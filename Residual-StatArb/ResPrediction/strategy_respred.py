"""
Residual Prediction - a model that forecasts each stock's next residual return from many features, trades the cross-section of forecasts, hedges market and sector with ETFs.

Krauss, Do & Huck (2017), "Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500" (EJOR 259(2)) and Gu, Kelly & Xiu (2020), "Empirical asset
pricing via machine learning" (RFS 33(5)) predict the cross-section of returns from lagged returns, momentum, liquidity and volatility features; Guijarro-Ordonez, Pelger & Zanotti
("Deep learning statistical arbitrage", Management Science 2025) apply the idea to factor-model RESIDUALS. Here the target is the residual open-to-open return over the next ``horizon``
bars (from the bar the order fills in), the features are the residual-based predictors in resarb/features.py (each from a paper), and the model is refitted walk-forward: every
prediction is made by a model trained only on data whose targets were known before that day.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from resarb import features as X
from resarb.strategy import Param, Strategy

SELECTED = "selected"


class ResidualPrediction(Strategy):
    KEY = "ResPrediction"
    NAME = "Residual Prediction"
    PREFIX = "RESPRED"
    SUMMARY = ("A walk-forward model (ridge regression or gradient-boosted trees) forecasts each member's residual return over the next horizon from residual-based features; the engine "
               "buys the highest forecasts, shorts the lowest, and hedges market and sector beta with ETFs.")
    PAPERS = [
        ("Krauss, Do & Huck (2017), Deep neural networks, gradient-boosted trees, random forests: statistical arbitrage on the S&P 500, EJOR 259(2) 689-702",
         "https://ideas.repec.org/a/eee/ejores/v259y2017i2p689-702.html", "lagged-return features, tree models, daily long/short top-k"),
        ("Gu, Kelly & Xiu (2020), Empirical asset pricing via machine learning, RFS 33(5)", "https://www.nber.org/papers/w25398", "dominant predictors: momentum, liquidity, volatility"),
        ("Guijarro-Ordonez, Pelger & Zanotti (2021/2025), Deep learning statistical arbitrage", "https://arxiv.org/abs/2106.04028", "predict factor-model residuals, then trade them"),
        ("Heston, Korajczyk & Sadka (2010), Intraday patterns in the cross-section of stock returns, J. Finance 65(4)", "https://ar5iv.arxiv.org/html/1005.3535", "same-time-of-day feature"),
        ("Gao, Han, Li & Zhou (2018), Market intraday momentum, JFE", "https://profiles.wustl.edu/en/publications/market-intraday-momentum/", "market-state features"),
        ("Lou, Polk & Skouras (2019), A tug of war: overnight versus intraday expected returns, JFE", "https://personal.lse.ac.uk/polk/research/TugOfWar.pdf", "overnight / intraday split"),
    ]
    PARAMS = [
        Param("model", "Model", "ridge", "choice", choices=("ridge", "gbm"), help="ridge = linear on ranked features; gbm = histogram gradient-boosted trees (slower)"),
        Param("horizon", "Forecast horizon (bars)", 26, "int", 1, 130, 1, help="residual return from the fill over this many 15-min bars (26 = one session)"),
        Param("features", "Features", SELECTED, "text", help="'selected' (the ones that passed the IC test), 'all', or a comma list"),
        Param("train_years", "Training window (years)", 3.0, "float", 0.5, 8.0, 0.5, dec=1),
        Param("min_train_years", "Minimum history before the first forecast (years)", 1.5, "float", 0.5, 5.0, 0.5, dec=1),
        Param("retrain", "Retrain", "quarterly", "choice", choices=("monthly", "quarterly", "yearly")),
        Param("sample_every", "Training sample: every n-th bar", 2, "int", 1, 26, 1),
        Param("ridge_alpha", "Ridge penalty", 10.0, "float", 0.0, 1e5, 10.0, dec=1),
        Param("gbm_iter", "GBM trees", 200, "int", 20, 2000, 20),
        Param("gbm_leaf", "GBM min samples per leaf", 2000, "int", 50, 50000, 250),
    ]
    # research choice (in-sample best of 25 variants, research/r4 + r8): ridge on the 15 selected features, 5-day horizon, rebalanced daily at 15:45
    DEFAULTS = {"model": "ridge", "horizon": 130, "rebalance": "daily", "trade_time": "15:45", "method": "quantile", "q_in": 0.2, "q_out": 0.35, "hedge": "etf", "beta_days": 126,
                "min_train_years": 2.0, "gross": 0.8}                                      # RM: stocks + hedges <= 1.9x overnight at the 95th percentile (research/r9_margin.py)
    PRESETS = {
        "Research choice: ridge, 5-day horizon, daily at 15:45 (in-sample best)": {"model": "ridge", "horizon": 130, "rebalance": "daily", "trade_time": "15:45", "q_in": 0.2,
                                                                                   "q_out": 0.35, "features": SELECTED},
        "Ridge, 1-day horizon, hourly": {"model": "ridge", "horizon": 26, "rebalance": "1h", "q_in": 0.2, "q_out": 0.3},
        "Ridge, 1-hour horizon, every 15 min": {"model": "ridge", "horizon": 4, "rebalance": "15min", "q_in": 0.1, "q_out": 0.25},
        "Gradient boosting, 1-day horizon, daily at 15:45": {"model": "gbm", "horizon": 26, "rebalance": "daily", "trade_time": "15:45"},
    }
    HOWTO = ("The model is refitted on the last few years (every quarter by default; live: press 'Train model' or let the engine refit when the model is older than the retrain period). "
             "Every rebalance it forecasts each member's residual return, buys the top fraction, shorts the bottom fraction and hedges with ETFs.")
    # chosen by research/r1_feature_ic.py on IN-SAMPLE data only: |Newey-West t| >= 3 at a 4, 13 or 26-bar horizon (2016-09 .. 2022, top-100, ETF residuals; REPORT.md section 3)
    SELECTED_FEATURES: list = ["rr_1", "rr_2", "cloc", "rr_4", "vwap_dev", "rr_13", "id_today", "ivol", "rr_26", "ou_s", "r2_fit", "range", "ou_kappa", "vol_surp", "hks"]

    def warmup_days(self) -> int:
        return max(int(self.v["beta_days"]), 252) + 60

    def feature_list(self) -> list:
        f = str(self.v["features"]).strip()
        if f == SELECTED:
            names = list(self.SELECTED_FEATURES)
        elif f == "all" or not f:
            names = list(X.FEATURES)
        else:
            names = [x.strip() for x in f.split(",") if x.strip() in X.FEATURES]
        if self.v["model"] == "ridge":
            names = [n for n in names if n not in X.MARKET_WIDE]                             # the same for every stock: a linear cross-sectional model can't use them
        return names

    # ── design matrix ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def design(self, ctx, rows: np.ndarray, log=lambda m: None) -> tuple:
        """Long-format X (cells = member stock-bars of the given rows), cell coordinates, and the cross-sectionally ranked target."""
        names = self.feature_list()
        ranked = self.ranked(ctx, names, log)
        mem = ctx.member[rows]
        ti, ii = np.nonzero(mem)
        Xl = np.empty((len(ti), len(names)), np.float32)
        for k, n in enumerate(names):
            Xl[:, k] = np.nan_to_num(ranked[n][rows][ti, ii], nan=0.0)                         # a missing feature = the cross-sectional median
        return Xl, rows[ti], ii, names

    @staticmethod
    def ranked(ctx, names: list, log=lambda m: None) -> dict:
        """Cross-sectional ranks in [-0.5, 0.5] among the members (market-wide features stay raw: no statistic of the panel may leak in, and live must see the same scale), computed once."""
        out = {}
        todo = [n for n in names if ("_xr", n) not in ctx._feat]
        if todo:
            log(f"ranking {len(todo)} features ...")
            feats = ctx.features(todo)
            for n in todo:
                a = feats[n]
                if n in X.MARKET_WIDE:
                    ctx._feat[("_xr", n)] = a.astype(np.float32)
                else:
                    ctx._feat[("_xr", n)] = X.xs_rank(a, ctx.member).astype(np.float32)
        for n in names:
            out[n] = ctx._feat[("_xr", n)]
        return out

    def target(self, ctx) -> np.ndarray:
        key = ("_target", int(self.v["horizon"]), ctx.lag)
        if key not in ctx._feat:
            fwd = X.forward(ctx.res, int(self.v["horizon"]), ctx.lag)
            ctx._feat[key] = X.xs_rank(fwd, ctx.member).astype(np.float32)
        return ctx._feat[key]

    def _fit(self, Xtr: np.ndarray, ytr: np.ndarray):
        if self.v["model"] == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_iter=int(self.v["gbm_iter"]), learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=int(self.v["gbm_leaf"]), l2_regularization=1.0,
                                              early_stopping=False, random_state=0)
            m.fit(Xtr, ytr)
            return ("gbm", m)
        mu = Xtr.mean(0)
        Xc = Xtr - mu
        A = Xc.T.astype(float) @ Xc.astype(float) + float(self.v["ridge_alpha"]) * len(Xc) * 1e-4 * np.eye(Xc.shape[1])
        beta = np.linalg.solve(A, Xc.T.astype(float) @ (ytr - ytr.mean()))
        return ("ridge", (mu, beta, float(ytr.mean())))

    @staticmethod
    def _predict(model, Xp: np.ndarray) -> np.ndarray:
        kind, m = model
        if kind == "gbm":
            return m.predict(Xp)
        mu, beta, b0 = m
        return (Xp - mu) @ beta + b0

    def _retrain_dates(self, p, first: pd.Timestamp) -> list:
        step = {"monthly": 1, "quarterly": 3, "yearly": 12}[str(self.v["retrain"])]
        months = pd.date_range(first.to_period("M").to_timestamp(), p.dates[-1] + pd.offsets.MonthBegin(1), freq=f"{step}MS")
        return [m for m in months if m <= p.dates[-1]]

    def cache_key(self, ctx) -> str:
        cfg = {k: self.v[k] for k in ("model", "horizon", "features", "train_years", "min_train_years", "retrain", "sample_every", "ridge_alpha", "gbm_iter", "gbm_leaf", "universe_n",
                                      "factor_model", "beta_days", "pca_k", "feed")}
        cfg["feature_list"] = self.feature_list()
        cfg["span"] = [str(ctx.p.ts[0]), str(ctx.p.ts[-1]), len(ctx.p.syms)]
        return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def score(self, ctx, log=lambda m: None) -> np.ndarray:
        """Walk-forward forecasts for every member bar (NaN before the first model). Cached on disk per configuration."""
        cache = os.path.join(self.folder, "models", f"wf_{self.cache_key(ctx)}.npy")
        if os.path.exists(cache):
            log(f"walk-forward forecasts from cache ({os.path.basename(cache)})")
            return np.load(cache)
        p = ctx.p
        T, N = p.close.shape
        y = self.target(ctx)
        out = np.full((T, N), np.nan, np.float32)
        start = pd.Timestamp(self.v["start"])
        first_member = p.dates[np.argmax(ctx.member_d.any(1))]
        first = max(start, first_member + pd.DateOffset(months=int(round(12 * float(self.v["min_train_years"])))))
        dates = self._retrain_dates(p, first)
        emb = int(self.v["horizon"]) + 1 + ctx.lag
        se = int(self.v["sample_every"])
        t_all = np.arange(T)
        log(f"walk-forward: {len(dates)} refits ({self.v['retrain']}), model {self.v['model']}, {len(self.feature_list())} features, horizon {self.v['horizon']} bars")
        t0 = time.time()
        for k, R in enumerate(dates):
            iR = int(np.searchsorted(p.ts, R))
            iN = int(np.searchsorted(p.ts, dates[k + 1])) if k + 1 < len(dates) else T
            lo = int(np.searchsorted(p.ts, R - pd.DateOffset(months=int(round(12 * float(self.v["train_years"]))))))
            tr = t_all[lo:max(iR - emb, lo)]
            tr = tr[(p.bar[tr] % se) == 0]
            if len(tr) < 500:
                continue
            Xtr, tt, ii, _ = self.design(ctx, tr)
            ytr = y[tt, ii]
            ok = np.isfinite(ytr)
            if ok.sum() < 5000:
                continue
            model = self._fit(Xtr[ok], ytr[ok].astype(float))
            pr = t_all[iR:iN]
            Xp, tp, ip, _ = self.design(ctx, pr)
            out[tp, ip] = self._predict(model, Xp).astype(np.float32)
            log(f"  refit {R:%Y-%m}: {ok.sum():,} training cells, forecasts to {p.ts[iN - 1]:%Y-%m-%d} ({time.time() - t0:.0f} s)")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, out)
        return out

    # ── live model ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def model_path(self) -> str:
        return os.path.join(self.folder, "models", "live_model.pkl")

    def fit_latest(self, ctx, log=lambda m: print(m)) -> dict:
        """Fit on the most recent ``train_years`` of data whose targets are known; save it for the live engine."""
        p = ctx.p
        T = len(p.ts)
        y = self.target(ctx)
        emb = int(self.v["horizon"]) + 1 + ctx.lag
        lo = int(np.searchsorted(p.ts, p.ts[-1] - pd.DateOffset(months=int(round(12 * float(self.v["train_years"]))))))
        tr = np.arange(lo, max(T - emb, lo))
        tr = tr[(p.bar[tr] % int(self.v["sample_every"])) == 0]
        Xtr, tt, ii, names = self.design(ctx, tr)
        ytr = y[tt, ii]
        ok = np.isfinite(ytr)
        if ok.sum() < 5000:
            raise RuntimeError(f"only {ok.sum()} training cells - load more history")
        model = self._fit(Xtr[ok], ytr[ok].astype(float))
        meta = {"trained": str(pd.Timestamp.now()), "data_end": str(p.ts[-1]), "cells": int(ok.sum()), "features": names, "config": {k: self.v[k] for k in self.v}}
        os.makedirs(os.path.dirname(self.model_path()), exist_ok=True)
        with open(self.model_path(), "wb") as f:
            pickle.dump({"model": model, "meta": meta}, f)
        log(f"model trained on {ok.sum():,} cells through {p.ts[-1]:%Y-%m-%d %H:%M} and saved")
        return meta

    def live_score(self, ctx, log=lambda m: None) -> np.ndarray:
        """Forecasts of the saved model for the last bars of the context (live engine)."""
        if not os.path.exists(self.model_path()):
            raise FileNotFoundError("no trained model - press 'Train model' (Live tab) first")
        with open(self.model_path(), "rb") as f:
            blob = pickle.load(f)
        names = blob["meta"]["features"]
        if names != self.feature_list():
            raise RuntimeError("the saved model was trained with other features - train it again")
        T, N = ctx.p.close.shape
        rows = np.arange(max(T - 30, 0), T)
        Xp, tp, ip, _ = self.design(ctx, rows)
        out = np.full((T, N), np.nan)
        out[tp, ip] = self._predict(blob["model"], Xp)
        return out

    def model_info(self) -> dict:
        if not os.path.exists(self.model_path()):
            return {}
        with open(self.model_path(), "rb") as f:
            return pickle.load(f)["meta"]
