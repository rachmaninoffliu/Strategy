"""Does a model that combines the Kalshi quote with BTC spot features beat the rules?  Walk-forward (expanding) by calendar month.

Per (market, decision minute) row, features known at the decision instant:
  logit(mid), z = ln(S/K)/(sigma*sqrt(tau)) with K = Coinbase strike proxy, logit(Phi(z)), z vs open, short/long vol ratio, fast/slow vol,
  minute, spread, 3-minute and 10-minute spot return in sigma units, (optionally) hour-of-day sin/cos.
Target: YES.  Models: market mid (baseline), logistic regression, HistGradientBoosting.
Trading: at every decision minute 2..13 compute EV of buying YES at the ask and NO at 1-bid with predicted p, take the best one per market if
EV_hat >= threshold (after 2c slippage + Polymarket-US fee); realised P&L is computed only on the TEST month (never seen in training).
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from panel import build_panel, load_spot
from trader.fees import fee_per_contract

SLIP = 0.02
pd.set_option("display.width", 220)


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def make_features() -> pd.DataFrame:
    P = build_panel()
    sp = load_spot()
    ok = (P.bid_c > 0) & (P.ask_c < 1) & (P.ask_c >= P.bid_c) & ((P.ask_c - P.bid_c) <= 0.06 + 1e-9) & P.S.notna() & P.sig.notna() & P.cb_k.notna() & P.vr.notna()
    P = P[ok & P.minute.between(2, 13)].copy()
    P["mid"] = (P.bid_c + P.ask_c) / 2
    P["spread"] = P.ask_c - P.bid_c
    sq = np.sqrt(P.tau_min)
    P["z"] = np.log(P.S / P.cb_k) / (P.sig * sq)
    P["z_open"] = np.log(P.S / P.S0) / (P.sig * sq)
    P["lm"] = logit(P.mid)
    P["lpz"] = logit(norm.cdf(P.z))
    P["fast"] = P.sig_fast / P.sig
    t_end = (P.open_ts + 60 * P.minute - 60).values
    cl = sp.close
    for k in (3, 10):
        prev = cl.reindex(t_end - 60 * k).values
        P[f"r{k}"] = np.log(P.S.values / prev) / (P.sig * np.sqrt(k))
    hr = pd.to_datetime(P.open_ts, unit="s", utc=True).dt.hour
    P["hour"] = hr
    P["hs"], P["hc"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    P["month_i"] = (pd.to_datetime(P.open_ts, unit="s", utc=True).dt.year - 2025) * 12 + pd.to_datetime(P.open_ts, unit="s", utc=True).dt.month - 12
    P = P.dropna(subset=["r3", "r10", "z", "z_open", "fast"]).reset_index(drop=True)
    return P


BASE = ["lm", "lpz", "z", "z_open", "vr", "fast", "minute", "spread", "r3", "r10"]
HOUR = ["hs", "hc"]


def models():
    return {
        "LR (quote+spot)": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)), BASE),
        "HGB (quote+spot)": (lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=300, l2_regularization=5.0, random_state=0), BASE),
        "HGB (+hour of day)": (lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=300, l2_regularization=5.0, random_state=0), BASE + HOUR),
    }


def trade_pnl(df: pd.DataFrame, p: np.ndarray, thr: float) -> pd.DataFrame:
    """Best-EV side per market at any minute where EV_hat >= thr; realised pnl per contract."""
    yes_cost = np.minimum(df.ask_c.values + SLIP, 0.99)
    no_cost = np.minimum(1 - df.bid_c.values + SLIP, 0.99)
    fee = lambda c: np.array([fee_per_contract("Polymarket US", float(x)) for x in np.round(c, 3)])
    ev_y = p - yes_cost - fee(yes_cost)
    ev_n = (1 - p) - no_cost - fee(no_cost)
    buy_yes = ev_y >= ev_n
    ev = np.where(buy_yes, ev_y, ev_n)
    cost = np.where(buy_yes, yes_cost, no_cost)
    win = np.where(buy_yes, df.yes.values, 1 - df.yes.values)
    out = pd.DataFrame({"ticker": df.ticker.values, "minute": df.minute.values, "open_ts": df.open_ts.values, "hour": df.hour.values,
                        "ev_hat": ev, "cost": cost, "win": win, "month_i": df.month_i.values})
    out = out[out.ev_hat >= thr]
    out = out.sort_values(["ticker", "ev_hat"], ascending=[True, False]).drop_duplicates("ticker")
    f = fee(np.minimum(out.cost.values, 0.99))
    out["pnl"] = out.win - out.cost - f
    return out


def tstat(x):
    x = np.asarray(x)
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 3 and x.std() > 0 else np.nan


def main():
    P = make_features()
    print(f"panel rows {len(P):,}, markets {P.ticker.nunique():,}, months {P.month_i.min()}..{P.month_i.max()}")
    first = 3                                                  # first test month = Mar-26 (train on Dec-Feb)
    M = int(P.month_i.max())
    preds = {k: np.full(len(P), np.nan) for k in models()}
    preds["market mid"] = P.mid.values.copy()
    for k in range(first, M + 1):
        tr = P[P.month_i < k]
        te_idx = np.where(P.month_i == k)[0]
        if len(te_idx) == 0:
            continue
        for name, (mk, feats) in models().items():
            m = mk().fit(tr[feats].values, tr.yes.values)
            preds[name][te_idx] = m.predict_proba(P.loc[te_idx, feats].values)[:, 1]
    T = P[P.month_i >= first].copy()
    print("\n=== A. Out-of-fold log-loss (lower is better), Mar-26 .. Oct-26, minutes 2..13 ===")
    rows = []
    for name, pr in preds.items():
        pr = np.clip(pr[T.index.values], 1e-3, 1 - 1e-3)
        rows.append((name, log_loss(T.yes, pr)))
    base = rows[-1][1]
    for name, ll in rows:
        print(f"  {name:22s} {ll:.5f}   vs market {1e5 * (ll - base):+8.1f} (x1e-5)")
    print("  (market mid is the benchmark: a model that cannot beat it has no information beyond the quote)")

    print("\n=== B. Walk-forward trading, realised OOF P&L per contract (all hours, all minutes 2-13, best side per market) ===")
    for name in models():
        pr = preds[name][T.index.values]
        for thr in (0.02, 0.04, 0.06, 0.08):
            tr_ = trade_pnl(T, pr, thr)
            if len(tr_) < 10:
                print(f"  {name:22s} thr {thr * 100:.0f}c: n={len(tr_)}")
                continue
            by_m = tr_.groupby("month_i").pnl.sum()
            print(f"  {name:22s} thr {thr * 100:.0f}c: n={len(tr_):5d} EV={tr_.pnl.mean() * 100:+6.2f}c win={tr_.win.mean():.3f} t={tstat(tr_.pnl):+5.2f} | months>0: {(by_m > 0).sum()}/{len(by_m)} | mean predicted EV {tr_.ev_hat.mean() * 100:+.1f}c")

    print("\n=== C. Same, but split the OOF trades by regime (before / after 2026-08-24) ===")
    cut = pd.Timestamp("2026-08-24", tz="UTC").timestamp()
    for name in ("LR (quote+spot)", "HGB (quote+spot)", "HGB (+hour of day)"):
        pr = preds[name][T.index.values]
        tr_ = trade_pnl(T, pr, 0.04)
        for lab, x in (("before Aug 24", tr_[tr_.open_ts < cut]), ("since Aug 24", tr_[tr_.open_ts >= cut])):
            if len(x) > 5:
                print(f"  {name:22s} thr 4c {lab:14s}: n={len(x):5d} EV={x.pnl.mean() * 100:+6.2f}c win={x.win.mean():.3f} t={tstat(x.pnl):+5.2f}")
    print("\n=== D. Which hours does the (+hour) model trade, and does it make money there?  (thr 4c, OOF) ===")
    tr_ = trade_pnl(T, preds["HGB (+hour of day)"][T.index.values], 0.04)
    g = tr_.groupby("hour").agg(n=("pnl", "size"), ev_c=("pnl", lambda v: v.mean() * 100), win=("win", "mean"))
    print(g.round(2).T.to_string())


if __name__ == "__main__":
    main()
