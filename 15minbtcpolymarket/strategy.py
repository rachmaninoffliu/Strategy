"""Calibration-Fade Kelly strategy for Kalshi KXBTC15M ("BTC price up in next 15 mins?").

Thesis (measured, see analyze*.py): Kalshi's quote is a well-calibrated probability, but the *shape* of its
miscalibration changes through the 15-minute window --
    minutes 3-9   market slightly UNDER-confident  (favourites cheap, dogs rich)   slope ~1.11-1.15
    minutes 12-13 market OVER-confident            (favourites rich, dogs cheap)   slope ~0.92-0.97
After taker fees the edge is ~1c (early favourite leg, high win-rate) to ~3-5c (late fade leg, low win-rate), so
fee-aware entry, shrinkage and fractional Kelly matter more than the signal itself.

Pure functions only -- the same `decide()` is used by the backtest and by the paper/live monitor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── fees ──────────────────────────────────────────────────────────────────────
KALSHI_COEF = 0.07          # taker; series fee_type == "quadratic" (no maker fee) per /series/KXBTC15M


def fee_total(contracts: int, price: float, coef: float = KALSHI_COEF) -> float:
    """Kalshi taker fee for one order: round_up_to_cent(coef * C * P * (1-P))."""
    return ceil(coef * contracts * price * (1 - price) * 100 - 1e-9) / 100


def fee_pc(price, contracts: int = 100, coef: float = KALSHI_COEF):
    """Per-contract fee (vectorised) for an order of ``contracts``."""
    p = np.asarray(price, dtype=float)
    return np.ceil(coef * contracts * p * (1 - p) * 100 - 1e-9) / 100 / contracts


# ── Kelly ─────────────────────────────────────────────────────────────────────
def kelly_binary(p: float, cost: float) -> float:
    """Full-Kelly fraction of bankroll for a contract costing ``cost`` (fee included) that pays $1 w.p. ``p``.
    Net odds b = (1-cost)/cost  ->  f* = (p*b - (1-p)) / b = (p - cost) / (1 - cost)."""
    if cost <= 0 or cost >= 1:
        return 0.0
    return max(0.0, (p - cost) / (1.0 - cost))


# ── calibrator ────────────────────────────────────────────────────────────────
MINUTE_GROUPS = ((1, 4), (5, 8), (9, 11), (12, 13), (14, 14))


def group_of(minute: int) -> int:
    for i, (a, b) in enumerate(MINUTE_GROUPS):
        if a <= minute <= b:
            return i
    return -1


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    """logit P(YES) = a_g + b_g * logit(mid), one (a,b) per minute group, fitted by logistic regression."""
    params: dict = field(default_factory=dict)       # g -> (a, b)
    cov: dict = field(default_factory=dict)          # g -> 2x2 covariance (cluster-robust by market)

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Calibrator":
        cal = cls()
        d = df.assign(lm=logit((df.bid_c + df.ask_c) / 2), g=df.minute.map(group_of))
        for g in range(len(MINUTE_GROUPS)):
            s = d[d.g == g]
            if len(s) < 300 or s.yes.nunique() < 2:
                cal.params[g] = (0.0, 1.0)
                cal.cov[g] = np.diag([1e-2, 1e-2])
                continue
            X = sm.add_constant(s[["lm"]])
            r = sm.Logit(s.yes, X).fit(disp=0, cov_type="cluster",
                                       cov_kwds={"groups": s.ticker.astype("category").cat.codes})
            cal.params[g] = (float(r.params["const"]), float(r.params["lm"]))
            cal.cov[g] = np.asarray(r.cov_params())
        return cal

    def predict(self, minute, bid, ask):
        """Vectorised: returns (p_hat, se_p) for the YES outcome."""
        minute = np.asarray(minute)
        lm = logit((np.asarray(bid) + np.asarray(ask)) / 2)
        g = np.array([group_of(int(m)) for m in minute])
        a = np.array([self.params.get(int(i), (0.0, 1.0))[0] for i in g])
        b = np.array([self.params.get(int(i), (0.0, 1.0))[1] for i in g])
        z = a + b * lm
        p = 1 / (1 + np.exp(-z))
        var = np.array([(lambda C, x: C[0, 0] + 2 * C[0, 1] * x + C[1, 1] * x * x)(self.cov.get(int(i), np.diag([1e-2, 1e-2])), l)
                        for i, l in zip(g, lm)])
        se_p = p * (1 - p) * np.sqrt(np.maximum(var, 0))
        return p, se_p


MID_EDGES = np.array([0, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .98, 1.0001])


@dataclass
class BucketCalibrator:
    """Non-parametric: empirical P(YES) per (minute group, mid-price bucket), shrunk to the bucket's mean mid with a
    Beta-style prior of ``prior_n`` pseudo-observations. Captures the NON-linear 0.70-0.95 late-favourite premium that a
    single logit slope cannot."""
    prior_n: float = 300.0
    cluster: float = 2.5                  # avg correlated rows per market inside one bucket -> n_eff = n / cluster
    table: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, prior_n: float = 300.0) -> "BucketCalibrator":
        cal = cls(prior_n=prior_n)
        mid = ((df.bid_c + df.ask_c) / 2).values
        g = df.minute.map(group_of).values
        b = np.clip(np.digitize(mid, MID_EDGES) - 1, 0, len(MID_EDGES) - 2)
        d = pd.DataFrame({"g": g, "b": b, "y": df.yes.values, "m": mid})
        for (gi, bi), s in d.groupby(["g", "b"]):
            cal.table[(int(gi), int(bi))] = (len(s), float(s.y.sum()), float(s.m.mean()))
        return cal

    def predict(self, minute, bid, ask):
        mid = (np.asarray(bid) + np.asarray(ask)) / 2
        g = np.array([group_of(int(m)) for m in np.asarray(minute)])
        b = np.clip(np.digitize(mid, MID_EDGES) - 1, 0, len(MID_EDGES) - 2)
        p = mid.copy()
        se = np.full(len(mid), 0.02)
        for i, (gi, bi) in enumerate(zip(g, b)):
            n, wins, mm = self.table.get((int(gi), int(bi)), (0, 0.0, mid[i]))
            ph = (wins + self.prior_n * mm) / (n + self.prior_n)
            # the bucket's own offset (freq - mean mid) is applied to THIS row's mid -> keeps within-bucket gradient
            p[i] = float(np.clip(mid[i] + (ph - mm), 0.001, 0.999))
            se[i] = np.sqrt(max(ph * (1 - ph), 1e-4) / max((n + self.prior_n) / self.cluster, 1.0))
        return p, se


# ── decision rule ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    legs: tuple = ("FAV", "FADE")       # FAV: favourite, cost>=fav_min, minutes fav_minutes ; FADE: dog in minutes fade_minutes
    fav_minutes: tuple = (3, 11)
    fav_min_cost: float = 0.80
    fade_minutes: tuple = (12, 13)
    fade_max_cost: float = 0.35
    min_edge: float = 0.010             # required edge after fee, after shrinkage ($ per contract)
    shrink: float = 0.5                 # keep this fraction of the calibrated edge (guards non-stationarity)
    se_k: float = 1.0                   # extra penalty of se_k * se(p_hat)
    kelly_mult: float = 0.5             # fractional Kelly applied on top of the shrunk edge
    max_frac: float = 0.05              # hard cap per trade (fraction of bankroll)
    max_contracts: int = 2000           # depth cap (top of book is typically 5k+ contracts)
    slip: float = 0.0                   # extra adverse price on fills, $ per contract
    max_vr: float | None = None         # FADE leg only: skip when short/long realised-vol ratio >= max_vr (hot regime)
    contracts_for_fee: int = 100


def decide(cal: Calibrator, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Row-wise decision for quote snapshots (needs: minute, bid_c, ask_c). Adds side/cost/p_win/edge/frac/leg."""
    d = df.copy()
    p, se = cal.predict(d.minute.values, d.bid_c.values, d.ask_c.values)
    mid = (d.bid_c + d.ask_c) / 2
    # shrink toward the market: p_adj = q + shrink * (p_hat - q), then subtract the estimation penalty
    p_adj = mid.values + cfg.shrink * (p - mid.values)
    pen = cfg.se_k * se * cfg.shrink
    cost_y = np.minimum(d.ask_c.values + cfg.slip, 0.995)
    cost_n = np.minimum(1 - d.bid_c.values + cfg.slip, 0.995)
    cy = cost_y + fee_pc(cost_y, cfg.contracts_for_fee)
    cn = cost_n + fee_pc(cost_n, cfg.contracts_for_fee)
    edge_y = (p_adj - pen) - cy
    edge_n = ((1 - p_adj) - pen) - cn
    use_y = edge_y >= edge_n
    d["side"] = np.where(use_y, "YES", "NO")
    d["cost"] = np.where(use_y, cost_y, cost_n)
    d["cost_all"] = np.where(use_y, cy, cn)
    d["p_win"] = np.where(use_y, p_adj - pen, (1 - p_adj) - pen)
    d["edge"] = np.where(use_y, edge_y, edge_n)
    fav = np.where(use_y, mid.values >= 0.5, mid.values < 0.5)
    mn = d.minute.values
    leg_fav = fav & (d.cost.values >= cfg.fav_min_cost) & (mn >= cfg.fav_minutes[0]) & (mn <= cfg.fav_minutes[1])
    leg_fade = (~fav) & (d.cost.values <= cfg.fade_max_cost) & (mn >= cfg.fade_minutes[0]) & (mn <= cfg.fade_minutes[1])
    if cfg.max_vr is not None and "vr" in d:
        leg_fade &= (d.vr.values < cfg.max_vr)
    ok = np.zeros(len(d), bool)
    d["leg"] = ""
    if "FAV" in cfg.legs:
        ok |= leg_fav
        d.loc[leg_fav, "leg"] = "FAV"
    if "FADE" in cfg.legs:
        ok |= leg_fade
        d.loc[leg_fade & ~leg_fav, "leg"] = "FADE"
    d["trade"] = ok & (d.edge.values >= cfg.min_edge)
    f_full = np.array([kelly_binary(pw, ca) for pw, ca in zip(d.p_win.values, d.cost_all.values)])
    d["frac"] = np.minimum(cfg.kelly_mult * f_full, cfg.max_frac)
    return d


# ── backtest ──────────────────────────────────────────────────────────────────
def settle_trade(row, contracts: int, slip_fee_contracts: int | None = None) -> float:
    """Net P&L in dollars for one filled order held to settlement (fee rounded up on the whole order)."""
    win = (row.yes == 1) if row.side == "YES" else (row.yes == 0)
    fee = fee_total(contracts, row.cost)
    return contracts * ((1.0 if win else 0.0) - row.cost) - fee


def simulate(panel: pd.DataFrame, cal_fn, cfg: Config, bankroll0: float = 10_000.0, compound: bool = True,
             train_days: int = 28, test_days: int = 7, start_ts: int | None = None) -> pd.DataFrame:
    """Walk-forward: refit calibrator on trailing ``train_days``, trade the next ``test_days``. One entry/market
    (first qualifying minute). Returns the trade log."""
    P = panel.sort_values(["open_ts", "minute"])
    t0, t1 = int(P.open_ts.min()), int(P.open_ts.max())
    day = 86400
    t = (start_ts or t0) + train_days * day
    bank = bankroll0
    out = []
    while t <= t1:
        tr = P[(P.open_ts >= t - train_days * day) & (P.open_ts < t)]
        te = P[(P.open_ts >= t) & (P.open_ts < t + test_days * day)]
        if len(tr) > 5000 and len(te):
            cal = cal_fn(tr)
            d = decide(cal, te, cfg)
            d = d[d.trade].sort_values(["open_ts", "minute"]).groupby("ticker", as_index=False).first()
            for r in d.sort_values("open_ts").itertuples():
                stake = (bank if compound else bankroll0) * r.frac
                n = int(min(floor(stake / r.cost_all), cfg.max_contracts))
                if n < 1:
                    continue
                pnl = settle_trade(r, n)
                bank += pnl
                out.append(dict(ticker=r.ticker, open_ts=r.open_ts, minute=r.minute, leg=r.leg, side=r.side, cost=r.cost,
                                p_win=r.p_win, edge=r.edge, frac=r.frac, n=n, stake=n * r.cost, pnl=pnl,
                                win=int((r.yes == 1) if r.side == "YES" else (r.yes == 0)), bank=bank))
        t += test_days * day
    return pd.DataFrame(out)


def summarize(tr: pd.DataFrame, bankroll0: float = 10_000.0, label: str = "") -> dict:
    if tr.empty:
        return dict(label=label, trades=0)
    tr = tr.sort_values("open_ts")
    days = max((tr.open_ts.max() - tr.open_ts.min()) / 86400, 1)
    daily = tr.assign(d=pd.to_datetime(tr.open_ts, unit="s").dt.date).groupby("d").pnl.sum()
    ret = tr.pnl / (tr.stake + 1e-9)
    eq = tr.bank.values
    peak = np.maximum.accumulate(np.r_[bankroll0, eq])[1:]
    mdd = ((eq - peak) / peak).min()
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan
    # bootstrap mean per-trade return (trades are one-per-market -> ~independent)
    rng = np.random.default_rng(0)
    bs = [rng.choice(ret.values, len(ret)).mean() for _ in range(2000)]
    return dict(label=label, trades=len(tr), per_day=len(tr) / days, win=tr.win.mean(), avg_cost=tr.cost.mean(),
                ev_c=(tr.pnl / tr.n).mean() * 100, ret_per_trade=ret.mean() * 100,
                ci_lo=np.percentile(bs, 2.5) * 100, ci_hi=np.percentile(bs, 97.5) * 100,
                total_pnl=tr.pnl.sum(), end_bank=tr.bank.iloc[-1], total_ret=tr.bank.iloc[-1] / bankroll0 - 1,
                maxdd=mdd, sharpe=sharpe, avg_frac=tr.frac.mean())
