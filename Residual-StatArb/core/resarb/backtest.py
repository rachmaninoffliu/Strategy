"""
Bar-by-bar backtest of a residual long/short book (no Qt).

Timing (identical to the live engine): at the OPEN of bar k the engine sees the data of bars <= k-1-lag (lag 0 = a real-time feed: IEX on the free plan or paid SIP; lag 1 = the free plan's
SIP, which is 15 minutes late), decides with what it holds at that moment, and trades at bar k's open price. Positions then earn open-to-open returns, and their weights drift with prices
(exact fixed-share accounting in weight space). Costs: a per-side charge on every traded dollar (stocks and hedge ETFs separately) plus a borrow fee on short stock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import data as D
from .factors import Residuals
from .portfolio import PortCfg, approach, decide, hedge_etf, project

TD = 252
# Half-spread model fitted on SIP NBBO quotes (research/r0_spreads.py: top-100 members on 16 sample days 2017-2026, 10:00 / 12:00 / 15:45 ET, 4,602 symbol x time observations,
# R^2 0.32):  log(half spread, bp) = c[year] - 0.32 log(ADV / $1.17bn) + 0.227 log(price / $169) + 0.373 [first hour] - 0.316 [last half hour]
# c[year] = the midday half spread of a typical member: 2017 0.98 bp, 2018 1.12, 2019 1.05, 2020 1.78, 2021 1.89, 2024 1.77, 2025 2.64, 2026 2.58 (2016 = 2017, 2022-23 interpolated)
SPREAD = dict(c=0.947, adv=-0.32, px=0.227, first_hour=0.373, last_half_hour=-0.316, adv_med=1.17e9, px_med=169.0, lo=0.3, hi=25.0,
              c_by_year={2016: -0.025, 2017: -0.025, 2018: 0.115, 2019: 0.05, 2020: 0.576, 2021: 0.637, 2022: 0.616, 2023: 0.594, 2024: 0.573, 2025: 0.969, 2026: 0.947})


@dataclass
class BtCfg:
    start: str = "2016-09-01"
    end: str = ""
    split: str = "2023-01-01"            # in-sample before, out-of-sample from this date
    lag: int = 0
    cost_model: str = "spread"           # 'spread' = each name's modelled half spread at that time of day (x cost_mult) + slip_bp; 'flat' = cost_bp
    cost_mult: float = 1.0
    slip_bp: float = 0.5
    cost_bp: float = 2.0                 # per side, stocks, when cost_model = 'flat'
    etf_cost_bp: float = 0.5             # per side, hedge ETFs
    borrow_bp_yr: float = 30.0           # on short stock notional
    flat_eod: bool = False               # intraday only: flat at the last bar's open, no overnight risk
    band: float = 0.0                    # no-trade band: a name is only traded when its weight changes by more than band x (gross / names held)
    max_turnover: float = 0.0            # > 0: cap the traded fraction of capital per rebalance (closing trades first)
    fill: str = "open"                   # 'open' = the bar's first print; 'vwap' = the bar's VWAP (a check against fills that sit on the favourable side of the spread)
    max_gross: float = 1.9               # stocks + ETF hedges, x capital: the live engine's cap (Reg-T 2x overnight) - the whole book is scaled down when a target exceeds it


@dataclass
class BtResult:
    daily: pd.DataFrame                  # date-indexed: ret, gross_ret, cost, borrow, turnover, gross, net, longs, shorts, equity
    bar_pnl: np.ndarray
    name_pnl: pd.Series                  # P&L contribution by symbol (sum of weight x return)
    name_trades: pd.Series
    last_w: pd.Series
    cfg: BtCfg
    bar_cost: np.ndarray = None
    stats: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def log_adv(p: D.Panel, u: Optional[pd.DataFrame] = None) -> np.ndarray:
    """days x symbols: log of the 63-day median dollar volume from the universe table (the month's value; carried forward / back for months a name was not ranked)."""
    u = D.load_universe() if u is None else u
    months = p.dates.to_period("M").to_timestamp()
    t = u.pivot_table(index="month", columns="symbol", values="adv")
    t = t.reindex(columns=[s for s in p.syms if s in t.columns])
    full = t.reindex(t.index.union(pd.DatetimeIndex(months.unique()))).sort_index().ffill().bfill()
    out = np.full((len(p.dates), len(p.syms)), np.log(SPREAD["adv_med"]))
    if len(t.columns):
        vals = np.log(full.reindex(months).to_numpy(float))
        for j_t, s in enumerate(t.columns):
            out[:, p.col[s]] = np.where(np.isfinite(vals[:, j_t]), vals[:, j_t], out[:, p.col[s]])
    return out


def spread_c(year: Optional[int]) -> float:
    cy = SPREAD["c_by_year"]
    if year is None:
        return SPREAD["c"]
    return cy.get(int(year), cy[min(cy)] if int(year) < min(cy) else cy[max(cy)])


def half_spread_bp(ladv_row: np.ndarray, price_row: np.ndarray, bar: int, n_bars: int = 26, year: Optional[int] = None) -> np.ndarray:
    """Modelled half spread (bp) of each name at this bar of the day (``year`` picks the level of that year; None = the latest)."""
    s = SPREAD
    c = spread_c(year)
    tod = s["first_hour"] if bar <= 3 else (s["last_half_hour"] if bar >= n_bars - 2 else 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = c + s["adv"] * (ladv_row - np.log(s["adv_med"])) + s["px"] * (np.log(price_row) - np.log(s["px_med"])) + tod
    return np.clip(np.exp(np.nan_to_num(x, nan=c)), s["lo"], s["hi"])


def schedule(p: D.Panel, every: int = 1, bar_of_day: Optional[int] = None, weekday: Optional[int] = None, monthly: bool = False) -> np.ndarray:
    """Decision (data) bars: every n-th bar of each day (counted from the first), or one bar a day (``bar_of_day``, -1 = last), optionally only on one weekday (0 = Mon) or the last session
    of each month."""
    T = len(p.ts)
    if bar_of_day is None:
        m = (p.bar % max(every, 1)) == (max(every, 1) - 1)
    else:
        nb = p.day_last[p.day] - p.day_first[p.day] + 1
        b = np.where(bar_of_day < 0, nb + bar_of_day, np.minimum(bar_of_day, nb - 1))
        m = p.bar == b
    if weekday is not None:
        m &= p.ts.weekday == weekday
    if monthly:                                                                                # the exchange calendar, not "the last date in the data": in live, today is always the last date
        last_day = p.dates.isin(D.month_end_sessions(p.dates[0], p.dates[-1]))
        m &= last_day[p.day]
    return np.asarray(m, bool)


def run(p: D.Panel, res: Residuals, score: np.ndarray, member: np.ndarray, decide_at: np.ndarray, pcfg: PortCfg, bcfg: BtCfg, log: Callable[[str], None] = lambda m: None,
        stop: Callable[[], bool] = lambda: False) -> BtResult:
    """``score`` T x N (NaN = no view), ``member`` T x N bool, ``decide_at`` T bool (data bars on which the strategy re-decides)."""
    T, M = p.close.shape
    t0 = np.searchsorted(p.ts, pd.Timestamp(bcfg.start)) if bcfg.start else 0
    t1 = np.searchsorted(p.ts, pd.Timestamp(bcfg.end)) if bcfg.end else T
    is_etf = np.zeros(M, bool)
    is_etf[res.etf_idx] = True
    stock = ~is_etf
    stock[[p.col[s] for s in p.syms if s in D.ETFS]] = False
    sidx = np.where(stock)[0]
    sec = res.sector_col[sidx]
    r_oo = np.nan_to_num(res.r_oo)
    opn = p.open
    c_side = np.where(is_etf, bcfg.etf_cost_bp, bcfg.cost_bp) / 1e4
    spread = bcfg.cost_model == "spread"
    ladv = log_adv(p) if spread else None
    nb_day = p.day_last - p.day_first + 1
    years = p.dates.year.values
    borrow_bar = bcfg.borrow_bp_yr / 1e4 / (TD * 26)
    w = np.zeros(M)
    pnl = np.zeros(T)
    gross_pnl = np.zeros(T)
    cost = np.zeros(T)
    borrow = np.zeros(T)
    turn = np.zeros(T)
    gross = np.zeros(T)
    gross_all = np.zeros(T)
    net = np.zeros(T)
    nlong = np.zeros(T, np.int32)
    nshort = np.zeros(T, np.int32)
    name_pnl = np.zeros(M)
    name_trades = np.zeros(M)
    last_bar = p.day_last[p.day]
    lag = int(bcfg.lag)
    for k in range(max(t0, 1 + lag), t1):
        if stop():
            break
        tradable = np.isfinite(opn[k])
        if spread:
            hs = half_spread_bp(ladv[p.day[k]], opn[k], int(p.bar[k]), int(nb_day[p.day[k]]), int(years[p.day[k]]))
            c_side = np.where(is_etf, bcfg.etf_cost_bp, hs * bcfg.cost_mult + bcfg.slip_bp) / 1e4
        dead = ~tradable & (w != 0)
        if dead.any():                                                                     # no price any more (delisted / halted for the day): cash out at the last price
            cost[k] += float(np.abs(w[dead]) @ c_side[dead])
            name_trades[dead] += 1
            w[dead] = 0.0
        td = k - 1 - lag
        eod = bcfg.flat_eod and k == last_bar[k]
        if decide_at[td] or eod:
            held = w[sidx]
            if eod:
                tgt_s = np.zeros(len(sidx))
            else:
                tgt_s = decide(score[td, sidx], member[td, sidx], held, pcfg, sec, tradable[sidx])
                tgt_s = approach(held, tgt_s, pcfg.trade_rate, pcfg.gross, tradable[sidx])
            dd = p.day[td]
            if pcfg.hedge == "project":
                full = np.zeros(M)
                full[sidx] = tgt_s
                tgt_s = project(full, res, dd)[sidx]
            tgt = np.zeros(M)
            tgt[sidx] = tgt_s
            if pcfg.hedge == "etf" and not eod:
                full = np.zeros(M)
                full[sidx] = tgt_s
                h = hedge_etf(full, res, dd)
                tgt[res.etf_idx] += h
            tgt = np.where(tradable, tgt, w)
            if bcfg.max_gross > 0:                                                         # the live engine's gross cap: scale the tradable part down
                g_fixed = float(np.abs(tgt[~tradable]).sum())
                g_trad = float(np.abs(tgt[tradable]).sum())
                if g_fixed + g_trad > bcfg.max_gross and g_trad > 0:
                    tgt = np.where(tradable, tgt * max(bcfg.max_gross - g_fixed, 0.0) / g_trad, tgt)
            dw = tgt - w
            if bcfg.band > 0:
                nh = max(int((np.abs(tgt[sidx]) > 0).sum()), 1)
                thr = bcfg.band * pcfg.gross / nh
                small = (np.abs(dw) < thr) & (tgt != 0) & (w != 0) & (np.sign(tgt) == np.sign(w))
                dw[small] = 0.0
            if bcfg.max_turnover > 0 and np.abs(dw).sum() > bcfg.max_turnover:
                closing = (tgt == 0) & (w != 0)
                rest = ~closing & (dw != 0)
                room = max(bcfg.max_turnover - np.abs(dw[closing]).sum(), 0.0)
                tot = np.abs(dw[rest]).sum()
                if tot > 0:
                    dw[rest] *= min(1.0, room / tot)
            traded = np.abs(dw)
            cost[k] += float(traded @ c_side)
            if bcfg.fill == "vwap":                                                       # bought at the bar's VWAP instead of its open: pay (vwap / open - 1) per $ bought
                with np.errstate(invalid="ignore", divide="ignore"):
                    cost[k] += float(dw @ np.nan_to_num(p.vwap[k] / opn[k] - 1.0))
            turn[k] += float(traded.sum())
            name_trades[traded > 0] += 1
            w = w + dw
        r = r_oo[k]
        g = float(w @ r)
        b = float(-w[stock & (w < 0)].sum()) * borrow_bar
        name_pnl += w * r
        gross_pnl[k] = g
        borrow[k] = b
        pnl[k] = g - cost[k] - b
        gross[k] = float(np.abs(w[stock]).sum())
        gross_all[k] = float(np.abs(w).sum())
        net[k] = float(w.sum())
        nlong[k] = int((w[stock] > 0).sum())
        nshort[k] = int((w[stock] < 0).sum())
        denom = 1.0 + g
        w = w * (1.0 + r) / (denom if denom > 0.05 else 0.05)
        if (k - t0) % 5000 == 0:
            log(f"  backtest {p.ts[k]:%Y-%m-%d} ({k - t0:,}/{t1 - t0:,} bars)")
    rng = slice(max(t0, 1 + lag), t1)
    df = pd.DataFrame({"pnl": pnl[rng], "gross_pnl": gross_pnl[rng], "cost": cost[rng], "borrow": borrow[rng], "turnover": turn[rng], "gross": gross[rng], "gross_all": gross_all[rng],
                       "net": net[rng], "longs": nlong[rng], "shorts": nshort[rng], "day": p.day[rng]})
    agg = df.groupby("day").agg(ret=("pnl", lambda x: float(np.prod(1 + x) - 1)), gross_ret=("gross_pnl", lambda x: float(np.prod(1 + x) - 1)), cost=("cost", "sum"),
                                borrow=("borrow", "sum"), turnover=("turnover", "sum"), gross=("gross", "mean"), gross_all=("gross_all", "last"), net=("net", "mean"),
                                longs=("longs", "mean"), shorts=("shorts", "mean"))
    agg.index = p.dates[agg.index.values]
    agg["equity"] = (1 + agg.ret).cumprod()
    out = BtResult(daily=agg, bar_pnl=pnl, name_pnl=pd.Series(name_pnl, index=p.syms), name_trades=pd.Series(name_trades, index=p.syms), last_w=pd.Series(w, index=p.syms), cfg=bcfg,
                   bar_cost=cost)
    out.stats = stats(agg, bcfg.split, p)
    return out


# ── statistics ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _block(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"days": len(r)}
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    yrs = len(r) / TD
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0
    sh = float(mu / sd * math.sqrt(TD)) if sd > 0 else float("nan")
    return {"days": len(r), "cagr": cagr, "vol": float(sd * math.sqrt(TD)), "sharpe": sh, "t_stat": float(mu / sd * math.sqrt(len(r))) if sd > 0 else float("nan"),
            "max_dd": dd, "hit": float((r > 0).mean()), "skew": float(r.skew()), "total": float(eq.iloc[-1] - 1)}


def psr(sr_ann: float, n: int, skew: float, kurt: float, sr0_ann: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey-Lopez de Prado): P(true Sharpe > sr0) from n daily observations."""
    from math import erf, sqrt
    sr, s0 = sr_ann / sqrt(TD), sr0_ann / sqrt(TD)
    den = sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    z = (sr - s0) * sqrt(max(n - 1, 1)) / den
    return 0.5 * (1 + erf(z / sqrt(2)))


def dsr(sr_ann: float, n: int, skew: float, kurt: float, trials: int, sr_var_ann: float) -> float:
    """Deflated Sharpe ratio: PSR against the best Sharpe expected from ``trials`` unskilled tries with the given cross-trial variance of (annual) Sharpe."""
    from statistics import NormalDist
    if trials <= 1:
        return psr(sr_ann, n, skew, kurt)
    g = 0.5772156649
    nd = NormalDist()
    e_max = math.sqrt(max(sr_var_ann, 1e-12)) * ((1 - g) * nd.inv_cdf(1 - 1 / trials) + g * nd.inv_cdf(1 - 1 / (trials * math.e)))
    return psr(sr_ann, n, skew, kurt, e_max)


def stats(daily: pd.DataFrame, split: str, p: Optional[D.Panel] = None) -> dict:
    r = daily.ret
    s = {"all": _block(r)}
    if split:
        sp = pd.Timestamp(split)
        s["is"] = _block(r[r.index < sp])
        s["oos"] = _block(r[r.index >= sp])
    s["gross_all"] = _block(daily.gross_ret)
    s["turnover_ann"] = float(daily.turnover.mean() * TD)
    s["cost_ann"] = float(daily.cost.sum() / max(len(daily) / TD, 1e-9))
    s["borrow_ann"] = float(daily.borrow.sum() / max(len(daily) / TD, 1e-9))
    s["avg_gross"] = float(daily.gross.mean())
    if "gross_all" in daily:
        s["overnight_gross_all"] = float(daily.gross_all.mean())                              # stocks + ETF hedges held over the close, x capital
        s["overnight_gross_p95"] = float(daily.gross_all.quantile(0.95))
    s["avg_names"] = float((daily.longs + daily.shorts).mean())
    s["hold_days"] = float(2 * daily.gross.mean() / daily.turnover.mean()) if daily.turnover.mean() > 0 else float("nan")
    if p is not None and D.MARKET in p.col:
        m = p.col[D.MARKET]
        cd = p.daily_close()[:, m]
        mr = pd.Series(np.r_[np.nan, cd[1:] / cd[:-1] - 1], index=p.dates).reindex(r.index)
        ok = mr.notna() & r.notna()
        if ok.sum() > 50:
            s["beta_spy"] = float(np.cov(r[ok], mr[ok])[0, 1] / mr[ok].var())
            s["corr_spy"] = float(np.corrcoef(r[ok], mr[ok])[0, 1])
    a = s["all"]
    if "sharpe" in a and np.isfinite(a.get("sharpe", np.nan)):
        rr = r.dropna()
        s["psr"] = psr(a["sharpe"], len(rr), float(rr.skew()), float(rr.kurt() + 3))
    by = r.groupby(r.index.year).apply(lambda x: float(np.prod(1 + x) - 1))
    s["by_year"] = {int(k): float(v) for k, v in by.items()}
    return s


def cost_curve(bt: BtResult, mults: list = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)) -> pd.DataFrame:
    """Re-price the same trades with the trading cost scaled by each multiplier (the decisions do not depend on the cost). avg_bp_side = the average cost per traded dollar."""
    d = bt.daily
    base = d.gross_ret - d.borrow
    avg_bp = float(d.cost.sum() / d.turnover.sum() * 1e4) if d.turnover.sum() > 0 else 0.0
    rows = []
    for m in mults:
        b = _block(base - d.cost * m)
        b_is = _block((base - d.cost * m)[d.index < pd.Timestamp(bt.cfg.split)]) if bt.cfg.split else {}
        rows.append({"cost_x": m, "avg_bp_side": avg_bp * m, "sharpe": b.get("sharpe"), "sharpe_is": b_is.get("sharpe"), "cagr": b.get("cagr"), "max_dd": b.get("max_dd")})
    return pd.DataFrame(rows)


def summary_line(st: dict) -> str:
    a, i, o = st.get("all", {}), st.get("is", {}), st.get("oos", {})
    f = lambda b, k, fmt: (fmt.format(b[k]) if k in b and b[k] is not None and np.isfinite(b[k]) else "-")
    return (f"Sharpe {f(a, 'sharpe', '{:.2f}')} (IS {f(i, 'sharpe', '{:.2f}')}, OOS {f(o, 'sharpe', '{:.2f}')}) · CAGR {f(a, 'cagr', '{:.1%}')} · vol {f(a, 'vol', '{:.1%}')} · "
            f"maxDD {f(a, 'max_dd', '{:.1%}')} · turnover {st.get('turnover_ann', float('nan')):.0f}x/yr · hold {st.get('hold_days', float('nan')):.1f} d")
