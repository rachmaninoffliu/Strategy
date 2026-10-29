"""
Stage B: trading rules, costs, portfolio and metrics for the leveraged-ETF pairs book. The rules are one vectorised state machine (loop over bars, arrays over sessions), so a 1-minute,
ten-year backtest of a pair takes seconds. The live engine calls the very same ``decide_bar`` step on the latest bar, so what is tested is what trades.

Position convention: +1 = long the spread (long $1 of the 3x ETF, short $beta of the 1x ETF), -1 = short the spread. A unit position earns ``eps`` per bar (rL - beta rB) per $1 of L notional and
costs ``cL + beta cB`` (bp of the L notional) every time it changes by one unit. Everything is flat before the close, nothing is held overnight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Callable, Optional

import numpy as np
import pandas as pd

import levpairs_data as D
from levpairs_core import TD, PairSignals

PAIR_ORDER = [D.pair_name(p) for p in D.PAIRS]
EXIT = {0: "", 1: "target", 2: "stop", 3: "time", 4: "end of day", 5: "no signal", 6: "day stop"}


@dataclass(frozen=True)
class SimCfg:
    start: str = "2017-01-01"
    end: Optional[str] = None
    capital: float = 100_000.0
    pair_w: tuple = (1.0, 1.0, 1.0)         # TQQQ/QQQ, SOXL/SMH, SPXL/SPY; 0 switches a pair off
    weight_mode: str = "equal"              # equal | inv_vol (trailing 60-session volatility of each pair's unit return)
    # rules
    s_open: float = 1.5
    s_close_long: float = 0.5
    s_close_short: float = 0.5
    s_stop: float = 4.0                     # model-failure stop: the spread keeps running away
    max_hold_min: int = 120
    max_halflife_min: float = 120.0         # entries need a fitted reversion half-life under this (0 = off)
    min_edge_ratio: float = 0.0             # skip an entry unless (|s| - s_close) x sigma_eq >= ratio x round-trip cost
    skip_open_min: int = 15
    no_entry_min: int = 30
    flat_min: int = 10
    cooldown_min: int = 30
    delay_bars: int = 0                     # act on the score of N bars ago (extra decision + order latency)
    fill: str = "next_open"                 # next_open: the order fills at the OPEN of the bar after the signal (realistic: the signal bar's last trade is not a price you can get)
                                            # mid: half way between the signal bar's close and the next open (what a very fast order might get)
                                            # close: instant fill at the signal bar's last trade (optimistic - it collects the snap-back of stale prints, kept to show the bias)
    move_z: float = 0.0                     # no new entries when |1x ETF move since the prior close| > z x its daily sigma (0 = off)
    day_stop_bp: float = 0.0                # stop a pair for the day after losing this much (bp of the L notional; 0 = off)
    # costs
    slip_bp: float = 0.3                    # one-way slippage per leg on top of half the quoted spread
    cost_mult: float = 1.0
    filter_slip_bp: float = 0.3             # the entry filter (min_edge_ratio) always prices a trade at THESE costs, so sweeping slip_bp / cost_mult changes what a trade costs,
    filter_cost_mult: float = 1.0           # not which trades are taken (Valid: otherwise 'zero slippage' looked worse than 'more slippage')
    # sizing
    vol_on: bool = True
    vol_target: float = 0.10
    g_init: float = 1.0
    g_min: float = 0.25
    g_max: float = 3.0                      # x equity; Alpaca day-trading buying power is 4x
    vol_halflife: int = 30
    vol_warmup: int = 20


def cfg_fields() -> set:
    return {f.name for f in fields(SimCfg)}


# ── one pair ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class PairResult:
    pair: tuple
    dates: pd.DatetimeIndex
    tf: int
    net_bar: np.ndarray
    gross_bar: np.ndarray
    cost_bar: np.ndarray
    hold: np.ndarray                        # position held during each bar
    dec: np.ndarray                         # position decided at each bar close
    reason: np.ndarray
    beta: np.ndarray
    unit_cost: np.ndarray                   # return units per unit change, per session
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    fill: str = "next_open"

    @property
    def daily_unit(self) -> np.ndarray:
        return self.net_bar.sum(axis=1)


def decide_bar(t: int, nb: int, tf: int, c: SimCfg, pos, age, cool, stopped, st, fin, hl_ok, edge_ok, move_block):
    """One decision step at the close of bar ``t`` for arrays of sessions (or a single live session). Returns (pos, age, cool, reason)."""
    mh = max(int(math.ceil(c.max_hold_min / tf)), 1)
    cool_bars = int(math.ceil(c.cooldown_min / tf))
    skip, last_entry = int(math.ceil(c.skip_open_min / tf)), nb - 1 - int(math.ceil(c.no_entry_min / tf))
    eod_dec = nb - 1 - int(math.ceil(c.flat_min / tf))
    age = np.where(pos != 0, age + 1, 0)
    long_run, short_run = (pos == 1) & fin & (st < -c.s_stop), (pos == -1) & fin & (st > c.s_stop)
    r = np.zeros(pos.shape, dtype=np.int8)
    ex = np.zeros(pos.shape, dtype=bool)
    for code, cond in ((1, ((pos == 1) & fin & (st > -c.s_close_long)) | ((pos == -1) & fin & (st < c.s_close_short))), (2, long_run | short_run), (3, (pos != 0) & (age >= mh)),
                       (5, (pos != 0) & ~fin), (6, (pos != 0) & stopped), (4, (pos != 0) & (t >= eod_dec))):
        new = cond & ~ex
        r = np.where(new, code, r).astype(np.int8)
        ex |= cond
    pos = np.where(ex, 0, pos)
    cooling = cool > 0                                                                   # bars left of the cooldown that started at an earlier stop
    cool = np.where(long_run | short_run, cool_bars, np.maximum(cool - 1, 0))
    can = (pos == 0) & ~ex & fin & hl_ok & edge_ok & ~stopped & ~move_block & ~cooling & (skip <= t) & (t <= last_entry) & (t < eod_dec)
    enter_l, enter_s = can & (st < -c.s_open), can & (st > c.s_open)
    pos = np.where(enter_l, 1, np.where(enter_s, -1, pos)).astype(np.int8)
    age = np.where(enter_l | enter_s, 0, age)
    return pos, age, cool, r


def entry_filters(s: np.ndarray, hl_min: np.ndarray, sig_eq_bp: np.ndarray, beta: np.ndarray, c: SimCfg, sig_mode: str, cL_bp: float, cB_bp: float):
    hl_ok = np.ones(s.shape, dtype=bool) if (c.max_halflife_min <= 0 or sig_mode == "z") else (np.isfinite(hl_min) & (hl_min <= c.max_halflife_min))
    if c.min_edge_ratio > 0:
        rt = 2.0 * (cL_bp + beta[:, None] * cB_bp)
        edge_ok = (np.abs(s) - 0.5 * (c.s_close_long + c.s_close_short)) * sig_eq_bp >= c.min_edge_ratio * rt
    else:
        edge_ok = np.ones(s.shape, dtype=bool)
    return hl_ok, edge_ok


def daily_sigma(close: np.ndarray, lag: int = 20) -> np.ndarray:
    """Trailing daily close-to-close sigma of the 1x ETF (known before the session)."""
    last = close[:, -1]
    r = pd.Series(np.concatenate([[np.nan], last[1:] / last[:-1] - 1.0]))
    return r.rolling(lag, min_periods=5).std().shift(1).to_numpy()


def leg_costs(pair: tuple, c: SimCfg) -> tuple[float, float]:
    return ((D.half_spread_bp(pair[0]) + c.slip_bp) * c.cost_mult, (D.half_spread_bp(pair[1]) + c.slip_bp) * c.cost_mult)


FILL_ALPHA = {"next_open": 1.0, "mid": 0.5, "close": 0.0}


def bar_gross(fill: str, hp, h, Lent, Bent, beta, Lc0, Lc1, Lo, Bc0, Bc1, Bo):
    """P&L of one bar per $1 of the 3x leg's notional AT ENTRY (the 1x leg holds beta x that in dollars at entry), with FIXED SHARES from the fill - what a real account earns.
    hp / h = position held before / during the bar; Lent / Bent = the fill prices of the open position; Lc0, Lc1, Lo = previous close, close and open of the bar (Bc0 ... for the 1x leg).
    A change of position is filled at  F = Lc0 + alpha x (Lo - Lc0): the old position earns Lc0 -> F, the new one F -> Lc1; an unchanged position earns close to close.
    Returns (pnl, new Lent, new Bent)."""
    a = FILL_ALPHA[fill]
    FL, FB = Lc0 + a * (Lo - Lc0), Bc0 + a * (Bo - Bc0)
    same = hp == h
    sl = np.where(Lent > 0, Lent, 1.0)
    sb = np.where(Bent > 0, Bent, 1.0)
    h, hp = h.astype(float), hp.astype(float)
    p_same = h * ((Lc1 - Lc0) / sl - beta * (Bc1 - Bc0) / sb)
    p_old = hp * ((FL - Lc0) / sl - beta * (FB - Bc0) / sb)
    p_new = h * ((Lc1 - FL) / FL - beta * (Bc1 - FB) / FB)
    pnl = np.where(same, p_same, p_old + p_new)
    Lent = np.where(same, Lent, np.where(h != 0, FL, 0.0))
    Bent = np.where(same, Bent, np.where(h != 0, FB, 0.0))
    return pnl, Lent, Bent


def filter_costs(pair: tuple, c: SimCfg) -> tuple[float, float]:
    return ((D.half_spread_bp(pair[0]) + c.filter_slip_bp) * c.filter_cost_mult, (D.half_spread_bp(pair[1]) + c.filter_slip_bp) * c.filter_cost_mult)


def simulate_pair(sig: PairSignals, c: SimCfg, progress: Optional[Callable] = None) -> PairResult:
    if c.fill not in FILL_ALPHA:
        raise ValueError(f"fill must be one of {list(FILL_ALPHA)}, not {c.fill!r}")
    if sig.L_open is None or sig.B_open is None:
        raise ValueError("these signals carry no open prices (an old cache): rebuild them")
    n, nb = sig.s.shape
    tf = sig.cfg.tf
    cL, cB = leg_costs(sig.pair, c)
    fL, fB = filter_costs(sig.pair, c)
    s, hl, se = sig.s, sig.hl_min, sig.sig_eq_bp
    d = c.delay_bars
    if d > 0:
        sh = lambda a: np.concatenate([np.full((n, d), np.nan), a[:, :-d]], axis=1)
        s, hl, se = sh(s), sh(hl), sh(se)
    fin = np.isfinite(s)
    hl_ok, edge_ok = entry_filters(s, hl, se, sig.beta, c, sig.cfg.s_mode, fL, fB)
    if c.move_z > 0:
        sg = daily_sigma(sig.B_close)
        move_block = np.isfinite(sg)[:, None] & (np.abs(sig.rB_day) > c.move_z * np.nan_to_num(sg, nan=1.0)[:, None])
    else:
        move_block = np.zeros((n, nb), dtype=bool)
    unit_cost = (cL + sig.beta * cB) / 1e4
    pos = np.zeros(n, dtype=np.int8)
    age, cool = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    stopped = np.zeros(n, dtype=bool)
    cum = np.zeros(n)
    hold_prev = np.zeros(n, dtype=np.int8)
    Lent, Bent = np.zeros(n), np.zeros(n)
    gross, cost_b = np.zeros((n, nb)), np.zeros((n, nb))
    hold, dec, reason = np.zeros((n, nb), dtype=np.int8), np.zeros((n, nb), dtype=np.int8), np.zeros((n, nb), dtype=np.int8)
    for t in range(nb):
        h = pos.copy()                                                              # position decided at the close of t-1 is held through bar t
        hold[:, t] = h
        if t >= 1:
            gross[:, t], Lent, Bent = bar_gross(c.fill, hold_prev, h, Lent, Bent, sig.beta, sig.L_close[:, t - 1], sig.L_close[:, t], sig.L_open[:, t], sig.B_close[:, t - 1], sig.B_close[:, t], sig.B_open[:, t])
        cost_b[:, t] = np.abs(h.astype(int) - hold_prev) * unit_cost
        cum += gross[:, t] - cost_b[:, t]
        if c.day_stop_bp > 0:
            stopped |= cum * 1e4 <= -c.day_stop_bp
        hold_prev = h
        pos, age, cool, r = decide_bar(t, nb, tf, c, pos, age, cool, stopped, s[:, t], fin[:, t], hl_ok[:, t], edge_ok[:, t], move_block[:, t])
        dec[:, t], reason[:, t] = pos, r
        if progress and t % 20 == 0:
            progress(t, nb)
    cost_b[:, nb - 1] += np.abs(hold_prev.astype(int)) * unit_cost                    # anything still held at the end of the session is closed there
    net = gross - cost_b
    res = PairResult(sig.pair, sig.dates, tf, net, gross, cost_b, hold, dec, reason, sig.beta, unit_cost, fill=c.fill)
    res.trades = extract_trades(res, sig)
    return res


TRADE_COLS = ["pair", "date", "side", "position", "entry_time", "exit_time", "hold_min", "long_leg", "short_leg", "L_entry", "L_exit", "B_entry", "B_exit", "beta", "gross_bp", "net_bp", "price_bp",
              "gap_in_bp", "gap_out_bp", "s_entry", "reason", "entry_bar", "bars"]
def extract_trades(r: PairResult, sig: PairSignals) -> pd.DataFrame:
    """One row per round trip: when it was opened and closed, which product was bought and which sold, the fill prices of both legs at entry and exit, the P&L from the bar accounting
    (``gross_bp`` / ``net_bp``, per $1 of the 3x leg, i.e. bp of its notional) and, as a cross-check you can do by hand, ``price_bp`` computed from the four fill prices with fixed shares.
    ``gap_in_bp`` / ``gap_out_bp``: how far the spread moved, in the trade's favour, between the signal bar's close and the next bar's open at entry / at exit - what a fill at the signal close
    would collect (in) or give up (out) compared with a fill at the next open."""
    n, nb = r.hold.shape
    tf = r.tf
    L, B = r.pair
    alpha = FILL_ALPHA.get(r.fill, 1.0)
    flat = r.hold.ravel().astype(int)
    prev = np.concatenate([[0], flat[:-1]])
    prev[::nb] = 0
    chg = np.flatnonzero(flat != prev)
    g, sflat, rsn = r.gross_bar.ravel(), sig.s.ravel(), r.reason.ravel()
    Lc, Bc = sig.L_close.ravel(), sig.B_close.ravel()
    Lo = sig.L_open.ravel() if sig.L_open is not None else Lc
    Bo = sig.B_open.ravel() if sig.B_open is not None else Bc
    clock = lambda day, bar: pd.Timestamp(r.dates[day]) + pd.Timedelta(minutes=D.OPEN_MIN + bar * tf)

    def fill(close, open_, i):                                                       # the price a change that starts at flat index i is executed at
        return close[i - 1] + alpha * (open_[i] - close[i - 1])

    def build(e_i: int, side: int, s_in: float, x_i: int, why: str):
        day = e_i // nb
        end_of_day = x_i >= (day + 1) * nb
        x_i = min(x_i, (day + 1) * nb)
        beta, cu = float(r.beta[day]), float(r.unit_cost[day])
        gb = g[e_i:min(x_i + 1, (day + 1) * nb)].sum()                                 # + the exit bar: its gap belongs to the position being closed
        Le, Be = fill(Lc, Lo, e_i), fill(Bc, Bo, e_i)
        if end_of_day:
            Lx, Bx, t_out = Lc[x_i - 1], Bc[x_i - 1], clock(day, nb)
            gap_out = 0.0
        else:
            Lx, Bx, t_out = fill(Lc, Lo, x_i), fill(Bc, Bo, x_i), clock(day, x_i % nb)
            gap_out = side * ((Lo[x_i] / Lc[x_i - 1] - 1.0) - beta * (Bo[x_i] / Bc[x_i - 1] - 1.0)) * 1e4
        gap_in = side * ((Lo[e_i] / Lc[e_i - 1] - 1.0) - beta * (Bo[e_i] / Bc[e_i - 1] - 1.0)) * 1e4
        long_leg, short_leg = (L, B) if side > 0 else (B, L)
        return dict(pair=D.pair_name(r.pair), date=r.dates[day], side="long" if side > 0 else "short", position=f"LONG {long_leg} / SHORT {short_leg}", entry_time=clock(day, e_i % nb), exit_time=t_out,
                    hold_min=int(x_i - e_i) * tf, long_leg=long_leg, short_leg=short_leg, L_entry=Le, L_exit=Lx, B_entry=Be, B_exit=Bx, beta=beta, gross_bp=gb * 1e4, net_bp=(gb - 2 * cu) * 1e4,
                    price_bp=side * ((Lx / Le - 1.0) - beta * (Bx / Be - 1.0)) * 1e4, gap_in_bp=gap_in, gap_out_bp=gap_out, s_entry=s_in, reason=why, entry_bar=e_i % nb, bars=int(x_i - e_i))

    rows, cur = [], None
    for i in chg:
        if cur is not None:                                                           # something was open: it ends at this change (or at the end of its session)
            e_i = cur["i"]
            x_i = int(i) if i // nb == e_i // nb else (e_i // nb + 1) * nb
            rows.append(build(e_i, cur["side"], cur["s"], x_i, EXIT.get(int(rsn[min(x_i, flat.size) - 1]), "")))
            cur = None
        if flat[i] != 0:
            cur = dict(i=int(i), side=int(flat[i]), s=sflat[i - 1] if i > 0 else np.nan)
    if cur is not None:
        e_i = cur["i"]
        rows.append(build(e_i, cur["side"], cur["s"], (e_i // nb + 1) * nb, "end of day"))
    return pd.DataFrame(rows, columns=TRADE_COLS)


# ── the book ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Result:
    cfg: SimCfg
    daily: pd.DataFrame                      # ret, u, G, cost, exposure, n_trades
    per_pair: dict                           # pair -> PairResult
    unit: pd.DataFrame                       # per-pair unit-gross daily returns
    weights: pd.DataFrame
    trades: pd.DataFrame

    @property
    def ret(self) -> pd.Series:
        return self.daily["ret"]


def simulate(sigs: dict, c: SimCfg, progress: Optional[Callable] = None, cancelled: Callable[[], bool] = lambda: False) -> Result:
    active = [(i, k) for i, k in enumerate(PAIR_ORDER) if k in sigs and i < len(c.pair_w) and c.pair_w[i] > 0]
    if not active:
        raise ValueError("no pair is enabled")
    per, unit, costs, expo, ntr = {}, {}, {}, {}, {}
    for j, (i, k) in enumerate(active):
        if cancelled():
            break
        r = simulate_pair(sigs[k], c)
        per[k] = r
        u = pd.Series(r.net_bar.sum(axis=1) / (1.0 + r.beta), index=r.dates)             # per $ of gross (L + beta B)
        unit[k] = u
        costs[k] = pd.Series(r.cost_bar.sum(axis=1) / (1.0 + r.beta), index=r.dates)
        expo[k] = pd.Series((r.hold != 0).mean(axis=1), index=r.dates)
        ntr[k] = pd.Series(np.abs(np.diff(np.concatenate([np.zeros((r.hold.shape[0], 1)), r.hold.astype(int)], axis=1), axis=1)).sum(axis=1) / 2.0, index=r.dates)
        if progress:
            progress(j + 1, len(active))
    U = pd.DataFrame(unit).sort_index()
    U = U[(U.index >= pd.Timestamp(c.start)) & ((U.index <= pd.Timestamp(c.end)) if c.end else True)].fillna(0.0)
    base = pd.Series({k: c.pair_w[i] for i, k in active if k in U.columns})
    if c.weight_mode == "inv_vol":
        vol = U.rolling(60, min_periods=20).std().shift(1)
        w = (1.0 / vol.replace(0, np.nan)).mul(base, axis=1)
        w = w.div(w.sum(axis=1), axis=0).fillna(base / base.sum())
    else:
        w = pd.DataFrame({k: base[k] / base.sum() for k in U.columns}, index=U.index)
    u = (U * w).sum(axis=1)
    G = unit_gross_leverage(u, c)
    ret = u * G
    cost = (pd.DataFrame(costs).reindex(U.index).fillna(0.0) * w).sum(axis=1) * G
    ex = (pd.DataFrame(expo).reindex(U.index).fillna(0.0) * w).sum(axis=1)
    nt = (pd.DataFrame(ntr).reindex(U.index).fillna(0.0)).sum(axis=1)
    daily = pd.DataFrame({"ret": ret, "u": u, "G": G, "cost": cost, "exposure": ex, "n_trades": nt})
    tr = pd.concat([r.trades for r in per.values()], ignore_index=True) if per else pd.DataFrame()
    if len(tr):
        tr = tr[(tr.date >= pd.Timestamp(c.start)) & ((tr.date <= pd.Timestamp(c.end)) if c.end else True)]
    return Result(c, daily, per, U, w, tr.reset_index(drop=True))


def unit_gross_leverage(u: pd.Series, c: SimCfg) -> pd.Series:
    """Gross leverage for each session from the EWMA volatility of the strategy's own unit-gross return, using only returns known before the session."""
    if not c.vol_on:
        return pd.Series(c.g_init, index=u.index)
    alpha = 1.0 - 0.5 ** (1.0 / max(c.vol_halflife, 1))
    var, n_seen, out = None, 0, []
    for x in u.to_numpy():
        if var is not None and n_seen >= c.vol_warmup and var > 0:
            out.append(float(np.clip(c.vol_target / math.sqrt(var * TD), c.g_min, c.g_max)))
        else:
            out.append(min(c.g_init, c.g_max))
        if x != 0.0 or n_seen > 0:
            var = x * x if var is None else (1 - alpha) * var + alpha * x * x
            n_seen += 1
    return pd.Series(out, index=u.index)


# ── metrics ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def psr(sr: float, n: int, skew: float, kurt: float, sr0: float = 0.0) -> float:
    den = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return _norm_cdf((sr - sr0) * math.sqrt(max(n - 1, 1)) / den)


def dsr(sr: float, n: int, skew: float, kurt: float, n_trials: int, sr_var: float) -> float:
    if n_trials <= 1:
        return psr(sr, n, skew, kurt, 0.0)
    from scipy.stats import norm
    g = 0.5772156649
    sr0 = math.sqrt(sr_var) * ((1 - g) * float(norm.ppf(1 - 1.0 / n_trials)) + g * float(norm.ppf(1 - 1.0 / (n_trials * math.e))))
    return psr(sr, n, skew, kurt, sr0)


def metrics(ret: pd.Series, bench: Optional[pd.Series] = None, n_trials: int = 1, sr_var: float = 0.0) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {"n": len(r)}
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(r) / TD
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    down = r[r < 0].std(ddof=1)
    sr_d = mu / sd if sd > 0 else 0.0
    skew, kurt = float(r.skew()), float(r.kurt() + 3.0)
    out = {"n": len(r), "years": yrs, "cagr": float(cagr), "mean_ann": float(mu * TD), "vol": float(sd * math.sqrt(TD)), "sharpe": float(sr_d * math.sqrt(TD)),
           "sortino": float(mu / down * math.sqrt(TD)) if down > 0 else float("nan"), "maxdd": float(dd.min()), "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else float("nan"),
           "hit": float((r > 0).mean()), "skew": skew, "kurt": kurt, "best_day": float(r.max()), "worst_day": float(r.min()), "worst_21d": float(r.rolling(21).sum().min()),
           "psr0": psr(sr_d, len(r), skew, kurt, 0.0), "total": float(eq.iloc[-1] - 1),
           "dsr": dsr(sr_d, len(r), skew, kurt, n_trials, sr_var) if n_trials > 1 and sr_var > 0 else float("nan")}
    out["sharpe_se"] = float(math.sqrt((1 + sr_d ** 2 / 2) / len(r)) * math.sqrt(TD))
    if bench is not None:
        b = bench.reindex(r.index).dropna()
        rr = r.reindex(b.index)
        if len(b) > 30 and b.std() > 0:
            beta = float(np.cov(rr, b)[0, 1] / b.var())
            out.update(beta=beta, alpha_ann=float((rr.mean() - beta * b.mean()) * TD), corr=float(np.corrcoef(rr, b)[0, 1]), bench_sharpe=float(b.mean() / b.std() * math.sqrt(TD)),
                       bench_cagr=float((1 + b).prod() ** (TD / len(b)) - 1), bench_maxdd=float(((1 + b).cumprod() / (1 + b).cumprod().cummax() - 1).min()))
    return out


def yearly(ret: pd.Series) -> pd.DataFrame:
    rows = []
    for y, g in ret.groupby(ret.index.year):
        eq = (1 + g).cumprod()
        rows.append({"year": int(y), "return": float(eq.iloc[-1] - 1), "vol": float(g.std() * math.sqrt(TD)), "sharpe": float(g.mean() / g.std() * math.sqrt(TD)) if g.std() > 0 else float("nan"),
                     "maxdd": float((eq / eq.cummax() - 1).min()), "days": len(g)})
    return pd.DataFrame(rows).set_index("year")


def fmt_metrics(m: dict) -> str:
    if m.get("n", 0) < 20:
        return "n/a"
    s = (f"CAGR {m['cagr']:+.1%} · vol {m['vol']:.1%} · Sharpe {m['sharpe']:.2f} (±{1.96 * m['sharpe_se']:.2f}) · Sortino {m['sortino']:.2f} · MaxDD {m['maxdd']:.1%} · "
         f"Calmar {m['calmar']:.2f} · PSR {m['psr0']:.0%}")
    if "beta" in m:
        s += f" · β {m['beta']:+.2f} · corr {m['corr']:+.2f}"
    return s
