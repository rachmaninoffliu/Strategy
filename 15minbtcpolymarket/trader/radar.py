"""Regime radar: where in the day is the edge *today*?  (Qt-free; used by the GUI tab and the pre-flight check.)

Every hour of the UTC day is scouted continuously with the same machinery the all-hours adaptive engine would use - underdog signals in the strategy's price band at
the decision minutes, net of slippage and fee, an exponentially decaying memory, hour estimates shrunk toward the all-hours estimate - but nothing is traded. It answers:

  * is the hour(s) I am trading still hot?                 (status of the active hours)
  * has the edge moved to another hour?                    (TRADE / WATCH hours outside the active set)
  * is there a market-wide shift (all hours pooled)?       (the 'ALL' row; pooled over ~24x more signals, so it is detected much faster than any single hour)

One hour yields only ~1 signal per day, so a single hour's edge cannot be distinguished from luck unless it is large (>~10c); the radar therefore reports the
shrunk estimate the engine would actually use, not just the raw trailing average.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data import extract_signals
from .evfilter import DEFAULT_MIN_N, TrailingEVStats
from .fees import fee_per_contract
from .probmodel import SignalStats
from .strategy import StrategyParams

SCOUT_HALFLIFE = 14.0
SCOUT_HOUR_K = 30.0
SCOUT_PRIOR_N = 20.0
LUCK_Z = 2.9          # one hour out of 24 searched: its raw trailing EV needs z = EV / se of about 2.9 (Bonferroni, 5%) before it can be told from luck

EVIDENCE_NOTE = ("Why the 14 d / 28 d columns can look good while the engine estimate does not: an hour yields only ~1 signal a day (se about 8-11c), so among 24 hours several look good by luck. "
                 "Walk-forward test (ev_walkforward.py, Mar-Oct 2026, 4,788 hour-days, point-in-time): an hour's raw trailing EV had ~0 correlation with what it earned in the NEXT 14 days "
                 "(top fifth -3.7c, bottom fifth -4.2c); hours that showed +8c or more on BOTH 14 d and 28 d (average +14.7c / +20.3c) then earned -2.1c - the shrunk engine estimate for them (-1.6c) was close, the raw average was ~16c off. "
                 "The estimate is deliberately low; the raw dots are shown for transparency, not as a signal.")


@dataclass(frozen=True)
class RadarConfig:
    cost_min: float = 0.15
    cost_max: float = 0.35
    minutes: tuple = (12, 13)
    fee_model: str = "Kalshi"
    slippage: float = 0.02
    min_edge: float = 0.04              # same conviction hurdle as the engine
    halflife_days: float = SCOUT_HALFLIFE
    hour_k: float = SCOUT_HOUR_K
    prior_n: float = SCOUT_PRIOR_N
    core_hours: tuple = ()               # the engine's own hour shrinkage (GUI / pre-flight pass the live settings), so a radar 'TRADE' means the engine's gate is open for a typical signal
    core_k: float = 0.0
    near_k: float = 0.0
    near_width: int = 2
    window_days: int = 28
    ev_rules: tuple = ()                 # EV-regime filter of the preset (trader.evfilter): when set, 'TRADE' means the hour passes every rule and the estimate column shows the filter's own EV
    ev_min_n: int = DEFAULT_MIN_N


def _net_pnl(s: pd.DataFrame, cfg: RadarConfig) -> np.ndarray:
    px = np.minimum(s.cost.values + cfg.slippage, 0.99)
    fee = np.array([fee_per_contract(cfg.fee_model, float(c)) for c in px]) if len(px) else np.array([])
    return s.win.values - px - fee


def _stats(pnl: np.ndarray, s: pd.DataFrame) -> dict:
    n = len(pnl)
    return dict(n=n, win=float(s.win.mean()) if n else np.nan, price=float(s.cost.mean()) if n else np.nan,
                ev=float(pnl.mean()) if n else np.nan, se=float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan)


def hour_radar(rows: pd.DataFrame, now: float, cfg: RadarConfig = RadarConfig(), active_hours: tuple = ()) -> pd.DataFrame:
    """One row per UTC hour (0-23) plus an 'ALL' row.  Columns: hour, active, n, win, price, ev (net, trailing window), se, z (= ev / se), n14, ev14, est_edge, status.
    est_edge = what the engine would use: net edge (p - all-in cost) of the shrunk, decayed estimate at the middle of the band.  status: TRADE / WATCH / cold."""
    p = StrategyParams(hours=tuple(range(24)), minutes=cfg.minutes, cost_min=cfg.cost_min, cost_max=cfg.cost_max, legs=("DOG",))
    sig = extract_signals(rows, p)
    sig = sig[sig.close_ts <= now]
    stats = SignalStats.from_signals(sig, mode="by_hour", hour_k=cfg.hour_k, prior_n=cfg.prior_n, halflife_days=cfg.halflife_days, basis="edge", core_hours=cfg.core_hours, core_k=cfg.core_k,
                                     near_k=cfg.near_k, near_width=cfg.near_width)
    typ = (cfg.cost_min + cfg.cost_max) / 2
    px = min(typ + cfg.slippage, 0.99)
    cost_all = px + fee_per_contract(cfg.fee_model, px)
    evs = TrailingEVStats.from_signals(sig, cfg.ev_rules, cfg.ev_min_n, cfg.slippage, cfg.fee_model) if cfg.ev_rules else None
    w = sig[sig.open_ts >= now - cfg.window_days * 86400]
    w14 = sig[sig.open_ts >= now - 14 * 86400]
    out = []
    for h in list(range(24)) + [None]:
        a = w if h is None else w[w.hour == h]
        b = w14 if h is None else w14[w14.hour == h]
        st, st14 = _stats(_net_pnl(a, cfg), a), _stats(_net_pnl(b, cfg), b)
        if evs is not None:                                                                # the preset trades by EV filter: show what the filter sees
            if h is None:
                est, status = float("nan"), "cold"
            else:
                ok, _g, parts = evs.check(h, now)
                nets = [net for _d, n, net, _t, _ok in parts if n]
                est, status = (min(nets) if nets else float("nan")), ("TRADE" if ok else "cold")
            out.append(dict(hour="ALL" if h is None else h, active=(h in active_hours) if h is not None else False, n=st["n"], win=st["win"], price=st["price"], ev=st["ev"], se=st["se"],
                            z=(st["ev"] / st["se"]) if (st["n"] > 1 and st["se"] and st["se"] > 0) else np.nan, n14=st14["n"], ev14=st14["ev"], est_edge=est, status=status))
            continue
        if h is None:
            pr, _n, _nh = stats.p("DOG", 22, typ, now)                                    # overall level: the hour-shrinkage target
            d = stats.decayed(("DOG", None), now)
            est = (d[2] / (d[0] + cfg.prior_n)) + typ - cost_all if d[0] + cfg.prior_n > 0 else np.nan
        else:
            pr, _n, _nh = stats.p("DOG", h, typ, now)
            est = pr - cost_all
        status = "TRADE" if est >= cfg.min_edge else "WATCH" if est >= cfg.min_edge / 2 else "cold"
        out.append(dict(hour="ALL" if h is None else h, active=(h in active_hours) if h is not None else False, n=st["n"], win=st["win"], price=st["price"], ev=st["ev"], se=st["se"],
                        z=(st["ev"] / st["se"]) if (st["n"] > 1 and st["se"] and st["se"] > 0) else np.nan, n14=st14["n"], ev14=st14["ev"], est_edge=est, status=status))
    return pd.DataFrame(out)


def luck_note(df: pd.DataFrame, cfg: RadarConfig = RadarConfig()) -> str:
    """One sentence on the raw trailing columns: the best hour's EV and z, and whether any hour clears the bar for 'not just luck among 24 hours'."""
    h = df[(df.hour != "ALL") & df.z.notna()]
    if h.empty:
        return "Raw trailing EV by hour: not enough signals."
    best = h.loc[h.z.idxmax()]
    ok = h[h.z >= LUCK_Z]
    verdict = (f"{len(ok)} hour(s) clear it: {', '.join(str(int(x)) for x in ok.hour)}." if len(ok) else "none clears it - every raw edge is consistent with luck.")
    return (f"Raw {cfg.window_days}-day EV by hour: the best is hour {int(best.hour)} at {best.ev * 100:+.1f}c (z = {best.z:.1f}); with 24 hours searched an edge needs z >= {LUCK_Z:g} to be told from luck - {verdict}")


def summarize(df: pd.DataFrame, active_hours: tuple, cfg: RadarConfig = RadarConfig()) -> tuple[str, str]:
    """(status, one-line text) for the pre-flight check.  status: 'ok' | 'moved' | 'cold' | 'warm'."""
    hrs = df[df.hour != "ALL"].copy()
    hrs["hour"] = hrs.hour.astype(int)
    act = hrs[hrs.hour.isin(active_hours)]
    oth = hrs[~hrs.hour.isin(active_hours)]
    allr = df[df.hour == "ALL"].iloc[0]
    act_trade = act[act.status == "TRADE"].hour.tolist()
    oth_trade = oth[oth.status == "TRADE"].hour.tolist()
    oth_watch = oth[oth.status == "WATCH"].hour.tolist()
    base = f"all hours pooled: net EV {allr.ev * 100:+.1f}c over {int(allr.n)} signals ({cfg.window_days} d)"
    if act_trade and not oth_trade:
        return "ok", f"your hours {','.join(map(str, act_trade))} are hot; no other hour qualifies. {base}."
    if oth_trade and not act_trade:
        return "moved", f"your hours are NOT hot but hour(s) {','.join(map(str, oth_trade))} qualify: the edge may have moved. {base}. Consider the all-hours adaptive preset."
    if oth_trade and act_trade:
        return "ok", f"your hours {','.join(map(str, act_trade))} are hot and hour(s) {','.join(map(str, oth_trade))} also qualify (only traded by an all-hours preset). {base}."
    if oth_watch or (act.status == "WATCH").any():
        return "warm", f"nothing qualifies yet; on watch: {','.join(map(str, sorted(set(oth_watch + act[act.status == 'WATCH'].hour.tolist()))))}. {base}."
    return "cold", f"no hour of the day currently shows a tradable edge. {base}."
