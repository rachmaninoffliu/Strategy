"""
Glue between data / signals / simulator and any front end (GUI, tests, research scripts). No Qt.

Signals (stage A) are cached on disk per (pair, time frame, feed, SigCfg, last data date) for time frames of 5 minutes and up; everything after that (stage B) takes seconds.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

import levpairs_data as D
from levpairs_core import PairSignals, SigCfg, build_signals
from levpairs_sim import PAIR_ORDER, Result, SimCfg, fmt_metrics, metrics, simulate, yearly

CACHE_DIR = os.path.join(D.REPORTS, "cache")
CACHE_VERSION = "v3"           # bump when PairSignals changes: pickles of an older layout are not read
Panel = dict          # pair name -> PairData


def load_panel(tf: int, feed: str = "sip", start: Optional[str] = None, log: Callable[[str], None] = print) -> Panel:
    out = {}
    for p in D.PAIRS:
        d = D.load_pairdata(p, tf, feed, start=start)
        out[D.pair_name(p)] = d
        log(f"{D.pair_name(p)} {tf}m {feed}: {len(d.dates):,} sessions {d.dates[0]:%Y-%m-%d} .. {d.dates[-1]:%Y-%m-%d}" + (" · " + "; ".join(d.notes) if d.notes else ""))
    return out


def bench_spy(panel: Panel) -> pd.Series:
    """SPY close-to-close buy & hold, for context (the strategy itself is flat overnight)."""
    d = panel[PAIR_ORDER[2]]
    c = pd.Series(d.B_close[:, -1], index=d.dates)
    return c.pct_change().dropna()


def sig_path(name: str, sc: SigCfg, feed: str, last) -> str:
    h = hashlib.md5((CACHE_VERSION + name + feed + repr(sorted(asdict(sc).items())) + str(pd.Timestamp(last).date())).encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"sig_{name.replace('/', '-')}_{sc.tf}m_{h}.pkl")


def get_signals(panel: Panel, sc: SigCfg, feed: str = "sip", log: Callable[[str], None] = print, cancelled: Callable[[], bool] = lambda: False, use_cache: bool = True) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = {}
    for name, d in panel.items():
        if cancelled():
            break
        f = sig_path(name, sc, feed, d.last_date())
        if use_cache and sc.tf >= 5 and os.path.exists(f):
            try:
                out[name] = pickle.load(open(f, "rb"))
                continue
            except Exception:                                                           # noqa: BLE001
                pass
        t = time.time()
        s = build_signals(d, sc)
        out[name] = s
        if sc.tf >= 5:
            pickle.dump(s, open(f, "wb"), protocol=4)
            for old in os.listdir(CACHE_DIR):
                fp = os.path.join(CACHE_DIR, old)
                if old.startswith(f"sig_{name.replace('/', '-')}_{sc.tf}m_") and fp != f and time.time() - os.path.getmtime(fp) > 14 * 86400:
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
        log(f"signals {name} {sc.tf}m in {time.time() - t:.1f}s")
    return out


@dataclass
class BtOut:
    result: Result
    ret: pd.Series
    bench: pd.Series
    m_all: dict
    m_is: dict
    m_oos: dict
    split: pd.Timestamp
    yearly: pd.DataFrame
    equity: pd.Series
    capital: float
    pairs: pd.DataFrame
    notes: list = field(default_factory=list)


def pair_table(res: Result) -> pd.DataFrame:
    rows = []
    for k, pr in res.per_pair.items():
        u = res.unit[k] if k in res.unit else pd.Series(dtype=float)
        tr = res.trades[res.trades.pair == k] if len(res.trades) else pd.DataFrame()
        m = metrics(u)
        rows.append(dict(pair=k, sharpe=m.get("sharpe", np.nan), net_bp_day=float(u.mean() * 1e4) if len(u) else np.nan, trades=len(tr), win=float((tr.net_bp > 0).mean()) if len(tr) else np.nan,
                         avg_net_bp=float(tr.net_bp.mean()) if len(tr) else np.nan, avg_gross_bp=float(tr.gross_bp.mean()) if len(tr) else np.nan, avg_bars=float(tr.bars.mean()) if len(tr) else np.nan,
                         cost_bp_day=float(pr.cost_bar.sum(axis=1).mean() * 1e4), tf=pr.tf))
    return pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()


def run_backtest(panel: Panel, sigs: dict, c: SimCfg, split: Optional[str] = "2022-01-01", progress=None, cancelled=lambda: False) -> BtOut:
    res = simulate(sigs, c, progress=progress, cancelled=cancelled)
    ret = res.ret
    spy = bench_spy(panel)
    split_ts = pd.Timestamp(split) if split else ret.index[len(ret) // 2]
    m_all, m_is, m_oos = metrics(ret, spy), metrics(ret[ret.index < split_ts], spy), metrics(ret[ret.index >= split_ts], spy)
    notes = []
    if len(res.trades):
        t = res.trades
        notes.append(f"{len(t):,} round trips · win rate {(t.net_bp > 0).mean():.0%} · average {t.net_bp.mean():+.1f} bp net ({t.gross_bp.mean():+.1f} gross) per $1 of the 3x leg · average hold {t.bars.mean() * next(iter(res.per_pair.values())).tf:.0f} min")
    return BtOut(res, ret, spy, m_all, m_is, m_oos, split_ts, yearly(ret) if len(ret) > 20 else pd.DataFrame(), (1 + ret).cumprod() * c.capital, c.capital, pair_table(res), notes)


# ── ablation / sensitivity / cost / time frames ──────────────────────────────────────────────────────────────────────────────────────────────────────
PAPER_RULES = dict(s_open=1.25, s_close_long=0.50, s_close_short=0.75, s_stop=99.0, max_hold_min=10_000, max_halflife_min=0.0, min_edge_ratio=0.0, skip_open_min=0, no_entry_min=0, cooldown_min=0,
                   move_z=0.0, day_stop_bp=0.0, vol_on=False, g_init=1.0, weight_mode="equal")


def _row(name: str, r: pd.Series, bench: pd.Series, sp: pd.Timestamp) -> dict:
    m, mo = metrics(r, bench), metrics(r[r.index >= sp], bench)
    return dict(variant=name, sharpe=m.get("sharpe"), cagr=m.get("cagr"), vol=m.get("vol"), maxdd=m.get("maxdd"), beta=m.get("beta"), sharpe_oos=mo.get("sharpe"), cagr_oos=mo.get("cagr"))


def ablation(panel: Panel, sc: SigCfg, base: SimCfg, feed: str = "sip", split: str = "2022-01-01", log: Callable[[str], None] = print, cancelled=lambda: False) -> pd.DataFrame:
    bench, sp = bench_spy(panel), pd.Timestamp(split)
    sigs = get_signals(panel, sc, feed, log=lambda m: None)
    steps: list = [("CURRENT SETTING", base, sigs), ("A&L 2008 rules (fixed thresholds, no filters, no vol target)", replace(base, **PAPER_RULES), sigs)]
    toggles = [("fill at the midpoint of the signal close and the next open", dict(fill="mid")), ("fill at the signal bar's last trade (optimistic)", dict(fill="close")),
               ("no half-life filter", dict(max_halflife_min=0.0)), ("no edge / cost filter", dict(min_edge_ratio=0.0)), ("no model-failure stop", dict(s_stop=99.0)), ("no cooldown after a stop", dict(cooldown_min=0)),
               ("trade the open (no 15-min wait)", dict(skip_open_min=0)), ("no volatility target (fixed gross)", dict(vol_on=False)), ("equal pair weights", dict(weight_mode="equal")),
               ("inverse-vol pair weights", dict(weight_mode="inv_vol")), ("one bar of latency", dict(delay_bars=base.delay_bars + 1)), ("costs x2", dict(cost_mult=base.cost_mult * 2)),
               ("costs x0.5", dict(cost_mult=base.cost_mult * 0.5)), ("costs x0.1", dict(cost_mult=base.cost_mult * 0.1)), ("zero cost, same trades", dict(cost_mult=0.0, slip_bp=0.0)),
               ("zero cost, every signal (no cost filter)", dict(cost_mult=0.0, slip_bp=0.0, min_edge_ratio=0.0))]
    steps += [(n, replace(base, **kw), sigs) for n, kw in toggles if any(getattr(base, k) != v for k, v in kw.items())]
    for name, kw in (("fixed beta 3.0", dict(beta_mode="fixed")), ("'reset' spread (to the prior close)", dict(spread="reset")), ("plain z-score instead of OU fit", dict(s_mode="z")),
                     ("OU window x2", dict(win_days=sc.win_days * 2)), ("OU window /2", dict(win_days=max(sc.win_days / 2, 1.0)))):
        if any(getattr(sc, k) != v for k, v in kw.items()):
            steps.append((name, base, get_signals(panel, replace(sc, **kw), feed, log=lambda m: None)))
    for i, k in enumerate(PAIR_ORDER):
        if base.pair_w[i] > 0 and sum(1 for w in base.pair_w if w > 0) > 1:
            pw = tuple(1.0 if j == i else 0.0 for j in range(3))
            steps.append((f"only {k}", replace(base, pair_w=pw), sigs))
    rows = []
    for name, c, sg in steps:
        if cancelled():
            break
        r = simulate(sg, c).ret
        rows.append(_row(name, r, bench, sp))
        log(f"  {name}: Sharpe {rows[-1]['sharpe']:.2f}")
    return pd.DataFrame(rows).set_index("variant")


def sensitivity(panel: Panel, sigs: dict, base: SimCfg, log: Callable[[str], None] = print, cancelled=lambda: False) -> pd.DataFrame:
    bench = bench_spy(panel)
    rows = []
    for so in (base.s_open - 0.5, base.s_open - 0.25, base.s_open, base.s_open + 0.25, base.s_open + 0.5):
        for sc_ in (0.0, 0.25, 0.5, 1.0):
            if cancelled():
                break
            if so <= sc_ + 0.2:
                continue
            c = replace(base, s_open=so, s_close_long=sc_, s_close_short=sc_)
            m = metrics(simulate(sigs, c).ret, bench)
            rows.append(dict(s_open=so, s_close=sc_, sharpe=m.get("sharpe"), cagr=m.get("cagr"), maxdd=m.get("maxdd")))
    return pd.DataFrame(rows)


def cost_breakeven(panel: Panel, sigs: dict, base: SimCfg, slips=(0.0, 0.3, 0.6, 1.0, 1.5, 2.5)) -> pd.DataFrame:
    bench = bench_spy(panel)
    rows = []
    for sl in slips:
        m = metrics(simulate(sigs, replace(base, slip_bp=sl)).ret, bench)
        rows.append(dict(slip_bp=sl, sharpe=m.get("sharpe"), cagr=m.get("cagr"), maxdd=m.get("maxdd")))
    return pd.DataFrame(rows)


FILL_LABEL = {"next_open": "at the OPEN of the next bar (realistic)", "mid": "half way: midpoint of the signal close and the next open", "close": "at the signal bar's last trade (optimistic)"}


def fill_comparison(panel: Panel, sigs: dict, base: SimCfg, split: str = "2022-01-01") -> tuple[pd.DataFrame, dict]:
    """The same signals and the same trades, accounted with three fill prices. The decisions do not depend on the fill, so the trade lists are identical and every difference is the price the orders
    get. Also returns how far the spread moves, in the trade's favour, between the signal close and the next open, at entry and at exit (what a close fill would collect / give up)."""
    bench, sp = bench_spy(panel), pd.Timestamp(split)
    rows, gaps = [], {}
    for fill in ("next_open", "mid", "close"):
        res = simulate(sigs, replace(base, fill=fill))
        r, tr = res.ret, res.trades
        m, mo = metrics(r, bench), metrics(r[r.index >= sp], bench)
        rows.append(dict(fill=fill, label=FILL_LABEL[fill], sharpe=m.get("sharpe"), sharpe_oos=mo.get("sharpe"), trades=len(tr), gross_bp=float(tr.gross_bp.mean()) if len(tr) else np.nan,
                         net_bp=float(tr.net_bp.mean()) if len(tr) else np.nan))
        if len(tr):
            gaps = dict(gap_in=float(tr.gap_in_bp.mean()), gap_out=float(tr.gap_out_bp.mean()), trades=len(tr))
    return pd.DataFrame(rows).set_index("fill"), gaps


def timeframe_scan(sc: SigCfg, base: SimCfg, feed: str = "sip", split: str = "2022-01-01", tfs=D.TIMEFRAMES, log: Callable[[str], None] = print, cancelled=lambda: False) -> pd.DataFrame:
    """Same rules (all times are in minutes) on every time frame. Returns Sharpe / CAGR in sample and out of sample, trades and costs."""
    sp = pd.Timestamp(split)
    rows = []
    for tf in tfs:
        if cancelled():
            break
        try:
            panel = load_panel(tf, feed, log=lambda m: None)
            sg = get_signals(panel, replace(sc, tf=tf), feed, log=lambda m: None)
            res = simulate(sg, base)
        except Exception as e:                                                          # noqa: BLE001
            log(f"{tf}m failed: {type(e).__name__}: {str(e)[:100]}")
            continue
        r, bench = res.ret, bench_spy(panel)
        m, mi, mo = metrics(r, bench), metrics(r[r.index < sp], bench), metrics(r[r.index >= sp], bench)
        rows.append(dict(tf=tf, sharpe=m.get("sharpe"), sharpe_is=mi.get("sharpe"), sharpe_oos=mo.get("sharpe"), cagr=m.get("cagr"), cagr_oos=mo.get("cagr"), maxdd=m.get("maxdd"),
                         trades=len(res.trades), net_bp=float(res.trades.net_bp.mean()) if len(res.trades) else np.nan, gross_bp=float(res.trades.gross_bp.mean()) if len(res.trades) else np.nan,
                         cost_pct=float(res.daily.cost.mean() * 252)))
        log(f"  {tf}m: Sharpe {rows[-1]['sharpe']:.2f} (IS {rows[-1]['sharpe_is']:.2f}, OOS {rows[-1]['sharpe_oos']:.2f}) · {rows[-1]['trades']:,} trades")
    return pd.DataFrame(rows).set_index("tf")
