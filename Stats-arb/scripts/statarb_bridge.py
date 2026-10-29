"""
Glue between the data / model / simulator and any front end (GUI, tests, the research scripts): cached signals, one-call backtests, ablations. No Qt.

Signals (stage A) are the expensive part (about a minute on a multi-core PC), so they are cached on disk per (universe, FactorCfg, last data date). Everything after that (stage B) takes seconds.
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

import statarb_data as D
from statarb_core import FactorCfg, PortCfg, Signals, build_signals
from statarb_sim import Result, fmt_metrics, metrics, simulate, spy_open_to_open, yearly

CACHE_DIR = os.path.join(D.REPORTS, "cache")
UNIVERSES = {"core": "Curated large caps (today's constituents)", "broad": "Broad, survivorship-free (incl. delisted)"}


def load_panel(universe: str = "core", refresh: bool = False, log: Callable[[str], None] = print) -> D.Panel:
    if universe == "broad":
        import statarb_broad as B
        if refresh:
            B.update_broad_cache(log=log)
        return B.load_broad_panel(log)
    return D.load_panel(refresh=refresh, log=log)


def earnings_fingerprint() -> str:
    try:
        e = D.load_earnings()
        return f"{len(e)}-{e.symbol.nunique()}-{pd.Timestamp(e.ts.max()).date() if len(e) else ''}"
    except Exception:                                                                  # noqa: BLE001
        return "none"


def cfg_key(universe: str, f: FactorCfg, last_date) -> str:
    h = hashlib.md5((universe + repr(sorted(asdict(f).items())) + str(pd.Timestamp(last_date).date()) + earnings_fingerprint()).encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"sig_{universe}_{h}.pkl")


def get_signals(panel: D.Panel, universe: str, f: FactorCfg, earn: Optional[pd.DataFrame] = None, log: Callable[[str], None] = print,
                progress: Optional[Callable[[int, int], None]] = None, cancelled: Callable[[], bool] = lambda: False, n_jobs: Optional[int] = None, use_cache: bool = True) -> Signals:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cfg_key(universe, f, panel.dates[-1])
    if use_cache and os.path.exists(path):
        try:
            sig = pickle.load(open(path, "rb"))
            log(f"signals loaded from cache ({os.path.basename(path)})")
            return sig
        except Exception:                                                              # noqa: BLE001
            pass
    if earn is None:
        earn = D.load_earnings()
    n_jobs = n_jobs if n_jobs is not None else max(1, min(8, (os.cpu_count() or 2) - 1))
    log(f"building signals: {len(panel.stocks)} names x {len(panel.dates)} sessions on {n_jobs} worker(s) ...")
    t = time.time()
    sig = build_signals(panel, f, earnings=earn, progress=progress, cancelled=cancelled, n_jobs=n_jobs)
    log(f"signals built in {time.time() - t:.0f}s")
    if not cancelled():
        pickle.dump(sig, open(path, "wb"), protocol=4)
        for old in os.listdir(CACHE_DIR):                                              # keep the cache from growing without bound: drop stale files of this universe
            fp = os.path.join(CACHE_DIR, old)
            if old.startswith(f"sig_{universe}_") and fp != path and time.time() - os.path.getmtime(fp) > 14 * 86400:
                try:
                    os.remove(fp)
                except OSError:
                    pass
    return sig


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
    notes: list = field(default_factory=list)


def run_backtest(panel: D.Panel, sig: Signals, p: PortCfg, split: Optional[str] = "2022-01-01", progress=None, cancelled=lambda: False) -> BtOut:
    res = simulate(sig, panel, p, progress=progress, cancelled=cancelled)
    ret = res.ret
    spy = spy_open_to_open(panel)
    split_ts = pd.Timestamp(split) if split else ret.index[len(ret) // 2]
    m_all = metrics(ret, spy)
    m_is = metrics(ret[ret.index < split_ts], spy)
    m_oos = metrics(ret[ret.index >= split_ts], spy)
    eq = (1 + ret).cumprod() * p.capital
    notes = []
    if len(res.trades):
        notes.append(f"{len(res.trades):,} closed trades · win rate {(res.trades.ret > 0).mean():.0%} · average {res.trades.ret.mean() * 1e4:+.0f} bp · average hold {res.trades.days.mean():.1f} sessions")
    return BtOut(result=res, ret=ret, bench=spy, m_all=m_all, m_is=m_is, m_oos=m_oos, split=split_ts, yearly=yearly(ret) if len(ret) > 20 else pd.DataFrame(), equity=eq, capital=p.capital, notes=notes)


# ── ablation: what does each ingredient add? ────────────────────────────────────────
PAPER_RULES = dict(w_rev=1.0, w_eap=0.0, neutral=False, style_neutral=False, macro_neutral=False, event_filter=False, earn_filter=False, weight_pow=0.0, shock=False, band=0.0, scale_band=0.0, vol_target=1.0,
                   g_init=1.0, g_min=1.0, g_max=1.0, name_cap=1.0, vol_boost=0.0, allow_short=True, hedge="none", borrow_bps=0.0)


def ablation(panel: D.Panel, sig: Signals, base: PortCfg, split: str = "2022-01-01", log: Callable[[str], None] = print, cancelled=lambda: False) -> pd.DataFrame:
    """Leave-one-out against the current setting, plus the paper's own rules at the same costs. Returns Sharpe / CAGR / vol / max drawdown, all-sample and out-of-sample."""
    steps: list[tuple[str, dict]] = [("CURRENT SETTING", {}), ("A&L 2008 rules (equal $, no filters, no overlays)", PAPER_RULES)]
    toggles = [("no earnings-premium sleeve (reversion only)", dict(w_eap=0.0, w_rev=max(base.w_rev, 1.0))), ("no reversion sleeve (earnings premium only)", dict(w_rev=0.0, w_eap=max(base.w_eap, 1.0))),
               ("earnings sleeve unhedged", dict(eap_hedge=False)), ("earnings sleeve: all announcers (no OU screen)", dict(eap_pool="all")),
               ("no news / post-earnings filter", dict(event_filter=False)), ("no earnings-window filter", dict(earn_filter=False)), ("no factor / sector neutrality", dict(neutral=False)),
               ("no style (mom / vol / size) neutrality", dict(style_neutral=False)), ("no macro (oil / rates / USD) neutrality", dict(macro_neutral=False)),
               ("no volatility target", dict(vol_target=1.0, g_init=1.0, g_min=1.0)), ("no geopolitical shock multiplier", dict(shock=False)),
               ("equal-dollar sizing (no 1/vol)", dict(weight_pow=0.0)), ("no volume boost", dict(vol_boost=0.0)), ("no no-trade band", dict(band=0.0, scale_band=0.0)),
               ("costs x2", dict(tc_bps=base.tc_bps * 2)), ("costs x0.5", dict(tc_bps=base.tc_bps * 0.5))]
    steps += [(n, kw) for n, kw in toggles if any(getattr(base, k) != v for k, v in kw.items())]
    spy = spy_open_to_open(panel)
    sp = pd.Timestamp(split)
    rows = []
    for name, kw in steps:
        if cancelled():
            break
        p = replace(base, **kw)
        r = simulate(sig, panel, p).ret
        m, mo = metrics(r, spy), metrics(r[r.index >= sp], spy)
        rows.append(dict(variant=name, sharpe=m.get("sharpe"), cagr=m.get("cagr"), vol=m.get("vol"), maxdd=m.get("maxdd"), beta=m.get("beta"), sharpe_oos=mo.get("sharpe"), cagr_oos=mo.get("cagr")))
        log(f"  {name}: Sharpe {m.get('sharpe', float('nan')):.2f}")
    return pd.DataFrame(rows).set_index("variant")


def sensitivity(panel: D.Panel, sig: Signals, base: PortCfg, log: Callable[[str], None] = print, cancelled=lambda: False) -> pd.DataFrame:
    """Neighbourhood of the main knobs (entry threshold x close threshold) - a plateau means the result is not a parameter spike."""
    spy = spy_open_to_open(panel)
    rows = []
    for so in (base.s_open - 0.25, base.s_open, base.s_open + 0.25, base.s_open + 0.5):
        for sc in (0.25, 0.5, 0.75):
            if cancelled():
                break
            p = replace(base, s_open=so, s_close_long=sc, s_close_short=sc + 0.25)
            r = simulate(sig, panel, p).ret
            m = metrics(r, spy)
            rows.append(dict(s_open=so, s_close=sc, sharpe=m.get("sharpe"), cagr=m.get("cagr"), maxdd=m.get("maxdd")))
    return pd.DataFrame(rows)


def cost_breakeven(panel: D.Panel, sig: Signals, base: PortCfg, costs=(0.0, 1.0, 1.5, 2.5, 4.0, 6.0)) -> pd.DataFrame:
    spy = spy_open_to_open(panel)
    rows = []
    for c in costs:
        r = simulate(sig, panel, replace(base, tc_bps=c)).ret
        m = metrics(r, spy)
        rows.append(dict(tc_bps=c, sharpe=m.get("sharpe"), cagr=m.get("cagr"), maxdd=m.get("maxdd")))
    return pd.DataFrame(rows)
