"""Backtest runner and the robustness (neighbour) check - no Qt, so a worker thread or a test can call it directly."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

import supertrend_gui as GU
import supertrend_strategy as S

from . import data as D
from . import metrics as M
from .params import BacktestSettings, to_config


@dataclass
class BtResult:
    settings: BacktestSettings
    values: dict
    cfg: S.Config
    eq: pd.Series
    bh: Optional[pd.Series]
    trades: pd.DataFrame
    metrics: dict
    first_bar: Optional[pd.Timestamp] = None
    last_bar: Optional[pd.Timestamp] = None
    n_bars: int = 0
    warm_bars: int = 0
    seconds: float = 0.0
    notes: list = field(default_factory=list)
    open_trade: Optional[dict] = None      # the position the backtest still holds after its last bar (supertrend_gui.backtest_open_view shape), None = flat


def bars_per_year(tf: int) -> int:
    return round(390 / tf) * 252


def enrich_trades(tr: pd.DataFrame) -> pd.DataFrame:
    if tr is None or tr.empty:
        return pd.DataFrame(columns=["side", "entry_time", "price", "qty", "exit_time", "exit_price", "exit_reason", "return_pct", "pnl_usd", "equity_after"])
    t = tr.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    return t.reset_index(drop=True)


def run_backtest(bars: pd.DataFrame, values: dict, bt: BacktestSettings, log: Callable = lambda *_: None) -> BtResult:
    """Run the strategy over [bt.start, bt.end]. With ``bt.prime`` the bars BEFORE the start date prime every indicator and no trade is taken before it, so a late start date
    behaves like the live engine (which also starts with a warm Supertrend) instead of sitting out the first ~19 sessions."""
    t0 = time.time()
    warm = D.warmup_bars(values) if bt.prime else 0
    sub, first = D.window(bars, bt.start, bt.end, warm)
    if first is None or len(sub) < 50:
        raise ValueError(f"no {bt.symbol} bars between {bt.start} and {bt.end} (the data covers {D.describe_bars(bars)})")
    extra = dict(initial_capital=float(bt.capital), strict_entries=bool(bt.strict))
    if bt.prime:
        extra.update(useDateFilter=True, startDate=str(first), endDate="2099-12-31")
    cfg = to_config(values, **extra)
    log(f"  simulating {len(sub):,} bars ({warm:,} of them warm-up) …")
    strat = S.SupertrendStrategy(cfg)
    eq, tr = strat.run(sub)
    eq = eq[eq.index >= first]
    if len(eq) < 2:
        raise ValueError("the window is too short to simulate (needs at least a few bars after the warm-up)")
    trades = enrich_trades(tr)
    shown = sub[sub.index >= first]
    bh = (shown["close"] / float(shown["close"].iloc[0]) * float(bt.capital)).reindex(eq.index).ffill() if bt.benchmark else None
    m = M.compute_metrics(eq, trades, float(bt.capital), bars_per_year(bt.tf), float(values.get("commission_bps", 0.0)), bh)
    notes = []
    before = len(sub) - len(shown)
    if bt.prime and 0 < before < warm:
        notes.append(f"only {before:,} bars exist before {first:%Y-%m-%d}: the indicators were primed with less than the full {warm:,}-bar warm-up")
    if not len(trades):
        notes.append("the strategy took no trades in this window")
    open_trade = GU.backtest_open_view(strat.open_trade)
    return BtResult(settings=bt, values=dict(values), cfg=cfg, eq=eq, bh=bh, trades=trades, metrics=m, first_bar=first, last_bar=shown.index[-1], n_bars=len(shown),
                    warm_bars=len(sub) - len(shown), seconds=time.time() - t0, notes=notes, open_trade=open_trade)


# ── robustness: is the result a plateau or a spike? ───────────────────────────────
@dataclass
class Robustness:
    factors: list
    atrs: list
    sharpe: np.ndarray                         # rows = atr periods, columns = factors
    cagr: np.ndarray
    maxdd: np.ndarray
    trades: np.ndarray
    centre: tuple                              # (row, col) of the current setting
    verdict: str = ""
    level: str = "info"                        # ok | warn | bad | info
    seconds: float = 0.0


def robustness(bars: pd.DataFrame, values: dict, bt: BacktestSettings, log: Callable = lambda *_: None,
               progress: Callable = lambda i, n: None, cancelled: Callable = lambda: False) -> Robustness:
    """Re-run the strategy on a small grid around the current factor and ATR period (factor ± 0.5 / 1.0, ATR period x 0.75 / 1 / 1.25), same window and costs.
    A result that is only good at the exact setting (a SPIKE) is the usual signature of over-fitting; neighbours that are nearly as good (a PLATEAU) are what you want."""
    f0, a0 = float(values["factor"]), int(values["atrPeriod"])
    factors = sorted({round(max(0.5, f0 + d), 4) for d in (-1.0, -0.5, 0.0, 0.5, 1.0)})
    atrs = sorted({int(round(a0 * k)) for k in (0.75, 1.0, 1.25)})
    sh, cg, dd, nt = (np.full((len(atrs), len(factors)), np.nan) for _ in range(4))
    t0, total, done = time.time(), len(atrs) * len(factors), 0
    for i, a in enumerate(atrs):
        for j, f in enumerate(factors):
            if cancelled():
                break
            try:
                r = run_backtest(bars, {**values, "factor": f, "atrPeriod": a}, BacktestSettings(**{**bt.__dict__, "benchmark": False}))
                sh[i, j], cg[i, j], dd[i, j], nt[i, j] = r.metrics.get("sharpe", np.nan), r.metrics["cagr"], r.metrics["max_drawdown"], r.metrics["trades"]
            except Exception as exc:                  # noqa: BLE001
                log(f"  factor {f:g} / ATR {a}: {exc}")
            done += 1
            progress(done, total)
    ci, cj = atrs.index(a0), factors.index(round(f0, 4))
    res = Robustness(factors, atrs, sh, cg, dd, nt, (ci, cj), seconds=time.time() - t0)
    centre = sh[ci, cj]
    mask = np.ones_like(sh, dtype=bool); mask[ci, cj] = False
    nb = sh[mask]
    nb = nb[np.isfinite(nb)]
    if not np.isfinite(centre) or not len(nb):
        res.verdict, res.level = "Not enough results to judge robustness.", "info"
    else:
        med, worst = float(np.median(nb)), float(nb.min())
        ratio = med / centre if centre > 0 else float("nan")
        txt = f"Sharpe {centre:.2f} at the current setting; neighbours: median {med:.2f}, worst {worst:.2f} ({ratio:.0%} of the centre)."
        if not np.isfinite(ratio) or centre <= 0:
            res.verdict, res.level = txt + " The current setting itself has no positive Sharpe.", "bad"
        elif ratio >= 0.8:
            res.verdict, res.level = txt + " PLATEAU - neighbouring settings hold up.", "ok"
        elif ratio >= 0.6:
            res.verdict, res.level = txt + " PARTLY FRAGILE - neighbours lose a good part of the edge.", "warn"
        else:
            res.verdict, res.level = txt + " SPIKE - the result is only good near this exact setting; expect live results nearer the neighbours.", "bad"
    return res
