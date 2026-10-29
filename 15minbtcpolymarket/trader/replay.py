"""Replay the LIVE engine over historical candles: the proof that the backtest and live trading take the same decisions.

``replay_live`` drives the very engine that trades real money (``LiveEngine`` with a ``PaperBroker``) through every 15-minute window of a past period on a simulated clock.  Its quotes are the Kalshi candle closes the
backtest trades on, its probability model starts from the history before the period and learns window by window as results arrive (exactly what a running session does).  ``compare_trades`` then diffs the
engine's trades with ``run_backtest``'s: same markets, sides, contracts, prices, probabilities, P&L - or a list of what differs.  Used by ``replay_live.py`` (command line), the Backtest tab's "Verify" button and the tests.

What a replay cannot contain is the real market at the second of the order: the order book's depth (the live depth guard needs a book) and the real fill.  ``reconcile`` measures those on your real trades.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .backtest import PROB_MODES, BacktestConfig, BacktestResult, bounds, run_backtest
from .broker import PaperBroker
from .data import SETTLE_DELAY, WINDOW
from .live import LiveConfig, LiveEngine, build_stats, trades_frame


# ── configuration mapping (the one place that says which backtest field is which live field) ──
def live_config_from(b: BacktestConfig, **live_only) -> LiveConfig:
    """The LiveConfig that makes the live engine take the decisions ``run_backtest(b)`` takes.  ``live_only`` sets what only exists live (mode, max_slippage, grace_s)."""
    if not b.compounding:
        raise ValueError("the live engine always compounds (sizes on current equity): a non-compounding backtest has no live equivalent")
    if b.prob_mode == PROB_MODES[2]:
        raise ValueError("the full-sample probability uses hindsight: it cannot be traded live")
    manual = b.prob_mode == PROB_MODES[1]
    return LiveConfig(capital=b.capital, params=b.params, fee_model=b.fee_model, kelly_mult=b.kelly_mult, max_frac=b.max_frac, min_edge=b.min_edge, max_contracts=b.max_contracts,
                      paper_slippage=b.slippage, prob_source="manual" if manual else "history", manual_p=b.fixed_p, lookback_days=b.lookback_days, halflife_days=b.halflife_days,
                      dd_throttle=b.dd_throttle, core_hours=b.core_hours, core_k=b.core_k, near_k=b.near_k, near_width=b.near_width, p_model=b.p_model, p_basis=b.p_basis, hour_k=b.hour_k,
                      prior_n=b.prior_n, daily_loss_limit=b.daily_loss_limit, max_drawdown_stop=b.max_drawdown_stop, min_history=b.min_history, auto=b.auto, ev_rules=b.ev_rules, ev_min_n=b.ev_min_n, **live_only)


def backtest_config_from(c: LiveConfig, start: date, end: date, capital: Optional[float] = None) -> BacktestConfig:
    """The BacktestConfig that mirrors a live configuration over [start, end] (inclusive)."""
    manual = c.prob_source == "manual"
    return BacktestConfig(start=start, end=end, capital=c.capital if capital is None else capital, params=c.params, fee_model=c.fee_model, slippage=c.paper_slippage, kelly_mult=c.kelly_mult,
                          max_frac=c.max_frac, min_edge=c.min_edge, max_contracts=c.max_contracts, prob_mode=PROB_MODES[1] if manual else PROB_MODES[0], fixed_p=c.manual_p, prior_n=c.prior_n,
                          lookback_days=c.lookback_days, halflife_days=c.halflife_days, daily_loss_limit=c.daily_loss_limit, max_drawdown_stop=c.max_drawdown_stop, min_history=c.min_history,
                          dd_throttle=c.dd_throttle, core_hours=c.core_hours, core_k=c.core_k, near_k=c.near_k, near_width=c.near_width, auto=c.auto, p_model=c.p_model, p_basis=c.p_basis, hour_k=c.hour_k,
                          ev_rules=c.ev_rules, ev_min_n=c.ev_min_n)


# ── the replay's quote source ──
class SimClock:
    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


class HistoricalQuotes:
    """Quote source of a replay: the Kalshi candle CLOSE at the simulated minute (what the backtest trades on), the candle's volume, and the result once the window has closed.
    It has no order book, so the engine's depth guard (a live-only extra) stays out of the way."""

    def __init__(self, rows: pd.DataFrame, clock: SimClock):
        self.clock = clock
        self._open: dict = {}
        self._win: dict = {}
        self._q: dict = {}
        self._res: dict = {}
        vol = rows["vol"].values if "vol" in rows else np.full(len(rows), np.nan)
        for tk, ot, mn, b, a, v, y in zip(rows.ticker.values, rows.open_ts.values, rows.minute.values, rows.bid.values, rows.ask.values, vol, rows.yes.values):
            self._open[tk] = int(ot)
            self._win[int(ot)] = tk
            self._q[(tk, int(mn))] = (float(b), float(a), float(v))
            self._res[tk] = "yes" if int(y) else "no"

    def window_ticker(self, window: int) -> Optional[str]:
        return self._win.get(int(window))

    def quote(self, ticker: str) -> Optional[tuple[float, float]]:
        ot = self._open.get(ticker)
        if ot is None:
            return None
        q = self._q.get((ticker, int((self.clock.t - ot) // 60)))
        return None if q is None else (q[0], q[1])

    def minute_volume(self, ticker: str, window: int, minute: int) -> Optional[float]:
        q = self._q.get((ticker, int(minute)))
        return None if q is None or q[2] != q[2] else q[2]

    def result(self, ticker: str) -> Optional[str]:
        ot = self._open.get(ticker)
        return self._res.get(ticker) if ot is not None and self.clock.t >= ot + WINDOW else None


@dataclass
class Replay:
    trades: pd.DataFrame                         # the live engine's trades in the backtest's TRADE_COLS format
    engine: LiveEngine
    log: list = field(default_factory=list)
    windows: int = 0
    seconds: float = 0.0


def replay_live(rows: pd.DataFrame, cfg: BacktestConfig, keep_log: bool = False, progress: Optional[Callable[[float], None]] = None, ledger=None, session_id: Optional[str] = None,
                broker=None) -> Replay:
    """Run the live engine over the backtest's period on the candles in ``rows`` (the probability model starts from the history before the period).
    ``ledger`` records the replayed session and its trades like a real one (the reconcile tests use it); ``broker`` replaces the paper broker (e.g. one that fills worse than assumed)."""
    t0 = time.time()
    lc = live_config_from(cfg)
    start_ts, end_ts = bounds(cfg)
    clock = SimClock(start_ts)
    quotes = HistoricalQuotes(rows[(rows.open_ts >= start_ts) & (rows.open_ts < end_ts)], clock)
    stats = build_stats(rows[rows.open_ts < start_ts], lc)                       # what a session starting at ``start`` would know
    log: list = []
    eng = LiveEngine(lc, quotes, broker or PaperBroker(lc.fee_model, lc.paper_slippage), stats, emit=lambda kind, d: log.append(d["msg"]) if (keep_log and kind == "log") else None, clock=clock,
                     ledger=ledger, session_id=session_id or (f"replay_{start_ts}" if ledger is not None else None))
    hours = set(lc.params.hours)
    first = ((start_ts + WINDOW - 1) // WINDOW) * WINDOW
    windows = list(range(first, end_ts, WINDOW))
    ticks: list = []
    for w in windows:                                                                       # the instants a continuously running session acts on, in time order
        ticks.append(w + 0.1)                                                               # first look at a window (also stamps a new UTC day before the last window's result arrives)
        if (w // 3600) % 24 in hours:
            ticks.extend(w + 60.0 * m for m in lc.params.minutes)                           # decision instants: the quote is read right at the minute mark
        ticks.append(w + WINDOW + SETTLE_DELAY + 0.1)                                       # the result is looked for once it is posted
    ticks.sort()
    n = len(ticks)
    for i, t in enumerate(ticks):
        clock.t = t
        eng.tick()
        if progress is not None and i % 2000 == 0:
            progress(i / n)
    eng.close("replay finished")
    return Replay(trades_frame(eng.trades), eng, log, len(windows), time.time() - t0)


# ── comparison ──
EXACT = ("side", "leg", "entry_ts", "contracts", "outcome", "regime")
NUMERIC = dict(quoted=1e-9, entry=1e-9, fee=1e-9, pnl=1e-6, p_used=1e-9, f_full=1e-9, f_used=1e-9, bankroll_before=1e-6, equity_after=1e-6, mult_used=1e-9, cap_used=1e-9)


@dataclass
class Comparison:
    n_a: int
    n_b: int
    only_a: list                                 # markets only the first set traded
    only_b: list
    diffs: list                                  # (market, column, a value, b value)
    names: tuple = ("backtest", "live engine")

    @property
    def identical(self) -> bool:
        return not (self.only_a or self.only_b or self.diffs)

    def text(self, limit: int = 12) -> str:
        a, b = self.names
        if self.identical:
            return f"IDENTICAL: {self.n_a} trades in the {a}, {self.n_b} in the {b}; every market, side, contract count, price, probability and P&L matches."
        out = [f"DIFFERENT: {self.n_a} trades in the {a}, {self.n_b} in the {b}; {len(self.only_a)} only in the {a}, {len(self.only_b)} only in the {b}, {len(self.diffs)} field differences."]
        out += [f"  only in the {a}: {m}" for m in self.only_a[:limit]]
        out += [f"  only in the {b}: {m}" for m in self.only_b[:limit]]
        out += [f"  {m}: {c} {x!r} vs {y!r}" for m, c, x, y in self.diffs[:limit]]
        return "\n".join(out)


def compare_trades(a: pd.DataFrame, b: pd.DataFrame, names: tuple = ("backtest", "live engine")) -> Comparison:
    """Trade-by-trade diff of two trade tables in the backtest's format (keyed by market: at most one trade per window)."""
    ia = a.set_index("market") if len(a) else pd.DataFrame(columns=["entry_ts"]).set_index("entry_ts")
    ib = b.set_index("market") if len(b) else pd.DataFrame(columns=["entry_ts"]).set_index("entry_ts")
    only_a = [m for m in ia.index if m not in ib.index]
    only_b = [m for m in ib.index if m not in ia.index]
    diffs = []
    for m in ia.index.intersection(ib.index):
        ra, rb = ia.loc[m], ib.loc[m]
        for c in EXACT:
            if c in ia and c in ib and ra[c] != rb[c] and not (ra[c] != ra[c] and rb[c] != rb[c]):
                diffs.append((m, c, ra[c], rb[c]))
        for c, tol in NUMERIC.items():
            if c in ia and c in ib:
                x, y = float(ra[c]), float(rb[c])
                if (x != x) != (y != y) or (x == x and abs(x - y) > tol):
                    diffs.append((m, c, x, y))
    return Comparison(len(a), len(b), only_a, only_b, diffs, names)


def verify(rows: pd.DataFrame, cfg: BacktestConfig, keep_log: bool = False) -> tuple[Comparison, BacktestResult, Replay]:
    """Backtest and live-engine replay of the same period and settings, and their comparison."""
    bt = run_backtest(rows, cfg)
    rp = replay_live(rows, cfg, keep_log=keep_log)
    return compare_trades(bt.trades, rp.trades), bt, rp
