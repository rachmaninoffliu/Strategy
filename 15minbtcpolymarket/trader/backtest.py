"""Backtest engine: replays the strategy on local Kalshi history with Kelly sizing, fees and slippage.

Per 15-minute window: every leg that qualified produces a signal (first qualifying decision minute). The trade is the earliest-minute
signal whose Kelly stake is at least one contract (ties: larger edge). At most one trade per window; all signals still feed the
win-probability model (so it learns without having to bet)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .data import SETTLE_DELAY, extract_signals
from .fees import fee_total
from .metrics import compute_metrics, drawdown_series, equity_curve
from .adaptive import AdaptiveStats, AutoConfig, decide, resolve_volume
from .evfilter import DEFAULT_MIN_N, TrailingEVStats
from .probmodel import SignalStats
from .strategy import MIN_HISTORY, StrategyParams, size_bet, throttled_sizing

PROB_MODES = ("Expanding historical (no look-ahead)", "Fixed value", "Full-sample (look-ahead!)")


@dataclass
class BacktestConfig:
    start: date
    end: date                                   # inclusive
    capital: float = 1_000.0
    params: StrategyParams = field(default_factory=StrategyParams)
    fee_model: str = "Kalshi"
    slippage: float = 0.02                      # added to the quoted ask (tape replay showed ~2-3c worse than candle closes)
    kelly_mult: float = 0.25
    max_frac: float = 0.02
    min_edge: float = 0.04                      # net edge (p - all-in cost) the learner must show before betting; 4c cut the dead-regime bleed from -20% to ~0 (cv_gate2.py)
    max_contracts: int = 2000
    prob_mode: str = PROB_MODES[0]
    fixed_p: float = 0.25
    prior_n: float = 20.0                       # strength of the skeptical prior (pseudo-signals at the market-implied price)
    lookback_days: int = 0                      # history window for the estimate; 0 = all history before the trade
    halflife_days: float = 0.0                  # >0: exponentially decaying memory instead of the hard window (see probmodel)
    daily_loss_limit: float = 0.10              # the live engine's risk rules, applied identically here (0 = off): no new trades until the next UTC day once equity is this far below the day's start ...
    max_drawdown_stop: float = 0.25             # ... and stop for good once equity is this far below its peak (0 = off)
    min_history: int = MIN_HISTORY              # the live engine refuses to trade a leg on fewer historical signals than this (expanding-probability mode only)
    dd_throttle: bool = True                    # scale the Kelly multiplier AND the stake cap down as the drawdown below the peak deepens (see strategy.drawdown_throttle)
    core_hours: tuple = ()                      # 'by_hour' mode: hours with an informative prior (smaller shrinkage core_k) ...
    core_k: float = 0.0                         # ... 0 = off
    near_k: float = 0.0                         # hours within near_width of a core hour use this (lighter than hour_k) shrinkage; 0 = off
    near_width: int = 2
    auto: Optional[AutoConfig] = None           # Auto mode (trader.adaptive): regime-aware memory, evidence-aware Kelly, equity ladder, liquidity cap; replaces kelly_mult / max_frac / max_contracts / halflife
    compounding: bool = True
    p_model: str = "overall"                    # 'overall' (per leg) or 'by_hour' (per leg and UTC hour, shrunk to the leg estimate)
    p_basis: str = "edge"                       # 'edge' = price-adjusted historical edge (default) | 'win_rate' = raw historical win rate
    hour_k: float = 40.0                        # shrinkage strength toward the leg estimate in 'by_hour' mode
    ev_rules: tuple = ()                        # EV-regime filter (trader.evfilter): ((days, threshold), ...) - an hour trades only while its trailing net EV beats EVERY rule; () = off
    ev_min_n: int = DEFAULT_MIN_N               # signals an hour needs inside a rule's window before its EV counts
    start_ts: Optional[int] = None              # exact period (UTC epoch seconds, [start_ts, end_ts) by window open) instead of whole days; the live-vs-backtest reconcile compares a session's own span
    end_ts: Optional[int] = None


@dataclass
class BacktestResult:
    cfg: BacktestConfig
    trades: pd.DataFrame
    equity: pd.DataFrame
    drawdown: pd.Series
    metrics: dict
    signals_in_range: int
    skipped: dict


def _ts(d: date, end: bool = False) -> int:
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp()) + (86400 if end else 0)


def full_sample_p(sig_rng: pd.DataFrame, leg: str, hour: int, by_hour: bool, k: float, cost: float, basis: str) -> float:
    """Hindsight estimate over the whole backtest range (no prior) - optimistic, only for reference."""
    s = sig_rng[sig_rng.leg == leg]
    if basis == "edge":
        d = float((s.win - s.cost).mean()) if len(s) else 0.0
        if by_hour:
            h = s[s.hour == hour]
            d = ((h.win - h.cost).sum() + k * d) / (len(h) + k)
        return min(max(cost + d, 0.0), 1.0)
    p0 = float(s.win.mean()) if len(s) else 0.0
    if not by_hour:
        return p0
    h = s[s.hour == hour]
    return (h.win.sum() + k * p0) / (len(h) + k)


TRADE_COLS = ["entry_ts", "settle_ts", "entry_time", "market", "leg", "hour", "side", "quoted", "entry", "contracts", "stake", "fee", "outcome",
              "win", "pnl", "ret_on_stake", "p_used", "f_full", "f_used", "bankroll_before", "equity_after", "regime", "mult_used", "cap_used"]


def bounds(cfg: BacktestConfig) -> tuple[int, int]:
    """[start, end) of the backtest in epoch seconds: the exact ``start_ts`` / ``end_ts`` when given, else the whole days ``start`` .. ``end`` (inclusive)."""
    return (_ts(cfg.start) if cfg.start_ts is None else int(cfg.start_ts)), (_ts(cfg.end, end=True) if cfg.end_ts is None else int(cfg.end_ts))


def run_backtest(rows: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    start_ts, end_ts = bounds(cfg)
    sig = extract_signals(rows, cfg.params)                         # full history: it also burns in the probability model
    in_rng = sig[(sig.open_ts >= start_ts) & (sig.open_ts < end_ts)]
    ev_on = bool(cfg.ev_rules) and cfg.prob_mode == PROB_MODES[0]                                 # the EV filter needs the history (expanding) mode too
    auto_on = cfg.auto is not None and cfg.auto.enabled and cfg.prob_mode == PROB_MODES[0] and not ev_on       # Auto needs the learned (expanding) probability
    if ev_on:
        stats = TrailingEVStats.from_signals(sig, cfg.ev_rules, cfg.ev_min_n, cfg.slippage, cfg.fee_model)
    elif auto_on:
        stats = AdaptiveStats.from_signals(sig, cfg.auto, mode=cfg.p_model, hour_k=cfg.hour_k, prior_n=cfg.prior_n, basis=cfg.p_basis, core_hours=cfg.core_hours, core_k=cfg.core_k, near_k=cfg.near_k, near_width=cfg.near_width)
    else:
        stats = SignalStats.from_signals(sig, mode=cfg.p_model, hour_k=cfg.hour_k, prior_n=cfg.prior_n, lookback_days=cfg.lookback_days, basis=cfg.p_basis, halflife_days=cfg.halflife_days, core_hours=cfg.core_hours, core_k=cfg.core_k, near_k=cfg.near_k, near_width=cfg.near_width)
    by_hour = cfg.p_model == "by_hour"
    bank = float(cfg.capital)
    recs: list[dict] = []
    peak = bank
    day_key, day_start, paused_until, halted = None, bank, 0.0, False
    last_pnl, last_close = 0.0, -1.0                                  # the previous trade: the live engine books a P&L when the result arrives (SETTLE_DELAY after the close), see the day-start rule below
    vol_seen: list = []                                               # liquidity-cap input, exactly as the live engine resolves it (adaptive.resolve_volume)
    vol_key, vol_now = None, 0.0
    skipped = {"no edge (Kelly = 0)": 0, "stake < 1 contract": 0, "paused (daily loss limit)": 0, "halted (drawdown stop)": 0}
    for ticker, g in in_rng.groupby("ticker", sort=False):
        ots = int(g.open_ts.iloc[0])                                  # same rules as LiveEngine._risk_checks, evaluated at the start of each window
        day = ots // 86400
        if day != day_key:
            day_key = day
            day_start = bank - (last_pnl if last_close + SETTLE_DELAY > day * 86400 else 0.0)        # live stamps the day's start at 00:00:00; a trade whose window closed at that instant is not booked yet
        if halted or (cfg.max_drawdown_stop and peak > 0 and bank <= peak * (1 - cfg.max_drawdown_stop)):
            halted = True
            skipped["halted (drawdown stop)"] += 1
            continue
        if cfg.daily_loss_limit and ots >= paused_until and bank <= day_start * (1 - cfg.daily_loss_limit):
            paused_until = (day + 1) * 86400
        if ots < paused_until:
            skipped["paused (daily loss limit)"] += 1
            continue
        best = None
        no_edge = small = 0
        for r in g.itertuples():
            entry_ts = int(r.open_ts + 60 * r.minute)
            base = bank if cfg.compounding else cfg.capital
            px = min(float(r.cost) + cfg.slippage, 0.99)
            regime, mult_u, cap_u = "", 0.0, 0.0
            if auto_on:
                pred = stats.predict(r.leg, int(r.hour), float(r.cost), entry_ts)
                if cfg.min_history and pred.n < cfg.min_history:
                    no_edge += 1
                    continue
                if (r.ticker, int(r.minute)) != vol_key:                      # one reading per decision instant, like the live engine
                    vol_key = (r.ticker, int(r.minute))
                    vol_now = resolve_volume(getattr(r, "vol", float("nan")), vol_seen)[0]
                dec = decide(cfg.auto, stats, r.leg, int(r.hour), float(r.cost), entry_ts, bank, peak, base, vol_now, cfg.fee_model, cfg.slippage, cfg.min_edge, pred)
                p, sz, regime, mult_u, cap_u = dec.p_used, dec.sizing, dec.regime, dec.mult, dec.cap
            else:
                if cfg.prob_mode == PROB_MODES[1]:
                    p = float(cfg.fixed_p)
                elif cfg.prob_mode == PROB_MODES[2]:
                    p = full_sample_p(in_rng, r.leg, int(r.hour), by_hour, cfg.hour_k, float(r.cost), cfg.p_basis)
                else:
                    p, n_hist, _nh = stats.p(r.leg, int(r.hour), float(r.cost), entry_ts)
                    if cfg.min_history and n_hist < cfg.min_history:
                        no_edge += 1
                        continue
                mult_u, cap_u = throttled_sizing(cfg.kelly_mult, cfg.max_frac, bank, peak, cfg.dd_throttle)
                sz = size_bet(base, p, px, cfg.fee_model, mult_u, cap_u, cfg.max_contracts, cfg.min_edge)
            if sz.contracts < 1:
                if sz.reason.startswith("stake"):
                    small += 1
                else:
                    no_edge += 1
                continue
            key = (r.minute, -sz.edge)
            if best is None or key < best[0]:
                best = (key, r, p, px, sz, entry_ts, (regime, mult_u, cap_u))
        if best is None:
            skipped["no edge (Kelly = 0)"] += no_edge
            skipped["stake < 1 contract"] += small
            continue
        _, r, p, px, sz, entry_ts, (regime, mult_u, cap_u) = best
        n = sz.contracts
        fee = fee_total(cfg.fee_model, n, px)
        win = int(r.win)
        pnl = n * (1.0 if win else 0.0) - n * px - fee
        bank += pnl
        peak = max(peak, bank)
        last_pnl, last_close = pnl, float(r.close_ts)
        recs.append(dict(
            entry_ts=entry_ts, settle_ts=int(r.close_ts), entry_time=datetime.fromtimestamp(entry_ts, timezone.utc), market=r.ticker,
            leg=r.leg, hour=int(r.hour), side=r.side, quoted=float(r.cost), entry=px, contracts=n, stake=n * px + fee, fee=fee,
            outcome="WIN" if win else "LOSS", win=win, pnl=pnl, ret_on_stake=pnl / (n * px + fee), p_used=p, f_full=sz.f_full,
            f_used=sz.f_used, bankroll_before=bank - pnl, equity_after=bank, regime=regime, mult_used=mult_u, cap_used=cap_u))
    trades = pd.DataFrame(recs, columns=TRADE_COLS)
    if not trades.empty:
        trades = trades.sort_values("settle_ts").reset_index(drop=True)
    eq = equity_curve(trades, cfg.capital, start_ts)
    dd = drawdown_series(eq)
    n_windows = in_rng.ticker.nunique()
    extra = dict(signals_in_range=len(in_rng), windows_with_signal=n_windows, skipped_no_edge=skipped["no edge (Kelly = 0)"],
                 skipped_small=skipped["stake < 1 contract"], windows_paused=skipped["paused (daily loss limit)"], windows_halted=skipped["halted (drawdown stop)"])
    metrics = compute_metrics(trades, cfg.capital, start_ts, end_ts, extra)
    return BacktestResult(cfg=cfg, trades=trades, equity=eq, drawdown=dd, metrics=metrics, signals_in_range=len(in_rng), skipped=skipped)
