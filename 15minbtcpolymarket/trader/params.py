"""Which parameters does each side actually use?  (Qt-free; the GUI's 'Parameters in use' table is built from this.)

Backtest and Live/paper share ONE strategy, ONE probability model, ONE sizing rule, ONE set of risk limits and ONE evaluation (metrics.evaluate_trades) - they differ only where
the world differs: a backtest has no real exchange (fills are assumed) and no account (capital is typed in); a live session has an order limit, buying power and a clock.
Every row below is a (backtest value, live value) pair, formatted identically, so equal strings = the same parameter and different strings = a deliberate difference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .adaptive import describe_liquidity, describe_regime_scale, format_ladder
from .backtest import PROB_MODES, BacktestConfig
from .live import MIN_HISTORY, LiveConfig
from .strategy import fmt_int_list

NA = "n/a"


@dataclass(frozen=True)
class ParamRow:
    group: str
    name: str
    backtest: str
    live: str
    note: str = ""

    @property
    def kind(self) -> str:
        if self.backtest == self.live:
            return "shared"
        if self.backtest.startswith(NA) and self.live.startswith(NA):
            return "not-used"
        if self.live.startswith(NA):
            return "backtest-only"
        if self.backtest.startswith(NA):
            return "live-only"
        return "differs"


def _band(lo: float, hi: float, on: bool) -> str:
    return f"{lo:.2f} - {hi:.2f}" if on else "off"


def auto_in_effect(c) -> bool:
    """Is Auto mode in effect for this config?  (It needs the learned probability: the backtest's history mode / the live 'history' source.)"""
    a = getattr(c, "auto", None)
    if a is None or not a.enabled:
        return False
    return c.prob_mode == PROB_MODES[0] if hasattr(c, "prob_mode") else c.prob_source == "history"


_auto_on = auto_in_effect


def _memory(c) -> str:
    if _auto_on(c):
        return "AUTO: learners with half-lives " + " / ".join(f"{h:g}" for h in c.auto.half_lives) + " days, weighted by their recent accuracy"
    if c.halflife_days and c.halflife_days > 0:
        return f"decaying, half-life {c.halflife_days:g} days"
    return f"hard window, last {c.lookback_days} days" if c.lookback_days else "all history"


def _slip(c) -> float:
    return c.slippage if hasattr(c, "slippage") else c.paper_slippage


def describe_parameters(bt: Optional[BacktestConfig], lv: LiveConfig, preset: str = "", live_mode: str = "", bt_error: str = "") -> list:
    """Rows of (group, parameter, backtest value, live value, note)."""
    bad = f"({bt_error or 'invalid backtest settings'})"
    rows: list[ParamRow] = []

    def both(group: str, name: str, f, note: str = "") -> None:
        rows.append(ParamRow(group, name, f(bt) if bt is not None else bad, f(lv), note))

    def sep(group: str, name: str, b: str, l: str, note: str = "") -> None:
        rows.append(ParamRow(group, name, b if bt is not None else bad, l, note))

    g = "Strategy (shared)"
    sep(g, "Preset", preset, preset, "chosen in the left panel")
    both(g, "Window-open hours (UTC)", lambda c: fmt_int_list(c.params.hours))
    both(g, "Decision minute(s)", lambda c: fmt_int_list(c.params.minutes))
    both(g, "Underdog price band", lambda c: _band(c.params.cost_min, c.params.cost_max, "DOG" in c.params.legs))
    both(g, "Favourite price band", lambda c: _band(c.params.fav_min, c.params.fav_max, "FAV" in c.params.legs))
    both(g, "Max book spread", lambda c: f"{c.params.max_spread * 100:.0f}c")

    g = "Win probability (shared estimator)"
    both(g, "Basis", lambda c: "price-adjusted edge" if c.p_basis == "edge" else "raw historical win rate")
    both(g, "Model", lambda c: "by UTC hour (adaptive)" if c.p_model == "by_hour" else "overall (per leg)")
    both(g, "Hourly shrinkage k", lambda c: f"{c.hour_k:g}" if c.p_model == "by_hour" else "—")
    both(g, "Core hours (lighter shrinkage)", lambda c: f"{fmt_int_list(c.core_hours)}  (k = {c.core_k:g})" if (c.p_model == "by_hour" and c.core_hours and c.core_k) else "none")
    both(g, "Near-core hours (lighter shrinkage)", lambda c: f"within ±{c.near_width} h of the core: k = {c.near_k:g}" if (c.p_model == "by_hour" and c.core_hours and c.core_k and c.near_k) else "none")
    both(g, "Memory", _memory)
    both(g, "Prior strength (pseudo-signals at zero edge)", lambda c: f"{c.prior_n:g}")
    if bt is None:
        b_src = bad
    elif bt.prob_mode == PROB_MODES[0]:
        b_src = "history, only signals settled before each trade"
    elif bt.prob_mode == PROB_MODES[1]:
        b_src = f"FIXED {bt.fixed_p * 100:.0f}% (not learned)"
    else:
        b_src = "FULL-SAMPLE hindsight (optimistic)"
    l_src = f"MANUAL {lv.manual_p * 100:.0f}% (not learned)" if lv.prob_source == "manual" else "history, only signals settled before each trade"
    rows.append(ParamRow(g, "Where p comes from", b_src, l_src, "'history' = the same learner in both; the live session also adds the signals that settle while it runs"))
    rows.append(ParamRow(g, "Minimum history before betting",
                         bad if bt is None else (f">= {bt.min_history} signals" if (bt.min_history and bt.prob_mode == PROB_MODES[0]) else "off"),
                         f">= {MIN_HISTORY} signals" if lv.prob_source == "history" else "off", "the live engine refuses a leg with fewer; the GUI backtest applies the same rule"))

    g = "Sizing (shared)"
    both(g, "Kelly multiplier", lambda c: "AUTO: equity ladder x regime x throttle" if _auto_on(c) else f"{c.kelly_mult:.2f}x")
    both(g, "Max stake per trade", lambda c: "AUTO: equity ladder x regime x throttle" if _auto_on(c) else f"{c.max_frac * 100:.1f}% of equity")
    both(g, "Min edge to bet", lambda c: f"{c.min_edge * 100:.1f}c")
    both(g, "Max contracts", lambda c: f"AUTO: {describe_liquidity(c.auto)}" if _auto_on(c) else f"{c.max_contracts:,}")
    both(g, "Drawdown throttle", lambda c: (("on (1/2 at -8%, 1/4 at -15%; scales the multiplier AND the cap)" if c.auto.throttle else "off") if _auto_on(c) else
                                            "on (1/2 size at -8%, 1/4 at -15%)" if c.dd_throttle else "off"))
    rows.append(ParamRow(g, "Sizes off", "current equity" if (bt is not None and bt.compounding) else (bad if bt is None else "FIXED initial capital (no compounding)"),
                         "current equity" if lv.mode == "paper" else "current equity, capped by your buying power", "the live engine always compounds; tick 'Compound' in the backtest to match it"))

    if _auto_on(lv) or (bt is not None and _auto_on(bt)):
        g = "Auto mode (shared)"
        both(g, "Auto mode", lambda c: "ON" if _auto_on(c) else "off", "the strategy sets its own memory, Kelly multiplier, stake cap and max size at each trade")
        both(g, "Learner memories (days)", lambda c: " / ".join(f"{h:g}" for h in c.auto.half_lives) if _auto_on(c) else "—", "predictions are averaged with weights from each learner's recent accuracy (floor 5%)")
        both(g, "Evidence haircut z", lambda c: f"{c.auto.z:g} standard errors" if _auto_on(c) else "—", "Kelly uses edge - z x standard error: thin evidence = smaller bet")
        both(g, "Equity ladder", lambda c: format_ladder(c.auto.ladder).replace(", ", "  |  ") if _auto_on(c) else "—", "equity >= threshold : Kelly multiplier / stake cap %")
        both(g, "Regime scaling", lambda c: describe_regime_scale(c.auto) if _auto_on(c) else "—", "HOT = fast and slow memories agree; BUILDING = only the fast one; FADING = the fast one dropped; COLD = no trade")
        both(g, "Liquidity cap", lambda c: describe_liquidity(c.auto) if _auto_on(c) else "—", "the backtest uses Kalshi candle volume (June 2026 on; earlier days have none = only the ceiling)")

    g = "Costs (shared)"
    both(g, "Fee model", lambda c: c.fee_model)
    both(g, "Assumed slippage", lambda c: f"{_slip(c) * 100:.1f}c", "backtest fills, paper fills AND the price live bets are sized with")
    rows.append(ParamRow(g, "Actual fill price", "assumed: ask + assumed slippage", "paper: same as backtest; LIVE: the exchange's real fill", "the one thing a backtest cannot know"))
    rows.append(ParamRow(g, "Order limit price", f"{NA} (no real order)", f"ask + {lv.max_slippage * 100:.1f}c (IOC, never pays more)" if lv.mode == "live" else f"{NA} in paper mode (would be ask + {lv.max_slippage * 100:.1f}c)",
                         "live-only box; it caps what a real order can pay and does not change the position size"))

    g = "Risk limits (shared)"
    both(g, "Daily loss limit", lambda c: f"{c.daily_loss_limit * 100:.0f}% (pause until next UTC day)" if c.daily_loss_limit else "off")
    both(g, "Max drawdown stop", lambda c: f"{c.max_drawdown_stop * 100:.0f}% (halt)" if c.max_drawdown_stop else "off")

    g = "Capital, period, mode (differ by nature)"
    rows.append(ParamRow(g, "Initial capital", bad if bt is None else f"${bt.capital:,.0f}", f"${lv.capital:,.0f}", "each tab has its own capital box; metrics are percentages, so scale does not matter"))
    rows.append(ParamRow(g, "Period", bad if bt is None else f"{bt.start} -> {bt.end}", "from the moment you press Start, in real time", "backtest-only date range"))
    rows.append(ParamRow(g, "Mode", "simulation on downloaded Kalshi history", live_mode or ("LIVE — real orders" if lv.mode == "live" else "Paper — simulated fills, no orders"), ""))
    return rows
