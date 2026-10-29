"""Named parameter sets. 'values' are overrides on the SigCfg / SimCfg defaults; the Settings panel shows every one of them, so nothing is hidden behind a preset.

How these were chosen (LEVPAIRS_REPORT.md): the research scan ranked 1,458 rule sets on 2017-2021 only and showed 2022-2026 next to it. With fills at the next bar's open and quoted-spread costs
EVERY one lost money, so no preset below is a 'winner'. DEFAULT is the structurally motivated setting (a daily-reset 3x ETF is 3x its index since the prior close, so the spread to that line is
the thing to fade; Valid notes that 96% of that spread's variance is a between-session offset, so the rationale is weaker than it sounds), not the best number in the scan."""
from __future__ import annotations

from dataclasses import fields
from typing import Optional

from levpairs_core import SigCfg
from levpairs_sim import SimCfg

PRESETS: dict = {}
DEFAULT_PRESET = "Premium-to-reset reversion, 5-min (research default)"

PRESETS[DEFAULT_PRESET] = dict(
    note="The Avellaneda-Lee construction with one factor: the spread of each 3x ETF to 3x its 1x ETF since the previous close, an OU fit over 5 sessions, fade at |s| > 2 (take profit at 0.5), "
         "half-life under 60 min, only when the expected capture is at least 1.5x the round-trip cost. Fills at the OPEN of the bar after the signal, quoted spreads + 0.3 bp slippage per leg. "
         "With these fills the gross edge is under 1 bp per trade; costs are 5-11 bp per round trip, so the net Sharpe is about -5. Research / paper only.",
    values=dict(tf=5, spread="reset", beta_mode="fixed", win_days=5.0, s_mode="ou", s_open=2.0, s_close_long=0.5, s_close_short=0.5, s_stop=4.0, max_halflife_min=60.0, min_edge_ratio=1.5, fill="next_open",
                slip_bp=0.3, vol_on=True, vol_target=0.10, g_max=3.0))

PRESETS["30-minute, patient (least bad in the scan)"] = dict(
    note="Fewer, larger trades on 30-minute bars with a rolling 2-session OU fit and a 3-sigma entry: the least negative family in the scan (in-sample Sharpe about -1), but the gross edge per trade is "
         "about 2.5 bp on only 88 in-sample trades against 7 bp of cost. Shown so the trade-off between horizon and cost is visible.",
    values=dict(tf=30, spread="reset", beta_mode="rolling", win_days=2.0, s_mode="ou", s_open=3.0, s_close_long=0.5, s_close_short=0.5, max_halflife_min=60.0, min_edge_ratio=1.5, fill="next_open",
                slip_bp=0.3, skip_open_min=30, no_entry_min=60, flat_min=30, vol_on=True, vol_target=0.10))

PRESETS["1-minute, edge/cost filtered"] = dict(
    note="One-minute bars, entries only when the expected capture is 1.5x the round-trip cost. The most trades and the highest gross edge (Sharpe about 4 at zero cost, only ~0.7 bp per trade); the 1-minute signal is also the one "
         "most exposed to stale last-trade prices, and it cannot be run on the IEX feed (too many empty minutes in the thinner ETFs).",
    values=dict(tf=1, spread="reset", beta_mode="fixed", win_days=5.0, s_mode="ou", s_open=2.0, s_close_long=0.5, s_close_short=0.5, max_halflife_min=60.0, min_edge_ratio=1.5, fill="next_open", slip_bp=0.3,
                vol_on=True, vol_target=0.10))

PRESETS["Paper replica - A&L thresholds on 5-min bars"] = dict(
    note="The paper's rules as published (cumulative bar residual X, open at |s| 1.25, close longs at -0.50 and shorts at +0.75, no filters, no vol target) on 5-minute bars of the three pairs, "
         "with realistic next-open fills. What the 2008 recipe is worth here.",
    values=dict(tf=5, spread="chain", beta_mode="rolling", win_days=5.0, s_mode="ou", s_open=1.25, s_close_long=0.50, s_close_short=0.75, s_stop=99.0, max_hold_min=10_000, max_halflife_min=0.0,
                min_edge_ratio=0.0, skip_open_min=0, no_entry_min=0, cooldown_min=0, fill="next_open", slip_bp=0.3, vol_on=False, g_init=1.0, weight_mode="equal"))

PRESETS["OPTIMISTIC: fill at the signal bar's close (bias demo)"] = dict(
    note="The same default rules but the order fills at the last trade of the bar that produced the signal - the usual shortcut of a bar-based backtest. It earns the snap-back of two stale, "
         "asynchronous last-trade prints (bounce, lag between the legs, selection of extremes) and shows a handsome Sharpe that no order can capture. Keep it to see how large the bias is; never trade on it.",
    values=dict(tf=5, spread="reset", beta_mode="fixed", win_days=5.0, s_mode="ou", s_open=2.0, s_close_long=0.5, s_close_short=0.5, max_halflife_min=60.0, min_edge_ratio=1.5, fill="close", slip_bp=0.3,
                vol_on=True, vol_target=0.10))


def cfgs(name: str, **override) -> tuple:
    """(SigCfg, SimCfg) for a preset - the same mapping the Settings panel uses, without Qt (used by the report script and the tests)."""
    v = {**PRESETS[name]["values"], **override}
    sg = {x.name: getattr(SigCfg(), x.name) for x in fields(SigCfg)}
    sm = {x.name: getattr(SimCfg(), x.name) for x in fields(SimCfg)}
    for k, val in v.items():
        if k in sg:
            sg[k] = val
        if k in sm:
            sm[k] = val
    if not v.get("vol_on", True):
        sm["g_min"] = sm["g_init"]
    return SigCfg(**sg), SimCfg(**sm)
