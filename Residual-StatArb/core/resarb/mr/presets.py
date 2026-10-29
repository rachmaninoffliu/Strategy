"""
Ready-made mean-reversion models (the GUI's model picker). Every one is long-only, unlevered (gross <= 1x equity), decided at the snapshot time and traded in the closing auction.
The research behind them (in-sample 2016-09 .. 2022, out-of-sample 2023 .. 2026-10, ~1,900 variants): research/mr_grid.py, mr_select.py, mr_final.py and MeanReversion/REPORT.md.

The core rule is the same in all IBS presets - the one the stock grid, the ETF grid and every later stage agreed on:
  enter: IBS < 0.2 (the price sits in the bottom fifth of the day's range at the decision time) while the price is above its 200-day mean
  exit:  IBS > 0.7, or a close above the 5-day mean, or 10 sessions
"""
from __future__ import annotations

from .models import MRModel

_IBS = dict(entry_signal="ibs", entry_op="<", entry_value=0.2, trend_n=200, exit_signal="ibs", exit_op=">", exit_value=0.7, exit_sma_n=5, max_hold=10, rank_by="signal")

PRESETS: dict = {
    "Recommended: IBS < 0.2 in up-trends - top-50 stocks + 59 ETFs, 20 slots, 15:45": MRModel(
        name="IBS top50+ETF k20 1545", universe="top50+etf", minute=1545, slots=20, **_IBS,
        note="The research pick: the 50 most liquid stocks of each month plus 59 ETFs (sectors, countries, bonds, credit, gold, oil, ...). 15:45 needs the 'sip+iex' (free) or "
             "the real-time 'sip' feed in live."),
    "Higher return: the same rule with 10 slots, 15:45": MRModel(
        name="IBS top50+ETF k10 1545", universe="top50+etf", minute=1545, slots=10, **_IBS,
        note="Fewer, bigger positions: more of the equity at work (75% on average), a higher CAGR and a deeper drawdown."),
    "ETFs only: IBS < 0.2 in up-trends - 59 ETFs, 5 slots, 15:45": MRModel(
        name="IBS ETF k5 1545", universe="etf", minute=1545, slots=5, **_IBS,
        note="No stock universe to maintain: 59 liquid ETFs across asset classes, 5 positions of ~20% each."),
    "Free feed, exact backtest data: top-50 + ETFs, 20 slots, 15:30": MRModel(
        name="IBS top50+ETF k20 1530", universe="top50+etf", minute=1530, slots=20, **_IBS,
        note="Decided at 15:30 so the free 16-minute-delayed SIP feed gives live exactly the backtest's bars. Deciding 15 minutes earlier costs ~0.3 Sharpe."),
    "Reference: Connors RSI(2) < 10 above the 200-day mean - top-100 stocks, 10 slots, 15:45": MRModel(
        name="RSI2 top100 k10 1545", universe="top100", minute=1545, entry_signal="rsi2", entry_op="<", entry_value=10, trend_n=200,
        exit_signal="", exit_sma_n=5, max_hold=10, slots=10, rank_by="signal",
        note="Connors & Alvarez's classic, decided honestly at 15:45 (published versions decide on the close they trade at)."),
}


def preset_names() -> list:
    return list(PRESETS)


def get(name: str) -> MRModel:
    return MRModel.from_dict(PRESETS[name].to_dict())
