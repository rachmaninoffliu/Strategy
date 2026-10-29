"""Strategy vs SPY buy & hold: equity (log scale, out-of-sample shaded) and both drawdowns."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ...ui.charts import Canvas, _style
from ...ui.theme import ACCENT, BORDER, MUTED, PANEL, RED, TEXT, YELLOW


class CompareChart(Canvas):
    def __init__(self):
        super().__init__(2, [3, 1])
        self.message("Run a backtest")

    def show(self, equity: pd.Series, bench_ret: pd.Series, split: Optional[pd.Timestamp], name: str) -> None:
        top, bot = self.axes
        for a in self.axes:
            a.clear(); _style(a)
        cap = float(equity.iloc[0]) if len(equity) else 1.0
        spy = cap * (1 + bench_ret.fillna(0)).cumprod()
        top.plot(equity.index, equity.values, color=ACCENT, linewidth=1.5, label=name)
        if bench_ret.notna().any():
            top.plot(spy.index, spy.values, color=MUTED, linewidth=1.1, label="SPY buy & hold")
        top.set_yscale("log")
        if split is not None and len(equity) and equity.index[0] < split < equity.index[-1]:
            top.axvspan(split, equity.index[-1], color=YELLOW, alpha=0.07)
            top.text(split, 0.97, "  out-of-sample", transform=top.get_xaxis_transform(), color=YELLOW, fontsize=8, va="top")
        top.legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        top.tick_params(labelbottom=False)
        dd = equity / equity.cummax() - 1
        bot.fill_between(dd.index, dd.values * 100, 0, color=RED, alpha=0.5, label=name)
        if bench_ret.notna().any():
            ds = spy / spy.cummax() - 1
            bot.plot(ds.index, ds.values * 100, color=MUTED, linewidth=0.9, label="SPY")
        bot.set_ylabel("drawdown %")
        self.draw_idle()
