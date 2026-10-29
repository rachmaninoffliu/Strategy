"""Matplotlib canvases in the app's dark palette."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from supertrend_ui.theme import ACCENT, BG, BORDER, GREEN, MUTED, PANEL, RED, TEXT, YELLOW


def _style(ax) -> None:
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=BORDER, alpha=0.35, linewidth=0.5)
    ax.title.set_color(TEXT)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


class Canvas(FigureCanvasQTAgg):
    def __init__(self, rows: int = 1, height_ratios: Optional[list] = None, w: float = 8, h: float = 4):
        self.fig = Figure(figsize=(w, h), facecolor=BG)
        super().__init__(self.fig)
        gs = self.fig.add_gridspec(rows, 1, height_ratios=height_ratios or [1] * rows, hspace=0.12)
        self.axes = [self.fig.add_subplot(gs[i, 0]) for i in range(rows)]
        for a in self.axes:
            _style(a)
        self.fig.subplots_adjust(left=0.07, right=0.985, top=0.95, bottom=0.08)

    def message(self, text: str) -> None:
        for a in self.axes:
            a.clear(); _style(a)
        self.axes[0].text(0.5, 0.5, text, transform=self.axes[0].transAxes, ha="center", va="center", color=MUTED, fontsize=10, wrap=True)
        self.draw_idle()


class EquityChart(Canvas):
    """Equity vs SPY (log scale) with the out-of-sample region shaded, drawdown below."""

    def __init__(self):
        super().__init__(2, [3, 1])
        self.message("Run a backtest")

    def show_curves(self, eq: pd.Series, bench: Optional[pd.Series], split: Optional[pd.Timestamp], label: str = "Strategy") -> None:
        top, bot = self.axes
        for a in self.axes:
            a.clear(); _style(a)
        top.plot(eq.index, eq.values, color=ACCENT, linewidth=1.5, label=label)
        if bench is not None and len(bench):
            b = bench.reindex(eq.index).fillna(0.0)
            top.plot(eq.index, eq.iloc[0] * (1 + b).cumprod().values, color=MUTED, linewidth=1.0, label="SPY buy & hold (open-to-open)")
        top.set_yscale("log")
        if split is not None and eq.index[0] < split < eq.index[-1]:
            top.axvspan(split, eq.index[-1], color=YELLOW, alpha=0.07)
            top.text(split, 0.97, "  out-of-sample", transform=top.get_xaxis_transform(), color=YELLOW, fontsize=8, va="top")
        top.legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        top.tick_params(labelbottom=False)
        dd = eq / eq.cummax() - 1
        bot.fill_between(dd.index, dd.values * 100, 0, color=RED, alpha=0.5)
        bot.set_ylabel("drawdown %")
        self.draw_idle()


class LineChart(Canvas):
    def __init__(self, ylabel: str = "", h: float = 2.5):
        super().__init__(1, None, 8, h)
        self.ylabel = ylabel

    def show_series(self, series: dict, hlines: Optional[list] = None, fill: bool = False) -> None:
        ax = self.axes[0]
        ax.clear(); _style(ax)
        cols = [ACCENT, GREEN, YELLOW, RED, MUTED]
        for i, (name, s) in enumerate(series.items()):
            ax.plot(s.index, s.values, color=cols[i % len(cols)], linewidth=1.2, label=name)
        for y in hlines or []:
            ax.axhline(y, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_ylabel(self.ylabel)
        if len(series) > 1:
            ax.legend(loc="upper left", facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        self.draw_idle()


class BarChart(Canvas):
    def __init__(self, h: float = 3.2):
        super().__init__(1, None, 6, h)

    def show_bars(self, labels: list, values: list, colors: Optional[list] = None, title: str = "", horizontal: bool = True) -> None:
        ax = self.axes[0]
        ax.clear(); _style(ax)
        colors = colors or [GREEN if v >= 0 else RED for v in values]
        if horizontal:
            ax.barh(range(len(values)), values, color=colors)
            ax.set_yticks(range(len(values))); ax.set_yticklabels(labels, fontsize=8, color=TEXT)
            ax.invert_yaxis()
            ax.axvline(0, color=MUTED, linewidth=0.8)
        else:
            ax.bar(range(len(values)), values, color=colors)
            ax.set_xticks(range(len(values))); ax.set_xticklabels(labels, fontsize=8, color=TEXT, rotation=45, ha="right")
            ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=9)
        self.fig.subplots_adjust(left=0.22 if horizontal else 0.08, bottom=0.08 if horizontal else 0.25)
        self.draw_idle()


class Histogram(Canvas):
    def __init__(self, h: float = 2.8):
        super().__init__(1, None, 6, h)

    def show_hist(self, x: np.ndarray, bins: int = 60, title: str = "") -> None:
        ax = self.axes[0]
        ax.clear(); _style(ax)
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        ax.hist(x, bins=bins, color=ACCENT, alpha=0.8)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, fontsize=9)
        self.draw_idle()
