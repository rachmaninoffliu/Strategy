"""PyQt6 GUI: shared Strategy/Kelly panel + Live Trading tab (Kalshi) + Backtest tab (equity curve, metrics, trades)."""
from __future__ import annotations

import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QDate, QEvent, QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox as _QComboBox, QDateEdit as _QDateEdit, QDialog, QDoubleSpinBox as _QDoubleSpinBox, QFileDialog,
                             QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox as _QSpinBox, QSplitter,
                             QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)


class _NoStrayWheel:
    """Mixin: a spin box / drop-down / date field only reacts to the mouse wheel once you have clicked into it. Otherwise scrolling the settings panel silently changes whatever
    field is under the cursor (Kelly multiplier, stake cap, ...) - a real risk for a trading app. The ignored wheel event scrolls the panel instead."""

    def _quiet(self) -> None:
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, e) -> None:                      # noqa: N802 (Qt override)
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class QDoubleSpinBox(_NoStrayWheel, _QDoubleSpinBox):
    def __init__(self, *a):
        super().__init__(*a); self._quiet()


class QSpinBox(_NoStrayWheel, _QSpinBox):
    def __init__(self, *a):
        super().__init__(*a); self._quiet()


class QComboBox(_NoStrayWheel, _QComboBox):
    def __init__(self, *a):
        super().__init__(*a); self._quiet()


class QDateEdit(_NoStrayWheel, _QDateEdit):
    def __init__(self, *a):
        super().__init__(*a); self._quiet()

from . import kalshi, preflight, radar
from .adaptive import DEFAULT_LADDER, AutoConfig, decide, format_ladder, parse_ladder, snapshot
from .backtest import PROB_MODES, BacktestConfig, BacktestResult, run_backtest
from .data import DATA, backfill_history, backfill_plan, coverage, data_range, extract_signals, load_rows, needs_refresh, update_data
from .fees import FEE_MODELS, fee_per_contract
from .kalshi import ENVS, ApiError, KalshiBroker, KalshiClient, KalshiQuotes, base_url, load_credentials
from .ledger import Ledger, ledger_dir, pooled_frame
from .live import MIN_HISTORY, LiveConfig, LiveEngine, PaperBroker, auto_active, build_stats, trades_frame
from .presets import PRESETS
from .metrics import evaluate_trades
from .params import auto_in_effect, describe_parameters
from .probmodel import SignalStats
from .reconcile import reconcile as reconcile_sessions
from .replay import live_config_from, verify as verify_replay
from .strategy import StrategyParams, kelly_full, parse_int_list, size_bet, throttled_sizing

# ── theme (same palette as the arbitrage app) ─────────────────────────────────
BG, PANEL, PANEL2, BORDER = "#0e1116", "#151a21", "#1b222b", "#2a3441"
TEXT, MUTED, ACCENT = "#d7dee8", "#8b98a9", "#3d8bfd"
GREEN, YELLOW, RED = "#2ea06b", "#d9a520", "#e05561"

STYLE = f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 12px; color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; background: {BG}; }}
QTabBar::tab {{ background: {PANEL}; padding: 9px 22px; border: 1px solid {BORDER}; border-bottom: none; margin-right: 2px;
               color: {MUTED}; font-weight: 600; }}
QTabBar::tab:selected {{ background: {BG}; color: {TEXT}; border-top: 2px solid {ACCENT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 14px; padding: 10px 8px 8px 8px; background: {PANEL}; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QLabel {{ background: transparent; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#warn {{ color: {YELLOW}; }}
QLabel#kpiValue {{ font-size: 18px; font-weight: 700; }}
QLabel#kpiName {{ color: {MUTED}; font-size: 11px; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 6px;
    selection-background-color: {ACCENT}; min-height: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; selection-background-color: {ACCENT}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 14px; font-weight: 600; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #566172; border-color: #212a35; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton#primary:disabled {{ background: #27446d; color: #8fa6c8; }}
QPushButton#go {{ background: #1f6f4a; border-color: {GREEN}; color: white; }}
QPushButton#go:disabled {{ background: #1a3a2c; color: #7da892; border-color: #24503c; }}
QPushButton#danger {{ background: #7a2430; border-color: {RED}; color: white; }}
QPushButton#danger:disabled {{ background: #3a1d22; color: #a07a80; border-color: #55303a; }}
QCheckBox {{ spacing: 6px; background: transparent; }}
QTableWidget {{ background: {PANEL}; alternate-background-color: #171d25; gridline-color: {BORDER}; border: 1px solid {BORDER};
    selection-background-color: #23406b; selection-color: white; }}
QHeaderView::section {{ background: {PANEL2}; color: {MUTED}; padding: 6px; border: none; border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER}; font-weight: 600; }}
QPlainTextEdit {{ background: #0b0e13; border: 1px solid {BORDER}; font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)


# ── small widgets ─────────────────────────────────────────────────────────────
class Kpi(QFrame):
    def __init__(self, name: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        self.n = QLabel(name)
        self.n.setObjectName("kpiName")
        self.v = QLabel("—")
        self.v.setObjectName("kpiValue")
        lay.addWidget(self.n)
        lay.addWidget(self.v)

    def set(self, text: str, color: Optional[str] = None) -> None:
        self.v.setText(text)
        self.v.setStyleSheet(f"color: {color};" if color else "")
        self.setToolTip(f"{self.n.text()}: {text}")


def money(x: float, sign: bool = False) -> str:
    return f"{x:+,.2f}" if sign else f"{x:,.2f}"


def pct(x: float, d: int = 2, sign: bool = False) -> str:
    return "—" if x is None or not np.isfinite(x) else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")


def num(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{d}f}"


def spin(lo: float, hi: float, val: float, dec: int = 2, step: float = 0.1, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    s.setGroupSeparatorShown(True)
    return s


def make_table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.setTextElideMode(Qt.TextElideMode.ElideNone)                         # never cut a cell's text with "..."
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)    # wide tables scroll sideways instead of clipping
    t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return t


class PageScroll(QScrollArea):
    """A scrolling page. When the window is smaller than the page needs it scrolls (both directions) instead of cutting text off.
    Unlike QScrollArea(widgetResizable=True) it sizes the page by its MINIMUM height for the current width: Qt would otherwise take the page's *preferred*
    height-for-width as a hard minimum as soon as one word-wrapped label is on it (here: 1073 px instead of the 647 px really needed), forcing a scroll bar even on a big window."""

    def __init__(self, inner: QWidget, min_w: int = 0, min_h: int = 0):
        super().__init__()
        self._min = (min_w, min_h)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setWidget(inner)
        inner.installEventFilter(self)
        self.viewport().installEventFilter(self)
        self._fit()

    def eventFilter(self, obj, ev) -> bool:           # noqa: N802 (Qt override)
        if obj is self.widget() and ev.type() == QEvent.Type.LayoutRequest:
            QTimer.singleShot(0, self._fit)
        elif obj is self.viewport() and ev.type() == QEvent.Type.Resize:
            self._fit()
        return False

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._fit()

    def _fit(self) -> None:
        inner = self.widget()
        if inner is None:
            return
        vp = self.viewport().size()
        w = max(vp.width(), inner.minimumSizeHint().width(), self._min[0])
        lay = inner.layout()
        h_need = lay.minimumHeightForWidth(w) if (lay is not None and lay.hasHeightForWidth()) else inner.minimumSizeHint().height()
        h = max(vp.height(), h_need, self._min[1])
        if (inner.width(), inner.height()) != (w, h):
            inner.resize(w, h)


def scrollable(inner: QWidget, min_w: int = 0, min_h: int = 0) -> QScrollArea:
    """Wrap a page so that it scrolls instead of cutting text off (see PageScroll)."""
    return PageScroll(inner, min_w, min_h)


def set_row(t: QTableWidget, r: int, vals: list, colors: Optional[dict] = None) -> None:
    for c, v in enumerate(vals):
        it = QTableWidgetItem(str(v))
        it.setToolTip(str(v))
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter if c > 0 else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if colors and c in colors and colors[c]:
            it.setForeground(QColor(colors[c]))
        t.setItem(r, c, it)


class EquityChart(FigureCanvasQTAgg):
    """Equity curve (top) + drawdown (bottom), dark themed and interactive:
       mouse wheel = zoom in / out around the cursor (Shift+wheel = zoom the vertical axis only) · drag = pan · double-click = reset the view ·
       hover a dot = trade details · click a dot = pin its details (click empty space to unpin)."""
    HIT_PX = 10

    def __init__(self):
        self.fig = Figure(figsize=(8, 5), facecolor=BG)
        super().__init__(self.fig)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.07, left=0.075, right=0.985, top=0.96, bottom=0.09)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._reset_state()
        self._style()
        self.message("Run a backtest to see the equity curve")
        for name, fn in (("scroll_event", self._on_scroll), ("button_press_event", self._on_press), ("button_release_event", self._on_release),
                         ("motion_notify_event", self._on_motion), ("axes_leave_event", self._on_leave)):
            self.mpl_connect(name, fn)

    # ── state ──
    def _reset_state(self) -> None:
        self._tx = np.array([]); self._ty = np.array([]); self._trades = pd.DataFrame()
        self._ex = np.array([]); self._ey = np.array([]); self._capital = 0.0
        self._home = None
        self._ann = None; self._hot = None; self._hover = None; self._pinned = None
        self._press = None; self._moved = False

    def _style(self) -> None:
        for a in (self.ax, self.ax2):
            a.set_facecolor(PANEL)
            a.tick_params(colors=MUTED, labelsize=9)
            for sp in a.spines.values():
                sp.set_color(BORDER)
            a.grid(True, color=BORDER, lw=0.5, alpha=0.7)
        plt_labels = self.ax.get_xticklabels()
        for l in plt_labels:
            l.set_visible(False)

    def message(self, text: str) -> None:
        self._reset_state()
        self.ax.clear(); self.ax2.clear(); self._style()
        self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes, ha="center", va="center", color=MUTED, fontsize=11)
        for a in (self.ax, self.ax2):
            a.set_xticks([]); a.set_yticks([])
        self.draw_idle()

    def plot(self, eq: pd.DataFrame, dd: pd.Series, trades: pd.DataFrame, capital: float) -> None:
        self._reset_state()
        self.ax.clear(); self.ax2.clear(); self._style()
        x = pd.to_datetime(eq.ts.values, unit="s", utc=True).tz_convert(None)
        self.ax.step(x, eq.equity.values, where="post", color=ACCENT, lw=1.5, label="Equity")
        self.ax.axhline(capital, color=MUTED, ls="--", lw=0.8)
        self._ex, self._ey, self._capital = mdates.date2num(x.to_pydatetime()), np.asarray(eq.equity.values, float), float(capital)
        if len(trades):
            tx = pd.to_datetime(trades.settle_ts.values, unit="s", utc=True).tz_convert(None)
            win = trades.pnl.values > 0
            self.ax.scatter(tx[win], trades.equity_after.values[win], s=10, color=GREEN, zorder=3, label="Win")
            self.ax.scatter(tx[~win], trades.equity_after.values[~win], s=10, color=RED, zorder=3, label="Loss")
            self.ax.legend(loc="upper left", fontsize=8, facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT)
            self._tx, self._ty = mdates.date2num(tx.to_pydatetime()), np.asarray(trades.equity_after.values, float)
            self._trades = trades.reset_index(drop=True)
        self.ax.set_ylabel("Equity ($)", color=MUTED, fontsize=9)
        self.ax2.fill_between(x, dd.values * 100, 0, step="post", color=RED, alpha=0.55)
        self.ax2.set_ylabel("Drawdown %", color=MUTED, fontsize=9)
        loc = mdates.AutoDateLocator(minticks=3, maxticks=8)
        self.ax2.xaxis.set_major_locator(loc)
        self.ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        for l in self.ax.get_xticklabels():
            l.set_visible(False)
        self.ax.margins(x=0.01)
        self._ann = self.ax.annotate("", xy=(0, 0), xytext=(14, 14), textcoords="offset points", fontsize=8.5, color=TEXT, zorder=20, annotation_clip=False,
                                     bbox=dict(boxstyle="round,pad=0.5", fc=PANEL2, ec=ACCENT, lw=1.0), arrowprops=dict(arrowstyle="->", color=ACCENT))
        self._ann.set_visible(False)
        (self._hot,) = self.ax.plot([], [], "o", ms=10, mfc="none", mec="white", mew=1.6, zorder=15)
        self.draw()                                                          # a real draw first, so the "home" view is the one the user actually sees
        self._home = (self.ax.get_xlim(), self.ax.get_ylim(), self.ax2.get_ylim())

    # ── view control ──
    def reset_view(self) -> None:
        if self._home is None:
            return
        self.ax.set_xlim(self._home[0]); self.ax.set_ylim(self._home[1]); self.ax2.set_ylim(self._home[2])
        self.draw_idle()

    def zoom_by(self, factor: float, x_center: Optional[float] = None, y_only: bool = False) -> None:
        """factor < 1 zooms in. The x axis is zoomed around ``x_center`` (default: the middle of the view) and the equity axis is refitted to what is visible."""
        if self._home is None:
            return
        if y_only:
            lo, hi = self.ax.get_ylim()
            mid = (lo + hi) / 2
            self.ax.set_ylim(mid - (mid - lo) * factor, mid + (hi - mid) * factor)
            self.draw_idle()
            return
        lo, hi = self.ax.get_xlim()
        c = (lo + hi) / 2 if x_center is None else x_center
        nlo, nhi = c - (c - lo) * factor, c + (hi - c) * factor
        h0, h1 = self._home[0]
        span = h1 - h0
        if nhi - nlo > span * 1.5:                                           # no point zooming out far beyond the data
            nlo, nhi = h0 - span * 0.25, h1 + span * 0.25
        if nhi - nlo < 2e-4:                                                 # ~17 s: enough
            return
        self.ax.set_xlim(nlo, nhi)
        self._fit_y()
        self.draw_idle()

    def _fit_y(self) -> None:
        if not len(self._ex):
            return
        lo, hi = self.ax.get_xlim()
        i0 = max(int(np.searchsorted(self._ex, lo, side="right")) - 1, 0)
        i1 = max(int(np.searchsorted(self._ex, hi, side="right")), i0 + 1)
        ys = self._ey[i0:i1]
        if len(self._ty):
            m = (self._tx >= lo) & (self._tx <= hi)
            if m.any():
                ys = np.r_[ys, self._ty[m]]
        if not len(ys):
            return
        y0, y1 = float(ys.min()), float(ys.max())
        pad = max((y1 - y0) * 0.08, abs(y1) * 0.002, 1e-9)
        self.ax.set_ylim(y0 - pad, y1 + pad)

    # ── trade info ──
    def _nearest(self, event) -> Optional[int]:
        if not len(self._tx) or event.x is None or event.y is None:
            return None
        pts = self.ax.transData.transform(np.column_stack([self._tx, self._ty]))
        d = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] <= self.HIT_PX else None

    def trade_text(self, i: int) -> str:
        r = self._trades.iloc[i]
        et = pd.Timestamp(r.entry_time)
        st = pd.to_datetime(int(r.settle_ts), unit="s", utc=True)
        ret = r.pnl / r.stake if r.stake else 0.0
        return (f"Trade #{i + 1} — {r.outcome}\n"
                f"Entry   {et:%Y-%m-%d %H:%M} UTC   settled {st:%H:%M}\n"
                f"{r.market}\n"
                f"{r.leg}: buy {r.side}   quoted {r.quoted:.3f} → paid {r.entry:.3f}\n"
                f"{int(r.contracts):,} contracts   stake ${r.stake:,.2f}   fee ${r.fee:,.2f}\n"
                f"P&L {'-' if r.pnl < 0 else '+'}${abs(r.pnl):,.2f}  ({ret * 100:+.1f}% on stake)\n"
                f"p used {r.p_used * 100:.1f}%   Kelly {r.f_full * 100:.1f}% → used {r.f_used * 100:.2f}%\n"
                f"{self._sizing_line(r)}"
                f"{self._session_line(r)}"
                f"Equity after  ${r.equity_after:,.2f}")

    @staticmethod
    def _session_line(r) -> str:
        s = r.get("session", "")
        return f"Session {s}{' (settled after a restart)' if r.get('recovered', False) else ''}\n" if isinstance(s, str) and s else ""

    @staticmethod
    def _sizing_line(r) -> str:
        mu, cu, reg = r.get("mult_used", np.nan), r.get("cap_used", np.nan), r.get("regime", "")
        if not (isinstance(mu, (int, float)) and np.isfinite(mu)):
            return ""
        return f"{'Auto ' + reg + ': ' if isinstance(reg, str) and reg else 'Sizing: '}Kelly x{mu:.3f}, stake cap {cu * 100:.2f}%\n"

    def _show_trade(self, i: Optional[int]) -> None:
        if self._ann is None:
            return
        if i is None:
            if self._ann.get_visible():
                self._ann.set_visible(False)
                self._hot.set_data([], [])
                self.draw_idle()
            self._hover = None
            return
        x, y = self._tx[i], self._ty[i]
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        right, upper = x > (x0 + x1) / 2, y > (y0 + y1) / 2
        self._ann.xy = (x, y)
        self._ann.set_text(self.trade_text(i).replace("$", r"\$"))       # '$' pairs would otherwise be read as math text
        self._ann.xyann = (-16 if right else 16, -16 if upper else 16)
        self._ann.set_horizontalalignment("right" if right else "left")
        self._ann.set_verticalalignment("top" if upper else "bottom")
        self._ann.set_visible(True)
        self._hot.set_data([x], [y])
        self._hover = i
        self.draw_idle()

    # ── mouse ──
    def _on_scroll(self, event) -> None:
        if self._home is None or event.inaxes not in (self.ax, self.ax2):
            return
        f = 0.8 if event.button == "up" else 1.25
        if event.key == "shift":
            self.zoom_by(f, y_only=True)
        else:
            self.zoom_by(f, x_center=event.xdata)
        if self._pinned is not None:
            self._show_trade(self._pinned)

    def _on_press(self, event) -> None:
        if self._home is None or event.inaxes not in (self.ax, self.ax2) or event.button != 1:
            return
        if getattr(event, "dblclick", False):
            self._pinned = None
            self._show_trade(None)
            self.reset_view()
            return
        inv = self.ax.transData.inverted()
        self._press = dict(x=event.x, y=event.y, inv=inv, start=inv.transform((event.x, event.y)), xlim=self.ax.get_xlim(), ylim=self.ax.get_ylim())
        self._moved = False

    def _on_motion(self, event) -> None:
        if self._home is None:
            return
        p = self._press
        if p is not None and event.x is not None:
            if abs(event.x - p["x"]) + abs(event.y - p["y"]) > 4:
                self._moved = True
            if self._moved:
                cur = p["inv"].transform((event.x, event.y))
                dx, dy = p["start"][0] - cur[0], p["start"][1] - cur[1]
                self.ax.set_xlim(p["xlim"][0] + dx, p["xlim"][1] + dx)
                self.ax.set_ylim(p["ylim"][0] + dy, p["ylim"][1] + dy)
                self.draw_idle()
                return
        if self._pinned is not None:
            return
        i = self._nearest(event) if event.inaxes is self.ax else None
        if i != self._hover:
            self._show_trade(i)

    def _on_release(self, event) -> None:
        p, moved = self._press, self._moved
        self._press, self._moved = None, False
        if p is None or moved or self._home is None:
            return
        i = self._nearest(event) if event.inaxes is self.ax else None        # a click without dragging: pin / unpin
        self._pinned = i
        self._show_trade(i)

    def _on_leave(self, event) -> None:
        if self._pinned is None:
            self._show_trade(None)

    def save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save equity curve", str(DATA / "equity_curve.png"), "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=140, facecolor=BG)


class ChartPanel(QWidget):
    """The equity chart with a slim toolbar (the mouse controls are always listed, so nothing has to be guessed)."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        hint = QLabel("Wheel = zoom · drag = pan · double-click = reset · hover / click a dot = trade info · Shift+wheel = zoom vertically")
        hint.setObjectName("muted")                       # deliberately NOT word-wrapped: a wrapping label makes the whole page "height depends on width" and inflates its minimum height
        self.chart = EquityChart()
        self.btn_in, self.btn_out, self.btn_reset, self.btn_save = QPushButton("Zoom in"), QPushButton("Zoom out"), QPushButton("Reset view"), QPushButton("Save image…")
        self.btn_in.clicked.connect(lambda: self.chart.zoom_by(0.7))
        self.btn_out.clicked.connect(lambda: self.chart.zoom_by(1.4))
        self.btn_reset.clicked.connect(self.chart.reset_view)
        self.btn_save.clicked.connect(self.chart.save_image)
        row.addWidget(hint, 1)
        for b in (self.btn_in, self.btn_out, self.btn_reset, self.btn_save):
            row.addWidget(b)
        v.addLayout(row)
        v.addWidget(self.chart, 1)


# ── background workers ────────────────────────────────────────────────────────
class FnWorker(QThread):
    """Runs fn(log) in a thread; emits log lines and the result/exception."""
    log = pyqtSignal(str)
    done = pyqtSignal(object, object)           # result, error-text

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self.fn(self.log.emit), None)
        except Exception as e:                    # noqa: BLE001
            self.done.emit(None, f"{type(e).__name__}: {e}")


def read_cash(cli) -> dict:
    """The connected account's cash: {shard, shard_name, bal (cash on the trading shard), total (all shards)}."""
    q = KalshiQuotes(base=cli.base)
    shard = q.exchange_index()
    bal = cli.balance(shard)
    total = cli.balance() if shard is not None else bal
    names = q.shard_names() if shard is not None else {}
    return dict(shard=shard, shard_name=names.get(shard, ""), bal=bal, total=total)


class LiveWorker(QThread):
    ev = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()
        self.engine: Optional[LiveEngine] = None
        self._stop = False

    def run(self) -> None:
        assert self.engine is not None
        self.engine.push_status()
        last_status = 0.0
        while not self._stop:
            try:
                self.engine.tick()
                if time.time() - last_status > 1.0:
                    self.engine.push_status()
                    last_status = time.time()
            except Exception as e:                # noqa: BLE001  - never let a thread die silently with an order possibly open
                self.engine.halt(f"engine error: {type(e).__name__}: {e}")
            self.msleep(int(max(0.005, min(0.25, self.engine.sleep_hint(time.time()))) * 1000))      # coarse while idle, 5-50 ms right before a decision so the quote is read at the minute mark

    def stop(self) -> None:
        self._stop = True


# ── shared settings panel ─────────────────────────────────────────────────────
def sizing_text(cfg) -> str:
    """One-line description of the sizing rule a BacktestConfig / LiveConfig really uses (fixed Kelly + cap, or the Auto ladder)."""
    if auto_in_effect(cfg):
        a = cfg.auto
        return f"AUTO sizing (equity ladder {format_ladder(a.ladder)}; evidence haircut z={a.z:g}; liquidity {a.participation * 100:g}% of last-minute volume)"
    return f"Kelly {cfg.kelly_mult:.2f}x, cap {cfg.max_frac * 100:.1f}%"


def fmt_auto(s: dict, equity: float) -> str:
    """Status line of the Auto layer (LiveEngine.auto_snapshot / adaptive.snapshot)."""
    hrs = ", ".join(f"{h:02d} {r}" for h, r in s["live_hours"]) or "none - every hour is COLD"
    return (f"AUTO · this hour: {s['regime']} | Kelly x{s['mult']:.3f} (ladder step {s['tier'] + 1}) | stake cap {s['cap'] * 100:.2f}% of ${equity:,.0f} | max contracts {s['max_contracts']:,} | "
            f"edge fast {s['d_fast'] * 100:+.1f}c / slow {s['d_slow'] * 100:+.1f}c ± {s['se'] * 100:.1f}c | learner weights {'/'.join(f'{w * 100:.0f}%' for w in s['weights'])} "
            f"for memories {'/'.join(f'{h:g}' for h in s['half_lives'])} d | hours with an edge now: {hrs}")




def app_settings() -> QSettings:
    """The app's saved settings (Windows registry by default). Set TRADER_SETTINGS_FILE to use a throw-away .ini instead - the tests do, so they can never touch your real settings."""
    f = os.environ.get("TRADER_SETTINGS_FILE")
    return QSettings(f, QSettings.Format.IniFormat) if f else QSettings("AlphaSyndicate", "BTC15mTrader")


PRESET_KEYS = ("auto", "hours", "minutes", "dog", "cmin", "cmax", "fav", "fmin", "fmax", "pmodel", "pbasis", "look", "half", "hourk", "corehours", "corek", "neark", "nearw")      # the fields a preset sets


class SettingsPanel(QWidget):
    """Strategy rule + execution + Kelly + risk settings shared by the Live and Backtest tabs (the SAME values drive both)."""
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._applying = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        banner = QLabel("SHARED SETTINGS — every field in this panel is used by BOTH the Backtest and the Live / paper session. "
                        "Fields that exist for only one of them are in that tab's own box (\"Backtest-only\" / \"Live-only\"). The tab \"Parameters in use\" lists everything side by side.")
        banner.setWordWrap(True)
        banner.setStyleSheet(f"background: rgba(61,139,253,0.14); border: 1px solid {ACCENT}; border-radius: 6px; padding: 8px; color: {TEXT};")
        lay.addWidget(banner)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet(f"background: rgba(217,165,32,0.16); border: 1px solid {YELLOW}; border-radius: 6px; padding: 8px; color: {TEXT};")
        self.notice.hide()
        lay.addWidget(self.notice)

        g = QGroupBox("Strategy preset")
        f = QFormLayout(g)
        self.preset = QComboBox(); self.preset.addItems(list(PRESETS))
        self.preset.setToolTip("Presets fill in the fields below. R2 (default) tested best: 39% win rate, Sharpe 3.7. R3 trades more but with lower Sharpe. "
                               "The experimental all-day / favourite presets exist so you can see for yourself that they test poorly after fees and slippage.")
        self.preset.activated.connect(lambda *_: self.apply_preset(self.preset.currentText()))
        self.preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.preset.setMinimumContentsLength(18)
        f.addRow(self.preset)
        lay.addWidget(g)

        g = QGroupBox("Auto mode — the strategy adapts itself")
        f = QFormLayout(g)
        A = AutoConfig()
        self.auto_on = QCheckBox("Auto: detect the regime, set memory, Kelly, stake and size by itself")
        self.auto_on.setToolTip("At EVERY trade the engine (1) mixes several learners of the same edge with different memories (fast right after a regime change, slow while it is stable), "
                                "(2) labels the regime HOT / BUILDING / FADING / COLD and scales the size by it (COLD = no trade), (3) sizes Kelly on the edge minus z standard errors, "
                                "(4) reads the Kelly multiplier and stake cap off the equity ladder (a bigger account bets a smaller fraction) and the drawdown throttle, and (5) caps the order at a share of "
                                "the last minute's traded volume. Backtest and Live run exactly the same code.")
        self.auto_hl = QLineEdit(",".join(f"{h:g}" for h in A.half_lives))
        self.auto_hl.setToolTip("Memories (half-lives in days) of the learners that run side by side, fastest first, e.g. 7,14,30. Their predictions are averaged with weights that follow how well each one "
                                "predicted the signals that settled recently - nothing is chosen by past profit.")
        self.auto_z = spin(0, 3, A.z, 2, 0.05)
        self.auto_z.setToolTip("Evidence haircut: Kelly uses (estimated edge - z x its standard error). 0 = take the estimate at face value; 0.25 = default; larger = smaller bets while evidence is thin "
                               "(z = 1 almost never trades).")
        self.auto_ladder = QLineEdit(format_ladder(A.ladder))
        self.auto_ladder.setToolTip("Equity ladder: 'equity from : Kelly multiplier / stake cap %', comma separated, e.g.  0:0.5/3, 25000:0.35/2.5, 100000:0.25/2, 500000:0.15/1. "
                                    "The step whose equity threshold the account has reached sets the Kelly multiplier and the per-trade stake cap; both are then scaled by the regime and by the drawdown throttle.")
        self.auto_part = spin(0, 100, A.participation * 100, 0, 5, " % of last minute")
        self.auto_part.setToolTip("Liquidity: an order may be at most this share of the contracts traded in the last completed minute (Kalshi candle volume, ~170k at 22-23 UTC). 0 = ignore liquidity. "
                                  "The backtest has volume for June 2026 onwards only; before that only the hard ceiling applies.")
        self.auto_maxc = QSpinBox(); self.auto_maxc.setRange(1, 10_000_000); self.auto_maxc.setValue(A.max_contracts); self.auto_maxc.setGroupSeparatorShown(True)
        self.auto_maxc.setToolTip("Absolute ceiling on the contracts of one order, whatever the liquidity says.")
        self.auto_note = QLabel(""); self.auto_note.setObjectName("muted"); self.auto_note.setWordWrap(True)
        f.addRow(self.auto_on)
        f.addRow("Learner memories (days)", self.auto_hl)
        f.addRow("Evidence haircut z", self.auto_z)
        self.auto_ladder_view = QLabel(""); self.auto_ladder_view.setObjectName("muted"); self.auto_ladder_view.setWordWrap(True)      # the ladder box is narrow: show every step in full below it
        f.addRow("Equity ladder", self.auto_ladder)
        f.addRow("", self.auto_ladder_view)
        f.addRow("Liquidity cap", self.auto_part)
        f.addRow("Hard max contracts", self.auto_maxc)
        f.addRow(self.auto_note)
        lay.addWidget(g)

        g = QGroupBox("Rules")
        f = QFormLayout(g)
        self.hours = QLineEdit("22,23")
        self.hours.setToolTip("UTC hours at which the 15-min window OPENS, e.g. 22,23 or 20-23 or 0-23")
        self.minutes = QLineEdit("12,13")
        self.minutes.setToolTip("Decision minute(s) after the window opens; each leg's first qualifying minute is its signal")
        self.leg_dog = QCheckBox("Underdog leg"); self.leg_dog.setChecked(True)
        self.leg_dog.setToolTip("Buy the less likely side: low win rate, large payoff")
        self.cmin = spin(0.01, 0.95, 0.15, 2, 0.01)
        self.cmax = spin(0.02, 0.95, 0.35, 2, 0.01)
        self.leg_fav = QCheckBox("Favourite leg")
        self.leg_fav.setToolTip("Buy the more likely side: high win rate, small payoff (tests negative after fees)")
        self.fmin = spin(0.50, 0.99, 0.85, 2, 0.01)
        self.fmax = spin(0.51, 0.999, 0.97, 3, 0.01)
        self.spread = spin(0.01, 0.50, 0.06, 2, 0.01)
        f.addRow("Window-open hours (UTC)", self.hours)
        f.addRow("Decision minute(s)", self.minutes)
        f.addRow(self.leg_dog)
        f.addRow("   price from", self.cmin)
        f.addRow("   price to", self.cmax)
        f.addRow(self.leg_fav)
        f.addRow("   price from", self.fmin)
        f.addRow("   price to", self.fmax)
        f.addRow("Max book spread", self.spread)
        lay.addWidget(g)

        g = QGroupBox("Execution and fees")
        f = QFormLayout(g)
        self.fee = QComboBox(); self.fee.addItems(FEE_MODELS)
        self.slip = spin(0, 20, 2.0, 1, 0.5, " ¢")
        self.slip.setToolTip("Used by the backtest fills, the paper fills AND to size live bets (price = quoted ask + this). Tape replay showed real fills ~2-3¢ worse than candle closes. "
                             "A live order's own limit price is set separately, in the Live-only box.")
        f.addRow("Fee model", self.fee)
        f.addRow("Assumed slippage", self.slip)
        lay.addWidget(g)

        g = QGroupBox("Kelly sizing and win probability")
        f = QFormLayout(g)
        self.mult = spin(0.01, 1.0, 0.25, 2, 0.05)
        self.mult.setToolTip("Fraction of full Kelly actually bet. 1.0 = full Kelly (very aggressive).")
        self.cap = spin(0.1, 100, 2.0, 1, 0.5, " %")
        self.cap.setToolTip("Hard cap on one trade's stake as % of bankroll")
        self.edge = spin(0, 20, 4.0, 1, 0.1, " ¢")
        self.edge.setToolTip("Skip the trade unless (win probability - all-in cost) is at least this. 4c is a conviction hurdle: on 31 months of dead-regime data it cut the "
                             "learner's probing losses from -20% to ~0 and still caught the live regime about a week after it started (0.5c = old default).")
        self.throttle = QCheckBox("Drawdown throttle (half size at -8%, quarter at -15%)")
        self.throttle.setChecked(True)
        self.throttle.setToolTip("Scales the Kelly multiplier down as the equity falls below its peak. In simulated regime endings it cut the 60-day drawdown from -26% to -19% and lifted the "
                                 "10th-percentile outcome from -29% to -19%, at almost no cost while the edge persists.")
        self.maxc = QSpinBox(); self.maxc.setRange(1, 1_000_000); self.maxc.setValue(2000); self.maxc.setGroupSeparatorShown(True)
        self.pbasis = QComboBox(); self.pbasis.addItems(["Price-adjusted edge (recommended)", "Raw historical win rate"])
        self.pbasis.setToolTip("Raw win rate treats a 6¢ contract as if it won as often as a 30¢ one, so it over-bets cheap contracts (it lost 99% when "
                               "tested all-day). Price-adjusted uses p = this trade's price + the strategy's historical edge (win - price), starting at zero edge.")
        self.pmodel = QComboBox(); self.pmodel.addItems(["Overall (per leg)", "By UTC hour (adaptive)"])
        self.pmodel.setToolTip("By UTC hour learns each hour's edge separately (shrunk toward the leg's overall edge) and only bets hours where history supports it")
        self.corehours = QLineEdit("")
        self.corehours.setPlaceholderText("e.g. 22,23  (empty = none)")
        self.corehours.setToolTip("'By UTC hour' model only: hours with an informative prior (the hours you chose to trade). They use the smaller 'core shrinkage' below, so the engine acts on them "
                                  "quickly; every other hour keeps the skeptical shrinkage and only trades if its own evidence is strong - this is what lets the strategy follow an edge that moves to another hour.")
        self.corek = spin(0, 500, 0, 0, 5)
        self.corek.setToolTip("Shrinkage strength for the core hours (0 = off, core hours treated like any other hour). Smaller = reacts faster. 10 tested best with other-hour shrinkage 100.")
        self.neark = spin(0, 500, 0, 0, 5)
        self.neark.setToolTip("Hours within the width below of a core hour (e.g. 20, 21, 0, 1 for core 22-23) use this lighter shrinkage than the far scout hours, so an edge that drifts to a neighbouring hour is "
                              "followed sooner. 0 = off. 50 (default) lets a hot neighbouring hour in after ~2 weeks instead of ~4 in simulation, at a cost of ~7% of the Jun-Oct return and ~4 points of drawdown "
                              "(40 is faster and costs more, 60 is slower and cheaper). Needs core hours.")
        self.nearw = QSpinBox(); self.nearw.setRange(1, 6); self.nearw.setValue(2); self.nearw.setSuffix(" hours either side")
        self.hourk = spin(1, 500, 40, 0, 5)
        self.hourk.setToolTip("Shrinkage strength for the hourly estimate: higher = trust the overall estimate more")
        self.half = spin(0, 365, 10, 0, 1, " days  (0 = use the window)")
        self.half.setToolTip("Exponentially decaying memory: a signal's weight halves every N days (min 3). Reacts faster than a hard window at BOTH ends of a regime - it started betting a week "
                             "earlier in the real history and, in a simulated regime end, stops betting weeks sooner. 0 = use the hard lookback window below instead.")
        self.look = QSpinBox(); self.look.setRange(0, 3650); self.look.setValue(45); self.look.setSuffix(" days  (0 = all)")
        self.look.setToolTip("History window used to estimate the strategy's edge. Shorter = notices a regime change sooner (45 days tested best: Sharpe 4.6 vs 3.9 for all history, "
                              "and it started betting about a week earlier); too short = noisy.")
        f.addRow("Kelly multiplier", self.mult)
        f.addRow("Max stake / trade", self.cap)
        f.addRow("Min edge", self.edge)
        f.addRow("Max contracts", self.maxc)
        f.addRow("", self.throttle)
        f.addRow("Probability basis", self.pbasis)
        f.addRow("Probability model", self.pmodel)
        f.addRow("Hourly shrinkage k", self.hourk)
        f.addRow("Core hours", self.corehours)
        f.addRow("Core shrinkage k", self.corek)
        f.addRow("Near-core shrinkage k", self.neark)
        f.addRow("Near-core width", self.nearw)
        f.addRow("Probability lookback", self.look)
        f.addRow("Memory half-life", self.half)
        lay.addWidget(g)

        g = QGroupBox("Risk limits (same rules in Backtest and Live)")
        f = QFormLayout(g)
        self.dloss = spin(1, 100, 10, 0, 1, " %")
        self.dloss.setToolTip("If equity falls this far below the day's starting equity, no new trades until the next UTC day. Applied by the live engine AND by the backtest.")
        self.ddstop = spin(1, 100, 25, 0, 1, " %")
        self.ddstop.setToolTip("If equity falls this far below its peak, trading halts for good (live: until you restart; backtest: the rest of the range). Applied by both.")
        f.addRow("Daily loss limit", self.dloss)
        f.addRow("Max drawdown stop", self.ddstop)
        self.btn_reset = QPushButton("Reset sizing, costs and risk to the validated defaults")
        self.btn_reset.setToolTip("Kelly 0.25x, max stake 2%, min edge 4c, max contracts 2,000, drawdown throttle on, slippage 2c, daily loss 10%, drawdown stop 25%, and the Auto ladder / haircut / liquidity cap "
                                  "back to their defaults. Does not touch the strategy rules or whether Auto is on.")
        self.btn_reset.clicked.connect(self.reset_sizing)
        f.addRow(self.btn_reset)
        lay.addWidget(g)

        for cb in (self.fee, self.pbasis, self.pmodel):
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); cb.setMinimumContentsLength(12)
        self.data_lbl = QLabel("")
        self.data_lbl.setObjectName("muted")
        self.data_lbl.setWordWrap(True)
        self.btn_update = QPushButton("Update data")
        self.btn_update.setToolTip("Download new Kalshi markets/candles (public API) so the backtest and probability estimate include recent days")
        self.btn_backfill = QPushButton("Download older history")
        self.btn_backfill.setToolTip("Backtests can only use days that have been downloaded. Kalshi's BTC 15-minute series started on 2025-12-15, so that is the furthest back "
                                     "possible. Downloads the hours in the Rules box back to that date (a few thousand markets for 2-4 hours; ~3 hours for all 24).")
        lay.addWidget(self.data_lbl)
        lay.addWidget(self.btn_update)
        lay.addWidget(self.btn_backfill)
        lay.addStretch(1)
        for key, w in self._widgets().items():
            sig = getattr(w, "valueChanged", None) or getattr(w, "textChanged", None) or getattr(w, "currentIndexChanged", None) or getattr(w, "toggled", None)
            if sig:
                if key in PRESET_KEYS:
                    sig.connect(lambda *_: self._on_edit())          # a field a preset defines: the preset label becomes "Custom"
                else:
                    sig.connect(lambda *_: self.changed.emit())      # sizing / costs / risk limits are shared by every preset: the preset label stays
        self.apply_preset(next(iter(PRESETS)))
        self.changed.connect(self.refresh_notice)
        self.changed.connect(self._sync_auto)
        self.refresh_notice()
        self._sync_auto()

    def sizing_notice(self) -> str:
        """Text describing any sizing / risk setting above the validated limits (empty if none)."""
        bits = []
        if self.auto_on.isChecked():
            try:
                a = self.auto_config()
            except ValueError as e:
                return f"⚠ Auto settings problem: {e}"
            _t, m0, c0 = a.ladder[0]
            if m0 > 0.5 + 1e-9 or c0 > 0.03 + 1e-9:
                bits.append(f"Auto ladder starts at {m0:g}x Kelly / {c0 * 100:g}% per trade (tested start: 0.5x / 3%)")
            if a.participation <= 0:
                bits.append("the Auto liquidity cap is off")
        else:
            if self.mult.value() > 0.5 + 1e-9:
                bits.append(f"Kelly multiplier {self.mult.value():.2f}x (validated up to 0.5x)")
            if self.cap.value() > 2.0 + 1e-9:
                bits.append(f"max stake {self.cap.value():.1f}% per trade (validated up to 2%)")
            if self.maxc.value() > 5000:
                bits.append(f"max contracts {self.maxc.value():,} (default 2,000)")
        if self.dloss.value() > 10 + 1e-9:
            bits.append(f"daily loss limit {self.dloss.value():.0f}% (default 10%)")
        if self.ddstop.value() > 25 + 1e-9:
            bits.append(f"drawdown stop {self.ddstop.value():.0f}% (default 25%)")
        if not self.throttle.isChecked():
            bits.append("drawdown throttle is off")
        if not bits:
            return ""
        return ("⚠ These sizing / risk settings are above the validated limits: " + "; ".join(bits) + ". If you did not choose this on purpose, click \"Reset sizing, costs and risk to the "
                "validated defaults\" below (scrolling the mouse wheel over a field used to change it by accident).")

    def refresh_notice(self) -> None:
        t = self.sizing_notice()
        self.notice.setText(t)
        self.notice.setVisible(bool(t))

    def reset_sizing(self) -> None:
        for w, v in ((self.mult, 0.25), (self.cap, 2.0), (self.edge, 4.0), (self.maxc, 2000), (self.slip, 2.0), (self.dloss, 10), (self.ddstop, 25)):
            w.setValue(v)
        self.throttle.setChecked(True)
        A = AutoConfig()
        self.auto_hl.setText(",".join(f"{h:g}" for h in A.half_lives)); self.auto_z.setValue(A.z); self.auto_ladder.setText(format_ladder(A.ladder))
        self.auto_part.setValue(A.participation * 100); self.auto_maxc.setValue(A.max_contracts)
        self.refresh_notice()

    def _sync_auto(self) -> None:
        """With Auto on, the fixed Kelly / stake / contracts / memory fields are not used - grey them out so nobody tunes a dead control."""
        on = self.auto_on.isChecked()
        for w in (self.mult, self.cap, self.maxc, self.look, self.half):
            w.setEnabled(not on)
        for w in (self.auto_hl, self.auto_z, self.auto_ladder, self.auto_part, self.auto_maxc):
            w.setEnabled(on)
        try:
            lad = parse_ladder(self.auto_ladder.text())
            self.auto_ladder_view.setText("   ·   ".join(f"equity ≥ ${t:,.0f}: {m:g}x Kelly, {c * 100:g}% cap" for t, m, c in lad))
            self.auto_ladder_view.setStyleSheet("")
        except ValueError as e:
            self.auto_ladder_view.setText(str(e))
            self.auto_ladder_view.setStyleSheet(f"color: {RED};")
        self.auto_note.setText(
            "ON — the Kelly multiplier, Max stake, Max contracts, probability lookback and memory half-life further down are set by Auto at every trade (greyed out). Everything else is still used: "
            "rules, min edge, probability basis / hour model, costs, drawdown throttle and the risk limits. Needs the learned win probability (Backtest 'history' mode, Live 'Historical win rate')."
            if on else "Off — the fixed Kelly multiplier, stake cap, max contracts and memory in the boxes below are used.")

    def auto_config(self) -> Optional[AutoConfig]:
        """The AutoConfig the Backtest and Live tabs pass on (None = Auto off).  Raises ValueError on a bad entry."""
        if not self.auto_on.isChecked():
            return None
        try:
            hl = tuple(sorted({float(x) for x in self.auto_hl.text().replace(";", ",").replace(" ", ",").split(",") if x.strip()}))
        except ValueError:
            raise ValueError("Auto learner memories must be numbers of days, e.g. 7,14,30") from None
        if not 1 <= len(hl) <= 5 or min(hl) < 3:
            raise ValueError("Auto needs 1-5 learner memories of at least 3 days each, e.g. 7,14,30")
        return AutoConfig(half_lives=hl, z=self.auto_z.value(), ladder=parse_ladder(self.auto_ladder.text()), participation=self.auto_part.value() / 100,
                          max_contracts=self.auto_maxc.value(), throttle=self.throttle.isChecked())

    def effective_cap(self) -> float:
        """Per-trade stake cap (fraction of equity) the pre-flight compares with: the first ladder step in Auto mode, else the Max stake field."""
        try:
            a = self.auto_config()
        except ValueError:
            a = None
        return a.ladder[0][2] if a else self.cap.value() / 100

    def cap_auto_ladder(self, cap: float) -> None:
        """The pre-flight's optional cap in Auto mode: no ladder step may stake more than ``cap`` of equity."""
        lad = tuple((t, m, min(c, cap)) for t, m, c in parse_ladder(self.auto_ladder.text()))
        self.auto_ladder.setText(format_ladder(lad))

    def _widgets(self) -> dict:
        return dict(hours=self.hours, minutes=self.minutes, dog=self.leg_dog, cmin=self.cmin, cmax=self.cmax, fav=self.leg_fav, fmin=self.fmin,
                    fmax=self.fmax, spread=self.spread, fee=self.fee, slip=self.slip, mult=self.mult, cap=self.cap, edge=self.edge, maxc=self.maxc,
                    pbasis=self.pbasis, pmodel=self.pmodel, hourk=self.hourk, look=self.look, half=self.half, corehours=self.corehours, corek=self.corek, neark=self.neark, nearw=self.nearw, throttle=self.throttle, dloss=self.dloss, ddstop=self.ddstop,
                    auto=self.auto_on, auto_hl=self.auto_hl, auto_z=self.auto_z, auto_ladder=self.auto_ladder, auto_part=self.auto_part, auto_maxc=self.auto_maxc)

    def _on_edit(self) -> None:
        if not self._applying and self.preset.currentText() != "Custom":
            self.preset.setCurrentText("Custom")
        self.changed.emit()

    def apply_preset(self, name: str) -> None:
        vals = PRESETS.get(name)
        self._applying = True
        try:
            self.preset.setCurrentText(name)
            if vals:
                self.hours.setText(vals["hours"]); self.minutes.setText(vals["minutes"])
                self.leg_dog.setChecked(vals["dog"]); self.cmin.setValue(vals["cmin"]); self.cmax.setValue(vals["cmax"])
                self.leg_fav.setChecked(vals["fav"]); self.fmin.setValue(vals["fmin"]); self.fmax.setValue(vals["fmax"])
                self.pmodel.setCurrentIndex(vals["pmodel"]); self.pbasis.setCurrentIndex(vals["pbasis"]); self.look.setValue(vals.get("look", 45)); self.half.setValue(vals.get("half", 0))
                self.hourk.setValue(vals.get("hourk", 40)); self.corehours.setText(vals.get("core", "")); self.corek.setValue(vals.get("corek", 0))
                self.neark.setValue(vals.get("near", 0)); self.nearw.setValue(vals.get("nearw", 2))
                self.auto_on.setChecked(bool(vals.get("auto", False)))
        finally:
            self._applying = False
        self.changed.emit()

    def params(self) -> StrategyParams:
        legs = tuple(n for n, on in (("DOG", self.leg_dog.isChecked()), ("FAV", self.leg_fav.isChecked())) if on)
        if not legs:
            raise ValueError("Enable at least one leg (underdog or favourite)")
        if self.leg_dog.isChecked() and self.cmin.value() >= self.cmax.value():
            raise ValueError("Underdog 'price from' must be below 'price to'")
        if self.leg_fav.isChecked() and self.fmin.value() >= self.fmax.value():
            raise ValueError("Favourite 'price from' must be below 'price to'")
        return StrategyParams(hours=parse_int_list(self.hours.text(), 0, 23), minutes=parse_int_list(self.minutes.text(), 1, 14),
                              cost_min=self.cmin.value(), cost_max=self.cmax.value(), max_spread=self.spread.value(), legs=legs,
                              fav_min=self.fmin.value(), fav_max=self.fmax.value())

    def core_hours(self) -> tuple:
        t = self.corehours.text().strip()
        return parse_int_list(t, 0, 23) if t and self.corek.value() > 0 else ()

    def near_k(self) -> float:
        """Near-core shrinkage (0 unless core hours are in use)."""
        return self.neark.value() if self.core_hours() else 0.0

    def active_hours(self) -> tuple:
        """The hours the preset is built around: its core hours if it has any (all-hours scout presets), else its trading hours."""
        c = self.core_hours()
        return c if (c and self.p_model() == "by_hour") else tuple(self.params().hours)

    def halflife(self) -> float:
        h = self.half.value()
        return 0.0 if h <= 0 else max(3.0, h)

    def p_model(self) -> str:
        return "by_hour" if self.pmodel.currentIndex() == 1 else "overall"

    def p_basis(self) -> str:
        return "win_rate" if self.pbasis.currentIndex() == 1 else "edge"

    def save(self, qs: QSettings) -> None:
        qs.setValue("settings/preset", self.preset.currentText())
        qs.setValue("settings/gate_v", 2)
        for k, w in self._widgets().items():
            v = (w.text() if isinstance(w, QLineEdit) else w.currentText() if isinstance(w, QComboBox) else w.isChecked() if isinstance(w, QCheckBox) else w.value())
            qs.setValue(f"settings/{k}", v)

    def load(self, qs: QSettings) -> None:
        if qs.value("settings/preset") is None:            # settings saved by an older version: use the new default preset
            return
        self._applying = True
        old_mem = qs.value("settings/half") is None             # saved before the decaying memory existed: non-default presets keep their hard window
        migrate = qs.value("settings/gate_v") is None          # settings saved before the 4c edge hurdle: keep the new default for 'edge'
        try:
            for k, w in self._widgets().items():
                v = qs.value(f"settings/{k}")
                if v is None or (migrate and k == "edge"):
                    continue
                try:
                    if isinstance(w, QLineEdit):
                        w.setText(str(v))
                    elif isinstance(w, QComboBox):
                        i = w.findText(str(v))
                        if i >= 0:
                            w.setCurrentIndex(i)
                    elif isinstance(w, QCheckBox):
                        w.setChecked(str(v).lower() in ("true", "1"))
                    elif isinstance(w, QSpinBox):
                        w.setValue(int(float(v)))
                    else:
                        w.setValue(float(v))
                except (TypeError, ValueError):
                    pass
            i = self.preset.findText(str(qs.value("settings/preset")))
            if i >= 0:
                self.preset.setCurrentIndex(i)
            self.refresh_notice()
            self._sync_auto()
            if old_mem and "R2" not in str(qs.value("settings/preset")):
                self.half.setValue(0)
            if qs.value("settings/corek") is None:                  # saved before core hours existed: the old default preset was the hard-wired 22-23 one
                self.corek.setValue(0); self.corehours.setText("")
                if str(qs.value("settings/preset")).startswith("R2 ·") or "(default)" in str(qs.value("settings/preset")):
                    self.preset.setCurrentText("R2 · 22-23 UTC only (hours hard-wired)")
        finally:
            self._applying = False

    def set_enabled_all(self, on: bool) -> None:
        for w in list(self._widgets().values()) + [self.preset]:
            w.setEnabled(on)
        if on:
            self._sync_auto()


# ── backtest tab ──────────────────────────────────────────────────────────────
METRIC_SECTIONS = [
    ("Returns", [("Initial capital", "initial_capital", "$"), ("Final equity", "final_equity", "$"), ("Total P&L", "total_pnl", "$s"),
                 ("Total return", "total_return", "%s"), ("CAGR (annualised; extrapolated if < 1 year)", "cagr", "%"), ("Period (days)", "period_days", "1"),
                 ("Fees paid", "fees_paid", "$")]),
    ("Risk", [("Max drawdown", "max_drawdown", "%"), ("Max drawdown ($)", "max_drawdown_usd", "$s"), ("Max drawdown duration (days)", "max_dd_duration_days", "1"),
              ("Average drawdown", "avg_drawdown", "%"), ("Ulcer index", "ulcer_index", "%"), ("Annualised volatility", "daily_vol_ann", "%"),
              ("Sharpe ratio (daily, √365)", "sharpe", "2"), ("Sortino ratio", "sortino", "2"), ("Calmar ratio", "calmar", "2"),
              ("Best day", "best_day", "%s"), ("Worst day", "worst_day", "%s"), ("% days up", "pct_days_up", "%")]),
    ("Trades", [("Trades", "trades", "0"), ("Wins", "wins", "0"), ("Losses", "losses", "0"), ("Win rate", "win_rate", "%"), ("Average win ($)", "avg_win", "$s"),
                ("Average loss ($)", "avg_loss", "$s"), ("Payoff ratio (avg win / avg loss)", "payoff_ratio", "2"), ("Profit factor", "profit_factor", "2"),
                ("Expectancy per trade ($)", "expectancy", "$s"), ("Avg return per trade (on stake)", "avg_return_per_trade", "%s"),
                ("95% CI of return per trade — low", "ret_ci_lo", "%s"), ("95% CI of return per trade — high", "ret_ci_hi", "%s"),
                ("t-stat / SQN", "sqn", "2"), ("Best trade ($)", "best_trade", "$s"), ("Worst trade ($)", "worst_trade", "$s")]),
    ("Streaks", [("Max consecutive wins", "max_consec_wins", "0"), ("Max consecutive losses", "max_consec_losses", "0"),
                 ("Average win streak", "avg_win_streak", "2"), ("Average loss streak", "avg_loss_streak", "2"), ("Current streak (+win / −loss)", "current_streak", "0")]),
    ("Sizing & activity", [("Average stake (% of bankroll)", "avg_stake_pct", "%"), ("Max stake (% of bankroll)", "max_stake_pct", "%"),
                           ("Average full-Kelly fraction", "avg_kelly_full", "%"), ("Average Kelly fraction used", "avg_kelly_used", "%"),
                           ("Trades per day", "trades_per_day", "2"), ("Average hold (min)", "avg_hold_min", "1"), ("Exposure (time in market)", "exposure", "%"),
                           ("Signals in range", "signals_in_range", "0"), ("Skipped: Kelly = 0 (no edge)", "skipped_no_edge", "0"),
                           ("Skipped: stake < 1 contract", "skipped_small", "0")]),
]


def fmt_metric(v, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "$":
        return f"${money(v)}"
    if kind == "$s":
        return f"{'-' if v < 0 else '+'}${money(abs(v))}"
    if kind == "%":
        return pct(v)
    if kind == "%s":
        return pct(v, 2, True)
    if kind == "0":
        return f"{int(v):,}"
    if kind == "1":
        return num(v, 1)
    return num(v, 2)


class PerformanceView(QWidget):
    """The ONE performance display used by BOTH the Backtest and the Live tab: the same KPI cards, the same Metrics table (all computed by metrics.evaluate_trades),
    the same equity curve and the same trades / monthly / by-hour tables."""
    KPI_NAMES = ("Total return", "Final equity", "Max drawdown", "Sharpe", "Win rate", "Profit factor", "Trades", "Max consec. losses")

    def __init__(self, export_name: str = "trades.csv"):
        super().__init__()
        self.trades = pd.DataFrame()
        self.export_name = export_name
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        k = QHBoxLayout()
        self.kpis = {n: Kpi(n) for n in self.KPI_NAMES}
        for w in self.kpis.values():
            k.addWidget(w)
        root.addLayout(k)
        self.tabs = QTabWidget()
        self.chart_panel = ChartPanel()
        self.chart = self.chart_panel.chart
        self.tabs.addTab(self.chart_panel, "Equity curve")
        self.mt = make_table(["Metric", "Value"])
        self.mt.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.mt.setColumnWidth(0, 380)
        self.tabs.addTab(self.mt, "Metrics")
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        self._tt_headers = ["#", "Entry time (UTC)", "Market", "Leg", "Side", "Quoted", "Entry", "Contracts", "Stake $", "Fee $", "Result", "P&L $",
                            "Return", "p used", "Kelly full", "Kelly used", "Regime (Auto)", "Kelly x", "Stake cap", "Equity $"]
        self.tt = make_table(self._tt_headers)
        row = QHBoxLayout(); self.export_btn = QPushButton("Export trades to CSV"); self.export_btn.clicked.connect(self.export)
        row.addStretch(1); row.addWidget(self.export_btn)
        v.addLayout(row); v.addWidget(self.tt)
        self.tabs.addTab(w, "Trades")
        self.mo = make_table(["Month", "Trades", "Win rate", "P&L $", "Return (on month-start equity)"])
        self.tabs.addTab(self.mo, "Monthly")
        self.bh = make_table(["Leg", "UTC hour", "Trades", "Win rate", "Avg price paid", "P&L $", "Avg return / trade"])
        self.tabs.addTab(self.bh, "By hour and leg")
        root.addWidget(self.tabs, 1)

    def show_result(self, trades: pd.DataFrame, eq: pd.DataFrame, dd: pd.Series, m: dict, capital: float, empty_msg: str) -> None:
        self.trades = trades
        n = m["trades"]
        self.kpis["Trades"].set(f"{n}")
        if n == 0:
            for name, kpi in self.kpis.items():
                if name != "Trades":
                    kpi.set("—")
            self.chart.message(empty_msg)
            self.mt.setRowCount(0); self.tt.setRowCount(0); self.mo.setRowCount(0); self.bh.setRowCount(0)
            return
        self.kpis["Total return"].set(pct(m["total_return"], 1, True), GREEN if m["total_return"] >= 0 else RED)
        self.kpis["Final equity"].set(f"${m['final_equity']:,.0f}")
        self.kpis["Max drawdown"].set(pct(m["max_drawdown"], 1), RED if m["max_drawdown"] < -0.2 else YELLOW if m["max_drawdown"] < -0.1 else None)
        self.kpis["Sharpe"].set(num(m["sharpe"], 2))
        self.kpis["Win rate"].set(pct(m["win_rate"], 1))
        self.kpis["Profit factor"].set(num(m["profit_factor"], 2))
        self.kpis["Max consec. losses"].set(f"{m['max_consec_losses']}  (wins {m['max_consec_wins']})")
        self.chart.plot(eq, dd, trades, capital)
        self._fill_metrics(m)
        self._fill_trades(trades)
        self._fill_monthly(trades, capital)
        self._fill_by_hour(trades)

    def _fill_metrics(self, m: dict) -> None:
        rows = sum(len(items) + 1 for _, items in METRIC_SECTIONS)
        self.mt.setRowCount(rows)
        r = 0
        for title, items in METRIC_SECTIONS:
            it = QTableWidgetItem(title); f = QFont(); f.setBold(True); it.setFont(f); it.setForeground(QColor(ACCENT))
            self.mt.setItem(r, 0, it); self.mt.setItem(r, 1, QTableWidgetItem("")); r += 1
            for label, key, kind in items:
                v = m.get(key)
                col = None
                if kind in ("$s", "%s") and isinstance(v, (int, float)) and np.isfinite(v):
                    col = GREEN if v > 0 else RED if v < 0 else None
                set_row(self.mt, r, [label, fmt_metric(v, kind)], {1: col})
                self.mt.item(r, 1).setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                r += 1

    def _fill_trades(self, t: pd.DataFrame) -> None:
        sess = "session" in t.columns                       # trades from the ledger say which session (and mode) placed them
        heads = self._tt_headers + (["Session"] if sess else [])
        self.tt.setColumnCount(len(heads))
        self.tt.setHorizontalHeaderLabels(heads)
        self.tt.setRowCount(len(t))
        for i, x in enumerate(t.itertuples(), start=0):
            col = GREEN if x.pnl > 0 else RED
            vals = [i + 1, x.entry_time.strftime("%Y-%m-%d %H:%M"), x.market, x.leg, x.side, f"{x.quoted:.3f}", f"{x.entry:.3f}", f"{x.contracts:,}",
                    money(x.stake), money(x.fee), x.outcome, money(x.pnl, True), pct(x.ret_on_stake, 1, True), pct(x.p_used, 1),
                    pct(x.f_full, 1), pct(x.f_used, 2), getattr(x, "regime", "") or "—",
                    f"{x.mult_used:.3f}" if np.isfinite(getattr(x, "mult_used", np.nan)) else "—", pct(x.cap_used, 2) if np.isfinite(getattr(x, "cap_used", np.nan)) else "—",
                    money(x.equity_after)]
            if sess:
                vals.append(f"{x.session}{' (settled after a restart)' if getattr(x, 'recovered', False) else ''}")
            set_row(self.tt, i, vals, {10: col, 11: col, 12: col})

    def _fill_by_hour(self, t: pd.DataFrame) -> None:
        rows = []
        for leg, x in t.groupby("leg"):
            rows.append((leg, "all", x))
            for h, y in x.groupby("hour"):
                rows.append((leg, f"{int(h):02d}", y))
        self.bh.setRowCount(len(rows))
        for i, (leg, h, x) in enumerate(rows):
            pnl = float(x.pnl.sum())
            c = GREEN if pnl > 0 else RED
            set_row(self.bh, i, [leg, h, len(x), pct((x.pnl > 0).mean(), 1), f"{x.entry.mean():.3f}", money(pnl, True), pct(x.ret_on_stake.mean(), 1, True)],
                    {5: c, 6: c})
            if h == "all":
                for j in range(self.bh.columnCount()):
                    it = self.bh.item(i, j)
                    f = QFont(); f.setBold(True); it.setFont(f)

    def _fill_monthly(self, t: pd.DataFrame, capital: float) -> None:
        d = t.copy()
        d["month"] = pd.to_datetime(d.settle_ts, unit="s", utc=True).dt.strftime("%Y-%m")
        g = d.groupby("month")
        self.mo.setRowCount(len(g))
        for i, (mth, x) in enumerate(g):
            start_eq = float(x.equity_after.iloc[0] - x.pnl.iloc[0])
            pnl = float(x.pnl.sum())
            c = GREEN if pnl > 0 else RED
            set_row(self.mo, i, [mth, len(x), pct((x.pnl > 0).mean(), 1), money(pnl, True), pct(pnl / start_eq if start_eq else 0, 2, True)], {3: c, 4: c})

    def export(self) -> None:
        if self.trades is None or self.trades.empty:
            QMessageBox.information(self, "Export", "There are no trades to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save trades", str(DATA / self.export_name), "CSV (*.csv)")
        if path:
            self.trades.drop(columns=["entry_time"]).to_csv(path, index=False)


class ParamsView(QWidget):
    """'Parameters in use': every parameter, with the value the Backtest uses next to the value the Live / paper session uses.
    White = shared (same value on both sides); yellow = deliberately different; grey = used by one side only."""

    def __init__(self):
        super().__init__()
        self.get_state = None                     # set by MainWindow: () -> (bt_cfg | None, bt_error, live_cfg, live_mode_label, frozen, preset)
        v = QVBoxLayout(self)
        self.head = QLabel(""); self.head.setObjectName("muted"); self.head.setWordWrap(True)
        self.table = make_table(["Parameter", "Backtest", "Live / paper session", "Note"])
        self.table.setWordWrap(True)
        v.addWidget(self.head)
        v.addWidget(self.table, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.refresh() if self.isVisible() else None)
        self.timer.start(1500)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.refresh()

    def refresh(self) -> None:
        if self.get_state is None:
            return
        try:
            bt, bt_err, lv, mode_label, frozen, preset = self.get_state()
            rows = describe_parameters(bt, lv, preset, mode_label, bt_err)
        except Exception as e:                    # noqa: BLE001
            self.head.setText(f"cannot describe the parameters: {e}")
            return
        self.head.setText(("The Live column shows the RUNNING session (settings frozen when you pressed Start). " if frozen else
                           "The Live column shows what a session started now would use. ") +
                          "White rows are shared: the same value on both sides. Yellow rows differ on purpose (see the note). Grey cells are used by only one side.")
        groups = []
        for r in rows:
            if not groups or groups[-1] != r.group:
                groups.append(r.group)
        self.table.setRowCount(len(rows) + len(groups))
        i, last = 0, None
        colors = {"shared": None, "differs": YELLOW, "backtest-only": None, "live-only": None, "not-used": MUTED}
        for r in rows:
            if r.group != last:
                it = QTableWidgetItem(r.group); f = QFont(); f.setBold(True); it.setFont(f); it.setForeground(QColor(ACCENT))
                self.table.setItem(i, 0, it)
                for j in (1, 2, 3):
                    self.table.setItem(i, j, QTableWidgetItem(""))
                i += 1
                last = r.group
            k = r.kind
            cells = [r.name, r.backtest, r.live, r.note]
            for j, txt in enumerate(cells):
                it = QTableWidgetItem(txt)
                it.setToolTip(txt)
                it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                col = None
                if j in (1, 2):
                    col = YELLOW if k == "differs" else MUTED if (k == "not-used" or (k == "backtest-only" and j == 2) or (k == "live-only" and j == 1)) else None
                elif j == 3:
                    col = MUTED
                if col:
                    it.setForeground(QColor(col))
                self.table.setItem(i, j, it)
            i += 1
        self.table.resizeRowsToContents()


class BacktestTab(QWidget):
    def __init__(self, settings: SettingsPanel, get_rows):
        super().__init__()
        self.sp, self.get_rows = settings, get_rows
        self.result: Optional[BacktestResult] = None
        root = QVBoxLayout(self)

        ctl = QGroupBox("Backtest-only settings (not used by the Live tab)")
        h = QGridLayout(ctl)
        today = QDate.currentDate()
        self.d0, self.d1 = QDateEdit(QDate(2026, 6, 1)), QDateEdit(today)
        for d in (self.d0, self.d1):
            d.setCalendarPopup(True)
            d.setDisplayFormat("yyyy-MM-dd")
        self.cap = spin(100, 1e9, 1000, 0, 100, "")
        self.cap.setPrefix("$ ")
        self.mode = QComboBox(); self.mode.addItems(PROB_MODES)
        self.mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.mode.setMinimumContentsLength(18)
        self.fixed_p = spin(0.1, 99.9, 25.0, 1, 1, " %")
        self.fixed_p.setEnabled(False)
        self.mode.currentIndexChanged.connect(lambda i: self.fixed_p.setEnabled(i == 1))
        self.compound = QCheckBox("Compound (size off current equity)"); self.compound.setChecked(True)
        self.compound.setToolTip("Leave ticked: the live engine always sizes off its current equity, so only a compounding backtest is comparable to live.")
        self.refresh_chk = QCheckBox("Refresh data first")
        self.refresh_chk.setChecked(True)
        self.refresh_chk.setToolTip("Before each run, download any newly settled Kalshi windows (about 10-20 s) if the local data is more than 20 minutes behind, so the backtest always includes the most recent data. "
                                    "Untick to run on exactly what is on disk.")
        self.ensure_fresh = None                    # set by MainWindow: (callback) -> refresh the data if needed, then call it
        self._to_pinned = self._from_pinned = False  # True once YOU edited the date; until then 'To' follows the newest data and 'From' the oldest
        self._programmatic = False
        self._lo = self._hi = None
        h.addWidget(QLabel("From"), 0, 0); h.addWidget(self.d0, 0, 1)
        h.addWidget(QLabel("To"), 0, 2); h.addWidget(self.d1, 0, 3)
        for i, (label, days) in enumerate((("30d", 30), ("60d", 60), ("90d", 90), ("All", 0))):
            b = QPushButton(label); b.setFixedWidth(52)
            b.clicked.connect(lambda _=False, dd=days: self.quick_range(dd))
            h.addWidget(b, 0, 4 + i)
        self.btn_latest = QPushButton("→ latest"); self.btn_latest.setFixedWidth(78)
        self.btn_latest.setToolTip("Set 'To' to the newest data and keep it following the data as it refreshes")
        self.btn_latest.clicked.connect(self.follow_latest)
        h.addWidget(self.btn_latest, 0, 8)
        h.addWidget(QLabel("Initial capital"), 1, 0); h.addWidget(self.cap, 1, 1)
        h.addWidget(QLabel("Win probability"), 1, 2); h.addWidget(self.mode, 1, 3, 1, 3)
        h.addWidget(QLabel("Fixed p"), 1, 6); h.addWidget(self.fixed_p, 1, 7)
        h.addWidget(self.compound, 2, 0, 1, 3)
        h.addWidget(self.refresh_chk, 2, 3, 1, 3)
        self.run_btn = QPushButton("Run backtest"); self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run)
        h.addWidget(self.run_btn, 2, 6, 1, 2)
        self.verify_btn = QPushButton("Verify live = backtest")
        self.verify_btn.setToolTip("Replays the LIVE trading engine (the code that sends the orders, on a paper broker) over exactly this period with exactly these settings, and compares every trade with the backtest: "
                                   "same markets, sides, contracts, prices, probabilities and P&L.  IDENTICAL means live trading takes the trades the backtest shows.  Takes a few seconds.")
        self.verify_btn.clicked.connect(self.verify_live)
        h.addWidget(self.verify_btn, 2, 8)
        self.using_lbl = QLabel(""); self.using_lbl.setObjectName("muted"); self.using_lbl.setWordWrap(True)
        h.addWidget(self.using_lbl, 3, 0, 1, 8)
        self.info = QLabel(""); self.info.setObjectName("muted"); self.info.setWordWrap(True)
        h.addWidget(self.info, 4, 0, 1, 8)
        self.fresh_lbl = QLabel(""); self.fresh_lbl.setWordWrap(True)
        h.addWidget(self.fresh_lbl, 5, 0, 1, 8)
        self.verify_lbl = QLabel(""); self.verify_lbl.setWordWrap(True); self.verify_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        h.addWidget(self.verify_lbl, 6, 0, 1, 9)
        self._workers: list = []
        root.addWidget(ctl)
        self.d0.dateChanged.connect(lambda *_: setattr(self, "_from_pinned", self._from_pinned or not self._programmatic))
        self.d1.dateChanged.connect(lambda *_: (setattr(self, "_to_pinned", self._to_pinned or not self._programmatic), self.refresh_fresh_label()))
        self._fresh_timer = QTimer(self); self._fresh_timer.timeout.connect(self.refresh_fresh_label); self._fresh_timer.start(30000)

        self.perf = PerformanceView("backtest_trades.csv")
        self.kpis, self.tabs, self.chart = self.perf.kpis, self.perf.tabs, self.perf.chart
        self.mt, self.tt, self.mo, self.bh = self.perf.mt, self.perf.tt, self.perf.mo, self.perf.bh
        self.params_view = ParamsView()
        self.tabs.addTab(self.params_view, "Parameters in use")
        root.addWidget(self.perf, 1)
        self.sp.changed.connect(self.refresh_using)
        for w in (self.cap, self.fixed_p):
            w.valueChanged.connect(lambda *_: self.refresh_using())
        self.mode.currentIndexChanged.connect(lambda *_: self.refresh_using())
        self.compound.toggled.connect(lambda *_: self.refresh_using())
        self.refresh_using()

    def refresh_using(self) -> None:
        try:
            p = self.sp.params()
            risk = f"daily loss {self.sp.dloss.value():.0f}% / drawdown stop {self.sp.ddstop.value():.0f}%"
            auto_on = self.sp.auto_on.isChecked() and self.mode.currentText() == PROB_MODES[0]
            sizing = f"AUTO sizing, ladder {self.sp.auto_ladder.text()}" if auto_on else f"Kelly {self.sp.mult.value():.2f}x, cap {self.sp.cap.value():.1f}%"
            warn = (" ⚠ Auto needs the learned probability: with a fixed / hindsight probability this backtest uses the fixed Kelly / stake / contracts boxes instead."
                    if self.sp.auto_on.isChecked() and not auto_on else "")
            self.using_lbl.setText(f"This backtest uses: {self.sp.preset.currentText()} | {p.describe()} | shared left-panel settings ({sizing}, "
                                   f"assumed slippage {self.sp.slip.value():.1f}c, {risk}) + the backtest-only box above (capital ${self.cap.value():,.0f}, "
                                   f"{'compounding' if self.compound.isChecked() else 'FIXED capital'}). Full list: tab \"Parameters in use\"." + warn)
        except ValueError as e:
            self.using_lbl.setText(f"Settings problem: {e}")

    def set_range(self, lo: datetime, hi: datetime) -> None:
        """New data arrived: 'To' follows the newest window (and 'From' the oldest) unless you set the date yourself."""
        self._lo, self._hi = lo, hi
        self._programmatic = True
        try:
            if not self._from_pinned:
                self.d0.setDate(QDate(lo.year, lo.month, lo.day))
            if not self._to_pinned:
                self.d1.setDate(QDate(hi.year, hi.month, hi.day))
        finally:
            self._programmatic = False
        self.refresh_fresh_label()

    def quick_range(self, days: int) -> None:
        hi = self._hi
        lo = self._lo if days == 0 else max(self._lo, hi - timedelta(days=days))
        self._programmatic = True
        try:
            self.d0.setDate(QDate(lo.year, lo.month, lo.day))
            self.d1.setDate(QDate(hi.year, hi.month, hi.day))
        finally:
            self._programmatic = False
        self._from_pinned, self._to_pinned = days != 0, False         # 'All' = the oldest data again; a 30/60/90-day button fixes From but 'To' keeps following the data
        self.refresh_fresh_label()

    def follow_latest(self) -> None:
        self._to_pinned = False
        if self._hi is not None:
            self._programmatic = True
            try:
                self.d1.setDate(QDate(self._hi.year, self._hi.month, self._hi.day))
            finally:
                self._programmatic = False
        self.refresh_fresh_label()

    def refresh_fresh_label(self) -> None:
        if self._hi is None:
            self.fresh_lbl.setText("No local data yet - use 'Update data' in the left panel.")
            return
        lag = (datetime.now(timezone.utc) - self._hi).total_seconds()
        age = f"{int(lag // 60)} min" if lag < 7200 else f"{lag / 3600:.1f} h"
        txt = f"Data through {self._hi:%Y-%m-%d %H:%M} UTC ({age} ago)."
        late = False
        if self._to_pinned and self.d1.date() < QDate(self._hi.year, self._hi.month, self._hi.day):
            txt += f"  ⚠ 'To' is {self.d1.date().toString('yyyy-MM-dd')}: newer data is not included - press '→ latest'."
            late = True
        if lag > 45 * 60:
            txt += "  The newest windows may not be downloaded yet - a run refreshes them first (if 'Refresh data first' is ticked)."
            late = True
        self.fresh_lbl.setText(txt)
        self.fresh_lbl.setStyleSheet(f"color: {YELLOW};" if late else f"color: {MUTED};")

    def config(self) -> BacktestConfig:
        a, b = self.d0.date(), self.d1.date()
        if a > b:
            raise ValueError("'From' date is after 'To' date")
        return BacktestConfig(
            start=date(a.year(), a.month(), a.day()), end=date(b.year(), b.month(), b.day()), capital=self.cap.value(), params=self.sp.params(),
            fee_model=self.sp.fee.currentText(), slippage=self.sp.slip.value() / 100, kelly_mult=self.sp.mult.value(), max_frac=self.sp.cap.value() / 100,
            min_edge=self.sp.edge.value() / 100, max_contracts=self.sp.maxc.value(), prob_mode=self.mode.currentText(),
            fixed_p=self.fixed_p.value() / 100, lookback_days=self.sp.look.value(), halflife_days=self.sp.halflife(), core_hours=self.sp.core_hours(), core_k=self.sp.corek.value(), near_k=self.sp.near_k(), near_width=self.sp.nearw.value(),
            dd_throttle=self.sp.throttle.isChecked(), daily_loss_limit=self.sp.dloss.value() / 100, max_drawdown_stop=self.sp.ddstop.value() / 100, min_history=MIN_HISTORY,
            compounding=self.compound.isChecked(), p_model=self.sp.p_model(), p_basis=self.sp.p_basis(), hour_k=self.sp.hourk.value(), auto=self.sp.auto_config())

    def safe_config(self):
        try:
            return self.config(), ""
        except ValueError as e:
            return None, str(e)

    def run(self) -> None:
        """Run button: refresh the data first (when stale and ticked), then backtest on the newest data."""
        if self.ensure_fresh is not None and self.refresh_chk.isChecked():
            self.run_btn.setEnabled(False)
            self.info.setText("Checking for newly settled Kalshi windows first …")
            self.ensure_fresh(self._run_now)
        else:
            self._run_now()

    def _run_now(self) -> None:
        self.run_btn.setEnabled(True)
        try:
            cfg = self.config()
            rows = self.get_rows()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                res = run_backtest(rows, cfg)
                cov = coverage(rows, cfg.params, int(datetime(cfg.start.year, cfg.start.month, cfg.start.day, tzinfo=timezone.utc).timestamp()),
                               int(datetime(cfg.end.year, cfg.end.month, cfg.end.day, tzinfo=timezone.utc).timestamp()) + 86400)
            finally:
                QApplication.restoreOverrideCursor()
        except ValueError as e:
            QMessageBox.warning(self, "Backtest", str(e))
            return
        self.result = res
        m = res.metrics
        hi = data_range(rows)[1]
        note = f"{cfg.start} → {cfg.end} (newest data {hi:%Y-%m-%d %H:%M} UTC) | {cfg.params.describe()} | {m['signals_in_range']} signals, {m['trades']} traded"
        if m.get("windows_paused") or m.get("windows_halted"):
            note += f" | risk limits skipped {m.get('windows_paused', 0)} windows (daily loss pause) and {m.get('windows_halted', 0)} (drawdown stop)"
        if cov[1]:
            note += f" | data coverage for these hours: {cov[0]}/{cov[1]} markets"
            if cov[0] < 0.97 * cov[1]:
                note += "  ⚠ some markets in this range have no candle data (older history is only downloaded for some hours - use 'Download older history' in the left panel)"
        if cfg.prob_mode == PROB_MODES[2]:
            note += "  ⚠ full-sample probability uses hindsight — optimistic"
        self.info.setText(note)
        self.show_result(res)

    def _verify_show(self, text: str, color: str) -> None:
        self.verify_lbl.setText(text)
        self.verify_lbl.setStyleSheet(f"color: {color};")

    def verify_live(self) -> None:
        """Replay the live engine over this period/settings in a worker thread and compare its trades with the backtest's."""
        try:
            cfg = self.config()
            live_config_from(cfg)                       # settings with no live equivalent (non-compounding, hindsight probability) are refused with a reason
        except ValueError as e:
            self._verify_show(f"Cannot verify: {e}", RED)
            return
        rows = self.get_rows()
        self.verify_btn.setEnabled(False)
        self._verify_show("Replaying the live engine over this period …", MUTED)
        w = FnWorker(lambda log: verify_replay(rows, cfg))
        self._workers.append(w)
        w.done.connect(self._verified)
        w.start()

    def _verified(self, res, err) -> None:
        self.verify_btn.setEnabled(True)
        if err:
            self._verify_show(f"Could not verify: {err}", RED)
            return
        cmp, _bt, rp = res
        tail = (f"  ({rp.windows:,} windows replayed in {rp.seconds:.1f} s.  The replay uses the same candle quotes as the backtest, so the only things it cannot contain are the real order book at the second "
                f"of the order and the real fill - the Live tab's 'Live vs backtest' measures those on your real trades.)")
        self._verify_show(("✔ Live engine = backtest.  " if cmp.identical else "✘ The live engine and the backtest DIFFER.  ") + cmp.text(6) + tail, GREEN if cmp.identical else RED)

    def show_result(self, res: BacktestResult) -> None:
        self.perf.show_result(res.trades, res.equity, res.drawdown, res.metrics, res.cfg.capital,
                              "No trades in this range under these settings\n(Kelly sizes to zero when the historical win probability is below the price)")

    def export(self) -> None:
        self.perf.export()


# ── live tab ──────────────────────────────────────────────────────────────────
class PreflightDialog(QDialog):
    """Progress + go / no-go report of the pre-flight check. ``action`` records how the user chose to proceed."""
    RANK = {preflight.FAIL: 0, preflight.WARN: 1, preflight.PASS: 2, preflight.INFO: 3, preflight.SKIP: 4}

    def __init__(self, parent, live: bool, then_start: bool, cap_now: float):
        super().__init__(parent)
        self.setWindowTitle("Pre-flight check")
        self.resize(1040, 700)
        self.live, self.then_start, self.cap_now = live, then_start, cap_now
        self.action: Optional[str] = None
        self.report: Optional[preflight.Report] = None
        self._go_action = "start"
        v = QVBoxLayout(self)
        self.title = QLabel("Running pre-flight checks…")
        self.title.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.verdict.hide()
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(130)
        self.table = make_table(["", "Check", "Result"])
        self.table.setWordWrap(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.hide()
        self.foot = QLabel(""); self.foot.setObjectName("muted"); self.foot.setWordWrap(True)
        row = QHBoxLayout()
        self.btn_copy = QPushButton("Copy report"); self.btn_copy.clicked.connect(self._copy); self.btn_copy.setEnabled(False)
        self.btn_paper = QPushButton("Run as paper instead"); self.btn_paper.clicked.connect(lambda: self._choose("paper")); self.btn_paper.hide()
        self.btn_cap = QPushButton("Use the suggested cap instead"); self.btn_cap.clicked.connect(lambda: self._choose("start_capped")); self.btn_cap.hide()
        self.btn_go = QPushButton("…"); self.btn_go.setObjectName("go"); self.btn_go.clicked.connect(lambda: self._choose(self._go_action)); self.btn_go.hide()
        self.btn_close = QPushButton("Cancel" if then_start else "Close"); self.btn_close.clicked.connect(self.reject)
        row.addWidget(self.btn_copy); row.addStretch(1)
        for b in (self.btn_paper, self.btn_cap, self.btn_close):
            row.addWidget(b)
        for w in (self.title, self.verdict, self.log, self.table, self.foot):
            v.addWidget(w)
        v.addLayout(row)
        self.btn_go.setMinimumHeight(38)
        v.addWidget(self.btn_go)                       # the main action gets its own full-width row
        self.setMinimumSize(640, 460)

    def add_log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _choose(self, action: str) -> None:
        self.action = action
        self.accept()

    def _copy(self) -> None:
        if self.report:
            QApplication.clipboard().setText(self.report.text())

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.table.resizeRowsToContents()

    def set_report(self, rep: "preflight.Report", saved: Optional[Path] = None) -> None:
        self.report = rep
        col = {"GO": GREEN, "CAUTION": YELLOW, "NO-GO": RED}[rep.verdict]
        c = QColor(col)
        self.title.setText(f"Pre-flight result — {datetime.fromtimestamp(rep.created, timezone.utc):%H:%M:%S} UTC")
        self.verdict.setText(rep.headline() + ("" if rep.live else "\n(Paper mode: the Kalshi account was not checked. The verdict is about going LIVE.)"))
        self.verdict.setStyleSheet(f"background: rgba({c.red()},{c.green()},{c.blue()},0.16); border: 1px solid {col}; border-radius: 6px; padding: 10px; font-size: 14px; font-weight: 600;")
        self.verdict.show()
        self.log.hide()
        rows = sorted(enumerate(rep.checks), key=lambda t: (self.RANK.get(t[1].status, 9), t[0]))
        items = [r for _, r in rows] + [preflight.Check("note", "Note", preflight.INFO, n) for n in rep.notes]
        self.table.setRowCount(len(items))
        colors = {preflight.PASS: GREEN, preflight.WARN: YELLOW, preflight.FAIL: RED, preflight.INFO: MUTED, preflight.SKIP: MUTED}
        for i, ck in enumerate(items):
            for j, txt in enumerate((ck.status, f"{ck.group} · {ck.name}", ck.detail)):
                it = QTableWidgetItem(txt)
                it.setToolTip(txt)
                it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                if j == 0:
                    it.setForeground(QColor(colors.get(ck.status, MUTED)))
                    f = it.font(); f.setBold(True); it.setFont(f)
                self.table.setItem(i, j, it)
        self.table.show()
        self.table.resizeColumnToContents(0); self.table.resizeColumnToContents(1)
        self.table.resizeRowsToContents()
        self.foot.setText((f"Report saved: {saved}" if saved else "") + ("   |   Forward test = the pre-registered R1 SPRT from forward_test.py, re-evaluated on the data refreshed just now." if rep.forward else ""))
        self.btn_copy.setEnabled(True)
        self._buttons(rep)

    def _buttons(self, rep: "preflight.Report") -> None:
        if not self.then_start:
            return
        cap = rep.stake_cap
        if not self.live:
            self.btn_go.setText("▶  Start paper session" + ("" if rep.verdict == "GO" else " (live verdict: " + rep.verdict + ")"))
            self._go_action = "start"
            self.btn_go.setEnabled(True)
            self.btn_go.show()
            return
        if rep.verdict == "GO":
            self.btn_go.setText("▶  Continue to LIVE confirmation"); self._go_action = "start"; self.btn_go.setEnabled(True)
        elif rep.verdict == "CAUTION":
            # CAUTION is advice, not a gate: the primary action runs LIVE with exactly the settings you entered; a lower cap is only an optional alternative
            self.btn_go.setText(f"▶  Continue to LIVE with my settings (max stake {self.cap_now * 100:.1f}% per trade)"); self._go_action = "start"
            self.btn_go.setEnabled(True)
            if cap is not None and self.cap_now > cap + 1e-9:
                self.btn_cap.setText(f"Use the suggested {cap * 100:.1f}% cap instead"); self.btn_cap.show()
            self.btn_paper.show()
        else:
            self.btn_go.setText("⛔  Live is blocked — fix the failed checks"); self.btn_go.setEnabled(False)
            self.btn_paper.show()
        self.btn_go.show()


class LiveTab(QWidget):
    def __init__(self, settings: SettingsPanel, get_rows, qs: QSettings):
        super().__init__()
        self.sp, self.get_rows, self.qs = settings, get_rows, qs
        self.client: Optional[KalshiClient] = None
        self.shard: Optional[int] = None            # exchange shard of the BTC 15-minute series: its cash funds the orders
        self._alerted: set = set()                  # alert titles already shown as a pop-up this session
        self._popups: list = []
        self.worker: Optional[LiveWorker] = None
        self.engine: Optional[LiveEngine] = None
        self._stats_cache: Optional[tuple] = None
        self._workers: list = []
        self.live_trades: list = []                 # this session's settled trades (LiveEngine rows) -> the shared PerformanceView
        self._session_start_ts: Optional[int] = None
        self._session_capital: Optional[float] = None
        self._pf_running = False
        self._pf_report: Optional[preflight.Report] = None
        self._pf_dlg: Optional[PreflightDialog] = None
        self._auto_key: Optional[str] = None
        self.adopt_rows = None          # set by MainWindow: callable(rows) to adopt data refreshed by the pre-flight
        self._exchange_note = ""
        self.ledger = Ledger()          # every trade any session places is saved here, so the history is still there after a restart
        self.reconcile_result_fn = None  # (ticker) -> 'yes' | 'no' | None; None = ask Kalshi (tests inject a fake)
        self.ensure_fresh = None         # set by MainWindow: (callback) -> refresh the candle data if stale, then call it
        root = QVBoxLayout(self)

        banner = QLabel("Heads-up: the edge is a transient evening-hours anomaly that is already fading (see REPORT.md); wider/all-day presets are experimental and test worse. Paper mode simulates fills and sends no orders. "
                        "LIVE sends real IOC orders to your Kalshi account — start small, and try the DEMO environment first.")
        banner.setObjectName("warn"); banner.setWordWrap(True)
        root.addWidget(banner)
        self.env_lbl = QLabel(""); self.env_lbl.setWordWrap(True)
        root.addWidget(self.env_lbl)

        top = QHBoxLayout()
        # connection
        g = QGroupBox("Kalshi account")
        f = QGridLayout(g)
        self.env = QComboBox()
        for e in ENVS.values():
            self.env.addItem(e.label, e.name)
        i = self.env.findData(str(self.qs.value("kalshi/env", "demo") or "demo"))
        self.env.setCurrentIndex(i if i >= 0 else 0)
        self.env.setToolTip("Demo = Kalshi's play-money environment (separate account and API key from demo.kalshi.co): the safe way to test the order path. Production = your real account.")
        self.key_id = QLineEdit(str(self.qs.value("kalshi/key_id", "") or ""))
        self.key_id.setPlaceholderText("API Key ID (the UUID Kalshi shows next to your key)")
        saved_key = str(self.qs.value("kalshi/keyfile", "") or "")
        self.keypath = QLineEdit(saved_key if saved_key and Path(saved_key).exists() else "")
        self.keypath.setPlaceholderText("Path to your private key file (.pem) from Kalshi")
        self.btn_browse = QPushButton("Browse…"); self.btn_browse.clicked.connect(self.browse)
        self.btn_connect = QPushButton("Connect and test"); self.btn_connect.setObjectName("primary"); self.btn_connect.clicked.connect(self.connect_account)
        self.btn_test = QPushButton("Order-path test"); self.btn_test.setEnabled(False)
        self.btn_test.setToolTip("Sends ONE 1-contract IMMEDIATE-OR-CANCEL bid at 1 cent on the current market to the selected environment: it cannot rest on the book and fills only if someone sells YES at 1 cent "
                                 "(cost at most one cent). It proves signing and the order format end to end.")
        self.btn_test.clicked.connect(self.order_path_test)
        self.acct = QLabel("Not connected"); self.acct.setObjectName("muted")
        self.bp = QLabel("—")
        self.health = QLabel("—"); self.health.setObjectName("muted"); self.health.setWordWrap(True)
        f.addWidget(QLabel("Environment"), 0, 0); f.addWidget(self.env, 0, 1, 1, 2)
        f.addWidget(self.key_id, 1, 0, 1, 3)
        f.addWidget(self.keypath, 2, 0, 1, 2); f.addWidget(self.btn_browse, 2, 2)
        f.addWidget(self.btn_connect, 3, 0); f.addWidget(self.btn_test, 3, 1, 1, 2)
        f.addWidget(QLabel("Account"), 4, 0); f.addWidget(self.acct, 4, 1, 1, 2)
        f.addWidget(QLabel("Cash (buying power)"), 5, 0); f.addWidget(self.bp, 5, 1, 1, 2)
        self.health.setMinimumHeight(56); self.health.setAlignment(Qt.AlignmentFlag.AlignTop)
        f.addWidget(self.health, 6, 0, 1, 3)
        g.setMinimumHeight(290)
        top.addWidget(g, 3)

        # session
        g = QGroupBox("Live-only settings (not used by the Backtest)")
        f = QFormLayout(g)
        self.capital = spin(1, 1e9, 1000, 2, 100); self.capital.setPrefix("$ ")
        self._capital_pinned = False                 # True once YOU typed a capital: the account's cash no longer overwrites it
        self._cap_programmatic = False
        self._cash: Optional[dict] = None            # last read of the connected account's cash (read_cash)
        self.btn_cap_auto = QPushButton("↺"); self.btn_cap_auto.setFixedWidth(34)
        self.btn_cap_auto.setToolTip("Use the default again: LIVE and connected = the cash available in your Kalshi account; otherwise $1,000")
        self.cap_note = QLabel(""); self.cap_note.setObjectName("muted"); self.cap_note.setWordWrap(True)
        self.capital.valueChanged.connect(self._capital_edited)
        self.btn_cap_auto.clicked.connect(self._capital_default)
        self.mode = QComboBox(); self.mode.addItems(["Paper — simulate fills, send no orders", "LIVE — send real orders to Kalshi (environment in the account box)"])
        self.max_slip = spin(0, 20, 3.0, 1, 0.5, " ¢")
        self.max_slip.setToolTip("LIVE orders only: the IOC limit price = the quoted ask + this, so an order never pays more. It does NOT change the position size - bets are sized with the "
                                 "shared 'Assumed slippage' in the left panel, exactly like the backtest. Keep it at or above the assumed slippage or orders will rarely fill.")
        self.prob_src = QComboBox(); self.prob_src.addItems(["Historical win rate of the strategy", "Manual value"])
        self.manual_p = spin(1, 99, 25, 1, 1, " %"); self.manual_p.setEnabled(False)
        for cb in (self.mode, self.prob_src):
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); cb.setMinimumContentsLength(16)
        self.prob_src.currentIndexChanged.connect(lambda i: self.manual_p.setEnabled(i == 1))
        cap_row = QWidget(); cr = QHBoxLayout(cap_row); cr.setContentsMargins(0, 0, 0, 0); cr.addWidget(self.capital, 1); cr.addWidget(self.btn_cap_auto)
        f.addRow("Initial capital (this session)", cap_row); f.addRow(self.cap_note); f.addRow("Mode", self.mode); f.addRow("Order limit: quoted ask +", self.max_slip)
        f.addRow("Win probability source", self.prob_src); f.addRow("Manual p", self.manual_p)
        self.autofund = QCheckBox("Auto-fund the trading shard (like the Kalshi app)")
        self.autofund.setChecked(str(self.qs.value("kalshi/autofund", "true")).lower() in ("true", "1"))
        self.autofund.setToolTip("Kalshi funds an API order only from the exchange shard BTC 15-minute markets trade on (shard 2), while the Kalshi app moves cash there for you. With this ticked the session does the same: "
                                 "at start, once early in every window, and again if an order is rejected for cash, it moves cash from your OTHER shards onto the trading shard. Cash only moves between shards of your own "
                                 "account, never out of it. LIVE only; untick to manage the shards yourself.")
        self.autofund.toggled.connect(lambda on: self.qs.setValue("kalshi/autofund", on))
        try:
            fm = float(self.qs.value("kalshi/fund_max", 0.0) or 0.0)
        except (TypeError, ValueError):
            fm = 0.0
        self.fund_max = spin(0, 1e9, fm, 2, 100); self.fund_max.setPrefix("$ "); self.fund_max.setSpecialValueText("session capital")
        self.fund_max.setToolTip("Auto-fund keeps at most this much cash on the trading shard (the engine sizes bets from it). 'session capital' = the initial capital above (plus any gains).")
        self.fund_max.valueChanged.connect(lambda v: self.qs.setValue("kalshi/fund_max", v))
        f.addRow(self.autofund); f.addRow("Keep at most on the trading shard", self.fund_max)
        note = QLabel("Risk limits (daily loss, drawdown stop), sizing, slippage and the strategy come from the shared left panel.")
        note.setObjectName("muted"); note.setWordWrap(True)
        f.addRow(note)
        top.addWidget(g, 3)

        # controls
        g = QGroupBox("Control")
        v = QVBoxLayout(g)
        self.btn_start = QPushButton("▶  Start trading"); self.btn_start.setObjectName("go"); self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("■  Stop"); self.btn_stop.clicked.connect(self.stop); self.btn_stop.setEnabled(False)
        self.btn_kill = QPushButton("⛔  KILL — cancel orders and halt"); self.btn_kill.setObjectName("danger"); self.btn_kill.clicked.connect(self.kill); self.btn_kill.setEnabled(False)
        self.state = QLabel("Idle"); self.state.setObjectName("muted"); self.state.setWordWrap(True)
        self.btn_pf = QPushButton("🔎  Pre-flight check (go / no-go)"); self.btn_pf.clicked.connect(lambda: self.run_preflight(then_start=False))
        self.btn_pf.setToolTip("Updates the data if stale, re-runs the pre-registered forward test, checks the edge, the feed, your risk settings and (LIVE) the account.")
        self.pf_lbl = QLabel("Pre-flight: not run yet"); self.pf_lbl.setObjectName("muted"); self.pf_lbl.setWordWrap(True)
        self.auto_pf = QCheckBox("Auto pre-flight at 21:45 UTC"); self.auto_pf.setToolTip("Runs the pre-flight check by itself at 21:45 UTC. The app must be open.")
        self.auto_pf.setChecked(str(self.qs.value("auto_preflight", "true")).lower() in ("true", "1"))
        self.auto_pf.toggled.connect(lambda on: self.qs.setValue("auto_preflight", on))
        for w in (self.btn_pf, self.btn_start, self.btn_stop, self.btn_kill, self.pf_lbl, self.auto_pf, self.state):
            v.addWidget(w)
        v.addStretch(1)
        top.addWidget(g, 2)
        root.addLayout(top)

        self.using_lbl = QLabel(""); self.using_lbl.setObjectName("muted"); self.using_lbl.setWordWrap(True)
        root.addWidget(self.using_lbl)
        self.auto_lbl = QLabel(""); self.auto_lbl.setWordWrap(True); self.auto_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.auto_lbl.setStyleSheet(f"background: rgba(46,160,107,0.12); border: 1px solid {GREEN}; border-radius: 6px; padding: 6px; color: {TEXT};")
        self.auto_lbl.hide()
        root.addWidget(self.auto_lbl)
        k = QHBoxLayout()
        self.kpis = {n: Kpi(n) for n in ("Next decision in", "Win probability (hist.)", "Kelly @ 16¢", "Open positions")}
        for w in self.kpis.values():
            k.addWidget(w)
        root.addLayout(k)

        # performance: the SAME view (KPIs, metrics, trades, monthly, by hour) the Backtest tab uses, fed with this session's trades
        self.perf = PerformanceView("live_session_trades.csv")
        self.perf.chart.message("Equity curve appears after the first settled trade")
        self.chart, self.tt = self.perf.chart, self.perf.tt
        sizing = QWidget(); sv = QVBoxLayout(sizing); sv.setContentsMargins(0, 0, 0, 0)
        sv.addWidget(QLabel("Sizing preview — what Kelly would stake at each price with the current settings"))
        self.pv = make_table(["Price (+slip)", "Full Kelly", "Used", "Stake $", "Contracts", "Auto: regime · Kelly x · cap"])
        sv.addWidget(self.pv)
        self.logbox = QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumBlockCount(2000)
        self.params_view = ParamsView()
        self.perf.tabs.addTab(sizing, "Sizing preview")
        cmpw = QWidget(); cv = QVBoxLayout(cmpw); cv.setContentsMargins(0, 0, 0, 0)
        self.cmp_lbl = QLabel("Press \"Compare with backtest\": every recorded session of the selected mode is re-run through the backtest on the same windows with the settings it traded with, and set against what it really did.")
        self.cmp_lbl.setWordWrap(True); self.cmp_lbl.setObjectName("muted"); self.cmp_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cmp_table = make_table(["Window (UTC)", "Status", "Side", "Quote live / candle", "Paid (slippage)", "Contracts live / asked / backtest", "P&L live", "P&L backtest", "Result"])
        cv.addWidget(self.cmp_lbl); cv.addWidget(self.cmp_table, 1)
        self.perf.tabs.addTab(cmpw, "Live vs backtest")
        self.perf.tabs.addTab(self.logbox, "Log")
        self.perf.tabs.addTab(self.params_view, "Parameters in use")
        hist = QHBoxLayout()
        hist.addWidget(QLabel("Trades shown:"))
        self.scope = QComboBox()
        self.scope.addItems(["This session only", "All recorded trades — this mode", "All recorded trades — both modes"])
        self.scope.setCurrentIndex(1)
        self.scope.setToolTip("Every trade the strategy places - paper or LIVE, in any session - is saved in the trade ledger the moment it fills and when it settles, so the full history is still here after you "
                              "restart the app or reconnect. 'This mode' follows the Mode box (LIVE shows only trades placed in LIVE sessions).")
        self.scope.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.scope.setMinimumContentsLength(34)
        self.btn_hist = QPushButton("Reload history"); self.btn_hist.setToolTip("Re-read the trade ledger and settle any position an earlier session left open (asks Kalshi for the results)")
        self.btn_hist.clicked.connect(lambda: self.refresh_history(True))
        self.btn_cmp = QPushButton("Compare with backtest")
        self.btn_cmp.setToolTip("Re-runs the backtest on every recorded session of the selected mode (same windows, same settings it traded with) and shows, trade by trade, where the real session followed it and where the "
                                "real market did not: quote at the second vs candle close, real fill vs assumed slippage, fills vs contracts asked for, windows one side skipped.")
        self.btn_cmp.clicked.connect(self.compare_with_backtest)
        hist.addWidget(self.scope); hist.addWidget(self.btn_hist); hist.addWidget(self.btn_cmp); hist.addStretch(1)
        root.addLayout(hist)
        self.hist_lbl = QLabel(""); self.hist_lbl.setObjectName("muted"); self.hist_lbl.setWordWrap(True)
        root.addWidget(self.hist_lbl)
        root.addWidget(self.perf, 1)

        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh_preview); self.timer.start(5000)
        for w in (self.capital, self.max_slip, self.manual_p):
            w.valueChanged.connect(lambda *_: self._refresh_using())
        self.mode.currentIndexChanged.connect(lambda *_: self._refresh_using())
        self.prob_src.currentIndexChanged.connect(lambda *_: self._refresh_using())
        self.clock_timer = QTimer(self); self.clock_timer.timeout.connect(self._clock_tick); self.clock_timer.start(15000)
        for w in (self.capital,):
            w.valueChanged.connect(lambda *_: self.refresh_preview())
        self.sp.changed.connect(self.refresh_preview)
        self.prob_src.currentIndexChanged.connect(lambda *_: self.refresh_preview())
        self.manual_p.valueChanged.connect(lambda *_: self.refresh_preview())
        self.scope.currentIndexChanged.connect(lambda *_: self._update_perf())
        self.mode.currentIndexChanged.connect(lambda *_: self._update_perf())
        self.mode.currentIndexChanged.connect(lambda *_: self._refresh_env())
        self.env.currentIndexChanged.connect(lambda *_: self._refresh_env())
        self.mode.currentIndexChanged.connect(lambda *_: self._sync_capital())
        self.autofund.toggled.connect(lambda *_: self._sync_capital())
        self._refresh_env()
        self._sync_capital()
        self._update_perf()                                           # show the saved history right away

    # ── helpers ──
    def log(self, msg: str) -> None:
        self.logbox.appendPlainText(msg)

    def make_config(self, live: bool = False) -> LiveConfig:
        manual = self.prob_src.currentIndex() == 1
        return LiveConfig(capital=self.capital.value(), params=self.sp.params(), fee_model=self.sp.fee.currentText(), kelly_mult=self.sp.mult.value(),
                          max_frac=self.sp.cap.value() / 100, min_edge=self.sp.edge.value() / 100, max_contracts=self.sp.maxc.value(),
                          mode="live" if live else "paper", paper_slippage=self.sp.slip.value() / 100, max_slippage=self.max_slip.value() / 100,
                          prob_source="manual" if manual else "history", manual_p=self.manual_p.value() / 100, lookback_days=self.sp.look.value(), halflife_days=self.sp.halflife(),
                          core_hours=self.sp.core_hours(), core_k=self.sp.corek.value(), near_k=self.sp.near_k(), near_width=self.sp.nearw.value(), dd_throttle=self.sp.throttle.isChecked(), p_model=self.sp.p_model(), p_basis=self.sp.p_basis(), hour_k=self.sp.hourk.value(),
                          daily_loss_limit=self.sp.dloss.value() / 100, max_drawdown_stop=self.sp.ddstop.value() / 100, auto=self.sp.auto_config(),
                          auto_fund=bool(live and self.autofund.isChecked()), fund_max=self.fund_max.value())

    def _stats(self, cfg: LiveConfig) -> SignalStats:
        key = (cfg.params, cfg.p_model, cfg.p_basis, cfg.hour_k, cfg.prior_n, cfg.lookback_days, cfg.halflife_days, cfg.core_hours, cfg.core_k, cfg.near_k, cfg.near_width, cfg.auto if auto_active(cfg) else None, id(self.get_rows()))
        if self._stats_cache is None or self._stats_cache[0] != key:
            self._stats_cache = (key, build_stats(self.get_rows(), cfg))
        return self._stats_cache[1]

    def _session_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning() and self.engine is not None

    def preview_p(self, cost: float, leg: Optional[str] = None) -> tuple[float, int]:
        """Win probability the engine would use for a signal at ``cost`` in the current UTC hour (and how many signals back it)."""
        if self.prob_src.currentIndex() == 1:
            return self.manual_p.value() / 100, 10**9
        cfg = self.make_config()
        st = self.engine.stats if self._session_running() else self._stats(cfg)
        leg = leg or cfg.params.legs[0]
        hour = datetime.now(timezone.utc).hour
        p, n, _nh = st.p(leg, hour, cost, time.time())
        return p, n

    def refresh_preview(self) -> None:
        try:
            cfg = self.make_config()
            leg = cfg.params.legs[0]
            typ = 0.16 if leg == "DOG" else 0.92
            p_typ, n = self.preview_p(typ, leg)
        except Exception:
            return
        auto = auto_active(cfg)
        fee = cfg.fee_model
        slip = cfg.paper_slippage                              # the same assumed slippage the backtest and the live engine size with
        running = self._session_running()
        bank = self.engine.equity if running else self.capital.value()
        peak = self.engine.peak if running else bank
        st = (self.engine.stats if running else self._stats(cfg)) if auto else None
        hour, now = datetime.now(timezone.utc).hour, time.time()
        prices = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.28] if leg == "DOG" else [0.86, 0.88, 0.90, 0.92, 0.94, 0.96]
        self.pv.setRowCount(len(prices))

        def sized(c: float, p: float):
            if auto:
                d = decide(cfg.auto, st, leg, hour, c, now, bank, peak, bank, None, fee, slip, cfg.min_edge)
                return d.sizing, f"{d.regime} · x{d.mult:.2f} · cap {d.cap * 100:.1f}%"
            m, cp = throttled_sizing(cfg.kelly_mult, cfg.max_frac, bank, peak, cfg.dd_throttle)
            return size_bet(bank, p, min(c + slip, 0.99), fee, m, cp, cfg.max_contracts, cfg.min_edge), "—"

        for i, c in enumerate(prices):
            p, _ = self.preview_p(c, leg)
            if not np.isfinite(p):
                set_row(self.pv, i, [f"{c:.2f}", "—", "—", "—", "—", "—"]); continue
            z, atxt = sized(c, p)
            colr = GREEN if z.contracts else MUTED
            set_row(self.pv, i, [f"{c:.2f} (+{slip * 100:.0f}¢)  p={p * 100:.0f}%", pct(z.f_full, 1), pct(z.f_used, 2), money(z.stake) if z.contracts else "no bet",
                                 f"{z.contracts:,}" if z.contracts else "—", atxt], {1: colr, 2: colr, 3: colr, 4: colr})
        z, _atxt = sized(typ, p_typ if np.isfinite(p_typ) else 0.0)
        self.kpis["Kelly @ 16¢"].set(f"{pct(z.f_used, 2)} of bankroll" if z.contracts else "no bet", GREEN if z.contracts else MUTED)
        self.kpis["Win probability (hist.)"].set("—" if not np.isfinite(p_typ) else f"{p_typ * 100:.1f}%  (n={n if n < 10**8 else 'manual'})")
        if auto and not running:
            try:
                self.auto_lbl.setText(fmt_auto(snapshot(cfg.auto, st, leg, hour, typ, now, bank, peak, fee, slip, cfg.min_edge), bank))
                self.auto_lbl.show()
            except Exception:
                self.auto_lbl.hide()
        elif not auto:
            self.auto_lbl.hide()

    # ── connection ──
    def env_name(self) -> str:
        return str(self.env.currentData() or "demo")

    def _refresh_env(self) -> None:
        live = self.mode.currentIndex() == 1
        env = self.env_name()
        if not live:
            self.env_lbl.setText("PAPER mode: nothing is sent to any exchange. Signals come from Kalshi's live order book.")
            self.env_lbl.setStyleSheet(f"background: rgba(139,152,169,0.12); border: 1px solid {BORDER}; border-radius: 6px; padding: 6px; color: {TEXT};")
        elif env == "production":
            self.env_lbl.setText("⚠ LIVE on Kalshi PRODUCTION — orders use REAL MONEY.")
            self.env_lbl.setStyleSheet(f"background: rgba(224,85,97,0.18); border: 1px solid {RED}; border-radius: 6px; padding: 6px; color: {TEXT}; font-weight: 700;")
        else:
            self.env_lbl.setText("LIVE on Kalshi DEMO — real order flow, play money (separate demo account and key).")
            self.env_lbl.setStyleSheet(f"background: rgba(46,160,107,0.14); border: 1px solid {GREEN}; border-radius: 6px; padding: 6px; color: {TEXT};")
        if self.client is not None and self.client.env != env:                  # a connection belongs to one environment
            self.client = None
            self.acct.setText("Not connected"); self.bp.setText("—"); self.health.setText("—")
            self.btn_test.setEnabled(False)
            self._cash = None
            self._sync_capital()

    # ── initial capital follows the account's cash (until you type your own) ──
    def _capital_edited(self, *_a) -> None:
        if not self._cap_programmatic:
            self._capital_pinned = True
            self._sync_capital()

    def _capital_default(self) -> None:
        self._capital_pinned = False
        self._sync_capital()

    def _available_cash(self) -> Optional[tuple[float, str]]:
        """(cash the session can use, where it is) from the connected account, or None (paper mode / not connected).  With auto-fund all the cash on the account counts - it is moved to the trading shard -
        without it only the cash already ON the trading shard does."""
        if self.mode.currentIndex() != 1 or self.client is None or not self._cash:
            return None
        c = self._cash
        if c.get("shard") is not None and self.autofund.isChecked():
            return c["total"]["buying_power"], "all the cash on the account (auto-fund moves it to the trading shard)"
        if c.get("shard") is not None:
            return c["bal"]["buying_power"], f"the cash on exchange shard {c['shard']} (auto-fund is off)"
        return c["bal"]["buying_power"], "the cash on the account"

    def _sync_capital(self) -> None:
        """Default of 'Initial capital': LIVE and connected = the cash available in the account; otherwise $1,000.  A value you typed is kept (the button next to it restores the default)."""
        if self._session_running():
            return
        avail = self._available_cash()
        value = 1000.0 if avail is None else max(1.0, math.floor(avail[0] * 100) / 100)
        if avail is None:
            self.cap_note.setText("Default $1,000. In LIVE mode, once connected, it becomes the cash available in your Kalshi account." + ("  (You set this by hand.)" if self._capital_pinned else ""))
        elif self._capital_pinned:
            self.cap_note.setText(f"You set this by hand. The account has ${avail[0]:,.2f} available ({avail[1]}) - the button restores that.")
        else:
            self.cap_note.setText(f"Default = ${avail[0]:,.2f}: {avail[1]}. Type a smaller amount to risk less; the button restores the default.")
        if not self._capital_pinned:
            self._cap_programmatic = True
            try:
                self.capital.setValue(value)
            finally:
                self._cap_programmatic = False

    def _refresh_cash_sync(self) -> None:
        """Re-read the account's cash (one quick request) so the default capital is current when a session starts."""
        if self.client is None or self._capital_pinned or self._session_running():
            return
        try:
            snap = read_cash(self.client)
        except Exception:                                 # noqa: BLE001  (offline: keep what we have)
            return
        self._cash = snap
        self._sync_capital()

    def _refresh_cash_async(self) -> None:
        if self.client is None:
            return
        cli = self.client
        w = FnWorker(lambda log: read_cash(cli)); self._workers.append(w)

        def done(res, err) -> None:
            if res is not None and not err and self.client is cli:
                self._cash = res
                self._sync_capital()

        w.done.connect(done); w.start()

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Kalshi private key file", str(Path(self.keypath.text()).parent if self.keypath.text() else Path.home()), "Key files (*.pem *.key *.txt);;All files (*)")
        if path:
            self.keypath.setText(path)

    def connect_account(self) -> None:
        path, kid, env = self.keypath.text().strip(), self.key_id.text().strip(), self.env_name()
        self.btn_connect.setEnabled(False)
        self.acct.setText("Connecting…")

        def work(_log):
            creds = load_credentials(path, kid)
            cli = KalshiClient(creds, env)
            q = KalshiQuotes(base=cli.base)
            shard = q.exchange_index()                    # Kalshi funds an order only from the exchange shard its market trades on
            t0 = time.time()
            bal = cli.balance(shard)
            lat = time.time() - t0
            total = cli.balance() if shard is not None else bal
            names = q.shard_names() if shard is not None else {}
            orders = positions = fills = None
            try:
                orders = len(cli.open_orders())
            except Exception:                         # noqa: BLE001
                pass
            try:
                positions = [p for p in cli.positions() if str(p.get("ticker", "")).startswith(kalshi.SERIES + "-")]
            except Exception:                         # noqa: BLE001
                pass
            try:
                fills = cli.fills(limit=200, max_pages=1)
            except Exception:                         # noqa: BLE001
                pass
            return dict(client=cli, creds=creds, bal=bal, total=total, shard=shard, shard_name=names.get(shard, ""), lat=lat, orders=orders, positions=positions, fills=fills)

        w = FnWorker(work); self._workers.append(w)
        w.done.connect(self._connected); w.start()

    def _connected(self, res, err) -> None:
        self.btn_connect.setEnabled(True)
        if err:
            self.client = None
            self.acct.setText("Connection failed")
            self.bp.setText("—")
            self.health.setText(f"❌ {err}")
            self.btn_test.setEnabled(False)
            self.log(f"connect failed: {err}")
            return
        self.client = res["client"]
        self.shard = res.get("shard")
        self.qs.setValue("kalshi/env", self.env_name()); self.qs.setValue("kalshi/key_id", self.key_id.text().strip()); self.qs.setValue("kalshi/keyfile", self.keypath.text().strip())
        self.acct.setText(f"✅ key {res['creds'].masked()} · {self.client.env}")
        bp = res["bal"]["buying_power"]
        if self.shard is None:
            self.bp.setText(f"${bp:,.2f}   (portfolio value ${res['bal']['portfolio_value']:,.2f})")
            self.bp.setStyleSheet("")
        else:
            nm = f" ({res['shard_name']})" if res.get("shard_name") else ""
            self.bp.setText(f"${bp:,.2f} on exchange shard {self.shard}{nm} — total across shards ${res['total']['buying_power']:,.2f}")
            self.bp.setStyleSheet(f"color: {RED}; font-weight: 700;" if bp < 1.0 else "")
        self._cash = dict(bal=res["bal"], total=res["total"], shard=res.get("shard"))
        self._sync_capital()
        skew = self.client.skew_s
        bits = [f"API latency {res['lat'] * 1000:.0f} ms"]
        if self.shard is not None and bp < 1.0:
            split = ", ".join(f"shard {i}: ${v:,.2f}" for i, v in sorted(res["total"].get("breakdown", {}).items()))
            bits.insert(0, f"❌ NO CASH on exchange shard {self.shard}, where BTC 15-minute orders are funded from (total cash ${res['total']['buying_power']:,.2f}{': ' + split if split else ''}). "
                           f"Kalshi would reject every order. Transfer cash to shard {self.shard} on the Exchange Indexes page of your Kalshi account settings - the API does not do it for you")
        if skew is not None:
            bits.append(f"clock skew {skew:+.1f}s" + ("  ⚠ check the PC clock" if abs(skew) > 5 else ""))
        if res.get("orders") is not None:
            bits.append(f"{res['orders']} resting orders" + (f", {len(res['positions'])} open BTC-15m positions" if res.get("positions") is not None else ""))
        self._exchange_note = ""
        if res.get("fills") is not None:
            miss = self.ledger.unmatched_exchange_fills(res["fills"])
            n_btc = len({f.get("order_id") for f in res["fills"] if str(f.get("ticker", "")).startswith(kalshi.SERIES + "-")})
            bits.append(f"{n_btc} recent BTC-15m orders on the account, {len(miss)} not in the ledger")
            if miss:
                self._exchange_note = f" ⚠ {len(miss)} order(s) on your Kalshi account in these markets are not in the ledger (placed outside this app, or by a session that crashed before recording) - see the Log."
                for t in miss[:20]:
                    self.log(f"  account order not in the ledger: {t['ticker']} order {t['order_id']} qty {t['qty']:g} {t['time']}")
        self.health.setText("   |   ".join(bits))
        self.btn_test.setEnabled(True)
        self.log(f"connected to Kalshi {self.client.env} (read-only checks passed)")
        self._update_perf()
        self.refresh_history(True)                                    # show everything the strategy placed before, and settle positions an earlier session left open

    def order_path_test(self) -> None:
        if not self.client:
            return
        env = self.client.env
        if QMessageBox.question(self, "Order-path test", f"Send ONE 1-contract immediate-or-cancel BID at 1 cent to Kalshi {env.upper()}?\n\nIt cannot rest on the book and fills only if someone sells YES at 1 cent "
                                f"(cost at most $0.01). It checks signing and the order format end to end.") != QMessageBox.StandardButton.Yes:
            return
        cli = self.client

        def work(_log):
            q = KalshiQuotes(base=cli.base)
            tk = q.window_ticker(int(time.time() // 900 * 900))
            if not tk:
                raise RuntimeError("no open BTC 15-minute market right now - retry in a moment")
            br = KalshiBroker(cli, exchange_index=q.exchange_index())
            note = ""
            if auto_fund and br.exchange_index is not None and br.buying_power() < 0.05:
                r = br.fund(1.0)                          # the test needs a cent on the trading shard: move $1 there like the Kalshi app would
                note = f" (auto-fund moved ${r['moved']:,.2f} to the trading shard first)" if r["moved"] else ""
            return tk, br.order_path_test(tk), note

        auto_fund = self.autofund.isChecked()
        w = FnWorker(work); self._workers.append(w)
        w.done.connect(lambda res, err: self.log(f"order-path test {env}: " + (f"FAILED {err}" if err else f"{res[0]}: accepted (order {res[1]['order_id']}, filled {res[1]['fill_count']:g}, remaining {res[1]['remaining_count']:g}) - signing and order format are fine{res[2]}")))
        w.start()

    # ── session control ──
    def _lock(self, running: bool) -> None:
        self.btn_start.setEnabled(not running and not self._pf_running)
        self.btn_pf.setEnabled(not running and not self._pf_running)
        self.btn_stop.setEnabled(running)
        self.btn_kill.setEnabled(running)
        for w in (self.capital, self.btn_cap_auto, self.mode, self.max_slip, self.prob_src, self.manual_p, self.btn_connect, self.keypath, self.btn_browse, self.env, self.key_id, self.autofund, self.fund_max):
            w.setEnabled(not running)
        self.sp.set_enabled_all(not running)
        if not running:
            self.manual_p.setEnabled(self.prob_src.currentIndex() == 1)

    # ── pre-flight (runs automatically when you press Start) ──
    def start(self) -> None:
        self.run_preflight(then_start=True)

    def run_preflight(self, then_start: bool, auto: bool = False) -> None:
        if self._pf_running or (self.worker is not None and self.worker.isRunning()):
            if not auto:
                QMessageBox.information(self, "Pre-flight", "A pre-flight check or a trading session is already running.")
            return
        live = self.mode.currentIndex() == 1
        if live:
            self._refresh_cash_sync()                 # the default capital = the account's cash right now
        try:
            cfg = self.make_config(live)
        except Exception as e:                    # noqa: BLE001
            QMessageBox.warning(self, "Pre-flight", f"Settings problem: {e}")
            return
        key_path, key_id, client = self.keypath.text().strip(), self.key_id.text().strip(), self.client
        env = self.env_name() if live else "production"            # paper reads Kalshi's real (production) order book; LIVE uses the selected environment
        if client is not None and client.env != env:
            client = None
        dlg = PreflightDialog(self, live, then_start, self.sp.effective_cap())
        self._pf_dlg = dlg
        self._pf_running = True
        self.btn_start.setEnabled(False); self.btn_pf.setEnabled(False)
        self.log("=== pre-flight check started ===")
        w = FnWorker(lambda log: preflight.run_preflight(cfg, live=live, key_path=key_path, key_id=key_id, env=env, client=client, log=log))
        self._workers.append(w)
        w.log.connect(dlg.add_log)
        w.done.connect(lambda res, err: self._pf_done(dlg, res, err, live, cfg))
        w.start()
        if then_start:
            dlg.exec()
            self._after_preflight_dialog(dlg)
        else:
            dlg.show()
            if auto:
                QApplication.alert(self.window())

    def _pf_done(self, dlg: PreflightDialog, res, err, live: bool, cfg: LiveConfig) -> None:
        self._pf_running = False
        if err or res is None:
            rep = preflight.Report(created=time.time(), live=live, capital=cfg.capital)
            rep.checks.append(preflight.Check("system", "Pre-flight", preflight.FAIL, f"the check itself failed: {err}"))
        else:
            rep = res
        self._pf_report = rep
        if rep.rows is not None and rep.updated_data and self.adopt_rows:
            self.adopt_rows(rep.rows)
        if rep.client is not None and self.client is None:
            self.client = rep.client
            self.acct.setText(f"✅ connected (pre-flight) · {rep.client.env}")
            self.btn_test.setEnabled(True)
        saved = preflight.save_report(rep)
        col = {"GO": GREEN, "CAUTION": YELLOW, "NO-GO": RED}[rep.verdict]
        cap = rep.stake_cap
        self.pf_lbl.setText(f"Pre-flight {datetime.fromtimestamp(rep.created, timezone.utc):%H:%M} UTC: {rep.verdict}" + (f" (suggested cap {cap * 100:.1f}%, optional)" if cap and rep.verdict == "CAUTION" else ""))
        self.pf_lbl.setStyleSheet(f"color: {col}; font-weight: 600;")
        self.log(f"=== pre-flight: {rep.headline()}")
        for c in rep.checks:
            if c.status in (preflight.FAIL, preflight.WARN):
                self.log(f"   [{c.status}] {c.name}: {c.detail}")
        running = self.worker is not None and self.worker.isRunning()
        self.btn_start.setEnabled(not running); self.btn_pf.setEnabled(not running)
        self.refresh_preview()
        dlg.set_report(rep, saved)

    def _after_preflight_dialog(self, dlg: PreflightDialog) -> None:
        act, rep = dlg.action, dlg.report
        if act == "start_capped" and rep is not None and rep.stake_cap:
            if self.sp.auto_on.isChecked():
                self.sp.cap_auto_ladder(rep.stake_cap)
                self.log(f"every Auto ladder step capped at {rep.stake_cap * 100:.1f}% per trade by the pre-flight check")
            else:
                self.sp.cap.setValue(round(rep.stake_cap * 100, 2))
                self.log(f"max stake per trade set to {rep.stake_cap * 100:.1f}% by the pre-flight check")
        elif act == "paper":
            self.mode.setCurrentIndex(0)
        if act in ("start", "start_capped", "paper"):
            self._start_session(rep)
        else:
            self.log("start cancelled" if rep is not None else "start cancelled before the pre-flight finished")

    def _clock_tick(self) -> None:
        if not self.auto_pf.isChecked() or self._pf_running or (self.worker is not None and self.worker.isRunning()):
            return
        now = datetime.now(timezone.utc)
        key = f"{now:%Y-%m-%d}"
        if now.hour == 21 and now.minute >= 45 and self._auto_key != key:
            self._auto_key = key
            self.log("auto pre-flight (21:45 UTC)")
            self.run_preflight(then_start=False, auto=True)

    def _record_end(self, why: str) -> None:
        try:
            if self.engine is not None:
                preflight.record_session_end(why, self.engine.cfg.mode, self.engine.equity - self.engine.cfg.capital, len(self.engine.trades))
        except Exception:                          # noqa: BLE001
            pass

    def _start_session(self, report: Optional["preflight.Report"] = None) -> None:
        live = self.mode.currentIndex() == 1
        try:
            cfg = self.make_config(live)
            params = cfg.params
            stats = build_stats(self.get_rows(), cfg)
        except Exception as e:                    # noqa: BLE001
            QMessageBox.warning(self, "Start", f"Settings problem: {e}")
            return
        manual = cfg.prob_source == "manual"
        now = time.time()
        _lo, _hi = data_range(self.get_rows())
        age_h = (datetime.now(timezone.utc) - _hi).total_seconds() / 3600 if _hi else 1e9
        stale_note = (f"Local history ends {_hi:%Y-%m-%d %H:%M} UTC ({age_h / 24:.1f} days ago): the probability estimate will not include the days since. "
                      "Click 'Update data' first for the most accurate estimate.") if age_h > 36 else ""
        if stale_note:
            self.log("⚠ " + stale_note)
        thin = [(lg, stats.n_total(now, lg)) for lg in params.legs if stats.n_total(now, lg) < MIN_HISTORY]
        if not manual and thin:
            QMessageBox.warning(self, "Start", "Not enough historical signals for: " + ", ".join(f"{lg} (n={n}, need {MIN_HISTORY})" for lg, n in thin) +
                                ". Widen hours/prices, update data, or use a manual probability.")
            return
        if live:
            if not self.client or self.client.env != self.env_name():
                QMessageBox.warning(self, "Start", "Connect to your Kalshi account first (Connect and test) - in the environment selected in the account box.")
                return
            p_txt = "manual " + format(self.manual_p.value(), ".1f") + "%" if manual else (
                f"{'price-adjusted historical edge' if cfg.p_basis == 'edge' else 'raw historical win rate'}, "
                f"{'by UTC hour' if cfg.p_model == 'by_hour' else 'overall'} ({', '.join(f'{lg}: n={stats.n_total(now, lg)}' for lg in params.legs)})")
            cap0 = cfg.auto.ladder[0][2] if auto_active(cfg) else cfg.max_frac
            env = self.client.env
            word = "LIVE" if env == "production" else "DEMO"
            txt = (f"You are about to send {'REAL-MONEY' if env == 'production' else 'PLAY-MONEY (demo)'} orders to Kalshi {env.upper()}.\n\nStrategy: {params.describe()}\nInitial capital: ${cfg.capital:,.2f}\n"
                   f"Sizing: {sizing_text(cfg)}, min edge {cfg.min_edge * 100:.1f}c\n"
                   f"Win probability: {p_txt}\n"
                   f"Pre-flight: {(report.verdict + ' at ' + format(datetime.fromtimestamp(report.created, timezone.utc), '%H:%M') + ' UTC') if report else 'not run'}\n"
                   f"{('Pre-flight suggested a ' + format(report.stake_cap * 100, '.1f') + '% per-trade cap while the evidence is unconfirmed; you are using YOUR setting of ' + format(cap0 * 100, '.1f') + '%.' + chr(10)) if (report and report.stake_cap and cap0 > report.stake_cap + 1e-9) else ''}"
                   f"Loss limits: daily {cfg.daily_loss_limit * 100:.0f}% , drawdown stop {cfg.max_drawdown_stop * 100:.0f}%\n"
                   f"{('AUTO-FUND is ON: the session will move cash between YOUR OWN Kalshi exchange shards (never out of the account) so the trading shard holds up to ' + ('$' + format(cfg.fund_max, ',.2f') if cfg.fund_max > 0 else 'the session capital') + chr(10)) if cfg.auto_fund else ''}"
                   f"{('WARNING: ' + stale_note + chr(10)) if stale_note else ''}\nType {word} to confirm.")
            ans, ok = QInputDialog.getText(self, f"Confirm {word} trading", txt)
            if not ok or ans.strip() != word:
                self.log("start cancelled")
                return
        broker = KalshiBroker(self.client, exchange_index=self.shard) if live else PaperBroker(cfg.fee_model, cfg.paper_slippage)
        quotes = KalshiQuotes(base=self.client.base if live else base_url("production"))        # the signal comes from the same environment the orders go to
        self.worker = LiveWorker()
        self.engine = LiveEngine(cfg, quotes, broker, stats, lambda k, d: self.worker.ev.emit(k, d), log_dir=ledger_dir(), ledger=self.ledger)
        self.worker.engine = self.engine
        self.worker.ev.connect(self.on_event)
        self.live_trades = []
        self._alerted = set()
        self._session_start_ts, self._session_capital = int(time.time()), cfg.capital
        self._update_perf()
        self._refresh_using()
        self.log(f"=== session started: {cfg.mode.upper()} ${cfg.capital:,.2f} | {params.describe()} | p: {cfg.p_basis}/{cfg.p_model} ===")
        self._lock(True)
        self.state.setText(f"Running ({cfg.mode}) — waiting for the next decision window")
        self.worker.start()
        self._refresh_using()

    def stop(self) -> None:
        if self.engine and self.engine._positions:
            if QMessageBox.question(self, "Stop", "A position is still open. It stays in the trade ledger and is settled from Kalshi's result the next time the app starts (or when you press Reload history). Stop anyway?") != QMessageBox.StandardButton.Yes:
                return
        self._end_session("stopped by user")

    def kill(self) -> None:
        if self.engine:
            self.engine.kill("kill switch pressed")
        self._end_session("KILLED")

    def _end_session(self, why: str) -> None:
        self._record_end(why)
        if self.worker:
            self.worker.stop(); self.worker.wait(3000)
        if self.engine is not None:
            self.engine.close(why)                                   # the ledger learns when this session stopped trading
        if self.mode.currentIndex() == 1:
            self._refresh_cash_async()                               # the next session's default capital = what the account holds now
        self._lock(False)
        self._refresh_using()
        self.state.setText(f"Session ended: {why}")
        self.log(f"=== session ended: {why} ===")

    def on_event(self, kind: str, d: dict) -> None:
        if kind == "log":
            self.log(d["msg"])
        elif kind == "status":
            self.on_status(d)
        elif kind == "trade_close":
            self.live_trades.append(d)
            self._update_perf()
        elif kind == "alert":
            self.log(f"⚠ {d['title']}: {d['msg']}")
            self.state.setText(f"⚠ {d['title']} - see the Log")
            self.state.setStyleSheet(f"color: {YELLOW};")
            self._popup(d["title"], d["msg"], d["title"])
        elif kind == "halt":
            self._record_end(f"HALTED: {d['reason']}")
            self.state.setText(f"HALTED: {d['reason']}")
            self.state.setStyleSheet(f"color: {RED};")
            self.btn_start.setEnabled(False)
            self._popup("Trading HALTED", f"The session stopped trading and placed no further orders:\n\n{d['reason']}", "halt:" + d["reason"][:40])

    def _popup(self, title: str, text: str, key: str) -> None:
        """A non-blocking window (and a beep) the first time a problem appears in this session, so a halted or order-rejecting session cannot go unnoticed."""
        if key in self._alerted:
            return
        self._alerted.add(key)
        QApplication.beep()
        box = QMessageBox(QMessageBox.Icon.Warning, title, text, QMessageBox.StandardButton.Ok, self)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._popups.append(box)
        box.show()

    def refresh_history(self, reconcile: bool = True) -> None:
        """Re-read the trade ledger; with ``reconcile`` also settle positions an earlier session left open (Kalshi results, in a background thread)."""
        if not reconcile:
            self._update_perf()
            return
        skip = [self.engine.session_id] if self._session_running() else []         # a running session settles its own positions
        fn = self.reconcile_result_fn or KalshiQuotes().result
        w = FnWorker(lambda log: self.ledger.reconcile(fn, time.time(), skip_sessions=skip))
        self._workers.append(w)
        w.done.connect(self._history_ready)
        w.start()

    def compare_with_backtest(self) -> None:
        mode = "live" if self.mode.currentIndex() == 1 else "paper"
        self.btn_cmp.setEnabled(False)
        self.cmp_lbl.setObjectName("muted")
        self.cmp_lbl.setText(f"Updating the candle data, then re-running the backtest on every recorded {mode.upper()} session …")

        def go() -> None:
            rows = self.get_rows()
            w = FnWorker(lambda log: reconcile_sessions(self.ledger, rows, mode=mode))
            self._workers.append(w)
            w.done.connect(lambda res, err: self._compared(res, err, mode))
            w.start()

        if self.ensure_fresh is not None:
            self.ensure_fresh(go)
        else:
            go()

    def _compared(self, rec, err, mode: str) -> None:
        self.btn_cmp.setEnabled(True)
        self.cmp_table.setRowCount(0)
        if err:
            self.cmp_lbl.setText(f"Could not compare: {err}")
            self.cmp_lbl.setStyleSheet(f"color: {RED};")
            return
        self.cmp_lbl.setText(f"{mode.upper()} sessions. " + rec.text())
        self.cmp_lbl.setStyleSheet("")
        t = rec.table()
        if t.empty:
            return
        t = t.sort_values(["window", "market"], ascending=False).reset_index(drop=True)
        colors = {"matched": GREEN, "live only": YELLOW, "backtest only": RED}
        self.cmp_table.setRowCount(len(t))
        for i, r in enumerate(t.itertuples()):
            sides = r.side_live if r.status == "live only" else r.side_bt if r.status == "backtest only" else (r.side_live if r.side_live == r.side_bt else f"{r.side_live} ≠ {r.side_bt}")
            quote = "—" if r.status == "backtest only" else (f"{r.quoted_live:.2f}" if r.status == "live only" else f"{r.quoted_live:.2f} / {r.quoted_bt:.2f}")
            paid = "—" if r.status == "backtest only" else f"{r.entry_live:.3f} ({r.slip_c:+.1f}c)"
            size = (f"— / — / {r.contracts_bt:,}" if r.status == "backtest only" else f"{r.contracts_live:,} / {r.requested_live:,} / —" if r.status == "live only"
                    else f"{r.contracts_live:,} / {r.requested_live:,} / {r.contracts_bt:,}")
            pl = "—" if r.pnl_live != r.pnl_live else f"{r.pnl_live:+,.2f}"
            pb = "—" if r.pnl_bt != r.pnl_bt else f"{r.pnl_bt:+,.2f}"
            set_row(self.cmp_table, i, [r.window, r.status, sides, quote, paid, size, pl, pb, r.outcome_live or "—"], {1: colors.get(r.status)})

    def _history_ready(self, got, err) -> None:
        if err:
            self.log(f"trade ledger check problem: {err}")
        elif got:
            self.log(f"trade ledger: settled {len(got)} position(s) an earlier session had left open: " + ", ".join(f"{g['market']} {g['outcome']} {g['pnl']:+.2f}" for g in got))
        self._update_perf()

    def _update_perf(self) -> None:
        """Show the trades chosen in 'Trades shown' (this session / all recorded trades of this mode / both modes) through the SAME evaluation and display as the backtest."""
        idx = self.scope.currentIndex()
        if idx == 0:
            self.hist_lbl.setText(f"Showing this session only. Earlier sessions are kept in the trade ledger ({self.ledger.path}) - choose 'All recorded trades' to see them." + self._exchange_note)
            self._update_perf_session()
            return
        mode = None if idx == 2 else ("live" if self.mode.currentIndex() == 1 else "paper")
        try:
            st = self.ledger.load(mode=mode)
        except Exception as e:                                       # noqa: BLE001
            self.hist_lbl.setText(f"⚠ could not read the trade ledger ({e}); showing this session only")
            self._update_perf_session()
            return
        t, cap0 = pooled_frame(st.settled)
        what = "all recorded trades (paper + LIVE)" if mode is None else f"all recorded {'LIVE' if mode == 'live' else 'paper'} trades"
        parts = [f"Showing {what}: {len(st.settled)} settled"]
        if st.open:
            parts.append(f"{len(st.open)} open (not settled yet)")
        rec = sum(1 for r in st.settled if r.get("recovered"))
        if rec:
            parts.append(f"{rec} settled after a restart")
        if st.legacy:
            parts.append(f"{st.legacy} imported from older session files")
        if st.bad_lines:
            parts.append(f"⚠ {st.bad_lines} unreadable ledger line(s) skipped")
        self.hist_lbl.setText(" · ".join(parts) + f". File: {self.ledger.path}." + (f" Equity = first recorded bankroll ${cap0:,.0f} + cumulative P&L of these trades." if len(t) else "") + self._exchange_note)
        now = time.time()
        if t.empty:
            self.perf.show_result(t, pd.DataFrame({"ts": [int(now)], "equity": [self.capital.value()]}), pd.Series([0.0]), dict(trades=0), self.capital.value(),
                                  "No recorded trades yet\n(every trade a session places - paper or LIVE - is saved and shown here after a restart)")
            return
        eq, dd, m = evaluate_trades(t, cap0, int(t.entry_ts.min()) - 900, int(max(now, float(t.settle_ts.max()))))
        self.perf.show_result(t, eq, dd, m, cap0, "")

    def _update_perf_session(self) -> None:
        """This session's settled trades only (the session starts from its own capital)."""
        cap = self._session_capital or self.capital.value()
        t = trades_frame(self.live_trades)
        now = time.time()
        if t.empty:
            start = self._session_start_ts or int(now)
            self.perf.show_result(t, pd.DataFrame({"ts": [start], "equity": [cap]}), pd.Series([0.0]), dict(trades=0), cap,
                                  "No settled trades in this session yet\n(the same metrics as the Backtest tab appear after the first trade)")
            return
        start = self._session_start_ts or int(t.entry_ts.min()) - 900
        end = int(max(now, float(t.settle_ts.max())))
        eq, dd, m = evaluate_trades(t, cap, int(start), end)
        self.perf.show_result(t, eq, dd, m, cap, "")

    def params_state(self):
        """(live config, mode label, frozen?) for the 'Parameters in use' table: the running session's frozen config if one is running, else what Start would use now."""
        running = self.worker is not None and self.worker.isRunning() and self.engine is not None
        cfg = self.engine.cfg if running else self.make_config(self.mode.currentIndex() == 1)
        label = ("LIVE — real orders" if cfg.mode == "live" else "Paper — simulated fills, no orders") + (" — running session" if running else " — if started now")
        return cfg, label, running

    def _refresh_using(self) -> None:
        try:
            cfg, label, running = self.params_state()
            p = cfg.params
            self.using_lbl.setText(f"{'THIS RUNNING SESSION uses' if running else 'A session started now would use'}: {label} | {self.sp.preset.currentText()} | {p.describe()} | shared left-panel "
                                   f"settings ({sizing_text(cfg)}, assumed slippage {cfg.paper_slippage * 100:.1f}c, daily loss {cfg.daily_loss_limit * 100:.0f}% / "
                                   f"drawdown stop {cfg.max_drawdown_stop * 100:.0f}%) + live-only: capital ${cfg.capital:,.0f}, order limit ask +{cfg.max_slippage * 100:.1f}c, "
                                   f"p source {'MANUAL ' + format(cfg.manual_p * 100, '.0f') + '%' if cfg.prob_source == 'manual' else 'history'}. Full list: tab \"Parameters in use\".")
        except ValueError as e:
            self.using_lbl.setText(f"Settings problem: {e}")

    def on_status(self, s: dict) -> None:
        if s.get("auto"):
            self.auto_lbl.setText(fmt_auto(s["auto"], s["equity"]))
            self.auto_lbl.show()
        self.kpis["Win probability (hist.)"].set("—" if not np.isfinite(s["p"]) else f"{s['p'] * 100:.1f}%  (n={s['n'] if s['n'] < 10**8 else 'manual'})")
        ni = s["next_in"]
        self.kpis["Next decision in"].set("—" if ni is None else f"{int(ni // 3600)}h {int(ni % 3600 // 60):02d}m {int(ni % 60):02d}s" if ni >= 3600 else f"{int(ni // 60)}m {int(ni % 60):02d}s")
        self.kpis["Open positions"].set(str(s["open_positions"]))
        if not self.live_trades:                                  # no settled trade yet: the performance cards are empty, so show the current equity
            self.perf.kpis["Final equity"].set(f"${s['equity']:,.0f}")
        if not s["halted"]:
            extra = "  | PAUSED (daily loss limit)" if s["paused"] else ""
            pos = f"  | {s['open_positions']} open position" if s["open_positions"] else ""
            self.state.setText(f"Running ({s['mode']}){extra}{pos}")
            self.state.setStyleSheet("")


HELP_HTML = """
<h2>How the strategy works</h2>
<p><b>The market.</b> Every 15 minutes Kalshi opens a BTC 'Up or Down' contract: it pays $1 if the BRTI Bitcoin index (60-second average)
ends the window at or above where it started, else $0. The contract price is the market's probability (a 25c YES = 25% chance of 'up').</p>
<h2>The rule (per 15-minute window)</h2>
<ol>
<li><b>When:</b> windows that open in the selected UTC hours (the core is 22-23 = 6-8 pm Eastern; see R2+ below), and only at minute 12 of the window (then minute 13 if nothing qualified).
By then ~80% of the window is over, the price has moved away from 50/50, and one side is the <i>underdog</i> (the side priced below 50%).</li>
<li><b>What:</b> buy the underdog if its ask is inside the price band (default 15-35c). If YES is below 50% buy YES, otherwise buy NO. Hold to settlement; no stop, no exit.</li>
<li><b>Why it can work:</b> in these evening hours the market has been overpricing the late favourite, so the underdog has paid more often than its price implies
(it wins ~40% of the time at an average ~25c price). This is an observed anomaly, not a law: it appeared in late August 2026, it is fading, and it does not exist in other hours.</li>
<li>At most <b>one trade per window</b>. One venue: the signal is read from Kalshi's live <b>order book</b> and the order goes to the same market on Kalshi as an immediate-or-cancel limit order that can never pay more than
the quoted ask + the live-only 'order limit' setting (it never rests on the book). The bet size itself is calculated with the shared 'assumed slippage', exactly as in the backtest, and is cut to what the book can fill inside the limit.</li>
</ol>
<h2>Why 22-23 UTC and minutes 12-13? (tested, with caveats)</h2>
<p>A scan of every hour and decision minute (Jun 1 - Oct 10, underdog 15-35c, after fees and 2c slippage, gate-independent):</p>
<ul>
<li><b>Hours:</b> 17 of the 24 hours have a zero or negative edge; none is significant on its own. 22-23 UTC ranks <b>#1 of the 24 two-hour windows</b> (+6.7c per contract, t = 2.5, 303 signals);
the next best (23-00, 21-22) are +2.8c / +2.4c. Widening dilutes it: 4 hours 20-23 +2.8c, 6 hours 22-03 +2.2c, all other hours -3c.</li>
<li><b>Minutes:</b> inside 22-23 the edge is concentrated at <b>minute 12 (+8.1c) and 13 (+6.0c)</b>; minute 11 is about 0, minute 14 is -9c, minutes 2-10 are -2 to -5c. In every other hour all minutes are negative.</li>
<li><b>Chance check:</b> shuffling the hour labels, a best two-hour window this good turns up ~1.3% of the time.</li>
</ul>
<p><b>Caveats:</b> the hours/minutes/price band were found by searching this same data (so the real chance of luck is higher than 1.3%); no mechanism explains them; <b>in the 8 months before late August (Dec 15 - Aug 23, 510 signals) it earned -3.8c per contract, i.e. no edge - all the profit comes from the last ~6 weeks;</b>
neighbouring hours (00-01) were strong in the burst but negative since; and the pattern did not appear in 8 other crypto series. Treat the window as a hypothesis that forward results must confirm.
Other contract lengths (hourly, daily, 5-minute) were not tested - no usable history exists.</p>
<h2>The default design: R2+ = a core plus a skeptical scout</h2>
<p>The rule above is the same in every preset; what differs is <i>which hours the engine is allowed to act on</i>. <b>R2+</b> (default) treats <b>22-23 UTC as its core</b> - the hours where the edge was found, so they get a lighter
shrinkage (core k=10) and the engine acts on them within days - but it also scouts <b>every other hour</b> with a very skeptical shrinkage (k=100): another hour trades only if its own evidence is strong. Why:
the evidence for the 22-23 concentration is the weakest part of the story (one regime, hours found after the fact, no earlier episode in 2.7 years of BTC data), so hard-wiring it is overconfident, while a fully hour-agnostic engine has to
discover the hot hour from scratch (+91..+157% on Jun-Oct vs +314% for R2+, deeper drawdowns). R2+ costs ~10% of the hard-wired preset's return in the observed regime and, in a simulated 60-day future where the hot hour moves to 01-02 UTC,
it ends roughly flat to slightly positive (+7% median) where the hard-wired preset loses (-12%). <b>R2</b> (22-23 only) is available as its own preset. The Regime radar tab shows what the scout currently sees.</p>
<p><b>How fast does R2+ follow a hot hour that moves? (default since 2026-10-12)</b> One hour yields only ~1.4 signals a day, so evidence for a new hot hour accumulates slowly; making the scout more eager everywhere costs real money in quiet hours
(Jun-Oct real history: scout k 100 / 60 / 40 / 30 returned +217% / +182% / +145% / +80%). So only the hours <b>right next to the core</b> (within 2 hours: 20, 21, 0, 1 for core 22-23) get a lighter shrinkage (k = 50); everything farther keeps k = 100.
In 60-day simulations where the hot hour moved by 2 hours, the first trade in the new hour came after about 10 days instead of 25 and the P&amp;L from it was about 4x larger (+187% vs +93% median return; +12% vs -10% for a move the other way); a move of 6 hours is still slow (first trade ~16 days, found in 7 of 8 paths).
Cost on the real Jun-Oct history, where the hot hour did not move: +202% instead of +217%, drawdown -16% instead of -12%. No change in how fast it enters a regime in 22-23 itself or in quiet periods. The fields <i>Near-core shrinkage k</i> and <i>Near-core width</i> in the left panel set this (0 = off; 40 is faster and costs more, 60 is slower and cheaper).
It trades a hot hour only while the app is running during that hour: run it around the clock if you want it to act on a moved hour.</p>
<p><b>Drawdown throttle (on by default).</b> Sizing is scaled down as the equity falls below its peak: full size to -8%, half size to -15%, quarter beyond (the -25% hard stop is unchanged). It scales <b>both the Kelly multiplier and the per-trade stake cap</b>. (Until 2026-10-10 it scaled only the multiplier, which did nothing whenever the cap was the binding limit, e.g. 1.0x Kelly / 5%: in simulated 60-day futures where the regime persists that sizing then hit the 25% drawdown stop in 80% of paths and halted after ~50 trades; with the fix it never did and the median return was several times higher.) At the default 0.25x / 2% sizing the fix changes little: median drawdown -14% to -12% when the regime persists, -17% to -15% when it ends, for about a tenth of the return when the regime persists (+476% to +421%).</p>
<h2>The probability gate - why there can be weeks with no trades</h2>
<p>The engine does not bet just because a signal appears. It keeps a record of <b>every</b> signal (traded or not) and, before each bet, asks:
<i>has this strategy historically beaten the price by more than it costs me to trade?</i></p>
<ul>
<li><b>Skeptical start:</b> the estimated win probability starts equal to the price (zero edge). It rises only if past signals won more often than their prices implied.</li>
<li><b>Price-adjusted:</b> p = this trade's price + the historical edge (win - price paid), so a 6c contract is not credited with the win rate of a 30c one.</li>
<li><b>Hurdle:</b> to bet, the learned edge must exceed the costs: slippage (2c) + fee (~1.2c) + minimum edge (4c), i.e. about <b>7c per contract</b>.
If it does not, Kelly says 'bet nothing' and the signal is only recorded.</li>
<li><b>Why a 4c conviction hurdle (was 0.5c):</b> a noisy learner always sees some positive-looking stretches. Replaying the engine on 31 months without the live regime (23 months reconstructed from BTC spot + 8 months of real quotes), the 0.5c hurdle made 127 probing trades and lost 20%; 4c made none. In the live regime the capture is unchanged (+247% vs +200% on the 10 months, max drawdown -11% vs -23%); it starts betting about a week later (Sep 5 vs Aug 29).</li>
<li><b>Memory:</b> the edge is estimated with an <b>exponentially decaying memory</b> (default half-life 10 days: a signal's weight halves every 10 days). Compared with a hard 45-day window it started betting about a week earlier in the real history (Aug 29 instead of Sep 5) and, when a regime ends, it stops betting weeks sooner instead of living off stale evidence. Set the half-life to 0 to use the hard lookback window instead.</li>
</ul>
<p><b>Example of a quiet stretch (July 2026):</b> the 73 July signals won only 20.5% at an average 24.6c price (-4c edge), so the gate stayed shut (2 trades). Betting all of them would have
lost 56%. In August the signals turned profitable (+11c) but an all-history learner was still weighed down by July and only opened in September; with a 45-day window it opens about a week earlier (19 bets in the week of Aug 31 instead of 4).
<i>A flat equity curve means 'the strategy had no edge then', not that something is broken.</i></p>
<h2>Regime radar - why the strategy can sit out, and how it watches for the next regime</h2>
<p>A strategy is only worth trading when it beats its costs (~3.5c per contract: slippage + fee). In the 8 months before late August <b>no hour, minute or price bucket</b> did - the quotes were calibrated, the median cell lost exactly the costs - so the engine correctly stayed out
(always-on, the same signals lose about 2c per contract). The edge that appeared on Aug 24 is specific to BTC (the 8 sibling crypto series show nothing) and concentrated at 22-23 UTC.
The <b>Regime radar</b> tab scouts all 24 hours continuously (nothing is traded) and shows the shrunk estimate the engine would use per hour (with the engine's own shrinkage settings: a TRADE hour is one whose gate is open for a typical signal), so you can see if the edge fades, persists, or <b>moves to another hour</b>;
the pre-flight check reports the same and warns if your hours are cold while another hour is hot. A single hour produces ~1 signal a day, so raw hourly averages are noisy - trust the shrunk bars.</p>
<h2>Pre-flight check - the go / no-go before every session</h2>
<p>Pressing <b>Start trading</b> first runs a pre-flight check (you can also press <b>Pre-flight check</b>, and the app runs it by itself at <b>21:45 UTC</b> if it is open). It never sends an order.</p>
<ul>
<li><b>Data:</b> if the newest settled window is stale it downloads the latest Kalshi data (about 15 s) and tells you what changed. Kalshi publishes candle history ~2 h late; that is normal.</li>
<li><b>Forward test:</b> it detects whether new windows have settled since the last check and re-evaluates the pre-registered R1 SPRT (frozen 2026-10-07) on the fresh data: ACCEPT / CONTINUE / REJECT.</li>
<li><b>Is the edge there now?</b> net-of-cost edge of your own preset over the last 28 and 14 days, and whether the learner gate is open (would it bet today?).</li>
<li><b>Feed and clock:</b> Kalshi exchange status, a live quote for the current window, your PC clock vs the server.</li>
<li><b>Risk:</b> sizing vs the validated limits (0.5x Kelly, 2% cap), the preset is a tested one, the previous session did not end by a kill / halt.</li>
<li><b>Account (LIVE only):</b> Kalshi key, buying power vs your capital, clock skew, resting orders / open positions, and the live market's price-tick structure (no order is sent; the separate <i>Order-path test</i> button sends one 1-contract IOC bid at 1 cent if you want an end-to-end check).</li>
</ul>
<p><b>Verdict:</b> <b>GO</b> = everything passed. <b>CAUTION</b> = advice, not a gate: live runs with exactly the settings you entered; while the forward test has not accepted the edge the dialog merely <i>suggests</i> a 0.5% per-trade cap and offers it as an optional alternative button. <b>NO-GO</b> = live is blocked (forward test rejected, trailing edge negative, feed or account problem...); paper is always allowed.
Every report is saved in data/live/preflight_*.txt.</p>
<h2>Kelly sizing</h2>
<p>For a contract costing c (price + slippage + fee) that pays $1 with probability p, the Kelly stake is <b>f* = (p - c) / (1 - c)</b> of your bankroll. You bet
<b>multiplier x f*</b> (default 0.25 = quarter Kelly), capped at 2% of bankroll, as whole contracts.<br>
Example, $10,000 bankroll: underdog at 22c, 2c slippage, fee 1.3c, so c = 25.3c; learned edge +9c, so p = 31%. f* = (0.31 - 0.253)/(1 - 0.253) = 7.7%; x 0.25 = 1.9% = $192, about 760 contracts.</p>
<p><b>Tested sizing (R2, Jun 1 - Oct 10):</b> Sharpe is nearly flat (4.1-4.7) for any multiplier up to ~0.5, so sizing is a return-vs-drawdown dial, not a source of Sharpe.
Quarter Kelly with a 2% cap gave +239% with a 12% max drawdown and, in a Monte-Carlo of reshuffled trades, only a 0.4% chance of a 30% drawdown. A 3% cap roughly doubles the return
but raises that risk to ~6%; a 5% cap to ~39%. Full Kelly or no cap is not recommended: the edge estimate is noisy and the regime is fading.</p>
<h2>Auto mode - the strategy sets its own memory, Kelly, stake and size</h2>
<p>Tick <b>Auto</b> (or choose the preset <b>AUTO · Adaptive regime + equity-scaled sizing</b>) and, at <i>every trade</i>, the engine decides five things from the history it has at that moment - in the backtest, in paper and in LIVE, with the very same code
(<code>trader/adaptive.py</code>; <code>test_auto_live_engine_and_backtest_produce_identical_trades</code> checks that both place identical trades):</p>
<ol>
<li><b>Memory = regime detection.</b> Three learners of the same edge run side by side with half-lives of 7, 14 and 30 days. Their predictions are averaged with weights that follow how well each one predicted the signals that settled recently (floor 5% each).
In a stable regime the weights stay near equal; right after a change the fast learner earns the larger weight (e.g. 84 / 11 / 5% twenty days into a new regime). Nothing is chosen by past profit - that kind of selection failed out of sample.</li>
<li><b>Regime label</b> for the signal: <b>HOT</b> (fast and slow edge both clear costs + hurdle), <b>BUILDING</b> (only the fast one does: a new regime on thin evidence), <b>FADING</b> (the slow one does, the fast one has dropped), <b>COLD</b> (neither: no trade).
Sizes are scaled 1.0 / 0.5 / 0.4 / 0.</li>
<li><b>Evidence-aware Kelly:</b> p = price + edge - z x standard error (default z = 0.25), so thin evidence means a smaller bet.</li>
<li><b>Equity ladder:</b> the Kelly multiplier and stake cap come from a table keyed by the account's <i>current equity</i> - default 0.5x / 3% from $0, 0.35x / 2.5% from $25k, 0.25x / 2% from $100k, 0.15x / 1% from $500k - then are multiplied by the regime scale and the drawdown throttle. A bigger account bets a smaller fraction.
Edit the ladder as text: <code>0:1.0/5, 25000:0.5/3, 100000:0.35/2, 500000:0.2/1</code> is a more aggressive one.</li>
<li><b>Liquidity at trade time:</b> max contracts = 10% of the contracts traded in the last completed minute (Kalshi candle volume, typically 150-200k at 22-23 UTC), capped by a hard ceiling (default 200,000). If the live volume read fails, the engine uses the median of recent readings or a small default - never 'unlimited'.
The backtest has volume from June 2026 on; earlier days only have the ceiling.</li>
</ol>
<p><b>What the tests say - honest version.</b> Auto is a different risk profile, <b>not more alpha</b>. Real history Jun 1 - Oct 10, same rules and safety limits: fixed 0.25x/2% +223% (Sharpe 4.6, max DD -11.7%); fixed 0.5x/3% +352% (4.4, -16.1%); <b>Auto default +212% (4.1, -11.6%)</b>;
Auto with the aggressive ladder +337% (3.8, -16.8%); your fixed 1.0x/5%/200k +560% (3.8, -19.7%) but with orders up to ~19,800 contracts and no liquidity cap. In 60-day simulated futures (10 bootstrap paths per future; median return / median max drawdown):</p>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Future</th><th>fixed 0.25x/2%</th><th>fixed 0.5x/3%</th><th>AUTO default (starts 0.5x/3%)</th><th>fixed 1.0x/5%/200k</th><th>AUTO aggressive ladder</th></tr>
<tr><td>Regime persists</td><td>+421% / -12%</td><td>+994% / -17%</td><td>+504% / -15%</td><td>+2,654% / -21%</td><td>+834% / -17%</td></tr>
<tr><td>Fades (half the days quiet)</td><td>+63% / -15%</td><td>+75% / -18%</td><td>+14% / -16%</td><td>+107% / -22%</td><td>-7% / -21%</td></tr>
<tr><td>Ends at day 20</td><td>+56% / -15%</td><td>+90% / -19%</td><td>+61% / -16%</td><td>+116% / -25%</td><td>+95% / -19%</td></tr>
<tr><td>Quiet (no regime)</td><td>-14% / -16%</td><td>-16% / -19%</td><td>-12% / -13%</td><td>-16% / -24%</td><td>-14% / -16%</td></tr>
<tr><td>Hot hour moves to 19-20 UTC</td><td>+12% / -18%</td><td>+12% / -21%</td><td>+14% / -15%</td><td>+4% / -25%</td><td>+17% / -20%</td></tr>
<tr><td>Paths halted by the 25% stop</td><td>0%</td><td>0-10%</td><td>0%</td><td>0-50% (50% when the regime ends / moves, 40% when quiet)</td><td>0-10%</td></tr>
</table>
<ul>
<li><b>What Auto buys you:</b> smaller positions as the account grows (the ladder), the least bleed when there is no regime (-12% vs -16% at the same starting size), it stops trading ~9 days sooner after a regime ends (last trade day 24 vs 33 after the end) and loses less after the end
(-7% vs -11% at the same starting size), and it can never size beyond what the market trades. <b>What it costs:</b> about half the upside while the regime persists; it is no better than the fixed engine when the regime fades or the hot hour moves (a single hot hour among 24 cannot be told apart from noise within weeks).</li>
<li>The internals are dials, not free lunches: a single 14-day memory earns more while the regime persists (+744%) but bleeds more when quiet (-16%); turning regime scaling off does the same (+669% persists, quiet -12.8%). z = 0 trades more (+585% / quiet -14%), z = 0.5 trades less (+360% / quiet -10%).
Defaults sit at the conservative end because the regime has weakened since its first weeks (net edge after costs: last 28 days +13c, last 14 days +7c per contract, against about +30c at the start).</li>
<li>Only one regime exists in the data, 10 paths per future is noisy (differences of 10-20 points between neighbouring cells are not reliable), and the fixed-sizing columns assume 2c slippage at any size, which gets optimistic above a few thousand contracts.</li>
</ul>
<p>With Auto on, the Kelly multiplier, Max stake, Max contracts, probability lookback and memory half-life boxes are greyed out (Auto sets them); min edge, the probability basis / hour model, costs, drawdown throttle and the risk limits still apply.
The Live tab shows an <b>Auto status</b> strip (regime, effective Kelly multiplier, stake cap, learner weights, hours with an edge now) and every trade records its regime, Kelly multiplier and cap (trades table and chart tooltip).
Auto needs the learned win probability: with a fixed / hindsight probability (Backtest) or a manual probability (Live) it is ignored and the fixed boxes apply - the tab 'Parameters in use' shows which.</p>
<h2>Backtest and Live use the same engine - what is shared and what is not</h2>
<p>The <b>left panel is shared</b>: strategy rules, probability model, memory, Kelly sizing, drawdown throttle, fees, assumed slippage, and the <b>risk limits (daily loss pause, drawdown stop)</b> drive the Backtest, paper mode and live trading alike.
The backtest applies the live engine's risk rules and minimum-history rule window by window; a test (<code>test_live_engine_and_backtest_produce_identical_trades</code>) replays the same data through both and checks that they place the same trades with the same size, price, P&amp;L and equity.</p>
<p>Both tabs show the <b>same performance view</b> (same KPI cards, same Metrics table, same equity curve, trades, monthly and by-hour tables), computed by the same function from the trades; the Live tab simply feeds it the session's settled trades.</p>
<p>Only what the world forces to differ differs: the Backtest has its own <b>date range, capital, probability mode (history / fixed / hindsight) and compounding</b> box; the Live tab has its own <b>capital, mode (paper / LIVE), order limit price and manual-probability</b> box.
Real fills can be better or worse than the assumed slippage - the one thing a backtest cannot know. The tab <b>Parameters in use</b> (in both tabs) lists every parameter with the Backtest value next to the Live value, so you can see exactly what each side uses.</p>
<h2>Kalshi: signal and execution on ONE venue</h2>
<p>The strategy was found, tested and is learned on Kalshi's prices, so Kalshi is also where it trades: no cross-venue price gap, no stale second feed, and the backtest's candle data and the live order book are the same market.</p>
<ul>
<li><b>Account:</b> create an API key on Kalshi (Account &rarr; API keys; the demo environment at demo.kalshi.co has its own account and key). Paste the <b>Key ID</b>, choose the downloaded <b>private key (.pem)</b>, pick <b>Demo</b> or <b>Production</b>, press <b>Connect and test</b>. The key stays in memory only; the Key ID and file path are remembered.</li>
<li><b>Orders:</b> every order is immediate-or-cancel at <i>ask + order limit</i>; a NO is bought as a sell of YES at 1 - price (Kalshi's single-book model). Fills, prices and fees are read back from your account's fills and checked: wrong side or a price above your limit stops trading instead of booking it.</li>
<li><b>Execution quality:</b> the order is checked against the live order-book depth first (it is cut to what is available inside the limit, and skipped if nothing is); the quote is read right at the minute mark; the book snapshot and expected average price are saved with every trade.</li>
<li><b>Backtest = live:</b> both run the same rules and the same sizing code. The Backtest tab's <b>Verify live = backtest</b> button (and <code>python replay_live.py</code>) replays the real live engine over the period you chose and checks it trades exactly what the backtest trades. The Live tab's <b>Compare with backtest</b> re-runs the backtest on each recorded session (same windows, the settings it traded with) and shows where the real market differed: the quote at that second, the real fill price and size, windows one side skipped.</li>
<li><b>Cash and exchange shards:</b> Kalshi splits its markets over several matching engines (&quot;exchange shards&quot;; BTC 15-minute is on shard 2, Crypto &amp; Commodities) and an API order is funded only by the cash held ON that shard - the Kalshi app moves cash there for you, the API does not. <b>Auto-fund</b> (Live-only settings, on by default) does what the app does: at session start, once early in every window and again if an order is rejected for cash, it moves cash from your OTHER shards onto the trading shard (never out of your account), up to the amount you set. The account box and the pre-flight show your cash per shard. <i>(The transfer call was built from Kalshi's documentation and tested against a mock exchange only: check it on a small amount first.)</i></li>
<li><b>Safety:</b> start in <b>Demo</b>; production asks you to type LIVE; a red banner shows when real money is in play.</li>
</ul>
<h2>Trade history and fresh data</h2>
<p><b>Your trades are kept.</b> Every trade a session places - paper or LIVE - is written to the <b>trade ledger</b> (<code>data/live/ledger.jsonl</code>) the moment it fills and again when it settles. The Live tab's <b>Trades shown</b> box chooses what the table, equity curve and metrics display:
this session only, all recorded trades of the selected mode (Paper or LIVE), or both modes. After a restart or a reconnect the history is simply there. Details:</p>
<ul>
<li>A position that is still open when you close the app is not lost: at the next start (and on <b>Connect and test</b> / <b>Reload history</b>) the app asks Kalshi for the result of every window an earlier session left open and settles it (marked "settled after a restart").</li>
<li>Older per-session files (<code>session_*.csv</code>) are read too, so trades from before the ledger existed show up as well.</li>
<li>The pooled equity curve is the first recorded bankroll plus the cumulative P&amp;L of the recorded trades (the ledger cannot know about deposits or withdrawals); compare it with your real account balance.</li>
<li>Each LIVE trade also stores the raw fill price the exchange reported and your account's buying power right after it settled, so fill prices can be audited. On connect, the app reads your account's recent fills (read-only) and warns if some orders in the BTC 15-minute markets are not in the ledger. Trades from the retired Polymarket venue are not shown (their files are kept in the archive_polymarket folder). <i>(The Kalshi order and fill reads were built from Kalshi's documentation and tested against a signature-verifying mock exchange; try the DEMO environment first.)</i></li>
</ul>
<p><b>The backtest always uses the newest data.</b> The app downloads newly settled Kalshi windows in the background when it starts, and the Backtest tab does it again before each run when the local data is more than 20 minutes behind (tick <i>Refresh data first</i>; about 10-20 seconds). The newest evening's windows used to be missing for up to two hours because the bulk download lags;
a per-market download now fills them in within a couple of minutes of the close. The <b>To</b> date follows the newest data automatically until you set it yourself (then a warning says newer data is excluded; <b>→ latest</b> re-attaches it), and a line under the controls shows how old the data is.</p>
<h2>Using the equity chart and the pages</h2>
<p><b>Equity chart</b> (Backtest and Live): <b>mouse wheel</b> zooms in / out around the cursor (the vertical axis refits to what is visible; <b>Shift+wheel</b> zooms the vertical axis only), <b>drag</b> pans, <b>double-click</b> resets, and the buttons above the chart do the same.
<b>Hover a dot</b> to see that trade (entry and settle time, market, side, quoted vs paid price, contracts, stake, fee, P&amp;L, p used, Kelly, equity after); <b>click a dot</b> to pin the details, click empty space to unpin. Green dots are wins, red dots are losses.</p>
<p><b>Nothing is cut off:</b> every page scrolls (vertically and horizontally) when the window is too small for it, the settings panel scrolls too and its width can be changed by dragging the divider between it and the tabs, tables never shorten text with "..." (they scroll sideways, and hovering a cell shows its full text).</p>
<h2>Safety limits</h2>
<p>Daily-loss pause and drawdown halt (shared settings, defaults 10% / 25%, applied identically in Backtest and Live), the drawdown throttle, a cap by your real buying power (live), the KILL button, and the engine halts if an order's final state is ever unclear.</p>
<h2>What does not work (tested)</h2>
<p>Trading all day: Sharpe 0.6-0.75 with ~40% drawdowns (vs 3.7-4.6 for the evening presets). Buying favourites (85-97c): ~90% win rate but negative after fees. A single raw win rate for sizing:
over-bets cheap contracts. All numbers are in-sample and regime-dependent; nothing here is a guarantee.</p>
"""


class RadarTab(QWidget):
    """Where in the day is the edge today?  Scouts all 24 UTC hours with the engine's own machinery; nothing is traded."""

    def __init__(self, settings: SettingsPanel, get_rows):
        super().__init__()
        self.sp, self.get_rows = settings, get_rows
        v = QVBoxLayout(self)
        intro = QLabel("Regime radar — scouts every UTC hour with the same estimator the all-hours adaptive engine uses (underdog signals in your price band, net of slippage and fee, "
                       "decaying memory, each hour shrunk toward the all-hours estimate). Nothing is traded here. A single hour yields only ~1 signal a day, so a raw trailing average is noisy: "
                       "trust the shrunk 'engine estimate' (bars), not the dots.")
        intro.setObjectName("muted"); intro.setWordWrap(True)
        v.addWidget(intro)
        row = QHBoxLayout()
        self.btn = QPushButton("Refresh radar"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.refresh)
        self.summary = QLabel("—"); self.summary.setWordWrap(True)
        row.addWidget(self.btn); row.addWidget(self.summary, 1)
        v.addLayout(row)
        self.fig = Figure(figsize=(9, 3.4), facecolor=BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self.canvas, 3)
        self.table = make_table(["Hour (UTC)", "Active", "Signals (28 d)", "Win", "Avg price", "Net EV 28 d", "± se", "Net EV 14 d", "Engine estimate", "Status"])
        v.addWidget(self.table, 4)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.refresh()

    def config(self) -> "radar.RadarConfig":
        p = self.sp.params()
        return radar.RadarConfig(cost_min=p.cost_min, cost_max=p.cost_max, minutes=p.minutes, fee_model=self.sp.fee.currentText(), slippage=self.sp.slip.value() / 100, min_edge=self.sp.edge.value() / 100,
                                 hour_k=self.sp.hourk.value(), halflife_days=self.sp.halflife() or radar.SCOUT_HALFLIFE, core_hours=self.sp.core_hours(), core_k=self.sp.corek.value(),
                                 near_k=self.sp.near_k(), near_width=self.sp.nearw.value())

    def refresh(self) -> None:
        try:
            rows = self.get_rows()
            cfg = self.config()
            active = self.sp.active_hours()
            df = radar.hour_radar(rows, time.time(), cfg, active)
        except Exception as e:                    # noqa: BLE001
            self.summary.setText(f"radar unavailable: {e}")
            return
        state, txt = radar.summarize(df, active, cfg)
        self.summary.setText(txt)
        self.summary.setStyleSheet(f"color: {GREEN if state == 'ok' else YELLOW if state in ('moved', 'warm') else MUTED}; font-weight: 600;")
        col = {"TRADE": GREEN, "WATCH": YELLOW, "cold": MUTED}
        h = df[df.hour != "ALL"]
        x = h.hour.astype(int).values
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(PANEL)
        ax.bar(x, h.est_edge.values * 100, color=[col[s] for s in h.status], width=0.7, zorder=2, label="Engine estimate (shrunk, net of costs)")
        ax.errorbar(x, h.ev.values * 100, yerr=h.se.values * 100, fmt="o", color=ACCENT, ms=3.5, lw=0.8, alpha=0.8, zorder=3, label="Raw net EV, last 28 d (± se)")
        ax.axhline(cfg.min_edge * 100, color=YELLOW, ls="--", lw=0.9, label=f"Conviction hurdle ({cfg.min_edge * 100:.0f}c)")
        ax.axhline(0, color=MUTED, lw=0.7)
        for hh in active:
            ax.axvspan(hh - 0.5, hh + 0.5, color=ACCENT, alpha=0.10, zorder=1)
        ax.set_xticks(range(24)); ax.set_xlim(-0.6, 23.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("UTC hour of window open (shaded = hours your preset trades)", color=MUTED, fontsize=9)
        ax.set_ylabel("cents per contract", color=MUTED, fontsize=9)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(True, color=BORDER, lw=0.5, alpha=0.7, zorder=0)
        ax.legend(loc="upper left", fontsize=8, facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT)
        self.fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.17)
        self.canvas.draw_idle()
        self.table.setRowCount(len(df))
        for i, r in enumerate(df.itertuples()):
            c = col[r.status]
            set_row(self.table, i, [r.hour, "●" if r.active else "", r.n, "—" if not r.n else f"{r.win * 100:.0f}%", "—" if not r.n else f"{r.price * 100:.0f}c",
                                    "—" if not r.n else f"{r.ev * 100:+.1f}c", "—" if r.n < 2 else f"{r.se * 100:.1f}c", "—" if not r.n14 else f"{r.ev14 * 100:+.1f}c",
                                    f"{r.est_edge * 100:+.1f}c", r.status], {8: c, 9: c})


class HelpTab(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setOpenExternalLinks(False)
        self.setHtml(HELP_HTML)
        self.setStyleSheet(f"QTextBrowser {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 12px; font-size: 13px; }}")


# ── main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BTC 15-min Up/Down — Kalshi trader")
        self.resize(1480, 900)
        self.qs = app_settings()
        self._rows: Optional[pd.DataFrame] = None
        self.auto_update = os.environ.get("TRADER_NO_AUTO_UPDATE") != "1"          # tests switch it off; they exercise the refresh path with a fake updater
        self.updater = update_data
        self._refresh_busy = False
        self._after_refresh: list = []
        self.settings = SettingsPanel()
        self.settings.load(self.qs)
        self.settings.btn_update.clicked.connect(self.update_data)
        self.settings.btn_backfill.clicked.connect(self.backfill_history)
        self.backtest = BacktestTab(self.settings, self.rows)
        self.backtest.ensure_fresh = self.ensure_fresh
        self.live = LiveTab(self.settings, self.rows, self.qs)
        self.live.adopt_rows = self._adopt_rows
        self.live.ensure_fresh = self.ensure_fresh
        self.radar = RadarTab(self.settings, self.rows)
        for pv in (self.backtest.params_view, self.live.params_view):
            pv.get_state = self._params_state
        self.settings.changed.connect(lambda: (self.backtest.params_view.refresh(), self.live.params_view.refresh(), self.live._refresh_using()))

        self.settings.setMinimumWidth(400)
        side = QScrollArea(); side.setWidgetResizable(True); side.setWidget(self.settings); side.setMinimumWidth(300)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded); side.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tabs = QTabWidget()
        tabs.addTab(scrollable(self.live, 900, 780), "Live trading"); tabs.addTab(scrollable(self.backtest, 900, 780), "Backtest")
        tabs.addTab(scrollable(self.radar, 900, 640), "Regime radar"); tabs.addTab(HelpTab(), "How it works")
        self.main_tabs = tabs
        self.splitter = QSplitter(Qt.Orientation.Horizontal)                      # drag the handle to give the settings (or the tabs) more room
        self.splitter.addWidget(side); self.splitter.addWidget(tabs)
        self.splitter.setStretchFactor(0, 0); self.splitter.setStretchFactor(1, 1); self.splitter.setSizes([430, 1050]); self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)
        self.reload_data()
        self.live.refresh_preview()
        self.live._refresh_using()
        QTimer.singleShot(1500, self.ensure_fresh)                       # bring the data up to date in the background right after start
        QTimer.singleShot(2500, lambda: self.live.refresh_history(reconcile=self.auto_update))        # settle positions an earlier session left open (needs Kalshi results)

    def ensure_fresh(self, then=None) -> None:
        """Download newly settled windows if the local data is stale (see data.needs_refresh), reload, then call ``then``.  Never blocks the window: the download runs in a worker thread."""
        try:
            state = preflight.load_state()
            need = self.auto_update and needs_refresh(self.rows(), time.time(), float(state.get("data_update_ts", 0.0)))
        except Exception:                                                # noqa: BLE001
            need = False
        if not need and not self._refresh_busy:
            if then:
                then()
            return
        if then:
            self._after_refresh.append(then)
        if self._refresh_busy:
            return
        self._refresh_busy = True
        try:
            state = preflight.load_state()
            state["data_update_ts"] = time.time()
            preflight.save_state(state)
        except Exception:                                                # noqa: BLE001
            pass
        self.live.log("refreshing local data from Kalshi (newly settled windows) …")
        self.statusBar().showMessage("Refreshing data from Kalshi …")
        w = FnWorker(lambda log: self.updater(lambda s: None))
        self._rw = w
        w.done.connect(self._refreshed)
        w.start()

    def _refreshed(self, ok, err) -> None:
        self._refresh_busy = False
        self.live.log("data refreshed" if not err and ok else f"data refresh problem ({err or 'a collector step failed'}) - using what is on disk")
        self.reload_data()
        self.live.refresh_preview()
        cbs, self._after_refresh = self._after_refresh, []
        for cb in cbs:
            cb()

    def rows(self) -> pd.DataFrame:
        if self._rows is None:
            self._rows = load_rows()
        return self._rows

    def _params_state(self):
        bt, bt_err = self.backtest.safe_config()
        lv, label, frozen = self.live.params_state()
        return bt, bt_err, lv, label, frozen, self.settings.preset.currentText()

    def _adopt_rows(self, rows: pd.DataFrame) -> None:
        """Data refreshed by the pre-flight check: use it without re-reading the files."""
        self.reload_data(rows)

    def reload_data(self, rows: Optional[pd.DataFrame] = None) -> None:
        self._rows = rows
        try:
            rows = self.rows()
            lo, hi = data_range(rows)
            self.backtest.set_range(lo, hi)
            sig = extract_signals(rows, self.settings.params())
            self.settings.data_lbl.setText(f"Local data: {lo:%Y-%m-%d} → {hi:%Y-%m-%d %H:%M} UTC · {len(rows):,} decision rows · {len(sig)} strategy signals")
            self.statusBar().showMessage(f"Data through {hi:%Y-%m-%d %H:%M} UTC")
        except Exception as e:                    # noqa: BLE001
            self.settings.data_lbl.setText(f"⚠ could not load local data: {e}")

    def update_data(self) -> None:
        self.settings.btn_update.setEnabled(False)
        self.live.log("updating local data from Kalshi (public API)…")
        w = FnWorker(lambda log: update_data(log))
        self._uw = w
        w.log.connect(lambda s: self.live.log("  " + s))
        w.done.connect(self._updated)
        w.start()

    def backfill_history(self) -> None:
        try:
            hours = self.settings.params().hours
        except ValueError as e:
            QMessageBox.warning(self, "Download older history", str(e))
            return
        n = backfill_plan(hours)
        if n == 0:
            QMessageBox.information(self, "Download older history", "All markets for these hours are already downloaded (Kalshi's series starts on 2025-12-15).")
            return
        mins = n / 1.5 / 60
        if QMessageBox.question(self, "Download older history", f"This will download about {n:,} markets for UTC hours {self.settings.hours.text()} back to 2025-12-15 "
                                f"(roughly {mins:.0f} minutes, public API, no account needed). Early months had very thin liquidity, so results there are less reliable. Continue?"
                                ) != QMessageBox.StandardButton.Yes:
            return
        self.settings.btn_backfill.setEnabled(False)
        self.live.log(f"downloading older history for hours {self.settings.hours.text()} ({n:,} markets)…")
        w = FnWorker(lambda log: backfill_history(hours, log))
        self._bw = w
        w.log.connect(lambda s: self.live.log("  " + s) if ("/" in s or "done" in s.lower() or "markets" in s) else None)
        w.done.connect(self._backfilled)
        w.start()

    def _backfilled(self, ok, err) -> None:
        self.settings.btn_backfill.setEnabled(True)
        self.live.log("older history download finished" if not err and ok else f"history download problem: {err or 'the downloader reported an error'}")
        self.reload_data()
        self.live.refresh_preview()

    def _updated(self, ok, err) -> None:
        self.settings.btn_update.setEnabled(True)
        self.live.log("data update finished" if not err and ok else f"data update problem: {err or 'a collector step failed'}")
        self.reload_data()
        self.live.refresh_preview()

    def closeEvent(self, e) -> None:
        if self.live.worker and self.live.worker.isRunning():
            if QMessageBox.question(self, "Quit", "A trading session is running. Stop it and quit?") != QMessageBox.StandardButton.Yes:
                e.ignore(); return
            self.live._end_session("window closed")
        self.settings.save(self.qs)
        e.accept()
