"""Charts: the interactive equity curve (Backtest + Live) and the price / Supertrend chart (Signal monitor). Dark themed, matplotlib on Qt.

Equity chart:  wheel = zoom around the cursor (Shift+wheel = value axis only) · drag = pan · double-click = reset · hover a dot = trade details · click a dot = pin them."""
from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, FuncFormatter, Locator, LogLocator, NullFormatter, NullLocator
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .theme import ACCENT, BG, BORDER, COL_LONG, COL_SHORT, GREEN, MUTED, PANEL, PANEL2, RED, TEXT, QComboBox, big_money


def fmt_dollar(y: float, _pos=None) -> str:
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(y) >= div:
            return f"${y / div:,.2f}{suf}".replace(".00", "")
    return f"${y:,.0f}"


class PercentLocator(Locator):
    """Ticks at round PERCENT values (0, 100%, 1K%, ...) on a dollar axis. A log dollar axis ticks on decades, which in percent reads 0% / 900% / 9.9K% - correct but unreadable."""

    def __init__(self, chart: "EquityChart"):
        self.chart = chart

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        base = self.chart._capital or 1.0
        lo, hi = sorted(((vmin / base - 1) * 100.0, (vmax / base - 1) * 100.0))
        top = max(abs(lo), abs(hi))
        if top <= 0:
            return [base]
        pcts = [0.0] if lo <= 0 <= hi else []
        lo_e = max(-2, int(np.floor(np.log10(top))) - 5)                      # cap the ladder: a 2.4M% range would otherwise drag in a pile of 0.1% ticks near the base
        exps = [e for e in range(lo_e, 11) if 10 ** e <= top * 10]
        mants = (1,) if len(exps) > 5 else (1, 2, 5)
        for e in exps:
            for m in mants:
                for p in (m * 10 ** e, -m * 10 ** e):
                    if lo <= p <= hi:
                        pcts.append(float(p))
        log = getattr(self.axis.axes, "get_yscale", lambda: "linear")() == "log"

        def pos(p):
            y = base * (1 + p / 100.0)
            if log:
                if y <= 0 or vmin <= 0 or vmax <= 0 or vmax == vmin:
                    return None
                return (np.log10(y) - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
            return None if vmax == vmin else (y - vmin) / (vmax - vmin)

        kept = []                                                              # drop ticks that would overlap: keep 0% first, then the largest moves
        for p in sorted(set(pcts), key=lambda v: (v != 0.0, -abs(v))):
            q = pos(p)
            if q is None or all(abs(q - pos(k)) >= 0.045 for k in kept if pos(k) is not None):
                kept.append(p)
        return [base * (1 + p / 100.0) for p in sorted(kept)]


class EquityChart(FigureCanvasQTAgg):
    """Equity curve (top, optional buy & hold) + drawdown (bottom), with the trades as dots."""
    HIT_PX = 10
    readout = pyqtSignal(str)                 # crosshair text: date, equity, return, drawdown
    trade_picked = pyqtSignal(int)            # a dot was clicked (index into the trade list), -1 = unpinned

    def __init__(self):
        self.fig = Figure(figsize=(8, 5), facecolor=BG)
        super().__init__(self.fig)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.07, left=0.085, right=0.985, top=0.96, bottom=0.09)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log, self.pct_axis, self.show_bh, self.show_dots = True, False, True, True
        self._reset_state()
        self._style()
        self.message("Run a backtest to see the equity curve")
        for name, fn in (("scroll_event", self._on_scroll), ("button_press_event", self._on_press), ("button_release_event", self._on_release),
                         ("motion_notify_event", self._on_motion), ("axes_leave_event", self._on_leave)):
            self.mpl_connect(name, fn)

    # ── state ──
    def _reset_state(self) -> None:
        self._eq = None; self._bh = None
        self._ex = np.array([]); self._ey = np.array([]); self._by = np.array([])
        self._tx = np.array([]); self._ty = np.array([]); self._trades = pd.DataFrame()
        self._capital = 0.0
        self._home = None
        self._ann = None; self._hot = None; self._hover = None; self._pinned = None
        self._press = None; self._moved = False
        self._marks: list = []
        self._dots_art: list = []; self._bh_art = None

    def _style(self) -> None:
        for a in (self.ax, self.ax2):
            a.set_facecolor(PANEL)
            a.tick_params(colors=MUTED, labelsize=9)
            for sp in a.spines.values():
                sp.set_color(BORDER)
            a.grid(True, color=BORDER, lw=0.5, alpha=0.7)
        for l in self.ax.get_xticklabels():
            l.set_visible(False)

    def message(self, text: str) -> None:
        self._reset_state()
        self.ax.clear(); self.ax2.clear(); self._style()
        self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes, ha="center", va="center", color=MUTED, fontsize=11)
        for a in (self.ax, self.ax2):
            a.set_xticks([]); a.set_yticks([])
        self.draw_idle()

    # ── plot ──
    def plot(self, eq: pd.Series, bh: Optional[pd.Series], trades: pd.DataFrame, capital: float, step: bool = False, bh_label: str = "Buy & hold") -> None:
        self._reset_state()
        self.ax.clear(); self.ax2.clear(); self._style()
        self._eq, self._bh, self._capital = eq, bh, float(capital)
        self._ex, self._ey = mdates.date2num(eq.index.values), eq.to_numpy(float)
        ds = "steps-post" if step else "default"
        self.ax.plot(eq.index, self._ey, color=ACCENT, lw=1.3, drawstyle=ds, label="Strategy", zorder=4)
        if bh is not None and len(bh):
            self._by = bh.reindex(eq.index).ffill().to_numpy(float)
            (self._bh_art,) = self.ax.plot(eq.index, self._by, color=MUTED, lw=1.0, alpha=0.85, label=bh_label, zorder=3)
            self._bh_art.set_visible(self.show_bh)
        self.ax.axhline(capital, color=MUTED, ls="--", lw=0.7, zorder=2)
        dd = (eq / eq.cummax() - 1.0) * 100.0
        self.ax2.fill_between(eq.index, dd.values, 0, step="post" if step else None, color=RED, alpha=0.55, lw=0)
        if bh is not None and len(bh):
            self.ax2.plot(eq.index, ((bh / bh.cummax() - 1.0) * 100.0).values, color=MUTED, lw=0.7, alpha=0.7)
        self.ax2.set_ylabel("Drawdown %", color=MUTED, fontsize=9)
        if trades is not None and len(trades):
            t = trades.reset_index(drop=True)
            tx = mdates.date2num(pd.to_datetime(t["exit_time"]).values)
            ty = t["equity_after"].to_numpy(float)
            win = (t["pnl_usd"].to_numpy(float) > 0)
            s = 14 if len(t) < 150 else 9 if len(t) < 500 else 5 if len(t) < 2000 else 3        # a thousand big dots would hide the curve they sit on
            self._dots_art = [self.ax.scatter(tx[win], ty[win], s=s, color=GREEN, zorder=5, label="Winning trade", linewidths=0, alpha=0.9),
                              self.ax.scatter(tx[~win], ty[~win], s=s, color=RED, zorder=5, label="Losing trade", linewidths=0, alpha=0.9)]
            for a in self._dots_art:
                a.set_visible(self.show_dots)
            self._tx, self._ty, self._trades = tx, ty, t
        self.ax.legend(loc="upper left", fontsize=8, facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT)
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
        self.apply_axes(redraw=False)
        self.ax.relim(); self.ax.autoscale_view()
        self.draw()                                                          # a real draw first, so the "home" view is the one the user actually sees
        self._home = (self.ax.get_xlim(), self.ax.get_ylim(), self.ax2.get_ylim())

    # ── axes: $ or %, linear or log ──
    def set_scale(self, pct_axis: Optional[bool] = None, log: Optional[bool] = None) -> None:
        if pct_axis is not None:
            self.pct_axis = pct_axis
        if log is not None:
            self.log = log
        if self._eq is not None:
            self.apply_axes()
            self._fit_y()
            self.draw_idle()

    def set_overlays(self, show_bh: Optional[bool] = None, show_dots: Optional[bool] = None) -> None:
        if show_bh is not None:
            self.show_bh = show_bh
            if self._bh_art is not None:
                self._bh_art.set_visible(show_bh)
        if show_dots is not None:
            self.show_dots = show_dots
            for a in self._dots_art:
                a.set_visible(show_dots)
            if not show_dots:
                self._pinned = None; self._show_trade(None)
        self.draw_idle()

    def apply_axes(self, redraw: bool = True) -> None:
        if self._eq is None:
            return
        ymin = float(min(np.nanmin(self._ey), np.nanmin(self._by) if len(self._by) else np.inf))
        log_ok = self.log and ymin > 0
        self.ax.set_yscale("log" if log_ok else "linear")
        self.ax.yaxis.set_major_formatter(FuncFormatter(self._fmt_pct if self.pct_axis else fmt_dollar))
        if self.pct_axis:
            self.ax.yaxis.set_major_locator(PercentLocator(self))
            self.ax.yaxis.set_minor_locator(NullLocator())
        else:
            self.ax.yaxis.set_major_locator(LogLocator() if log_ok else AutoLocator())
            self.ax.yaxis.set_minor_locator(LogLocator(subs="auto") if log_ok else NullLocator())
        if log_ok:
            self.ax.yaxis.set_minor_formatter(NullFormatter())            # else the decade minor ticks get labelled too
        self.ax.set_ylabel("Cumulative return" if self.pct_axis else "Equity", color=MUTED, fontsize=9)
        if redraw:
            self.draw_idle()

    def _fmt_pct(self, y: float, _pos=None) -> str:
        p = (y / (self._capital or 1.0) - 1.0) * 100.0
        if abs(p) >= 1e6:
            return f"{p / 1e6:,.2f}M%"
        if abs(p) >= 1e3:
            return f"{p / 1e3:,.1f}K%"
        return f"{p:,.1f}%"

    # ── view control ──
    def reset_view(self) -> None:
        if self._home is None:
            return
        self.ax.set_xlim(self._home[0]); self.ax.set_ylim(self._home[1]); self.ax2.set_ylim(self._home[2])
        self.apply_axes(redraw=False)
        self.draw_idle()

    def zoom_by(self, factor: float, x_center: Optional[float] = None, y_only: bool = False) -> None:
        """factor < 1 zooms in. The x axis is zoomed around ``x_center`` (default: the middle of the view); the equity axis is refitted to what is visible."""
        if self._home is None:
            return
        if y_only:
            lo, hi = self.ax.get_ylim()
            if self.ax.get_yscale() == "log" and lo > 0:
                a, b = np.log10(lo), np.log10(hi); mid = (a + b) / 2
                self.ax.set_ylim(10 ** (mid - (mid - a) * factor), 10 ** (mid + (b - mid) * factor))
            else:
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
        if nhi - nlo < 2e-3:                                                 # ~3 minutes: enough
            return
        self.ax.set_xlim(nlo, nhi)
        self._fit_y()
        self.draw_idle()

    def _fit_y(self) -> None:
        """Rescale the equity axis to the data in view (the TradingView behaviour)."""
        if not len(self._ex):
            return
        lo, hi = self.ax.get_xlim()
        i0 = max(int(np.searchsorted(self._ex, lo, side="right")) - 1, 0)
        i1 = max(int(np.searchsorted(self._ex, hi, side="right")), i0 + 1)
        ys = self._ey[i0:i1]
        if len(self._by) and self.show_bh:
            ys = np.r_[ys, self._by[i0:i1]]
        if len(self._ty) and self.show_dots:
            m = (self._tx >= lo) & (self._tx <= hi)
            if m.any():
                ys = np.r_[ys, self._ty[m]]
        ys = ys[np.isfinite(ys)]
        if not len(ys):
            return
        y0, y1 = float(ys.min()), float(ys.max())
        if self.ax.get_yscale() == "log" and y0 > 0:
            pad = max((np.log10(y1) - np.log10(y0)) * 0.06, 0.02)
            self.ax.set_ylim(10 ** (np.log10(y0) - pad), 10 ** (np.log10(y1) + pad))
        else:
            pad = max((y1 - y0) * 0.08, abs(y1) * 0.002, 1e-9)
            self.ax.set_ylim(y0 - pad, y1 + pad)

    def highlight(self, t0, t1, side: str) -> None:
        """Shade one trade (entry → exit) on both panels; links a row of the trades table to the curve."""
        self.clear_highlight()
        if self._eq is None or pd.isna(t1):
            return
        col = COL_LONG if str(side).lower() == "long" else COL_SHORT
        a1 = mdates.date2num(pd.Timestamp(t1))
        a0 = mdates.date2num(pd.Timestamp(t0)) if not pd.isna(t0) else a1
        for ax in (self.ax, self.ax2):
            self._marks.append(ax.axvspan(a0, a1, color=col, alpha=0.20, zorder=1, lw=0))
        self._marks.append(self.ax.axvline(a0, color=col, lw=0.9, ls="--", zorder=1))
        self._marks.append(self.ax.axvline(a1, color=col, lw=0.9, ls=":", zorder=1))
        self.draw_idle()

    def clear_highlight(self) -> None:
        for m in self._marks:
            try:
                m.remove()
            except Exception:                                                # noqa: BLE001
                pass
        self._marks = []

    def focus_trade(self, i: int) -> None:
        """Zoom to trade ``i`` and pin its details."""
        if self._home is None or not (0 <= i < len(self._trades)):
            return
        r = self._trades.iloc[i]
        x1 = self._tx[i]
        x0 = mdates.date2num(pd.Timestamp(r.entry_time)) if pd.notna(r.entry_time) else x1 - 1.0
        pad = max((x1 - x0) * 1.5, 3.0)
        self.ax.set_xlim(x0 - pad, x1 + pad)
        self._fit_y()
        self._pinned = i
        self._show_trade(i)

    # ── trade info ──
    def _nearest(self, event) -> Optional[int]:
        if not len(self._tx) or not self.show_dots or event.x is None or event.y is None:
            return None
        pts = self.ax.transData.transform(np.column_stack([self._tx, self._ty]))
        d = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] <= self.HIT_PX else None

    def trade_text(self, i: int) -> str:
        r = self._trades.iloc[i]
        et = pd.Timestamp(r.entry_time) if pd.notna(r.entry_time) else None
        xt = pd.Timestamp(r.exit_time)
        why = str(r.exit_reason) if "exit_reason" in r.index and str(r.exit_reason) not in ("", "—", "nan") else ""
        sym = f" · {r.symbol}" if "symbol" in r.index and isinstance(r.symbol, str) and r.symbol else ""
        lines = [f"Trade #{i + 1} — {'WIN' if r.pnl_usd > 0 else 'LOSS'}" + (f" · exit by {why}" if why else ""),
                 f"{str(r.side).upper()}{sym}   qty {int(r.qty):,}",
                 f"In   {et:%Y-%m-%d %H:%M}  @ {r.price:,.2f}" if et is not None else f"In   @ {r.price:,.2f}",
                 f"Out  {xt:%Y-%m-%d %H:%M}  @ {r.exit_price:,.2f}" + (f"   (held {self._hold(xt - et)})" if et is not None else ""),
                 f"Return {r.return_pct:+.2f}%   P&L {'-' if r.pnl_usd < 0 else '+'}${abs(r.pnl_usd):,.2f}",
                 f"Equity after ${r.equity_after:,.2f}"]
        return "\n".join(lines)

    @staticmethod
    def _hold(td: pd.Timedelta) -> str:
        h = td.total_seconds() / 3600.0
        return f"{h / 24:.1f} d" if h >= 48 else f"{h:.1f} h"

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
        right = x > (x0 + x1) / 2
        upper = (np.log10(y) > (np.log10(y0) + np.log10(y1)) / 2) if (self.ax.get_yscale() == "log" and y > 0 and y0 > 0) else y > (y0 + y1) / 2
        self._ann.xy = (x, y)
        self._ann.set_text(self.trade_text(i).replace("$", r"\$"))           # '$' pairs would otherwise be read as math text
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
            self.trade_picked.emit(-1)
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
                if self.ax.get_yscale() == "log":                           # a pixel is a fixed RATIO on a log axis, not a fixed difference
                    k = p["start"][1] / cur[1]
                    self.ax.set_ylim(p["ylim"][0] * k, p["ylim"][1] * k)
                else:
                    self.ax.set_ylim(p["ylim"][0] + dy, p["ylim"][1] + dy)
                self.draw_idle()
                return
        self._emit_readout(event)
        if self._pinned is not None:
            return
        i = self._nearest(event) if event.inaxes is self.ax else None
        if i != self._hover:
            self._show_trade(i)

    def _emit_readout(self, event) -> None:
        if self._eq is None or event.inaxes not in (self.ax, self.ax2) or event.xdata is None:
            return
        k = int(np.clip(np.searchsorted(self._ex, event.xdata), 0, len(self._ex) - 1))
        v = float(self._ey[k])
        peak = float(np.nanmax(self._ey[:k + 1]))
        txt = f"{self._eq.index[k]:%Y-%m-%d %H:%M}   equity {big_money(v)}   return {(v / self._capital - 1) * 100:+,.1f}%   drawdown {(v / peak - 1) * 100:+.1f}%"
        if len(self._by):
            txt += f"   buy & hold {big_money(float(self._by[k]))}"
        self.readout.emit(txt)

    def _on_release(self, event) -> None:
        p, moved = self._press, self._moved
        self._press, self._moved = None, False
        if p is None or moved or self._home is None:
            return
        i = self._nearest(event) if event.inaxes is self.ax else None        # a click without dragging: pin / unpin
        self._pinned = i
        self._show_trade(i)
        self.trade_picked.emit(-1 if i is None else int(i))

    def _on_leave(self, event) -> None:
        if self._pinned is None:
            self._show_trade(None)
        self.readout.emit("")

    def save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save equity curve", "equity_curve.png", "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=140, facecolor=BG)


class ChartPanel(QWidget):
    """The equity chart with a slim toolbar (the mouse controls are always listed, so nothing has to be guessed)."""

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.chart = EquityChart()
        row = QHBoxLayout()
        self.scale = QComboBox(); self.scale.addItems(["Equity $", "Return %"]); self.scale.setToolTip("Same curve, y axis labelled in dollars or as cumulative return")
        self.chk_log = QCheckBox("log scale"); self.chk_log.setChecked(True); self.chk_log.setToolTip("Log scale keeps early and late years comparable when equity grows 1000x")
        self.chk_bh = QCheckBox("buy && hold"); self.chk_bh.setChecked(True)
        self.chk_dots = QCheckBox("trades"); self.chk_dots.setChecked(True)
        self.btn_in, self.btn_out, self.btn_reset, self.btn_save = QPushButton("Zoom in"), QPushButton("Zoom out"), QPushButton("Reset view"), QPushButton("Save image…")
        for w in (self.scale, self.chk_log, self.chk_bh, self.chk_dots):
            row.addWidget(w)
        row.addStretch(1)
        for b in (self.btn_in, self.btn_out, self.btn_reset, self.btn_save):
            row.addWidget(b)
        v.addLayout(row)
        v.addWidget(self.chart, 1)
        self.readout = QLabel(""); self.readout.setObjectName("muted"); self.readout.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        hint = QLabel("Wheel = zoom · drag = pan · double-click = reset · hover / click a dot = trade info · Shift+wheel = zoom vertically")
        hint.setObjectName("muted")                       # deliberately NOT word-wrapped: a wrapping label makes the whole page "height depends on width" and inflates its minimum height
        v.addWidget(self.readout)
        v.addWidget(hint)
        self.scale.currentIndexChanged.connect(lambda i: self.chart.set_scale(pct_axis=bool(i)))
        self.chk_log.toggled.connect(lambda on: self.chart.set_scale(log=on))
        self.chk_bh.toggled.connect(lambda on: self.chart.set_overlays(show_bh=on))
        self.chk_dots.toggled.connect(lambda on: self.chart.set_overlays(show_dots=on))
        self.btn_in.clicked.connect(lambda: self.chart.zoom_by(0.7))
        self.btn_out.clicked.connect(lambda: self.chart.zoom_by(1.4))
        self.btn_reset.clicked.connect(self.chart.reset_view)
        self.btn_save.clicked.connect(self.chart.save_image)
        self.chart.readout.connect(self.readout.setText)

    def plot(self, eq: pd.Series, bh: Optional[pd.Series], trades: pd.DataFrame, capital: float, step: bool = False, bh_label: str = "Buy & hold") -> None:
        """Default the axis to log when the curve spans more than 20x (otherwise the first years are an unreadable flat line)."""
        top = float(eq.max()) / max(float(eq.min()), 1e-9) if len(eq) else 1.0
        self.chart.log = self.chk_log.isChecked() and top > 3.0
        self.chart.pct_axis = self.scale.currentIndex() == 1
        self.chart.show_bh, self.chart.show_dots = self.chk_bh.isChecked(), self.chk_dots.isChecked()
        self.chart.plot(eq, bh, trades, capital, step=step, bh_label=bh_label)
        self.chk_bh.setEnabled(bh is not None and len(bh) > 0)


class SignalChart(FigureCanvasQTAgg):
    """Price bars with the Supertrend line (green = uptrend/long, red = downtrend/short), the re-entry limit and, when a position is open, its frozen exit levels."""
    readout = pyqtSignal(str)

    def __init__(self):
        self.fig = Figure(figsize=(8, 4), facecolor=BG)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.07, right=0.865, top=0.96, bottom=0.1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._bars = None; self._st = None; self._home = None; self._press = None; self._moved = False
        self.message("Press 'Refresh' to load the latest bars")
        for name, fn in (("scroll_event", self._on_scroll), ("button_press_event", self._on_press), ("button_release_event", self._on_release), ("motion_notify_event", self._on_motion)):
            self.mpl_connect(name, fn)

    def _style(self) -> None:
        self.ax.set_facecolor(PANEL)
        self.ax.tick_params(colors=MUTED, labelsize=8)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.grid(True, color=BORDER, lw=0.5, alpha=0.7)

    def message(self, text: str) -> None:
        self._bars = None
        self.ax.clear(); self._style()
        self.ax.text(0.5, 0.5, text, transform=self.ax.transAxes, ha="center", va="center", color=MUTED, fontsize=11)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.draw_idle()

    def plot(self, bars: pd.DataFrame, st: np.ndarray, dirn: np.ndarray, levels: list, signal_bar=None) -> None:
        """``bars``: the bars to show (OHLC, naive ET index); ``st`` / ``dirn``: Supertrend and direction aligned to them (-1 = up); ``levels``: [(label, price, colour, style)]."""
        self.ax.clear(); self._style()
        self._bars, self._st = bars, st
        x = np.arange(len(bars))
        o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        up = c >= o
        col = np.where(up, GREEN, RED)
        self.ax.vlines(x, l, h, color=col, lw=1.0, zorder=3)
        self.ax.bar(x, np.where(up, c - o, o - c) + 1e-9, bottom=np.minimum(o, c), width=0.62, color=col, zorder=4, linewidth=0)
        long_ = np.where(dirn == -1, st, np.nan)
        short_ = np.where(dirn == 1, st, np.nan)
        self.ax.plot(x, long_, color=GREEN, lw=1.8, zorder=2, label="Supertrend (long)")
        self.ax.plot(x, short_, color=RED, lw=1.8, zorder=2, label="Supertrend (short)")
        n = len(bars)
        for label, price, colr, style in levels:
            self.ax.axhline(price, color=colr, lw=1.0, ls=style, zorder=1, alpha=0.9)
            self.ax.annotate(f"{label} {price:,.2f}", xy=(1.0, price), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=colr,
                             annotation_clip=False)
        if signal_bar is not None and signal_bar in bars.index:
            k = bars.index.get_loc(signal_bar)
            self.ax.axvspan(k - 0.5, k + 0.5, color=ACCENT, alpha=0.12, zorder=0, lw=0)
        days = pd.Series(bars.index.date)
        first = np.where(days.ne(days.shift()).to_numpy())[0]
        self.ax.set_xticks(first)
        self.ax.set_xticklabels([f"{bars.index[i]:%a %d %b}" for i in first])
        for i in first[1:]:
            self.ax.axvline(i - 0.5, color=BORDER, lw=0.7, zorder=0)
        self.ax.set_xlim(-1, n + 2)
        lo = min(l.min(), np.nanmin(st[-n:]) if np.isfinite(st[-n:]).any() else l.min())
        hi = max(h.max(), np.nanmax(st[-n:]) if np.isfinite(st[-n:]).any() else h.max())
        lv = [p for _, p, _, _ in levels]
        lo, hi = min([lo] + lv), max([hi] + lv)
        pad = (hi - lo) * 0.06 or 1.0
        self.ax.set_ylim(lo - pad, hi + pad)
        self.ax.legend(loc="upper left", fontsize=8, facecolor=PANEL2, edgecolor=BORDER, labelcolor=TEXT)
        self.draw()
        self._home = (self.ax.get_xlim(), self.ax.get_ylim())

    def _on_scroll(self, event) -> None:
        if self._bars is None or event.inaxes is not self.ax:
            return
        f = 0.8 if event.button == "up" else 1.25
        lo, hi = self.ax.get_xlim()
        c = event.xdata if event.xdata is not None else (lo + hi) / 2
        self.ax.set_xlim(c - (c - lo) * f, c + (hi - c) * f)
        self.draw_idle()

    def _on_press(self, event) -> None:
        if self._bars is None or event.inaxes is not self.ax or event.button != 1:
            return
        if getattr(event, "dblclick", False):
            self.ax.set_xlim(self._home[0]); self.ax.set_ylim(self._home[1]); self.draw_idle()
            return
        inv = self.ax.transData.inverted()
        self._press = dict(x=event.x, y=event.y, inv=inv, start=inv.transform((event.x, event.y)), xlim=self.ax.get_xlim(), ylim=self.ax.get_ylim())
        self._moved = False

    def _on_motion(self, event) -> None:
        if self._bars is None:
            return
        p = self._press
        if p is not None and event.x is not None and abs(event.x - p["x"]) + abs(event.y - p["y"]) > 4:
            self._moved = True
            cur = p["inv"].transform((event.x, event.y))
            dx, dy = p["start"][0] - cur[0], p["start"][1] - cur[1]
            self.ax.set_xlim(p["xlim"][0] + dx, p["xlim"][1] + dx); self.ax.set_ylim(p["ylim"][0] + dy, p["ylim"][1] + dy)
            self.draw_idle()
            return
        if event.inaxes is self.ax and event.xdata is not None:
            k = int(round(event.xdata))
            if 0 <= k < len(self._bars):
                r = self._bars.iloc[k]
                s = self._st[k] if k < len(self._st) else float("nan")
                self.readout.emit(f"{self._bars.index[k]:%a %d %b %H:%M}   O {r.open:,.2f}  H {r.high:,.2f}  L {r.low:,.2f}  C {r.close:,.2f}   Supertrend {s:,.2f}")

    def _on_release(self, event) -> None:
        self._press, self._moved = None, False
