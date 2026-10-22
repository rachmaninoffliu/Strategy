"""The ONE performance display used by BOTH the Backtest and the Live tab (same KPI cards, metrics, curve, trades, monthly heat-map and breakdown),
plus the 'Parameters in use' table that shows what each side really uses."""
from __future__ import annotations

import html
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget

from . import metrics as M
from .chart import ChartPanel
from .theme import ACCENT, COL_LONG, COL_SHORT, GREEN, MUTED, RED, YELLOW, Kpi, big_money, box_style, make_table, money, num, pct, set_row

TRADE_HEADERS = ["#", "Entry time (ET)", "Exit time (ET)", "Side", "Symbol", "Qty", "Entry", "Exit", "Return", "P&L $", "Exit by", "Held", "Equity $"]
IDX_ROLE = Qt.ItemDataRole.UserRole + 1


class SortItem(QTableWidgetItem):
    """Table cell that sorts by a number / timestamp stored in UserRole instead of by its text ('9' > '10' as text)."""

    def __lt__(self, other) -> bool:
        a, b = self.data(Qt.ItemDataRole.UserRole), other.data(Qt.ItemDataRole.UserRole)
        try:
            return a < b
        except TypeError:
            return super().__lt__(other)


def side_color(side: str) -> str:
    return COL_SHORT if str(side).lower().startswith("s") else COL_LONG


def held_text(entry, exit_) -> str:
    if pd.isna(entry) or pd.isna(exit_):
        return "—"
    h = (pd.Timestamp(exit_) - pd.Timestamp(entry)).total_seconds() / 3600.0
    return f"{h / 24:.1f} d" if h >= 48 else f"{h:.1f} h"


def _since(opened) -> str:
    try:
        return f" since {pd.Timestamp(opened):%m-%d %H:%M}" if opened else ""
    except (ValueError, TypeError):
        return ""


def open_html(v: dict, label: str) -> str:
    """One open position (the common shape of supertrend_gui.backtest_open_view / Engine.live_open) as two lines of rich text."""
    col = COL_SHORT if v["side"] == "short" else COL_LONG
    pnl = v.get("pnl_pct")
    pc = GREEN if (pnl or 0) >= 0 else RED
    head = (f"<span style='color:{MUTED}'>{html.escape(label)}</span> &nbsp; <b style='color:{col}'>{'▼ SHORT' if v['side'] == 'short' else '▲ LONG'}</b> "
            f"<b>{v['qty']:,} {html.escape(str(v['symbol']))}</b> @ {v['price']:,.2f}{_since(v.get('opened'))}")
    if v.get("now"):
        head += f" · now {v['now']:,.2f}"
    if pnl is not None:
        head += f" · <b style='color:{pc}'>{pnl:+.2f}%</b> ({money(v.get('pnl_usd') or 0.0, True)})"
    if not v.get("tracked", True):
        lv = f"<span style='color:{YELLOW}'>no frozen exit levels: this app did not open the position</span>"
    else:
        trail = (f"trailing stop <b>{v['trail']:,.2f}</b>" if v.get("trail") is not None else
                 f"trailing stop not armed (arms at {v['target']:,.2f}" + (f", best so far {v['extreme']:,.2f}" if v.get("extreme") is not None else "") + ")")
        lv = (f"SOXL levels: entry reference {v['entry_ref']:,.2f} · take-profit <b>{v['tp']:,.2f}</b> · {trail} · "
              f"{'fixed' if v.get('fixed') else 'dynamic'} mode")
    return head + f"<br><span style='color:{MUTED}'>{lv}</span>"


class PerformanceView(QWidget):
    KPI_NAMES = ("Total return", "Final equity", "CAGR", "Max drawdown", "Sharpe", "Win rate", "Profit factor", "Trades")

    def __init__(self, export_name: str = "trades.csv", export_dir: Callable[[], str] = lambda: "."):
        super().__init__()
        self.trades = pd.DataFrame()
        self.export_name, self.export_dir = export_name, export_dir
        self._linking = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        k = QHBoxLayout()
        self.kpis = {n: Kpi(n) for n in self.KPI_NAMES}
        for w in self.kpis.values():
            k.addWidget(w)
        root.addLayout(k)
        self.open_lbl = QLabel()                                     # the open position: Backtest = held after the last bar, Live = the account's (and whether they match)
        self.open_lbl.setTextFormat(Qt.TextFormat.RichText); self.open_lbl.setWordWrap(True)
        self.open_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.open_lbl)
        self.show_open(None, "Open position", "no result yet")
        self.tabs = QTabWidget()
        self.chart_panel = ChartPanel()
        self.chart = self.chart_panel.chart
        self.tabs.addTab(self.chart_panel, "Equity curve")
        self.mt = make_table(["Metric", "Value"])
        self.mt.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.mt.setColumnWidth(0, 430)
        self.tabs.addTab(self.mt, "Metrics")
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        self.tt = make_table(TRADE_HEADERS)
        self.tt.setSortingEnabled(True)
        self.tt.setToolTip("Click a row to mark that trade on the equity curve · double-click to zoom to it · click a heading to sort")
        row = QHBoxLayout()
        self.lbl_n = QLabel(""); self.lbl_n.setObjectName("muted")
        self.export_btn = QPushButton("Export trades to CSV"); self.export_btn.clicked.connect(self.export)
        row.addWidget(self.lbl_n, 1); row.addWidget(self.export_btn)
        v.addLayout(row); v.addWidget(self.tt)
        self.tabs.addTab(w, "Trades")
        self.mo = make_table(["Year"] + [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)] + ["Full year"])
        self.mo.horizontalHeader().setStretchLastSection(False)
        self.tabs.addTab(self.mo, "Monthly")
        self.bd = make_table(["Group", "Trades", "Win rate", "Avg return", "Median", "Best", "Worst", "Total P&L $", "Avg held (h)"])
        self.tabs.addTab(self.bd, "By side and exit")
        root.addWidget(self.tabs, 1)
        self.tt.itemSelectionChanged.connect(self._row_selected)
        self.tt.cellDoubleClicked.connect(lambda r, _c: self._row_zoom(r))
        self.chart.trade_picked.connect(self._chart_picked)

    # ── the open position ──
    def show_open(self, primary: Optional[dict], title: str, flat_text: str = "flat", other: Optional[dict] = None, other_title: str = "", cmp: Optional[dict] = None) -> None:
        """primary: the position to show (None = flat, ``flat_text`` says why). other/cmp: the position it is compared with (Live: the strict backtest) and the verdict
        of supertrend_gui.compare_open, which colours the strip."""
        self.open_state = {"primary": primary, "other": other, "cmp": cmp}
        lines = [open_html(primary, title) if primary else f"<span style='color:{MUTED}'>{html.escape(title)}</span> &nbsp; <b>{html.escape(flat_text)}</b>"]
        color = MUTED
        if cmp is not None:
            if other is not None:
                lines.append(open_html(other, other_title))
            status = cmp.get("status")
            color, word = {"match": (GREEN, "MATCHES THE BACKTEST"), "differs": (RED, "DIFFERS FROM THE BACKTEST"), "timing": (YELLOW, "TIMING (bar still forming)"),
                           "untracked": (YELLOW, "NOT TRACKED BY THIS APP"), "dry": (YELLOW, "DRY RUN")}.get(status, (MUTED, ""))
            lines.append(f"<b style='color:{color}'>{word}</b> &nbsp; {html.escape(cmp.get('text', ''))}")
        elif primary:
            color = COL_SHORT if primary["side"] == "short" else COL_LONG
        self.open_lbl.setText("<br>".join(lines))
        self.open_lbl.setStyleSheet(box_style(color, 0.10))

    # ── fill ──
    def show_result(self, eq: Optional[pd.Series], bh: Optional[pd.Series], trades: pd.DataFrame, m: dict, capital: float, empty_msg: str, step: bool = False,
                    bh_label: str = "Buy & hold", always_curve: bool = False) -> None:
        self.trades = trades if trades is not None else pd.DataFrame()
        n = int(m.get("trades", 0))
        self.kpis["Trades"].set(f"{n:,}")
        no_curve = eq is None or len(eq) < 2
        if n == 0 and (no_curve or not always_curve):
            for name, kpi in self.kpis.items():
                if name != "Trades":
                    kpi.set("—")
            self.chart.message(empty_msg)
            for t in (self.mt, self.tt, self.mo, self.bd):
                t.setRowCount(0)
            self.lbl_n.setText("")
            return
        self._fill_kpis(m, n)
        self.chart_panel.plot(eq, bh, self.trades, capital, step=step, bh_label=bh_label)
        self._fill_metrics(m)
        self._fill_trades(self.trades)
        self._fill_monthly(eq, capital)
        self._fill_breakdown(self.trades)

    def _fill_kpis(self, m: dict, n: int) -> None:
        tr = m.get("total_return", 0.0)
        self.kpis["Total return"].set(pct(tr, 1, True), GREEN if tr >= 0 else RED)
        self.kpis["Final equity"].set(big_money(m["final_equity"]), sub=f"from {big_money(m['initial_capital'])}")
        cg, yrs = m.get("cagr", float("nan")), m.get("period_years", 0.0)
        if yrs < 1 / 12 or not np.isfinite(cg):                              # annualising a few days of results says nothing: show the period instead of a silly number
            self.kpis["CAGR"].set("—", sub="needs ~1 month" if yrs < 1 / 12 else "")
        else:
            self.kpis["CAGR"].set(pct(cg, 1, True), GREEN if cg >= 0 else RED, sub=f"{yrs:.1f} years" + (" · extrapolated" if yrs < 1 else ""))
        dd = m.get("max_drawdown", 0.0)
        self.kpis["Max drawdown"].set(pct(dd, 1), RED if dd < -0.3 else YELLOW if dd < -0.15 else None)
        self.kpis["Sharpe"].set(num(m.get("sharpe"), 2), sub=f"per-bar {num(m.get('sharpe_bar'), 2)}" if np.isfinite(m.get("sharpe_bar", float("nan"))) else "daily, √252")
        self.kpis["Win rate"].set(pct(m.get("win_rate", float("nan")), 1), sub=f"payoff {num(m.get('payoff_ratio'), 2)}" if n else "")
        self.kpis["Profit factor"].set(num(m.get("profit_factor"), 2))

    def _fill_metrics(self, m: dict) -> None:
        sections = [(t, [it for it in items if it[1] in m]) for t, items in M.METRIC_SECTIONS]
        sections = [(t, items) for t, items in sections if items]
        self.mt.setRowCount(sum(len(items) + 1 for _, items in sections))
        r = 0
        for title, items in sections:
            it = QTableWidgetItem(title); f = QFont(); f.setBold(True); it.setFont(f); it.setForeground(QColor(ACCENT))
            self.mt.setItem(r, 0, it); self.mt.setItem(r, 1, QTableWidgetItem("")); r += 1
            for label, key, kind in items:
                v = m.get(key)
                col = None
                if kind in ("$s", "%s") and isinstance(v, (int, float)) and np.isfinite(v):
                    col = GREEN if v > 0 else RED if v < 0 else None
                set_row(self.mt, r, [label, M.fmt_metric(v, kind)], {1: col})
                self.mt.item(r, 1).setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                r += 1

    def _cell(self, text: str, key=None, color: Optional[str] = None, left: bool = False) -> SortItem:
        it = SortItem(text)
        it.setData(Qt.ItemDataRole.UserRole, key if key is not None else text)
        it.setToolTip(text)
        it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if left else Qt.AlignmentFlag.AlignRight) | Qt.AlignmentFlag.AlignVCenter)
        if color:
            it.setForeground(QColor(color))
        return it

    def _fill_trades(self, t: pd.DataFrame) -> None:
        self.tt.setSortingEnabled(False)
        self.tt.setRowCount(len(t))
        has_entry = len(t) and t["entry_time"].notna().any()
        for i, x in enumerate(t.itertuples(index=False)):
            et = pd.Timestamp(x.entry_time) if pd.notna(x.entry_time) else None
            xt = pd.Timestamp(x.exit_time)
            win = x.pnl_usd > 0
            c = GREEN if win else RED
            sym = getattr(x, "symbol", "") or ("SOXL" if str(x.side).lower() == "long" else "SOXL short")
            cells = [self._cell(str(i + 1), i + 1), self._cell(f"{et:%Y-%m-%d %H:%M}" if et is not None else "—", et.value if et is not None else 0, left=True),
                     self._cell(f"{xt:%Y-%m-%d %H:%M}", xt.value, left=True), self._cell(str(x.side).upper(), str(x.side), side_color(x.side), left=True), self._cell(str(sym), str(sym), left=True),
                     self._cell(f"{int(x.qty):,}", float(x.qty)), self._cell(f"{x.price:,.2f}", float(x.price)), self._cell(f"{x.exit_price:,.2f}", float(x.exit_price)),
                     self._cell(f"{x.return_pct:+.2f}%", float(x.return_pct), c), self._cell(money(x.pnl_usd, True), float(x.pnl_usd), c),
                     self._cell(str(x.exit_reason), str(x.exit_reason), left=True),
                     self._cell(held_text(x.entry_time, x.exit_time), float((xt - et).total_seconds()) if et is not None else 0.0), self._cell(money(x.equity_after), float(x.equity_after))]
            for j, it in enumerate(cells):
                it.setData(IDX_ROLE, i)
                self.tt.setItem(i, j, it)
        self.tt.setSortingEnabled(True)
        self.tt.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.lbl_n.setText(f"{len(t):,} trades" + (f" · {t['exit_time'].min():%Y-%m-%d} → {t['exit_time'].max():%Y-%m-%d}" if len(t) else "") + ("" if has_entry else " · entry times are not known from closing fills"))
        if len(t):
            self.tt.scrollToBottom()

    def _fill_monthly(self, eq: pd.Series, capital: float) -> None:
        tab = M.monthly_table(eq, capital)
        self.mo.setRowCount(len(tab))
        for i, (yr, r) in enumerate(tab.iterrows()):
            self.mo.setItem(i, 0, QTableWidgetItem(str(yr)))
            for j, v in enumerate(list(r.values), start=1):
                it = QTableWidgetItem("" if not np.isfinite(v) else pct(v, 1, True) if abs(v) < 100 else f"{v * 100:+,.0f}%")
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if np.isfinite(v):
                    a = min(abs(v) / (0.6 if j == 13 else 0.3), 1.0) * 0.6
                    it.setBackground(self._tint(GREEN if v >= 0 else RED, a))
                    it.setToolTip(f"{yr}  {'year' if j == 13 else pd.Timestamp(2000, j, 1).strftime('%B')}: {v * 100:+,.2f}%")
                    if j == 13:
                        f = QFont(); f.setBold(True); it.setFont(f)
                self.mo.setItem(i, j, it)
        for j in range(14):
            self.mo.setColumnWidth(j, 62 if j < 13 else 84)

    @staticmethod
    def _tint(color: str, alpha: float) -> QColor:
        c = QColor(color); c.setAlphaF(alpha)
        return c

    def _fill_breakdown(self, t: pd.DataFrame) -> None:
        b = M.breakdown(t)
        self.bd.setRowCount(len(b))
        for i, r in enumerate(b.itertuples()):
            c = GREEN if r.pnl > 0 else RED
            set_row(self.bd, i, [r.label, f"{r.n:,}", pct(r.win, 1), pct(r.avg, 2, True), pct(r.median, 2, True), pct(r.best, 1, True), pct(r.worst, 1, True), money(r.pnl, True),
                                 "—" if not np.isfinite(r.hold) else f"{r.hold:.1f}"], {3: c, 7: c})
            if r.bold:
                for j in range(self.bd.columnCount()):
                    f = QFont(); f.setBold(True); self.bd.item(i, j).setFont(f)

    # ── table <-> chart ──
    def _trade_index(self, row: int) -> Optional[int]:
        it = self.tt.item(row, 0)
        return None if it is None else it.data(IDX_ROLE)

    def _row_selected(self) -> None:
        if self._linking:
            return
        rows = self.tt.selectionModel().selectedRows() if self.tt.selectionModel() else []
        if not rows:
            self.chart.clear_highlight(); self.chart.draw_idle()
            return
        i = self._trade_index(rows[0].row())
        if i is None or i >= len(self.trades):
            return
        r = self.trades.iloc[i]
        self.chart.highlight(r.entry_time, r.exit_time, r.side)

    def _row_zoom(self, row: int) -> None:
        i = self._trade_index(row)
        if i is not None:
            self.tabs.setCurrentIndex(0)
            self.chart.focus_trade(int(i))

    def _chart_picked(self, i: int) -> None:
        """A dot was clicked on the chart: select (and scroll to) that trade's row."""
        if i < 0:
            return
        self._linking = True
        try:
            for r in range(self.tt.rowCount()):
                if self._trade_index(r) == i:
                    self.tt.selectRow(r)
                    self.tt.scrollToItem(self.tt.item(r, 0), QTableWidget.ScrollHint.PositionAtCenter)
                    break
        finally:
            self._linking = False

    def export(self) -> None:
        if self.trades is None or self.trades.empty:
            QMessageBox.information(self, "Export", "There are no trades to export yet.")
            return
        import os
        path, _ = QFileDialog.getSaveFileName(self, "Save trades", os.path.join(self.export_dir(), self.export_name), "CSV (*.csv)")
        if path:
            self.trades.drop(columns=[c for c in ("entry_alert", "exit_alert") if c in self.trades.columns]).to_csv(path, index=False)


class ParamsView(QWidget):
    """'Parameters in use': every parameter, with the value the Backtest uses next to the value the Live engine uses.
    White = shared; yellow = the two sides differ (see the note); grey = used by only one side or not read in the chosen mode."""

    def __init__(self):
        super().__init__()
        self.get_state: Optional[Callable] = None                     # set by MainWindow: () -> list[Row]
        v = QVBoxLayout(self)
        self.head = QLabel(""); self.head.setObjectName("muted"); self.head.setWordWrap(True)
        self.table = make_table(["Parameter", "Backtest", "Live engine", "Note"])
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
            rows, frozen = self.get_state()
        except Exception as e:                    # noqa: BLE001
            self.head.setText(f"cannot describe the parameters: {e}")
            return
        n_diff = sum(1 for r in rows if r.kind == "differs")
        self.head.setText(("The Live column shows the RUNNING session (settings frozen when you pressed Start). " if frozen else "The Live column shows what a session started now would use. ") +
                          "White rows are shared: the same value on both sides. Yellow rows differ (see the note) - " + (f"{n_diff} right now. " if n_diff else "none right now. ") +
                          "Grey cells are used by only one side, or not read in the chosen strategy mode.")
        groups = []
        for r in rows:
            if not groups or groups[-1] != r.group:
                groups.append(r.group)
        self.table.setRowCount(len(rows) + len(groups))
        i, last = 0, None
        for r in rows:
            if r.group != last:
                it = QTableWidgetItem(r.group); f = QFont(); f.setBold(True); it.setFont(f); it.setForeground(QColor(ACCENT))
                self.table.setItem(i, 0, it)
                for j in (1, 2, 3):
                    self.table.setItem(i, j, QTableWidgetItem(""))
                i += 1
                last = r.group
            k = r.kind
            for j, txt in enumerate([r.name, r.backtest, r.live, r.note]):
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
