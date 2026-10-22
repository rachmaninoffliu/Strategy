"""Tab 3 - Full historical backtesting engine: parameters, async progress, equity / lockup / drawdown charts."""
from __future__ import annotations

import html
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QDate, Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea,
                             QSpinBox, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from config import FeeConfig
from core.backtester import BacktestConfig, BacktestResult, run_backtest
from core.models import utcnow
from database import Database
from ui.common import (ACCENT, BORDER, GREEN, MUTED, RED, YELLOW, AsyncWorker, DataTable, KpiCard, RowTableModel,
                       SortProxy, dark_canvas, fmt_usd, style_ax)
from ui.tab_history import COLS, HistRow

KPI_ORDER = ["CAGR / APY (%)", "Max Drawdown (%)", "Win Rate (%)", "Avg Lockup (days)", "Sharpe Ratio",
             "Sortino Ratio", "Total Net PnL ($)", "Return on Capital (%)", "Trades", "Capital Utilisation (%)"]


def _dspin(val, lo, hi, dec=2, suffix="", prefix="", step=None) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setValue(val)
    s.setSuffix(suffix)
    s.setPrefix(prefix)
    s.setGroupSeparatorShown(True)
    if step:
        s.setSingleStep(step)
    return s


class BacktestTab(QWidget):
    status = pyqtSignal(str)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.worker: Optional[AsyncWorker] = None
        self.result: Optional[BacktestResult] = None

        root = QHBoxLayout(self)
        root.addWidget(self._sidebar())
        root.addWidget(self._main(), 1)

    # ══ sidebar ═══════════════════════════════════════════════════════════════
    def _sidebar(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)

        g = QGroupBox("Date range")
        f = QFormLayout(g)
        today = QDate.currentDate()
        self.d_start = QDateEdit(today.addDays(-182)); self.d_start.setCalendarPopup(True)
        self.d_end = QDateEdit(today.addDays(-1)); self.d_end.setCalendarPopup(True)
        for d in (self.d_start, self.d_end):
            d.setDisplayFormat("yyyy-MM-dd")
        f.addRow("Start", self.d_start)
        f.addRow("End", self.d_end)
        pr = QHBoxLayout()
        for name, days in (("3M", 91), ("6M", 182), ("1Y", 365), ("Full", 1800)):
            b = QPushButton(name)
            b.setFixedWidth(46)
            b.clicked.connect(lambda _=False, d=days: self._preset(d))
            pr.addWidget(b)
        f.addRow(pr)
        lay.addWidget(g)

        g = QGroupBox("Capital and sizing")
        f = QFormLayout(g)
        self.capital = _dspin(10_000, 100, 1e9, 0, prefix="$ ", step=1000)
        self.hurdle = _dspin(7.0, 0, 1e5, 1, " %")
        self.per_trade = _dspin(1_000, 10, 1e8, 0, prefix="$ ", step=100)
        self.cmax = _dspin(10_000, 100, 1e9, 0, prefix="$ ", step=1000)
        self.sizing = QComboBox(); self.sizing.addItems(["Fixed allocation", "Fractional Kelly"])
        self.kelly = _dspin(0.25, 0.01, 1, 2)
        self.basis = _dspin(2.0, 0, 50, 1, " %")
        self.basis.setToolTip("Prior probability that the two venues resolve differently (total loss of the trade)")
        f.addRow("Initial capital", self.capital)
        f.addRow("Min APY hurdle", self.hurdle)
        f.addRow("Max capital / trade", self.per_trade)
        f.addRow("Max concurrent (C max)", self.cmax)
        f.addRow("Sizing", self.sizing)
        f.addRow("Kelly fraction", self.kelly)
        f.addRow("Mismatch risk (Kelly)", self.basis)
        lay.addWidget(g)

        g = QGroupBox("Fees and fills")
        f = QFormLayout(g)
        self.kfee = _dspin(0.07, 0, 1, 3)
        self.kfee.setToolTip("Kalshi taker fee = ceil-to-cent( coef x C x P x (1-P) )")
        self.pmode = QComboBox(); self.pmode.addItems(["schedule", "flat", "zero"])
        self.pmode.setToolTip("schedule: per-market feeSchedule (rate, exponent) when fee-enabled; flat: bps of notional")
        self.pflat = _dspin(0, 0, 1000, 1, " bps")
        self.extra = _dspin(0, 0, 1000, 1, " bps")
        self.slip = _dspin(25, 0, 2000, 1, " bps")
        self.bar = QComboBox(); self.bar.addItems(["Daily bars", "Hourly bars"])
        self.delay = QSpinBox(); self.delay.setRange(0, 5); self.delay.setValue(1)
        self.half = _dspin(0.5, 0, 10, 2, " c")
        self.half.setToolTip("Polymarket history is price-only: half-spread assumed on each side")
        self.depth = _dspin(500, 10, 1e8, 0, prefix="$ ")
        f.addRow("Kalshi fee coef", self.kfee)
        f.addRow("Polymarket fees", self.pmode)
        f.addRow("  flat rate", self.pflat)
        f.addRow("Extra fee (both venues)", self.extra)
        f.addRow("Slippage buffer", self.slip)
        f.addRow("Bar size", self.bar)
        f.addRow("Execution delay (bars)", self.delay)
        f.addRow("Poly half-spread", self.half)
        f.addRow("Assumed depth / leg", self.depth)
        lay.addWidget(g)

        g = QGroupBox("Universe")
        f = QFormLayout(g)
        self.minvol = _dspin(500, 0, 1e9, 0)
        self.minconf = _dspin(0.80, 0, 1, 2)
        self.maxst = QSpinBox(); self.maxst.setRange(10, 5000); self.maxst.setValue(300)
        self.kp = QSpinBox(); self.kp.setRange(1, 500); self.kp.setValue(60)
        self.pp = QSpinBox(); self.pp.setRange(1, 500); self.pp.setValue(60)
        self.t_par = QCheckBox("I Parity"); self.t_mec = QCheckBox("II MECE"); self.t_sub = QCheckBox("III Subset")
        for c in (self.t_par, self.t_mec, self.t_sub):
            c.setChecked(True)
        tr = QHBoxLayout()
        for c in (self.t_par, self.t_mec, self.t_sub):
            tr.addWidget(c)
        f.addRow("Min volume", self.minvol)
        f.addRow("Min match conf.", self.minconf)
        f.addRow("Max structures", self.maxst)
        f.addRow("Kalshi pages", self.kp)
        f.addRow("Polymarket pages", self.pp)
        f.addRow(tr)
        lay.addWidget(g)

        self.b_run = QPushButton("Run Backtest")
        self.b_run.setObjectName("primary")
        self.b_run.setMinimumHeight(34)
        self.b_run.clicked.connect(self.run)
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.setObjectName("danger")
        self.b_cancel.setEnabled(False)
        self.b_cancel.clicked.connect(lambda: self.worker and self.worker.cancel())
        row = QHBoxLayout()
        row.addWidget(self.b_run, 2)
        row.addWidget(self.b_cancel, 1)
        lay.addLayout(row)
        self.progress = QProgressBar(); self.progress.setRange(0, 100)
        lay.addWidget(self.progress)
        self.lbl = QLabel("Ready.")
        self.lbl.setObjectName("muted")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        lay.addStretch(1)

        sa = QScrollArea()
        sa.setWidget(box)
        sa.setWidgetResizable(True)
        sa.setFixedWidth(352)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        return sa

    def _preset(self, days: int):
        today = QDate.currentDate()
        self.d_end.setDate(today.addDays(-1))
        self.d_start.setDate(today.addDays(-days))

    # ══ main panel ════════════════════════════════════════════════════════════
    def _main(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        note = QLabel("NOT US-TRADABLE DATA: Polymarket price history and settled markets come from international polymarket.com (the only source with history), which prohibits US persons. These results show how Kalshi and Polymarket priced the same events - they are NOT a record of trades a US person could have made.")
        note.setWordWrap(True)
        note.setStyleSheet("background:#3a1b20;border:1px solid #e05561;border-radius:4px;padding:5px 8px;"
                           "color:#ffb3ba;font-weight:600;")
        lay.addWidget(note)
        grid = QGridLayout()
        self.k: dict[str, KpiCard] = {}
        for i, name in enumerate(KPI_ORDER):
            c = KpiCard(name)
            self.k[name] = c
            grid.addWidget(c, i // 5, i % 5)
        lay.addLayout(grid)

        split = QSplitter(Qt.Orientation.Vertical)
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        chart = QWidget()
        cl = QVBoxLayout(chart)
        cl.setContentsMargins(0, 0, 0, 0)
        self.fig, self.canvas = dark_canvas(9, 6.4)
        self.toolbar = NavigationToolbar2QT(self.canvas, chart)
        self.toolbar.setStyleSheet(f"background:{BORDER};")
        cl.addWidget(self.toolbar)
        cl.addWidget(self.canvas, 1)
        split.addWidget(chart)

        tabs = QTabWidget()
        self.model = RowTableModel(COLS)
        self.proxy = SortProxy(self)
        self.proxy.setSourceModel(self.model)
        self.table = DataTable()
        self.table.setModel(self.proxy)
        tabs.addTab(self.table, "Trades")
        self.sheet = QTextBrowser()
        tabs.addTab(self.sheet, "Tear sheet and assumptions")
        split.addWidget(tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)

        ex = QHBoxLayout()
        self.b_csv = QPushButton("Export trades (CSV)")
        self.b_csv.clicked.connect(self._export_csv)
        self.b_html = QPushButton("Export tear sheet (HTML)")
        self.b_html.clicked.connect(self._export_html)
        ex.addStretch(1)
        ex.addWidget(self.b_csv)
        ex.addWidget(self.b_html)
        lay.addLayout(ex)
        self._draw(None)
        return w

    # ══ run ═══════════════════════════════════════════════════════════════════
    def _config(self) -> BacktestConfig:
        qd = lambda d: datetime(d.year(), d.month(), d.day(), tzinfo=timezone.utc)   # noqa: E731
        types = tuple(n for n, c in (("parity", self.t_par), ("mece", self.t_mec), ("subset", self.t_sub))
                      if c.isChecked())
        return BacktestConfig(
            start=qd(self.d_start.date()), end=qd(self.d_end.date()) + timedelta(days=1) - timedelta(seconds=1),
            initial_capital=self.capital.value(), max_capital_per_trade=self.per_trade.value(),
            max_concurrent_capital=self.cmax.value(), hurdle_apy=self.hurdle.value() / 100.0,
            fees=FeeConfig(kalshi_coef=self.kfee.value(), poly_mode=self.pmode.currentText(),
                           poly_flat_bps=self.pflat.value(), extra_bps=self.extra.value()),
            slippage_bps=self.slip.value(), sizing="kelly" if self.sizing.currentIndex() == 1 else "fixed",
            kelly_mult=self.kelly.value(), basis_risk_prob=self.basis.value() / 100.0,
            exec_delay_bars=self.delay.value(), bar_minutes=60 if self.bar.currentIndex() == 1 else 1440,
            min_match_confidence=self.minconf.value(), min_volume=self.minvol.value(),
            assumed_depth_usd=self.depth.value(), poly_half_spread=self.half.value() / 100.0,
            types=types or ("parity",), max_structures=self.maxst.value(), kalshi_max_pages=self.kp.value(),
            poly_max_pages=self.pp.value())

    def run(self):
        if self.worker and self.worker.isRunning():
            return
        cfg = self._config()
        if cfg.start >= cfg.end:
            self.lbl.setText("Start date must be before end date.")
            return
        db = self.db
        self.worker = AsyncWorker(lambda prog, evt: run_backtest(cfg, db, prog, evt))
        self.worker.progress.connect(lambda p, m: (self.progress.setValue(p), self.lbl.setText(m)))
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(lambda e: self._end(f"Backtest failed: {e}"))
        self.worker.cancelled.connect(lambda: self._end("Backtest cancelled."))
        self.b_run.setEnabled(False)
        self.b_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.worker.start()

    def _end(self, msg: str):
        self.b_run.setEnabled(True)
        self.b_cancel.setEnabled(False)
        self.lbl.setText(msg)
        self.status.emit(msg)

    def _done(self, res: BacktestResult):
        self.result = res
        m = res.metrics
        for name in KPI_ORDER:
            v = m.get(name)
            txt = "-" if v is None else (f"{v:,.2f}" if isinstance(v, float) else f"{v:,}")
            color = None
            if name in ("Total Net PnL ($)", "Return on Capital (%)", "CAGR / APY (%)") and isinstance(v, (int, float)):
                color = GREEN if v > 0 else (RED if v < 0 else None)
            if name == "Max Drawdown (%)" and isinstance(v, (int, float)) and v < 0:
                color = RED
            if name.endswith("(%)") and v is not None:
                txt += "%"
            if name == "Total Net PnL ($)" and v is not None:
                txt = fmt_usd(v)
            self.k[name].set(txt, color)
        self.model.set_rows([HistRow(t) for t in res.trades])
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(2, 240)
        self.table.setColumnWidth(3, 215)
        self._draw(res)
        self.sheet.setHtml(self._tear_sheet(res))
        self._end(f"Backtest done in {res.stats.get('seconds')}s: {len(res.trades)} trades, "
                  f"{res.stats.get('structures', 0)} structures, {res.stats.get('markets_with_history', 0)} price histories.")

    # ══ charts ════════════════════════════════════════════════════════════════
    def _draw(self, res: Optional[BacktestResult]):
        self.fig.clear()
        gs = self.fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.28)
        a1 = self.fig.add_subplot(gs[0])
        a2 = self.fig.add_subplot(gs[1], sharex=a1)
        a3 = self.fig.add_subplot(gs[2], sharex=a1)
        style_ax(a1, "Cumulative portfolio equity vs cash benchmark (risk-free)", "$")
        style_ax(a2, "Locked capital utilisation", "$ locked")
        style_ax(a3, "Drawdown (underwater)", "%")
        if res is None or res.equity.empty:
            a1.text(0.5, 0.5, "Run a backtest to see results" if res is None else "No trades - nothing cleared the "
                    "hurdle / data gates in this window", transform=a1.transAxes, ha="center", color=MUTED)
        else:
            eq, bm = res.equity, res.benchmark
            a1.plot(eq.index, eq.values, color=ACCENT, linewidth=1.8, label="Strategy equity (realised basis)")
            a1.plot(bm.index, bm.values, color=YELLOW, linewidth=1.1, linestyle="--",
                    label=f"Cash @ {res.cfg.risk_free:.0%} Treasury")
            a1.fill_between(eq.index, eq.values, bm.values, where=eq.values >= bm.values, color=GREEN, alpha=0.15)
            a1.fill_between(eq.index, eq.values, bm.values, where=eq.values < bm.values, color=RED, alpha=0.15)
            a1.legend(facecolor="#151a21", edgecolor=BORDER, labelcolor="#d7dee8", fontsize=8, loc="upper left")
            lk = res.locked
            a2.fill_between(lk.index, lk.values, step="post", color=YELLOW, alpha=0.35)
            a2.step(lk.index, lk.values, where="post", color=YELLOW, linewidth=1.2)
            a2.axhline(res.cfg.c_max, color=RED, linestyle=":", linewidth=1)
            dd = res.drawdown * 100
            a3.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.45)
            a3.plot(dd.index, dd.values, color=RED, linewidth=1)
        for a in (a1, a2):
            a.tick_params(labelbottom=False)
        self.fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.08)
        self.canvas.draw_idle()

    # ══ tear sheet / export ═══════════════════════════════════════════════════
    def _tear_sheet(self, res: BacktestResult) -> str:
        cfg = res.cfg
        rows = "".join(f"<tr><td style='padding:2px 14px 2px 0;color:{MUTED}'>{html.escape(k)}</td>"
                       f"<td><b>{'-' if v is None else html.escape(str(v))}</b></td></tr>" for k, v in res.metrics.items())
        st = res.stats
        sk = st.get("skipped", {})
        info = (f"universe: {st.get('kalshi_universe', 0):,} Kalshi / {st.get('poly_universe', 0):,} Polymarket settled "
                f"contracts &middot; {st.get('pairs', 0)} matched pairs &middot; {st.get('subset_edges', 0):,} ladder edges "
                f"&middot; {st.get('mece_baskets', 0)} MECE baskets &middot; {st.get('structures', 0)} structures replayed "
                f"&middot; {st.get('markets_with_history', 0)}/{st.get('markets_requested', 0)} histories available "
                f"&middot; {st.get('bars_rejected_by_data_gate', 0):,} bars rejected by the mid-sum data gate "
                f"&middot; skipped: capital {sk.get('capital', 0)}, concurrency {sk.get('concurrency', 0)}, "
                f"min-size {sk.get('min_size', 0)}, Kelly&le;0 {sk.get('kelly_zero', 0)}")
        ass = "".join(f"<li>{html.escape(a)}</li>" for a in res.assumptions)
        return (f"<h3>Backtest tear sheet</h3><p style='color:{MUTED}'>{cfg.start:%Y-%m-%d} &rarr; {cfg.end:%Y-%m-%d} "
                f"&middot; hurdle {cfg.hurdle_apy:.1%} APY &middot; ${cfg.initial_capital:,.0f} capital "
                f"&middot; C max ${cfg.c_max:,.0f} &middot; {cfg.sizing} sizing</p>"
                f"<table>{rows}</table><p style='color:{MUTED}'>{info}</p>"
                f"<h4>Assumptions &amp; data flags</h4><ul>{ass}</ul>")

    def _export_csv(self):
        if not self.result or not self.result.trades:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export trades", "backtest_trades.csv", "CSV (*.csv)")
        if path:
            rows = []
            for t in self.result.trades:
                d = t.model_dump(exclude={"legs"})
                d["legs"] = " | ".join(f"{l['venue']} {l['side']} {l['market']} @ {l['price']:.4f}" for l in t.legs)
                d["pnl"] = t.pnl
                rows.append(d)
            pd.DataFrame(rows).to_csv(path, index=False)
            self.status.emit(f"Exported {len(rows)} trades to {path}")

    def _export_html(self):
        if not self.result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export tear sheet", "tear_sheet.html", "HTML (*.html)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("<html><body style='font-family:Segoe UI;background:#0e1116;color:#d7dee8'>"
                        + self._tear_sheet(self.result) + "</body></html>")
            self.status.emit(f"Exported tear sheet to {path}")
