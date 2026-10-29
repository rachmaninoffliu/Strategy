"""Main window of the mean-reversion app: the model picker on the left; Backtest / Live trading / How it works on the right."""
from __future__ import annotations

import html
import os
import sys

import pandas as pd
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QScrollArea, QSplitter, QTabWidget, QTextBrowser

from ... import data as D
from ...ui.main_window import app_settings
from ...ui.theme import apply_theme, scrollable
from ...ui.workers import FnWorker, WorkerBag
from .. import engine as E
from ..market import load_daily, update_daily
from ..models import universe_parts
from .backtest_tab import BacktestTab
from .live_tab import LiveTab
from .model_panel import ModelPanel

KEY = "MeanReversion"

HELP = """
<h2>Mean reversion - backtest + live (Alpaca)</h2>
<p>Buy what was pushed down (a close near the day's low, a short-term RSI washout, a few days of losses) and sell it into the rebound. Short-term reversal is one of the oldest
anomalies in equities (Lehmann 1990, Jegadeesh 1990); the daily rules here are the practitioners' versions of it: <b>IBS</b> (internal bar strength, Pagonidis 2013),
<b>Connors RSI(2)</b> (Connors &amp; Alvarez 2008), N-day returns and z-scores.</p>
<h3>Timing - why the results are honest</h3>
<p>Most published IBS / RSI(2) backtests decide on the closing price and also trade at that close, which nobody can do: the close is not known until the auction prints. Here the
decision uses only the <b>15-minute bars that ended by the decision time</b> (15:30 with the free 16-minute-delayed SIP feed, 15:45 with a real-time feed) plus earlier official closes;
orders are <b>market-on-close</b> (sent before Alpaca's 15:50 cut-off) and fill at the official close. 'Overnight' models sell in the next <b>opening auction</b> (market-on-open). On
early-close days everything moves with the close (decision 12:30 for a 13:00 close). Costs: the auction cost per side (default 1 bp) on every fill.</p>
<h3>Capital - no leverage</h3>
<p>The book never holds more than 1x equity. A model has <b>slots</b>: at most that many positions, each sized at (1 − cash reserve) / slots of the equity in whole shares at the
decision price. When more names qualify than there are free slots, <b>which signals get the capital</b> decides: <i>signal</i> = the most extreme reading first (the lowest IBS /
RSI / return), <i>vol_low</i> = the calmest names first, <i>vol_high</i> = the most volatile first. The cash reserve (default 2%) absorbs a close above the decision price.</p>
<h3>Backtest = live</h3>
<p>The backtest and the live engine call the same <code>decide()</code> with the same signals, sizes and auctions. The tests replay backtests session by session through the live
engine with a simulated broker and require identical orders, fills and final equity. In live, 'Check live decisions against the backtest data' rebuilds each journaled decision from
the backtest's data with the positions live had. The Live tab trades exactly the model in the panel (the panel is locked while the loop runs) and warns when that model has not been
backtested in this session.</p>
<h3>Daily cycle (live)</h3>
<ul><li>08:45 - top up the daily bars; an overnight model sends its market-on-open sells.</li>
<li>decision time (15:46:06 for the 15:30 snapshot on the delayed SIP; 15:45:06 for 15:45 on a real-time feed) - exits and entries as market-on-close orders.</li>
<li>16:05 - the auction fills are journaled and the position book (entry day / price) is rebuilt from the broker's positions.</li></ul>
<h3>Safety</h3>
<ul><li>Starts on PAPER and DRY RUN; ARM needs the typed phrase <code>ARM PAPER</code>; real money is disabled in code until Valid signs off.</li>
<li>Keys: the project's <code>Alpaca Key.txt</code>; <code>MR_ALPACA_API_KEY</code> / <code>MR_ALPACA_SECRET_KEY</code> select a separate paper account. An account another strategy
(ResMomentum, ...) has claimed is refused - two strategies on one account would close each other's positions.</li>
<li>Positions in symbols the model can never trade are reported and left alone. KILL cancels and flattens. Optional live-only overlays: daily-loss halt, drawdown stop.</li></ul>
<h3>Papers</h3>
<ul><li>Lehmann (1990), Fads, martingales, and market efficiency, QJE 105(1)</li><li>Jegadeesh (1990), Evidence of predictable behavior of security returns, J. Finance 45(3)</li>
<li>Connors &amp; Alvarez (2008), Short Term Trading Strategies That Work</li><li>Pagonidis (2013), The IBS effect: mean reversion in equity ETFs, NAAIM</li>
<li>Da, Liu &amp; Schaumburg (2014), A closer look at the short-term return reversal, Management Science 60(3)</li></ul>
"""


class MRWindow(QMainWindow):
    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        self.setWindowTitle("Mean Reversion - Backtest + Live (Alpaca) - no leverage")
        self.resize(1560, 960)
        self.qs = app_settings(KEY)
        self._workers = WorkerBag()
        self.panel = ModelPanel(folder)
        self.panel.load(self.qs)
        self.panel.btn_update.clicked.connect(self.update_data)
        self.backtest = BacktestTab(self.panel, folder)
        self.live = LiveTab(self.panel, self.backtest, self.qs, folder)
        self.live.load()
        side = QScrollArea(); side.setWidgetResizable(True); side.setWidget(self.panel); side.setMinimumWidth(460)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tabs = QTabWidget()
        tabs.addTab(scrollable(self.backtest, 900, 760), "Backtest")
        tabs.addTab(scrollable(self.live, 900, 860), "Live trading")
        hb = QTextBrowser(); hb.setOpenExternalLinks(True)
        rep = os.path.join(folder, "REPORT.md")
        hb.setHtml(HELP + f"<p>Research, the team review and the verdict: <code>{html.escape(rep)}</code></p>")
        tabs.addTab(hb, "How it works")
        self.main_tabs = tabs
        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(side); sp.addWidget(tabs)
        sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1); sp.setSizes([480, 1080]); sp.setChildrenCollapsible(False)
        self.setCentralWidget(sp)
        self.state_lbl = QLabel("")
        self.statusBar().addPermanentWidget(self.state_lbl)
        self._menus()
        self._restore()
        self._data_label()
        self._t = QTimer(self); self._t.timeout.connect(self._status); self._t.start(1000)

    def _act(self, menu, text, fn, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(lambda *_: fn())
        menu.addAction(a); self.addAction(a)

    def _menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        self._act(m, "Save model…", self.panel.save_file, "Ctrl+S")
        self._act(m, "Load model…", self.panel.load_file, "Ctrl+O")
        m.addSeparator()
        self._act(m, "Open the app folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.folder)))
        self._act(m, "Quit", self.close, "Ctrl+Q")
        m = mb.addMenu("&Run")
        self._act(m, "Run backtest", lambda: (self.main_tabs.setCurrentIndex(0), self.backtest.run()), "Ctrl+R")
        self._act(m, "Plan now (dry run)", lambda: (self.main_tabs.setCurrentIndex(1), self.live.plan_now()), "Ctrl+L")
        self._act(m, "Update data", self.update_data)

    def _data_label(self) -> None:
        try:
            d = load_daily(["SPY"])
            u = D.load_universe()
            self.panel.data_lbl.setText(f"daily bars through {d.date.max():%Y-%m-%d}" + (f" · stock universe through {u.month.max():%Y-%m}" if len(u) else ""))
        except Exception as e:                                                                # noqa: BLE001
            self.panel.data_lbl.setText(f"⚠ {e}")

    def update_data(self) -> None:
        if self._workers.busy():
            return
        self.panel.btn_update.setEnabled(False)
        m = self.panel.model()

        def fn(log):
            syms: list = []
            for part in universe_parts(m.universe):
                if part.startswith("top"):
                    D.update_universe(log=log)
                    u = D.load_universe()
                    syms += ["SPY"] + sorted(set(u[(u.month >= u.month.max() - pd.DateOffset(months=13)) & (u["rank"] <= int(part[3:] or 100))].symbol))
                else:                                                                         # (delisted members keep their cached history)
                    syms += E._part_symbols(part)
            syms = list(dict.fromkeys(syms))
            update_daily(syms, log=log)
            D.update_bars(syms, "sip", log=log)
            return True
        self._workers.start(FnWorker(fn), self._updated, lambda s: self.statusBar().showMessage(s.strip()[:140]))

    def _updated(self, res, err) -> None:
        self.panel.btn_update.setEnabled(True)
        self.statusBar().showMessage("Data update failed: " + err if err else "Data updated", 8000)
        self.backtest.invalidate_data()
        self._data_label()

    def _status(self) -> None:
        armed = self.live.armed.isChecked()
        txt, col = ("● ARMED - PAPER", "#d9a520") if armed else ("DRY RUN - nothing is sent", "#8b98a9")
        self.state_lbl.setText(txt)
        self.state_lbl.setStyleSheet(f"color: {col}; font-weight: 600; padding-right: 8px;")

    def _restore(self) -> None:
        g = self.qs.value("window/geometry")
        try:
            if g is not None:
                self.restoreGeometry(g)
        except Exception:                                                                     # noqa: BLE001
            pass
        screens = [s.availableGeometry() for s in QGuiApplication.screens()]
        if screens and not any(s.intersects(self.frameGeometry()) and s.intersected(self.frameGeometry()).width() > 200 for s in screens):
            scr = QGuiApplication.primaryScreen().availableGeometry()
            self.resize(min(1560, scr.width() - 40), min(960, scr.height() - 60)); self.move(scr.x() + 20, scr.y() + 20)

    def closeEvent(self, e) -> None:
        if self.live._loop is not None and self.live._loop.is_alive():
            if QMessageBox.question(self, "Quit", "The live loop is running. Quitting stops it (open positions stay at the broker). Quit anyway?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        self.live.shutdown()
        self._workers.stop_all(2000)
        self.panel.save(self.qs); self.live.save()
        self.qs.setValue("window/geometry", self.saveGeometry())
        self.qs.sync()
        e.accept()


def run(folder: str) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    w = MRWindow(folder)
    w.show()
    return app.exec()
