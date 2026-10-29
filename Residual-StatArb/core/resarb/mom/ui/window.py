"""Main window of the momentum-book app: the model picker on the left; Backtest / Live trading / How it works on the right."""
from __future__ import annotations

import html
import os
import sys

import pandas as pd
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QScrollArea, QSplitter, QTabWidget, QTextBrowser

from ... import data as D
from ...mr import engine as MRE
from ...mr.market import load_daily, update_daily
from ...ui.main_window import app_settings
from ...ui.theme import apply_theme, scrollable
from ...ui.workers import FnWorker, WorkerBag
from .backtest_tab import BacktestTab
from .live_tab import LiveTab
from .model_panel import ModelPanel

KEY = "MomentumBook"

HELP = """
<h2>Momentum book - backtest + live (Alpaca)</h2>
<p>Two momentum effects on the <b>same capital</b>, never held at the same time, so the book is never above 1x equity:</p>
<ul><li><b>Overnight leg</b> (Lou, Polk &amp; Skouras 2019; Berkman et al. 2012): stocks that rose during the day tend to keep rising overnight. At 15:30 the top-150 stocks
and 59 ETFs are ranked; the 20 best are bought in the <b>closing auction</b> (market-on-close) and sold in the next <b>opening auction</b> (market-on-open). Only when SPY's
last close is above its 200-day mean.</li>
<li><b>Intraday leg</b> (Zarattini, Aziz &amp; Barbon 2024, "Beat the Market"): QQQ is bought when a 30-minute bar closes above its <i>noise band</i> - the day's open (or
yesterday's close) widened by the average move of the last 14 sessions at that time of day - and sold when it falls back below the band or the session VWAP, else in the
closing auction.</li></ul>
<h3>Ranking - which names get the capital</h3>
<p><i>rule</i>: today's move to the decision time (last price / yesterday's close). <i>rule+ml</i>: the percentile rank of that move plus the percentile rank of a
gradient-boosting forecast of the overnight return (36 inputs: the day's move and volume, past overnight and intraday returns, momentum, volatility, beta, the market's
state). One model per year, fitted only on the years before it (walk-forward), saved in <code>models/</code>: the backtest predicts each year with that year's file and live
loads the current year's file - the same object. Ties are broken by ticker. Each name gets (1 - cash reserve) / 20 of the capital in whole shares.</p>
<h3>Timing - why the results are honest</h3>
<p>The overnight decision uses only the 15-minute SIP bars that ended by 15:30 (available at 15:46 on the free, 16-minute-delayed feed) and earlier official closes; the
orders are market-on-close (before Alpaca's 15:50 cut-off). The intraday leg reads each 30-minute bar from the <b>IEX</b> feed (free and real time) a few seconds after it
ends; the backtest uses the cached IEX bars (from 2024; the SIP bars stand in before) and fills at that bar's close plus the half spread. On early-close days everything moves
with the close. Costs: 1 bp per auction side, 0.5 bp half spread intraday.</p>
<h3>Backtest = live</h3>
<p>The backtest and the live engine call the same functions: MomMarket, prepare() (features, forecast), decide_overnight(), noise_upper() / noise_action(), and the same
sizing. The tests replay backtests session by session through the live engine with a simulated broker - every opening-auction sale, intraday order and closing-auction buy must
equal a backtest trade and the final equity must match. In live, 'Check live vs backtest' rebuilds each journaled decision and check from the backtest's data.</p>
<h3>Shadow</h3>
<p>The live tab can run a second book in the shadow (default: the rule-only book): its overnight picks are computed on the same data at every decision and journaled, never
traded. 'Shadow report' compares the nights of both from the daily bars.</p>
<h3>Daily cycle (live)</h3>
<ul><li>08:45 - top up the daily and 15-minute bars; last night's names get market-on-open sells.</li>
<li>10:00:05, 10:30:05 ... 15:30:05 - the intraday checks (market orders on QQQ).</li>
<li>15:36 - today's universe bars are read ahead; 15:46:06 - QQQ (if held) is sold market-on-close, the overnight names are bought market-on-close.</li>
<li>16:05 - fills journaled, the overnight book rebuilt from the broker's positions, today's intraday round trips journaled.</li></ul>
<h3>Safety</h3>
<ul><li>Starts on PAPER and DRY RUN; ARM needs the typed phrase <code>ARM PAPER</code>; real money is disabled in code until Valid signs off.</li>
<li>Keys: the project's <code>Alpaca Kay.txt</code>; <code>MB_ALPACA_API_KEY</code> / <code>MB_ALPACA_SECRET_KEY</code> select a separate paper account. An account another strategy
(MeanReversion, ResMomentum, ...) has claimed is refused - two strategies on one account would close each other's positions.</li>
<li>The intraday leg day-trades: the pattern-day-trader rule needs a margin account with at least $25,000 (pre-flight checks it).</li>
<li>Positions in symbols the book can never trade are left alone. KILL cancels and flattens. Optional live-only overlays: daily-loss halt, drawdown stop.</li></ul>
<h3>Papers</h3>
<ul><li>Lou, Polk &amp; Skouras (2019), A tug of war: overnight versus intraday expected returns, JFE 134(1)</li>
<li>Berkman, Koch, Tuttle &amp; Zhang (2012), Paying attention: overnight returns and the hidden cost of buying at the open, JFQA 47(4)</li>
<li>Zarattini, Aziz &amp; Barbon (2024), Beat the market: an effective intraday momentum strategy for S&amp;P 500 ETF (SPY), SSRN 4824172</li>
<li>Gu, Kelly &amp; Xiu (2020), Empirical asset pricing via machine learning, RFS 33(5)</li></ul>
"""


class MomWindow(QMainWindow):
    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        self.setWindowTitle("Momentum Book - overnight momentum + QQQ intraday - Backtest + Live (Alpaca) - no leverage")
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
        hb.setHtml(HELP + f"<p>Results, the team review and the verdict: <code>{html.escape(rep)}</code></p>")
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
        self._act(m, "Train ML models", lambda: (self.main_tabs.setCurrentIndex(0), self.backtest.train()))
        self._act(m, "Plan overnight now (dry run)", lambda: (self.main_tabs.setCurrentIndex(1), self.live.plan_now()), "Ctrl+L")
        self._act(m, "Update data", self.update_data)

    def _data_label(self) -> None:
        try:
            d = load_daily(["SPY"])
            u = D.load_universe()
            b = D.load_bars("QQQ", "iex")
            self.panel.data_lbl.setText(f"daily bars through {d.date.max():%Y-%m-%d}" + (f" · stock universe through {u.month.max():%Y-%m}" if len(u) else "")
                                        + (f" · QQQ IEX bars {b.index.min():%Y-%m} .. {b.index.max():%Y-%m-%d}" if len(b) else " · no QQQ IEX bars (Update data)"))
        except Exception as e:                                                                # noqa: BLE001
            self.panel.data_lbl.setText(f"⚠ {e}")

    def update_data(self) -> None:
        if self._workers.busy():
            return
        self.panel.btn_update.setEnabled(False)
        m = self.panel.model()

        def fn(log):
            syms: list = []
            for part in [p.strip() for p in m.universe.split("+") if p.strip()]:
                if part.startswith("top"):
                    D.update_universe(log=log)
                    u = D.load_universe()
                    syms += sorted(set(u[(u.month >= u.month.max() - pd.DateOffset(months=13)) & (u["rank"] <= int(part[3:] or 100))].symbol))
                else:                                                                         # (delisted members keep their cached history)
                    syms += MRE._part_symbols(part)
            syms = list(dict.fromkeys(syms + ["SPY"] + ([m.intraday_symbol] if m.intraday_symbol else [])))
            update_daily(syms, log=log)
            D.update_bars(syms, "sip", log=log)
            if m.intraday_symbol and m.intraday_feed == "iex":
                D.update_bars([m.intraday_symbol], "iex", start="2024-01-01", log=log)
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
    w = MomWindow(folder)
    w.show()
    return app.exec()
