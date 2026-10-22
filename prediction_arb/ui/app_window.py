"""Main window: banner, 3-tab navigation, menu and status bar."""
from __future__ import annotations

import os

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QLabel, QMainWindow, QMessageBox, QStatusBar, QTabWidget, QVBoxLayout, QWidget)

from config import APP_NAME, BASE_DIR, DB_PATH
from database import Database
from ui.common import MUTED
from ui.tab_backtest import BacktestTab
from ui.tab_history import HistoryTab
from ui.tab_live import LiveTab
from ui.tab_longshots import LongShotTab


class AppWindow(QMainWindow):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.setWindowTitle(APP_NAME)
        from PyQt6.QtGui import QGuiApplication
        geo = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1720, geo.width() - 30), min(960, geo.height() - 50))

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(8, 6, 8, 4)
        banner = QLabel("READ-ONLY RESEARCH TOOL  -  public endpoints only, no keys, no order placement. "
                        "'Guaranteed' assumes both venues resolve identically: verify the rulebooks before trading.")
        banner.setStyleSheet(f"color:{MUTED};font-size:11px;padding:2px 4px;")
        lay.addWidget(banner)
        self.tabs = QTabWidget()
        self.live = LiveTab(db)
        self.hist = HistoryTab(db)
        self.bt = BacktestTab(db)
        self.ls = LongShotTab(db)
        self.tabs.addTab(self.live, "Live Arbitrage Scanner")
        self.tabs.addTab(self.hist, "Historical Opportunities && PnL")
        self.tabs.addTab(self.bt, "Backtesting Engine")
        self.tabs.addTab(self.ls, "Long-Shot Scanner")
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        for t in (self.live, self.hist, self.bt, self.ls):
            t.status.connect(lambda m: self.statusBar().showMessage(m, 30_000))
        self.live.manual_logged.connect(self.hist.reload)
        self.live.scanned.connect(self.ls.on_live_scan)
        self.tabs.currentChanged.connect(lambda i: self.hist.reload() if i == 1 else None)
        self._menus()
        self.statusBar().showMessage(f"DB: {DB_PATH}   |   {self._db_summary()}")

    def _db_summary(self) -> str:
        s = self.db.stats()
        return f"{s['markets']:,} cached markets, {s['candles']:,} candles, {s['opportunities']:,} logged opps"

    def _menus(self):
        m = self.menuBar().addMenu("&File")
        a = QAction("Open data folder", self)
        a.triggered.connect(lambda: os.startfile(str(BASE_DIR)) if hasattr(os, "startfile") else None)
        m.addAction(a)
        a = QAction("Clear market / candle cache", self)
        a.triggered.connect(self._clear_cache)
        m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("About / methodology", self)
        a.triggered.connect(self._about)
        h.addAction(a)

    def _clear_cache(self):
        if QMessageBox.question(self, "Clear cache", "Delete cached markets, candles and settlements?\n"
                                "(Logged opportunities and positions are kept.)") == QMessageBox.StandardButton.Yes:
            self.db.clear_cache(keep_trades=True)
            self.statusBar().showMessage("Cache cleared. " + self._db_summary())

    def _about(self):
        QMessageBox.information(self, "Methodology", (
            "Hurdle: D = max((T_settle - T_now)/86400, 0.25);  R = (min payout - cost)/cost;  "
            "APY = (1+R)^(365.25/D) - 1  >=  7%.\n\n"
            "Type I: YES(v1)+NO(v2) on equivalent questions (or YES+YES on opposite ones).\n"
            "Type II: all-YES of an exhaustive mutually-exclusive set (pays 1) or all-NO (pays >= n-1).\n"
            "Type III: A implies B - buy YES(B) + NO(A): pays 1, or 2 if B and not A.\n\n"
            "Sanity gates: mid-sum consistency, net-return cap, qualifier/period matching. "
            "Capital is locked until the LAST leg settles."))

    def closeEvent(self, e):
        for t in (self.live, self.hist, self.bt, self.ls):
            w = getattr(t, "worker", None)
            if w is not None and w.isRunning():
                w.cancel()
                w.wait(4000)
        super().closeEvent(e)
