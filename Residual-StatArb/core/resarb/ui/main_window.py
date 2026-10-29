"""Main window: the strategy's settings on the left; Live trading / Backtest / Research / Signals / How it works on the right. One window class serves all three strategies."""
from __future__ import annotations

import json
import os

import pandas as pd
from PyQt6.QtCore import QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QScrollArea, QSplitter, QTabWidget

from .. import data as D
from .backtest_tab import BacktestTab
from .help_tab import HelpTab
from .live_tab import LiveTab
from .research_tab import ResearchTab
from .settings_panel import SettingsPanel
from .signals_tab import SignalsTab
from .theme import scrollable
from .workers import FnWorker, WorkerBag


def app_settings(key: str) -> QSettings:
    """RESARB_SETTINGS_FILE selects a throw-away .ini instead of the registry (the tests use it so they never touch your settings)."""
    f = os.environ.get("RESARB_SETTINGS_FILE")
    return QSettings(f, QSettings.Format.IniFormat) if f else QSettings("AlphaSyndicate", f"ResArb-{key}")


class MainWindow(QMainWindow):
    TAB_LIVE, TAB_BT, TAB_RES, TAB_SIG, TAB_HELP = range(5)

    def __init__(self, strat_cls: type, folder: str):
        super().__init__()
        self.cls, self.folder = strat_cls, folder
        self.setWindowTitle(f"{strat_cls.NAME} - residual stat-arb - Backtest + Live (Alpaca)")
        self.resize(1500, 940)
        self.qs = app_settings(strat_cls.KEY)
        self._workers = WorkerBag()
        self.settings = SettingsPanel(strat_cls)
        self.settings.load(self.qs)
        self.settings.btn_update.clicked.connect(self.update_data)
        self.live = LiveTab(self.settings, self.qs, folder)
        self.backtest = BacktestTab(self.settings, self.qs, folder)
        self.research = ResearchTab(self.settings, folder)
        self.signals = SignalsTab(self.settings, folder)
        self.help = HelpTab(strat_cls, folder)
        self.live.load()
        side = QScrollArea(); side.setWidgetResizable(True); side.setWidget(self.settings); side.setMinimumWidth(340)
        tabs = QTabWidget()
        tabs.addTab(scrollable(self.live, 900, 860), "Live trading")
        tabs.addTab(scrollable(self.backtest, 900, 760), "Backtest")
        tabs.addTab(scrollable(self.research, 900, 640), "Research")
        tabs.addTab(scrollable(self.signals, 900, 600), "Signals")
        tabs.addTab(self.help, "How it works")
        self.main_tabs = tabs
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(side); self.splitter.addWidget(tabs)
        self.splitter.setStretchFactor(0, 0); self.splitter.setStretchFactor(1, 1); self.splitter.setSizes([440, 1060]); self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)
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
        self._act(m, "Save settings to file…", self.save_settings_file, "Ctrl+S")
        self._act(m, "Load settings from file…", self.load_settings_file, "Ctrl+O")
        m.addSeparator()
        self._act(m, "Open the strategy folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.folder)))
        self._act(m, "Quit", self.close, "Ctrl+Q")
        m = mb.addMenu("&Run")
        self._act(m, "Run backtest", lambda: (self.main_tabs.setCurrentIndex(self.TAB_BT), self.backtest.run()), "Ctrl+R")
        self._act(m, "Plan this bar (dry run)", lambda: (self.main_tabs.setCurrentIndex(self.TAB_LIVE), self.live.plan_now()), "Ctrl+L")
        self._act(m, "Update data", self.update_data)
        m = mb.addMenu("&View")
        for i, name in enumerate(("Live trading", "Backtest", "Research", "Signals", "How it works")):
            self._act(m, name, lambda i=i: self.main_tabs.setCurrentIndex(i), f"Ctrl+{i + 1}")

    def save_settings_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", os.path.join(self.folder, f"{self.cls.KEY}_settings.json"), "JSON (*.json)")
        if path:
            json.dump({"strategy": self.cls.KEY, "values": self.settings.values()}, open(path, "w", encoding="utf-8"), indent=2)
            self.statusBar().showMessage(f"saved {path}", 5000)

    def load_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", self.folder, "JSON (*.json)")
        if path:
            try:
                self.settings.set_values(json.load(open(path, encoding="utf-8"))["values"])
            except Exception as e:                                                         # noqa: BLE001
                QMessageBox.warning(self, "Load settings", f"Could not read that file: {e}")

    def _data_label(self) -> None:
        try:
            self.settings.data_lbl.setText(D.status())
        except Exception as e:                                                             # noqa: BLE001
            self.settings.data_lbl.setText(f"⚠ {e}")

    def update_data(self) -> None:
        if self._workers.busy():
            return
        self.settings.btn_update.setEnabled(False)
        n = int(self.settings.values()["universe_n"])

        def fn(log):
            D.update_universe(log=log)
            u = D.load_universe()
            recent = sorted(set(u[(u.month >= u.month.max() - pd.DateOffset(months=13)) & (u["rank"] <= max(n, 100))].symbol))
            D.update_bars(list(D.ETFS) + recent, "sip", log=log)
            return True
        self._workers.start(FnWorker(fn), self._updated, lambda s: self.statusBar().showMessage(s.strip()[:120]))

    def _updated(self, res, err) -> None:
        self.settings.btn_update.setEnabled(True)
        self.statusBar().showMessage("Data update failed: " + err if err else "Data updated", 8000)
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
        except Exception:                                                                  # noqa: BLE001
            pass
        screens = [s.availableGeometry() for s in QGuiApplication.screens()]
        if screens and not any(s.intersects(self.frameGeometry()) and s.intersected(self.frameGeometry()).width() > 200 for s in screens):
            scr = QGuiApplication.primaryScreen().availableGeometry()
            self.resize(min(1500, scr.width() - 40), min(940, scr.height() - 60)); self.move(scr.x() + 20, scr.y() + 20)

    def closeEvent(self, e) -> None:
        if self.live._loop is not None and self.live._loop.is_alive():
            if QMessageBox.question(self, "Quit", "The live loop is running. Quitting stops it (open positions stay at the broker). Quit anyway?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        self.live.shutdown()
        self._workers.stop_all(2000)
        self.settings.save(self.qs); self.live.save()
        self.qs.setValue("window/geometry", self.saveGeometry())
        self.qs.sync()
        e.accept()
