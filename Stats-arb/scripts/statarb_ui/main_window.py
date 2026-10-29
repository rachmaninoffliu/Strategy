"""Main window: shared settings on the left; Live trading / Backtest / Signals & risk / How it works on the right."""
from __future__ import annotations

import json
import os
from typing import Optional

from PyQt6.QtCore import QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QScrollArea, QSplitter, QTabWidget

import statarb_bridge as BR
import statarb_data as D
from supertrend_ui.theme import scrollable
from supertrend_ui.workers import FnWorker, WorkerBag

from . import presets
from .backtest_tab import BacktestTab
from .help_tab import HelpTab
from .live_tab import LiveTab
from .settings_panel import SettingsPanel
from .signals_tab import SignalsTab

TITLE = "Stat-Arb - Residual Mean Reversion - Backtest + Live (Alpaca)"


def app_settings() -> QSettings:
    """STATARB_SETTINGS_FILE selects a throw-away .ini instead of the registry - the tests do, so they can never touch your real settings."""
    f = os.environ.get("STATARB_SETTINGS_FILE")
    return QSettings(f, QSettings.Format.IniFormat) if f else QSettings("AlphaSyndicate", "StatArbTrader")


def reports_dir() -> str:
    os.makedirs(D.REPORTS, exist_ok=True)
    return D.REPORTS


class MainWindow(QMainWindow):
    TAB_LIVE, TAB_BT, TAB_SIG, TAB_HELP = range(4)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1500, 920)
        self.qs = app_settings()
        self._workers = WorkerBag()
        self.settings = SettingsPanel()
        self.settings.load(self.qs)
        self.settings.btn_update.clicked.connect(self.update_data)
        self.backtest = BacktestTab(self.settings, self.qs, reports_dir)
        self.live = LiveTab(self.settings, self.qs, reports_dir)
        self.signals = SignalsTab(self.settings, self.qs)
        self.help = HelpTab()
        self.backtest.load(); self.live.load()
        side = QScrollArea(); side.setWidgetResizable(True); side.setWidget(self.settings); side.setMinimumWidth(330)
        tabs = QTabWidget()
        tabs.addTab(scrollable(self.live, 900, 900), "Live trading")
        tabs.addTab(scrollable(self.backtest, 900, 760), "Backtest")
        tabs.addTab(scrollable(self.signals, 900, 640), "Signals & risk")
        tabs.addTab(self.help, "How it works")
        self.main_tabs = tabs
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(side); self.splitter.addWidget(tabs)
        self.splitter.setStretchFactor(0, 0); self.splitter.setStretchFactor(1, 1); self.splitter.setSizes([430, 1070]); self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)
        self.state_lbl = QLabel("")
        self.statusBar().addPermanentWidget(self.state_lbl)
        self._menus()
        self._restore()
        self._data_label()
        t = QTimer(self); t.timeout.connect(self._status); t.start(1000)
        self._t = t

    # ── menus ────────────────────────────────────────────────────────────────────
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
        self._act(m, "Open the reports folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(reports_dir())))
        self._act(m, "Quit", self.close, "Ctrl+Q")
        m = mb.addMenu("&Run")
        self._act(m, "Run backtest", lambda: (self.main_tabs.setCurrentIndex(self.TAB_BT), self.backtest.run()), "Ctrl+R")
        self._act(m, "Plan next session (dry run)", lambda: (self.main_tabs.setCurrentIndex(self.TAB_LIVE), self.live.make_plan(False)), "Ctrl+L")
        self._act(m, "Refresh signals", lambda: (self.main_tabs.setCurrentIndex(self.TAB_SIG), self.signals.refresh()))
        self._act(m, "Update data", self.update_data)
        m = mb.addMenu("&View")
        for i, name in enumerate(("Live trading", "Backtest", "Signals & risk", "How it works")):
            self._act(m, name, lambda i=i: self.main_tabs.setCurrentIndex(i), f"Ctrl+{i + 1}")

    def save_settings_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", os.path.join(reports_dir(), "statarb_settings.json"), "JSON (*.json)")
        if path:
            json.dump({"preset": self.settings.preset_name(), "values": self.settings.values()}, open(path, "w", encoding="utf-8"), indent=2)
            self.statusBar().showMessage(f"saved {path}", 5000)

    def load_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", reports_dir(), "JSON (*.json)")
        if path:
            try:
                self.settings.set_values(json.load(open(path, encoding="utf-8"))["values"])
            except Exception as e:                                                     # noqa: BLE001
                QMessageBox.warning(self, "Load settings", f"Could not read that file: {e}")

    # ── data ─────────────────────────────────────────────────────────────────────
    def _data_label(self) -> None:
        try:
            c = D.load_cache()
            e = D.load_earnings()
            self.settings.data_lbl.setText(f"Core cache: {c.symbol.nunique()} symbols through {c.date.max():%Y-%m-%d} · earnings calendar: {e.symbol.nunique() if len(e) else 0} symbols" if len(c) else
                                           "No data yet - press 'Update data'.")
        except Exception as e:                                                         # noqa: BLE001
            self.settings.data_lbl.setText(f"⚠ {e}")

    def update_data(self) -> None:
        if self._workers.busy():
            return
        self.settings.btn_update.setEnabled(False)
        uni = self.settings.universe()

        def fn(log):
            D.update_cache(log=log)                                                    # core universe + the factor ETFs
            if uni == "broad":
                import statarb_broad as B
                B.update_broad_cache(log=log)
                D.update_earnings(list(B.load_broad_panel().stocks), log=log)
            else:
                D.update_earnings(log=log)
            return True
        w = FnWorker(fn)
        self._workers.start(w, self._updated, lambda s: self.statusBar().showMessage(s.strip()[:120]))

    def _updated(self, res, err) -> None:
        self.settings.btn_update.setEnabled(True)
        self.statusBar().showMessage("Data update failed: " + err if err else "Data updated", 8000)
        self.backtest._panels.clear()
        self._data_label()

    # ── status / layout ──────────────────────────────────────────────────────────
    def _status(self) -> None:
        lv = self.live
        armed = lv.armed.isChecked()
        if armed:
            txt, col = (f"● ARMED - {'PAPER' if lv.paper() else 'LIVE'}", "#e05561" if not lv.paper() else "#d9a520")
        else:
            txt, col = ("DRY RUN - nothing is sent", "#8b98a9")
        self.state_lbl.setText(txt)
        self.state_lbl.setStyleSheet(f"color: {col}; font-weight: 600; padding-right: 8px;")

    def _restore(self) -> None:
        g = self.qs.value("window/geometry")
        try:
            if g is not None:
                self.restoreGeometry(g)
        except Exception:                                                              # noqa: BLE001
            pass
        screens = [s.availableGeometry() for s in QGuiApplication.screens()]
        if screens and not any(s.intersects(self.frameGeometry()) and s.intersected(self.frameGeometry()).width() > 200 for s in screens):
            scr = QGuiApplication.primaryScreen().availableGeometry()
            self.resize(min(1500, scr.width() - 40), min(920, scr.height() - 60)); self.move(scr.x() + 20, scr.y() + 20)
        try:
            sizes = [int(x) for x in self.qs.value("window/splitter", [])]
            if len(sizes) == 2 and min(sizes) > 100:
                self.splitter.setSizes(sizes)
        except (TypeError, ValueError):
            pass

    def closeEvent(self, e) -> None:
        if self.live.armed.isChecked():
            if QMessageBox.question(self, "Quit", "The engine is ARMED. Quitting does not cancel orders already queued at the broker. Quit anyway?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        self._workers.stop_all(2000)
        self.settings.save(self.qs); self.backtest.save(); self.live.save()
        self.qs.setValue("window/geometry", self.saveGeometry()); self.qs.setValue("window/splitter", self.splitter.sizes())
        self.qs.sync()
        e.accept()
