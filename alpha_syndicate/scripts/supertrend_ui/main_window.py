"""Main window: the shared settings panel on the left, Live trading / Backtest / Signal monitor / How it works on the right."""
from __future__ import annotations

import json
import os
from typing import Optional

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QScrollArea, QSplitter, QTabWidget

import supertrend_gui as GU

from . import data as D
from . import params as P
from .backtest_tab import BacktestTab
from .help_tab import HelpTab
from .live_tab import LiveTab
from .monitor_tab import MonitorTab
from .settings_panel import SettingsPanel
from .theme import scrollable
from .workers import FnWorker, WorkerBag

TITLE = "Supertrend SOXL / SOXS — Backtest + Live"


def app_settings() -> QSettings:
    """The app's saved settings (Windows registry by default). SUPERTREND_SETTINGS_FILE selects a throw-away .ini instead - the tests do, so they can never touch your real settings."""
    f = os.environ.get("SUPERTREND_SETTINGS_FILE")
    return QSettings(f, QSettings.Format.IniFormat) if f else QSettings("AlphaSyndicate", "SupertrendTrader")


def reports_dir() -> str:
    os.makedirs(GU.REPORTS, exist_ok=True)
    return GU.REPORTS


class MainWindow(QMainWindow):
    TAB_LIVE, TAB_BT, TAB_MON, TAB_HELP = range(4)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1480, 900)
        self.qs = app_settings()
        self.auto_update = os.environ.get("SUPERTREND_NO_AUTO_UPDATE") != "1"          # tests switch it off: the window must not hit the network by itself
        self._workers = WorkerBag()
        self.settings = SettingsPanel()
        self.settings.load(self.qs)
        self.settings.btn_update.clicked.connect(self.update_data)
        self.backtest = BacktestTab(self.settings, self.qs, reports_dir)
        self.live = LiveTab(self.settings, self.qs, reports_dir)
        self.monitor = MonitorTab(self.settings, self.qs)
        self.monitor.auto_load = self.auto_update
        self.help = HelpTab()
        self.backtest.load(); self.live.load()
        self.live.get_bt_settings = self.backtest.safe_settings
        for pv in (self.backtest.params_view, self.live.params_view):
            pv.get_state = self.live.params_state
        self.settings.changed.connect(lambda: (self.backtest.params_view.refresh(), self.live.params_view.refresh()))
        self.backtest.d0.dateChanged.connect(lambda *_: self.live.params_view.refresh())

        self.settings.setMinimumWidth(400)
        side = QScrollArea(); side.setWidgetResizable(True); side.setWidget(self.settings); side.setMinimumWidth(300)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded); side.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.side = side
        tabs = QTabWidget()
        tabs.addTab(scrollable(self.live, 900, 960), "Live trading")
        tabs.addTab(scrollable(self.backtest, 900, 780), "Backtest")
        tabs.addTab(scrollable(self.monitor, 900, 640), "Signal monitor")
        tabs.addTab(self.help, "How it works")
        self.main_tabs = tabs
        self.splitter = QSplitter(Qt.Orientation.Horizontal)                      # drag the handle to give the settings (or the tabs) more room
        self.splitter.addWidget(side); self.splitter.addWidget(tabs)
        self.splitter.setStretchFactor(0, 0); self.splitter.setStretchFactor(1, 1); self.splitter.setSizes([430, 1050]); self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        self.state_lbl = QLabel("")
        self.statusBar().addPermanentWidget(self.state_lbl)
        self._menus()
        self._restore_layout()
        self.reload_data()
        self._status_timer = QTimer(self); self._status_timer.timeout.connect(self._status); self._status_timer.start(1000)
        QTimer.singleShot(1500, lambda: self.ensure_fresh() if self.auto_update else None)

    # ══ menus and shortcuts ══════════════════════════════════════════════════════
    def _act(self, menu, text: str, fn, shortcut: Optional[str] = None, tip: str = "") -> QAction:
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setStatusTip(tip)
        a.triggered.connect(lambda *_: fn())
        menu.addAction(a)
        self.addAction(a)
        return a

    def _menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        self._act(m, "Save settings to file…", self.save_settings_file, "Ctrl+S", "Save every strategy input to a .json file you can load again")
        self._act(m, "Load settings from file…", self.load_settings_file, "Ctrl+O")
        m.addSeparator()
        self._act(m, "Export the current trades to CSV…", self.export_trades, "Ctrl+E")
        self._act(m, "Open the reports folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(reports_dir())))
        self._act(m, "Open the live log file", self.live.open_log_file)
        m.addSeparator()
        self._act(m, "Quit", self.close, "Ctrl+Q")
        m = mb.addMenu("&Run")
        self._act(m, "Run backtest", self.run_backtest, "Ctrl+R")
        self._act(m, "Run backtest (F5)", self.run_backtest, "F5")
        self._act(m, "Robustness check", lambda: (self.main_tabs.setCurrentIndex(self.TAB_BT), self.backtest.run_robustness()))
        self._act(m, "Pre-flight check", lambda: (self.main_tabs.setCurrentIndex(self.TAB_LIVE), self.live.run_preflight(then_start=False)), "Ctrl+L")
        self._act(m, "Refresh the signal monitor", lambda: (self.main_tabs.setCurrentIndex(self.TAB_MON), self.monitor.refresh()))
        self._act(m, "Update data", self.update_data)
        m = mb.addMenu("&View")
        for i, name in enumerate(("Live trading", "Backtest", "Signal monitor", "How it works")):
            self._act(m, name, lambda i=i: self.main_tabs.setCurrentIndex(i), f"Ctrl+{i + 1}")
        m = mb.addMenu("&Help")
        self._act(m, "How it works", lambda: self.main_tabs.setCurrentIndex(self.TAB_HELP), "F1")
        self._act(m, "About", self.about)

    def about(self) -> None:
        QMessageBox.about(self, "About", "<b>Supertrend SOXL / SOXS</b><br>Backtest, signal monitor and Alpaca trading for the Pine strategy port.<br><br>"
                                         "Always starts on Paper + DRY RUN. " + P.VALID_NOTE)

    def run_backtest(self) -> None:
        self.main_tabs.setCurrentIndex(self.TAB_BT)
        self.backtest.run()

    def current_perf(self):
        return self.live.perf if self.main_tabs.currentIndex() == self.TAB_LIVE else self.backtest.perf

    def export_trades(self) -> None:
        self.current_perf().export()

    def save_settings_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", os.path.join(reports_dir(), "supertrend_settings.json"), "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"preset": self.settings.preset_name(), "values": self.settings.values()}, fh, indent=2)
            self.statusBar().showMessage(f"settings saved to {path}", 5000)

    def load_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", reports_dir(), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                self.settings.set_values(json.load(fh)["values"])
            self.statusBar().showMessage(f"settings loaded from {path}", 5000)
        except Exception as e:                                    # noqa: BLE001
            QMessageBox.warning(self, "Load settings", f"Could not read that file: {e}")

    # ══ data ═════════════════════════════════════════════════════════════════════
    def reload_data(self) -> None:
        try:
            bars = D.read_cache("SOXL", 15)
            self.backtest.adopt_bars("SOXL", 15, bars)
            self.settings.data_lbl.setText(f"Local data: {D.describe_bars(bars)}")
            self.statusBar().showMessage(f"Data through {bars.index[-1]:%Y-%m-%d %H:%M} ET")
        except FileNotFoundError as e:
            self.settings.data_lbl.setText(f"⚠ {e}")
        except Exception as e:                                    # noqa: BLE001
            self.settings.data_lbl.setText(f"⚠ could not load local data: {e}")

    def ensure_fresh(self) -> None:
        """Top the bar cache up in the background when it is behind (never blocks the window)."""
        bars = self.backtest._bars.get(("SOXL", 15))
        last = bars.index[-1].to_pydatetime() if bars is not None and len(bars) else None
        if self._workers.busy() or not D.needs_refresh(last, D.now_et(), self.backtest._last_refresh_try):
            return
        self.update_data(quiet=True)

    def update_data(self, quiet: bool = False) -> None:
        import time
        self.settings.btn_update.setEnabled(False)
        self.backtest._last_refresh_try = time.time()
        if not quiet:
            self.live.add_log("updating the bar cache from Alpaca (read-only data request) …")
        self.statusBar().showMessage("Updating bars from Alpaca …")
        w = FnWorker(lambda log: D.load_bars("SOXL", 15, refresh=True, log=log))
        self._workers.start(w, self._updated, lambda s: self.statusBar().showMessage(s.strip()[:120]))

    def _updated(self, bars, err) -> None:
        self.settings.btn_update.setEnabled(True)
        if err or bars is None:
            self.live.add_log(f"data update problem: {err}", "warn")
            self.statusBar().showMessage("Data update failed - using the cached bars", 8000)
            return
        self.backtest.adopt_bars("SOXL", 15, bars)
        self.settings.data_lbl.setText(f"Local data: {D.describe_bars(bars)}")
        self.live.add_log(f"data updated: {D.describe_bars(bars)}", "ok")
        self.statusBar().showMessage(f"Data through {bars.index[-1]:%Y-%m-%d %H:%M} ET", 8000)

    # ══ status, layout, close ════════════════════════════════════════════════════
    def _status(self) -> None:
        lv = self.live
        if lv.running and lv._run:
            ls = lv._run["ls"]
            txt = f"● SESSION RUNNING — {self.live.mode_label(ls)}"
            col = "#e05561" if (ls.armed and not ls.paper) else "#2ea06b" if ls.armed else "#8b98a9"
        else:
            txt, col = "no session running", "#8b98a9"
        self.state_lbl.setText(txt)
        self.state_lbl.setStyleSheet(f"color: {col}; font-weight: 600; padding-right: 8px;")

    def _restore_layout(self) -> None:
        g = self.qs.value("window/geometry")
        try:
            if g is not None:
                self.restoreGeometry(g)
        except Exception:                                         # noqa: BLE001
            pass
        screens = [s.availableGeometry() for s in QGuiApplication.screens()]
        if screens and not any(s.intersects(self.frameGeometry()) and s.intersected(self.frameGeometry()).width() > 200 for s in screens):
            scr = QGuiApplication.primaryScreen().availableGeometry()                 # a saved window that is now off-screen (a monitor was unplugged) is brought back
            self.resize(min(1480, scr.width() - 40), min(900, scr.height() - 60))
            self.move(scr.x() + 20, scr.y() + 20)
        try:
            sizes = [int(x) for x in self.qs.value("window/splitter", [])]
            if len(sizes) == 2 and min(sizes) > 100:
                self.splitter.setSizes(sizes)
            t = int(self.qs.value("window/tab", 0))
            if 0 <= t < self.main_tabs.count():
                self.main_tabs.setCurrentIndex(t)
        except (TypeError, ValueError):
            pass

    def closeEvent(self, e) -> None:
        if self.live.running:
            if QMessageBox.question(self, "Quit", "A session is running. Stop it and quit?\n\nOpen positions stay open (SOXL's broker-side exits keep working; SOXS exits are managed by this app only).") \
                    != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            if self.live.engine is not None:
                self.live.engine.stop()
                self.live.engine.join(3)
            self.live._finish_session("window closed")
        self._workers.stop_all(2000)
        self.settings.save(self.qs); self.backtest.save(); self.live.save()
        self.qs.setValue("window/geometry", self.saveGeometry()); self.qs.setValue("window/splitter", self.splitter.sizes()); self.qs.setValue("window/tab", self.main_tabs.currentIndex())
        self.qs.sync()
        e.accept()
