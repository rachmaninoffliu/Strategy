"""Signal monitor: what the strategy sees RIGHT NOW (price bars, Supertrend, re-entry limit, the open position's levels) - read-only, nothing is traded here."""
from __future__ import annotations

import json
import queue
from datetime import datetime
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

import supertrend_gui as GU
import supertrend_strategy as S

from . import data as D
from .chart import SignalChart
from .perf_view import side_color
from .settings_panel import SettingsPanel
from .theme import ACCENT, GREEN, MUTED, RED, YELLOW, QComboBox, make_table, muted, set_row
from .workers import FnWorker, WorkerBag

SHOW = (("1 day", 26), ("3 days", 78), ("5 days", 130), ("10 days", 260), ("20 days", 520))


class MonitorTab(QWidget):
    def __init__(self, settings: SettingsPanel, qs: QSettings):
        super().__init__()
        self.sp, self.qs = settings, qs
        self.bars: Optional[pd.DataFrame] = None
        self.auto_load = True                                      # load the bars the first time the tab is shown (the tests switch this off)
        self._workers = WorkerBag()
        v = QVBoxLayout(self)
        v.addWidget(muted("Signal monitor — the same Supertrend the live engine computes, on the latest bars. Green line = uptrend (long SOXL), red line = downtrend (short). The shaded bar is the last CLOSED bar: "
                          "entries are decided on it; the bar after it may still be forming. Nothing is traded here."))
        row = QHBoxLayout()
        self.btn = QPushButton("Refresh"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.refresh)
        self.show_n = QComboBox(); self.show_n.addItems([s for s, _ in SHOW]); self.show_n.setCurrentIndex(2)
        self.show_n.currentIndexChanged.connect(lambda *_: self.redraw())
        self.auto = QCheckBox("auto-refresh every minute while visible"); self.auto.setChecked(False)
        self.summary = QLabel("—"); self.summary.setWordWrap(True)
        row.addWidget(self.btn); row.addWidget(QLabel("Show")); row.addWidget(self.show_n); row.addWidget(self.auto); row.addWidget(self.summary, 1)
        v.addLayout(row)
        self.chart = SignalChart()
        v.addWidget(self.chart, 3)
        self.readout = QLabel(""); self.readout.setObjectName("muted"); self.readout.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        v.addWidget(self.readout)
        self.chart.readout.connect(self.readout.setText)
        self.table = make_table(["What", "Value", "Meaning"])
        self.table.setWordWrap(True)
        v.addWidget(self.table, 2)
        self.timer = QTimer(self); self.timer.timeout.connect(self._auto); self.timer.start(60_000)
        self.sp.changed.connect(self.redraw)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if self.bars is None and self.auto_load:
            self.refresh()

    def _auto(self) -> None:
        if self.auto.isChecked() and self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        if self._workers.busy():
            return
        self.btn.setEnabled(False)
        self.summary.setText("Loading the latest bars …")

        def work(log):
            k, s = GU.creds(True)                                  # market data only: the paper keys are enough and cannot trade real money
            eng = GU.Engine(queue.Queue(), True, 1.0, 1.0, datetime.now().strftime("%Y-%m-%d"), False)
            eng.dclient = GU.StockHistoricalDataClient(k, s)
            return eng.fetch_bars()                                # the live engine's own fetch: the monitor sees exactly what it sees

        self._workers.start(FnWorker(work), self._loaded)

    def _loaded(self, bars, err) -> None:
        self.btn.setEnabled(True)
        if err:
            self.summary.setText(f"⚠ could not load bars: {err}")
            self.summary.setStyleSheet(f"color: {RED};")
            if self.bars is None:
                self.chart.message(f"Could not load bars\n{err[:150]}")
            return
        self.bars = bars
        self.redraw()

    def redraw(self) -> None:
        if self.bars is None or not len(self.bars):
            return
        try:
            cfg = self.sp.to_config()
        except Exception as e:                                    # noqa: BLE001
            self.summary.setText(f"settings problem: {e}")
            return
        now = D.now_et()
        sig = D.compute_signal(self.bars, cfg, now)
        if sig is None:
            self.summary.setText(f"Warming up: {len(self.bars)} bars loaded, the ATR period is {cfg.atrPeriod}.")
            self.summary.setStyleSheet(f"color: {YELLOW};")
            self.chart.message(f"Warming up ({len(self.bars)} bars, the ATR period needs {cfg.atrPeriod})")
            return
        try:
            with open(GU.STATE_FILE, encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception:                                         # noqa: BLE001
            state = {}
        pos = D.position_levels(state, cfg, sig.get("adx"))
        n = dict(SHOW)[self.show_n.currentText()]
        tail = self.bars.iloc[-n:]
        up = sig["want"] > 0
        levels = [("re-entry limit", sig["limit"], ACCENT, "--")]
        if pos:
            levels += [("entry ref", pos["entry_ref"], MUTED, ":"), ("take-profit", pos["tp"], GREEN, "-"), ("run-up target", pos["target"], YELLOW, "--")]
            if pos["trail"] is not None:
                levels.append(("trailing stop", pos["trail"], RED, "-"))
        self.chart.plot(tail, sig["st_series"][-len(tail):], sig["dir_series"][-len(tail):], levels, sig["bar"])
        self.summary.setText(f"{'UP → long SOXL' if up else 'DOWN → short SOXL'} on the {sig['bar']:%a %H:%M} bar · close {sig['close']:,.2f} · Supertrend {sig['st']:,.2f} · "
                             f"data through {self.bars.index[-1]:%a %d %b %H:%M} ET" + (" (forming bar included)" if sig["forming"] else ""))
        self.summary.setStyleSheet(f"color: {GREEN if up else RED}; font-weight: 600;")
        self._table(sig, pos, cfg, state)

    def _table(self, sig: dict, pos: Optional[dict], cfg: S.Config, state: dict) -> None:
        up = sig["want"] > 0
        dist = (sig["close"] / sig["st"] - 1) * 100
        reached = (sig["mon_low"] <= sig["limit"]) if up else (sig["mon_high"] >= sig["limit"])
        rows = [("Last closed bar", f"{sig['bar']:%Y-%m-%d %H:%M} · close {sig['close']:,.2f}", "Entries are decided on this bar only (a still-forming bar can repaint)."),
                ("Trend", "UP" if up else "DOWN", "Long SOXL while the close is above the Supertrend line, short below it." ),
                ("Supertrend line", f"{sig['st']:,.2f}  ({dist:+.1f}% from the close)", "The ratcheting band: it only moves with the trend until price closes through it."),
                ("Re-entry limit", f"{sig['limit']:,.2f}", f"After an exit the strategy waits for SOXL to {'dip to' if up else 'rally to'} Supertrend {'+' if up else '-'} {cfg.buyback * 100:g}% before re-entering the same trend."),
                ("Monitor bar range", f"{sig['mon_low']:,.2f} – {sig['mon_high']:,.2f}" + (" (forming)" if sig["forming"] else ""),
                 ("The re-entry limit has been reached on this bar." if reached else "The re-entry limit has not been reached on this bar."))]
        if pos:
            rows.append(("Open position", f"{pos['side'].upper()} via {pos['symbol']} · entry ref {pos['entry_ref']:,.2f}", "Frozen when the position was opened; all levels are SOXL prices."))
            rows.append(("Take-profit / run-up target", f"{pos['tp']:,.2f} / {pos['target']:,.2f}", "The take-profit closes the trade; reaching the run-up target arms the trailing stop."))
            rows.append(("Trailing stop", f"{pos['trail']:,.2f} (extreme {pos['extreme']:,.2f})" if pos["trail"] is not None else "not armed yet",
                         "Once armed it ratchets with the best price reached and never loosens."))
            if (pos["side"] == "long") != up:
                rows.append(("⚠ Position vs trend", "the saved position is against the current trend", "The engine closes it with a market order and reverses on its next poll while a session runs."))
        else:
            rows.append(("Open position", "none saved by this app", "Flat, or opened outside this app (then its exits are not managed)."))
        self.table.setRowCount(len(rows))
        for i, (a, b, c) in enumerate(rows):
            set_row(self.table, i, [a, b, c], {1: side_color("long" if up else "short") if a in ("Trend", "Supertrend line") else None, 2: MUTED}, align_left=(1, 2))
        self.table.resizeRowsToContents()
        self.table.resizeColumnToContents(0); self.table.resizeColumnToContents(1)
