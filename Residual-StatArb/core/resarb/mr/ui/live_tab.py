"""Live trading tab for the mean-reversion model picked on the left - the exact model object the backtest ran: connect, pre-flight, plan now (dry run), the daily loop, KILL.

Nothing is sent unless 'ARM' is ticked AND the typed phrase is given AND the engine has the project's own keys on an account no other strategy claimed. The window always starts on
PAPER + DRY RUN. Real money is disabled in code (Valid has not signed off)."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QTabWidget,
                             QVBoxLayout, QWidget)

from ...ui.theme import ACCENT, GREEN, RED, TEXT, YELLOW, Kpi, QComboBox, big_money, box_style, make_table, muted, set_row, spin, strip
from ...ui.workers import FnWorker, WorkerBag
from .. import live as ML
from .model_panel import describe

LOG_COLORS = {"err": RED, "warn": YELLOW, "ok": GREEN, "trade": ACCENT, "info": TEXT}


class _Bridge(QObject):
    log = pyqtSignal(str, str)
    plan = pyqtSignal(object)
    stopped = pyqtSignal(str)


class LiveTab(QWidget):
    def __init__(self, panel, backtest, qs: QSettings, folder: str):
        super().__init__()
        self.panel, self.bt, self.qs, self.folder = panel, backtest, qs, folder
        self.eng: Optional[ML.MRLiveEngine] = None
        self.plan: Optional[ML.MRPlan] = None
        self._workers = WorkerBag()
        self._busy = False
        self._loop: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.br = _Bridge()
        self.br.log.connect(self.add_log)
        self.br.plan.connect(self._planned)
        self.br.stopped.connect(self._loop_ended)
        root = QVBoxLayout(self)
        banner = QLabel("Valid has NOT signed this model off for capital. Run it DRY, then on paper. Keys come from 'Alpaca Key.txt' in the project folder "
                        "(MR_ALPACA_API_KEY / MR_ALPACA_SECRET_KEY would select a separate paper account). Real money is disabled in code.")
        banner.setObjectName("warn"); banner.setWordWrap(True)
        root.addWidget(banner)
        self.model_strip = strip("", ACCENT)
        root.addWidget(self.model_strip)
        self.env = strip("PAPER account - DRY RUN unless ARMED", GREEN, bold=True)
        root.addWidget(self.env)
        top = QHBoxLayout()
        top.addWidget(self._account_box(), 3); top.addWidget(self._mode_box(), 4); top.addWidget(self._control_box(), 3)
        root.addLayout(top)
        k = QHBoxLayout()
        self.k = {n: Kpi(n) for n in ("Today", "Capital", "Positions", "Orders", "Candidates", "Loop")}
        for w in self.k.values():
            k.addWidget(w)
        root.addLayout(k)
        self.tabs = QTabWidget()
        self.t_orders = make_table(["Ticker", "Side", "Shares", "Order", "Ref price", "Notional", "Why", "OK", "Note"])
        self.tabs.addTab(self.t_orders, "Orders (closing auction)")
        self.t_cand = make_table(["Rank", "Ticker", "Signal", "Price", "Gets capital"])
        self.tabs.addTab(self.t_cand, "Candidates (ranked)")
        self.t_pos = make_table(["Ticker", "Shares", "Entry date", "Entry price", "Price now", "Value", "P&L $", "Held (sessions)", "Exit today"])
        self.tabs.addTab(self.t_pos, "Positions")
        self.t_pf = make_table(["Check", "Result", "Detail"])
        self.tabs.addTab(self.t_pf, "Pre-flight")
        self.t_trades = make_table(["Ticker", "Entry date", "Entry price", "Exit date", "Exit price", "Shares", "P&L $", "Signal", "Rank"])
        self.tabs.addTab(self.t_trades, "Live trades (journal)")
        self.t_par = make_table(["Day", "Ticker", "Leg", "Live shares", "Backtest shares", "Match"])
        self.tabs.addTab(self.t_par, "Live vs backtest")
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumBlockCount(5000)
        self.tabs.addTab(self.log_box, "Log")
        root.addWidget(self.tabs, 1)
        self.armed.toggled.connect(self._arm_toggled)
        self.panel.changed.connect(self._model_strip)
        self.bt.finished.connect(self._model_strip)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(1000)
        self._model_strip()

    # ── construction ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _account_box(self) -> QGroupBox:
        g = QGroupBox("Alpaca account"); f = QGridLayout(g)
        self.btn_connect = QPushButton("Connect"); self.btn_connect.clicked.connect(self.connect_)
        self.acct_lbl = muted("not connected")
        f.addWidget(QLabel("Paper account (simulated money)"), 0, 0, 1, 2)
        f.addWidget(self.btn_connect, 1, 0, 1, 2); f.addWidget(self.acct_lbl, 2, 0, 1, 2)
        return g

    def _mode_box(self) -> QGroupBox:
        g = QGroupBox("Mode and limits"); f = QFormLayout(g)
        self.armed = QCheckBox("ARM - allow sending orders")
        self.armed.setToolTip("Unticked = DRY RUN: every decision is planned and journaled, nothing is sent.")
        self.feed = QComboBox(); self.feed.addItems(list(ML.FEEDS))
        self.feed.setToolTip("sip_delayed = the free plan's SIP bars, 16 minutes late (decision time 15:30 or earlier - exactly the backtest's data); sip+iex = free: delayed SIP plus "
                             "real-time IEX for the last ~30 minutes (allows 15:45; the last half hour's prints are IEX-only); iex = real time, IEX prints only; "
                             "sip = real-time SIP (paid plan - the backtest's data at 15:45)")
        self.cap = spin(0, 1e9, 0, 0, 5_000); self.cap.setPrefix("$ "); self.cap.setToolTip("0 = account equity x fraction")
        self.eq_frac = spin(0.05, 1.0, 1.0, 2, 0.05)
        self.halt = spin(0.0, 0.5, 0.0, 3, 0.005); self.halt.setToolTip("0 = off. Live-only overlay (the backtest has none): no new entries after this loss today")
        self.dd = spin(0.0, 0.9, 0.0, 2, 0.01); self.dd.setToolTip("0 = off. Live-only overlay: sell everything and stay flat after this drawdown, until reset")
        f.addRow(self.armed); f.addRow("Today's bars from", self.feed); f.addRow("Fixed capital", self.cap); f.addRow("…or equity fraction", self.eq_frac)
        f.addRow("Daily-loss halt", self.halt); f.addRow("Drawdown stop", self.dd)
        return g

    def _control_box(self) -> QGroupBox:
        g = QGroupBox("Run"); v = QVBoxLayout(g)
        self.btn_pf = QPushButton("Pre-flight check"); self.btn_pf.clicked.connect(self.preflight)
        self.btn_plan = QPushButton("Plan now (dry run)"); self.btn_plan.setObjectName("primary"); self.btn_plan.clicked.connect(self.plan_now)
        self.btn_start = QPushButton("Start the daily loop"); self.btn_start.setObjectName("go"); self.btn_start.clicked.connect(self.start_loop)
        self.btn_stop = QPushButton("Stop loop"); self.btn_stop.clicked.connect(self.stop_loop); self.btn_stop.setEnabled(False)
        self.btn_cmp = QPushButton("Check live vs backtest"); self.btn_cmp.clicked.connect(self.compare)
        self.btn_cmp.setToolTip("Rebuilds every journaled decision from the backtest's data (update the data first) with the positions and capital live had, and compares the orders.")
        self.btn_hwm = QPushButton("Reset drawdown base"); self.btn_hwm.clicked.connect(self.reset_hwm)
        self.btn_kill = QPushButton("KILL: cancel + flatten"); self.btn_kill.setObjectName("danger"); self.btn_kill.clicked.connect(self.kill)
        self.btn_kill.setToolTip("cancel this model's open orders and close every position in its symbols")
        for b in (self.btn_pf, self.btn_plan, self.btn_start, self.btn_stop, self.btn_cmp, self.btn_hwm, self.btn_kill):
            v.addWidget(b)
        return g

    # ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def live_settings(self) -> ML.MRLiveSettings:
        return ML.MRLiveSettings(paper=True, armed=self.armed.isChecked(), capital=self.cap.value(), equity_fraction=self.eq_frac.value(), feed=self.feed.currentText(),
                                 daily_loss_halt=self.halt.value(), dd_stop=self.dd.value())

    def engine(self) -> ML.MRLiveEngine:
        m = self.panel.model()
        ls = self.live_settings()
        if self.eng is None:
            self.eng = ML.MRLiveEngine(m, ls, self.folder, log=lambda msg, k="info": self.br.log.emit(msg, k))
        elif self._loop is None or not self._loop.is_alive():
            self.eng.m, self.eng.live = m, ls
        return self.eng

    def _model_strip(self) -> None:
        m = self.panel.model()
        running = self._loop is not None and self._loop.is_alive() and self.eng is not None
        if running:
            m = self.eng.m
        else:
            want = ML.default_feed(m.minute, self.feed.currentText())
            if want != self.feed.currentText():
                self.feed.setCurrentText(want)
                self.add_log(f"today's bars now come from '{want}' (a {m.minute // 100}:{m.minute % 100:02d} decision)", "info")
        bt = self.bt.ran_model
        same = bt is not None and bt.to_dict() == m.to_dict()
        tag = ("✓ the same model as the last backtest" if same else "⚠ not backtested in this session - run the Backtest tab first") if not running else "● running"
        self.model_strip.setText(f"LIVE MODEL - {describe(m)}   ·   {tag}")
        self.model_strip.setStyleSheet(box_style(GREEN if same or running else YELLOW, 0.12, True, 8))

    def add_log(self, msg: str, kind: str = "info") -> None:
        fmt = QTextCharFormat(); fmt.setForeground(QColor(LOG_COLORS.get(kind, TEXT)))
        c = self.log_box.textCursor(); c.movePosition(QTextCursor.MoveOperation.End)
        c.insertText(f"{datetime.now():%H:%M:%S}  {msg}\n", fmt)
        self.log_box.setTextCursor(c); self.log_box.ensureCursorVisible()

    def _arm_toggled(self, on: bool) -> None:
        if on:
            if self._loop is not None and self._loop.is_alive():
                self._set_armed(False)
                QMessageBox.information(self, "ARM", "Stop the loop first, then arm and start it again.")
                return
            txt, ok = QInputDialog.getText(self, "Arm paper trading", f"The model '{self.panel.model().name}' will SEND market-on-close orders to its paper account every session.\n\n"
                                                                       f"Type  {ML.ARM_PHRASE}  to continue:")
            if not ok or txt.strip() != ML.ARM_PHRASE:
                self._set_armed(False)
                return
            eng = self.engine()
            probs = eng.arm_problems() if eng.acct else ["connect first"]
            if probs:
                self._set_armed(False)
                QMessageBox.warning(self, "Cannot arm", "\n".join(probs))
                return
        elif self._loop is not None and self._loop.is_alive() and self.eng is not None:
            self.eng.live.armed = False                                                       # disarming takes effect at once, even inside a running loop
        self.env.setText("PAPER account - ARMED: market-on-close orders are sent at the decision time" if on else "PAPER account - DRY RUN unless ARMED")
        self.env.setStyleSheet(box_style(YELLOW if on else GREEN, 0.14, True, 8))
        self.add_log("ARMED" if on else "DISARMED - dry run", "warn" if on else "info")

    def _set_armed(self, v: bool) -> None:
        self.armed.blockSignals(True); self.armed.setChecked(v); self.armed.blockSignals(False)

    def _run(self, fn, on_done, label: str) -> None:
        if self._busy:
            self.add_log("another action is still running", "warn")
            return
        self._busy = True
        self.add_log(label)

        def done(res, err):
            self._busy = False
            if err:
                self.add_log(f"{label} failed: {err}", "err")
                return
            on_done(res)
        self._workers.start(FnWorker(fn), done, lambda s: self.add_log(s.strip()))

    # ── actions ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def connect_(self) -> None:
        eng = self.engine()
        self._run(lambda log: eng.connect(), self._connected, "connecting ...")

    def _connected(self, a: dict) -> None:
        self.acct_lbl.setText(f"{a['number'][:4]}**** · {a['status']} · equity {big_money(a['equity'])} · buying power {big_money(a['buying_power'])} · margin {a['multiplier']:.0f}x · "
                              f"keys: {a['keys'].upper()} · market {'open' if a['market_open'] else 'closed'} · {a['latency_ms']} ms")
        self.add_log(f"connected (keys from the {a.get('key_src', '?')})" + ("" if a["keys"] == "own" else " - fine for DRY RUN; put 'Alpaca Key.txt' in the project folder to arm"),
                     "ok" if a["keys"] == "own" else "warn")

    def preflight(self) -> None:
        eng = self.engine()
        self._run(lambda log: eng.preflight(), self._pf_done, "pre-flight ...")

    def _pf_done(self, checks: list) -> None:
        t = self.t_pf
        t.setRowCount(len(checks))
        for r, c in enumerate(checks):
            set_row(t, r, [c.name, c.level.upper(), c.msg], {1: {"ok": GREEN, "warn": YELLOW, "fail": RED}[c.level]}, align_left=(2,))
        self.tabs.setCurrentWidget(t)

    def plan_now(self) -> None:
        eng = self.engine()

        def fn(log):
            if eng.broker is None:
                eng.connect()
            return eng.plan(force=True)
        self._run(fn, self._planned, "planning (dry run) ...")

    def _planned(self, plan: ML.MRPlan) -> None:
        self.plan = plan
        o, c, p = plan.orders, plan.candidates, plan.positions
        self.k["Capital"].set(big_money(plan.capital), None, f"equity {big_money(plan.equity)}")
        self.k["Positions"].set(f"{len(p)} / {self.eng.m.slots if self.eng else '?'}", None, f"{int((p.exit != '').sum()) if len(p) else 0} leave today")
        self.k["Orders"].set(f"{len(o)}", None, f"{int((o.kind == 'exit').sum()) if len(o) else 0} exits · {int((o.kind == 'entry').sum()) if len(o) else 0} entries")
        self.k["Candidates"].set(f"{len(c)}", None, f"{int(c.taken.sum()) if len(c) else 0} get capital")
        t = self.t_orders
        t.setRowCount(len(o))
        for r, x in enumerate(o.itertuples()):
            set_row(t, r, [x.symbol, x.side.upper(), f"{x.qty:,}", "market-on-close" if x.tif == "cls" else x.tif.upper(), f"{x.ref_px:,.2f}",
                           f"${x.notional:,.0f}" if np.isfinite(x.notional) else "—", x.why, "yes" if x.ok else "NO", x.note],
                    {1: GREEN if x.side == "buy" else RED, 7: None if x.ok else RED}, align_left=(6, 8))
        t = self.t_cand
        t.setRowCount(len(c))
        for r, x in enumerate(c.itertuples()):
            set_row(t, r, [f"{x.rank}", x.symbol, f"{x.signal:.4f}", f"{x.last:,.2f}", "YES" if x.taken else "no (slots full / lower rank)"], {4: GREEN if x.taken else None},
                    align_left=(1, 4))
        t = self.t_pos
        t.setRowCount(len(p))
        for r, x in enumerate(p.itertuples()):
            set_row(t, r, [x.symbol, f"{x.shares:+,}", str(x.entry_date), f"{x.entry_px:,.2f}", f"{x.last:,.2f}", f"${x.value:,.0f}", f"{x.pnl:+,.0f}", f"{x.age}", x.exit],
                    {6: GREEN if x.pnl > 0 else RED}, align_left=(8,))
        for n in plan.notes:
            self.add_log(n, "warn")
        self.add_log(f"plan for {plan.day:%Y-%m-%d} ({'decision time' if plan.real else 'manual dry run'}): {len(o)} order(s), {len(c)} candidate(s)", "ok")
        self.tabs.setCurrentWidget(self.t_orders)
        self._journal_trades()

    def start_loop(self) -> None:
        if self._loop is not None and self._loop.is_alive():
            return
        m = self.panel.model()
        if self.bt.ran_model is None or self.bt.ran_model.to_dict() != m.to_dict():
            if QMessageBox.question(self, "Model not backtested", "This exact model has not been backtested in this session (Backtest tab). Start the loop anyway?") \
                    != QMessageBox.StandardButton.Yes:
                return
        eng = self.engine()
        if self.armed.isChecked():
            if QMessageBox.question(self, "Start ARMED loop", f"Start the daily loop ARMED on the paper account with\n\n{describe(m)}\n\nIt sends market-on-close orders every session "
                                                                f"until you stop it.") != QMessageBox.StandardButton.Yes:
                return
        try:
            os.makedirs(os.path.join(self.folder, "reports"), exist_ok=True)
            with open(os.path.join(self.folder, "live_model.json"), "w", encoding="utf-8") as fh:
                json.dump(m.to_dict(), fh, indent=2)                                         # the headless runner trades the same file
        except OSError as e:
            self.add_log(f"could not save live_model.json: {e}", "warn")
        self._stop.clear()

        def body():
            msg = ""
            try:
                eng.run(self._stop, on_plan=lambda p: self.br.plan.emit(p))
            except Exception as e:                                                            # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
            self.br.stopped.emit(msg)
        self._loop = threading.Thread(target=body, name="mr-loop", daemon=True)
        self._loop.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.panel.setEnabled(False)                                                          # the running model cannot be edited under the loop
        self.feed.setEnabled(False)
        self._model_strip()

    def stop_loop(self) -> None:
        self._stop.set()
        self.add_log("stopping the loop ...")

    def _loop_ended(self, msg: str) -> None:
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.panel.setEnabled(True); self.feed.setEnabled(True)
        if msg:
            self.add_log("loop ended: " + msg, "err")
        self._model_strip()

    def compare(self) -> None:
        eng = self.engine()

        def fn(log):
            return eng.check_against_backtest(log=log)

        def done(df):
            t = self.t_par
            t.setRowCount(len(df))
            for r, x in enumerate(df.itertuples()):
                set_row(t, r, [f"{pd.Timestamp(x.day):%Y-%m-%d}", x.symbol, x.kind, f"{x.live_qty:,}", f"{x.backtest_qty:,}", x.match], {5: GREEN if x.match == "same" else RED})
            same = int((df.match == "same").sum()) if len(df) else 0
            self.add_log(f"live vs backtest data: {same} of {len(df)} orders identical" + ("" if same == len(df) else " - see the 'Live vs backtest' tab"), "ok" if same == len(df) else "warn")
            self.tabs.setCurrentWidget(t)
        self._run(fn, done, "checking the journaled decisions against the backtest data ...")

    def reset_hwm(self) -> None:
        if QMessageBox.question(self, "Reset drawdown base", "Make the current equity the new high-water mark (and clear a drawdown stop)?") != QMessageBox.StandardButton.Yes:
            return
        eng = self.engine()

        def fn(log):
            if eng.broker is None:
                eng.connect()
            return eng.reset_drawdown()
        self._run(fn, lambda v: self.add_log(f"drawdown base reset to ${v:,.0f}", "ok"), "resetting ...")

    def kill(self) -> None:
        txt, ok = QInputDialog.getText(self, "KILL", "Cancel this model's open orders and CLOSE every position in its symbols on its paper account.\n\nType  KILL  to continue:")
        if not ok or txt.strip() != "KILL":
            return
        self.stop_loop()
        eng = self.engine()
        self._run(lambda log: eng.kill(), lambda res: self.add_log("KILL: " + "; ".join(f"{r['symbol']}:{r['status']}" for r in res[:40]), "warn"), "KILL ...")

    def _journal_trades(self) -> None:
        if self.eng is None:
            return
        rows = self.eng.read_journal(("trade_closed",))
        t = self.t_trades
        t.setRowCount(len(rows))
        for r, x in enumerate(reversed(rows)):
            pnl = x.get("pnl")
            set_row(t, r, [x.get("symbol", ""), str(x.get("entry_date", "")), f"{float(x.get('entry_px') or np.nan):,.2f}", str(x.get("exit_date", "")),
                           f"{float(x.get('exit_px') or np.nan):,.2f}", f"{float(x.get('shares') or 0):+,.0f}", f"{pnl:+,.0f}" if pnl is not None else "—",
                           f"{float(x['signal']):.3f}" if x.get("signal") is not None else "—", str(x.get("rank") or "—")],
                    {6: (GREEN if (pnl or 0) > 0 else RED) if pnl is not None else None})

    def _tick(self) -> None:
        now = ML.now_et()
        alive = self._loop is not None and self._loop.is_alive()
        self.k["Loop"].set("running" if alive else "stopped", GREEN if alive else None, f"{now:%a %H:%M:%S} ET")
        if now.second % 10 == 0:
            try:
                eng = self.eng or ML.MRLiveEngine(self.panel.model(), self.live_settings(), self.folder, log=lambda *a: None)
                tm = eng.times(now.normalize())
                if tm is None:
                    self.k["Today"].set("no session")
                else:
                    nxt = "decision" if now < tm["decision"] else ("after-close step" if now < tm["close"] + pd.Timedelta(minutes=5) else "done")
                    self.k["Today"].set(f"decide {tm['decision']:%H:%M:%S}", None, f"snapshot {tm['snapshot']:%H:%M} · cut-off {tm['cutoff']:%H:%M:%S} · next: {nxt}")
            except Exception:                                                                 # noqa: BLE001
                pass

    def shutdown(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.join(5)

    # ── persistence ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def save(self) -> None:
        for k, w in (("live/cap", self.cap), ("live/eq_frac", self.eq_frac), ("live/halt", self.halt), ("live/dd", self.dd)):
            self.qs.setValue(k, w.value())
        self.qs.setValue("live/feed", self.feed.currentText())

    def load(self) -> None:
        for k, w in (("live/cap", self.cap), ("live/eq_frac", self.eq_frac), ("live/halt", self.halt), ("live/dd", self.dd)):
            try:
                v = self.qs.value(k)
                if v is not None:
                    w.setValue(float(v))
            except (TypeError, ValueError):
                pass
        v = self.qs.value("live/feed")
        if v in ML.FEEDS:
            self.feed.setCurrentText(v)
        self._model_strip()
