"""Live trading tab: connect to Alpaca, pre-flight, plan the next session's orders (dry run), and - only when ARMED and confirmed - queue them. Reconcile, adopt, KILL.

Nothing is sent unless the 'ARM' box is ticked AND the confirmation dialog is accepted. A real-money account additionally asks for a typed phrase. The window always starts on PAPER + DRY RUN."""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget)

import statarb_data as D
import statarb_live as LV
from supertrend_ui.theme import ACCENT, GREEN, MUTED, RED, TEXT, YELLOW, Kpi, QComboBox, big_money, box_style, make_table, muted, pct, set_row, spin, strip
from supertrend_ui.workers import FnWorker, WorkerBag

from .charts import BarChart
from .settings_panel import SettingsPanel

LOG_COLORS = {"err": RED, "warn": YELLOW, "ok": GREEN, "trade": ACCENT, "info": TEXT}


class LiveTab(QWidget):
    def __init__(self, settings: SettingsPanel, qs: QSettings, reports_dir: Callable[[], str] = lambda: "."):
        super().__init__()
        self.sp, self.qs, self.reports_dir = settings, qs, reports_dir
        self.eng: Optional[LV.LiveEngine] = None
        self.plan: Optional[LV.Plan] = None
        self.acct: dict = {}
        self._workers = WorkerBag()
        self._busy = False
        self._auto_key = None
        self._live_ok = False                                    # set only by the typed 'ARM LIVE' phrase; cleared whenever the account changes
        root = QVBoxLayout(self)
        banner = QLabel("Heads-up: Valid has NOT signed off for live capital · the 2008 reversion signal has no net edge today; the earnings-premium sleeve is marginal (net Sharpe ~0.45, std. error 0.33) and lumpy · "
                        "start DRY RUN, then PAPER, then tiny. Orders are market-on-open, queued 19:00-09:28 ET.")
        banner.setObjectName("warn"); banner.setWordWrap(True)
        root.addWidget(banner)
        self.env = QLabel(""); self.env.setWordWrap(True)
        root.addWidget(self.env)

        top = QHBoxLayout()
        top.addWidget(self._account_box(), 3); top.addWidget(self._mode_box(), 4); top.addWidget(self._control_box(), 3)
        root.addLayout(top)
        names = ("Decision session", "Capital deployed", "Gross target", "Shock multiplier", "Positions", "Next window")
        k = QHBoxLayout()
        self.k = {n: Kpi(n) for n in names}
        for w in self.k.values():
            k.addWidget(w)
        root.addLayout(k)
        self.tabs = QTabWidget()
        self.t_orders = make_table(["Symbol", "Side", "Qty", "Ref px", "Notional", "Target w", "Now", "→ Target", "Why", "OK", "Note"])
        self.tabs.addTab(self.t_orders, "Planned orders")
        self.t_book = make_table(["Symbol", "Side", "Target w", "Shares (ledger)", "Broker", "Diff"])
        self.tabs.addTab(self.t_book, "Book / ledger vs broker")
        ex = QWidget(); ev = QVBoxLayout(ex)
        self.c_expo = BarChart(3.4)
        self.expo_note = muted("")
        ev.addWidget(self.expo_note); ev.addWidget(self.c_expo, 1)
        self.tabs.addTab(ex, "Exposures")
        self.t_reg = make_table(["Regime / risk reading", "Value"])
        self.tabs.addTab(self.t_reg, "Market regime")
        self.t_pf = make_table(["Check", "Result", "Detail"])
        self.tabs.addTab(self.t_pf, "Pre-flight")
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumBlockCount(3000)
        self.tabs.addTab(self.log_box, "Log")
        root.addWidget(self.tabs, 1)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(1000)
        self.auto_timer = QTimer(self); self.auto_timer.timeout.connect(self._auto); self.auto_timer.start(30000)
        self._refresh_env()
        self.account.currentIndexChanged.connect(self._account_changed)
        self.armed.toggled.connect(self._arm_toggled)

    # ── construction ─────────────────────────────────────────────────────────────
    def _account_box(self) -> QGroupBox:
        g = QGroupBox("Alpaca account"); f = QGridLayout(g)
        self.account = QComboBox(); self.account.addItems(["Paper (simulated money)", "LIVE (real money)"])
        self.btn_connect = QPushButton("Connect"); self.btn_connect.clicked.connect(self.connect_)
        self.acct_lbl = muted("not connected")
        f.addWidget(QLabel("Account"), 0, 0); f.addWidget(self.account, 0, 1); f.addWidget(self.btn_connect, 1, 0, 1, 2); f.addWidget(self.acct_lbl, 2, 0, 1, 2)
        return g

    def _mode_box(self) -> QGroupBox:
        g = QGroupBox("Mode & limits"); f = QFormLayout(g)
        self.armed = QCheckBox("ARM - allow sending orders"); self.armed.setToolTip("Unticked = DRY RUN: plans are computed and shown, nothing is sent.")
        self.exec_mode = QComboBox(); self.exec_mode.addItems(["opg", "day"]); self.exec_mode.setToolTip("opg = market-on-open queued overnight (the backtest's fill). day = plain market order right now (testing only).")
        self.eq_frac = spin(0.01, 1.0, 1.0, 2, 0.05); self.cap_over = spin(0, 1e9, 0, 0, 10_000); self.cap_over.setPrefix("$ "); self.cap_over.setToolTip("0 = use equity x fraction")
        self.min_trade = spin(0, 100_000, 250, 0, 50); self.min_trade.setPrefix("$ ")
        self.max_name = spin(0.005, 0.5, 0.05, 3, 0.005)
        self.halt = spin(0.0, 0.5, 0.025, 3, 0.005)
        self.etb = QCheckBox("Shorts only in easy-to-borrow names"); self.etb.setChecked(True)
        self.auto = QCheckBox("Auto: plan after 16:25 ET, submit in the OPG window (only if armed)")
        f.addRow(self.armed); f.addRow("Execution", self.exec_mode); f.addRow("Equity fraction", self.eq_frac); f.addRow("…or fixed capital", self.cap_over)
        f.addRow("Min ticket", self.min_trade); f.addRow("Max per name (of capital)", self.max_name); f.addRow("Daily-loss halt", self.halt); f.addRow(self.etb); f.addRow(self.auto)
        return g

    def _control_box(self) -> QGroupBox:
        g = QGroupBox("Run"); v = QVBoxLayout(g)
        self.btn_pf = QPushButton("Pre-flight check"); self.btn_pf.clicked.connect(self.preflight)
        self.btn_plan = QPushButton("Plan next session (dry run)"); self.btn_plan.setObjectName("primary"); self.btn_plan.clicked.connect(lambda: self.make_plan(False))
        self.btn_plan_r = QPushButton("Refresh data + plan"); self.btn_plan_r.clicked.connect(lambda: self.make_plan(True))
        self.btn_submit = QPushButton("Submit plan to Alpaca"); self.btn_submit.setObjectName("go"); self.btn_submit.clicked.connect(self.submit)
        self.btn_rec = QPushButton("Reconcile with broker"); self.btn_rec.clicked.connect(self.reconcile)
        self.btn_adopt = QPushButton("Adopt broker positions"); self.btn_adopt.clicked.connect(self.adopt)
        self.btn_adopt.setToolTip("After a partial fill / reject / manual change: make the engine's ledger equal the broker for the symbols it owns.")
        self.btn_hwm = QPushButton("Reset drawdown base"); self.btn_hwm.clicked.connect(self.reset_hwm)
        self.btn_hwm.setToolTip("After a drawdown stop (-8% halves the gross, -10% goes flat): make today's equity the new high-water mark.")
        self.btn_kill = QPushButton("KILL: cancel orders + flatten engine positions"); self.btn_kill.setObjectName("danger"); self.btn_kill.clicked.connect(self.kill)
        for b in (self.btn_pf, self.btn_plan, self.btn_plan_r, self.btn_submit, self.btn_rec, self.btn_adopt, self.btn_hwm, self.btn_kill):
            v.addWidget(b)
        return g

    # ── helpers ──────────────────────────────────────────────────────────────────
    def paper(self) -> bool:
        return self.account.currentIndex() == 0

    def live_settings(self) -> LV.LiveSettings:
        return LV.LiveSettings(paper=self.paper(), armed=self.armed.isChecked() and (self.paper() or self._live_ok), universe=self.sp.universe(), equity_fraction=self.eq_frac.value(), capital_override=self.cap_over.value(),
                               exec_mode=self.exec_mode.currentText(), min_trade_usd=self.min_trade.value(), max_name_pct=self.max_name.value(), daily_loss_halt=self.halt.value(),
                               require_etb=self.etb.isChecked())

    def engine(self) -> LV.LiveEngine:
        f, p, ls = self.sp.factor_cfg(), self.sp.port_cfg(), self.live_settings()
        if self.eng is None or self.eng.live.paper != ls.paper or self.eng.live.universe != ls.universe:
            self.eng = LV.LiveEngine(f, p, ls, log=self.add_log)
        else:
            self.eng.f, self.eng.p, self.eng.live = f, p, ls
        return self.eng

    def add_log(self, msg: str, kind: str = "info") -> None:
        fmt = QTextCharFormat(); fmt.setForeground(QColor(LOG_COLORS.get(kind, TEXT)))
        c = self.log_box.textCursor(); c.movePosition(QTextCursor.MoveOperation.End)
        c.insertText(f"{datetime.now():%H:%M:%S}  {msg}\n", fmt)
        self.log_box.setTextCursor(c); self.log_box.ensureCursorVisible()

    def _refresh_env(self) -> None:
        if self.paper():
            txt, col = ("PAPER account - simulated money. DRY RUN unless ARMED.", GREEN)
        else:
            txt, col = ("LIVE account - REAL MONEY. Orders you submit are real.", RED)
        self.env.setText(txt); self.env.setStyleSheet(box_style(col, 0.14, True, 6))

    def _account_changed(self, *_) -> None:
        """Switching accounts always disarms: arming on paper must never carry over to the real-money account (and the typed phrase has to be given again)."""
        self._live_ok = False
        if self.armed.isChecked():
            self.armed.blockSignals(True); self.armed.setChecked(False); self.armed.blockSignals(False)
            self.add_log("account changed - DISARMED", "warn")
        self._refresh_env()
        self.eng = None

    def _arm_toggled(self, on: bool) -> None:
        if on and not self.paper():
            txt, ok = QInputDialog.getText(self, "Arm LIVE trading", "This is a REAL-MONEY account. Orders will be sent when you press Submit.\n\nType  ARM LIVE  to continue:")
            if not ok or txt.strip() != "ARM LIVE":
                self.armed.blockSignals(True); self.armed.setChecked(False); self.armed.blockSignals(False)
                self._live_ok = False
                return
            self._live_ok = True
        if not on:
            self._live_ok = False
        self.add_log("ARMED - Submit will send orders" if on else "DISARMED - dry run", "warn" if on else "info")

    def _run(self, fn, on_done, label: str) -> None:
        if self._busy:
            self.add_log("another action is still running", "warn")
            return
        self._busy = True
        self.add_log(label)
        w = FnWorker(fn)
        def done(res, err):
            self._busy = False
            if err:
                self.add_log(f"{label} failed: {err}", "err")
                return
            on_done(res)
        self._workers.start(w, done, lambda s: self.add_log(s.strip()))

    # ── actions ──────────────────────────────────────────────────────────────────
    def connect_(self) -> None:
        eng = self.engine()
        self._run(lambda log: eng.connect(), self._connected, "connecting to Alpaca ...")

    def _connected(self, a: dict) -> None:
        self.acct = a
        self.acct_lbl.setText(f"{'PAPER' if a['paper'] else 'LIVE'} {a['number']} · {a['status']} · equity {big_money(a['equity'])} · buying power {big_money(a['buying_power'])} · "
                              f"margin {a['multiplier']}x · shorting {'on' if a['shorting'] else 'OFF'} · market {'open' if a['market_open'] else 'closed'} · {a['latency_ms']} ms")
        self.add_log("connected", "ok")

    def preflight(self) -> None:
        eng = self.engine()
        self._run(lambda log: eng.preflight(), self._pf_done, "pre-flight ...")

    def _pf_done(self, checks: list) -> None:
        t = self.t_pf
        t.setRowCount(len(checks))
        for r, c in enumerate(checks):
            set_row(t, r, [c.name, c.level.upper(), c.msg], {1: {"ok": GREEN, "warn": YELLOW, "fail": RED}[c.level]}, align_left=(2,))
        self.tabs.setCurrentWidget(t)
        fails = [c for c in checks if c.level == "fail"]
        self.add_log("pre-flight: " + ("FAILED - " + "; ".join(c.msg for c in fails) if fails else "no blocking problems"), "err" if fails else "ok")

    def make_plan(self, refresh: bool) -> None:
        eng = self.engine()
        def fn(log):
            eng.connect()
            return eng.plan(refresh=refresh)
        self._run(fn, self._planned, "planning (refresh data first)" if refresh else "planning ...")

    def _planned(self, plan: LV.Plan) -> None:
        self.plan = plan
        self.acct = self.eng.acct
        o = plan.orders
        self.k["Decision session"].set(f"{plan.decision_date:%Y-%m-%d}", None, "fills at the next open")
        self.k["Capital deployed"].set(big_money(plan.deploy), None, f"equity {big_money(plan.equity)}")
        self.k["Gross target"].set(f"{plan.g_target:.2f}x", None, "vol-target × shock")
        self.k["Shock multiplier"].set(f"{plan.shock_mult:.2f}", RED if plan.shock_mult < 0.6 else YELLOW if plan.shock_mult < 1 else GREEN)
        self.k["Positions"].set(f"{plan.n_long} L / {plan.n_short} S", None, f"{len(o)} orders")
        t = self.t_orders
        t.setRowCount(len(o))
        for r, x in enumerate(o.itertuples()):
            set_row(t, r, [x.symbol, x.side.upper(), f"{x.qty:,}", f"{x.ref_px:,.2f}" if np.isfinite(x.ref_px) else "—", f"${x.notional:,.0f}" if np.isfinite(x.notional) else "—", pct(x.target_w, 2),
                           f"{x.cur_qty:+,}", f"{x.tgt_qty:+,}", x.reason, "yes" if x.ok else "NO", x.note], {1: GREEN if x.side == "buy" else RED, 9: None if x.ok else RED}, align_left=(8, 10))
        book = plan.targets
        bt = self.t_book
        syms = sorted(set(book) | set(plan.ledger_after))
        bt.setRowCount(len(syms))
        for r, s in enumerate(syms):
            w = book.get(s, 0.0)
            set_row(bt, r, [s, "long" if w > 0 else "short" if w < 0 else "—", pct(w, 2), f"{plan.ledger_after.get(s, 0):+,}", "—", "—"], {1: GREEN if w > 0 else RED if w < 0 else None})
        ex = {k: v for k, v in plan.exposures.items() if not k.startswith("sector:") and k not in ("gross",)}
        sec = {k.split(":")[1]: v for k, v in plan.exposures.items() if k.startswith("sector:")}
        labels = list(ex) + [f"sector {k}" for k in sec]
        vals = list(ex.values()) + list(sec.values())
        self.c_expo.show_bars(labels, vals, title="net exposure per unit of gross (target book)")
        self.expo_note.setText(f"gross {plan.exposures['gross']:.2f}x equity · net dollar {plan.exposures['net_dollar']:+.3f} · SPY beta {plan.exposures['beta_spy']:+.3f} (per unit gross; ~0 = market neutral)")
        rt = self.t_reg
        reg = [(k, v) for k, v in plan.regime.items() if v is not None]
        rt.setRowCount(len(reg))
        for r, (k, v) in enumerate(reg):
            set_row(rt, r, [k, f"{v:.3f}"])
        for n in plan.notes:
            self.add_log(n, "warn")
        self.add_log(f"plan for {plan.decision_date:%Y-%m-%d}: {plan.n_long} long / {plan.n_short} short, {len(o)} orders, notional {big_money(float(o.notional.sum()) if len(o) else 0.0)}", "ok")
        self.tabs.setCurrentWidget(self.t_orders)

    def submit(self) -> None:
        if self.plan is None:
            QMessageBox.information(self, "Submit", "Make a plan first.")
            return
        if not self.armed.isChecked() or (not self.paper() and not self._live_ok):
            QMessageBox.information(self, "DRY RUN", "The engine is not ARMED, so nothing will be sent.\n\nTick 'ARM' (and use the paper account first) to send orders.")
            return
        o = self.plan.orders
        n_ok = int(o.ok.sum()) if len(o) else 0
        notional = float(o[o.ok].notional.sum()) if len(o) else 0.0
        msg = (f"Send {n_ok} market-on-open orders (about {big_money(notional)}) to the {'PAPER' if self.paper() else 'LIVE'} account for the session after "
               f"{self.plan.decision_date:%Y-%m-%d}?\n\nExecution: {self.exec_mode.currentText().upper()}")
        if QMessageBox.question(self, "Confirm submit", msg) != QMessageBox.StandardButton.Yes:
            return
        eng, plan = self.engine(), self.plan
        self._run(lambda log: eng.submit(plan), self._submitted, "submitting orders ...")

    def _submitted(self, res: list) -> None:
        bad = [r for r in res if r.get("status") in ("rejected",)]
        for r in res:
            self.add_log(f"{r['symbol']}: {r['status']} {r.get('note', '')}", "err" if r.get("status") == "rejected" else "trade")
        self.add_log(f"submitted {len(res) - len(bad)} / {len(res)} orders" + (f" ({len(bad)} rejected)" if bad else ""), "warn" if bad else "ok")

    def reconcile(self) -> None:
        eng = self.engine()
        def fn(log):
            eng.connect()
            return eng.reconcile()
        self._run(fn, self._reconciled, "reconciling ...")

    def _reconciled(self, df: pd.DataFrame) -> None:
        t = self.t_book
        t.setRowCount(len(df))
        for r, x in enumerate(df.itertuples()):
            set_row(t, r, [x.symbol, "long" if x.ledger > 0 else "short" if x.ledger < 0 else "—", "—", f"{x.ledger:+,}", f"{x.broker:+,}", f"{x.diff:+,}"], {5: RED if x.diff else GREEN})
        self.tabs.setCurrentWidget(t)
        self.add_log(f"reconcile: {int((df['diff'] != 0).sum()) if len(df) else 0} of {len(df)} positions differ from the broker", "warn" if len(df) and (df["diff"] != 0).any() else "ok")

    def adopt(self) -> None:
        if QMessageBox.question(self, "Adopt", "Set the engine's ledger equal to the broker's positions for the symbols the engine owns?") != QMessageBox.StandardButton.Yes:
            return
        eng = self.engine()
        def fn(log):
            eng.connect()
            return eng.adopt_broker()
        self._run(fn, lambda n: self.add_log(f"adopted broker quantities for {n} symbols", "ok"), "adopting ...")

    def reset_hwm(self) -> None:
        if QMessageBox.question(self, "Reset drawdown base", "Make today's account equity the new high-water mark for the drawdown stops?") != QMessageBox.StandardButton.Yes:
            return
        eng = self.engine()
        def fn(log):
            eng.connect()
            return eng.reset_drawdown_base()
        self._run(fn, lambda v: self.add_log(f"drawdown base reset to ${v:,.0f}", "ok"), "resetting the drawdown base ...")

    def kill(self) -> None:
        phrase = "KILL"
        txt, ok = QInputDialog.getText(self, "KILL", f"Cancel the engine's open orders and close every position in the engine's ledger on the {'PAPER' if self.paper() else 'LIVE'} account.\n\nType  {phrase}  to continue:")
        if not ok or txt.strip() != phrase:
            return
        eng = self.engine()
        self._run(lambda log: eng.flatten(), lambda res: self.add_log(f"KILL: {len(res)} actions: " + "; ".join(f"{r['symbol']}:{r['status']}" for r in res[:25]), "warn"), "KILL ...")

    # ── clock / auto ─────────────────────────────────────────────────────────────
    def _tick(self) -> None:
        now = LV.clock_now()
        ok, msg = LV.opg_window_open(now)
        self.k["Next window"].set("OPG open" if ok else "OPG 19:00 ET", GREEN if ok else None, f"{now:%a %H:%M} ET")

    def _auto(self) -> None:
        if not self.auto.isChecked() or self._busy:
            return
        now = LV.clock_now()
        if now.weekday() >= 5 and now.hour >= 9:
            pass
        key = now.strftime("%Y%m%d")
        if (now.hour, now.minute) >= (16, 25) and (now.hour, now.minute) < (19, 0) and self._auto_key != ("plan", key):
            self._auto_key = ("plan", key)
            self.add_log("auto: planning after the close", "info")
            self.make_plan(True)
        elif self.armed.isChecked() and self.plan is not None and LV.opg_window_open(now)[0] and self._auto_key != ("submit", self.plan.decision_date.strftime("%Y%m%d")):
            st = self.engine().load_state()
            if st.get("held_date") == str(self.plan.decision_date):
                self._auto_key = ("submit", self.plan.decision_date.strftime("%Y%m%d"))
                return
            self._auto_key = ("submit", self.plan.decision_date.strftime("%Y%m%d"))
            eng, plan = self.engine(), self.plan
            self.add_log("auto: submitting the plan", "warn")
            self._run(lambda log: eng.submit(plan), self._submitted, "auto-submitting ...")

    # ── persistence ──────────────────────────────────────────────────────────────
    def save(self) -> None:
        for k, w in (("live/eq_frac", self.eq_frac), ("live/cap", self.cap_over), ("live/min_trade", self.min_trade), ("live/max_name", self.max_name), ("live/halt", self.halt)):
            self.qs.setValue(k, w.value())

    def load(self) -> None:
        for k, w in (("live/eq_frac", self.eq_frac), ("live/cap", self.cap_over), ("live/min_trade", self.min_trade), ("live/max_name", self.max_name), ("live/halt", self.halt)):
            try:
                v = self.qs.value(k)
                if v is not None:
                    w.setValue(float(v))
            except (TypeError, ValueError):
                pass
