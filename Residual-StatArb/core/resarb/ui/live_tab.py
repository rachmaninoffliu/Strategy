"""Live trading tab: connect to this strategy's Alpaca account, pre-flight, plan the current bar (dry run), run the 15-minute loop, KILL.

Nothing is sent unless 'ARM' is ticked AND the typed phrase is given AND the engine has this strategy's own keys on an account no other strategy claimed. The window always starts on
PAPER + DRY RUN. Real money is disabled in code (Valid has not signed off)."""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout,
                             QWidget)

from .. import live as LV
from .. import strategy as S
from .theme import ACCENT, GREEN, RED, TEXT, YELLOW, Kpi, big_money, box_style, make_table, muted, pct, set_row, spin, strip
from .workers import FnWorker, WorkerBag

LOG_COLORS = {"err": RED, "warn": YELLOW, "ok": GREEN, "trade": ACCENT, "info": TEXT}


class _Bridge(QObject):
    log = pyqtSignal(str, str)
    plan = pyqtSignal(object)
    stopped = pyqtSignal(str)


class LiveTab(QWidget):
    def __init__(self, settings, qs: QSettings, folder: str):
        super().__init__()
        self.sp, self.qs, self.folder = settings, qs, folder
        self.cls = settings.cls
        self.eng: Optional[LV.LiveEngine] = None
        self.plan: Optional[LV.Plan] = None
        self._workers = WorkerBag()
        self._busy = False
        self._loop: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.br = _Bridge()
        self.br.log.connect(self.add_log)
        self.br.plan.connect(self._planned)
        self.br.stopped.connect(self._loop_ended)
        root = QVBoxLayout(self)
        banner = QLabel(f"Valid has NOT signed off {self.cls.NAME} for capital. Run it DRY, then on paper. Keys come from 'Alpaca Key.txt' in the project folder "
                        f"({self.cls.PREFIX}_ALPACA_API_KEY / _SECRET_KEY would select a separate account). Real money is disabled in code.")
        banner.setObjectName("warn"); banner.setWordWrap(True)
        root.addWidget(banner)
        self.env = strip("PAPER account - DRY RUN unless ARMED", GREEN, bold=True)
        root.addWidget(self.env)
        top = QHBoxLayout()
        top.addWidget(self._account_box(), 3); top.addWidget(self._mode_box(), 4); top.addWidget(self._control_box(), 3)
        root.addLayout(top)
        names = ("Data bar", "Trade bar", "Capital", "Book", "Gross", "Beta after hedge", "Loop")
        k = QHBoxLayout()
        self.k = {n: Kpi(n) for n in names}
        for w in self.k.values():
            k.addWidget(w)
        root.addLayout(k)
        self.tabs = QTabWidget()
        self.t_orders = make_table(["Symbol", "Side", "Qty", "Now", "→ Target", "Ref px", "Notional", "Why", "OK", "Note"])
        self.tabs.addTab(self.t_orders, "Orders for this bar")
        self.t_book = make_table(["Symbol", "Kind", "Score", "Target w", "Price", "Now (shares)", "Target (shares)"])
        self.tabs.addTab(self.t_book, "Target book")
        self.t_pf = make_table(["Check", "Result", "Detail"])
        self.tabs.addTab(self.t_pf, "Pre-flight")
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumBlockCount(5000)
        self.tabs.addTab(self.log_box, "Log")
        root.addWidget(self.tabs, 1)
        self.armed.toggled.connect(self._arm_toggled)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(1000)

    # ── construction ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _account_box(self) -> QGroupBox:
        g = QGroupBox("Alpaca account"); f = QGridLayout(g)
        self.btn_connect = QPushButton("Connect"); self.btn_connect.clicked.connect(self.connect_)
        self.acct_lbl = muted("not connected")
        f.addWidget(QLabel("Paper account (simulated money)"), 0, 0, 1, 2)
        f.addWidget(self.btn_connect, 1, 0, 1, 2); f.addWidget(self.acct_lbl, 2, 0, 1, 2)
        return g

    def _mode_box(self) -> QGroupBox:
        g = QGroupBox("Mode & limits"); f = QFormLayout(g)
        self.armed = QCheckBox("ARM - allow sending orders")
        self.armed.setToolTip("Unticked = DRY RUN: every bar is planned and journaled, nothing is sent.")
        self.eq_frac = spin(0.05, 1.0, 0.9, 2, 0.05)
        self.cap = spin(0, 1e8, 0, 0, 5_000); self.cap.setPrefix("$ "); self.cap.setToolTip("0 = equity x fraction")
        self.min_trade = spin(0, 100_000, 100, 0, 50); self.min_trade.setPrefix("$ ")
        self.max_gross = spin(0.2, 4.0, 1.9, 2, 0.1)
        self.max_gross.setToolTip("Stocks + ETF hedges as a multiple of capital. A Reg-T margin account allows 2x overnight: keep this under 2.")
        self.halt = spin(0.0, 0.5, 0.03, 3, 0.005)
        self.dd = spin(0.0, 0.9, 0.15, 2, 0.01)
        self.etb = QCheckBox("Shorts only in easy-to-borrow names"); self.etb.setChecked(True)
        f.addRow(self.armed); f.addRow("Equity fraction", self.eq_frac); f.addRow("…or fixed capital", self.cap); f.addRow("Min ticket", self.min_trade)
        f.addRow("Max gross (stocks + ETFs, x capital)", self.max_gross); f.addRow("Daily-loss halt", self.halt); f.addRow("Drawdown stop (from high)", self.dd); f.addRow(self.etb)
        return g

    def _control_box(self) -> QGroupBox:
        g = QGroupBox("Run"); v = QVBoxLayout(g)
        self.btn_pf = QPushButton("Pre-flight check"); self.btn_pf.clicked.connect(self.preflight)
        self.btn_plan = QPushButton("Plan this bar now (dry run)"); self.btn_plan.setObjectName("primary"); self.btn_plan.clicked.connect(self.plan_now)
        self.btn_start = QPushButton("Start 15-minute loop"); self.btn_start.setObjectName("go"); self.btn_start.clicked.connect(self.start_loop)
        self.btn_stop = QPushButton("Stop loop"); self.btn_stop.clicked.connect(self.stop_loop); self.btn_stop.setEnabled(False)
        self.btn_train = QPushButton("Train model (latest data)"); self.btn_train.clicked.connect(self.train)
        self.btn_train.setVisible(hasattr(self.cls, "fit_latest"))
        self.btn_cmp = QPushButton("Compare last session with the backtest"); self.btn_cmp.clicked.connect(self.compare)
        self.btn_cmp.setToolTip("Rebuilds the backtest's decisions for the journaled bars from the SIP history (top up the data first) and compares targets and fills.")
        self.btn_hwm = QPushButton("Reset drawdown base"); self.btn_hwm.clicked.connect(self.reset_hwm)
        self.btn_kill = QPushButton("KILL: cancel orders + flatten this strategy"); self.btn_kill.setObjectName("danger"); self.btn_kill.clicked.connect(self.kill)
        for b in (self.btn_pf, self.btn_plan, self.btn_start, self.btn_stop, self.btn_train, self.btn_cmp, self.btn_hwm, self.btn_kill):
            v.addWidget(b)
        return g

    # ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def live_settings(self) -> LV.LiveSettings:
        return LV.LiveSettings(paper=True, armed=self.armed.isChecked(), capital=self.cap.value(), equity_fraction=self.eq_frac.value(), max_gross_x=self.max_gross.value(),
                               min_trade_usd=self.min_trade.value(), daily_loss_halt=self.halt.value(), dd_stop=self.dd.value(), require_etb=self.etb.isChecked())

    def engine(self) -> LV.LiveEngine:
        strat = self.sp.strategy(self.folder)
        ls = self.live_settings()
        if self.eng is None:
            self.eng = LV.LiveEngine(strat, ls, log=lambda m, k="info": self.br.log.emit(m, k))
        elif self._loop is None or not self._loop.is_alive():
            self.eng.s, self.eng.live = strat, ls
        return self.eng

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
            txt, ok = QInputDialog.getText(self, "Arm paper trading", f"{self.cls.NAME} will SEND ORDERS to its paper account at every decision bar.\n\nType  {LV.ARM_PHRASE}  to continue:")
            if not ok or txt.strip() != LV.ARM_PHRASE:
                self._set_armed(False)
                return
            eng = self.engine()
            probs = eng.arm_problems() if eng.acct else ["connect first"]
            if probs:
                self._set_armed(False)
                QMessageBox.warning(self, "Cannot arm", "\n".join(probs))
                return
        elif self._loop is not None and self._loop.is_alive() and self.eng is not None:
            self.eng.live.armed = False                                                     # disarming takes effect at once, even inside a running loop
        self.env.setText("PAPER account - ARMED: orders are sent at decision bars" if on else "PAPER account - DRY RUN unless ARMED")
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
        self.acct_lbl.setText(f"{a['number'][:4]}**** · {a['status']} · equity {big_money(a['equity'])} · buying power {big_money(a['buying_power'])} · margin {a['multiplier']}x · "
                              f"shorting {'on' if a['shorting'] else 'OFF'} · keys: {a['keys'].upper()} · market {'open' if a['market_open'] else 'closed'} · {a['latency_ms']} ms")
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
            if eng.tc is None:
                eng.connect()
            return eng.plan(force=True)
        self._run(fn, self._planned, "planning the current bar (dry run) ...")

    def _planned(self, plan: LV.Plan) -> None:
        self.plan = plan
        o, b = plan.orders, plan.book
        self.k["Data bar"].set(f"{plan.data_bar:%m-%d %H:%M}", None, "decision bar" if plan.decision else "not a decision bar")
        self.k["Trade bar"].set(f"{plan.trade_bar:%H:%M}", None, "EOD flatten" if plan.eod else "")
        self.k["Capital"].set(big_money(plan.capital), None, f"equity {big_money(plan.equity)}")
        nst = b[b.kind == "stock"] if len(b) else b
        self.k["Book"].set(f"{int((nst.tgt_qty > 0).sum()) if len(nst) else 0} L / {int((nst.tgt_qty < 0).sum()) if len(nst) else 0} S", None, f"{len(o)} orders")
        self.k["Gross"].set(f"{plan.expo.get('gross_stocks', 0):.2f}x", None, f"ETF hedge {plan.expo.get('gross_etf', 0):.2f}x")
        self.k["Beta after hedge"].set(f"{plan.expo.get('beta_after_hedge', 0):+.3f}")
        t = self.t_orders
        t.setRowCount(len(o))
        for r, x in enumerate(o.itertuples()):
            set_row(t, r, [x.symbol, x.side.upper(), f"{x.qty:,}", f"{x.cur_qty:+,}", f"{x.tgt_qty:+,}", f"{x.ref_px:,.2f}", f"${x.notional:,.0f}" if np.isfinite(x.notional) else "—",
                           x.reason, "yes" if x.ok else "NO", x.note], {1: GREEN if x.side == "buy" else RED, 8: None if x.ok else RED}, align_left=(7, 9))
        t = self.t_book
        b = b.sort_values("target_w") if len(b) else b
        t.setRowCount(len(b))
        for r, x in enumerate(b.itertuples()):
            set_row(t, r, [x.symbol, x.kind, f"{x.score:+.3f}" if np.isfinite(x.score) else "—", pct(x.target_w, 2), f"{x.px:,.2f}", f"{x.cur_qty:+,}", f"{x.tgt_qty:+,}"],
                    {3: GREEN if x.target_w > 0 else RED if x.target_w < 0 else None})
        for n in plan.notes:
            self.add_log(n, "warn")
        self.tabs.setCurrentWidget(self.t_orders)

    def start_loop(self) -> None:
        if self._loop is not None and self._loop.is_alive():
            return
        eng = self.engine()
        if self.armed.isChecked():
            if QMessageBox.question(self, "Start ARMED loop", f"Start the 15-minute loop ARMED on the paper account?\nIt sends orders at every decision bar until you stop it.") \
                    != QMessageBox.StandardButton.Yes:
                return
        self._stop.clear()

        def body():
            msg = ""
            try:
                eng.run(self._stop, on_plan=lambda p: self.br.plan.emit(p))
            except Exception as e:                                                         # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
            self.br.stopped.emit(msg)
        self._loop = threading.Thread(target=body, name="resarb-loop", daemon=True)
        self._loop.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.sp.setEnabled(False)

    def stop_loop(self) -> None:
        self._stop.set()
        self.add_log("stopping the loop (finishes the current step) ...")

    def _loop_ended(self, msg: str) -> None:
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.sp.setEnabled(True)
        if msg:
            self.add_log("loop ended: " + msg, "err")

    def train(self) -> None:
        strat = self.sp.strategy(self.folder)

        def fn(log):
            end = pd.Timestamp.now()
            ctx = S.context(strat, start=(end - pd.DateOffset(years=int(float(strat.v["train_years"])) + 1)).strftime("%Y-%m-%d"), log=log, cache=False)
            return strat.fit_latest(ctx, log)
        self._run(fn, lambda meta: self.add_log(f"model trained: {meta['cells']:,} cells through {meta['data_end'][:16]}", "ok"), "training the model on the latest data ...")

    def compare(self) -> None:
        strat = self.sp.strategy(self.folder)
        from .. import compare as C

        def fn(log):
            return C.compare(strat, "paper", log=log)

        def done(out):
            s = out["summary"]
            self.add_log(f"live vs backtest {s['day']}: {s['decisions']} decisions · score corr {s['score_corr']:.2f} · target corr {s['target_corr']:.2f} · same side "
                         f"{s['same_side']:.0%} · target difference {s['diff_gross']:.0%} of gross · {s['fills']} fills · slippage vs ref {s['slip_vs_ref_bp']:+.1f} bp, vs the backtest's "
                         f"open {s['slip_vs_bt_open_bp']:+.1f} bp · latency {s['median_latency_s']:.1f} s (details: reports/compare_*.csv)", "ok")
        self._run(fn, done, "comparing the last session with the backtest ...")

    def reset_hwm(self) -> None:
        if QMessageBox.question(self, "Reset drawdown base", "Make the current equity the new high-water mark (and clear a drawdown stop)?") != QMessageBox.StandardButton.Yes:
            return
        eng = self.engine()

        def fn(log):
            if eng.tc is None:
                eng.connect()
            return eng.reset_drawdown()
        self._run(fn, lambda v: self.add_log(f"drawdown base reset to ${v:,.0f}", "ok"), "resetting ...")

    def kill(self) -> None:
        txt, ok = QInputDialog.getText(self, "KILL", f"Cancel {self.cls.NAME}'s open orders and CLOSE every position in its symbols on its paper account.\n\nType  KILL  to continue:")
        if not ok or txt.strip() != "KILL":
            return
        self.stop_loop()
        eng = self.engine()
        self._run(lambda log: eng.kill(), lambda res: self.add_log("KILL: " + "; ".join(f"{r['symbol']}:{r['status']}" for r in res[:40]), "warn"), "KILL ...")

    def _tick(self) -> None:
        now = LV.now_et()
        alive = self._loop is not None and self._loop.is_alive()
        nxt = LV.bar_start(now) + pd.Timedelta(minutes=15)
        self.k["Loop"].set("running" if alive else "stopped", GREEN if alive else None, f"next step {nxt:%H:%M}:25 ET" if alive else f"{now:%a %H:%M} ET")

    def shutdown(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.join(5)

    # ── persistence ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def save(self) -> None:
        for k, w in (("live/eq_frac", self.eq_frac), ("live/cap", self.cap), ("live/min_trade", self.min_trade), ("live/max_gross", self.max_gross), ("live/halt", self.halt), ("live/dd", self.dd)):
            self.qs.setValue(k, w.value())

    def load(self) -> None:
        for k, w in (("live/eq_frac", self.eq_frac), ("live/cap", self.cap), ("live/min_trade", self.min_trade), ("live/max_gross", self.max_gross), ("live/halt", self.halt), ("live/dd", self.dd)):
            try:
                v = self.qs.value(k)
                if v is not None:
                    w.setValue(float(v))
            except (TypeError, ValueError):
                pass
