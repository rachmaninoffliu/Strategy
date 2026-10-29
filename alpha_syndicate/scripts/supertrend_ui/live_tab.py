"""Live trading tab: connect to Alpaca, pre-flight, run the live Engine (dry run / paper / live), watch the signal, orders, positions and the session's performance.

The trading logic is supertrend_gui.Engine, unchanged: it runs in its own thread and reports through a queue, which this tab polls. Nothing here sends an order except the Engine
(on Start, when ARMED) and the explicit Flatten / KILL / Order-path-test buttons, each behind a confirmation."""
from __future__ import annotations

import json
import math
import os
import queue
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd
from PyQt6.QtCore import QDate, QSettings, Qt, QTimer
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (QApplication, QCheckBox, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                             QVBoxLayout, QWidget)

import supertrend_gui as GU

from . import data as D
from . import metrics as M
from . import params as P
from . import preflight as PF
from .perf_view import ParamsView, PerformanceView, side_color
from .preflight_dialog import PreflightDialog
from .settings_panel import SettingsPanel
from .theme import ACCENT, COL_FLAT, GREEN, MUTED, RED, TEXT, YELLOW, Kpi, QComboBox, QDateEdit, QSpinBox, big_money, box_style, make_table, muted, set_row, spin
from .workers import FnWorker, WorkerBag

LOG_COLORS = {"err": RED, "warn": YELLOW, "ok": GREEN, "trade": ACCENT, "info": TEXT}
MAX_LOG = 2500


def strategy_side(symbol: Optional[str] = None, side: Optional[str] = None) -> str:
    return GU.strategy_side(symbol, side)


class LiveTab(QWidget):
    def __init__(self, settings: SettingsPanel, qs: QSettings, reports_dir: Callable[[], str] = lambda: "."):
        super().__init__()
        self.sp, self.qs = settings, qs
        self.q: queue.Queue = queue.Queue()
        self.engine: Optional[GU.Engine] = None
        self.running = False
        self._run: Optional[dict] = None                         # the session's frozen settings (values, LiveSettings, paper, armed)
        self._workers = WorkerBag()
        self._popups: list = []
        self._acct: Optional[dict] = None                        # last connect result
        self._connected_paper: Optional[bool] = None
        self._capital_pinned = False
        self._cap_programmatic = False
        self._equity_now: Optional[float] = None
        self._day_base: Optional[float] = None
        self._held: dict = {}
        self._sess: Optional[dict] = None
        self._openpos: Optional[dict] = None                     # last account-vs-backtest open position (engine event "openpos")
        self._sig: Optional[dict] = None
        self._live_rows: list = []
        self._session_capital: Optional[float] = None
        self._session_start: Optional[datetime] = None
        self._log_entries: list = []
        self._dry_seen: set = set()
        self._pf_report = None
        self._pf_running = False
        self._pf_dlg: Optional[PreflightDialog] = None
        self._auto_key: Optional[str] = None
        self._stop_deadline: Optional[float] = None
        self._history_busy = False
        self.get_bt_settings: Callable = lambda: (None, "")       # set by MainWindow
        self.reports_dir = reports_dir
        root = QVBoxLayout(self)

        banner = QLabel("Heads-up: Valid has not signed off · the backtest does not reconcile with the TradingView run · factor 5.0 is a spike · SOXL / SOXS are 3x leveraged. Start dry, then paper, then small.")
        banner.setObjectName("warn"); banner.setWordWrap(True)
        root.addWidget(banner)
        self.env_lbl = QLabel(""); self.env_lbl.setWordWrap(True)
        root.addWidget(self.env_lbl)

        top = QHBoxLayout()
        self.setup = QWidget(); sl = QHBoxLayout(self.setup); sl.setContentsMargins(0, 0, 0, 0)       # account + live-only settings: locked while a session runs, so they make room for the results
        sl.addWidget(self._account_box(), 1); sl.addWidget(self._live_box(), 1)
        self.run_card = QFrame(); self.run_card.setObjectName("card")
        rv = QVBoxLayout(self.run_card); rv.setContentsMargins(12, 8, 12, 8)
        self.run_title = QLabel("This session"); self.run_title.setStyleSheet("font-weight: 700; font-size: 13px;")
        self.using_lbl = muted("")
        self.btn_setup = QPushButton("Show setup"); self.btn_setup.setCheckable(True); self.btn_setup.setToolTip("Show the account and live-only settings (read-only while the session runs).")
        self.btn_setup.toggled.connect(lambda on: (self.setup.setVisible(on or not self.running), self.btn_setup.setText("Hide setup" if on else "Show setup")))
        rv.addWidget(self.run_title); rv.addWidget(self.using_lbl, 1); rv.addWidget(self.btn_setup, 0, Qt.AlignmentFlag.AlignLeft)
        self.run_card.hide()
        top.addWidget(self.setup, 6); top.addWidget(self.run_card, 6); top.addWidget(self._control_box(), 2)
        root.addLayout(top)

        k = QHBoxLayout()
        self.kpis = {n: Kpi(n) for n in ("Next bar closes in", "Trend (last closed bar)", "Position", "Trades today", "Day P&L", "Market")}
        for w in self.kpis.values():
            k.addWidget(w)
        root.addLayout(k)
        card = QFrame(); card.setObjectName("card")
        cv = QVBoxLayout(card); cv.setContentsMargins(10, 6, 10, 6); cv.setSpacing(2)
        self.sig_lbl = QLabel("Signal: start a session (a dry run is fine) or press Refresh in the Signal monitor tab.")
        self.plan_lbl = QLabel("Next order: —")
        self.lvl_lbl = QLabel("Position levels: no open position.")
        for w in (self.sig_lbl, self.plan_lbl, self.lvl_lbl):
            w.setWordWrap(True); w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            cv.addWidget(w)
        self.sig_lbl.setStyleSheet(f"color: {TEXT};"); self.plan_lbl.setObjectName("muted"); self.lvl_lbl.setObjectName("muted")
        root.addWidget(card)

        self.perf = PerformanceView("live_session_trades.csv", reports_dir)
        self.perf.chart.message("The session's equity curve appears here after Connect (history) or once a session is running")
        self.orders = make_table(["Time", "Account", "Symbol", "Action", "Qty", "Type", "Limit", "Result", "Why"])
        self.perf.tabs.addTab(self.orders, "Orders")
        self.perf.tabs.addTab(self._log_page(), "Log")
        self.params_view = ParamsView()
        self.perf.tabs.addTab(self.params_view, "Parameters in use")
        hist = QHBoxLayout()
        self.hist_lbl = muted("")
        self.btn_hist = QPushButton("Reload history")
        self.btn_hist.setToolTip("Read the broker's fills since the start date (or, while the account has none, simulate the strategy over the same window) and show them below.")
        self.btn_hist.clicked.connect(self.reload_history)
        hist.addWidget(self.hist_lbl, 1); hist.addWidget(self.btn_hist)
        root.addLayout(hist)
        root.addWidget(self.perf, 1)

        self.poll = QTimer(self); self.poll.timeout.connect(self._drain); self.poll.start(150)
        self.clock = QTimer(self); self.clock.timeout.connect(self._clock_tick); self.clock.start(1000)
        self.sp.changed.connect(self._refresh_using)
        for w in (self.account, self.mode, self.short_route):
            w.currentIndexChanged.connect(lambda *_: (self._refresh_env(), self._refresh_using(), self._hint()))
        self.capital.valueChanged.connect(self._capital_edited)
        self.capital.valueChanged.connect(lambda *_: self._refresh_using())
        self._refresh_env(); self._refresh_using(); self._hint(); self._update_perf(); self._clock_tick()

    # ══ construction ═════════════════════════════════════════════════════════════
    def _account_box(self) -> QGroupBox:
        g = QGroupBox("Alpaca account")
        f = QGridLayout(g)
        self.account = QComboBox(); self.account.addItems(["Paper account (play money)", "LIVE account (real money)"])
        self.account.setToolTip("Which Alpaca account the engine reads and (when ARMED) sends orders to. Keys come from the project's key file 'Alpaca Key.txt' in the project folder (then ALPACA_API_KEY / ALPACA_SECRET_KEY, then ~/.claude.json). Always starts on Paper.")
        self.btn_connect = QPushButton("Connect and test"); self.btn_connect.setObjectName("primary"); self.btn_connect.clicked.connect(self.connect_account)
        self.btn_connect.setToolTip("Read-only check of the keys, the account, the market clock and the SOXL / SOXS instruments. Sends nothing.")
        self.btn_test = QPushButton("Order-path test"); self.btn_test.setEnabled(False)
        self.btn_test.setToolTip("Sends ONE 1-share DAY limit BUY for SOXL at half the market price (it cannot fill) and cancels it straight away: proves the keys, signing and the order format\n"
                                 "end to end on the selected account. Needs Connect first.")
        self.btn_test.clicked.connect(self.order_path_test)
        self.acct = QLabel("Not connected"); self.acct.setObjectName("muted"); self.acct.setWordWrap(True)
        self.bp = QLabel("—"); self.bp.setWordWrap(True)                # wrapping labels: long account text must never widen the page
        self.health = muted(""); self.health.setMinimumHeight(58); self.health.setAlignment(Qt.AlignmentFlag.AlignTop)
        f.addWidget(QLabel("Account"), 0, 0); f.addWidget(self.account, 0, 1, 1, 2)
        f.addWidget(self.btn_connect, 1, 0); f.addWidget(self.btn_test, 1, 1, 1, 2)
        f.addWidget(QLabel("Status"), 2, 0); f.addWidget(self.acct, 2, 1, 1, 2)
        f.addWidget(QLabel("Equity / buying power"), 3, 0); f.addWidget(self.bp, 3, 1, 1, 2)
        f.addWidget(self.health, 4, 0, 1, 3)
        return g

    def _live_box(self) -> QGroupBox:
        g = QGroupBox("Live-only settings (not used by the Backtest)")
        f = QFormLayout(g)
        self.capital = spin(1, 1e12, 100_000, 2, 1000); self.capital.setPrefix("$ ")
        self.capital.setToolTip("The session's STARTING equity: it is compounded by the session's P&L and the running figure - not the raw account balance - sizes every order.")
        self.btn_cap = QPushButton("Default"); self.btn_cap.setFixedWidth(70)
        self.btn_cap.setToolTip("Use the default again: the account's equity once connected, otherwise $100,000")
        self.btn_cap.clicked.connect(self._capital_default)
        self.cap_note = muted("")
        cap_row = QWidget(); cr = QHBoxLayout(cap_row); cr.setContentsMargins(0, 0, 0, 0); cr.addWidget(self.capital, 1); cr.addWidget(self.btn_cap)
        self.sdate = QDateEdit(QDate.currentDate()); self.sdate.setCalendarPopup(True); self.sdate.setDisplayFormat("yyyy-MM-dd")
        self.sdate.setToolTip("The session's P&L baseline. While the account has no fills since then, the engine re-simulates the strategy from here on EVERY poll to know its running equity - keep it recent.")
        self.mode = QComboBox(); self.mode.addItems(["DRY RUN (send nothing)", "ARMED (send orders)"])
        self.mode.setToolTip("DRY RUN shows every order the engine would send and sends none - it is the default every time the app starts.\nARMED sends orders to the account chosen on the left "
                             "(PAPER asks for a yes, LIVE needs you to type a phrase).")
        self.short_route = QComboBox(); self.short_route.addItems(["SOXL short, else SOXS", "Always SOXS"])
        self.short_route.setToolTip("How a short signal is held:\n• 'SOXL short, else SOXS': a direct SOXL short when the broker allows it, otherwise a long SOXS position.\n• 'Always SOXS': never short SOXL directly.\n"
                                    "SOXS is a separate 3x fund that does not mirror SOXL exactly, and its exits are POLLED by this app (nothing rests at the broker).")
        self.dloss = spin(0.5, 50, 8.0, 1, 0.5, " %")
        self.dloss.setToolTip("New entries pause when the account is down this much on the day. Open positions keep their exits.")
        self.maxtr = QSpinBox(); self.maxtr.setRange(1, 100); self.maxtr.setValue(12)
        self.maxtr.setToolTip("Circuit breaker on churn: no new orders after this many in a day.")
        self.chk_pop = QCheckBox("Pop up a window for every order sent"); self.chk_pop.setChecked(True)
        f.addRow("Starting capital", cap_row); f.addRow(self.cap_note); f.addRow("Start date (P&L baseline)", self.sdate); f.addRow("Orders", self.mode)
        f.addRow("Short route", self.short_route); f.addRow("Daily loss limit", self.dloss); f.addRow("Max trades / day", self.maxtr); f.addRow(self.chk_pop)
        f.addRow(muted("Strategy, sizing and exits come from the shared left panel."))
        return g

    def _control_box(self) -> QGroupBox:
        g = QGroupBox("Control")
        v = QVBoxLayout(g)
        self.btn_pf = QPushButton("🔎  Pre-flight check (go / no-go)"); self.btn_pf.clicked.connect(lambda: self.run_preflight(then_start=False))
        self.btn_pf.setToolTip("Reads the account, market, data and your settings and compares them with what the strategy assumes. Never sends an order.")
        self.btn_start = QPushButton("▶  Start"); self.btn_start.setObjectName("go"); self.btn_start.clicked.connect(self.start)
        self.btn_stop = QPushButton("■  Stop"); self.btn_stop.clicked.connect(self.stop); self.btn_stop.setEnabled(False)
        self.btn_flat = QPushButton("Flatten positions"); self.btn_flat.clicked.connect(self.flatten)
        self.btn_flat.setToolTip("Stops the engine first (otherwise it would re-enter on its next poll), then cancels all open orders and closes every position at market.")
        self.btn_kill = QPushButton("⛔  KILL — stop, flatten and latch"); self.btn_kill.setObjectName("danger"); self.btn_kill.clicked.connect(self.kill)
        self.btn_kill.setToolTip("Stop the engine, cancel every order, close every position and write the KILL file so nothing can restart until you clear it.")
        self.btn_unlatch = QPushButton("Clear kill latch"); self.btn_unlatch.clicked.connect(self.clear_latch); self.btn_unlatch.hide()
        self.state = muted("Idle")
        self.pf_lbl = muted("Pre-flight: not run yet")
        self.chk_pf = QCheckBox("Pre-flight before Start"); self.chk_pf.setChecked(str(self.qs.value("live/preflight_on_start", "true")).lower() in ("true", "1"))
        self.chk_pf.setToolTip("Run the pre-flight check (and show its report) every time you press Start.")
        self.auto_pf = QCheckBox("Auto pre-flight 09:10 ET"); self.auto_pf.setToolTip("Runs the pre-flight check by itself at 09:10 ET on market days (the app must be open). It never starts a session.")
        self.auto_pf.setChecked(str(self.qs.value("live/auto_preflight", "true")).lower() in ("true", "1"))
        self.hint = QLabel(""); self.hint.setWordWrap(True); self.hint.setStyleSheet(box_style(ACCENT, 0.12, pad=6))
        grid = QGridLayout(); grid.setSpacing(6)
        grid.addWidget(self.btn_start, 0, 0); grid.addWidget(self.btn_stop, 0, 1); grid.addWidget(self.btn_flat, 1, 0); grid.addWidget(self.btn_kill, 1, 1)
        self.btn_kill.setText("⛔  KILL")
        for w in (self.hint,):
            v.addWidget(w)
        v.addWidget(self.btn_pf)
        v.addLayout(grid)
        for w in (self.btn_unlatch, self.pf_lbl, self.chk_pf, self.auto_pf, self.state):
            v.addWidget(w)
        v.addStretch(1)
        return g

    def _log_page(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.log_filter = QComboBox(); self.log_filter.addItems(["Everything", "Warnings and errors", "Orders and trades"])
        self.log_filter.currentIndexChanged.connect(lambda *_: self._render_log())
        self.chk_scroll = QCheckBox("auto-scroll"); self.chk_scroll.setChecked(True)
        b_clear, b_copy, b_open = QPushButton("Clear"), QPushButton("Copy all"), QPushButton("Open log file")
        b_clear.clicked.connect(lambda: (self._log_entries.clear(), self._render_log()))
        b_copy.clicked.connect(lambda: QApplication.clipboard().setText(self.logbox.toPlainText()))
        b_open.clicked.connect(self.open_log_file)
        for x in (QLabel("Show"), self.log_filter, self.chk_scroll):
            row.addWidget(x)
        row.addStretch(1)
        for b in (b_clear, b_copy, b_open):
            row.addWidget(b)
        self.logbox = QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumBlockCount(MAX_LOG + 50)
        v.addLayout(row); v.addWidget(self.logbox, 1)
        v.addWidget(muted("Repeated identical lines are folded into one with a ×count. The full history is in reports/live_log.jsonl."))
        return w

    # ══ log ══════════════════════════════════════════════════════════════════════
    def add_log(self, msg: str, level: str = "info", ts: Optional[str] = None) -> None:
        ts = ts or datetime.now().strftime("%H:%M:%S")
        last = self._log_entries[-1] if self._log_entries else None
        if last and last["msg"] == msg and last["level"] == level:
            last["n"] += 1; last["ts"] = ts
            self._render_log(last_only=True)
            return
        self._log_entries.append({"ts": ts, "msg": msg, "level": level, "n": 1})
        del self._log_entries[:-MAX_LOG]
        self._render_log(append=True)

    log = add_log

    def _visible(self, e: dict) -> bool:
        i = self.log_filter.currentIndex()
        return i == 0 or (i == 1 and e["level"] in ("err", "warn")) or (i == 2 and (e["level"] == "trade" or "order" in e["msg"].lower() or "DRY RUN" in e["msg"]))

    @staticmethod
    def _line(e: dict) -> str:
        return f"[{e['ts']}] {e['msg']}" + (f"   (×{e['n']})" if e["n"] > 1 else "")

    def _write(self, cur: QTextCursor, e: dict) -> None:
        fmt = QTextCharFormat(); fmt.setForeground(QColor(LOG_COLORS.get(e["level"], TEXT)))
        cur.insertText(self._line(e), fmt)

    def _render_log(self, append: bool = False, last_only: bool = False) -> None:
        box = self.logbox
        if append or last_only:
            e = self._log_entries[-1]
            if not self._visible(e):
                return
            cur = box.textCursor(); cur.movePosition(QTextCursor.MoveOperation.End)
            if last_only:
                cur.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
            elif box.document().characterCount() > 1:
                cur.insertBlock()
            self._write(cur, e)
        else:
            box.clear()
            cur = box.textCursor()
            first = True
            for e in self._log_entries:
                if self._visible(e):
                    if not first:
                        cur.insertBlock()
                    self._write(cur, e); first = False
        if self.chk_scroll.isChecked():
            box.verticalScrollBar().setValue(box.verticalScrollBar().maximum())

    def open_log_file(self) -> None:
        try:
            if os.path.exists(GU.LOG_FILE):
                os.startfile(GU.LOG_FILE)                          # noqa: S606 (Windows)
            else:
                QMessageBox.information(self, "Log file", f"No log file yet ({GU.LOG_FILE}).")
        except Exception as exc:                                  # noqa: BLE001
            QMessageBox.information(self, "Log file", f"{GU.LOG_FILE}\n\n{exc}")

    # ══ settings → objects ═══════════════════════════════════════════════════════
    def live_settings(self) -> P.LiveSettings:
        return P.LiveSettings(capital=self.capital.value(), start_date=self.sdate.date().toString("yyyy-MM-dd"), paper=self.account.currentIndex() == 0,
                              armed=self.mode.currentIndex() == 1, prefer_short=self.short_route.currentIndex() == 0, daily_loss_pct=self.dloss.value(), max_trades=int(self.maxtr.value()))

    def mode_label(self, ls: P.LiveSettings) -> str:
        return ("Alpaca PAPER" if ls.paper else "Alpaca LIVE (real money)") + (" · orders SENT" if ls.armed else " · DRY RUN (nothing sent)")

    def _refresh_env(self) -> None:
        ls = self.live_settings()
        if not ls.armed:
            txt, col, a, bold = (f"DRY RUN on the {'PAPER' if ls.paper else 'LIVE'} account: the engine reads the market and the account, shows every order it WOULD send and sends none.", MUTED, 0.12, False)
        elif ls.paper:
            txt, col, a, bold = ("PAPER, ARMED — orders are sent to your Alpaca PAPER account (play money).", GREEN, 0.14, False)
        else:
            txt, col, a, bold = ("⚠ LIVE, ARMED — orders are sent to your REAL Alpaca account and use REAL MONEY.", RED, 0.18, True)
        self.env_lbl.setText(txt)
        self.env_lbl.setStyleSheet(box_style(col, a, bold, pad=6))
        if self._connected_paper is not None and self._connected_paper != ls.paper:          # a connection belongs to one account
            self._acct = None; self._connected_paper = None
            self.acct.setText("Not connected"); self.bp.setText("—"); self.health.setText(""); self.btn_test.setEnabled(False)
            self._sync_capital()

    def _refresh_using(self) -> None:
        try:
            ls = self.live_settings()
            v = self._run["values"] if (self.running and self._run) else self.sp.values()
            lsx = self._run["ls"] if (self.running and self._run) else ls
            txt = (f"{'THIS RUNNING SESSION uses' if self.running else 'A session started now would use'}: {self.mode_label(lsx)} | {self.sp.preset_name()} | {v['strategyMode']}, ATR {v['atrPeriod']:g}, "
                   f"factor {v['factor']:g}, {v['equity_frac']:g}% of running equity per trade (shorts x{v['short_mult']:g}) | live-only: capital ${lsx.capital:,.0f} from {lsx.start_date}, "
                   f"-{lsx.daily_loss_pct:g}% daily loss, {lsx.max_trades} trades/day. Full list: tab \"Parameters in use\".")
            self.using_lbl.setText(txt)
            self.env_lbl.setToolTip(txt)
            self.run_title.setText("This session — " + self.mode_label(lsx))
        except Exception as e:                                    # noqa: BLE001
            self.using_lbl.setText(f"Settings problem: {e}")

    def params_state(self):
        """(rows, frozen?) for the 'Parameters in use' table: the running session's frozen settings if one is running, else what Start would use now."""
        bt, err = self.get_bt_settings()
        if self.running and self._run:
            vals, ls = self._run["values"], self._run["ls"]
        else:
            vals, ls = self.sp.values(), self.live_settings()
        return P.describe_parameters(vals, bt, ls, self.mode_label(ls), err), self.running

    # ── capital follows the account (until you type your own) ──
    def _capital_edited(self, *_a) -> None:
        if not self._cap_programmatic:
            self._capital_pinned = True
            self._sync_capital()

    def _capital_default(self) -> None:
        self._capital_pinned = False
        self._sync_capital()

    def _sync_capital(self) -> None:
        if self.running:
            return
        eq = self._acct["equity"] if self._acct else None
        value = 100_000.0 if eq is None else max(1.0, math.floor(eq * 100) / 100)
        if eq is None:
            self.cap_note.setText("Default $100,000 (a fresh paper account). Once connected it becomes the account's equity." + ("  (You set this by hand.)" if self._capital_pinned else ""))
        elif self._capital_pinned:
            self.cap_note.setText(f"You set this by hand. The account holds ${eq:,.2f} - the button restores that.")
        else:
            self.cap_note.setText(f"Default = the account's equity ${eq:,.2f}. Type a smaller amount to risk less; the button restores the default.")
        if not self._capital_pinned:
            self._cap_programmatic = True
            try:
                self.capital.setValue(value)
            finally:
                self._cap_programmatic = False

    # ══ connect ══════════════════════════════════════════════════════════════════
    def connect_account(self) -> None:
        paper = self.account.currentIndex() == 0
        self.btn_connect.setEnabled(False)
        self.acct.setText("Connecting…")

        def work(_log):
            k, s = GU.creds(paper)
            t0 = time.time()
            tc = GU.TradingClient(k, s, paper=paper)
            a = tc.get_account()
            lat = (time.time() - t0) * 1000
            out = dict(paper=paper, base=str(getattr(tc, "_base_url", "")), lat=lat, number=str(a.account_number)[:4] + "****", status=str(a.status), equity=float(a.equity),
                       last_equity=float(getattr(a, "last_equity", 0) or 0), bp=float(a.buying_power), shorting=bool(getattr(a, "shorting_enabled", True)),
                       pdt=bool(getattr(a, "pattern_day_trader", False)), dtc=int(float(getattr(a, "daytrade_count", 0) or 0)), assets={}, clock=None, positions=None, orders=None)
            try:
                c = tc.get_clock(); out["clock"] = (bool(c.is_open), str(c.next_open)[:16], str(c.next_close)[:16])
            except Exception:                                      # noqa: BLE001
                pass
            for sym in (GU.SYMBOL_LONG, GU.SYMBOL_SHORT):
                try:
                    x = tc.get_asset(sym)
                    out["assets"][sym] = dict(tradable=bool(getattr(x, "tradable", True)), shortable=bool(getattr(x, "shortable", False)), etb=bool(getattr(x, "easy_to_borrow", True)))
                except Exception:                                  # noqa: BLE001
                    pass
            try:
                held_ = [p for p in tc.get_all_positions() if abs(float(p.qty)) > 1e-9]
                out["positions"] = [(p.symbol, float(p.qty)) for p in held_]
                out["pos_detail"] = {p.symbol: dict(qty=float(p.qty), side=p.side.value, avg=float(p.avg_entry_price), mv=float(p.market_value), upl=float(p.unrealized_pl),
                                                    uplpc=float(p.unrealized_plpc) * 100, cur=float(p.current_price)) for p in held_}
                out["orders"] = len(tc.get_orders(GU.GetOrdersRequest(status=GU.QueryOrderStatus.OPEN)))
            except Exception:                                      # noqa: BLE001
                pass
            return out

        self._workers.start(FnWorker(work), self._connected)

    def _connected(self, res, err) -> None:
        self.btn_connect.setEnabled(not self.running)
        if err:
            self._acct = None; self._connected_paper = None
            self.acct.setText("Connection failed"); self.bp.setText("—")
            self.health.setText(f"❌ {err}")
            self.btn_test.setEnabled(False)
            self.add_log(f"connect failed: {err}", "err")
            self._hint()
            return
        self._acct, self._connected_paper = res, res["paper"]
        self._equity_now = res["equity"]
        self._day_base = res["last_equity"] or res["equity"]
        self.acct.setText(f"✅ {res['number']} · {res['status'].split('.')[-1]} · {'paper' if res['paper'] else 'LIVE'}")
        self.bp.setText(f"${res['equity']:,.2f}   /   buying power ${res['bp']:,.2f}")
        bits = [f"API latency {res['lat']:.0f} ms"]
        a_long = res["assets"].get(GU.SYMBOL_LONG, {})
        if a_long:
            ok = a_long.get("shortable") and res["shorting"]
            bits.append("SOXL: " + ("shortable" + ("" if a_long.get("etb") else ", hard to borrow") if ok else "NOT shortable → shorts go via SOXS"))
        if res["clock"]:
            bits.append(("market OPEN, closes " if res["clock"][0] else "market CLOSED, opens ") + (res["clock"][2] if res["clock"][0] else res["clock"][1]))
        if res["positions"] is not None:
            bits.append(f"{len(res['positions'])} open position(s)" + (": " + ", ".join(f"{s} {q:+,.0f}" for s, q in res["positions"]) if res["positions"] else "") + f", {res['orders']} resting order(s)")
        if res["pdt"]:
            bits.append("⚠ flagged pattern day trader")
        if not res["shorting"]:
            bits.append("⚠ shorting disabled")
        self.health.setText("   |   ".join(bits))
        self.btn_test.setEnabled(True)
        if res.get("pos_detail") is not None and not self.running:
            self._show_idle_position(res["pos_detail"])
        self.add_log(f"connected to Alpaca {'paper' if res['paper'] else 'LIVE'} {res['number']} (read-only checks passed)", "ok")
        self._sync_capital()
        self._on_account({"equity": res["equity"]})
        self._hint()
        self.reload_history()

    def _show_idle_position(self, pos: dict) -> None:
        """No session is running: show what the account holds right now (read-only), with the exit levels this app froze for it, if any."""
        probe = GU.Engine(queue.Queue(), True, 0.0, 1.0, "2000-01-01", False)         # never started: only its level maths and state-file reader are used
        live = probe.live_open({"mon_bar": None, "adx": None}, pos) if pos else None
        self.perf.show_open(live, "Account holds (no session running)", "FLAT - the account holds nothing")

    def order_path_test(self) -> None:
        if not self._acct:
            return
        paper = self._acct["paper"]
        if QMessageBox.question(self, "Order-path test", f"Send ONE 1-share DAY limit BUY for SOXL at half the market price to the Alpaca {'PAPER' if paper else 'LIVE (real money)'} account, "
                                                          "then cancel it straight away?\n\nIt cannot fill at that price. It proves the keys, signing and the order format end to end.") != QMessageBox.StandardButton.Yes:
            return

        def work(log):
            k, s = GU.creds(paper)
            tc, dc = GU.TradingClient(k, s, paper=paper), GU.StockHistoricalDataClient(k, s)
            q = dc.get_stock_latest_quote(GU.StockLatestQuoteRequest(symbol_or_symbols=[GU.SYMBOL_LONG]))[GU.SYMBOL_LONG]
            px = float(q.ask_price or q.bid_price or 0)
            if px <= 0:
                raise RuntimeError("no SOXL quote available to price the test order")
            lim = max(0.01, round(px * 0.5, 2))
            o = tc.submit_order(GU.LimitOrderRequest(symbol=GU.SYMBOL_LONG, qty=1, side=GU.OrderSide.BUY, limit_price=lim, time_in_force=GU.TimeInForce.DAY))
            time.sleep(1.0)
            st1 = str(tc.get_order_by_id(o.id).status)
            tc.cancel_order_by_id(o.id)
            time.sleep(1.0)
            st2 = str(tc.get_order_by_id(o.id).status)
            return dict(id=str(o.id), limit=lim, px=px, st1=st1, st2=st2)

        def done(res, err):
            self.add_log("order-path test: " + (f"FAILED {err}" if err else f"order {res['id'][:8]} accepted (limit {res['limit']:.2f} vs market {res['px']:.2f}, status {res['st1']}), "
                                                                           f"cancelled (status {res['st2']}) - keys, signing and order format are fine"), "err" if err else "ok")
        self._workers.start(FnWorker(work), done)

    # ══ history ══════════════════════════════════════════════════════════════════
    def reload_history(self) -> None:
        """The session's trades without running a session: broker fills since the start date, or the simulation when the account has none."""
        if self.running or self._history_busy or self._acct is None:
            return
        paper = self._acct["paper"]
        ls = self.live_settings()
        try:
            cfg = P.to_config(self.sp.values())
        except Exception:                                         # noqa: BLE001
            return
        self._history_busy = True
        self.btn_hist.setEnabled(False)
        self.hist_lbl.setText("Reading the history …")
        GU.CFG, GU.SHORT_EQUITY_MULT = cfg, cfg.short_mult               # the engine's simulation reads these module settings; set here on the GUI thread, never from a worker mid-session

        def work(log):
            k, s = GU.creds(paper)
            eng = GU.Engine(queue.Queue(), paper, ls.capital, cfg.equity_frac, ls.start_date, False)
            eng.tclient, eng.dclient = GU.TradingClient(k, s, paper=paper), GU.StockHistoricalDataClient(k, s)
            acct = eng.tclient.get_account()
            return eng.session_equity(float(acct.equity)), ls, cfg.equity_frac

        self._workers.start(FnWorker(work), self._history_ready)

    def _history_ready(self, res, err) -> None:
        self._history_busy = False
        self.btn_hist.setEnabled(True)
        if err:
            self.hist_lbl.setText(f"⚠ could not read the history: {err}")
            return
        sess, ls, frac = res
        if not self.running:
            self._session_capital = ls.capital
            self._session_start = datetime.strptime(ls.start_date, "%Y-%m-%d")
            self._live_rows = sess["trades"]
            self._on_session({**{k: v for k, v in sess.items() if k != "trades"}, "frac": frac, "start": ls.start_date})

    # ══ pre-flight ═══════════════════════════════════════════════════════════════
    def start(self) -> None:
        if self.chk_pf.isChecked():
            self.run_preflight(then_start=True)
        else:
            self._start_session(None)

    def run_preflight(self, then_start: bool, auto: bool = False) -> None:
        if self._pf_running or self.running:
            if not auto:
                QMessageBox.information(self, "Pre-flight", "A pre-flight check or a session is already running.")
            return
        try:
            values = self.sp.values()
            P.to_config(values)
        except Exception as e:                                    # noqa: BLE001
            QMessageBox.warning(self, "Pre-flight", f"Settings problem: {e}")
            return
        ls = self.live_settings()
        dlg = PreflightDialog(self, not ls.paper, ls.armed, then_start, values["equity_frac"])
        self._pf_dlg, self._pf_running = dlg, True
        self.btn_start.setEnabled(False); self.btn_pf.setEnabled(False)
        self.add_log("=== pre-flight check started ===")
        w = FnWorker(lambda log: PF.run_preflight(values, ls, log))
        self._workers.start(w, lambda res, err: self._pf_done(dlg, res, err, ls), dlg.add_log)
        if then_start:
            dlg.exec()
            self._after_preflight_dialog(dlg)
        else:
            dlg.show()
            if auto:
                QApplication.alert(self.window())

    def _pf_done(self, dlg: PreflightDialog, res, err, ls: P.LiveSettings) -> None:
        self._pf_running = False
        if err or res is None:
            rep = P.Report(created=time.time(), live=not ls.paper, armed=ls.armed, capital=ls.capital)
            rep.checks.append(P.Check("System", "Pre-flight", PF.FAIL, f"the check itself failed: {err}", "system"))
        else:
            rep = res
        self._pf_report = rep
        saved = PF.save_report(rep)
        col = {"GO": GREEN, "CAUTION": YELLOW, "NO-GO": RED}[rep.verdict]
        self.pf_lbl.setText(f"Pre-flight {datetime.fromtimestamp(rep.created):%H:%M}: {rep.verdict}" + (f" ({rep.counts()['WARN']} warning(s))" if rep.verdict == "CAUTION" else ""))
        self.pf_lbl.setStyleSheet(f"color: {col}; font-weight: 600;")
        self.add_log(f"=== pre-flight: {rep.headline()}", "ok" if rep.verdict == "GO" else "warn" if rep.verdict == "CAUTION" else "err")
        for c in rep.checks:
            if c.status in (PF.FAIL, PF.WARN):
                self.add_log(f"   [{c.status}] {c.name}: {c.detail}", "err" if c.status == PF.FAIL else "warn")
        self.btn_start.setEnabled(not self.running); self.btn_pf.setEnabled(not self.running)
        dlg.set_report(rep, saved)
        self._hint()

    def _after_preflight_dialog(self, dlg: PreflightDialog) -> None:
        act, rep = dlg.action, dlg.report
        if act == "start_frac" and rep is not None and rep.suggested_frac:
            self.sp.w["equity_frac"].setValue(rep.suggested_frac)
            self.add_log(f"equity per trade set to {rep.suggested_frac:g}% by the pre-flight check")
        elif act == "dry":
            self.mode.setCurrentIndex(0)
        if act in ("start", "start_frac", "dry"):
            self._start_session(rep)
        else:
            self.add_log("start cancelled" if rep is not None else "start cancelled before the pre-flight finished")

    def _clock_tick(self) -> None:
        now = D.now_et()
        nb = D.next_bar_close(now)
        if nb is not None:
            self.kpis["Next bar closes in"].set(D.fmt_td((nb - now).total_seconds()), sub=f"{nb:%H:%M} ET bar; entries use the last CLOSED bar")
            self.kpis["Market"].set("OPEN", GREEN, sub=f"{now:%a %H:%M:%S} ET")
        else:
            nx = D.next_open(now)
            self.kpis["Next bar closes in"].set("—", sub="market closed")
            self.kpis["Market"].set("CLOSED", MUTED, sub=f"opens ~{nx:%a %H:%M} ET")
        if self.auto_pf.isChecked() and not self._pf_running and not self.running and now.weekday() < 5 and now.hour == 9 and now.minute >= 10 and self._auto_key != f"{now:%Y-%m-%d}":
            self._auto_key = f"{now:%Y-%m-%d}"
            self.add_log("auto pre-flight (09:10 ET)")
            self.run_preflight(then_start=False, auto=True)
        if self._stop_deadline and self.running and time.time() > self._stop_deadline:      # the engine thread is stuck in a slow request: stop waiting for it
            self.add_log("the engine did not confirm the stop in time - treating the session as stopped", "warn")
            self._finish_session("stopped by user (engine slow to confirm)")
        self.btn_unlatch.setVisible(os.path.exists(GU.KILL_FILE))

    def _hint(self) -> None:
        ls = self.live_settings()
        if self.running:
            t = "Running (polls every minute). Stop ends the session only: positions stay open."
        elif os.path.exists(GU.KILL_FILE):
            t = "A KILL latch is set - press 'Clear kill latch' before you start again."
        elif self._acct is None:
            t = "Step 1 → press 'Connect and test' to check your keys and the account (read-only)."
        elif self._pf_report is None:
            t = "Step 2 → run the pre-flight check, or just press Start (it runs one first)."
        else:
            t = "Step 3 → press Start. " + ("Dry run: orders are only shown." if not ls.armed else "ARMED: orders will be sent after you confirm.")
        self.hint.setText(t)

    # ══ session control ══════════════════════════════════════════════════════════
    def _lock(self, running: bool) -> None:
        self.btn_start.setEnabled(not running and not self._pf_running)
        self.btn_pf.setEnabled(not running and not self._pf_running)
        self.btn_stop.setEnabled(running)
        for w in (self.account, self.btn_connect, self.capital, self.btn_cap, self.sdate, self.mode, self.short_route, self.dloss, self.maxtr, self.btn_hist):
            w.setEnabled(not running)
        self.sp.set_enabled_all(not running)
        if not running and self.btn_setup.isChecked():
            self.btn_setup.blockSignals(True); self.btn_setup.setChecked(False); self.btn_setup.setText("Show setup"); self.btn_setup.blockSignals(False)
        self.setup.setVisible(not running or self.btn_setup.isChecked())          # while running: a summary card instead of the (locked) setup boxes
        self.run_card.setVisible(running)
        for w in (self.btn_pf, self.chk_pf, self.auto_pf, self.pf_lbl):
            w.setVisible(not running)

    def push_config(self, cfg, ls: P.LiveSettings) -> None:
        """The Engine reads these module-level settings of supertrend_gui at run time (that is how the app has always handed it the tuned parameters)."""
        GU.CFG = cfg
        GU.SHORT_EQUITY_MULT = cfg.short_mult
        GU.MAX_TRADES_PER_DAY = int(ls.max_trades)
        GU.DAILY_LOSS_LIMIT_PCT = -abs(float(ls.daily_loss_pct))
        GU.PREFER_SHORT_SOXL = bool(ls.prefer_short)

    def _start_session(self, report) -> None:
        try:
            values = self.sp.values()
            cfg = P.to_config(values)
            ls = self.live_settings()
            datetime.strptime(ls.start_date, "%Y-%m-%d")
        except Exception as e:                                    # noqa: BLE001
            QMessageBox.warning(self, "Start", f"Settings problem: {e}")
            return
        if os.path.exists(GU.KILL_FILE):
            QMessageBox.warning(self, "Start", "A KILL latch is set. Press 'Clear kill latch' first.")
            return
        if ls.armed:
            summary = (f"Strategy: {cfg.strategyMode}, ATR {cfg.atrPeriod}, factor {cfg.factor:g}\nCapital ${ls.capital:,.0f} from {ls.start_date}, {values['equity_frac']:g}% of running equity per trade "
                       f"(shorts x{cfg.short_mult:g})\nShort route: {'direct SOXL short if possible, else SOXS' if ls.prefer_short else 'always SOXS'}\n"
                       f"Loss limits: -{ls.daily_loss_pct:g}% a day, {ls.max_trades} trades a day\nPre-flight: {report.verdict if report else 'not run'}\n")
            if ls.paper:
                if QMessageBox.question(self, "Arm the paper session", f"Send orders to your Alpaca PAPER account?\n\n{summary}") != QMessageBox.StandardButton.Yes:
                    self.add_log("start cancelled"); return
            else:
                ans, ok = QInputDialog.getText(self, "Confirm LIVE trading", f"You are about to send REAL-MONEY orders to your Alpaca LIVE account.\n\n{summary}\n{P.VALID_NOTE}\n\nType  {GU.LIVE_CONFIRM}  to confirm.")
                if not ok or ans.strip() != GU.LIVE_CONFIRM:
                    self.add_log("start cancelled"); return
        self.push_config(cfg, ls)
        self._run = dict(values=dict(values), ls=ls, cfg=cfg, paper=ls.paper, armed=ls.armed)
        self.q = queue.Queue()
        self._dry_seen.clear()
        self.orders.setRowCount(0)                                 # the Orders tab lists THIS session's orders (the Log keeps the history)
        self.engine = GU.Engine(self.q, ls.paper, ls.capital, cfg.equity_frac, ls.start_date, ls.armed)
        self._session_capital, self._session_start = ls.capital, datetime.strptime(ls.start_date, "%Y-%m-%d")
        self._live_rows, self._sess, self._sig, self._held = [], None, None, {}
        self.perf.show_open(None, "Account holds", "waiting for the first poll …")
        self._stop_deadline = None
        self.running = True
        self.add_log(f"=== session started: {self.mode_label(ls)} · capital ${ls.capital:,.0f} · {cfg.strategyMode} ATR {cfg.atrPeriod} factor {cfg.factor:g} ===", "ok")
        self._lock(True)
        self.state.setText(f"Running ({'ARMED' if ls.armed else 'DRY RUN'}, {'paper' if ls.paper else 'LIVE'}) — first poll in a moment")
        self.state.setStyleSheet("")
        self._refresh_using(); self._hint(); self._update_perf()
        self.engine.start()

    def _held_text(self) -> str:
        return ", ".join(f"{s} {p['qty']:+,.0f}" for s, p in self._held.items() if abs(p["qty"]) > 0)

    def stop(self) -> None:
        if not self.running:
            return
        if self._held_text():
            soxs = GU.SYMBOL_SHORT in self._held
            if QMessageBox.question(self, "Stop", f"You hold {self._held_text()}.\n\nStopping ends the engine only: the position stays open. " +
                                    ("SOXS exits are managed by THIS app only (nothing rests at the broker), so the position will be UNMANAGED until you start again or flatten it.\n\n" if soxs else
                                     "A SOXL take-profit / stop that rests at the broker stays active.\n\n") + "Stop anyway?") != QMessageBox.StandardButton.Yes:
                return
        self._request_stop("stopped by user")

    def _request_stop(self, why: str) -> None:
        self.state.setText("Stopping …")
        self.btn_stop.setEnabled(False)
        self._stop_why = why
        self._stop_deadline = time.time() + 25
        if self.engine is not None:
            self.engine.stop()

    def _on_engine_stopped(self) -> None:
        if self.running:
            self._finish_session(getattr(self, "_stop_why", None) or "the engine stopped")

    def _finish_session(self, why: str) -> None:
        if not self.running:
            return
        self.running = False
        self._stop_deadline = None
        try:
            PF.record_session_end(why, "live" if (self._run and not self._run["paper"]) else "paper", (self._sess or {}).get("realised", 0.0), len(self._live_rows))
        except Exception:                                         # noqa: BLE001
            pass
        self._lock(False)
        self.state.setText(f"Session ended: {why}")
        self.state.setStyleSheet(f"color: {RED};" if "KILL" in why.upper() else "")
        self.add_log(f"=== session ended: {why} ===", "warn")
        self._refresh_using(); self._hint()
        QTimer.singleShot(500, self.reload_history)

    def _stand_down(self, kill: bool) -> None:
        paper = self._run["paper"] if (self.running and self._run) else self.account.currentIndex() == 0
        eng = self.engine if self.running else None
        self._stop_why = "KILLED" if kill else "flattened by user"                  # the engine's own 'stopped' event can arrive before this finishes
        self.btn_flat.setEnabled(False); self.btn_kill.setEnabled(False)
        self.state.setText("KILL: stopping the engine, cancelling orders, closing positions …" if kill else "Flattening …")

        def work(log):
            if eng is not None and eng.is_alive():
                log("stopping the engine …")
                eng.stop(); eng.join(25)
            k, s = GU.creds(paper)
            tc = GU.TradingClient(k, s, paper=paper)
            try:
                tc.cancel_orders()
            except Exception as exc:                              # noqa: BLE001
                log(f"cancel orders: {exc}")
            res = tc.close_all_positions(cancel_orders=True)
            try:
                with open(GU.STATE_FILE, "w", encoding="utf-8") as fh:
                    json.dump({}, fh)
            except Exception:                                     # noqa: BLE001
                pass
            if kill:
                os.makedirs(os.path.dirname(GU.KILL_FILE), exist_ok=True)
                with open(GU.KILL_FILE, "w", encoding="utf-8") as fh:
                    fh.write(f"killed from the GUI {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            return dict(closed=len(res) if res else 0)

        def done(res, err):
            self.btn_flat.setEnabled(True); self.btn_kill.setEnabled(True)
            if err:
                self.add_log(f"{'KILL' if kill else 'flatten'} failed: {err}", "err")
                self.state.setText(f"⚠ {'KILL' if kill else 'flatten'} failed - check the account by hand: {err}")
                self.state.setStyleSheet(f"color: {RED};")
            else:
                self.add_log(("KILL" if kill else "FLATTEN") + f" done: orders cancelled, {res['closed']} position(s) closed" + (", KILL latch written" if kill else ""), "trade")
            if self.running:
                self._finish_session("KILLED" if kill else "flattened by user")
            elif kill and not err:
                try:
                    PF.record_session_end("KILLED", "live" if not paper else "paper")
                except Exception:                                 # noqa: BLE001
                    pass
            self._hint()
            QTimer.singleShot(800, self.reload_history)

        self._workers.start(FnWorker(work), done, lambda m: self.add_log(m))

    def flatten(self) -> None:
        paper = self._run["paper"] if (self.running and self._run) else self.account.currentIndex() == 0
        if QMessageBox.question(self, "Flatten", f"Close ALL positions and cancel ALL open orders on the Alpaca {'PAPER' if paper else 'LIVE (real money)'} account now?\n\n" +
                                ("The running session is stopped first (otherwise the engine would re-enter on its next poll).\n\n" if self.running else "") + "Continue?") == QMessageBox.StandardButton.Yes:
            self._stand_down(kill=False)

    def kill(self) -> None:
        paper = self._run["paper"] if (self.running and self._run) else self.account.currentIndex() == 0
        if QMessageBox.question(self, "KILL", f"KILL stops the engine, cancels every open order, closes every position at market on the Alpaca {'PAPER' if paper else 'LIVE (real money)'} account, "
                                              "and writes the KILL latch so nothing can restart until you clear it.\n\nContinue?") == QMessageBox.StandardButton.Yes:
            self._stand_down(kill=True)

    def clear_latch(self) -> None:
        try:
            os.remove(GU.KILL_FILE)
            self.add_log("KILL latch cleared", "ok")
        except Exception as exc:                                  # noqa: BLE001
            self.add_log(f"could not remove the KILL file: {exc}", "err")
        self.btn_unlatch.setVisible(os.path.exists(GU.KILL_FILE))
        self._hint()

    # ══ engine events ════════════════════════════════════════════════════════════
    def _drain(self) -> None:
        try:
            for _ in range(400):
                self.on_event(self.q.get_nowait())
        except queue.Empty:
            pass

    def on_event(self, d: dict) -> None:
        k = d["kind"]
        if k == "log":
            self.add_log(d["msg"], d.get("level", "info"), d.get("ts"))
        elif k == "account":
            self._on_account(d)
        elif k == "session":
            self._on_session(d)
        elif k == "trades":
            self._live_rows = d["trades"]
            self._update_perf()
        elif k == "positions":
            self._on_positions(d)
        elif k == "signal":
            self._on_signal(d)
        elif k == "plan":
            self._on_plan(d)
        elif k == "guard":
            self._on_guard(d)
        elif k == "levels":
            self._on_levels(d)
        elif k == "openpos":
            self._on_openpos(d)
        elif k == "trade_popup":
            self._on_order(d)
        elif k == "stopped":
            self._on_engine_stopped()

    def _on_account(self, d: dict) -> None:
        self._equity_now = d["equity"]
        if self._day_base is None:
            self._day_base = d["equity"]
        if "buying_power" in d:
            self.bp.setText(f"${d['equity']:,.2f}   /   buying power ${d['buying_power']:,.2f}")
        base = self._day_base or d["equity"]
        pnl = d["equity"] - base
        self.kpis["Day P&L"].set(f"{'-' if pnl < 0 else '+'}${abs(pnl):,.2f}", GREEN if pnl > 0 else RED if pnl < 0 else None, sub=f"{pnl / base * 100:+.2f}% · equity ${d['equity']:,.0f}" if base else "")

    def _on_session(self, d: dict) -> None:
        self._sess = d
        self.hist_lbl.setText(f"Since {d['start']}: baseline ${d['baseline']:,.0f} · {d['n_trades']} closed trades [{d.get('source', '')}] · realised ${d['realised']:+,.2f} · open ${d['unrealised']:+,.2f} · "
                              f"running equity ${d['equity']:,.2f} → next long ${d['equity'] * d['frac']:,.0f} / short ${d['equity'] * d['frac'] * GU.SHORT_EQUITY_MULT:,.0f}")
        self._update_perf()

    def _on_positions(self, d: dict) -> None:
        self._held = {s: p for s, p in d["positions"].items() if abs(p["qty"]) > 1e-9}
        if not self._held:
            self.kpis["Position"].set("FLAT", COL_FLAT)
            return
        sym, p = next(iter(self._held.items()))
        ss = strategy_side(sym, p["side"])
        note = f" (long {GU.SYMBOL_SHORT} = short {GU.SYMBOL_LONG})" if sym == GU.SYMBOL_SHORT else ""
        self.kpis["Position"].set(f"{ss.upper()} {abs(p['qty']):,.0f} {sym}", side_color(ss), sub=f"entry {p['avg']:.2f} · last {p['cur']:.2f} · P&L ${p['upl']:+,.0f} ({p['uplpc']:+.2f}%){note}")

    def _on_signal(self, d: dict) -> None:
        self._sig = d
        up = d["want"] > 0
        self.kpis["Trend (last closed bar)"].set("UP · long" if up else "DOWN · short", side_color("long" if up else "short"), sub=f"Supertrend {d['st']:.2f} · re-entry {d['limit']:.2f}")
        self.sig_lbl.setText(f"Signal: bar {d['bar']} closed {d['close']:.2f} · Supertrend {d['st']:.2f} · trend {'UP' if up else 'DOWN'} · re-entry limit {d['limit']:.2f} · "
                             f"monitor bar {d['mon_low']:.2f}–{d['mon_high']:.2f}{' (still forming)' if d.get('forming') else ''}")

    def _on_plan(self, d: dict) -> None:
        ss = strategy_side(d["symbol"], d["side"])
        if d["in_place"]:
            self.plan_lbl.setText(f"Next order: none — the account already holds the right position ({d['symbol']}).")
            self.plan_lbl.setStyleSheet(f"color: {MUTED};")
        else:
            self.plan_lbl.setText(f"Next order: {ss.upper()} {d['qty']:,} {d['symbol']} @ ~{d['price']:.2f} = ${d['notional']:,.0f} · budget ${d['budget']:,.0f} ({d['frac']:.0%} × {d.get('mult', 1.0):.0%}) · route: {d['route']} · {d['when']}")
            self.plan_lbl.setStyleSheet(f"color: {side_color(ss)}; font-weight: 600;")

    def _on_guard(self, d: dict) -> None:
        self.kpis["Trades today"].set(f"{d['trades_today']} / {d['cap']}", YELLOW if d["trades_today"] >= d["cap"] else None)
        if not d["ok"]:
            self.state.setText(f"Entries paused: {d['reason']}")
            self.state.setStyleSheet(f"color: {YELLOW};")
        elif self.running:
            self.state.setText(f"Running ({'ARMED' if self._run and self._run['armed'] else 'DRY RUN'}, {'paper' if self._run and self._run['paper'] else 'LIVE'})")
            self.state.setStyleSheet("")

    def _on_openpos(self, d: dict) -> None:
        """The account's open position next to the one the strict backtest holds on the same bars, and whether they match."""
        self._openpos = d
        pend = d.get("pending")
        nxt = "long" if d["want"] > 0 else "short"
        flat = ("FLAT" + (f" · resting {pend['side']} limit {pend['qty']:,} {pend['symbol']} @ {pend['limit']:.2f}" if pend else
                          f" · waiting for a {nxt} re-entry at {d['limit']:.2f} or better (trend {'UP' if d['want'] > 0 else 'DOWN'})"))
        cmp_ = d["cmp"]
        if self._run is not None and not self._run["armed"] and cmp_["status"] != "match":
            cmp_ = {**cmp_, "status": "dry", "text": "DRY RUN: no orders are sent, so the account stays flat and cannot match. " + cmp_["text"]}
        self.perf.show_open(d["live"], "Account holds", flat, d["expected"], "Backtest holds", cmp_)

    def _on_levels(self, d: dict) -> None:
        trail = f"armed at {d['trail']:.2f} (extreme {d['extreme']:.2f})" if d.get("armed") and d.get("trail") else "not armed yet (price has not reached the run-up target)"
        self.lvl_lbl.setText(f"Position levels (SOXL prices): {d['side'].upper()} · entry ref {d['entry_ref']:.2f} · run-up target {d['target']:.2f} · take-profit {d['tp']:.2f} · trailing stop {trail}")
        self.lvl_lbl.setStyleSheet(f"color: {side_color(d['side'])};")

    def _on_order(self, d: dict) -> None:
        p, dry = d["payload"], d["dry"]
        key = (p.get("ticker"), p.get("action"), p.get("quantity"), p.get("type"), p.get("limit_price"), d["why"])
        if dry:
            if key in self._dry_seen:                                            # the engine re-proposes the same order every poll: show it once
                return
            self._dry_seen.add(key)
        r = self.orders.rowCount()
        self.orders.insertRow(r)
        res = "DRY RUN — not sent" if dry else f"SENT {p.get('status', '')}".strip()
        set_row(self.orders, r, [datetime.now().strftime("%H:%M:%S"), d["mode"], p.get("ticker", ""), str(p.get("action", "")).upper(), f"{int(p.get('quantity', 0)):,}", p.get("type", ""),
                                 f"{p['limit_price']:.2f}" if "limit_price" in p else "—", res, d["why"]], {7: MUTED if dry else GREEN}, align_left=(1, 2, 3, 5, 7, 8))
        self.orders.scrollToBottom()
        self.add_log(f"{'DRY RUN would send' if dry else 'ORDER SENT'} {json.dumps(p)}  — {d['why']}", "warn" if dry else "trade")
        if not dry and self.chk_pop.isChecked():
            self._popup(f"ORDER SENT — {d['mode']}", f"{d['why']}\n\n{json.dumps(p, indent=2)}", real=(d["mode"] != "PAPER"))

    def _popup(self, title: str, text: str, real: bool = False) -> None:
        """A non-blocking window (and a beep for real money) so an order cannot go unnoticed."""
        if real:
            QApplication.beep()
        box = QMessageBox(QMessageBox.Icon.Information if not real else QMessageBox.Icon.Warning, title, text, QMessageBox.StandardButton.Ok, self)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._popups.append(box)
        box.show()

    # ══ performance ══════════════════════════════════════════════════════════════
    def _update_perf(self) -> None:
        cap = self._session_capital or self.capital.value()
        if self._sess is None and not self._live_rows:
            if not self.running:
                self.perf.show_result(None, None, pd.DataFrame(), dict(trades=0), cap, "Connect to load the session's history, or start a session\n(the same metrics as the Backtest tab appear here)")
            else:
                self.perf.show_result(None, None, pd.DataFrame(), dict(trades=0), cap, "Waiting for the first poll …")
            return
        tr = M.trades_from_live(self._live_rows, cap)
        start = self._session_start or datetime.combine(datetime.now().date(), datetime.min.time())
        now = max(D.now_et(), start + timedelta(minutes=1))
        eq_now = float(self._sess["equity"]) if self._sess else float(cap)
        eq = M.curve_from_trades(tr, cap, start, now, eq_now)
        m = M.compute_metrics(eq, tr, cap, fill_days=True)
        m.pop("sharpe_bar", None)                                           # a per-bar figure means nothing on a step curve of closed trades
        src = (self._sess or {}).get("source", "")
        self.perf.show_result(eq, None, tr, m, cap, "No closed trades since the start date yet", step=True, always_curve=True)
        if len(tr) == 0:
            self.perf.kpis["Final equity"].set(big_money(eq_now), sub=f"from {big_money(cap)} · {src}")

    # ══ persistence ══════════════════════════════════════════════════════════════
    def save(self) -> None:
        q = self.qs
        q.setValue("live/preflight_on_start", self.chk_pf.isChecked()); q.setValue("live/auto_preflight", self.auto_pf.isChecked())
        q.setValue("live/prefer_short", self.short_route.currentIndex() == 0); q.setValue("live/dloss", self.dloss.value()); q.setValue("live/maxtr", int(self.maxtr.value()))
        q.setValue("live/popups", self.chk_pop.isChecked()); q.setValue("live/capital_pinned", self._capital_pinned)
        if self._capital_pinned:
            q.setValue("live/capital", self.capital.value())
        # NOT saved on purpose: account (Paper/Live) and Orders (dry/armed) - every launch starts on Paper + DRY RUN

    def load(self) -> None:
        q = self.qs
        tb = lambda k, d: str(q.value(k, d)).lower() in ("true", "1")      # noqa: E731
        self.short_route.setCurrentIndex(0 if tb("live/prefer_short", True) else 1)
        try:
            self.dloss.setValue(float(q.value("live/dloss", 8.0))); self.maxtr.setValue(int(float(q.value("live/maxtr", 12))))
        except (TypeError, ValueError):
            pass
        self.chk_pop.setChecked(tb("live/popups", True))
        if tb("live/capital_pinned", False):
            try:
                self._cap_programmatic = True
                self.capital.setValue(float(q.value("live/capital", 100_000)))
                self._capital_pinned = True
            except (TypeError, ValueError):
                pass
            finally:
                self._cap_programmatic = False
        self._sync_capital()

