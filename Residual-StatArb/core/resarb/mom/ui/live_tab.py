"""Live trading tab for the book picked on the left - the exact model object the backtest ran: connect, pre-flight, plan the overnight decision now (dry run), run the
intraday check now (dry run), the daily loop (morning OPG sells, the 30-minute intraday checks on IEX, the 15:46 overnight decision on the delayed SIP), a shadow model,
the check against the backtest's data, KILL.

Nothing is sent unless 'ARM' is ticked AND the typed phrase is given AND the engine has the project's own keys on an account no other strategy claimed. The window always starts
on PAPER + DRY RUN. Real money is disabled in code (Valid has not signed off)."""
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
from .. import presets as PR
from .model_panel import describe

LOG_COLORS = {"err": RED, "warn": YELLOW, "ok": GREEN, "trade": ACCENT, "info": TEXT}
NO_SHADOW = "(no shadow)"


class _Bridge(QObject):
    log = pyqtSignal(str, str)
    plan = pyqtSignal(object)
    check = pyqtSignal(object)
    stopped = pyqtSignal(str)


class LiveTab(QWidget):
    def __init__(self, panel, backtest, qs: QSettings, folder: str):
        super().__init__()
        self.panel, self.bt, self.qs, self.folder = panel, backtest, qs, folder
        self.eng: Optional[ML.MomLiveEngine] = None
        self.plan: Optional[ML.MomPlan] = None
        self._workers = WorkerBag()
        self._busy = False
        self._book_busy = False
        self._loop: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._checks: list = []
        self.br = _Bridge()
        self.br.log.connect(self.add_log)
        self.br.plan.connect(self._planned)
        self.br.check.connect(self._checked)
        self.br.stopped.connect(self._loop_ended)
        root = QVBoxLayout(self)
        banner = QLabel("Valid has NOT signed this book off for capital. Run it DRY, then on paper. Keys come from 'Alpaca Kay.txt' in the project folder "
                        "(MB_ALPACA_API_KEY / MB_ALPACA_SECRET_KEY would select a separate account). The intraday leg day-trades: a margin account with at least $25,000. "
                        "A live (real-money) account can be viewed; sending real orders is disabled in code unless RESARB_ALLOW_LIVE=1.")
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
        self.k = {n: Kpi(n) for n in ("Today", "Capital", "Overnight book", "Orders", "Intraday", "Loop")}
        for w in self.k.values():
            k.addWidget(w)
        root.addLayout(k)
        self.tabs = QTabWidget()
        self.t_orders = make_table(["Ticker", "Side", "Shares", "Order", "Ref price", "Notional", "Why", "OK", "Note"])
        self.tabs.addTab(self.t_orders, "Overnight orders (planned)")
        self.t_cand = make_table(["Rank", "Ticker", "Score", "Move to decision", "ML forecast", "Price", "Bought"])
        self.tabs.addTab(self.t_cand, "Candidates (ranked)")
        self.t_checks = make_table(["Bar end", "Price", "Band", "VWAP", "Held", "Action", "Shares", "Status", "Feed"])
        self.tabs.addTab(self.t_checks, "Intraday checks")
        self.t_pos = make_table(["Leg", "Ticker", "Shares", "Ordered", "Entry", "Entry price", "Backtest entry", "Price now", "Value", "P&L $", "Status"])
        self.tabs.addTab(self.t_pos, "Positions (broker)")
        sw = QWidget(); sv = QVBoxLayout(sw)
        self.shadow_lbl = muted("No shadow model.")
        self.t_shadow = make_table(["Ticker", "Shares", "Score", "Rank", "Move", "Also traded"])
        self.t_shadow_rep = make_table(["Day", "Traded names", "Traded night return", "Shadow names", "Shadow night return", "In common"])
        sv.addWidget(self.shadow_lbl); sv.addWidget(self.t_shadow, 1); sv.addWidget(muted("Every journaled night (close -> next official open, before costs, from the cached daily bars):"))
        sv.addWidget(self.t_shadow_rep, 1)
        self.tabs.addTab(sw, "Shadow")
        self.t_pf = make_table(["Check", "Result", "Detail"])
        self.tabs.addTab(self.t_pf, "Pre-flight")
        self.t_trades = make_table(["Leg", "Ticker", "Shares", "Entry", "Entry price", "Backtest entry", "Exit", "Exit price", "Backtest exit", "Exit via", "P&L $",
                                    "Backtest P&L $", "Diff $"])
        self.tabs.addTab(self.t_trades, "Trade history (broker)")
        self.t_hist = make_table(["Submitted", "Ticker", "Side", "Shares", "Order", "Leg", "Status", "Filled", "Fill price", "Filled at"])
        self.tabs.addTab(self.t_hist, "Order history (broker)")
        self.t_par = make_table(["Day", "Leg", "What", "Ticker", "Live", "Backtest", "Match"])
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
        self.account = QComboBox(); self.account.addItems(["Paper (simulated money)", "Live (real money)"])
        self.account.setToolTip("Paper = the paper account of the project's keys. Live = the real-money account: it can be connected and viewed (positions, orders, trade "
                                "history) at any time; sending orders needs RESARB_ALLOW_LIVE=1 in the environment and the typed phrase "
                                f"'{ML.LIVE_PHRASE}' (real money is disabled in code until Valid signs the book off).")
        self.account.currentIndexChanged.connect(self._account_changed)
        self.btn_connect = QPushButton("Connect"); self.btn_connect.clicked.connect(self.connect_)
        self.acct_lbl = muted("not connected")
        f.addWidget(self.account, 0, 0, 1, 2)
        f.addWidget(self.btn_connect, 1, 0, 1, 2); f.addWidget(self.acct_lbl, 2, 0, 1, 2)
        return g

    @property
    def paper(self) -> bool:
        return self.account.currentIndex() == 0

    def _env_text(self) -> None:
        on = self.armed.isChecked()
        if self.paper:
            self.env.setText("PAPER account - ARMED: orders are sent at the backtest's times" if on else "PAPER account - DRY RUN unless ARMED")
            self.env.setStyleSheet(box_style(YELLOW if on else GREEN, 0.14, True, 8))
        else:
            allowed = os.environ.get("RESARB_ALLOW_LIVE") == "1"
            self.env.setText("LIVE (REAL MONEY) account - ARMED: real orders are sent" if on else
                             ("LIVE (REAL MONEY) account - DRY RUN unless ARMED" if allowed else "LIVE (REAL MONEY) account - VIEW ONLY: real money is disabled in code"))
            self.env.setStyleSheet(box_style(RED if on or allowed else YELLOW, 0.18, True, 8))

    def _account_changed(self, _i: int) -> None:
        if self._loop is not None and self._loop.is_alive():
            self.account.blockSignals(True); self.account.setCurrentIndex(0 if self.eng and self.eng.live.paper else 1); self.account.blockSignals(False)
            QMessageBox.information(self, "Account", "Stop the loop first, then switch the account.")
            return
        self._set_armed(False)
        self.eng = None                                                                         # another account: another broker, journal and state files
        for t in (self.t_pos, self.t_trades, self.t_hist, self.t_par):
            t.setRowCount(0)
        self.acct_lbl.setText("not connected")
        self._env_text()
        self.add_log(f"account: {'PAPER' if self.paper else 'LIVE (real money)'} - connect to load its positions and history", "warn" if not self.paper else "info")

    def _mode_box(self) -> QGroupBox:
        g = QGroupBox("Mode and limits"); f = QFormLayout(g)
        self.armed = QCheckBox("ARM - allow sending orders")
        self.armed.setToolTip("Unticked = DRY RUN: every decision is planned and journaled, nothing is sent.")
        self.feed = QComboBox(); self.feed.addItems(list(ML.FEEDS))
        self.feed.setToolTip("Overnight decision bars: sip_delayed = the free plan's SIP bars, 16 minutes late (decision 15:30 - exactly the backtest's data); sip = real-time SIP "
                             "(paid plan, allows 15:45). The intraday leg reads its own feed (the model's 'Intraday feed', IEX by default).")
        self.auction = QComboBox()
        d = ML.MomLiveSettings()
        for txt, val in (("MOC / MOO orders, as the model", "auction"),
                         (f"market DAY orders: {d.mkt_close_s} s before the close, {-d.mkt_open_s // 60} min before the open", "market")):
            self.auction.addItem(txt, val)
        self.auction.setToolTip("Only the model's auction orders: the overnight buys at the close and sales at the open, and the QQQ sale at the close when it is still held "
                                "after the last check. The model, its decisions and sizes are the same either way; the intraday checks' market orders are not affected.\n"
                                "MOC / MOO: market-on-close at the 15:46 decision, market-on-open at 08:45 - as the backtest.\n"
                                f"market DAY orders: the 15:46 decision is queued and sent at 15:59:{60 - d.mkt_close_s:02d} (the QQQ sale and the exits first, then the "
                                f"buys, in parallel; nothing in the last 2 s); last night's names are sold at 09:{30 + d.mkt_open_s // 60:02d}:00, queued for the open. "
                                "Their fills are compared with the backtest's auction prices (Positions + trade history: diff; Check live vs backtest: cost in bp).\n"
                                "Alpaca's PAPER account does not run the auctions: it fills MOC / MOO orders at the quote in the last seconds, partly at random, and "
                                "expires the rest (2026-10-28: 6 of 20 filled).\n"
                                "Greyed out when the model sends no auction order.")
        self.auction.currentIndexChanged.connect(lambda *_: self._model_strip())
        self.shadow = QComboBox(); self.shadow.addItems([NO_SHADOW] + PR.preset_names())
        self.shadow.setToolTip("A second book whose overnight picks are computed on the same data at every decision and journaled - never traded.")
        self.cap = spin(0, 1e9, 0, 0, 5_000); self.cap.setPrefix("$ "); self.cap.setToolTip("0 (default) = the account equity x fraction. The equity is the cash plus the book's own positions: positions outside the book are not "
                                                                                          "part of it, buying power / margin is never used, and a fixed amount above the equity is capped at it.")
        self.eq_frac = spin(0.05, 1.0, 1.0, 2, 0.05); self.eq_frac.setToolTip("share of the account equity the book trades (1.00 = all of it)")
        self.halt = spin(0.0, 0.5, 0.0, 3, 0.005); self.halt.setToolTip("0 = off. Live-only overlay (the backtest has none): no new entries after this loss today")
        self.dd = spin(0.0, 0.9, 0.0, 2, 0.01); self.dd.setToolTip("0 = off. Live-only overlay: sell everything and stay flat after this drawdown, until reset")
        f.addRow(self.armed); f.addRow("Overnight bars from", self.feed); f.addRow("Auction orders", self.auction); f.addRow("Shadow model", self.shadow)
        f.addRow("Fixed capital", self.cap)
        f.addRow("…or equity fraction", self.eq_frac); f.addRow("Daily-loss halt", self.halt); f.addRow("Drawdown stop", self.dd)
        return g

    def _control_box(self) -> QGroupBox:
        g = QGroupBox("Run"); v = QVBoxLayout(g)
        self.btn_pf = QPushButton("Pre-flight check"); self.btn_pf.clicked.connect(self.preflight)
        self.btn_plan = QPushButton("Plan overnight now (dry run)"); self.btn_plan.setObjectName("primary"); self.btn_plan.clicked.connect(self.plan_now)
        self.btn_check = QPushButton("Intraday check now (dry run)"); self.btn_check.clicked.connect(self.check_now)
        self.btn_start = QPushButton("Start the daily loop"); self.btn_start.setObjectName("go"); self.btn_start.clicked.connect(self.start_loop)
        self.btn_stop = QPushButton("Stop loop"); self.btn_stop.clicked.connect(self.stop_loop); self.btn_stop.setEnabled(False)
        self.btn_cmp = QPushButton("Check live vs backtest"); self.btn_cmp.clicked.connect(self.compare)
        self.btn_cmp.setToolTip("Rebuilds every journaled overnight decision and intraday check from the backtest's data (update the data first) with the positions and capital "
                                "live had, and compares.")
        self.btn_book = QPushButton("Positions + trade history"); self.btn_book.clicked.connect(self.load_book)
        self.btn_book.setToolTip("Reload the open positions and every trade of this book from the broker's order history (read-only), each next to the backtest's price "
                                 "and P&L for it. Also loads on Connect, after each plan and every 10 minutes.")
        self.btn_shrep = QPushButton("Shadow report"); self.btn_shrep.clicked.connect(self.shadow_report)
        self.btn_hwm = QPushButton("Reset drawdown base / clear KILL"); self.btn_hwm.clicked.connect(self.reset_hwm)
        self.btn_hwm.setToolTip("Make the current equity the new high-water mark, clear a drawdown stop and clear a KILL (after a KILL nothing is sent until this is done)")
        self.btn_kill = QPushButton("KILL: cancel + flatten"); self.btn_kill.setObjectName("danger"); self.btn_kill.clicked.connect(self.kill)
        self.btn_kill.setToolTip("cancel this book's open orders and close every position in its symbols")
        for b in (self.btn_pf, self.btn_plan, self.btn_check, self.btn_start, self.btn_stop, self.btn_book, self.btn_cmp, self.btn_shrep, self.btn_hwm, self.btn_kill):
            v.addWidget(b)
        return g

    # ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def live_settings(self) -> ML.MomLiveSettings:
        return ML.MomLiveSettings(paper=self.paper, armed=self.armed.isChecked(), capital=self.cap.value(), equity_fraction=self.eq_frac.value(),
                                  feed=self.feed.currentText(), auction_orders=self.auction.currentData() or "auction", daily_loss_halt=self.halt.value(),
                                  dd_stop=self.dd.value())

    def shadow_model(self):
        s = self.shadow.currentText()
        return PR.get(s) if s in PR.PRESETS else None

    def engine(self) -> ML.MomLiveEngine:
        m = self.panel.model()
        ls = self.live_settings()
        if self.eng is None or (self.eng.live.paper != ls.paper and (self._loop is None or not self._loop.is_alive())):
            self.eng = ML.MomLiveEngine(m, ls, self.folder, log=lambda msg, k="info": self.br.log.emit(msg, k), shadow=self.shadow_model())
        elif self._loop is None or not self._loop.is_alive():
            self.eng.m, self.eng.live, self.eng.shadow = m, ls, self.shadow_model()
        return self.eng

    def _model_strip(self) -> None:
        m = self.panel.model()
        running = self._loop is not None and self._loop.is_alive() and self.eng is not None
        if running:
            m = self.eng.m
        else:
            want = "sip_delayed" if m.minute <= 1530 else "sip"
            if want != self.feed.currentText():
                self.feed.setCurrentText(want)
                self.add_log(f"overnight bars now come from '{want}' (a {m.minute // 100}:{m.minute % 100:02d} decision)", "info")
        bt = self.bt.ran_model
        same = bt is not None and bt.to_dict() == m.to_dict()
        tag = ("✓ the same model as the last backtest" if same else "⚠ not backtested in this session - run the Backtest tab first") if not running else "● running"
        self.auction.setEnabled(m.uses_auctions and not running)
        ls = self.eng.live if running else self.live_settings()
        orders = (f"   ·   live orders: market DAY {ls.mkt_close_s} s before the close / {-ls.mkt_open_s // 60} min before the open in place of MOC / MOO"
                  if m.uses_auctions and ls.auction_orders == "market" else "")
        self.model_strip.setText(f"LIVE MODEL - {describe(m)}{orders}   ·   {tag}")
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
            phrase = ML.ARM_PHRASE if self.paper else ML.LIVE_PHRASE
            txt, ok = QInputDialog.getText(self, "Arm " + ("paper trading" if self.paper else "REAL-MONEY trading"),
                                           f"The book '{self.panel.model().name}' will SEND orders to its {'paper' if self.paper else 'REAL-MONEY'} account every session: the "
                                           f"opening sales, intraday market orders and the closing buys.\n\nType  {phrase}  to continue:")
            if not ok or txt.strip() != phrase:
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
        self._env_text()
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
        self._run(lambda log: eng.connect(), lambda a: (self._connected(a), self.load_book()), "connecting ...")

    def load_book(self) -> None:
        """Open positions + trade history from the broker (read-only), next to the backtest's prices - on Connect, after each plan, every 10 minutes and on the button."""
        if self._book_busy:
            return
        eng = self.engine()
        self._book_busy = True

        def fn(log):
            if eng.broker is None:
                eng.connect()
            return eng.book_view()

        def done(v, err):
            self._book_busy = False
            if err:
                self.add_log(f"positions / trade history: {err}", "err")
            else:
                self._show_book(v, eng.mode)
        self._workers.start(FnWorker(fn), done, lambda s: None)

    def _show_book(self, v: dict, mode: str) -> None:
        f = lambda x, d=2: f"{x:,.{d}f}" if x is not None and np.isfinite(x) else "—"
        ts = lambda x: f"{pd.Timestamp(x):%m-%d %H:%M}" if x is not None and pd.notna(x) else "—"
        p, tr, tot = v["positions"], v["trades"], v["totals"]
        t = self.t_pos
        t.setRowCount(len(p))
        for r, x in enumerate(p.itertuples()):
            set_row(t, r, [x.leg, x.symbol, f(x.shares, 0), f(x.ordered, 0), ts(x.entry_at), f(x.entry_px), f(x.bt_entry), f(x.price), f"${x.value:,.0f}" if np.isfinite(x.value)
                           else "—", f"{x.pnl:+,.0f}" if np.isfinite(x.pnl) else "—", x.status],
                    {9: GREEN if np.isfinite(x.pnl) and x.pnl > 0 else RED, 10: GREEN if x.status == "filled" else YELLOW}, align_left=(0, 1, 10))
        t = self.t_trades
        t.setRowCount(len(tr))
        for r, x in enumerate(tr.iloc[::-1].itertuples()):
            set_row(t, r, [x.leg, x.symbol, f(x.shares, 0), ts(x.entry_at), f(x.entry_px), f(x.bt_entry), ts(x.exit_at), f(x.exit_px), f(x.bt_exit), x.exit_how,
                           f"{x.pnl:+,.2f}", f"{x.bt_pnl:+,.2f}" if np.isfinite(x.bt_pnl) else "—", f"{x.diff:+,.2f}" if np.isfinite(x.diff) else "—"],
                    {10: GREEN if x.pnl > 0 else RED, 12: GREEN if np.isfinite(x.diff) and x.diff >= 0 else RED}, align_left=(0, 1, 9))
        oh = v.get("orders")
        t = self.t_hist
        t.setRowCount(0 if oh is None else len(oh))
        for r, x in enumerate(oh.itertuples() if oh is not None else []):
            done = x.filled_qty >= x.qty
            set_row(t, r, [ts(x.submitted_at), x.symbol, x.side.upper(), f(x.qty, 0), x.order, x.leg, x.status, f(x.filled_qty, 0), f(x.filled_avg_price),
                           ts(x.filled_at)], {2: GREEN if x.side == "buy" else RED, 6: GREEN if done else (YELLOW if x.filled_qty else RED)}, align_left=(1, 4, 5, 6))
        self.add_log(f"{mode.upper()} account: {len(p)} open position(s), {tot.get('orders', 0)} order(s) ({tot.get('unfilled', 0)} not filled), "
                     f"{tot['trades']} closed trade(s) since {pd.Timestamp(tot['since']):%Y-%m-%d} - live P&L "
                     f"${tot['pnl']:+,.0f} vs the backtest's ${tot['bt_pnl']:+,.0f} on the same shares ({tot['priced']} priced; difference ${tot['diff']:+,.0f})", "ok")
        for n in v["notes"]:
            self.add_log(n, "warn")

    def _connected(self, a: dict) -> None:
        self.acct_lbl.setText(f"{a['number'][:4]}**** · {a['status']} · equity {big_money(a['equity'])} · buying power {big_money(a['buying_power'])} · margin {a['multiplier']:.0f}x · "
                              f"keys: {a['keys'].upper()} · market {'open' if a['market_open'] else 'closed'} · {a['latency_ms']} ms")
        self.add_log(f"connected (keys from the {a.get('key_src', '?')})" + ("" if a["keys"] == "own" else " - fine for DRY RUN; the project's key file is needed to arm"),
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
        self._run(fn, self._planned, "planning the overnight decision (dry run) ...")

    def check_now(self) -> None:
        eng = self.engine()
        if not eng.m.intraday_symbol:
            self.add_log("this book has no intraday leg", "warn")
            return

        def fn(log):
            if eng.broker is None:
                eng.connect()
            now = ML.now_et()
            tm = eng.times(now.normalize())
            past = [s for s, end, due in (tm["checks"] if tm else []) if due <= now]
            if not past:
                raise RuntimeError("no intraday check has come due today yet (the first is at 10:00 ET on a session day)")
            return eng.intraday_check(past[-1], now, force=True)
        self._run(fn, self._checked, "intraday check (dry run) ...")

    def _checked(self, row: dict) -> None:
        self._checks = [r for r in self._checks if pd.Timestamp(r["day"]) == pd.Timestamp(row["day"])] + [row]
        t = self.t_checks
        t.setRowCount(len(self._checks))
        for r, x in enumerate(self._checks):
            act = x.get("action") or "-"
            set_row(t, r, [f"{pd.Timestamp(x['bar_end']):%H:%M}", f"{x['price']:,.2f}", f"{x['band']:,.2f}", f"{x['vwap']:,.2f}", f"{x['held']:,}", act.upper(),
                           f"{x.get('qty', 0):,}", x.get("status", ""), x.get("src", "")], {5: GREEN if act == "buy" else (RED if act == "sell" else None)}, align_left=(7,))
        last = self._checks[-1]
        self.k["Intraday"].set(f"{last['held']:,} held" if last["held"] else "flat", None,
                               f"{pd.Timestamp(last['bar_end']):%H:%M}: {last['price']:,.2f} vs band {last['band']:,.2f} → {(last.get('action') or 'hold').upper()}")
        for n in last.get("notes") or []:
            self.add_log(n, "warn")

    def _planned(self, plan: ML.MomPlan) -> None:
        self.plan = plan
        o, c, p = plan.orders, plan.candidates, plan.positions
        self.k["Capital"].set(big_money(plan.capital), None, f"equity {big_money(plan.equity)}")
        ent = int((o.kind == "entry").sum()) if len(o) else 0
        self.k["Overnight book"].set(f"{ent} names" if plan.trend else "flat (trend)", None,
                                     ("SPY above its 200-day mean" if plan.trend else "SPY below its 200-day mean: no overnight positions") +
                                     (f" · ML model {plan.ml['year']}" if plan.ml else ""))
        self.k["Orders"].set(f"{len(o)}", None, f"{int((o.kind == 'exit').sum()) if len(o) else 0} exits · {ent} entries")
        t = self.t_orders
        t.setRowCount(len(o))
        for r, x in enumerate(o.itertuples()):
            set_row(t, r, [x.symbol, x.side.upper(), f"{x.qty:,}", "market-on-close" if x.tif == "cls" else x.tif.upper(), f"{x.ref_px:,.2f}",
                           f"${x.notional:,.0f}" if np.isfinite(x.notional) else "—", x.why, "yes" if x.ok else "NO", x.note],
                    {1: GREEN if x.side == "buy" else RED, 7: None if x.ok else RED}, align_left=(6, 8))
        t = self.t_cand
        t.setRowCount(len(c))
        for r, x in enumerate(c.itertuples()):
            set_row(t, r, [f"{x.rank}", x.symbol, f"{x.score:.4f}", f"{x.move * 100:+.2f}%", f"{x.forecast:.4f}" if np.isfinite(x.forecast) else "—", f"{x.last:,.2f}",
                           "YES" if x.taken else "no"], {6: GREEN if x.taken else None}, align_left=(1,))
        if plan.shadow is not None:
            sm = self.eng.shadow if self.eng else None
            traded = set(o[o.kind == "entry"].symbol) if len(o) else set()
            self.shadow_lbl.setText(f"Shadow '{sm.name if sm else '?'}' on the same data: {len(plan.shadow)} name(s), {len(traded & set(plan.shadow.symbol))} also traded.")
            t = self.t_shadow
            t.setRowCount(len(plan.shadow))
            for r, x in enumerate(plan.shadow.itertuples()):
                set_row(t, r, [x.symbol, f"{x.qty:,}", f"{x.score:.4f}", f"{x.rank}", f"{x.move * 100:+.2f}%", "yes" if x.symbol in traded else "no"], align_left=(0,))
        for n in plan.notes:
            self.add_log(n, "warn")
        self.add_log(f"overnight plan for {plan.day:%Y-%m-%d} ({'decision time' if plan.real else 'manual dry run'}): {len(o)} order(s)", "ok")
        for x in p.itertuples():                                                                # what happens to the positions held at the decision
            if x.exit:
                self.add_log(f"{x.symbol} ({x.shares:+,} held): {x.exit}", "info")
        self.tabs.setCurrentWidget(self.t_orders)
        self.load_book()

    def start_loop(self) -> None:
        if self._loop is not None and self._loop.is_alive():
            return
        m = self.panel.model()
        if self.bt.ran_model is None or self.bt.ran_model.to_dict() != m.to_dict():
            if QMessageBox.question(self, "Model not backtested", "This exact model has not been backtested in this session (Backtest tab). Start the loop anyway?") \
                    != QMessageBox.StandardButton.Yes:
                return
        eng = self.engine()
        strip_id = lambda x: {k: v for k, v in x.to_dict().items() if k not in ("name", "note")}
        if eng.shadow is not None and strip_id(eng.shadow) == strip_id(m):
            if QMessageBox.question(self, "Shadow = traded book", "The shadow model is the same book as the one being traded, so the comparison tells nothing.\n\n"
                                    f"The traded book is the model in the left panel; the shadow should differ (e.g. '{PR.SHADOW}').\n\nStart anyway?") \
                    != QMessageBox.StandardButton.Yes:
                return
        if self.armed.isChecked():
            eng_ = self.engine()
            how = (f"market DAY orders {eng_.live.mkt_close_s} s before the close and {-eng_.live.mkt_open_s // 60} min before the open, in place of the "
                   f"market-on-close / market-on-open orders" if eng_.emulate else "market-on-close / market-on-open orders")
            if QMessageBox.question(self, "Start ARMED loop", f"Start the daily loop ARMED on the {'paper' if self.paper else 'REAL-MONEY'} account with\n\n{describe(m)}\n\n"
                                                                f"It sends the overnight sales and buys ({how}) and the intraday market orders every session until you "
                                                                f"stop it.") != QMessageBox.StandardButton.Yes:
                return
        try:
            os.makedirs(os.path.join(self.folder, "reports"), exist_ok=True)
            with open(os.path.join(self.folder, "live_model.json"), "w", encoding="utf-8") as fh:
                json.dump({**m.to_dict(), "_shadow": (eng.shadow.to_dict() if eng.shadow else None), "_auction_orders": eng.live.auction_orders}, fh,
                          indent=2)                                                             # the headless runner trades the same file, with the same orders
        except OSError as e:
            self.add_log(f"could not save live_model.json: {e}", "warn")
        self._stop.clear()

        def body():
            msg = ""
            try:
                eng.run(self._stop, on_plan=lambda p: self.br.plan.emit(p), on_check=lambda r: self.br.check.emit(r))
            except Exception as e:                                                            # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
            self.br.stopped.emit(msg)
        self._loop = threading.Thread(target=body, name="mb-loop", daemon=True)
        self._loop.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.panel.setEnabled(False)                                                          # the running model cannot be edited under the loop
        self.feed.setEnabled(False); self.shadow.setEnabled(False); self.auction.setEnabled(False); self.account.setEnabled(False)
        self._model_strip()

    def stop_loop(self) -> None:
        self._stop.set()
        self.add_log("stopping the loop ...")

    def _loop_ended(self, msg: str) -> None:
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.panel.setEnabled(True); self.feed.setEnabled(True); self.shadow.setEnabled(True); self.account.setEnabled(True)   # the auction box: _model_strip
        if msg:
            self.add_log("loop ended: " + msg, "err")
        self._model_strip()

    def compare(self) -> None:
        eng = self.engine()

        def done(res):
            df, fr = res
            t = self.t_par
            t.setRowCount(len(df) + len(fr))
            for r, x in enumerate(df.itertuples()):
                set_row(t, r, [f"{pd.Timestamp(x.day):%Y-%m-%d}", x.leg, x.what, x.symbol, str(x.live), str(x.backtest), x.match], {6: GREEN if x.match == "same" else RED})
            for r, x in enumerate(fr.itertuples(), start=len(df)):                             # fills vs the backtest's prices (official close / open)
                full = x.filled >= x.qty
                live = f"{x.filled:,.0f} / {x.qty:,.0f} @ {x.fill_px:,.2f}" if x.filled else f"0 / {x.qty:,.0f}"
                ref = f"official {x.ref} {x.ref_px:,.2f}" if np.isfinite(x.ref_px) else f"official {x.ref}: update the data"
                res_ = "not filled" if not x.filled else ((f"{x.cost_bp:+.1f} bp" if np.isfinite(x.cost_bp) else "filled") + ("" if full else " (part)"))
                set_row(t, r, [f"{pd.Timestamp(x.day):%Y-%m-%d}", "fill", f"{x.what} ({x.tif})", x.symbol, live, ref, res_], {6: GREEN if full else (YELLOW if x.filled else RED)})
            same = int((df.match == "same").sum()) if len(df) else 0
            self.add_log(f"live vs backtest data: {same} of {len(df)} identical" + ("" if same == len(df) else " - see the 'Live vs backtest' tab"), "ok" if same == len(df) else "warn")
            if len(fr):
                full = int((fr.filled >= fr.qty).sum())
                bt_note = (f"the backtest's auction prices plus {eng.m.cost_auction_bp:g} bp per side" + (" - market DAY orders in their place: this is what they cost"
                                                                                                        if eng.emulate else ""))
                self.add_log(f"fills vs the official close / open: {full} of {len(fr)} orders filled completely, cost {fr.cost_bp.mean():+.1f} bp on average (filled ones; "
                             f"{bt_note})", "ok" if full == len(fr) else "warn")
            self.tabs.setCurrentWidget(t)
        self._run(lambda log: (eng.check_against_backtest(log=log), eng.fill_report()), done, "checking the journal against the backtest data ...")

    def shadow_report(self) -> None:
        eng = self.engine()

        def done(df):
            t = self.t_shadow_rep
            t.setRowCount(len(df))
            for r, x in enumerate(df.itertuples()):
                f = lambda v: f"{v * 100:+.2f}%" if np.isfinite(v) else "—"
                set_row(t, r, [f"{pd.Timestamp(x.day):%Y-%m-%d}", f"{x.traded}", f(x.traded_ret), f"{x.shadow}", f(x.shadow_ret), f"{x.common}"],
                        {2: GREEN if np.isfinite(x.traded_ret) and x.traded_ret > 0 else RED, 4: GREEN if np.isfinite(x.shadow_ret) and x.shadow_ret > 0 else RED})
            if len(df):
                a, b = df.traded_ret.dropna(), df.shadow_ret.dropna()
                self.add_log(f"shadow report: {len(df)} night(s) · traded mean {a.mean() * 1e4:+.1f} bp · shadow mean {b.mean() * 1e4:+.1f} bp", "ok")
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.t_shadow_rep.parentWidget()))
        self._run(lambda log: eng.shadow_report(), done, "shadow report ...")

    def reset_hwm(self) -> None:
        if QMessageBox.question(self, "Reset drawdown base / clear KILL", "Make the current equity the new high-water mark, clear a drawdown stop and clear a KILL "
                                                                          "(the book may trade again once armed)?") != QMessageBox.StandardButton.Yes:
            return
        eng = self.engine()

        def fn(log):
            if eng.broker is None:
                eng.connect()
            return eng.reset_drawdown()
        self._run(fn, lambda v: self.add_log(f"drawdown base reset to ${v:,.0f}; drawdown stop and KILL cleared", "ok"), "resetting ...")

    def kill(self) -> None:
        txt, ok = QInputDialog.getText(self, "KILL", f"Cancel this book's open orders and CLOSE every position in its symbols on its {'paper' if self.paper else 'REAL-MONEY'} "
                                                     f"account.\n\nType  KILL  to continue:")
        if not ok or txt.strip() != "KILL":
            return
        self.stop_loop()
        if self._loop is not None and self._loop.is_alive():
            self._loop.join(20)                                                                 # the loop finishes its step (a submit in progress) before KILL acts
        self._set_armed(False)
        self._env_text()
        eng = self.engine()
        self.add_log("KILL ...", "warn")

        def done(res, err):
            if err:
                self.add_log(f"KILL failed: {err}", "err")
                return
            self.add_log("KILL: " + "; ".join(f"{r['symbol']}:{r['status']}" for r in res[:40]) + " - nothing more is sent until 'Reset drawdown base / clear KILL'",
                         "warn")
            self.load_book()
        self._workers.start(FnWorker(lambda log: eng.kill()), done, lambda s: None)              # not through _run: KILL never waits for another action

    def _tick(self) -> None:
        now = ML.now_et()
        alive = self._loop is not None and self._loop.is_alive()
        self.k["Loop"].set("running" if alive else "stopped", GREEN if alive else None, f"{now:%a %H:%M:%S} ET")
        if now.second == 30 and now.minute % 10 == 0 and self.eng is not None and self.eng.broker is not None:
            self.load_book()                                                                    # fills / the after-close step show up without a click
        if now.second % 10 == 0:
            try:
                eng = self.eng or ML.MomLiveEngine(self.panel.model(), self.live_settings(), self.folder, log=lambda *a: None)
                tm = eng.times(now.normalize())
                if tm is None:
                    self.k["Today"].set("no session")
                else:
                    nxt = next((f"check {end:%H:%M}" for _, end, due in tm["checks"] if due > now), None)
                    emu = eng.emulate
                    nxt = nxt if nxt and now < tm["decision"] else ("overnight decision" if now < tm["decision"] else
                                                                   ("market orders at the close" if emu and now < tm["emu_close"] else
                                                                    ("after-close step" if now < tm["close"] + pd.Timedelta(minutes=5) else "done")))
                    when = f"market orders {tm['emu_open']:%H:%M:%S} / {tm['emu_close']:%H:%M:%S}" if emu else f"MOC cut-off {tm['cutoff']:%H:%M:%S}"
                    self.k["Today"].set(f"decide {tm['decision']:%H:%M:%S}", None, f"next: {nxt} · {when}")
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
        self.qs.setValue("live/auction", self.auction.currentData())
        self.qs.setValue("live/shadow", self.shadow.currentText())

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
        v = ML._OLD_ORDERS.get(self.qs.value("live/auction"), self.qs.value("live/auction"))     # the account is not restored: the tab always starts on PAPER
        self.auction.setCurrentIndex(max(self.auction.findData(v if v in ML.AUCTION_ORDERS else "auction"), 0))
        v = self.qs.value("live/shadow")
        v = PR.RENAMED.get(v, v)
        self.shadow.setCurrentText(v if v == NO_SHADOW or v in PR.PRESETS else PR.SHADOW)          # default: the rule-only book in the shadow
        self._model_strip()
