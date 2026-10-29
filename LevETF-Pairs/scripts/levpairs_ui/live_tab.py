"""Live trading tab: connect to Alpaca, pre-flight, run the intraday engine (DRY RUN shadow book, or ARMED market orders on the paper / live account), replay the last session, reconcile, KILL.

Nothing is sent unless the 'ARM' box is ticked; a real-money account additionally asks for a typed phrase. The window always starts on PAPER + DRY RUN."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget)

import levpairs_data as D
import levpairs_live as LV
from levpairs_sim import PAIR_ORDER
from supertrend_ui.theme import ACCENT, GREEN, MUTED, RED, TEXT, YELLOW, Kpi, QComboBox, big_money, box_style, make_table, muted, pct, set_row, spin
from supertrend_ui.workers import FnWorker, WorkerBag

from .settings_panel import SettingsPanel

LOG_COLORS = {"err": RED, "warn": YELLOW, "ok": GREEN, "trade": ACCENT, "info": TEXT}


class LiveTab(QWidget):
    log_sig = pyqtSignal(str, str)                               # engine threads emit here; the GUI thread writes the log

    def __init__(self, settings: SettingsPanel, qs: QSettings, reports_dir: Callable[[], str] = lambda: "."):
        super().__init__()
        self._session_engine: Optional[LV.LiveEngine] = None
        self.sp, self.qs, self.reports_dir = settings, qs, reports_dir
        self.eng: Optional[LV.LiveEngine] = None
        self.acct: dict = {}
        self._workers = WorkerBag()
        self._busy = False
        self._session_worker: Optional[FnWorker] = None
        self._live_ok = False                                    # set only by the typed 'ARM LIVE' phrase; cleared whenever the account changes
        self._auto_day = None
        root = QVBoxLayout(self)
        banner = QLabel("Heads-up: Valid has NOT signed off · with next-open fills and quoted-spread costs the backtest loses money on every time frame (gross edge under 1 bp per trade, costs 5-11 bp) · "
                        "this tab is for DRY-RUN observation and measuring REAL fill costs on the paper account · start DRY RUN, then PAPER, never size it. Everything is flat before the close.")
        banner.setObjectName("warn"); banner.setWordWrap(True)
        root.addWidget(banner)
        self.env = QLabel(""); self.env.setWordWrap(True)
        root.addWidget(self.env)
        top = QHBoxLayout()
        top.addWidget(self._account_box(), 3); top.addWidget(self._mode_box(), 4); top.addWidget(self._control_box(), 3)
        root.addLayout(top)
        k = QHBoxLayout()
        self.k = {n: Kpi(n) for n in ("Mode", "Session", "Deployed × gross", "Day P&L (shadow)", "Open pairs", "Engine")}
        for w in self.k.values():
            k.addWidget(w)
        root.addLayout(k)
        self.tabs = QTabWidget()
        self.t_pairs = make_table(["Pair", "Position now (long / short which product)", "Shares 3x / 1x", "s now", "Half-life (min)", "σ eq (bp)", "β", "Entry s", "Trades", "P&L $", "Status"])
        self.tabs.addTab(self.t_pairs, "Pairs now")
        self.t_dec = make_table(["Time", "Pair", "s", "What the engine does", "Position after", "Note"])
        self.tabs.addTab(self.t_dec, "Decisions")
        self.t_book = make_table(["Symbol", "Ledger", "Broker", "Diff"])
        self.tabs.addTab(self.t_book, "Ledger vs broker")
        self.t_pf = make_table(["Check", "Result", "Detail"])
        self.tabs.addTab(self.t_pf, "Pre-flight")
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumBlockCount(3000)
        self.tabs.addTab(self.log_box, "Log")
        root.addWidget(self.tabs, 1)
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick); self.timer.start(1000)
        self._refresh_env()
        self.log_sig.connect(self.add_log)
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
        self.armed = QCheckBox("ARM - allow sending orders"); self.armed.setToolTip("Unticked = DRY RUN: the shadow book runs (own state file), nothing is sent.")
        self.feed = QComboBox(); self.feed.addItems(["iex", "sip"]); self.feed.setToolTip("Feed of the live signals. This account may not read real-time SIP, only IEX.")
        self.eq_frac = spin(0.01, 1.0, 0.25, 2, 0.05); self.cap_over = spin(0, 1e9, 0, 0, 10_000); self.cap_over.setPrefix("$ "); self.cap_over.setToolTip("0 = use equity x fraction")
        self.min_leg = spin(0, 100_000, 200, 0, 50); self.min_leg.setPrefix("$ ")
        self.max_gross = spin(0.05, 1.0, 0.25, 2, 0.05); self.max_gross.setToolTip("Gross of all open pairs (both legs, each at its own price) as a share of ACCOUNT EQUITY. RM: start at 25%.")
        self.max_leg = spin(0.01, 0.5, 0.10, 2, 0.01); self.max_leg.setToolTip("No single leg above this share of account equity (the hedge leg is about 3x the other one).")
        self.halt = spin(0.0, 0.2, 0.01, 3, 0.005)
        self.accept_iex = QCheckBox("Accept IEX signals when ARMED (unreliable: corr 0.2-0.57 with SIP)")
        self.accept_iex.toggled.connect(self._accept_iex_toggled)
        self.accept_iex.setToolTip("Without real-time SIP data the signals come from IEX bars, which are empty in many minutes for the thinner ETFs. The report's fidelity test shows the IEX s-scores "
                                   "only weakly track the consolidated ones and the strategy trades several times more on them. Leave unticked: an ARMED session on IEX is then refused.")
        self.orph_eod = QCheckBox("DEDICATED account: also close unexplained positions at 15:52")
        self.orph_eod.setToolTip("At the hard flatten, close anything held in the six ETFs that this engine's ledger does not explain. Only for an account used by this strategy alone.")
        self.auto = QCheckBox("Auto: start the session engine at 09:25 ET on trading days")
        f.addRow(self.armed); f.addRow("Signal feed", self.feed); f.addRow("Equity fraction", self.eq_frac); f.addRow("…or fixed capital", self.cap_over); f.addRow("Min leg ticket", self.min_leg)
        f.addRow("Max gross (of equity)", self.max_gross); f.addRow("Max single leg (of equity)", self.max_leg); f.addRow("Daily-loss halt (of deployed)", self.halt); f.addRow(self.accept_iex); f.addRow(self.orph_eod); f.addRow(self.auto)
        return g

    def _control_box(self) -> QGroupBox:
        g = QGroupBox("Run"); v = QVBoxLayout(g)
        self.btn_pf = QPushButton("Pre-flight check"); self.btn_pf.clicked.connect(self.preflight)
        self.btn_start = QPushButton("Start session engine"); self.btn_start.setObjectName("primary"); self.btn_start.clicked.connect(self.start_session)
        self.btn_stop = QPushButton("Stop engine (flattens if armed)"); self.btn_stop.clicked.connect(self.stop_session); self.btn_stop.setEnabled(False)
        self.btn_replay = QPushButton("Replay the last cached session (dry run)"); self.btn_replay.clicked.connect(self.replay_last)
        self.btn_rec = QPushButton("Reconcile with broker"); self.btn_rec.clicked.connect(self.reconcile)
        self.btn_reset = QPushButton("Clear 'broken' flags"); self.btn_reset.clicked.connect(self.reset_broken)
        self.btn_reset.setToolTip("After a leg failed and the other was unwound: re-enable the pair once you have checked the account.")
        self.btn_kill = QPushButton("KILL: cancel orders + flatten engine positions"); self.btn_kill.setObjectName("danger"); self.btn_kill.clicked.connect(self.kill)
        self.btn_orph = QPushButton("Flatten unexplained positions (the six ETFs)"); self.btn_orph.setObjectName("danger"); self.btn_orph.clicked.connect(self.flatten_orphans)
        self.btn_orph.setToolTip("Positions in TQQQ/QQQ/SOXL/SMH/SPXL/SPY at the broker that the ledger does not explain (after a crash between legs, an unknown order state, manual trades). "
                                 "Armed engines only. On a shared account this closes anyone's position in those symbols.")
        for b in (self.btn_pf, self.btn_start, self.btn_stop, self.btn_replay, self.btn_rec, self.btn_reset, self.btn_kill, self.btn_orph):
            v.addWidget(b)
        return g

    # ── helpers ──────────────────────────────────────────────────────────────────
    def paper(self) -> bool:
        return self.account.currentIndex() == 0

    def live_settings(self) -> LV.LiveSettings:
        return LV.LiveSettings(paper=self.paper(), armed=self.armed.isChecked() and (self.paper() or self._live_ok), feed=self.feed.currentText(), equity_fraction=self.eq_frac.value(),
                               capital_override=self.cap_over.value(), min_leg_usd=self.min_leg.value(), max_gross_nav=self.max_gross.value(), max_leg_nav=self.max_leg.value(), daily_loss_halt=self.halt.value(), accept_iex=self.accept_iex.isChecked(), flatten_orphans_eod=self.orph_eod.isChecked())

    def new_engine(self) -> LV.LiveEngine:
        return LV.LiveEngine(self.sp.sig_cfg(), self.sp.sim_cfg(), self.live_settings(), log=self._log_threadsafe)

    def _log_threadsafe(self, msg: str, kind: str = "info") -> None:
        self.log_sig.emit(msg, kind)                               # queued to the GUI thread whichever thread logs

    def session_running(self) -> bool:
        return self._session_worker is not None and self._session_worker.isRunning()

    def effective_mode(self) -> tuple[bool, bool]:
        """(armed, paper) of what is actually running: the session's own settings while one runs, the controls otherwise."""
        if self.session_running() and self._session_engine is not None:
            return self._session_engine.live.armed, self._session_engine.live.paper
        ls = self.live_settings()
        return ls.armed, ls.paper

    def engine_for_action(self) -> LV.LiveEngine:
        """The engine KILL / flatten / reconcile act on: the running session's, else the last engine ONLY if it has the account and mode the controls show now and is not a replay (a replay engine is unarmed
        and has no broker: acting through it logged 'none' and sent nothing while the broker held positions), else a fresh engine built from the controls."""
        if self.session_running() and self._session_engine is not None:
            return self._session_engine
        e = self.eng
        if e is not None and not getattr(e, "is_replay", False):
            ls = self.live_settings()
            if e.live.armed == ls.armed and e.live.paper == ls.paper:
                return e
        return self.new_engine()

    def _set_locked(self, on: bool) -> None:
        """While a session runs the account, ARM, feed and limits cannot be changed: the running engine keeps the settings it started with, and the controls must not say otherwise."""
        for w in (self.account, self.armed, self.feed, self.accept_iex, self.orph_eod, self.eq_frac, self.cap_over, self.min_leg, self.max_gross, self.max_leg, self.halt, self.btn_replay, self.btn_pf):
            w.setEnabled(not on)

    def add_log(self, msg: str, kind: str = "info") -> None:
        if "|" in msg[:6] and msg.split("|", 1)[0] in LOG_COLORS:
            kind, msg = msg.split("|", 1)
        fmt = QTextCharFormat(); fmt.setForeground(QColor(LOG_COLORS.get(kind, TEXT)))
        c = self.log_box.textCursor(); c.movePosition(QTextCursor.MoveOperation.End)
        c.insertText(f"{datetime.now():%H:%M:%S}  {msg}\n", fmt)
        self.log_box.setTextCursor(c); self.log_box.ensureCursorVisible()

    def _refresh_env(self) -> None:
        if self.paper():
            txt, col = ("PAPER account - simulated money. DRY RUN unless ARMED.", GREEN)
        else:
            txt, col = ("LIVE account - REAL MONEY. Orders sent while ARMED are real.", RED)
        self.env.setText(txt); self.env.setStyleSheet(box_style(col, 0.14, True, 6))

    def _account_changed(self, *_) -> None:
        """Switching accounts always disarms: arming on paper must never carry over to the real-money account (and the typed phrase has to be given again)."""
        self._live_ok = False
        if self.armed.isChecked():
            self.armed.blockSignals(True); self.armed.setChecked(False); self.armed.blockSignals(False)
            self.add_log("account changed - DISARMED", "warn")
        self._refresh_env()
        if not self.session_running():
            self.eng = None

    def _arm_toggled(self, on: bool) -> None:
        if self._session_worker is not None and self._session_worker.isRunning():
            self.add_log("the running session keeps the mode it started with - stop and start the engine to apply the change", "warn")
        if on and not self.paper():
            txt, ok = QInputDialog.getText(self, "Arm LIVE trading", "This is a REAL-MONEY account. Orders will be sent while the engine runs.\n\nType  ARM LIVE  to continue:")
            if not ok or txt.strip() != "ARM LIVE":
                self.armed.blockSignals(True); self.armed.setChecked(False); self.armed.blockSignals(False)
                self._live_ok = False
                return
            self._live_ok = True
        if not on:
            self._live_ok = False
        self.add_log("ARMED - the engine will send orders" if on else "DISARMED - dry run", "warn" if on else "info")

    def _accept_iex_toggled(self, on: bool) -> None:
        if on:
            txt, ok = QInputDialog.getText(self, "Accept IEX signals", "The IEX s-scores of these pairs correlate only 0.2-0.57 with the consolidated-tape scores the strategy was tested on, and the rules trade 8x as often "
                                           "on them. Sending orders on them is not supported by any evidence in this project.\n\nType  ACCEPT IEX  to allow it:")
            if not ok or txt.strip() != "ACCEPT IEX":
                self.accept_iex.blockSignals(True); self.accept_iex.setChecked(False); self.accept_iex.blockSignals(False)

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
        eng = self.new_engine()
        self._run(lambda log: eng.connect(), self._connected, "connecting to Alpaca ...")

    def _connected(self, a: dict) -> None:
        self.acct = a
        self.acct_lbl.setText(f"{'PAPER' if a['paper'] else 'LIVE'} {a['number']} · {a['status']} · equity {big_money(a['equity'])} · buying power {big_money(a['buying_power'])} · "
                              f"margin {a['multiplier']}x · shorting {'on' if a['shorting'] else 'OFF'} · market {'open' if a['market_open'] else 'closed'} · {a['latency_ms']} ms")
        self.add_log("connected", "ok")

    def preflight(self) -> None:
        eng = self.new_engine()
        self._run(lambda log: eng.preflight(), self._pf_done, "pre-flight ...")

    def _pf_done(self, checks: list) -> None:
        t = self.t_pf
        t.setRowCount(len(checks))
        for r, c in enumerate(checks):
            set_row(t, r, [c.name, c.level.upper(), c.msg], {1: {"ok": GREEN, "warn": YELLOW, "fail": RED}[c.level]}, align_left=(2,))
        self.tabs.setCurrentWidget(t)
        fails = [c for c in checks if c.level == "fail"]
        self.add_log("pre-flight: " + ("FAILED - " + "; ".join(c.msg for c in fails) if fails else "no blocking problems"), "err" if fails else "ok")

    def start_session(self) -> None:
        if self._session_worker is not None and self._session_worker.isRunning():
            return
        ls = self.live_settings()
        if ls.armed and QMessageBox.question(self, "Start ARMED session", f"The engine will send MARKET orders to the {'PAPER' if ls.paper else 'LIVE'} account for the rest of the session, flatten before the close "
                                             f"and stop.\n\nDeployed capital: {'$%s' % format(ls.capital_override, ',.0f') if ls.capital_override > 0 else '%.0f%% of equity' % (100 * ls.equity_fraction)}. Continue?") != QMessageBox.StandardButton.Yes:
            return
        eng = self.new_engine()
        self.eng = self._session_engine = eng
        holder: dict = {}

        def fn(log):
            eng.run_session(should_stop=holder["w"].cancelled)
            return True
        w = FnWorker(fn)
        holder["w"] = w
        self._session_worker = w
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self._set_locked(True)
        self._workers.start(w, self._session_done, lambda s: self.add_log(s.strip()))
        self.add_log(f"session engine started ({LV.mode_name(ls).upper()})", "ok")

    def _session_done(self, res, err) -> None:
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self._session_worker = None
        self._session_engine = None
        self._set_locked(False)
        self.add_log("session engine stopped" if not err else f"session engine failed: {err}", "ok" if not err else "err")

    def stop_session(self) -> None:
        if self._session_worker is not None:
            self._session_worker.cancel()
            self.add_log("stopping: an ARMED engine flattens its ledger on the way out", "warn")

    def replay_last(self) -> None:
        sc, c, ls = self.sp.sig_cfg(), self.sp.sim_cfg(), self.live_settings()
        eq = self.acct.get("equity", 100_000.0)

        def fn(log):
            import levpairs_data as DD
            b = pd.read_parquet(DD.path1("SPY", ls.feed))
            date = b.index.normalize().unique()[-1]
            log(f"replaying {date:%Y-%m-%d} on {ls.feed.upper()} bars through the live decision path ({sc.tf}-minute bars) ...")
            eng, rows = LV.replay_session(date, sc, c, ls, equity=eq, log=lambda m, k="info": log(m))
            return date, eng, rows
        self._run(fn, self._replayed, "replay ...")

    def _replayed(self, res) -> None:
        date, eng, rows = res
        self.eng = eng
        self._show_rows(rows)
        pnl = sum(p.get("pnl_usd", 0.0) for p in eng.state["pairs"].values())
        self.add_log(f"replay of {date:%Y-%m-%d}: {int((rows.get('action', pd.Series(dtype=str)).astype(str).str.startswith('ENTER')).sum()) if len(rows) else 0} entries, shadow P&L ${pnl:,.2f} "
                     f"on ${eng.deploy * eng.G:,.0f} gross", "ok")
        self.tabs.setCurrentWidget(self.t_dec)

    def _show_rows(self, rows: pd.DataFrame) -> None:
        if rows is None or not len(rows):
            return
        note = rows["note"].fillna("") if "note" in rows else pd.Series("", index=rows.index)
        r = rows[(rows["action"].notna()) | (note != "")] if "action" in rows else rows
        r = r.tail(400)
        t = self.t_dec
        t.setRowCount(len(r))
        for i, x in enumerate(r.itertuples()):
            act = getattr(x, "action", "") if isinstance(getattr(x, "action", ""), str) else ""
            pos = getattr(x, "position", "") if isinstance(getattr(x, "position", ""), str) else ""
            set_row(t, i, [x.time, x.pair, f"{x.s:+.2f}" if hasattr(x, "s") and np.isfinite(x.s) else "—", act or "—", pos or "—", getattr(x, "note", "") if isinstance(getattr(x, "note", ""), str) else ""],
                    {3: GREEN if act.startswith("ENTER long") else RED if act.startswith("ENTER short") else YELLOW if act.startswith("EXIT") else None}, align_left=(3, 4, 5))

    def reconcile(self) -> None:
        eng = self.engine_for_action()

        def fn(log):
            eng.connect()
            if not hasattr(eng, "state"):
                eng.state = eng.load_state(pd.Timestamp.now().normalize())
            return eng.reconcile()
        self._run(fn, self._reconciled, "reconciling ...")

    def _reconciled(self, df: pd.DataFrame) -> None:
        t = self.t_book
        t.setRowCount(len(df))
        for r, x in enumerate(df.itertuples()):
            set_row(t, r, [x.symbol, f"{x.ledger:+,}", f"{x.broker:+,}", f"{x.diff:+,}"], {3: RED if x.diff else GREEN})
        self.tabs.setCurrentWidget(t)
        self.add_log(f"reconcile: {int((df['diff'] != 0).sum())} of {len(df)} symbols differ from the broker", "warn" if (df["diff"] != 0).any() else "ok")

    def reset_broken(self) -> None:
        eng = self.engine_for_action()                              # never a replay engine; an armed engine reads the broker before it clears anything

        def fn(log):
            if eng.live.armed and eng.tc is None:
                eng.connect()
            if not hasattr(eng, "state"):
                eng.state = eng._latest_state()
            eng.reset_broken()
            return True
        self._run(fn, lambda _: self.add_log("'broken' flags cleared", "ok"), "clearing the 'broken' flags ...")

    def kill(self) -> None:
        txt, ok = QInputDialog.getText(self, "KILL", f"Cancel the engine's open orders and close every position in the engine's ledger ({LV.mode_name(self.live_settings()).upper()} book).\n\nType  KILL  to continue:")
        if not ok or txt.strip() != "KILL":
            return
        eng = self.engine_for_action()
        eng.kill_flag.set()                                         # the running loop stops entering at once
        if self._session_worker is not None:
            self._session_worker.cancel()
        w = FnWorker(lambda log: eng.flatten("KILL"))               # not through _run: KILL must work while another action is busy; the engine lock serialises it with the session thread
        self._workers.start(w, lambda res, err: self.add_log(("KILL failed: " + err) if err else "KILL: " + "; ".join(f"{r['symbol']}:{r['status']}" for r in res[:12]), "err" if err else "warn"))
        self.add_log("KILL ...", "warn")
        if not eng.live.armed:
            for mode in ("paper", "live"):
                try:
                    import json as _json
                    st = _json.load(open(LV.state_path(mode), encoding="utf-8"))
                    open_pairs = [k for k, v in st.get("pairs", {}).items() if v.get("side")]
                    if open_pairs:
                        self.add_log(f"WARNING: this KILL acted on the DRY RUN book. The {mode.upper()} ledger on disk still holds {open_pairs}: tick ARM on that account and KILL again.", "err")
                except (OSError, ValueError):
                    pass

    def flatten_orphans(self) -> None:
        armed, paper = self.effective_mode()
        if not armed:
            QMessageBox.information(self, "Unexplained positions", "Only an ARMED engine reads the broker. Tick ARM (paper account first), then press this again.")
            return
        txt, ok = QInputDialog.getText(self, "Flatten unexplained positions", f"Close every position in TQQQ, QQQ, SOXL, SMH, SPXL, SPY on the {'PAPER' if paper else 'LIVE'} account that this engine's ledger does not explain - "
                                                                           "including positions someone else opened.\n\nType  FLATTEN  to continue:")
        if not ok or txt.strip() != "FLATTEN":
            return
        eng = self.engine_for_action()
        w = FnWorker(lambda log: (eng.connect() if eng.tc is None else None, eng.flatten_orphans())[1])
        self._workers.start(w, lambda res, err: self.add_log(("flatten unexplained failed: " + err) if err else "unexplained positions: " + ("; ".join(f"{r['symbol']}:{r['status']}" for r in res[:12]) or "none"), "err" if err else "warn"))
        self.add_log("flattening unexplained positions ...", "warn")

    def shutdown(self, wait_ms: int = 150_000) -> bool:
        """Stop the session thread and wait for its final flatten. Returns False if it is still running."""
        w = self._session_worker
        if w is None or not w.isRunning():
            return True
        w.cancel()
        return bool(w.wait(wait_ms))

    # ── clock / refresh ──────────────────────────────────────────────────────────
    def _tick(self) -> None:
        now = LV.clock_now()
        running = self._session_worker is not None and self._session_worker.isRunning()
        ls = self.live_settings()
        armed_now, paper_now = self.effective_mode()
        self.k["Mode"].set(("ARMED " + ("PAPER" if paper_now else "LIVE")) if armed_now else "DRY RUN", RED if (armed_now and not paper_now) else YELLOW if armed_now else None, f"feed {ls.feed.upper()}")
        self.k["Engine"].set("RUNNING" if running else "idle", GREEN if running else None, f"{now:%a %H:%M:%S} ET")
        eng = self.eng
        if eng is not None and hasattr(eng, "state"):
            st = eng.state
            pnl = sum(p.get("pnl_usd", 0.0) for p in st["pairs"].values())
            self.k["Session"].set(st["date"], None, f"last bar {st['last_j']}")
            self.k["Deployed × gross"].set(f"{big_money(eng.deploy)} × {eng.G:.2f}", None, f"= {big_money(eng.deploy * eng.G)} gross")
            self.k["Day P&L (shadow)"].set(f"${pnl:,.0f}", GREEN if pnl > 0 else RED if pnl < 0 else None, f"{pnl / max(eng.deploy, 1):+.2%} of deployed")
            alarms = st.get("alarms") or []
            self.k["Open pairs"].set(str(sum(1 for p in st["pairs"].values() if p.get("side"))), RED if alarms else None, (f"ALARMS: {len(alarms)} - {alarms[-1][:40]}" if alarms else "halted" if eng.halted else ""))
            self._show_pairs(eng)
        if self.auto.isChecked() and not running and not self._busy:
            key = now.strftime("%Y%m%d")
            if now.weekday() < 5 and (9, 25) <= (now.hour, now.minute) < (15, 45) and self._auto_day != key:
                self._auto_day = key
                self.add_log("auto: starting the session engine", "info")
                self.start_session()

    def _show_pairs(self, eng: LV.LiveEngine) -> None:
        t = self.t_pairs
        names = [n for n in PAIR_ORDER if n in eng.state["pairs"]]
        t.setRowCount(len(names))
        for r, n in enumerate(names):
            ps, v = eng.state["pairs"][n], eng.views.get(n)
            side = ps.get("side", 0)
            status = "BROKEN" if ps.get("broken") else "halted" if eng.halted else "ok"
            set_row(t, r, [n, LV.pair_position_text(n, ps), f"{ps.get('shares_L', 0):+,} / {ps.get('shares_B', 0):+,}",
                           f"{v.s:+.2f}" if v and np.isfinite(v.s) else "—", f"{v.hl_min:.0f}" if v and np.isfinite(v.hl_min) else "—", f"{v.sig_eq_bp:.1f}" if v and np.isfinite(v.sig_eq_bp) else "—",
                           f"{v.beta:.2f}" if v else "—", f"{ps['entry_s']:+.2f}" if ps.get("entry_s") is not None and np.isfinite(float(ps["entry_s"])) else "—", ps.get("n_trades", 0),
                           f"{ps.get('pnl_usd', 0.0):+,.0f}", status], {1: GREEN if side > 0 else RED if side < 0 else None, 9: GREEN if ps.get("pnl_usd", 0) > 0 else RED if ps.get("pnl_usd", 0) < 0 else None,
                                                                          10: RED if status != "ok" else None})

    # ── persistence ──────────────────────────────────────────────────────────────
    def save(self) -> None:
        for k, w in (("live/eq_frac", self.eq_frac), ("live/cap", self.cap_over), ("live/min_leg", self.min_leg), ("live/max_gross_nav", self.max_gross), ("live/max_leg_nav", self.max_leg), ("live/halt", self.halt)):
            self.qs.setValue(k, w.value())
        self.qs.setValue("live/feed", self.feed.currentText())

    def load(self) -> None:
        for k, w in (("live/eq_frac", self.eq_frac), ("live/cap", self.cap_over), ("live/min_leg", self.min_leg), ("live/max_gross_nav", self.max_gross), ("live/max_leg_nav", self.max_leg), ("live/halt", self.halt)):
            try:
                v = self.qs.value(k)
                if v is not None:
                    w.setValue(float(v))
            except (TypeError, ValueError):
                pass
        f = self.qs.value("live/feed")
        if f in ("iex", "sip"):
            self.feed.setCurrentText(str(f))
