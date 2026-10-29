"""Backtest tab: any date range, every trade, equity curve vs buy & hold, full metrics, and a robustness (neighbour) check."""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QDate, QSettings, Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QCheckBox, QGridLayout, QGroupBox, QLabel, QMessageBox, QProgressBar, QPushButton, QTableWidgetItem, QVBoxLayout, QWidget

from . import backtest as B
from . import data as D
from . import params as P
from .perf_view import ParamsView, PerformanceView
from .settings_panel import SettingsPanel
from .theme import ACCENT, GREEN, MUTED, RED, YELLOW, QComboBox, QDateEdit, QSpinBox, box_style, make_table, muted, spin, strip
from .verify import run_replay_process
from .workers import FnWorker, WorkerBag


class BacktestTab(QWidget):
    def __init__(self, settings: SettingsPanel, qs: QSettings, reports_dir: Callable[[], str] = lambda: "."):
        super().__init__()
        self.sp, self.qs = settings, qs
        self.result: Optional[B.BtResult] = None
        self.robust: Optional[B.Robustness] = None
        self._bars: dict = {}                              # (symbol, tf) -> bars already in memory
        self._last_refresh_try = 0.0
        self._lo = self._hi = None
        self._from_pinned = self._to_pinned = False        # True once YOU edited the date; until then 'To' follows the newest data
        self._programmatic = False
        self._workers = WorkerBag()
        self._bt_worker: Optional[FnWorker] = None
        root = QVBoxLayout(self)

        ctl = QGroupBox("Backtest-only settings (not used by the Live tab)")
        h = QGridLayout(ctl)
        self.symbol = QComboBox(); self.symbol.setEditable(True); self.symbol.addItems(["SOXL", "TQQQ", "SPXL", "UPRO", "TECL"])
        self.symbol.setToolTip("The strategy is built for SOXL (the live engine only reads SOXL bars). Other 3x ETFs download on first use for what-if tests.")
        self.tf = QComboBox(); self.tf.addItems(["5", "15", "30", "60"]); self.tf.setCurrentText("15")
        self.tf.setToolTip("Bar size in minutes. The live engine is fixed to 15, and the ATR period is counted in bars.")
        self.d0, self.d1 = QDateEdit(QDate(2016, 1, 4)), QDateEdit(QDate.currentDate())
        for d in (self.d0, self.d1):
            d.setCalendarPopup(True); d.setDisplayFormat("yyyy-MM-dd")
        self.cap = spin(100, 1e12, 100_000, 0, 10_000); self.cap.setPrefix("$ ")
        self.prime = QCheckBox("Prime indicators with bars before the start")
        self.prime.setChecked(True)
        self.prime.setToolTip("ON: the bars BEFORE the start date warm up the Supertrend / ATR, and no trade is taken before it - like the live engine, which always starts warm.\n"
                              "OFF: indicators start cold at the first bar, so the first ~19 sessions (500 bars) are dead time. The 2016-start full-history run is identical either way.")
        self.bench = QCheckBox("Buy && hold overlay"); self.bench.setChecked(True)
        self.refresh_chk = QCheckBox("Refresh data first"); self.refresh_chk.setChecked(True)
        self.refresh_chk.setToolTip("Before a run, download any newer 15-minute bars from Alpaca if the local cache is behind (a few seconds). If that fails the cached bars are used.")
        h.addWidget(QLabel("Symbol"), 0, 0); h.addWidget(self.symbol, 0, 1)
        h.addWidget(QLabel("Bar (min)"), 0, 2); h.addWidget(self.tf, 0, 3)
        h.addWidget(QLabel("From"), 0, 4); h.addWidget(self.d0, 0, 5)
        h.addWidget(QLabel("To"), 0, 6); h.addWidget(self.d1, 0, 7)
        self.quick = {}
        for i, (label, key) in enumerate((("1Y", 365), ("3Y", 3 * 365), ("5Y", 5 * 365), ("YTD", -1), ("All", 0))):
            b = QPushButton(label); b.setFixedWidth(56)
            b.clicked.connect(lambda _=False, k=key: self.quick_range(k))
            h.addWidget(b, 0, 8 + i); self.quick[label] = b
        self.btn_latest = QPushButton("→ latest"); self.btn_latest.setFixedWidth(78)
        self.btn_latest.setToolTip("Set 'To' to the newest data and keep it following the data as it refreshes")
        self.btn_latest.clicked.connect(self.follow_latest)
        h.addWidget(self.btn_latest, 0, 13)
        h.addWidget(QLabel("Initial capital"), 1, 0); h.addWidget(self.cap, 1, 1, 1, 2)
        h.addWidget(self.prime, 1, 3, 1, 5)
        h.addWidget(self.bench, 1, 8, 1, 2); h.addWidget(self.refresh_chk, 1, 10, 1, 3)
        self.strict = QCheckBox("Live-accurate entries")
        self.strict.setChecked(True)
        self.strict.setToolTip("ON (default): a trade is only taken when a live order COULD have known to take it - the trend, volatility regime and ADX as of the PREVIOUS bar's close, and the "
                               "re-entry limit priced off the previous bar. This is what the live engine does, and the replay check proves the two then trade identically.\n"
                               "OFF: the original backtest, which lets a flat re-entry use the entry bar's own close (about 13% of its trades - 210 of 1,580 - have no live equivalent, and they were "
                               "unusually good: +2.35% on average).")
        self.v_months = QSpinBox(); self.v_months.setRange(1, 12); self.v_months.setValue(2); self.v_months.setSuffix(" months")
        self.v_months.setToolTip("How far back from the end of the window to replay (about 25 s per month).")
        self.v_route = QComboBox(); self.v_route.addItems(["SOXL short", "long SOXS"])
        self.v_route.setToolTip("How the replayed broker handles a short: directly (SOXL shortable) or by buying SOXS (SOXL hard to borrow). SOXS exits are polled, so they can differ slightly.")
        self.verify_btn = QPushButton("Verify live = backtest")
        self.verify_btn.setToolTip("Replays the REAL live Engine, bar by bar, against a simulated broker over the last months of the window and compares every trade - and the position still open "
                                   "at the end - with this backtest. Takes a minute or two; runs in a separate process and sends nothing anywhere.")
        self.verify_btn.clicked.connect(self.verify)
        h.addWidget(self.strict, 2, 0, 1, 3); h.addWidget(QLabel("Verify the last"), 2, 3); h.addWidget(self.v_months, 2, 4, 1, 2); h.addWidget(QLabel("shorts as"), 2, 6)
        h.addWidget(self.v_route, 2, 7, 1, 2); h.addWidget(self.verify_btn, 2, 9, 1, 3)
        self.run_btn = QPushButton("Run backtest   (Ctrl+R)"); self.run_btn.setObjectName("primary"); self.run_btn.clicked.connect(self.run)
        self.rob_btn = QPushButton("Robustness check")
        self.rob_btn.setToolTip("Re-runs the strategy on a grid around the current factor and ATR period (15 runs, about 20 s) and says whether the result is a PLATEAU (neighbours hold up) or a SPIKE "
                                "(only good at the exact setting - the usual signature of over-fitting).")
        self.rob_btn.clicked.connect(self.run_robustness)
        self.prog = QProgressBar(); self.prog.setRange(0, 0); self.prog.setFixedWidth(120); self.prog.hide()
        self.status = muted("", wrap=False)
        h.addWidget(self.run_btn, 3, 0, 1, 3); h.addWidget(self.rob_btn, 3, 3, 1, 3); h.addWidget(self.prog, 3, 6, 1, 2); h.addWidget(self.status, 3, 8, 1, 6)
        self.using_lbl = muted(""); h.addWidget(self.using_lbl, 4, 0, 1, 14)
        self.info = muted(""); h.addWidget(self.info, 5, 0, 1, 14)
        self.fresh_lbl = QLabel(""); self.fresh_lbl.setWordWrap(True); h.addWidget(self.fresh_lbl, 6, 0, 1, 14)
        root.addWidget(ctl)

        self.perf = PerformanceView("backtest_trades.csv", reports_dir)
        self.params_view = ParamsView()
        self.perf.tabs.addTab(self._robust_page(), "Robustness")
        self.perf.tabs.addTab(self.params_view, "Parameters in use")
        root.addWidget(self.perf, 1)

        self.d0.dateChanged.connect(lambda *_: (setattr(self, "_from_pinned", self._from_pinned or not self._programmatic), self.refresh_using()))
        self.d1.dateChanged.connect(lambda *_: (setattr(self, "_to_pinned", self._to_pinned or not self._programmatic), self.refresh_fresh_label(), self.refresh_using()))
        for w in (self.cap,):
            w.valueChanged.connect(lambda *_: self.refresh_using())
        for w in (self.symbol, self.tf):
            w.currentTextChanged.connect(lambda *_: self.refresh_using())
        self.prime.toggled.connect(lambda *_: self.refresh_using()); self.bench.toggled.connect(lambda *_: self.refresh_using()); self.strict.toggled.connect(lambda *_: self.refresh_using())
        self.sp.changed.connect(self.refresh_using)
        self._fresh_timer = QTimer(self); self._fresh_timer.timeout.connect(self.refresh_fresh_label); self._fresh_timer.start(30000)
        self.refresh_using()

    # ── robustness page ──
    def _robust_page(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        self.rob_verdict = strip("Press \"Robustness check\": the strategy is re-run for factor ± 0.5 / 1.0 and ATR period x 0.75 / 1 / 1.25 over the same window and costs. "
                                 "A result that only works at the exact setting is probably over-fitted.", color=ACCENT)
        self.rob_table = make_table(["ATR period \\ factor"])
        v.addWidget(self.rob_verdict); v.addWidget(self.rob_table, 1)
        v.addWidget(muted("Cell: Sharpe (daily) · CAGR · max drawdown · trades. Green = close to or better than the current setting, red = much worse. The current setting is outlined (bold)."))
        return w

    # ── settings → objects ──
    def settings(self) -> P.BacktestSettings:
        a, b = self.d0.date(), self.d1.date()
        if a > b:
            raise ValueError("'From' date is after 'To' date")
        sym = self.symbol.currentText().strip().upper()
        if not sym:
            raise ValueError("enter a symbol")
        return P.BacktestSettings(symbol=sym, tf=int(self.tf.currentText()), start=a.toString("yyyy-MM-dd"), end=b.toString("yyyy-MM-dd"), capital=self.cap.value(),
                                  prime=self.prime.isChecked(), benchmark=self.bench.isChecked(), strict=self.strict.isChecked())

    def safe_settings(self) -> tuple:
        try:
            return self.settings(), ""
        except ValueError as e:
            return None, str(e)

    def refresh_using(self) -> None:
        bt, err = self.safe_settings()
        if bt is None:
            self.using_lbl.setText(f"Settings problem: {err}")
            return
        v = self.sp.values()
        self.using_lbl.setText(f"This backtest uses: {self.sp.preset_name()} | {P.to_config(v).strategyMode}, ATR {v['atrPeriod']:g}, factor {v['factor']:g}, run-up {v['riskReward']:g}%, take-profit {v['price_perc']:g}%, "
                               f"re-entry {v['buyback']:g}%, {v['equity_frac']:g}% per trade (shorts x{v['short_mult']:g}), {v['commission_bps']:g} bp/leg, "
                               f"{'touch' if v['touch_fill'] else 'pierce'} fills, {'live-accurate' if bt.strict else 'classic (look-ahead)'} entries | window {bt.start} → {bt.end}, capital ${bt.capital:,.0f}. Full list: tab \"Parameters in use\".")

    # ── dates ──
    def set_range(self, lo: datetime, hi: datetime) -> None:
        """New data arrived: 'To' follows the newest bar (and 'From' the oldest) unless you set the date yourself."""
        self._lo, self._hi = lo, hi
        self._programmatic = True
        try:
            if not self._from_pinned:
                self.d0.setDate(QDate(lo.year, lo.month, lo.day))
            if not self._to_pinned:
                self.d1.setDate(QDate(hi.year, hi.month, hi.day))
        finally:
            self._programmatic = False
        self.refresh_fresh_label()

    def quick_range(self, days: int) -> None:
        if self._hi is None:
            return
        hi = self._hi
        lo = self._lo if days == 0 else datetime(hi.year, 1, 1) if days < 0 else max(self._lo, hi - timedelta(days=days))
        self._programmatic = True
        try:
            self.d0.setDate(QDate(lo.year, lo.month, lo.day)); self.d1.setDate(QDate(hi.year, hi.month, hi.day))
        finally:
            self._programmatic = False
        self._from_pinned, self._to_pinned = days != 0, False
        self.refresh_fresh_label(); self.refresh_using()

    def follow_latest(self) -> None:
        self._to_pinned = False
        if self._hi is not None:
            self._programmatic = True
            try:
                self.d1.setDate(QDate(self._hi.year, self._hi.month, self._hi.day))
            finally:
                self._programmatic = False
        self.refresh_fresh_label()

    def refresh_fresh_label(self) -> None:
        if self._hi is None:
            self.fresh_lbl.setText("No local data yet - use 'Update data' in the left panel.")
            self.fresh_lbl.setStyleSheet(f"color: {YELLOW};")
            return
        txt, stale = D.freshness_text(self._hi, D.now_et())
        late = stale
        if self._to_pinned and self.d1.date() < QDate(self._hi.year, self._hi.month, self._hi.day):
            txt += f"  ⚠ 'To' is {self.d1.date().toString('yyyy-MM-dd')}: newer data is not included - press '→ latest'."
            late = True
        self.fresh_lbl.setText(txt)
        self.fresh_lbl.setStyleSheet(f"color: {YELLOW};" if late else f"color: {MUTED};")

    # ── data ──
    def bars_for(self, symbol: str, tf: int, refresh: bool, log: Callable = lambda m: None) -> pd.DataFrame:
        """Bars in memory, else the cache; topped up from Alpaca first when asked and the cache is behind (never more than once per 20 minutes)."""
        key = (symbol, tf)
        bars = self._bars.get(key)
        if bars is None:
            try:
                bars = D.read_cache(symbol, tf)
            except FileNotFoundError:
                bars = None
        last = bars.index[-1].to_pydatetime() if bars is not None and len(bars) else None
        if refresh and D.needs_refresh(last, D.now_et(), self._last_refresh_try, tf):
            self._last_refresh_try = time.time()
            log("  refreshing the bars from Alpaca …")
            bars = D.load_bars(symbol, tf, refresh=True, log=log)
        elif bars is None:
            bars = D.load_bars(symbol, tf, refresh=True, log=log)          # first use of this symbol: it has to be downloaded
        self._bars[key] = bars
        return bars

    def adopt_bars(self, symbol: str, tf: int, bars: pd.DataFrame) -> None:
        self._bars[(symbol, tf)] = bars
        if symbol == "SOXL" and tf == 15 and len(bars):
            self.set_range(bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime())

    # ── run ──
    def _busy(self, on: bool, msg: str = "") -> None:
        for b in (self.run_btn, self.rob_btn, self.verify_btn):
            b.setEnabled(not on)
        self.prog.setVisible(on)
        self.status.setText(msg)
        if not on:
            self.prog.setRange(0, 0)

    def run(self) -> None:
        if self._workers.busy():
            return
        try:
            bt = self.settings()
            values = self.sp.values()
            P.to_config(values)
        except ValueError as e:
            QMessageBox.warning(self, "Backtest", str(e)); return
        refresh = self.refresh_chk.isChecked()
        self._busy(True, "loading bars …")

        def work(log):
            bars = self.bars_for(bt.symbol, bt.tf, refresh, log)
            return B.run_backtest(bars, values, bt, log), bars

        w = FnWorker(work)
        self._bt_worker = self._workers.start(w, self._ran, lambda m: self.status.setText(m.strip()[:90]))

    def _ran(self, out, err) -> None:
        self._busy(False)
        if err:
            self.info.setText(f"⚠ {err}")
            self.perf.chart.message("The backtest could not run\n" + err[:160])
            QMessageBox.warning(self, "Backtest", err)
            return
        res, bars = out
        self._bars[(res.settings.symbol, res.settings.tf)] = bars
        if res.settings.symbol == "SOXL" and res.settings.tf == 15:
            self.set_range(bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime())
        self.show_result(res)

    # ── verify: does the LIVE engine trade like this backtest? ──
    def verify(self) -> None:
        if self._workers.busy():
            return
        try:
            bt = self.settings()
            values = self.sp.values()
            P.to_config(values)
        except ValueError as e:
            QMessageBox.warning(self, "Verify", str(e)); return
        if bt.symbol != "SOXL" or bt.tf != 15:
            QMessageBox.information(self, "Verify", "The live engine trades SOXL on 15-minute bars: set Symbol = SOXL and Bar = 15 to verify it."); return
        if self._hi is None:
            QMessageBox.information(self, "Verify", "No local bars yet - use 'Update data' in the left panel first."); return
        end = min(pd.Timestamp(bt.end), pd.Timestamp(self._hi.date()))
        start = end - pd.DateOffset(months=self.v_months.value())
        shortable = self.v_route.currentText() == "SOXL short"
        self._busy(True, f"replaying the live engine over {start:%Y-%m-%d} → {end:%Y-%m-%d} …")
        self.prog.setRange(0, 100)

        def work(log):
            w = self._bt_worker
            return run_replay_process(values, f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", shortable, lookahead=not bt.strict, log=lambda m: None,
                                      progress=lambda i, n: w.progress.emit(i, max(n, 1)), cancelled=lambda: w.cancelled())

        w = FnWorker(work)
        self._bt_worker = self._workers.start(w, self._verified, None, lambda i, n: self.prog.setValue(int(100 * i / max(n, 1))))

    def _verified(self, out, err) -> None:
        self._busy(False)
        if err:
            self.status.setText("verify failed")
            QMessageBox.warning(self, "Verify live = backtest", err); return
        self.last_verify = out
        st = out["stats"]
        kind = "live-accurate" if out.get("strict", True) else "classic (look-ahead)"
        head = (f"IDENTICAL - the live engine took exactly the backtest's {st['bt_trades']} trades" if out["identical"] else
                f"DIFFERENT - backtest {st['bt_trades']} trades, live engine {st['live_trades']}; {st['full_matches']} match")
        self.status.setText(head[:90])
        self.status.setStyleSheet(f"color: {GREEN if out['identical'] else RED};")
        box = QMessageBox(QMessageBox.Icon.Information if out["identical"] else QMessageBox.Icon.Warning, "Verify live = backtest",
                          f"{head}\n\nWindow {out['start']} → {out['end']} · shorts as {'a SOXL short' if out['shortable'] else 'long SOXS'} · backtest mode: {kind}", parent=self)
        box.setDetailedText(out["text"])
        box.exec()

    def show_result(self, res: B.BtResult) -> None:
        self.result = res
        m = res.metrics
        s = res.settings
        note = (f"{res.first_bar:%Y-%m-%d} → {res.last_bar:%Y-%m-%d %H:%M} ET · {res.n_bars:,} bars" + (f" (+{res.warm_bars:,} warm-up)" if res.warm_bars else "") +
                f" · {m['trades']:,} trades · simulated in {res.seconds:.1f} s · {s.symbol} {s.tf}-min")
        if res.notes:
            note += "  ⚠ " + "; ".join(res.notes)
        if s.symbol != "SOXL":
            note += "  ⚠ the live engine only trades SOXL: this is a what-if on another symbol"
        note += ("  · live-accurate entries ON: the classic backtest shows far higher returns because it lets a flat re-entry use the entry bar's own close (untick the box to see it)" if s.strict else
                 "  ⚠ CLASSIC entries (look-ahead): about 13% of these trades could not have been placed live; tick 'Live-accurate entries' for what the live engine can reproduce")
        self.info.setText(note)
        self.perf.show_result(res.eq, res.bh, res.trades, res.metrics, s.capital, "No trades in this window under these settings", bh_label=f"{s.symbol} buy & hold")
        self.perf.show_open(res.open_trade, f"Open position at the end of the window ({res.last_bar:%Y-%m-%d %H:%M})",
                            "FLAT - the backtest holds nothing after its last bar" + (" (a live account would be waiting for its next entry)" if s.symbol == "SOXL" else ""))
        self.perf.tabs.setCurrentIndex(0)
        self.refresh_fresh_label()

    # ── robustness ──
    def run_robustness(self) -> None:
        if self._workers.busy():
            return
        try:
            bt = self.settings()
            values = self.sp.values()
            P.to_config(values)
        except ValueError as e:
            QMessageBox.warning(self, "Robustness check", str(e)); return
        refresh = self.refresh_chk.isChecked()
        self._busy(True, "robustness: loading bars …")
        self.prog.setRange(0, 15)
        self.rob_verdict.setText("Running 15 backtests around the current setting … (about 20 seconds)")

        w = FnWorker(lambda log: B.robustness(self.bars_for(bt.symbol, bt.tf, refresh, log), values, bt, log, lambda i, n: w.progress.emit(i, n), w.cancelled))
        self._workers.start(w, self._robust_done, None, lambda i, n: (self.prog.setRange(0, n), self.prog.setValue(i), self.status.setText(f"robustness {i}/{n}")))

    def _robust_done(self, rb, err) -> None:
        self._busy(False)
        if err:
            self.rob_verdict.setText(f"⚠ could not run: {err}")
            return
        self.robust = rb
        col = {"ok": GREEN, "warn": YELLOW, "bad": RED}.get(rb.level, ACCENT)
        self.rob_verdict.setStyleSheet(box_style(col, 0.16, bold=True))
        self.rob_verdict.setText(rb.verdict + f"   ({rb.seconds:.0f} s)")
        t = self.rob_table
        t.horizontalHeader().setStretchLastSection(False)
        t.setColumnCount(len(rb.factors) + 1)
        t.setHorizontalHeaderLabels(["ATR period \\ factor"] + [f"{f:g}" for f in rb.factors])
        t.setRowCount(len(rb.atrs))
        ci, cj = rb.centre
        centre = rb.sharpe[ci, cj]
        for i, a in enumerate(rb.atrs):
            it = QTableWidgetItem(f"{a}")
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            t.setItem(i, 0, it)
            for j, f in enumerate(rb.factors):
                s, c, d, n = rb.sharpe[i, j], rb.cagr[i, j], rb.maxdd[i, j], rb.trades[i, j]
                cell = QTableWidgetItem("failed" if not np.isfinite(s) else f"Sharpe {s:.2f}\nCAGR {c * 100:+,.0f}%\nDD {d * 100:.0f}% · {int(n):,} tr")
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if np.isfinite(s) and np.isfinite(centre) and centre > 0:
                    r = s / centre
                    col = QColor(GREEN if r >= 0.8 else YELLOW if r >= 0.6 else RED)
                    col.setAlphaF(0.35 if r >= 0.8 else 0.28)
                    cell.setBackground(col)
                if (i, j) == (ci, cj):
                    f_ = QFont(); f_.setBold(True); cell.setFont(f_)
                    cell.setToolTip("the current setting")
                t.setItem(i, j + 1, cell)
        t.resizeRowsToContents()
        for j in range(1, t.columnCount()):
            t.setColumnWidth(j, 150)
        self.perf.tabs.setCurrentIndex(self.perf.tabs.count() - 2)

    def export(self) -> None:
        self.perf.export()

    # ── persistence ──
    def save(self) -> None:
        q = self.qs
        q.setValue("backtest/symbol", self.symbol.currentText()); q.setValue("backtest/tf", self.tf.currentText()); q.setValue("backtest/capital", self.cap.value())
        q.setValue("backtest/prime", self.prime.isChecked()); q.setValue("backtest/bench", self.bench.isChecked()); q.setValue("backtest/refresh", self.refresh_chk.isChecked()); q.setValue("backtest/strict", self.strict.isChecked())
        q.setValue("backtest/from_pinned", self._from_pinned); q.setValue("backtest/from", self.d0.date().toString("yyyy-MM-dd"))

    def load(self) -> None:
        q = self.qs
        tb = lambda k, d: str(q.value(k, d)).lower() in ("true", "1")      # noqa: E731
        if q.value("backtest/symbol"):
            self.symbol.setCurrentText(str(q.value("backtest/symbol")))
        if q.value("backtest/tf") in ("5", "15", "30", "60"):
            self.tf.setCurrentText(str(q.value("backtest/tf")))
        try:
            self.cap.setValue(float(q.value("backtest/capital", self.cap.value())))
        except (TypeError, ValueError):
            pass
        self.prime.setChecked(tb("backtest/prime", True)); self.bench.setChecked(tb("backtest/bench", True)); self.refresh_chk.setChecked(tb("backtest/refresh", True)); self.strict.setChecked(tb("backtest/strict", True))
        if tb("backtest/from_pinned", False):
            d = QDate.fromString(str(q.value("backtest/from", "")), "yyyy-MM-dd")
            if d.isValid():
                self._programmatic = True
                try:
                    self.d0.setDate(d)
                finally:
                    self._programmatic = False
                self._from_pinned = True
