"""Backtest tab: any date range, an in-sample / out-of-sample split, full metrics, yearly table, per-pair table, trades, exposure, a one-session spread viewer, and the time-frame scan, ablation and cost /
parameter-neighbourhood checks."""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QDate, QSettings, Qt
from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QTabWidget, QVBoxLayout, QWidget)

import levpairs_bridge as BR
import levpairs_data as D
from levpairs_sim import PAIR_ORDER, fmt_metrics
from supertrend_ui.theme import ACCENT, GREEN, MUTED, RED, YELLOW, Kpi, QComboBox, QDateEdit, make_table, muted, pct, set_row, spin, strip
from supertrend_ui.workers import FnWorker, WorkerBag

from .charts import EquityChart, Histogram, LineChart, SpreadChart
from .settings_panel import SettingsPanel


def _c(x: float, good: float = 0.0) -> Optional[str]:
    return GREEN if (x is not None and np.isfinite(x) and x > good) else RED if (x is not None and np.isfinite(x) and x < good) else None


class BacktestTab(QWidget):
    def __init__(self, settings: SettingsPanel, qs: QSettings, reports_dir: Callable[[], str] = lambda: "."):
        super().__init__()
        self.sp, self.qs, self.reports_dir = settings, qs, reports_dir
        self.out: Optional[BR.BtOut] = None
        self._sigs: dict = {}
        self._panels: dict = {}
        self._workers = WorkerBag()
        self._busy: Optional[FnWorker] = None
        root = QVBoxLayout(self)

        ctl = QGroupBox("Backtest-only settings")
        g = QGridLayout(ctl)
        self.d0, self.d1 = QDateEdit(QDate(2017, 1, 3)), QDateEdit(QDate.currentDate())
        for d in (self.d0, self.d1):
            d.setCalendarPopup(True); d.setDisplayFormat("yyyy-MM-dd")
        self.split = QDateEdit(QDate(2022, 1, 1)); self.split.setCalendarPopup(True); self.split.setDisplayFormat("yyyy-MM-dd")
        self.split.setToolTip("Everything on/after this date is reported as OUT-OF-SAMPLE. The research scan ranked rule sets on the earlier period only.")
        self.cap = spin(1000, 1e12, 100_000, 0, 10_000); self.cap.setPrefix("$ ")
        self.refresh_chk = QCheckBox("Refresh data first"); self.refresh_chk.setToolTip("Download newer 1-minute bars (read-only) before running.")
        self.run_btn = QPushButton("Run backtest   (Ctrl+R)"); self.run_btn.setObjectName("primary"); self.run_btn.clicked.connect(self.run)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.clicked.connect(self.stop); self.stop_btn.setEnabled(False)
        self.fill_btn = QPushButton("Fill models"); self.fill_btn.clicked.connect(self.run_fill)
        self.fill_btn.setToolTip("The same trades accounted with three fill prices: why 'fill at the close' and 'fill at the next open' differ so much.")
        self.tf_btn = QPushButton("Time frames"); self.tf_btn.clicked.connect(self.run_tf)
        self.tf_btn.setToolTip("The same rules on 1, 5, 15 and 30-minute bars: which horizon survives the costs?")
        self.abl_btn = QPushButton("Ablation"); self.abl_btn.clicked.connect(self.run_ablation)
        self.abl_btn.setToolTip("Re-runs the backtest with each ingredient switched off in turn (and with the paper's own rules, optimistic fills, zero cost) - what does each one add?")
        self.cost_btn = QPushButton("Cost & parameter checks"); self.cost_btn.clicked.connect(self.run_checks)
        self.cost_btn.setToolTip("Sharpe against the slippage assumption (where does the edge disappear?) and against the entry / exit thresholds (plateau or spike?).")
        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setFixedWidth(160); self.prog.hide()
        self.status = muted("", wrap=False)
        g.addWidget(QLabel("From"), 0, 0); g.addWidget(self.d0, 0, 1); g.addWidget(QLabel("To"), 0, 2); g.addWidget(self.d1, 0, 3)
        g.addWidget(QLabel("Out-of-sample from"), 0, 4); g.addWidget(self.split, 0, 5); g.addWidget(QLabel("Capital"), 0, 6); g.addWidget(self.cap, 0, 7)
        g.addWidget(self.refresh_chk, 0, 8)
        g.addWidget(self.run_btn, 1, 0, 1, 2); g.addWidget(self.stop_btn, 1, 2); g.addWidget(self.tf_btn, 1, 3); g.addWidget(self.fill_btn, 1, 4); g.addWidget(self.abl_btn, 1, 5)
        g.addWidget(self.cost_btn, 1, 6, 1, 2); g.addWidget(self.prog, 1, 8)
        g.addWidget(self.status, 2, 0, 1, 9)
        root.addWidget(ctl)

        self.verdict = strip("Run a backtest. Signals are cached on disk for 5-minute bars and up (first build takes a few seconds).", color=ACCENT)
        root.addWidget(self.verdict)
        names = ("CAGR", "Sharpe", "Volatility", "Max drawdown", "Out-of-sample Sharpe", "Beta to SPY", "Avg gross", "Trades / day")
        row = QHBoxLayout()
        self.k = {n: Kpi(n) for n in names}
        for w in self.k.values():
            row.addWidget(w)
        root.addLayout(row)

        self.tabs = QTabWidget()
        self.chart = EquityChart()
        self.tabs.addTab(self.chart, "Equity & drawdown")
        self.tbl_year = make_table(["Year", "Return", "Vol", "Sharpe", "Max DD", "Sessions"])
        self.tabs.addTab(self.tbl_year, "By year")
        self.tbl_met = make_table(["Metric", "All", "In-sample", "Out-of-sample"])
        self.tabs.addTab(self.tbl_met, "Metrics")
        self.tbl_pair = make_table(["Pair", "Sharpe", "Net bp / day", "Trades", "Win rate", "Net bp / trade", "Gross bp / trade", "Avg hold (bars)", "Cost bp / day"])
        self.tabs.addTab(self.tbl_pair, "Pairs")
        trp = QWidget(); trl = QVBoxLayout(trp)
        trl.addWidget(muted("Every row is one round trip. 'Long TQQQ / short QQQ' = you BUY the 3x ETF and SELL (short) the 1x ETF at the entry prices, and do the reverse at the exit prices. Amounts are per $10,000 of the 3x ETF; the 1x leg is beta "
                            "times that. Gross = the price P&L of the two legs (you can reproduce it: 3x-leg return minus beta x 1x-leg return, fill price to fill price); net = after the costs of the four fills."))
        self.tbl_tr = make_table(["Date", "Pair", "You hold", "Orders at entry (per $10k of the 3x ETF)", "Entry", "Exit", "Held (min)", "Entry price  3x / 1x", "Exit price  3x / 1x", "Gross bp", "Net bp", "Net $ per $10k",
                                  "s at entry", "Why it closed"])
        trl.addWidget(self.tbl_tr, 1)
        self.tabs.addTab(trp, "Trades")
        fp = QWidget(); fl = QVBoxLayout(fp)
        self.fill_note = muted("Press 'Fill models'.")
        self.tbl_fill = make_table(["Fill price", "Sharpe", "Out-of-sample Sharpe", "Round trips", "Gross bp / trip", "Net bp / trip"])
        fl.addWidget(self.fill_note); fl.addWidget(self.tbl_fill, 1)
        self.tabs.addTab(fp, "Fill models")
        self.fill_page = fp
        sv = QWidget(); sl = QVBoxLayout(sv)
        top = QHBoxLayout()
        self.v_pair = QComboBox(); self.v_pair.addItems(PAIR_ORDER)
        self.v_date = QDateEdit(QDate.currentDate()); self.v_date.setCalendarPopup(True); self.v_date.setDisplayFormat("yyyy-MM-dd")
        self.v_btn = QPushButton("Show session"); self.v_btn.clicked.connect(self.show_session)
        self.v_busy = QPushButton("Busiest session"); self.v_busy.clicked.connect(lambda: self.show_session(busiest=True))
        top.addWidget(QLabel("Pair")); top.addWidget(self.v_pair); top.addWidget(QLabel("Date")); top.addWidget(self.v_date); top.addWidget(self.v_btn); top.addWidget(self.v_busy); top.addStretch(1)
        self.v_chart = SpreadChart()
        sl.addLayout(top); sl.addWidget(self.v_chart, 1)
        self.v_page = sv
        self.tabs.addTab(sv, "Spread viewer")
        expo = QWidget(); ev = QVBoxLayout(expo)
        self.c_gross = LineChart("gross leverage (x equity)", 2.4); self.c_expo = LineChart("share of the session in a position", 2.0); self.c_tr = LineChart("round trips per session", 2.0)
        for w in (self.c_gross, self.c_expo, self.c_tr):
            ev.addWidget(w)
        self.tabs.addTab(expo, "Exposure & risk")
        self.hist = Histogram()
        self.tabs.addTab(self.hist, "Daily returns")
        self.tbl_tf = make_table(["Bar (min)", "Sharpe", "In-sample", "Out-of-sample", "CAGR", "OOS CAGR", "Max DD", "Round trips", "Gross bp / trip", "Net bp / trip", "Cost % equity / yr"])
        self.tabs.addTab(self.tbl_tf, "Time frames")
        self.tbl_abl = make_table(["Variant", "Sharpe", "CAGR", "Vol", "Max DD", "Beta", "OOS Sharpe", "OOS CAGR"])
        self.tabs.addTab(self.tbl_abl, "Ablation")
        chk = QWidget(); cv = QVBoxLayout(chk)
        self.tbl_cost = make_table(["Slippage per leg (bp)", "Sharpe", "CAGR", "Max DD"]); self.tbl_sens = make_table(["Open |s| >", "Close at", "Sharpe", "CAGR", "Max DD"])
        cv.addWidget(QLabel("Sharpe vs the slippage assumption")); cv.addWidget(self.tbl_cost); cv.addWidget(QLabel("Sharpe vs entry / exit thresholds (neighbourhood of the current setting)")); cv.addWidget(self.tbl_sens)
        self.tabs.addTab(chk, "Cost & parameter checks")
        root.addWidget(self.tabs, 1)
        exp = QPushButton("Export daily returns & trades to CSV"); exp.clicked.connect(self.export)
        root.addWidget(exp, 0, Qt.AlignmentFlag.AlignLeft)

    # ── persistence ──────────────────────────────────────────────────────────────
    def save(self) -> None:
        for k, w in (("bt/from", self.d0), ("bt/split", self.split)):
            self.qs.setValue(k, w.date().toString("yyyy-MM-dd"))
        self.qs.setValue("bt/capital", self.cap.value())

    def load(self) -> None:
        for k, w in (("bt/from", self.d0), ("bt/split", self.split)):
            v = self.qs.value(k)
            if v:
                d = QDate.fromString(str(v), "yyyy-MM-dd")
                if d.isValid():
                    w.setDate(d)
        try:
            self.cap.setValue(float(self.qs.value("bt/capital", 100_000)))
        except (TypeError, ValueError):
            pass

    # ── running ──────────────────────────────────────────────────────────────────
    def _busy_on(self, on: bool, text: str = "") -> None:
        for b in (self.run_btn, self.tf_btn, self.fill_btn, self.abl_btn, self.cost_btn):
            b.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        self.prog.setVisible(on)
        self.status.setText(text)

    def stop(self) -> None:
        if self._busy:
            self._busy.cancel()
            self.status.setText("stopping ...")

    def _panel(self, tf: int, feed: str, refresh: bool, log):
        key = (tf, feed)
        if refresh:
            D.update_cache(feeds=(feed,), log=log)
            self._panels.clear()
        if key not in self._panels:
            self._panels[key] = BR.load_panel(tf, feed, log=log)
        return self._panels[key]

    def _job(self, kind: str):
        sc = self.sp.sig_cfg()
        c = self.sp.sim_cfg(self.d0.date().toString("yyyy-MM-dd"), self.d1.date().toString("yyyy-MM-dd"), self.cap.value())
        split, feed, refresh = self.split.date().toString("yyyy-MM-dd"), self.sp.feed(), self.refresh_chk.isChecked()
        holder: dict = {}

        def fn(log):
            w = holder["w"]
            if kind == "tf":
                return ("tf", BR.timeframe_scan(sc, c, feed, split, log=log, cancelled=w.cancelled), None, None)
            if kind == "fill":
                panel = self._panel(sc.tf, feed, refresh, log)
                return ("fill", BR.fill_comparison(panel, BR.get_signals(panel, sc, feed, log=log, cancelled=w.cancelled), c, split), None, None)
            panel = self._panel(sc.tf, feed, refresh, log)
            sigs = BR.get_signals(panel, sc, feed, log=log, cancelled=w.cancelled)
            if w.cancelled():
                return None
            if kind == "bt":
                return ("bt", BR.run_backtest(panel, sigs, c, split, progress=lambda a, b: w.progress.emit(a, b), cancelled=w.cancelled), panel, sigs)
            if kind == "abl":
                return ("abl", BR.ablation(panel, sc, c, feed, split, log=log, cancelled=w.cancelled), None, None)
            return ("chk", (BR.cost_breakeven(panel, sigs, c), BR.sensitivity(panel, sigs, c, log=log, cancelled=w.cancelled)), None, None)
        w = FnWorker(fn)
        holder["w"] = w
        return w

    def _start(self, kind: str, text: str) -> None:
        if self._busy and self._busy.isRunning():
            return
        try:
            w = self._job(kind)
        except Exception as e:                                                         # noqa: BLE001
            self.verdict.setText(f"Cannot start: {e}")
            return
        self._busy = w
        self._busy_on(True, text)
        self.prog.setValue(0)
        self._workers.start(w, self._done, lambda s: self.status.setText(s.strip()[:90]), lambda a, b: self.prog.setValue(int(100 * a / max(b, 1))))

    def run(self) -> None:
        self._start("bt", "loading bars / building signals / simulating ...")

    def run_tf(self) -> None:
        self._start("tf", "time-frame scan ...")

    def run_fill(self) -> None:
        self._start("fill", "fill models ...")

    def run_ablation(self) -> None:
        self._start("abl", "ablation ...")

    def run_checks(self) -> None:
        self._start("chk", "cost & parameter checks ...")

    def _done(self, res, err) -> None:
        self._busy_on(False)
        self._busy = None
        if err:
            self.verdict.setText(f"Failed: {err}")
            return
        if res is None:
            self.status.setText("stopped")
            return
        kind, payload, panel, sigs = res
        if kind == "bt":
            self._sigs = sigs
            self.show_result(payload, panel)
        elif kind == "tf":
            self.tabs.setCurrentWidget(self.tbl_tf)
            t = self.tbl_tf
            t.setRowCount(len(payload))
            for r, (tf, x) in enumerate(payload.iterrows()):
                set_row(t, r, [int(tf), f"{x.sharpe:.2f}", f"{x.sharpe_is:.2f}", f"{x.sharpe_oos:.2f}", pct(x.cagr), pct(x.cagr_oos), pct(x.maxdd), f"{int(x.trades):,}", f"{x.gross_bp:+.2f}", f"{x.net_bp:+.2f}",
                               pct(x.cost_pct)], {1: _c(x.sharpe), 2: _c(x.sharpe_is), 3: _c(x.sharpe_oos), 8: _c(x.gross_bp), 9: _c(x.net_bp)})
            self.status.setText("time-frame scan done")
        elif kind == "fill":
            tab, gaps = payload
            self.tabs.setCurrentWidget(self.fill_page)
            t = self.tbl_fill
            t.setRowCount(len(tab))
            for r, (k, x) in enumerate(tab.iterrows()):
                set_row(t, r, [x.label, f"{x.sharpe:.2f}", f"{x.sharpe_oos:.2f}", f"{int(x.trades):,}", f"{x.gross_bp:+.2f}", f"{x.net_bp:+.2f}"], {1: _c(x.sharpe), 2: _c(x.sharpe_oos), 4: _c(x.gross_bp), 5: _c(x.net_bp)}, align_left=(0,))
            if gaps:
                self.fill_note.setText(f"The decisions and trades are identical in all three rows ({gaps['trades']:,} round trips); only the price the orders get differs. Between the close of the bar that gave the signal and the open of the next bar "
                                       f"the spread moves {gaps['gap_in']:+.2f} bp in the trade's favour at ENTRY (a fill at the old close would collect that, but no order can: the close is a stale last trade) and {gaps['gap_out']:+.2f} bp "
                                       f"at EXIT (a close fill would give that up). That is why the rows differ so much.")
            self.status.setText("fill models done")
        elif kind == "abl":
            self.tabs.setCurrentWidget(self.tbl_abl)
            t = self.tbl_abl
            t.setRowCount(len(payload))
            for r, (name, x) in enumerate(payload.iterrows()):
                set_row(t, r, [name, f"{x.sharpe:.2f}", pct(x.cagr), pct(x.vol), pct(x.maxdd), f"{x.beta:+.2f}" if pd.notna(x.beta) else "—", f"{x.sharpe_oos:.2f}", pct(x.cagr_oos)],
                        {1: _c(x.sharpe), 2: _c(x.cagr), 6: _c(x.sharpe_oos), 7: _c(x.cagr_oos)}, align_left=(0,))
            self.status.setText("ablation done")
        else:
            cost, sens = payload
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            self.tbl_cost.setRowCount(len(cost))
            for r, x in cost.reset_index(drop=True).iterrows():
                set_row(self.tbl_cost, r, [f"{x.slip_bp:g}", f"{x.sharpe:.2f}", pct(x.cagr), pct(x.maxdd)], {1: _c(x.sharpe), 2: _c(x.cagr)})
            self.tbl_sens.setRowCount(len(sens))
            for r, x in sens.reset_index(drop=True).iterrows():
                set_row(self.tbl_sens, r, [f"{x.s_open:.2f}", f"{x.s_close:.2f}", f"{x.sharpe:.2f}", pct(x.cagr), pct(x.maxdd)], {2: _c(x.sharpe), 3: _c(x.cagr)})
            self.status.setText("checks done")

    # ── showing ──────────────────────────────────────────────────────────────────
    def show_result(self, o: BR.BtOut, panel: BR.Panel) -> None:
        self.out = o
        a, i, s = o.m_all, o.m_is, o.m_oos
        d = o.result.daily
        if a.get("n", 0) < 20:
            self.verdict.setText("Not enough sessions in that window.")
            return
        self.k["CAGR"].set(pct(a["cagr"]), _c(a["cagr"]), f"total {pct(a['total'], 0)}")
        self.k["Sharpe"].set(f"{a['sharpe']:.2f}", _c(a["sharpe"]), f"± {1.96 * a['sharpe_se']:.2f} (95%) · PSR {a['psr0']:.0%}")
        self.k["Volatility"].set(pct(a["vol"]))
        self.k["Max drawdown"].set(pct(a["maxdd"]), RED if a["maxdd"] < -0.15 else None, f"worst 21d {pct(a['worst_21d'])}")
        self.k["Out-of-sample Sharpe"].set(f"{s['sharpe']:.2f}" if s.get("n", 0) >= 20 else "—", _c(s.get("sharpe", np.nan)), f"in-sample {i['sharpe']:.2f}" if i.get("n", 0) >= 20 else "")
        self.k["Beta to SPY"].set(f"{a.get('beta', float('nan')):+.2f}", None, f"corr {a.get('corr', float('nan')):+.2f}")
        self.k["Avg gross"].set(f"{d.G.mean():.2f}x", None, f"in a position {d.exposure.mean():.0%} of the session")
        self.k["Trades / day"].set(f"{d.n_trades.mean():.1f}", None, f"cost {d.cost.mean() * 252:.1%} of equity / yr")
        msg = f"All: {fmt_metrics(a)}"
        if i.get("n", 0) >= 20 and s.get("n", 0) >= 20:
            msg += f"\nIn-sample (<{o.split:%Y-%m-%d}): Sharpe {i['sharpe']:.2f}, CAGR {pct(i['cagr'])} · Out-of-sample: Sharpe {s['sharpe']:.2f}, CAGR {pct(s['cagr'])}, max DD {pct(s['maxdd'])}"
        msg += "\n" + " · ".join(o.notes)
        if self.out.result.cfg.fill == "close":
            msg += "\n⚠ Fill model 'close': orders fill at the signal bar's last trade - the bid-ask bounce is counted as profit. Not achievable."
        good = a["sharpe"] > 0.5 and s.get("sharpe", -1) > 0.3
        from supertrend_ui.theme import box_style
        self.verdict.setStyleSheet(box_style(GREEN if good else YELLOW if a["sharpe"] > 0 else RED, 0.12))
        self.verdict.setText(msg)
        self.chart.show_curves(o.equity, o.bench.reindex(o.equity.index), o.split)
        ty = self.tbl_year
        ty.setRowCount(len(o.yearly))
        for r, (y, x) in enumerate(o.yearly.iterrows()):
            set_row(ty, r, [y, pct(x["return"]), pct(x.vol), f"{x.sharpe:.2f}", pct(x.maxdd), int(x.days)], {1: _c(x["return"]), 3: _c(x.sharpe)})
        rows = [("CAGR", "cagr", pct), ("Mean return (ann.)", "mean_ann", pct), ("Volatility", "vol", pct), ("Sharpe", "sharpe", lambda v: f"{v:.2f}"), ("Sharpe 95% half-width", "sharpe_se", lambda v: f"{1.96 * v:.2f}"),
                ("Sortino", "sortino", lambda v: f"{v:.2f}"), ("Max drawdown", "maxdd", pct), ("Calmar", "calmar", lambda v: f"{v:.2f}"), ("Hit rate (days)", "hit", pct),
                ("Skew", "skew", lambda v: f"{v:.2f}"), ("Kurtosis", "kurt", lambda v: f"{v:.1f}"), ("Best day", "best_day", pct), ("Worst day", "worst_day", pct), ("Worst 21 days", "worst_21d", pct),
                ("Prob. Sharpe > 0 (PSR)", "psr0", pct), ("Beta to SPY", "beta", lambda v: f"{v:+.2f}"), ("Alpha vs SPY (ann.)", "alpha_ann", pct), ("SPY Sharpe", "bench_sharpe", lambda v: f"{v:.2f}"),
                ("SPY CAGR", "bench_cagr", pct), ("SPY max DD", "bench_maxdd", pct), ("Sessions", "n", lambda v: f"{int(v):,}")]
        tm = self.tbl_met
        tm.setRowCount(len(rows))
        for r, (lab, k, f) in enumerate(rows):
            vals = [f(m[k]) if (k in m and m[k] is not None and np.isfinite(m[k])) else "—" for m in (a, i, s)]
            set_row(tm, r, [lab] + vals)
        tp = self.tbl_pair
        tp.setRowCount(len(o.pairs))
        for r, (k, x) in enumerate(o.pairs.iterrows()):
            set_row(tp, r, [k, f"{x.sharpe:.2f}", f"{x.net_bp_day:+.2f}", f"{int(x.trades):,}", pct(x.win, 0), f"{x.avg_net_bp:+.2f}", f"{x.avg_gross_bp:+.2f}", f"{x.avg_bars:.1f}", f"{x.cost_bp_day:.2f}"],
                    {1: _c(x.sharpe), 2: _c(x.net_bp_day), 5: _c(x.avg_net_bp), 6: _c(x.avg_gross_bp)})
        tr = o.result.trades
        if len(tr):
            t = tr.sort_values("entry_time", ascending=False).head(2000)
            self.tbl_tr.setRowCount(len(t))
            for r, x in enumerate(t.itertuples()):
                L_, B_ = x.pair.split("/")
                orders = f"BUY {L_} $10,000  +  SELL {B_} ${10000 * x.beta:,.0f}" if x.side == "long" else f"SELL {L_} $10,000  +  BUY {B_} ${10000 * x.beta:,.0f}"
                set_row(self.tbl_tr, r, [f"{x.date:%Y-%m-%d}", x.pair, f"LONG {x.long_leg} / SHORT {x.short_leg}", orders,
                                         f"{x.entry_time:%H:%M}", f"{x.exit_time:%H:%M}", int(x.hold_min), f"{x.L_entry:,.2f} / {x.B_entry:,.2f}", f"{x.L_exit:,.2f} / {x.B_exit:,.2f}", f"{x.gross_bp:+.1f}", f"{x.net_bp:+.1f}",
                                         f"{x.net_bp:+.2f}", f"{x.s_entry:+.2f}" if np.isfinite(x.s_entry) else "—", x.reason],
                        {2: GREEN if x.side == "long" else RED, 9: _c(x.gross_bp), 10: _c(x.net_bp), 11: _c(x.net_bp)}, align_left=(1, 2, 3, 13))
        self.c_gross.show_series({"gross leverage G": d.G}, [1.0])
        self.c_expo.show_series({"time in position": d.exposure.rolling(21).mean()})
        self.c_tr.show_series({"round trips / session (21d avg)": d.n_trades.rolling(21).mean()})
        self.hist.show_hist(o.ret.values * 100, 80, "daily net return, %")
        if len(panel):
            last = next(iter(panel.values())).dates[-1]
            self.v_date.setDate(QDate(last.year, last.month, last.day))
        self.status.setText(f"done · {len(o.ret):,} sessions")

    def show_session(self, busiest: bool = False) -> None:
        if self.out is None or not self._sigs:
            self.status.setText("run a backtest first")
            return
        name = self.v_pair.currentText()
        sig, pr = self._sigs.get(name), self.out.result.per_pair.get(name)
        if sig is None or pr is None:
            self.status.setText(f"{name} is not part of this run")
            return
        if busiest:
            ix = int(np.argmax((np.abs(np.diff(pr.hold.astype(int), axis=1)) > 0).sum(axis=1)))
        else:
            qd = self.v_date.date()
            ix = int(np.clip(sig.dates.searchsorted(pd.Timestamp(qd.year(), qd.month(), qd.day())), 0, len(sig.dates) - 1))
        tf = sig.cfg.tf
        t = np.array([(D.OPEN_MIN + k * tf) / 60.0 for k in range(sig.nb)])
        c = self.out.result.cfg
        self.v_chart.show_session(t, sig.X[ix] * 1e4, sig.s[ix], pr.hold[ix], c.s_open, 0.5 * (c.s_close_long + c.s_close_short), f"{name} · {sig.dates[ix]:%Y-%m-%d} · beta {sig.beta[ix]:.2f} · {tf}-minute bars (x axis: hour of day)")
        self.tabs.setCurrentWidget(self.v_page)

    def export(self) -> None:
        if not self.out:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export", os.path.join(self.reports_dir(), "levpairs_backtest_daily.csv"), "CSV (*.csv)")
        if path:
            self.out.result.daily.to_csv(path)
            self.out.result.trades.to_csv(path.replace(".csv", "_trades.csv"), index=False)
            self.status.setText(f"saved {path}")
