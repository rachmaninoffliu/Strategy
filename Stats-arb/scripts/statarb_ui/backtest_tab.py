"""Backtest tab: any date range, an in-sample / out-of-sample split, full metrics, yearly table, trades, exposures, and the ablation / cost / parameter-neighbourhood checks."""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QDate, QSettings, Qt
from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QTabWidget, QVBoxLayout, QWidget)

import statarb_bridge as BR
import statarb_data as D
from statarb_sim import fmt_metrics
from supertrend_ui.theme import ACCENT, GREEN, MUTED, RED, YELLOW, Kpi, QComboBox, QDateEdit, make_table, muted, pct, set_row, spin, strip
from supertrend_ui.workers import FnWorker, WorkerBag

from .charts import EquityChart, Histogram, LineChart
from .settings_panel import SettingsPanel


def _c(x: float, good: float = 0.0) -> Optional[str]:
    return GREEN if (x is not None and np.isfinite(x) and x > good) else RED if (x is not None and np.isfinite(x) and x < good) else None


class BacktestTab(QWidget):
    def __init__(self, settings: SettingsPanel, qs: QSettings, reports_dir: Callable[[], str] = lambda: "."):
        super().__init__()
        self.sp, self.qs, self.reports_dir = settings, qs, reports_dir
        self.out: Optional[BR.BtOut] = None
        self._panels: dict = {}
        self._workers = WorkerBag()
        self._busy: Optional[FnWorker] = None
        root = QVBoxLayout(self)

        ctl = QGroupBox("Backtest-only settings")
        g = QGridLayout(ctl)
        self.d0, self.d1 = QDateEdit(QDate(2017, 6, 1)), QDateEdit(QDate.currentDate())
        for d in (self.d0, self.d1):
            d.setCalendarPopup(True); d.setDisplayFormat("yyyy-MM-dd")
        self.split = QDateEdit(QDate(2022, 1, 1)); self.split.setCalendarPopup(True); self.split.setDisplayFormat("yyyy-MM-dd")
        self.split.setToolTip("Everything on/after this date is reported as OUT-OF-SAMPLE. The default settings were chosen on the earlier period only.")
        self.cap = spin(1000, 1e12, 100_000, 0, 10_000); self.cap.setPrefix("$ ")
        self.refresh_chk = QCheckBox("Refresh data first"); self.refresh_chk.setToolTip("Download newer daily bars (read-only) before running.")
        self.run_btn = QPushButton("Run backtest   (Ctrl+R)"); self.run_btn.setObjectName("primary"); self.run_btn.clicked.connect(self.run)
        self.stop_btn = QPushButton("Stop"); self.stop_btn.clicked.connect(self.stop); self.stop_btn.setEnabled(False)
        self.abl_btn = QPushButton("Ablation"); self.abl_btn.clicked.connect(self.run_ablation)
        self.abl_btn.setToolTip("Re-runs the backtest with each ingredient switched off in turn (and with the paper's own rules) - what does each one add?")
        self.cost_btn = QPushButton("Cost & parameter checks"); self.cost_btn.clicked.connect(self.run_checks)
        self.cost_btn.setToolTip("Sharpe against the cost assumption (where does the edge disappear?) and against the entry / exit thresholds (is it a plateau or a spike?).")
        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setFixedWidth(160); self.prog.hide()
        self.status = muted("", wrap=False)
        g.addWidget(QLabel("From"), 0, 0); g.addWidget(self.d0, 0, 1); g.addWidget(QLabel("To"), 0, 2); g.addWidget(self.d1, 0, 3)
        g.addWidget(QLabel("Out-of-sample from"), 0, 4); g.addWidget(self.split, 0, 5); g.addWidget(QLabel("Capital"), 0, 6); g.addWidget(self.cap, 0, 7)
        g.addWidget(self.refresh_chk, 0, 8)
        g.addWidget(self.run_btn, 1, 0, 1, 2); g.addWidget(self.stop_btn, 1, 2); g.addWidget(self.abl_btn, 1, 3); g.addWidget(self.cost_btn, 1, 4, 1, 2); g.addWidget(self.prog, 1, 6, 1, 2)
        g.addWidget(self.status, 1, 8)
        root.addWidget(ctl)

        self.verdict = strip("Run a backtest. Signals are cached on disk (building them takes about a minute the first time, seconds afterwards).", color=ACCENT)
        root.addWidget(self.verdict)
        names = ("CAGR", "Sharpe", "Volatility", "Max drawdown", "Out-of-sample Sharpe", "Beta to SPY", "Avg gross", "Turnover / day")
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
        self.tbl_tr = make_table(["Symbol", "Sleeve", "Side", "Entry", "Exit", "Days", "Return", "s at entry"])
        self.tabs.addTab(self.tbl_tr, "Trades")
        expo = QWidget(); ev = QVBoxLayout(expo)
        self.c_gross = LineChart("gross / net / beta (x equity)", 2.4); self.c_pos = LineChart("positions", 2.0); self.c_mult = LineChart("shock multiplier", 2.0)
        for w in (self.c_gross, self.c_pos, self.c_mult):
            ev.addWidget(w)
        self.tabs.addTab(expo, "Exposure & risk")
        self.hist = Histogram()
        self.tabs.addTab(self.hist, "Daily returns")
        self.tbl_abl = make_table(["Variant", "Sharpe", "CAGR", "Vol", "Max DD", "Beta", "OOS Sharpe", "OOS CAGR"])
        self.tabs.addTab(self.tbl_abl, "Ablation")
        chk = QWidget(); cv = QVBoxLayout(chk)
        self.tbl_cost = make_table(["One-way cost (bp)", "Sharpe", "CAGR", "Max DD"]); self.tbl_sens = make_table(["Open |s| >", "Close at", "Sharpe", "CAGR", "Max DD"])
        cv.addWidget(QLabel("Sharpe vs the cost assumption")); cv.addWidget(self.tbl_cost); cv.addWidget(QLabel("Sharpe vs entry / exit thresholds (neighbourhood of the current setting)")); cv.addWidget(self.tbl_sens)
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
        for b in (self.run_btn, self.abl_btn, self.cost_btn):
            b.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        self.prog.setVisible(on)
        self.status.setText(text)

    def stop(self) -> None:
        if self._busy:
            self._busy.cancel()
            self.status.setText("stopping ...")

    def _prepare(self, log, uni: str, refresh: bool):
        if uni not in self._panels or refresh:
            self._panels[uni] = BR.load_panel(uni, refresh=refresh, log=log)
        return uni, self._panels[uni]

    def _job(self, kind: str):
        f, p = self.sp.factor_cfg(), self.sp.port_cfg(self.d0.date().toString("yyyy-MM-dd"), self.d1.date().toString("yyyy-MM-dd"), self.cap.value())
        split = self.split.date().toString("yyyy-MM-dd")
        uni, refresh = self.sp.universe(), self.refresh_chk.isChecked()
        w_holder: dict = {}

        def fn(log):
            w = w_holder["w"]
            uni_, panel = self._prepare(log, uni, refresh)
            sig = BR.get_signals(panel, uni_, f, log=log, progress=lambda a, b: w.progress.emit(a, b), cancelled=w.cancelled)
            if w.cancelled():
                return None
            if kind == "bt":
                return ("bt", BR.run_backtest(panel, sig, p, split, progress=lambda a, b: w.progress.emit(a, b), cancelled=w.cancelled), panel)
            if kind == "abl":
                return ("abl", BR.ablation(panel, sig, p, split, log=log, cancelled=w.cancelled), None)
            return ("chk", (BR.cost_breakeven(panel, sig, p), BR.sensitivity(panel, sig, p, log=log, cancelled=w.cancelled)), None)
        w = FnWorker(fn)
        w_holder["w"] = w
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
        self._start("bt", "building signals / simulating ...")

    def run_ablation(self) -> None:
        self._start("abl", "ablation ...")

    def run_checks(self) -> None:
        self._start("chk", "cost & parameter checks ...")

    def _done(self, res, err) -> None:
        self._busy_on(False)
        self._busy = None
        if err:
            self.verdict.setText(f"Failed: {err}")
            self.verdict.setStyleSheet(self.verdict.styleSheet())
            return
        if res is None:
            self.status.setText("stopped")
            return
        kind, payload, panel = res
        if kind == "bt":
            self.show_result(payload, panel)
        elif kind == "abl":
            self.tabs.setCurrentWidget(self.tbl_abl)
            t = self.tbl_abl
            t.setRowCount(len(payload))
            for r, (name, x) in enumerate(payload.iterrows()):
                set_row(t, r, [name, f"{x.sharpe:.2f}", pct(x.cagr), pct(x.vol), pct(x.maxdd), f"{x.beta:+.2f}" if pd.notna(x.beta) else "—", f"{x.sharpe_oos:.2f}", pct(x.cagr_oos)],
                        {1: _c(x.sharpe), 2: _c(x.cagr), 6: _c(x.sharpe_oos), 7: _c(x.cagr_oos)})
            self.status.setText("ablation done")
        else:
            cost, sens = payload
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            self.tbl_cost.setRowCount(len(cost))
            for r, x in cost.reset_index(drop=True).iterrows():
                set_row(self.tbl_cost, r, [f"{x.tc_bps:g}", f"{x.sharpe:.2f}", pct(x.cagr), pct(x.maxdd)], {1: _c(x.sharpe), 2: _c(x.cagr)})
            self.tbl_sens.setRowCount(len(sens))
            for r, x in sens.reset_index(drop=True).iterrows():
                set_row(self.tbl_sens, r, [f"{x.s_open:.2f}", f"{x.s_close:.2f}", f"{x.sharpe:.2f}", pct(x.cagr), pct(x.maxdd)], {2: _c(x.sharpe), 3: _c(x.cagr)})
            self.status.setText("checks done")

    # ── showing ──────────────────────────────────────────────────────────────────
    def show_result(self, o: BR.BtOut, panel: D.Panel) -> None:
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
        self.k["Avg gross"].set(f"{d.gross.mean():.2f}x", None, f"{d.n_long.mean():.0f} long / {d.n_short.mean():.0f} short")
        self.k["Turnover / day"].set(f"{d.turnover.mean():.2f}x", None, f"cost {d.cost.mean() * 252:.1%} of equity / yr")
        msg = f"All: {fmt_metrics(a)}"
        if i.get("n", 0) >= 20 and s.get("n", 0) >= 20:
            msg += f"\nIn-sample (<{o.split:%Y-%m-%d}): Sharpe {i['sharpe']:.2f}, CAGR {pct(i['cagr'])} · Out-of-sample: Sharpe {s['sharpe']:.2f}, CAGR {pct(s['cagr'])}, max DD {pct(s['maxdd'])}"
        msg += "\n" + " · ".join(o.notes)
        good = a["sharpe"] > 0.5 and s.get("sharpe", -1) > 0.3
        self.verdict.setStyleSheet("")
        self.verdict.setText(msg)
        from supertrend_ui.theme import box_style
        self.verdict.setStyleSheet(box_style(GREEN if good else YELLOW if a["sharpe"] > 0 else RED, 0.12))
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
        tr = o.result.trades
        if len(tr):
            t = tr.sort_values("exit", ascending=False).head(2000)
            self.tbl_tr.setRowCount(len(t))
            for r, x in enumerate(t.itertuples()):
                set_row(self.tbl_tr, r, [x.symbol, x.sleeve, x.side, f"{x.entry:%Y-%m-%d}", f"{x.exit:%Y-%m-%d}", x.days, pct(x.ret), f"{x.s_entry:+.2f}" if np.isfinite(x.s_entry) else "—"],
                        {6: _c(x.ret), 2: GREEN if x.side == "long" else RED})
        self.c_gross.show_series({"gross": d.gross, "net (long+short)": d.long + d.short, "beta": d.beta}, [0.0])
        self.c_pos.show_series({"long": d.n_long.astype(float), "short": d.n_short.astype(float)})
        self.c_mult.show_series({"shock multiplier": d.shock_mult, "gross target": d.g_target}, [1.0])
        self.hist.show_hist(o.ret.values * 100, 80, "daily net return, %")
        self.status.setText(f"done · {len(o.ret):,} sessions")

    def export(self) -> None:
        if not self.out:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export", os.path.join(self.reports_dir(), "statarb_backtest_daily.csv"), "CSV (*.csv)")
        if path:
            self.out.result.daily.to_csv(path)
            self.out.result.trades.to_csv(path.replace(".csv", "_trades.csv"), index=False)
            self.status.setText(f"saved {path}")
