"""Backtest tab: run the strategy with the settings on the left over the cached history (same decide() as live), show KPIs, equity / drawdown, cost sensitivity, years, names, exposure."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget

from .. import backtest as BT
from .. import strategy as S
from .charts import EquityChart, LineChart
from .theme import GREEN, RED, YELLOW, Kpi, make_table, muted, pct, set_row, strip
from .workers import FnWorker, WorkerBag


class BacktestTab(QWidget):
    def __init__(self, settings, qs: QSettings, folder: str):
        super().__init__()
        self.sp, self.qs, self.folder = settings, qs, folder
        self._workers = WorkerBag()
        self._w = None
        self.result = None
        root = QVBoxLayout(self)
        root.addWidget(strip("The backtest trades at the OPEN of the bar after the data it uses (one bar later with the delayed SIP feed), pays the per-side costs on every traded dollar "
                             "of stock and ETF hedge plus a borrow fee, and uses the same portfolio function as the live engine. In-sample before the split date, out-of-sample after: "
                             "pick settings on in-sample only.", YELLOW))
        top = QHBoxLayout()
        self.btn = QPushButton("Run backtest"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.run)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.clicked.connect(self.stop); self.btn_stop.setEnabled(False)
        self.status = muted("")
        top.addWidget(self.btn); top.addWidget(self.btn_stop); top.addWidget(self.status, 1)
        root.addLayout(top)
        names = ("Sharpe (net)", "In-sample", "Out-of-sample", "CAGR", "Volatility", "Max drawdown", "Turnover / yr", "Holding (days)", "Before costs", "Beta to SPY")
        k = QHBoxLayout()
        self.k = {n: Kpi(n) for n in names}
        for w in self.k.values():
            k.addWidget(w)
        root.addLayout(k)
        self.tabs = QTabWidget()
        self.c_eq = EquityChart()
        self.tabs.addTab(self.c_eq, "Equity")
        self.t_cost = make_table(["Cost x (of the model)", "Avg cost per side (bp)", "Sharpe", "In-sample Sharpe", "CAGR", "Max DD"])
        self.tabs.addTab(self.t_cost, "Cost sensitivity")
        self.t_year = make_table(["Year", "Return", "Sample"])
        self.tabs.addTab(self.t_year, "Years")
        self.t_names = make_table(["Symbol", "P&L contribution", "Trades"])
        self.tabs.addTab(self.t_names, "Names")
        ex = QWidget(); ev = QVBoxLayout(ex)
        self.c_gross = LineChart("x capital", 2.4); self.c_names = LineChart("names", 2.4)
        ev.addWidget(self.c_gross); ev.addWidget(self.c_names)
        self.tabs.addTab(ex, "Exposure")
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True)
        self.tabs.addTab(self.log_box, "Log")
        root.addWidget(self.tabs, 1)

    def log(self, m: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now():%H:%M:%S}  {m.strip()}")
        self.status.setText(m.strip()[:140])

    def run(self) -> None:
        if self._workers.busy():
            return
        strat = self.sp.strategy(self.folder)
        self.btn.setEnabled(False); self.btn_stop.setEnabled(True)
        self.tabs.setCurrentWidget(self.log_box)

        def fn(log):
            w = self._w
            bt, sc, ctx = S.backtest(strat, log=log, stop=lambda: w is not None and w.cancelled())
            curve = BT.cost_curve(bt, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
            return bt, curve, strat
        self._w = FnWorker(fn)
        self._workers.start(self._w, self._done, self.log)

    def stop(self) -> None:
        if self._w is not None:
            self._w.cancel()

    def _done(self, res, err) -> None:
        self.btn.setEnabled(True); self.btn_stop.setEnabled(False)
        if err:
            self.log("backtest failed: " + err)
            return
        bt, curve, strat = res
        self.result = bt
        st = bt.stats
        a, i, o, g = st.get("all", {}), st.get("is", {}), st.get("oos", {}), st.get("gross_all", {})
        col = lambda x: (GREEN if x > 0 else RED) if x is not None and np.isfinite(x) else None
        f2 = lambda b, k: f"{b[k]:.2f}" if k in b and np.isfinite(b[k]) else "—"
        self.k["Sharpe (net)"].set(f2(a, "sharpe"), col(a.get("sharpe")), f"t {a.get('t_stat', float('nan')):.1f} · PSR {st.get('psr', float('nan')):.2f}")
        self.k["In-sample"].set(f2(i, "sharpe"), col(i.get("sharpe")), f"to {strat.v['split']}")
        self.k["Out-of-sample"].set(f2(o, "sharpe"), col(o.get("sharpe")), f"from {strat.v['split']}")
        self.k["CAGR"].set(pct(a.get("cagr")), col(a.get("cagr")))
        self.k["Volatility"].set(pct(a.get("vol")))
        self.k["Max drawdown"].set(pct(a.get("max_dd")), RED)
        self.k["Turnover / yr"].set(f"{st.get('turnover_ann', float('nan')):.0f}x", None, f"costs {pct(st.get('cost_ann'))}/yr")
        self.k["Holding (days)"].set(f"{st.get('hold_days', float('nan')):.1f}", None, f"{st.get('avg_names', 0):.0f} names")
        self.k["Before costs"].set(f2(g, "sharpe"), col(g.get("sharpe")), "Sharpe at zero cost")
        self.k["Beta to SPY"].set(f"{st.get('beta_spy', float('nan')):+.3f}", None, f"corr {st.get('corr_spy', float('nan')):+.2f} · overnight gross "
                                                                                      f"{st.get('overnight_gross_all', float('nan')):.2f}x (p95 {st.get('overnight_gross_p95', float('nan')):.2f}x)")
        self.c_eq.show_curves(bt.daily, pd.Timestamp(strat.v["split"]))
        t = self.t_cost
        t.setRowCount(len(curve))
        f2_ = lambda v: f"{v:.2f}" if v is not None and np.isfinite(v) else "—"
        for r, x in enumerate(curve.itertuples()):
            set_row(t, r, [f"{x.cost_x:.1f}x", f"{x.avg_bp_side:.2f}", f2_(x.sharpe), f2_(x.sharpe_is), pct(x.cagr), pct(x.max_dd)], {2: col(x.sharpe), 3: col(x.sharpe_is)})
        by = st.get("by_year", {})
        t = self.t_year
        t.setRowCount(len(by))
        sp = pd.Timestamp(strat.v["split"]).year
        for r, (y, v) in enumerate(sorted(by.items())):
            set_row(t, r, [str(y), pct(v, 1), "out-of-sample" if y >= sp else "in-sample"], {1: col(v)})
        npnl = bt.name_pnl[bt.name_trades > 0].sort_values()
        show = pd.concat([npnl.tail(25)[::-1], npnl.head(25)])
        t = self.t_names
        t.setRowCount(len(show))
        for r, (s, v) in enumerate(show.items()):
            set_row(t, r, [s, pct(v, 2), f"{int(bt.name_trades[s]):,}"], {1: col(v)})
        d = bt.daily
        self.c_gross.show_series({"stock gross": d.gross, "net (incl. hedges)": d.net}, hlines=[0])
        self.c_names.show_series({"longs": d.longs, "shorts": d.shorts})
        self.tabs.setCurrentWidget(self.c_eq)
        self.log(BT.summary_line(st))
