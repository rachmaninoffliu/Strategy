"""Backtest tab: runs the picked book (the same decision functions as live) and shows every trade of both legs - leg, ticker, shares, entry / exit time and auction or market
order, prices, value, ranking score and rank, exit reason, P&L - plus the equity vs SPY, years, legs, tickers, exposure and the ML models used. Also trains the ML models."""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget

from ...ui.charts import LineChart
from ...ui.theme import ACCENT, GREEN, RED, YELLOW, Kpi, QComboBox, make_table, muted, pct, set_row, strip
from ...ui.workers import FnWorker, WorkerBag
from .. import engine as E
from .. import ml
from ...mr.ui.charts import CompareChart
from ...mr.ui.frame_model import FrameView, f_date, f_int, f_money, f_num, f_pct, f_px, f_str
from .model_panel import describe


def f_time(v) -> str:
    return "" if v is None or v is pd.NaT or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) else f"{pd.Timestamp(v):%H:%M}"


TRADE_COLS = [("#", "trade_id", lambda v: f"{int(v) + 1}"), ("Leg", "leg", f_str), ("Ticker", "symbol", f_str), ("Direction", "direction", f_str), ("Shares", "shares", f_int),
              ("Entry date", "entry_date", f_date), ("Entry time", "entry_time", f_time), ("Entry via", "entry_how", f_str), ("Entry price", "entry_px", f_px),
              ("Entry value $", "entry_value", f_px), ("Ranked by", "score", f_str), ("Score / band", "score_value", lambda v: f_num(v, 4)), ("Rank", "rank", f_int),
              ("Move to decision", "move", f_pct), ("ML forecast", "forecast", lambda v: f_num(v, 4)), ("Decision price", "decision_px", f_px),
              ("Exit date", "exit_date", f_date), ("Exit time", "exit_time", f_time), ("Exit via", "exit_how", f_str), ("Exit price", "exit_px", f_px),
              ("Exit reason", "exit_reason", f_str), ("Costs $", "fees", f_px), ("P&L $", "pnl", f_money), ("Return", "ret", f_pct)]


def _trade_color(row, c):
    if c == "leg":
        return ACCENT if row["leg"] == "INTRADAY" else None
    if c == "direction":
        return GREEN
    if c in ("pnl", "ret"):
        v = row[c]
        return None if not np.isfinite(v) else (GREEN if v > 0 else RED)
    return None


class BacktestTab(QWidget):
    finished = pyqtSignal()                                                                    # a backtest result arrived (the live tab compares its model with it)

    def __init__(self, panel, folder: str):
        super().__init__()
        self.panel, self.folder = panel, folder
        self._workers = WorkerBag()
        self._w = None
        self._mk: dict = {}
        self.result = None
        self.ran_model = None
        root = QVBoxLayout(self)
        root.addWidget(strip("Decisions use only what is known at the decision time: overnight = the 15-minute SIP bars that ended by 15:30 (15:45) + earlier official closes, "
                             "filled in the closing auction and sold in the next opening auction; intraday = each 30-minute bar of the intraday feed, filled at its close plus "
                             "the half spread. No leverage: gross <= 1x at every moment. In-sample before the split date, out-of-sample after.", YELLOW))
        top = QHBoxLayout()
        self.btn = QPushButton("Run backtest"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.run)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.clicked.connect(self.stop); self.btn_stop.setEnabled(False)
        self.btn_csv = QPushButton("Export trades (CSV)"); self.btn_csv.clicked.connect(self.export); self.btn_csv.setEnabled(False)
        self.btn_train = QPushButton("Train ML models"); self.btn_train.clicked.connect(self.train)
        self.btn_train.setToolTip("Fit the missing yearly forecast models (each on the data before its year) for this book's universe and decision time; saved in models/")
        self.status = muted("")
        top.addWidget(self.btn); top.addWidget(self.btn_stop); top.addWidget(self.btn_csv); top.addWidget(self.btn_train); top.addWidget(self.status, 1)
        root.addLayout(top)
        self.model_lbl = muted("")
        self.ml_lbl = muted("")
        root.addWidget(self.model_lbl); root.addWidget(self.ml_lbl)
        k1 = ("Sharpe", "In-sample", "Out-of-sample", "CAGR", "Max drawdown", "Volatility")
        k2 = ("Trades", "Win rate", "Overnight leg", "Intraday leg", "Overnight exposure", "Beta to SPY")
        self.k = {}
        for names in (k1, k2):
            row = QHBoxLayout()
            for n in names:
                self.k[n] = Kpi(n)
                row.addWidget(self.k[n])
            root.addLayout(row)
        self.tabs = QTabWidget()
        self.c_eq = CompareChart()
        self.tabs.addTab(self.c_eq, "Equity vs SPY")
        tw = QWidget(); tv = QVBoxLayout(tw)
        fl = QHBoxLayout()
        self.leg = QComboBox(); self.leg.addItems(["All legs", "OVERNIGHT", "INTRADAY"]); self.leg.currentTextChanged.connect(self._show_trades)
        self.filter = QLineEdit(); self.filter.setPlaceholderText("ticker filter: AA = tickers starting with AA · V, AAPL = exactly these")
        self.trade_info = muted("")
        fl.addWidget(self.leg); fl.addWidget(self.filter, 1); fl.addWidget(self.trade_info, 3)
        tv.addLayout(fl)
        self.t_trades = FrameView(TRADE_COLS, _trade_color, left=(1, 2, 3, 7, 10, 18, 20), filter_col=2)
        self.filter.textChanged.connect(self.t_trades.set_filter)
        tv.addWidget(self.t_trades, 1)
        self.tabs.addTab(tw, "Trades")
        self.t_year = make_table(["Year", "Book", "SPY", "Difference", "Overnight leg", "Intraday leg", "Sample"])
        self.tabs.addTab(self.t_year, "Years")
        lw = QWidget(); lv = QVBoxLayout(lw)
        self.t_legs = make_table(["Leg", "Sharpe all", "In-sample", "Out-of-sample", "CAGR (alone)", "Trades", "Win rate", "Avg trade", "P&L $"])
        self.c_legs = LineChart("cumulative return of each leg", 3.0)
        lv.addWidget(self.t_legs); lv.addWidget(self.c_legs, 1)
        self.tabs.addTab(lw, "Legs")
        self.t_names = make_table(["Ticker", "Leg", "P&L $", "Trades", "Win rate", "Avg return"])
        self.tabs.addTab(self.t_names, "Tickers")
        ex = QWidget(); ev = QVBoxLayout(ex)
        self.c_expo = LineChart("x equity", 2.4); self.c_npos = LineChart("names held overnight", 2.4)
        ev.addWidget(self.c_expo); ev.addWidget(self.c_npos)
        self.tabs.addTab(ex, "Exposure")
        self.t_ml = make_table(["Year", "Model file", "Fitted on rows", "Training data", "Fitted"])
        self.tabs.addTab(self.t_ml, "ML models")
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True)
        self.tabs.addTab(self.log_box, "Log")
        root.addWidget(self.tabs, 1)
        self.panel.changed.connect(self._model_changed)
        self._model_changed()

    def _model_changed(self) -> None:
        m = self.panel.model()
        same = self.ran_model is not None and self.ran_model.to_dict() == m.to_dict()
        self.model_lbl.setText(describe(m) + ("" if same or self.ran_model is None else "   ·   (the results below are for a different model - run again)"))
        if m.uses_ml:
            yrs = ml.available_years(self.folder, m)
            this = pd.Timestamp.now().year
            self.ml_lbl.setText(f"ML forecast models ({ml.model_key(m)}): " + (f"{yrs[0]} .. {yrs[-1]}" if yrs else "none") +
                                ("" if this in yrs else f"  ·  ⚠ no model for {this}: Train ML models (live cannot rank without it)"))
            self.ml_lbl.setStyleSheet(f"color: {GREEN if this in yrs else YELLOW};")
        else:
            self.ml_lbl.setText("ranking by the rule alone: no ML model needed")
            self.ml_lbl.setStyleSheet("")

    def log(self, m: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now():%H:%M:%S}  {m.strip()}")
        self.status.setText(m.strip()[:140])

    def market(self, m, log):
        syms = E.model_symbols(m)
        key = (tuple(syms), m.minute, m.intraday_symbol, m.intraday_feed)
        if key not in self._mk:
            self._mk = {key: E.load_market(m, log=log)}
        return self._mk[key]

    def run(self) -> None:
        if self._workers.busy():
            return
        m = self.panel.model()
        a = self.panel.backtest_args()
        self.btn.setEnabled(False); self.btn_stop.setEnabled(True); self.btn_train.setEnabled(False)
        self.tabs.setCurrentWidget(self.log_box)

        def fn(log):
            w = self._w
            mk = self.market(m, log)
            log(f"running {m.name} from {a['start']} ...")
            r = E.run(m, mk, self.folder, start=a["start"], capital=a["capital"], split=a["split"], log=log, stop=lambda: w is not None and w.cancelled())
            return r, m, a
        self._w = FnWorker(fn)
        self._workers.start(self._w, self._done, self.log)

    def train(self) -> None:
        if self._workers.busy():
            return
        m = self.panel.model()
        if not m.uses_ml:
            self.log("this book ranks by the rule alone - pick 'rule+ml' to use (and train) the forecast")
            return
        self.btn.setEnabled(False); self.btn_train.setEnabled(False)
        self.tabs.setCurrentWidget(self.log_box)

        def fn(log):
            mk = self.market(m, log)
            log("building the feature table (every rankable name of every session) ...")
            df = E.training_frame(m, mk)
            log(f"{len(df):,} rows; fitting the missing yearly models ({ml.FIRST_YEAR} .. {pd.Timestamp.now().year}) ...")
            return ml.train(self.folder, m, df, log=log)

        def done(res, err):
            self.btn.setEnabled(True); self.btn_train.setEnabled(True)
            self.log("training failed: " + err if err else (f"saved {len(res)} model(s)" if res else "every year already has its model"))
            self._model_changed()
        self._workers.start(FnWorker(fn), done, self.log)

    def stop(self) -> None:
        if self._w is not None:
            self._w.cancel()

    def invalidate_data(self) -> None:
        self._mk = {}

    def _show_trades(self, *_) -> None:
        if self.result is None:
            return
        tr = self.result.trades.copy()
        if len(tr):
            for c in ("entry_date", "exit_date", "entry_time", "exit_time"):
                tr[c] = pd.to_datetime(tr[c])
            if self.leg.currentText() != "All legs":
                tr = tr[tr.leg == self.leg.currentText()]
        self.t_trades.set_frame(tr)
        done = tr[tr.exit_date.notna()] if len(tr) else tr
        self.trade_info.setText(f"{len(tr):,} trades ({len(tr) - len(done)} still open at the end) · {tr.symbol.nunique() if len(tr) else 0} tickers · click a header to sort")

    def _done(self, res, err) -> None:
        self.btn.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_train.setEnabled(True)
        if err:
            self.log("backtest failed: " + err)
            return
        r, m, a = res
        self.result, self.ran_model = r, m
        self._model_changed()
        st = r.stats
        al, i, o, sa, si, so = (st.get(k, {}) for k in ("all", "is", "oos", "spy_all", "spy_is", "spy_oos"))
        col = lambda x: (GREEN if x > 0 else RED) if x is not None and np.isfinite(x) else None
        f2 = lambda b, k: f"{b[k]:.2f}" if k in b and np.isfinite(b[k]) else "—"
        self.k["Sharpe"].set(f2(al, "sharpe"), col(al.get("sharpe")), f"SPY {f2(sa, 'sharpe')}")
        self.k["In-sample"].set(f2(i, "sharpe"), col(i.get("sharpe")), f"SPY {f2(si, 'sharpe')} · to {a['split']}")
        self.k["Out-of-sample"].set(f2(o, "sharpe"), col(o.get("sharpe")), f"SPY {f2(so, 'sharpe')} · from {a['split']}")
        self.k["CAGR"].set(pct(al.get("cagr")), col(al.get("cagr")), f"SPY {pct(sa.get('cagr'))} · OOS {pct(o.get('cagr'))} vs {pct(so.get('cagr'))}")
        self.k["Max drawdown"].set(pct(al.get("max_dd")), RED, f"SPY {pct(sa.get('max_dd'))}")
        self.k["Volatility"].set(pct(al.get("vol")), None, f"SPY {pct(sa.get('vol'))}")
        self.k["Trades"].set(f"{st.get('trades', 0):,}", None, f"{st.get('overnight_trades', 0):,} overnight · {st.get('intraday_trades', 0):,} intraday")
        self.k["Win rate"].set(pct(st.get("win_rate"), 1), None, f"overnight {pct(st.get('overnight_win'), 0)} · intraday {pct(st.get('intraday_win'), 0)}")
        on, it = st.get("overnight_leg", {}), st.get("intraday_leg", {})
        self.k["Overnight leg"].set(f2(on.get("all", {}), "sharpe"), col(on.get("all", {}).get("sharpe")), f"Sharpe alone · OOS {f2(on.get('oos', {}), 'sharpe')}")
        self.k["Intraday leg"].set(f2(it.get("all", {}), "sharpe"), col(it.get("all", {}).get("sharpe")), f"Sharpe alone · OOS {f2(it.get('oos', {}), 'sharpe')}")
        self.k["Overnight exposure"].set(f"{st.get('avg_exposure', 0):.0%}", None, f"invested {st.get('invested_nights', 0):.0%} of nights · max {st.get('max_exposure', 0):.2f}x")
        self.k["Beta to SPY"].set(f"{st.get('beta_spy', float('nan')):+.2f}", None, f"corr {st.get('corr_spy', float('nan')):+.2f}")
        self.c_eq.show(r.equity, r.bench, pd.Timestamp(a["split"]), m.name)
        self._show_trades()
        tr = r.trades
        done = tr[tr.exit_date.notna()] if len(tr) else tr
        by, sb = st.get("by_year", {}), st.get("spy_by_year", {})
        dly = r.daily
        y_on = (1 + dly.overnight_ret).groupby(dly.index.year).prod() - 1
        y_id = (1 + dly.intraday_ret).groupby(dly.index.year).prod() - 1
        sp = pd.Timestamp(a["split"]).year
        t = self.t_year
        t.setRowCount(len(by))
        for rr, (y, v) in enumerate(sorted(by.items())):
            s_ = sb.get(y, float("nan"))
            set_row(t, rr, [str(y), pct(v, 1), pct(s_, 1), pct(v - s_, 1, True), pct(y_on.get(y), 1), pct(y_id.get(y), 1), "out-of-sample" if y >= sp else "in-sample"],
                    {1: col(v), 2: col(s_), 3: col(v - s_), 4: col(y_on.get(y)), 5: col(y_id.get(y))})
        t = self.t_legs
        t.setRowCount(2)
        for rr, leg in enumerate(("overnight", "intraday")):
            b = st.get(f"{leg}_leg", {})
            set_row(t, rr, [leg, f2(b.get("all", {}), "sharpe"), f2(b.get("is", {}), "sharpe"), f2(b.get("oos", {}), "sharpe"), pct(b.get("all", {}).get("cagr")),
                            f"{st.get(f'{leg}_trades', 0):,}", pct(st.get(f"{leg}_win"), 1), pct(st.get(f"{leg}_avg_ret"), 3), f"{st.get(f'{leg}_pnl', 0):+,.0f}"],
                    {1: col(b.get("all", {}).get("sharpe")), 8: col(st.get(f"{leg}_pnl"))}, align_left=(0,))
        self.c_legs.show_series({"overnight leg": (1 + dly.overnight_ret).cumprod() - 1, "intraday leg": (1 + dly.intraday_ret).cumprod() - 1,
                                 "book": (1 + dly.ret).cumprod() - 1})
        t = self.t_names
        if len(done):
            g = done.groupby(["symbol", "leg"]).agg(pnl=("pnl", "sum"), n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()), ret=("ret", "mean"))
            g = g.sort_values("pnl", ascending=False)
            t.setRowCount(len(g))
            for rr, ((s, lg), x) in enumerate(g.iterrows()):
                set_row(t, rr, [s, lg, f"{x.pnl:+,.0f}", f"{int(x.n):,}", pct(x.win, 0), pct(x.ret, 3)], {2: col(x.pnl), 5: col(x.ret)}, align_left=(0, 1))
        self.c_expo.show_series({"overnight value after the close (x equity)": r.exposure}, hlines=[1.0])
        self.c_npos.show_series({"names held overnight": dly.overnight_names}, hlines=[m.k])
        t = self.t_ml
        used = r.ml_used or {}
        t.setRowCount(len(used))
        for rr, (y, inf) in enumerate(sorted(used.items())):
            if inf:
                set_row(t, rr, [str(y), os.path.basename(ml.model_path(self.folder, m, y)), f"{inf['rows']:,}", f"{inf['first']} .. {inf['last']}", inf.get("fitted", "")])
            else:
                set_row(t, rr, [str(y), "MISSING - no overnight entries that year", "", "", ""], {1: RED})
        if st.get("sessions_without_forecast"):
            self.log(f"⚠ {st['sessions_without_forecast']} session(s) had no ML forecast (no model for that year) - no overnight entries on them; train the models or start later")
        self.btn_csv.setEnabled(len(tr) > 0)
        self.tabs.setCurrentIndex(0)
        self.log(f"{m.name}: {E.summary(st)}")
        self.finished.emit()

    def export(self) -> None:
        if self.result is None:
            return
        name = self.ran_model.name.replace(" ", "_").replace("+", "p")
        path, _ = QFileDialog.getSaveFileName(self, "Export trades", os.path.join(self.folder, "reports", f"trades_{name}.csv"), "CSV (*.csv)")
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.result.trades.to_csv(path, index=False)
            self.log(f"saved {len(self.result.trades):,} trades to {path}")
