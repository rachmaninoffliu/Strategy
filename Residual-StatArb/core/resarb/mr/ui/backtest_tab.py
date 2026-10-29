"""Backtest tab: runs the picked model (the same decide() as live) and shows every trade - entry, exit, ticker, shares, value, direction, signal, rank, P&L - plus the equity vs SPY."""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget

from ...ui.charts import LineChart
from ...ui.theme import GREEN, RED, YELLOW, Kpi, make_table, muted, pct, set_row, strip
from ...ui.workers import FnWorker, WorkerBag
from .. import engine as E
from ..market import Market
from .charts import CompareChart
from .frame_model import FrameView, f_date, f_int, f_money, f_num, f_pct, f_px, f_str
from .model_panel import describe

TRADE_COLS = [("#", "trade_id", lambda v: f"{int(v) + 1}"), ("Ticker", "symbol", f_str), ("Direction", "direction", f_str), ("Shares", "shares", f_int),
              ("Entry date", "entry_date", f_date), ("Entry auction", "entry_auction", f_str), ("Entry price", "entry_px", f_px), ("Entry value $", "entry_value", f_px),
              ("Signal", "signal", f_str), ("Signal value", "signal_value", f_num), ("Rank", "rank", f_int), ("Decision price", "decision_px", f_px),
              ("Exit date", "exit_date", f_date), ("Exit auction", "exit_auction", f_str), ("Exit price", "exit_px", f_px), ("Exit reason", "exit_reason", f_str),
              ("Held (sessions)", "held_sessions", f_int), ("P&L $", "pnl", f_money), ("Return", "ret", f_pct)]


def _trade_color(row, c):
    if c == "direction":
        return GREEN if row["direction"] == "LONG" else RED
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
        root.addWidget(strip("Decisions use only what is known at the decision time (the 15-minute bars that ended by then + earlier official closes); orders fill in the closing auction "
                             "at the official close, overnight holds leave in the next opening auction. No leverage: the book never exceeds 1x equity. In-sample before the split date, "
                             "out-of-sample after - the presets were picked on in-sample data only.", YELLOW))
        top = QHBoxLayout()
        self.btn = QPushButton("Run backtest"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.run)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.clicked.connect(self.stop); self.btn_stop.setEnabled(False)
        self.btn_csv = QPushButton("Export trades (CSV)"); self.btn_csv.clicked.connect(self.export); self.btn_csv.setEnabled(False)
        self.status = muted("")
        top.addWidget(self.btn); top.addWidget(self.btn_stop); top.addWidget(self.btn_csv); top.addWidget(self.status, 1)
        root.addLayout(top)
        self.model_lbl = muted("")
        root.addWidget(self.model_lbl)
        k1 = ("Sharpe", "In-sample", "Out-of-sample", "CAGR", "Max drawdown", "Volatility")
        k2 = ("Trades", "Win rate", "Avg trade", "Avg hold", "Exposure", "Beta to SPY")
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
        self.filter = QLineEdit(); self.filter.setPlaceholderText("filter: AA = tickers starting with AA · V, AAPL = exactly these")
        self.trade_info = muted("")
        fl.addWidget(self.filter, 1); fl.addWidget(self.trade_info, 3)
        tv.addLayout(fl)
        self.t_trades = FrameView(TRADE_COLS, _trade_color, left=(1, 2, 5, 8, 13, 15))
        self.filter.textChanged.connect(self.t_trades.set_filter)
        tv.addWidget(self.t_trades, 1)
        self.tabs.addTab(tw, "Trades")
        self.t_year = make_table(["Year", "Strategy", "SPY", "Difference", "Sample"])
        self.tabs.addTab(self.t_year, "Years")
        self.t_names = make_table(["Ticker", "P&L $", "Trades", "Win rate", "Avg return", "Avg hold"])
        self.tabs.addTab(self.t_names, "Tickers")
        self.t_reason = make_table(["Exit reason", "Trades", "Win rate", "Avg return", "P&L $"])
        self.tabs.addTab(self.t_reason, "Exit reasons")
        ex = QWidget(); ev = QVBoxLayout(ex)
        self.c_expo = LineChart("x equity", 2.4); self.c_npos = LineChart("positions", 2.4)
        ev.addWidget(self.c_expo); ev.addWidget(self.c_npos)
        self.tabs.addTab(ex, "Exposure")
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True)
        self.tabs.addTab(self.log_box, "Log")
        root.addWidget(self.tabs, 1)
        self.panel.changed.connect(self._model_changed)
        self._model_changed()

    def _model_changed(self) -> None:
        m = self.panel.model()
        same = self.ran_model is not None and self.ran_model.to_dict() == m.to_dict()
        self.model_lbl.setText(describe(m) + ("" if same or self.ran_model is None else "   ·   (the results below are for a different model - run again)"))

    def log(self, m: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now():%H:%M:%S}  {m.strip()}")
        self.status.setText(m.strip()[:140])

    def market(self, m, log) -> Market:
        syms = E.model_symbols(m)
        key = (tuple(syms), m.minute)
        if key not in self._mk:
            log(f"loading {len(syms)} symbols (daily bars + the {m.minute // 100}:{m.minute % 100:02d} snapshots) ...")
            self._mk = {key: Market(syms, minute=m.minute, start="2015-06-01")}
        return self._mk[key]

    def run(self) -> None:
        if self._workers.busy():
            return
        m = self.panel.model()
        a = self.panel.backtest_args()
        self.btn.setEnabled(False); self.btn_stop.setEnabled(True)
        self.tabs.setCurrentWidget(self.log_box)

        def fn(log):
            w = self._w
            mk = self.market(m, log)
            log(f"running {m.name} from {a['start']} ...")
            r = E.run(m, mk, start=a["start"], capital=a["capital"], split=a["split"], log=log, stop=lambda: w is not None and w.cancelled())
            return r, m, a
        self._w = FnWorker(fn)
        self._workers.start(self._w, self._done, self.log)

    def stop(self) -> None:
        if self._w is not None:
            self._w.cancel()

    def invalidate_data(self) -> None:
        self._mk = {}

    def _done(self, res, err) -> None:
        self.btn.setEnabled(True); self.btn_stop.setEnabled(False)
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
        self.k["Trades"].set(f"{st.get('trades', 0):,}", None, f"{st.get('trades', 0) / max(al.get('days', 1) / 252, 1e-9):.0f} / year")
        self.k["Win rate"].set(pct(st.get("win_rate"), 1), None, f"profit factor {st.get('profit_factor', float('nan')):.2f}")
        self.k["Avg trade"].set(pct(st.get("avg_ret"), 2), col(st.get("avg_ret")), "net of the auction costs")
        self.k["Avg hold"].set(f"{st.get('avg_hold', float('nan')):.1f} sessions")
        self.k["Exposure"].set(f"{st.get('avg_exposure', 0):.0%}", None, f"max {st.get('max_exposure', 0):.2f}x (1.00x = fully invested, no leverage)")
        self.k["Beta to SPY"].set(f"{st.get('beta_spy', float('nan')):+.2f}", None, f"corr {st.get('corr_spy', float('nan')):+.2f}")
        self.c_eq.show(r.equity, r.bench, pd.Timestamp(a["split"]), m.name)
        tr = r.trades.copy()
        if len(tr):
            tr["entry_date"] = pd.to_datetime(tr.entry_date)
            tr["exit_date"] = pd.to_datetime(tr.exit_date)
        self.t_trades.set_frame(tr)
        done = tr[tr.exit_date.notna()] if len(tr) else tr
        self.trade_info.setText(f"{len(tr):,} trades ({len(tr) - len(done)} still open at the end) · {tr.symbol.nunique() if len(tr) else 0} tickers · click a header to sort")
        by, sb = st.get("by_year", {}), st.get("spy_by_year", {})
        sp = pd.Timestamp(a["split"]).year
        t = self.t_year
        t.setRowCount(len(by))
        for rr, (y, v) in enumerate(sorted(by.items())):
            s_ = sb.get(y, float("nan"))
            set_row(t, rr, [str(y), pct(v, 1), pct(s_, 1), pct(v - s_, 1, True), "out-of-sample" if y >= sp else "in-sample"], {1: col(v), 2: col(s_), 3: col(v - s_)})
        t = self.t_names
        if len(done):
            g = done.groupby("symbol").agg(pnl=("pnl", "sum"), n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()), ret=("ret", "mean"), hold=("held_sessions", "mean"))
            g = g.sort_values("pnl", ascending=False)
            t.setRowCount(len(g))
            for rr, (s, x) in enumerate(g.iterrows()):
                set_row(t, rr, [s, f"{x.pnl:+,.0f}", f"{int(x.n):,}", pct(x.win, 0), pct(x.ret, 2), f"{x.hold:.1f}"], {1: col(x.pnl), 4: col(x.ret)})
            g = done.assign(reason=done.exit_reason.str.replace(r"[\d.]+", "#", regex=True)).groupby("reason").agg(
                n=("pnl", "size"), win=("pnl", lambda x: (x > 0).mean()), ret=("ret", "mean"), pnl=("pnl", "sum")).sort_values("n", ascending=False)
            t = self.t_reason
            t.setRowCount(len(g))
            for rr, (s, x) in enumerate(g.iterrows()):
                set_row(t, rr, [s, f"{int(x.n):,}", pct(x.win, 0), pct(x.ret, 2), f"{x.pnl:+,.0f}"], {3: col(x.ret), 4: col(x.pnl)}, align_left=(0,))
        self.c_expo.show_series({"gross exposure (x equity)": r.exposure}, hlines=[1.0])
        if r.npos is not None:
            self.c_npos.show_series({"positions held": r.npos}, hlines=[m.slots])
        self.btn_csv.setEnabled(len(tr) > 0)
        self.tabs.setCurrentIndex(0)
        self.log(f"{m.name}: {E.summary(st)}")
        self.finished.emit()

    def export(self) -> None:
        if self.result is None:
            return
        name = self.ran_model.name.replace(" ", "_").replace("<", "lt").replace(">", "gt")
        path, _ = QFileDialog.getSaveFileName(self, "Export trades", os.path.join(self.folder, "reports", f"trades_{name}.csv"), "CSV (*.csv)")
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.result.trades.to_csv(path, index=False)
            self.log(f"saved {len(self.result.trades):,} trades to {path}")
