"""Research tab: does the strategy's score (and, for the prediction strategy, each feature) predict the forward RESIDUAL return? Rank IC by horizon, in- and out-of-sample, and deciles."""
from __future__ import annotations

from datetime import datetime

import numpy as np
from PyQt6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget

from .. import features as X
from .. import research as R
from .. import strategy as S
from .charts import BarChart
from .theme import GREEN, RED, YELLOW, QSpinBox, make_table, muted, set_row, strip
from .workers import FnWorker, WorkerBag


class ResearchTab(QWidget):
    def __init__(self, settings, folder: str):
        super().__init__()
        self.sp, self.folder = settings, folder
        self._workers = WorkerBag()
        root = QVBoxLayout(self)
        root.addWidget(strip("Rank IC = the cross-sectional (Spearman) correlation between the score at a bar and the residual return that follows from the fill bar over the horizon; "
                             "averaged per day, t-statistics are Newey-West. |t| > 3 is the bar after testing many features (Harvey-Liu-Zhu 2016). A negative IC means the "
                             "feature predicts REVERSAL.", YELLOW))
        top = QHBoxLayout()
        self.btn = QPushButton("Test the strategy's score"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.run_score)
        self.btn_f = QPushButton("Test every feature"); self.btn_f.clicked.connect(self.run_features)
        self.dec_h = QSpinBox(); self.dec_h.setRange(1, 260); self.dec_h.setValue(26); self.dec_h.setPrefix("deciles at "); self.dec_h.setSuffix(" bars")
        self.every = QSpinBox(); self.every.setRange(1, 26); self.every.setValue(2); self.every.setPrefix("every "); self.every.setSuffix(" bars")
        self.status = muted("")
        for w in (self.btn, self.btn_f, self.dec_h, self.every):
            top.addWidget(w)
        top.addWidget(self.status, 1)
        root.addLayout(top)
        self.tabs = QTabWidget()
        self.t_ic = make_table(["Horizon (bars)", "IS IC", "IS t", "OOS IC", "OOS t", "IS days", "OOS days"])
        self.tabs.addTab(self.t_ic, "Score IC")
        dq = QWidget(); dv = QVBoxLayout(dq)
        self.c_dec = BarChart(3.0); self.c_dec_o = BarChart(3.0)
        dv.addWidget(self.c_dec); dv.addWidget(self.c_dec_o)
        self.tabs.addTab(dq, "Deciles")
        self.t_feat = make_table(["Feature", "Horizon", "IS IC", "IS t", "OOS IC", "OOS t", "Verdict"])
        self.tabs.addTab(self.t_feat, "Features")
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True)
        self.tabs.addTab(self.log_box, "Log")
        root.addWidget(self.tabs, 1)

    def log(self, m: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now():%H:%M:%S}  {m.strip()}")
        self.status.setText(m.strip()[:140])

    def run_score(self) -> None:
        if self._workers.busy():
            return
        strat = self.sp.strategy(self.folder)
        every, h = self.every.value(), self.dec_h.value()

        def fn(log):
            ctx = S.context(strat, log=log)
            sc = strat.score(ctx, log)
            ic = R.score_ic(ctx, sc, strat.v["start"], strat.v["split"], strat.v["end"] or None, every=every, log=log)
            dec = R.deciles(ctx, sc, h, strat.v["start"], strat.v["split"], strat.v["end"] or None, every=every)
            return ic, dec
        self.tabs.setCurrentWidget(self.log_box)
        self._workers.start(FnWorker(fn), self._score_done, self.log)

    def _score_done(self, res, err) -> None:
        if err:
            self.log("failed: " + err)
            return
        ic, dec = res
        t = self.t_ic
        t.setRowCount(len(ic))
        c = lambda v: (GREEN if v > 0 else RED) if np.isfinite(v) else None
        for r, x in enumerate(ic.itertuples()):
            set_row(t, r, [str(x.horizon_bars), f"{x.IS_ic:+.4f}", f"{x.IS_t:+.1f}", f"{x.OOS_ic:+.4f}", f"{x.OOS_t:+.1f}", str(x.IS_days), str(x.OOS_days)],
                    {1: c(x.IS_ic), 3: c(x.OOS_ic)})
        for chart, s in ((self.c_dec, "IS"), (self.c_dec_o, "OOS")):
            d = dec[dec["sample"] == s]
            chart.show_bars([f"D{k}" for k in d.decile], list(d.fwd_bp), title=f"{s}: forward residual return by score decile (bp, vs the average member)", horizontal=False)
        self.tabs.setCurrentWidget(self.t_ic)
        self.log("done")

    def run_features(self) -> None:
        if self._workers.busy():
            return
        strat = self.sp.strategy(self.folder)
        every = self.every.value()

        def fn(log):
            ctx = S.context(strat, log=log)
            return R.feature_ic(ctx, list(X.FEATURES), strat.v["start"], strat.v["split"], strat.v["end"] or None, every=every, log=log)
        self.tabs.setCurrentWidget(self.log_box)
        self._workers.start(FnWorker(fn), self._feat_done, self.log)

    def _feat_done(self, df, err) -> None:
        if err:
            self.log("failed: " + err)
            return
        t = self.t_feat
        t.setRowCount(len(df))
        for r, x in enumerate(df.itertuples()):
            same = np.sign(x.IS_ic) == np.sign(x.OOS_ic)
            v = "predicts (IS and OOS)" if abs(x.IS_t) >= 3 and same and abs(x.OOS_t) >= 2 else ("IS only" if abs(x.IS_t) >= 3 else "no")
            c = lambda v_: (GREEN if v_ > 0 else RED) if np.isfinite(v_) else None
            set_row(t, r, [x.feature, str(x.horizon_bars), f"{x.IS_ic:+.4f}", f"{x.IS_t:+.1f}", f"{x.OOS_ic:+.4f}", f"{x.OOS_t:+.1f}", v], {2: c(x.IS_ic), 4: c(x.OOS_ic)},
                    align_left=(6,))
        self.tabs.setCurrentWidget(t)
        self.log("done")
