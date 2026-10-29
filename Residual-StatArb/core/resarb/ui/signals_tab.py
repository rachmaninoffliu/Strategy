"""Signals tab: the latest scores from the cached history (no broker needed) - every member's score, percentile, beta, recent residual returns, and the book the strategy would
hold if it started from flat."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .. import data as D
from .. import features as X
from .. import strategy as S
from ..portfolio import decide, exposures, hedge_etf
from .theme import GREEN, RED, make_table, muted, pct, set_row
from .workers import FnWorker, WorkerBag


class SignalsTab(QWidget):
    def __init__(self, settings, folder: str):
        super().__init__()
        self.sp, self.folder = settings, folder
        self._workers = WorkerBag()
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn = QPushButton("Refresh signals (latest cached bar)"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.refresh)
        self.status = muted("")
        top.addWidget(self.btn); top.addWidget(self.status, 1)
        root.addLayout(top)
        self.t = make_table(["Symbol", "Sector ETF", "Score", "Percentile", "Target w (from flat)", "Beta (SPY)", "Residual 1 day", "Residual 5 days", "Residual vol (daily)"])
        root.addWidget(self.t, 1)
        self.expo = muted("")
        root.addWidget(self.expo)

    def refresh(self) -> None:
        if self._workers.busy():
            return
        strat = self.sp.strategy(self.folder)

        def fn(log):
            end = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
            start = (end - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            ctx = S.context(strat, start=start, log=log, cache=False)
            sc = strat.live_score(ctx) if hasattr(strat, "live_score") and getattr(strat, "model_info", lambda: {})() else strat.score(ctx, log)
            t = len(ctx.p.ts) - 1
            p, res = ctx.p, ctx.res
            is_etf = np.array([s in D.ETFS for s in p.syms])
            sidx = np.where(~is_etf)[0]
            w = np.zeros(len(p.syms))
            w[sidx] = decide(sc[t, sidx], ctx.member[t, sidx], np.zeros(len(sidx)), strat.port_cfg(), res.sector_col[sidx], np.isfinite(p.open[t, sidx]))
            d = p.day[t]
            h = hedge_etf(w, res, d) if strat.port_cfg().hedge == "etf" else np.zeros(len(res.etf_idx))
            rr26 = X.roll_sum(res.e_cc, 26)[t]
            rr130 = X.roll_sum(res.e_cc, 130)[t]
            rows = []
            inv = {c: s for s, c in p.col.items()}
            for j in sidx:
                if not ctx.member[t, j]:
                    continue
                rows.append(dict(symbol=p.syms[j], sector=inv.get(int(res.sector_col[j]), "-"), score=float(sc[t, j]), w=float(w[j]), beta=float(res.b_m[d, j]) if np.isfinite(res.b_m[d, j]) else np.nan,
                                 r1=float(rr26[j]), r5=float(rr130[j]), ivol=float(np.sqrt(res.idio_var[d, j])) if np.isfinite(res.idio_var[d, j]) else np.nan))
            df = pd.DataFrame(rows)
            if len(df):
                df["pct"] = df.score.rank(pct=True)
                df = df.sort_values("score", ascending=False)
            return df, exposures(w, h, res, d), p.ts[t]
        self.status.setText("computing ...")
        self._workers.start(FnWorker(fn), self._done, lambda s: self.status.setText(s.strip()[:140]))

    def _done(self, res, err) -> None:
        if err:
            self.status.setText("failed: " + err)
            return
        df, ex, ts = res
        t = self.t
        t.setRowCount(len(df))
        for r, x in enumerate(df.itertuples()):
            set_row(t, r, [x.symbol, x.sector, f"{x.score:+.3f}" if np.isfinite(x.score) else "—", f"{x.pct:.2f}" if np.isfinite(x.pct) else "—", pct(x.w, 2) if x.w else "",
                           f"{x.beta:.2f}" if np.isfinite(x.beta) else "—", pct(x.r1, 2), pct(x.r5, 2), pct(x.ivol, 2)],
                    {4: GREEN if x.w > 0 else RED if x.w < 0 else None, 6: GREEN if x.r1 > 0 else RED, 7: GREEN if x.r5 > 0 else RED})
        if not len(df) or not np.isfinite(df.score).any():
            self.status.setText(f"no scores at the bar starting {ts:%Y-%m-%d %H:%M} ET - the prediction strategy needs a trained model first (Live trading > Train model)")
            return
        self.status.setText(f"scores at the bar starting {ts:%Y-%m-%d %H:%M} ET (cached SIP history)")
        self.expo.setText(f"book from flat: {ex['longs']} long / {ex['shorts']} short · stock gross {ex['gross_stocks']:.2f}x · ETF hedge {ex['gross_etf']:.2f}x · "
                          f"net {ex['net_total']:+.3f} · beta after hedge {ex['beta_after_hedge']:+.3f}")
