"""Signal monitor: where each pair's spread and s-score are now, what the trade would cost against what it could capture, and the leveraged-ETF risk map."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget

import levpairs_data as D
import levpairs_live as LV
from levpairs_sim import PAIR_ORDER, leg_costs
from supertrend_ui.theme import ACCENT, GREEN, MUTED, RED, YELLOW, Kpi, QComboBox, make_table, muted, pct, set_row, strip
from supertrend_ui.workers import FnWorker, WorkerBag

from .charts import LineChart
from .settings_panel import SettingsPanel

ENV_HTML = """
<h3>Risk map - leveraged-ETF pairs, October 2026</h3>
<p style="color:#8b98a9">What can go wrong when you trade a 3x ETF against its 1x ETF, and what the app does about it.</p>
<table cellpadding="6">
<tr><th align="left">Risk</th><th align="left">What is going on</th><th align="left">What the strategy does about it</th></tr>
<tr><td><b>Costs are bigger than the edge</b></td><td>A pair trade pays the spread on two legs, and the hedge leg is 3x larger: 4.6 bp round trip for TQQQ/QQQ, 5.7 for SPXL/SPY, 11 for SOXL/SMH. The measured gross reversion per trade is under 1 bp.</td>
<td>Costs are explicit and charged on both legs; an entry needs the expected capture to be a multiple of the round-trip cost; the Cost tab shows where the edge disappears. Honest result: it does not clear the bar.</td></tr>
<tr><td><b>Stale prints look like alpha</b></td><td>The last trade of a bar is at the bid or the ask and up to seconds old; two such prints of different ETFs make a 'spread' that snaps back - and a backtest that fills at them earns the snap-back (8-12 bp a trade, more than the quoted bounce).</td>
<td>Default fill is the next bar's open; the 'OPTIMISTIC' preset shows how large the bias is (on 5-minute bars the Sharpe flips from about -5 to about +2.9; a midpoint fill gives about -0.7).</td></tr>
<tr><td><b>Daily reset and the closing rebalance</b></td><td>3x funds rebalance at the close: after an up day they must buy, after a down day sell, a flow that moves the underlying late in the day and can dislocate the pair into 16:00.</td>
<td>Everything is flat 10 minutes before the close; no entries in the last 30 minutes.</td></tr>
<tr><td><b>Gap days and halts</b></td><td>In a violent move the 3x ETF can trade away from 3x for hours, or halt (LULD) while the 1x ETF trades.</td>
<td>Model-failure stop at |s| 4, move filter (no entries when the index is far from its prior close), per-pair day stop, daily-loss halt, hard flatten at 15:52. A halted leg cannot be closed: size small.</td></tr>
<tr><td><b>Basis risk, SOXL vs SMH</b></td><td>SOXL tracks 3x an ICE semiconductor index, SMH a different (MVIS) index: the 'spread' contains real index differences, not only mispricing; Taiwan / tariff headlines hit them unevenly.</td>
<td>Rolling beta option, half-life filter, the pair is the most expensive and has its own weight (0 switches it off).</td></tr>
<tr><td><b>Real-time data</b></td><td>This account cannot read real-time SIP data, only IEX. IEX sees a few percent of the volume; the thinner ETFs have empty 1-minute bars.</td>
<td>IEX s-scores correlate only 0.2-0.57 with the consolidated-tape scores (fidelity test, report): fine for watching, not for sending orders. Real-time SIP data is needed. Stale bars hold the position, then exit.</td></tr>
<tr><td><b>Leg risk</b></td><td>One leg of the pair fills and the other is rejected: you are suddenly directional at 3x.</td>
<td>Both legs are confirmed; a failed leg unwinds the other; the pair is marked BROKEN until you clear it.</td></tr>
<tr><td><b>Pattern-day-trader rule</b></td><td>A margin account under $25,000 cannot open and close positions every day.</td><td>Pre-flight fails a real-money account below $25k; the paper account is exempt.</td></tr>
</table>
"""


class SignalsTab(QWidget):
    def __init__(self, settings: SettingsPanel, qs: QSettings):
        super().__init__()
        self.sp, self.qs = settings, qs
        self._workers = WorkerBag()
        self._busy = False
        self.views: dict = {}
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn = QPushButton("Refresh signals"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.refresh)
        self.live_chk = QCheckBox("Use the latest bars from Alpaca (IEX / SIP per the feed setting)"); self.live_chk.setToolTip("Unticked: the last session in the cached 1-minute bars.")
        self.pair_sel = QComboBox(); self.pair_sel.addItems(PAIR_ORDER); self.pair_sel.currentIndexChanged.connect(self._chart)
        self.status = muted("", wrap=False)
        top.addWidget(self.btn); top.addWidget(self.live_chk); top.addWidget(QLabel("Chart")); top.addWidget(self.pair_sel); top.addWidget(self.status, 1)
        root.addLayout(top)
        k = QHBoxLayout()
        self.k = {n: Kpi(n) for n in ("As of", "Pairs beyond the entry level", "Best edge / cost", "SPY since prior close", "Bar size")}
        for w in self.k.values():
            k.addWidget(w)
        root.addLayout(k)
        self.banner = strip("", color=ACCENT)
        root.addWidget(self.banner)
        self.tabs = QTabWidget()
        self.t_sig = make_table(["Pair", "Spread (bp)", "s-score", "Half-life (min)", "σ eq (bp)", "β", "Round trip (bp)", "Capture if reverts (bp)", "Capture / cost", "Verdict"])
        self.tabs.addTab(self.t_sig, "Pairs now")
        self.c_s = LineChart("s-score of the selected pair, this session", 3.0)
        self.tabs.addTab(self.c_s, "Session chart")
        self.t_cost = make_table(["Pair", "Half-spread L (bp)", "Half-spread B (bp)", "Slippage / leg (bp)", "β", "Round trip (bp of the 3x leg)", "Gross capture needed to break even"])
        self.tabs.addTab(self.t_cost, "Cost of a round trip")
        env = QTextBrowser(); env.setHtml(ENV_HTML)
        self.tabs.addTab(env, "Risk map")
        root.addWidget(self.tabs, 1)
        self._cost_table()

    def _cost_table(self) -> None:
        c = self.sp.sim_cfg()
        t = self.t_cost
        t.setRowCount(len(D.PAIRS))
        for r, p in enumerate(D.PAIRS):
            cL, cB = leg_costs(p, c)
            rt = 2 * (cL + 3.0 * cB)
            set_row(t, r, [D.pair_name(p), f"{D.half_spread_bp(p[0]):.2f}", f"{D.half_spread_bp(p[1]):.2f}", f"{c.slip_bp:.2f}", "3.0", f"{rt:.1f}", f"{rt:.1f} bp per trade"])

    def refresh(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.btn.setEnabled(False)
        self.status.setText("computing ...")
        sc, c, feed, live = self.sp.sig_cfg(), self.sp.sim_cfg(), self.sp.feed(), self.live_chk.isChecked()
        pairs = D.PAIRS

        def fn(log):
            if live:
                now = LV.clock_now().tz_localize(None)
                date = now.normalize() if now.hour >= 10 else now.normalize() - pd.Timedelta(days=1)
                while date.weekday() >= 5:
                    date -= pd.Timedelta(days=1)
                end = min(now, date + pd.Timedelta(hours=15, minutes=59, seconds=59)) if date == now.normalize() else date + pd.Timedelta(hours=15, minutes=59, seconds=59)
                src = LV.ApiSource(feed)
            else:
                b = pd.read_parquet(D.path1("SPY", feed if feed == "sip" else "sip"))
                date = b.index.normalize().unique()[-1]
                end = date + pd.Timedelta(hours=15, minutes=59, seconds=59)
                src = LV.ParquetSource(feed if feed == "sip" else "sip")
            views, _ = LV.pair_views(src, list(pairs), sc, c, date, end)
            return date, end, views, src.feed
        w = FnWorker(fn)
        self._workers.start(w, lambda res, err: self._done(res, err, sc, c), lambda s: self.status.setText(s.strip()[:90]))

    def _done(self, res, err, sc, c) -> None:
        self._busy = False
        self.btn.setEnabled(True)
        if err:
            self.status.setText(f"failed: {err}")
            return
        date, end, views, feed = res
        self.views, self._date, self._tf = views, date, sc.tf
        self.k["As of"].set(f"{end:%H:%M}", None, f"{date:%a %Y-%m-%d} · {feed.upper()} bars")
        self.k["Bar size"].set(f"{sc.tf} min")
        t = self.t_sig
        t.setRowCount(len(views))
        n_beyond, best = 0, 0.0
        rows = []
        for r, (name, v) in enumerate(views.items()):
            L, B = name.split("/")
            cL, cB = leg_costs((L, B), c)
            rt = 2 * (cL + v.beta * cB)
            s_close = 0.5 * (c.s_close_long + c.s_close_short)
            cap = max(abs(v.s) - s_close, 0.0) * v.sig_eq_bp if np.isfinite(v.s) else float("nan")
            ratio = cap / rt if rt > 0 and np.isfinite(cap) else float("nan")
            if np.isfinite(ratio):
                best = max(best, ratio)
            beyond = np.isfinite(v.s) and abs(v.s) >= c.s_open
            n_beyond += int(beyond)
            hl_ok = c.max_halflife_min <= 0 or (np.isfinite(v.hl_min) and v.hl_min <= c.max_halflife_min)
            if not np.isfinite(v.s):
                verdict = "no score (the OU fit does not revert)"
            elif not beyond:
                verdict = "inside the band"
            elif not hl_ok:
                verdict = f"blocked: half-life {v.hl_min:.0f} min"
            elif c.min_edge_ratio > 0 and ratio < c.min_edge_ratio:
                verdict = f"blocked: capture only {ratio:.1f}x cost"
            elif v.move_block:
                verdict = "blocked: index move filter"
            else:
                verdict = ("LONG" if v.s < 0 else "SHORT") + " the spread"
            rows.append((name, v, rt, cap, ratio, verdict))
            set_row(t, r, [name, f"{v.x_bp:+.1f}", f"{v.s:+.2f}" if np.isfinite(v.s) else "—", f"{v.hl_min:.0f}" if np.isfinite(v.hl_min) else "—", f"{v.sig_eq_bp:.1f}" if np.isfinite(v.sig_eq_bp) else "—",
                           f"{v.beta:.2f}", f"{rt:.1f}", f"{cap:.1f}" if np.isfinite(cap) else "—", f"{ratio:.2f}" if np.isfinite(ratio) else "—", verdict],
                    {2: GREEN if (np.isfinite(v.s) and v.s < -c.s_open) else RED if (np.isfinite(v.s) and v.s > c.s_open) else None, 9: GREEN if verdict.startswith("LONG") else RED if verdict.startswith("SHORT") else None},
                    align_left=(9,))
        spy = views.get(PAIR_ORDER[2])
        self.k["Pairs beyond the entry level"].set(str(n_beyond), None, f"|s| ≥ {c.s_open:g}")
        self.k["Best edge / cost"].set(f"{best:.2f}", GREEN if best >= 1.5 else YELLOW if best >= 1.0 else RED, "expected capture ÷ round-trip cost")
        self.k["SPY since prior close"].set(pct(spy.rB_day, 2) if spy else "—")
        self.banner.setText(f"{date:%A %d %B %Y}, {end:%H:%M} · {sc.tf}-minute bars · {n_beyond} pair(s) beyond |s| {c.s_open:g} · best capture/cost {best:.2f}x"
                            + (" - the trade pays only when this is well above 1" if best < 1.5 else ""))
        self._cost_table()
        self._chart()
        self.status.setText("updated")

    def _chart(self) -> None:
        v = self.views.get(self.pair_sel.currentText())
        if not v or not hasattr(self, "_date"):
            return
        idx = [self._date + pd.Timedelta(minutes=D.OPEN_MIN + (k + 1) * self._tf) for k in range(len(v.s_hist))]
        c = self.sp.sim_cfg()
        self.c_s.show_series({"s-score": pd.Series(v.s_hist, index=idx)}, [c.s_open, -c.s_open, 0.0])
