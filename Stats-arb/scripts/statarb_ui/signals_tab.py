"""Signal monitor: today's s-scores with the reason each candidate is (not) tradable, plus the macro / geopolitical risk dashboard the shock multiplier reads."""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget

import statarb_bridge as BR
import statarb_data as D
import statarb_live as LV
from statarb_core import FactorCfg, build_signals, shock_table
from statarb_sim import shock_multiplier
from supertrend_ui.theme import ACCENT, GREEN, MUTED, RED, YELLOW, Kpi, make_table, muted, pct, set_row, strip
from supertrend_ui.workers import FnWorker, WorkerBag

from .charts import Histogram, LineChart
from .settings_panel import SettingsPanel

ENV_HTML = """
<h3>Risk map - October 2026</h3>
<p style="color:#8b98a9">Sources are press summaries (Goldman Sachs breadth note 28 Sep 2026, Capital Group, prime-broker commentary on the 2026 quant drawdowns, FOMC minutes 7 Oct 2026) - treat the levels as approximate and check the live data.</p>
<table cellpadding="6">
<tr><th align="left">Risk</th><th align="left">What is going on</th><th align="left">What the strategy does about it</th></tr>
<tr><td><b>Idiosyncratic jump risk</b></td><td>~73% of a typical stock's 6-month variance is company-specific (58% on average since 2002). Median S&amp;P stock ~16% below its 52-week high while the index sits ~1% below its record - leadership is narrow and single names gap hard.</td>
<td>Earnings calendar filter (no entries within 6 sessions of an announcement, exit 2-3 sessions before); news / volume-spike filter; risk-sized positions (1/residual vol) with a per-name cap; model-failure stop at |s|&gt;3.5; time stop.</td></tr>
<tr><td><b>Crowding / quant unwinds</b></td><td>Repeated sharp drawdowns for systematic equity funds in Jan (junk squeeze), Mar and Jun-Jul 2026 (momentum / AI-memory unwind); stress correlation between funds ~0.9.</td>
<td>Book is projected to zero exposure to market beta, the top PCA factors, 11 sectors, momentum, volatility and size - the standard crowded factors; volatility target de-grosses after losses; gross capped at the Reg-T 2x limit.</td></tr>
<tr><td><b>Geopolitics: Middle East / Hormuz, oil</b></td><td>Strikes on Iran from late Feb, a fragile April truce, Brent in the $80s with spikes toward $100+ in tail scenarios; oil and yields rising together is the classic geopolitical signature.</td>
<td>Each stock's oil beta (126 sessions, controlling for SPY, rates and USD) is neutralised at book level; the shock score reads USO moves in sigma units and de-grosses for several sessions after one.</td></tr>
<tr><td><b>Trade policy / Taiwan</b></td><td>Chip-material and auto tariffs; Chinese coast-guard 'quarantine' drills near Taiwan - a semiconductor-specific gap risk that a market-wide hedge does not cover.</td>
<td>Sector and PCA-factor neutrality keep semis from becoming a net bet; the VIXY / SPY legs of the shock score catch the market-wide part; a news-day filter blocks fading a name that gaps on headlines.</td></tr>
<tr><td><b>Rates: the Fed is hiking</b></td><td>Inflation risks skewed to the upside in the October FOMC minutes; rates-sensitive sectors (utilities, REITs, banks) trade on yields.</td>
<td>Per-stock rates beta (TLT) neutralised; TLT 3-day moves feed the shock score.</td></tr>
<tr><td><b>Concentration / AI</b></td><td>Index driven by a handful of AI beneficiaries; the Mag-7 trade as one position at times.</td>
<td>PCA factors absorb the common AI factor; the model never holds the index, only residuals, and the first PCA factors are neutralised.</td></tr>
<tr><td><b>Decay of the edge itself</b></td><td>Avellaneda &amp; Lee (2008) already saw the Sharpe fall after 2002; on 2017-2026 data the raw signal's net edge is roughly zero (see Backtest, Ablation). The one surviving effect (earnings-announcement premium) is modest and lumpy.</td>
<td>The default sleeve is the earnings premium, not the reversion book; the reversion sleeve stays available at weight 0. Paper trade and size small until live out-of-sample evidence says otherwise.</td></tr>
<tr><td><b>Earnings-gap risk (we hold it on purpose)</b></td><td>The premium is the payment for holding announcements; single-name gaps of 10-20% happen. Q3 earnings season is under way now (late October).</td>
<td>Equal weight, 4% cap inside the sleeve (often far smaller: with 200+ announcers each name is under 0.5%), names with a history of huge gaps (avg &gt; 20%) skipped, every long hedged with beta x sector SPDR, volatility target.</td></tr>
</table>
"""


class SignalsTab(QWidget):
    def __init__(self, settings: SettingsPanel, qs: QSettings):
        super().__init__()
        self.sp, self.qs = settings, qs
        self._workers = WorkerBag()
        self._busy = False
        self._panel = None
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.btn = QPushButton("Refresh signals"); self.btn.setObjectName("primary"); self.btn.clicked.connect(self.refresh)
        self.status = muted("", wrap=False)
        top.addWidget(self.btn); top.addWidget(self.status, 1)
        root.addLayout(top)
        k = QHBoxLayout()
        self.k = {n: Kpi(n) for n in ("Session", "Tradable names", "Long candidates", "Short candidates", "Shock score", "Shock multiplier", "Absorption (top-5)", "Idiosyncratic share")}
        for w in self.k.values():
            k.addWidget(w)
        root.addLayout(k)
        self.banner = strip("", color=ACCENT)
        root.addWidget(self.banner)
        self.tabs = QTabWidget()
        self.t_sig = make_table(["Symbol", "Sector", "s-score", "Reversion (days)", "Resid vol (bp/d)", "Volume x", "Earnings in", "Flag", "Verdict"])
        self.tabs.addTab(self.t_sig, "Signals (|s| ≥ 1)")
        self.t_macro = make_table(["Macro / geopolitical proxy", "1d", "3d", "z (3d)", "21d realised vol"])
        mac = QWidget(); mv = QVBoxLayout(mac)
        mv.addWidget(self.t_macro)
        self.c_shock = LineChart("shock score (sigma)", 2.6)
        mv.addWidget(self.c_shock)
        self.tabs.addTab(mac, "Macro / geopolitical dashboard")
        self.c_s = Histogram(3.0)
        self.tabs.addTab(self.c_s, "s-score distribution")
        env = QTextBrowser(); env.setHtml(ENV_HTML)
        self.tabs.addTab(env, "Risk map 2026")
        root.addWidget(self.tabs, 1)

    def refresh(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.btn.setEnabled(False)
        self.status.setText("computing ...")
        f, p, uni = self.sp.factor_cfg(), self.sp.port_cfg(), self.sp.universe()

        def fn(log):
            panel = BR.load_panel(uni, refresh=False, log=log)
            i = LV.last_complete_session(panel)
            cut = LV._truncate(panel, i)
            ff = FactorCfg(**{**asdict(f), "start": (cut.dates[-1] - pd.Timedelta(days=400)).strftime("%Y-%m-%d")})
            sig = build_signals(cut, ff, earnings=D.load_earnings(), n_jobs=1)
            return panel, cut, sig
        w = FnWorker(fn)
        self._workers.start(w, lambda res, err: self._done(res, err, p), lambda s: self.status.setText(s.strip()[:90]))

    def _done(self, res, err, p) -> None:
        self._busy = False
        self.btn.setEnabled(True)
        if err:
            self.status.setText(f"failed: {err}")
            return
        panel, cut, sig = res
        i = len(cut.dates) - 1
        s = sig.s[i]
        univ = sig.univ[i] & np.isfinite(s)
        self.k["Session"].set(f"{cut.dates[-1]:%Y-%m-%d}")
        self.k["Tradable names"].set(f"{int(univ.sum()):,}")
        rg = sig.regime.iloc[i]
        mult = float(shock_multiplier(sig.regime, p).iloc[i])
        nl, ns = int((univ & (s < -p.s_open)).sum()), int((univ & (s > p.s_open)).sum())
        self.k["Long candidates"].set(str(nl)); self.k["Short candidates"].set(str(ns))
        self.k["Shock score"].set(f"{rg.shock:.2f}σ", RED if rg.shock > 3 else YELLOW if rg.shock > p.shock_z0 else GREEN, "max of SPY↓ VIXY↑ |oil| |TLT| gold↑")
        self.k["Shock multiplier"].set(f"{mult:.2f}", RED if mult < 0.6 else YELLOW if mult < 1 else GREEN)
        self.k["Absorption (top-5)"].set(f"{rg.absorb5:.0%}", None, f"avg corr {rg.avg_corr:.2f}")
        self.k["Idiosyncratic share"].set(f"{rg.idio_share:.0%}", None, "of 60d variance after PCA")
        px = cut.close.iloc[i].to_numpy(float)
        cols = np.array(sig.cols)
        order = np.argsort(-np.nan_to_num(np.abs(s) * univ))
        rows = []
        for j in order[:300]:
            if not univ[j] or abs(s[j]) < 1.0:
                break
            flag = []
            if sig.news[i, j]: flag.append("news")
            if sig.post_earn[i, j]: flag.append("post-earnings")
            if sig.earn_soon[i, j]: flag.append("earnings ≤6d")
            verdict = "—"
            if abs(s[j]) >= p.s_open:
                blocked = (p.event_filter and (sig.news[i, j] or sig.post_earn[i, j])) or (p.earn_filter and sig.earn_soon[i, j])
                side = "LONG" if s[j] < 0 else "SHORT"
                verdict = f"blocked ({'/'.join(flag)})" if blocked else (side + " candidate" if (s[j] < 0 or p.allow_short) else "short disabled")
            k = sig.kappa[i, j]
            rows.append((cols[j], panel.sector.get(cols[j], ""), s[j], 252.0 / k if np.isfinite(k) and k > 0 else np.nan, sig.rvol[i, j] * 1e4, sig.vol_ratio[i, j], ",".join(flag), verdict))
        t = self.t_sig
        t.setRowCount(len(rows))
        for r, (sym, sec, sv, tau, rv, vr, flag, verdict) in enumerate(rows):
            set_row(t, r, [sym, sec, f"{sv:+.2f}", f"{tau:.1f}" if np.isfinite(tau) else "—", f"{rv:.0f}", f"{vr:.1f}" if np.isfinite(vr) else "—", "", flag or "—", verdict],
                    {2: GREEN if sv < 0 else RED, 8: GREEN if "LONG" in verdict else RED if "SHORT" in verdict else None}, align_left=(1, 7, 8))
        # macro dashboard
        ec = cut.etf_close
        r1, r3 = ec.pct_change(fill_method=None).iloc[-1], ec.pct_change(3, fill_method=None).iloc[-1]
        sh = shock_table(cut).iloc[-1]
        zmap = {"SPY": -sh.z_spy, "VIXY": sh.z_vix, "USO": sh.z_oil, "TLT": sh.z_tlt, "GLD": sh.z_gld}
        names = {"SPY": "S&P 500 (SPY)", "VIXY": "Short-term VIX futures (VIXY)", "USO": "Crude oil (USO) - Hormuz", "TLT": "20y Treasuries (TLT) - Fed", "GLD": "Gold (GLD)", "UUP": "U.S. dollar (UUP)",
                 "ITA": "Defense (ITA)", "QQQ": "Nasdaq-100 (QQQ)", "IWM": "Russell 2000 (IWM)", "XLE": "Energy (XLE)", "XLK": "Technology (XLK)"}
        rv21 = ec.pct_change(fill_method=None).rolling(21).std().iloc[-1] * np.sqrt(252)
        tm = self.t_macro
        tm.setRowCount(len(names))
        for r, (sym, nm) in enumerate(names.items()):
            z = zmap.get(sym)
            set_row(tm, r, [nm, pct(r1[sym]), pct(r3[sym]), f"{z:+.2f}" if z is not None and np.isfinite(z) else "—", pct(rv21[sym], 1)], {1: GREEN if r1[sym] > 0 else RED, 2: GREEN if r3[sym] > 0 else RED})
        sht = shock_table(panel).shock.dropna().iloc[-500:]
        self.c_shock.show_series({"shock score": sht}, [p.shock_z0])
        self.c_s.show_hist(s[univ], 80, "s-score, tradable names today")
        reason = "quiet" if mult >= 1 else "de-grossing after a macro shock"
        self.banner.setText(f"{cut.dates[-1]:%A %d %B %Y} · {int(univ.sum())} tradable names · {nl} long / {ns} short candidates at |s|>{p.s_open:g} · shock multiplier {mult:.2f} ({reason})")
        self.status.setText("updated")
