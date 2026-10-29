"""Left-hand settings: every strategy input in one place. The Backtest and the Live engine both read these (what you test is what trades)."""
from __future__ import annotations

import json
from dataclasses import fields
from typing import Optional

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from statarb_core import FactorCfg, PortCfg
from supertrend_ui.theme import QComboBox, QDoubleSpinBox, QSpinBox, muted, strip

from .presets import DEFAULT_PRESET, PRESETS

# (key, label, kind, lo, hi, step, decimals, tooltip)   kind: f float · i int · b bool · c combo(lo = choices)
GROUPS: list = [
    ("Universe & factor model", [
        ("universe", "Universe", "c", ["core", "broad"], None, None, None, "core = ~430 curated large caps (today's constituents - survivorship-biased). broad = every U.S. stock that was ever liquid, INCLUDING delisted ones, filtered "
                                                                           "by price and dollar volume each day with data up to that day (needs the broad cache: python statarb_broad.py)."),
        ("factor_mode", "Systematic factors", "c", ["pca", "etf"], None, None, None, "pca = eigenportfolios of the 252-day correlation matrix (paper 2.1). etf = the stock's own sector SPDR (paper 2.3)."),
        ("n_factors", "PCA factors", "i", 3, 40, 1, 0, "Paper: 15 (or enough to explain ~55% of variance)."),
        ("var_explained", "…or % variance explained (0 = fixed)", "f", 0.0, 0.9, 0.05, 2, "0 uses the fixed count above; 0.55 picks the number of eigenportfolios that explain 55% of the variance on each date (paper 5.4)."),
        ("est_window", "OU estimation window (sessions)", "i", 20, 120, 5, 0, "Paper: 60 (about one earnings cycle)."),
        ("volume_time", "Trading-time returns (volume-weighted)", "b", None, None, None, None, "Paper section 6: scale each day's return by <V>/V."),
        ("kappa_min", "Min mean-reversion speed kappa", "f", 1.0, 40.0, 0.5, 1, "Paper: 252/30 = 8.4 (reversion time under ~1.5 months)."),
        ("min_price", "Min price ($)", "f", 1.0, 50.0, 1.0, 0, "Point-in-time filter."),
        ("min_adv_m", "Min 20-day median $ volume ($M)", "f", 1.0, 500.0, 5.0, 0, "Point-in-time liquidity filter. Costs rise steeply below $50M a day."),
    ]),
    ("Sleeves (what the capital does)", [
        ("w_rev", "Residual-reversion sleeve (Avellaneda-Lee)", "f", 0.0, 1.0, 0.05, 2, "Share of the risk budget in the paper's long/short residual reversion book. On 2017-2026 data its edge after costs is about zero: it is a research / hedge sleeve."),
        ("w_eap", "Earnings-premium sleeve", "f", 0.0, 1.0, 0.05, 2, "Share in the event sleeve: hold the stocks that announce earnings in the next few sessions (the announcement premium), hedged with beta-weighted sector SPDRs. The one "
                                                                    "component with a positive net result in this sample - still modest and noisy (Sharpe standard error ~0.35)."),
        ("eap_lo", "…hold announcers reporting in ≥ N sessions", "i", 1, 10, 1, 0, "Decision at the close; position from the next open. 2 = the gap is inside the first holding day."),
        ("eap_hi", "…and ≤ N sessions", "i", 2, 12, 1, 0, ""),
        ("eap_pool", "…screen with the OU s-score", "c", ["all", "cheap", "notrich"], None, None, None, "all = every announcer; cheap = only names the model calls cheap vs their factors (s < 0); notrich = skip s ≥ 1."),
        ("eap_cap", "…max weight per name inside the sleeve", "f", 0.01, 0.5, 0.01, 2, ""),
        ("eap_max_names", "…max names (0 = all; keeps the cheapest by OU s-score)", "i", 0, 400, 10, 0, "In peak earnings weeks 200-350 companies report; tickets of a few hundred dollars each are not worth trading. 100 loses nothing in the backtest."),
        ("eap_sector_cap", "…max share of the sleeve in one sector", "f", 0.05, 1.0, 0.05, 2, "Above this the excess is left in cash (earnings seasons cluster by sector)."),
        ("eap_max_move", "…skip names whose average earnings gap exceeds", "f", 0.02, 1.0, 0.01, 2, "Average |gap| over the stock's last 8 announcements."),
        ("eap_hedge", "…hedge with sector SPDRs", "b", None, None, None, None, "Short beta x the stock leg in each sector's SPDR."),
    ]),
    ("Entry / exit rules (reversion sleeve)", [
        ("s_open", "Open at |s| >", "f", 0.5, 4.0, 0.05, 2, "Paper: 1.25."),
        ("s_close_long", "Close longs when s > -", "f", 0.0, 2.0, 0.05, 2, "Paper: 0.50."),
        ("s_close_short", "Close shorts when s < +", "f", 0.0, 2.0, 0.05, 2, "Paper: 0.75."),
        ("s_stop", "Model-failure stop |s| >", "f", 2.0, 8.0, 0.25, 2, "If the residual keeps running away the OU model is wrong for that stock: exit."),
        ("max_hold", "Max holding (sessions)", "i", 3, 60, 1, 0, "Time stop (about 2-3 reversion half-lives)."),
        ("max_pos_per_side", "Max positions per side", "i", 5, 200, 5, 0, ""),
        ("vol_boost", "Volume boost (fade heavy-volume moves harder)", "f", 0.0, 3.0, 0.25, 2, "Entry score = s x (1 + boost x ln(volume ratio)). The 2008 paper believed heavy-volume moves (fade less); the 2017-2026 data say they revert MORE."),
        ("allow_short", "Trade the short side in single stocks", "b", None, None, None, None, "Off = long-only fade book (hedge it with sector ETFs below)."),
        ("hedge", "Hedge", "c", ["none", "etf"], None, None, None, "etf = short beta-weighted sector SPDRs against the stock book (market + sector neutral without shorting single names)."),
    ]),
    ("Idiosyncratic & macro risk", [
        ("event_filter", "No fading of news / post-earnings sessions", "b", None, None, None, None, "News = volume >= 2.5x its 60-session median (relative to the market that day) or an overnight gap beyond 4 sigma."),
        ("earn_filter", "Earnings window filter", "b", None, None, None, None, "No entry when earnings are scheduled within 6 sessions; close the position before the announcement. Dates from the Yahoo earnings calendar."),
        ("neutral", "Factor / sector neutrality (projection)", "b", None, None, None, None, "Project the book onto the null space of the exposures below."),
        ("n_pca_neutral", "…neutral to the top N PCA factors", "i", 0, 15, 1, 0, ""),
        ("style_neutral", "…and to momentum / volatility / size", "b", None, None, None, None, "Quant crowding showed up as style unwinds in Jan / Mar / Jun-Jul 2026."),
        ("macro_neutral", "…and to oil / rates / USD betas", "b", None, None, None, None, "Hormuz and tariff headlines move oil; Fed hiking moves rates. Betas are estimated per stock over 126 sessions."),
        ("shock", "Geopolitical shock multiplier", "b", None, None, None, None, "De-gross after a 3-day move beyond z0 sigma in SPY (down), VIXY (up), oil or Treasuries (either way) or gold (up)."),
        ("shock_z0", "…starts above z =", "f", 0.5, 6.0, 0.25, 2, ""),
        ("shock_k", "…de-gross per sigma", "f", 0.0, 1.0, 0.05, 2, ""),
        ("shock_min", "…never below", "f", 0.0, 1.0, 0.05, 2, ""),
    ]),
    ("Sizing & costs", [
        ("weight_pow", "Size by 1/(residual vol)^p, p =", "f", 0.0, 2.0, 0.5, 1, "0 = equal dollars (paper), 1 = equal risk."),
        ("name_cap", "Max weight per name (of gross)", "f", 0.01, 1.0, 0.01, 2, ""),
        ("vol_on", "Volatility targeting", "b", None, None, None, None, "Off = fixed gross exposure (the next box). On = gross floats so that the strategy's own EWMA volatility is the target below."),
        ("vol_target", "…target volatility (annual)", "f", 0.01, 0.5, 0.01, 2, "EWMA of the strategy's own unit-gross P&L, only using P&L known before each decision."),
        ("g_max", "Gross leverage (fixed, or the cap)", "f", 0.25, 4.0, 0.25, 2, "Reg-T overnight limit is 2.0x equity (long + short)."),
        ("band", "No-trade band (share of position)", "f", 0.0, 1.0, 0.05, 2, "Continuing positions are only re-traded if the target moved by more than this."),
        ("tc_bps", "One-way cost, large caps (bp)", "f", 0.0, 20.0, 0.5, 1, "Scaled up automatically for less liquid names (x1.7 below $200M a day, x2.7 below $50M, x4 below $20M). Paper: 5. Auction fills: 1-2."),
        ("borrow_bps", "Short borrow (bp / year)", "f", 0.0, 500.0, 10.0, 0, "Easy-to-borrow names are ~0-50 bp."),
    ]),
]

FACTOR_KEYS = {f.name for f in fields(FactorCfg)}
PORT_KEYS = {f.name for f in fields(PortCfg)}


class SettingsPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.w: dict = {}
        root = QVBoxLayout(self)
        root.addWidget(strip("Strategy inputs - shared by the Backtest and the Live engine.", alpha=0.10))
        self.preset = QComboBox()
        self.preset.addItem("(custom)")
        self.preset.addItems(list(PRESETS))
        self.preset.setToolTip("A preset only fills the boxes below - nothing is hidden.")
        self.preset.activated.connect(lambda *_: self.apply_preset(self.preset.currentText()))
        row = QHBoxLayout(); row.addWidget(QLabel("Preset")); row.addWidget(self.preset, 1)
        root.addLayout(row)
        self.preset_note = muted("")
        root.addWidget(self.preset_note)
        for title, rows in GROUPS:
            g = QGroupBox(title)
            f = QFormLayout(g)
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for key, label, kind, lo, hi, step, dec, tip in rows:
                if kind == "f":
                    w = QDoubleSpinBox(); w.setRange(lo, hi); w.setSingleStep(step); w.setDecimals(dec)
                    w.valueChanged.connect(self._edited)
                elif kind == "i":
                    w = QSpinBox(); w.setRange(lo, hi); w.setSingleStep(step)
                    w.valueChanged.connect(self._edited)
                elif kind == "b":
                    w = QCheckBox(); w.toggled.connect(self._edited)
                else:
                    w = QComboBox(); w.addItems(lo); w.currentIndexChanged.connect(self._edited)
                if tip:
                    w.setToolTip(tip)
                self.w[key] = w
                lb = QLabel(label); lb.setToolTip(tip)
                f.addRow(lb, w)
            root.addWidget(g)
        self.btn_update = QPushButton("Update data (Alpaca + earnings)")
        self.btn_update.setToolTip("Top up the daily bars from Alpaca (read-only data requests) and the Yahoo earnings calendar.")
        self.data_lbl = muted("")
        root.addWidget(self.btn_update); root.addWidget(self.data_lbl)
        root.addStretch(1)
        self._silent = False
        self.apply_preset(DEFAULT_PRESET)

    # ── values ───────────────────────────────────────────────────────────────────
    def _edited(self, *_):
        if not self._silent:
            self.preset.setCurrentIndex(0)
            self.changed.emit()

    def values(self) -> dict:
        out = {}
        for k, w in self.w.items():
            out[k] = w.isChecked() if isinstance(w, QCheckBox) else w.currentText() if isinstance(w, QComboBox) else w.value()
        return out

    def apply_values(self, vals: dict, universe: Optional[str] = None) -> None:
        self._silent = True
        try:
            v = dict(vals)
            if universe:
                v["universe"] = universe
            if "min_adv" in v:
                v["min_adv_m"] = v.pop("min_adv") / 1e6
            for k, x in v.items():
                w = self.w.get(k)
                if w is None:
                    continue
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(x))
                elif isinstance(w, QComboBox):
                    i = w.findText(str(x))
                    if i >= 0:
                        w.setCurrentIndex(i)
                else:
                    w.setValue(float(x) if hasattr(w, "decimals") and w.decimals() else int(x))
        finally:
            self._silent = False
        self.changed.emit()

    def set_values(self, vals: dict) -> None:
        self.apply_values(vals)
        self.preset.setCurrentIndex(0)

    def apply_preset(self, name: str) -> None:
        p = PRESETS.get(name)
        if not p:
            return
        base = {**self._defaults(), **p["values"]}
        self.apply_values(base)
        self.preset.setCurrentText(name)
        self.preset_note.setText(p.get("note", ""))

    def _defaults(self) -> dict:
        d = {f.name: getattr(FactorCfg(), f.name) for f in fields(FactorCfg)}
        d.update({f.name: getattr(PortCfg(), f.name) for f in fields(PortCfg)})
        d["var_explained"] = 0.0
        d["vol_on"] = True
        d.pop("min_adv", None)                                           # the box is in $M (min_adv_m); keeping both would let the default overwrite a preset's value
        d["min_adv_m"] = FactorCfg().min_adv / 1e6
        return d

    def preset_name(self) -> str:
        return self.preset.currentText()

    # ── objects ──────────────────────────────────────────────────────────────────
    def universe(self) -> str:
        return self.w["universe"].currentText()

    def factor_cfg(self) -> FactorCfg:
        v = self.values()
        kw = {k: v[k] for k in FACTOR_KEYS if k in v}
        kw["var_explained"] = v["var_explained"] or None
        kw["min_adv"] = v["min_adv_m"] * 1e6
        kw["n_factors"] = int(v["n_factors"])
        kw["est_window"] = int(v["est_window"])
        return FactorCfg(**{**{f.name: getattr(FactorCfg(), f.name) for f in fields(FactorCfg)}, **kw})

    def port_cfg(self, start: str = "2017-06-01", end: Optional[str] = None, capital: float = 100_000.0) -> PortCfg:
        v = self.values()
        kw = {k: v[k] for k in PORT_KEYS if k in v}
        for k in ("max_hold", "max_pos_per_side", "n_pca_neutral", "eap_lo", "eap_hi", "eap_max_names"):
            kw[k] = int(v[k])
        kw.update(start=start, end=end, capital=capital)
        if not v["vol_on"]:                                         # fixed gross: the leverage box is the gross
            kw.update(g_init=v["g_max"], g_min=v["g_max"])
        return PortCfg(**{**{f.name: getattr(PortCfg(), f.name) for f in fields(PortCfg)}, **kw})

    def save(self, qs: QSettings) -> None:
        qs.setValue("settings/values", json.dumps(self.values()))

    def load(self, qs: QSettings) -> None:
        try:
            raw = qs.value("settings/values")
            if raw:
                self.apply_values(json.loads(raw))
                self.preset.setCurrentIndex(0)
        except (ValueError, TypeError):
            pass
