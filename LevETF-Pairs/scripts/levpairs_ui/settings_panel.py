"""Left-hand settings: every strategy input in one place. The Backtest and the Live engine both read these (what you test is what trades)."""
from __future__ import annotations

import json
from dataclasses import fields
from typing import Optional

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from levpairs_core import SigCfg
from levpairs_sim import SimCfg
from supertrend_ui.theme import QComboBox, QDoubleSpinBox, QSpinBox, muted, strip

from .presets import DEFAULT_PRESET, PRESETS

# (key, label, kind, lo, hi, step, decimals, tooltip)   kind: f float · i int · b bool · c combo(lo = choices)
GROUPS: list = [
    ("Time frame & spread model", [
        ("tf", "Bar size (minutes)", "c", ["1", "5", "15", "30"], None, None, None, "Every other number below is in minutes or sessions, so the same rules run on any bar size. 1-minute bars have the highest gross edge per trade and the "
                                                                                    "most noise; 30-minute bars the least of both. The Backtest tab's 'Time frames' scan compares all four."),
        ("spread", "Spread definition", "c", ["reset", "chain"], None, None, None, "reset = premium of the 3x ETF to 3x its 1x ETF SINCE THE PREVIOUS CLOSE (exact for a daily-reset ETF). chain = the paper's cumulative bar residual, overnight excluded."),
        ("beta_mode", "Hedge ratio", "c", ["fixed", "rolling"], None, None, None, "fixed = 3.0 (what the fund promises). rolling = regression of the last N sessions, shrunk toward 3.0."),
        ("beta_days", "…rolling window (sessions)", "i", 5, 60, 5, 0, ""),
        ("beta_shrink", "…weight on the nominal 3.0", "f", 0.0, 1.0, 0.1, 2, ""),
        ("win_days", "OU estimation window (sessions)", "f", 1.0, 20.0, 0.5, 1, "Paper: 60 days of daily data. Here the window is in sessions of intraday bars."),
        ("s_mode", "Score", "c", ["ou", "z"], None, None, None, "ou = AR(1) fit of the paper (needs a reverting fit); z = plain rolling z-score of the spread."),
        ("feed", "Data feed (backtest)", "c", ["sip", "iex"], None, None, None, "sip = consolidated tape (the research data, 2016+). iex = the only feed this account may read in real time (history from 2020-07, thin in the 1x/3x semiconductor ETFs)."),
    ]),
    ("Pairs", [
        ("w_tqqq", "TQQQ / QQQ weight", "f", 0.0, 1.0, 0.25, 2, "0 switches the pair off. Cheapest pair to trade (about 4.6 bp round trip)."),
        ("w_soxl", "SOXL / SMH weight", "f", 0.0, 1.0, 0.25, 2, "Different indexes (ICE vs MVIS) - the most basis risk and the widest spreads (about 11 bp round trip)."),
        ("w_spxl", "SPXL / SPY weight", "f", 0.0, 1.0, 0.25, 2, "About 5.7 bp round trip."),
        ("weight_mode", "Weighting", "c", ["equal", "inv_vol"], None, None, None, "inv_vol = by the inverse of each pair's trailing 60-session volatility."),
    ]),
    ("Entry / exit rules", [
        ("s_open", "Open at |s| >", "f", 0.5, 5.0, 0.25, 2, "Paper: 1.25. Long the spread (long 3x ETF, short beta x 1x ETF) when s < -open; short the spread when s > +open."),
        ("s_close_long", "Close longs when s > -", "f", 0.0, 2.0, 0.05, 2, "Paper: 0.50."),
        ("s_close_short", "Close shorts when s < +", "f", 0.0, 2.0, 0.05, 2, "Paper: 0.75."),
        ("s_stop", "Model-failure stop |s| >", "f", 2.0, 99.0, 0.5, 1, "If the spread runs away the OU model is wrong: exit and wait out the cooldown."),
        ("max_hold_min", "Max holding (minutes)", "i", 5, 10000, 5, 0, "Time stop."),
        ("max_halflife_min", "Entry needs half-life under (min, 0 = off)", "f", 0.0, 600.0, 5.0, 0, "The fitted reversion half-life must be short enough to complete inside the holding limit."),
        ("min_edge_ratio", "Expected capture ≥ x round-trip cost (0 = off)", "f", 0.0, 5.0, 0.25, 2, "Entry only if (|s| - close level) x sigma_eq is at least this multiple of the pair's round-trip cost."),
        ("skip_open_min", "No entry in the first (min)", "i", 0, 120, 5, 0, "Opening prints are noisy and spreads are widest in the first minutes."),
        ("no_entry_min", "No entry in the last (min)", "i", 0, 120, 5, 0, ""),
        ("flat_min", "Flat by (min before the close)", "i", 0, 60, 5, 0, "Positions are never held overnight."),
        ("cooldown_min", "Cooldown after a stop (min)", "i", 0, 240, 5, 0, ""),
    ]),
    ("Execution realism & risk", [
        ("fill", "Fill model", "c", ["next_open", "mid", "close"], None, None, None, "next_open = the order fills at the open of the bar after the signal (realistic for a market order sent seconds after the bar close). mid = half way between the signal bar's last trade and the next open. close = fills at the signal bar's last trade, a stale print that no order can get - it collects the snap-back of the spread and overstates the edge (see the Fill models tab)."),
        ("delay_bars", "Extra latency (bars)", "i", 0, 5, 1, 0, "Act on the score of N bars ago."),
        ("slip_bp", "Slippage per leg, one way (bp)", "f", 0.0, 5.0, 0.1, 2, "On top of half the quoted spread, applied to BOTH legs (the hedge leg is beta times larger)."),
        ("cost_mult", "Cost multiplier", "f", 0.0, 5.0, 0.25, 2, "0 = gross edge (no costs at all)."),
        ("move_z", "No entries when the index has moved more than z sigma (0 = off)", "f", 0.0, 8.0, 0.5, 1, "Leveraged ETFs dislocate on violent days."),
        ("day_stop_bp", "Stop a pair for the day after losing (bp, 0 = off)", "f", 0.0, 500.0, 10.0, 0, ""),
    ]),
    ("Sizing", [
        ("vol_on", "Volatility targeting", "b", None, None, None, None, "Off = fixed gross exposure (the next box). On = gross floats so the strategy's own EWMA volatility is the target below."),
        ("vol_target", "…target volatility (annual)", "f", 0.01, 0.5, 0.01, 2, ""),
        ("g_init", "Gross (fixed, or the start)", "f", 0.1, 4.0, 0.25, 2, "Gross = long leg + hedge leg, x equity. Alpaca margin gives 4x intraday."),
        ("g_max", "Gross cap", "f", 0.25, 4.0, 0.25, 2, ""),
    ]),
]

SIG_KEYS = {f.name for f in fields(SigCfg)}
SIM_KEYS = {f.name for f in fields(SimCfg)}
INT_KEYS = ("max_hold_min", "skip_open_min", "no_entry_min", "flat_min", "cooldown_min", "delay_bars", "beta_days")


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
                lb = QLabel(label); lb.setToolTip(tip); lb.setWordWrap(True)
                f.addRow(lb, w)
            root.addWidget(g)
        self.btn_update = QPushButton("Update data (Alpaca, 1-minute bars)")
        self.btn_update.setToolTip("Top up the cached 1-minute bars of the six ETFs from Alpaca (read-only data requests).")
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

    def apply_values(self, vals: dict) -> None:
        self._silent = True
        try:
            for k, x in dict(vals).items():
                w = self.w.get(k)
                if w is None:
                    continue
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(x))
                elif isinstance(w, QComboBox):
                    i = w.findText(str(int(x)) if k == "tf" else str(x))
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
        d = {f.name: getattr(SigCfg(), f.name) for f in fields(SigCfg)}
        d.update({f.name: getattr(SimCfg(), f.name) for f in fields(SimCfg) if f.name not in ("pair_w", "start", "end", "capital")})
        d.update(w_tqqq=1.0, w_soxl=1.0, w_spxl=1.0, feed="sip")
        return d

    def preset_name(self) -> str:
        return self.preset.currentText()

    # ── objects ──────────────────────────────────────────────────────────────────
    def feed(self) -> str:
        return self.w["feed"].currentText()

    def tf(self) -> int:
        return int(self.w["tf"].currentText())

    def sig_cfg(self) -> SigCfg:
        v = self.values()
        kw = {k: v[k] for k in SIG_KEYS if k in v}
        kw["tf"] = int(v["tf"])
        kw["beta_days"] = int(v["beta_days"])
        return SigCfg(**kw)

    def sim_cfg(self, start: str = "2017-01-01", end: Optional[str] = None, capital: float = 100_000.0) -> SimCfg:
        v = self.values()
        kw = {k: v[k] for k in SIM_KEYS if k in v and k not in ("pair_w",)}
        for k in INT_KEYS:
            if k in kw:
                kw[k] = int(kw[k])
        kw["pair_w"] = (v["w_tqqq"], v["w_soxl"], v["w_spxl"])
        kw.update(start=start, end=end, capital=capital)
        if not v["vol_on"]:
            kw["g_min"] = v["g_init"]
        return SimCfg(**kw)

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
