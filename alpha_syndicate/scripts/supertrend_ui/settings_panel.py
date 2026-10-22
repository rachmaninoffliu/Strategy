"""The settings panel on the left: every strategy input, edited once and used by BOTH the Backtest and the Live engine."""
from __future__ import annotations

from PyQt6.QtCore import QSettings, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

import supertrend_strategy as S

from . import params as P
from .theme import YELLOW, QComboBox, QSpinBox, muted, spin, strip, box_style


class SettingsPanel(QWidget):
    """Strategy inputs shared by the Backtest and Live tabs (the SAME values drive both). Inputs the chosen mode does not read are greyed out."""
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._applying = False
        self._locked = False
        self.w: dict = {}                                   # attr -> input widget
        self.rows: dict = {}                                # attr -> (label widget | None, input widget)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(strip("SHARED SETTINGS — every field here is used by BOTH the Backtest and the Live engine, so what you test is what trades. "
                            "Fields that exist for only one side say so; their own boxes are in the Backtest / Live tabs. The tab \"Parameters in use\" lists everything side by side.",
                            color="#3d8bfd", alpha=0.14))
        self.notice = QLabel(""); self.notice.setWordWrap(True); self.notice.setStyleSheet(box_style(YELLOW, 0.16)); self.notice.hide()
        lay.addWidget(self.notice)

        g = QGroupBox("Strategy preset")
        f = QFormLayout(g)
        self.preset = QComboBox()
        self.preset.addItems(list(P.PRESETS) + [P.CUSTOM])
        self.preset.setToolTip("Presets fill in the fields below. The reference is the setup all the research notes are built on. The 'Neighbour check' presets show how fragile the factor is; "
                               "'Stress costs' shows what the edge looks like with heavier costs. Any manual edit turns the label into 'Custom'.")
        self.preset.activated.connect(lambda *_: self.apply_preset(self.preset.currentText()))
        self.preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); self.preset.setMinimumContentsLength(18)
        f.addRow(self.preset)
        lay.addWidget(g)

        forms: dict = {}
        for gname in P.GROUPS:
            box = QGroupBox(gname.replace("&", "&&"))                 # a single '&' in a title is a keyboard-mnemonic marker and would vanish
            forms[gname] = QFormLayout(box)
            lay.addWidget(box)
        for p in P.SPEC:
            wdg = self._make(p)
            self.w[p.attr] = wdg
            tip = p.help + ("\n\nUsed by the BACKTEST only - the live engine does not read it." if p.attr in P.BACKTEST_ONLY and p.attr != "commission_bps" else "")
            wdg.setToolTip(tip)
            if p.kind == "bool":
                wdg.setText(p.label + ("  (backtest only)" if p.attr in P.BACKTEST_ONLY else ""))
                forms[p.group].addRow(wdg)
                self.rows[p.attr] = (None, wdg)
            else:
                lab = QLabel(p.label + ("  (backtest)" if p.attr in P.BACKTEST_ONLY else ""))
                lab.setToolTip(tip)
                forms[p.group].addRow(lab, wdg)
                self.rows[p.attr] = (lab, wdg)
            sig = getattr(wdg, "valueChanged", None) or getattr(wdg, "currentIndexChanged", None) or getattr(wdg, "toggled", None)
            sig.connect(lambda *_: self._on_edit())

        self.btn_reset = QPushButton("Reset everything to the reference settings")
        self.btn_reset.setToolTip("Back to the setup the research notes are built on: Always Fixed, ATR 500, factor 5.0, 100% sizing, 5 bp costs.")
        self.btn_reset.clicked.connect(lambda: self.apply_preset(P.REFERENCE_PRESET))
        lay.addWidget(self.btn_reset)
        self.data_lbl = muted("")
        self.btn_update = QPushButton("Update data")
        self.btn_update.setToolTip("Download the newest 15-minute bars from Alpaca into the local cache (read-only data request) so backtests and the signal include the latest session.")
        lay.addWidget(self.data_lbl)
        lay.addWidget(self.btn_update)
        lay.addStretch(1)
        self.changed.connect(self.refresh_notice)
        self.changed.connect(self._sync_used)
        self.apply_preset(P.REFERENCE_PRESET)

    # ── building ──
    def _make(self, p: P.P):
        if p.kind == "choice":
            c = QComboBox(); c.addItems(list(p.options)); c.setCurrentText(str(p.default))
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon); c.setMinimumContentsLength(14)
            return c
        if p.kind == "int":
            s = QSpinBox(); s.setRange(int(p.lo), int(p.hi)); s.setSingleStep(int(p.step)); s.setValue(int(p.default)); s.setGroupSeparatorShown(True)
            return s
        if p.kind == "bool":
            c = QCheckBox(p.label); c.setChecked(bool(p.default))
            return c
        s = spin(p.lo, p.hi, float(p.default), p.dec, p.step, p.suffix if p.kind != "money" else "")
        if p.kind == "money":
            s.setPrefix("$ "); s.setSpecialValueText("no cap")
        return s

    # ── values ──
    def values(self) -> dict:
        out = {}
        for p in P.SPEC:
            w = self.w[p.attr]
            out[p.attr] = (w.currentText() if p.kind == "choice" else w.isChecked() if p.kind == "bool" else int(w.value()) if p.kind == "int" else float(w.value()))
        return out

    def set_values(self, vals: dict) -> None:
        self._applying = True
        try:
            for p in P.SPEC:
                if p.attr not in vals:
                    continue
                w, v = self.w[p.attr], vals[p.attr]
                if p.kind == "choice":
                    w.setCurrentText(str(v))
                elif p.kind == "bool":
                    w.setChecked(v if isinstance(v, bool) else str(v).lower() in ("true", "1"))
                elif p.kind == "int":
                    w.setValue(int(round(float(v))))
                else:
                    w.setValue(float(v))
        finally:
            self._applying = False
        self._on_edit()

    def to_config(self, **extra) -> S.Config:
        return P.to_config(self.values(), **extra)

    def _matching_preset(self) -> str:
        cur = self.values()
        for name in P.PRESETS:
            ref = P.preset_values(name)
            if all((cur[a] == ref[a]) if isinstance(ref[a], (str, bool)) else abs(float(cur[a]) - float(ref[a])) < 1e-9 for a in ref):
                return name
        return P.CUSTOM

    def apply_preset(self, name: str) -> None:
        vals = P.preset_values(name)
        if vals is None:
            return
        self.set_values(vals)

    def _on_edit(self) -> None:
        if self._applying:
            return
        self.preset.setCurrentText(self._matching_preset())
        self.changed.emit()

    def preset_name(self) -> str:
        return self.preset.currentText()

    # ── feedback ──
    def refresh_notice(self) -> None:
        items = P.notices(self.values())
        self.notice.setText("⚠ Differs from the tested reference or between Backtest and Live: " + "; ".join(items) + "." if items else "")
        self.notice.setVisible(bool(items))

    def _sync_used(self) -> None:
        """Grey out the inputs the chosen mode does not read, so nobody tunes a dead control."""
        use = P.used_flags(self.values())
        for a, (lab, w) in self.rows.items():
            on = use[a] and not self._locked
            w.setEnabled(on)
            if lab is not None:
                lab.setEnabled(use[a])

    def set_enabled_all(self, on: bool) -> None:
        self._locked = not on
        self.preset.setEnabled(on)
        self.btn_reset.setEnabled(on)
        self._sync_used()

    # ── persistence ──
    def save(self, qs: QSettings) -> None:
        qs.setValue("settings/preset", self.preset.currentText())
        for k, v in self.values().items():
            qs.setValue(f"settings/{k}", v)

    def load(self, qs: QSettings) -> None:
        if qs.value("settings/preset") is None:
            return
        vals = {}
        for p in P.SPEC:
            v = qs.value(f"settings/{p.attr}")
            if v is None:
                continue
            try:
                vals[p.attr] = (str(v) if p.kind == "choice" else (str(v).lower() in ("true", "1")) if p.kind == "bool" else float(v))
            except (TypeError, ValueError):
                pass
        if vals.get("strategyMode") not in P.MODES:
            vals.pop("strategyMode", None)
        self.set_values(vals)
