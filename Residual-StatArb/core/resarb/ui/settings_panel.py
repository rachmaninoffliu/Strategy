"""Left-hand settings, built from the strategy's parameter list (strategy-specific ones first, then the shared universe / trading / portfolio / cost settings) plus its presets."""
from __future__ import annotations

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..strategy import Param, Strategy
from .theme import QComboBox, QDoubleSpinBox, QSpinBox, muted


class SettingsPanel(QWidget):
    def __init__(self, strat_cls: type):
        super().__init__()
        self.cls = strat_cls
        self.w: dict = {}
        root = QVBoxLayout(self)
        g = QGroupBox(f"{strat_cls.NAME} - preset"); h = QHBoxLayout(g)
        self.preset = QComboBox(); self.preset.addItems(["(custom)"] + list(strat_cls.PRESETS))
        b = QPushButton("Apply"); b.clicked.connect(self.apply_preset)
        h.addWidget(self.preset, 1); h.addWidget(b)
        root.addWidget(g)
        groups: dict = {}
        for p in strat_cls.all_params():
            if p.group not in groups:
                box = QGroupBox(p.group if p.group != "Strategy" else f"{strat_cls.NAME} signal")
                groups[p.group] = QFormLayout(box)
                root.addWidget(box)
            groups[p.group].addRow(self._label(p), self._widget(p))
        d = QGroupBox("Data"); dv = QVBoxLayout(d)
        self.data_lbl = muted("")
        self.btn_update = QPushButton("Update data (universe + 15-min bars)")
        dv.addWidget(self.data_lbl); dv.addWidget(self.btn_update)
        root.addWidget(d)
        root.addStretch(1)
        self.set_values({p.key: strat_cls.DEFAULTS.get(p.key, p.default) for p in strat_cls.all_params()})

    def _label(self, p: Param) -> QLabel:
        lb = QLabel(p.label)
        lb.setToolTip(p.help)
        lb.setWordWrap(True)
        return lb

    def _widget(self, p: Param):
        if p.kind == "float":
            w = QDoubleSpinBox(); w.setRange(p.lo, p.hi); w.setDecimals(p.dec); w.setSingleStep(p.step)
        elif p.kind == "int":
            w = QSpinBox(); w.setRange(int(p.lo), int(p.hi)); w.setSingleStep(max(int(p.step), 1))
        elif p.kind == "choice":
            w = QComboBox(); w.addItems([str(c) for c in p.choices])
        elif p.kind == "bool":
            w = QCheckBox()
        else:
            w = QLineEdit()
        w.setToolTip(p.help)
        self.w[p.key] = (p, w)
        return w

    # ── values ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def values(self) -> dict:
        out = {}
        for k, (p, w) in self.w.items():
            if p.kind == "float":
                out[k] = float(w.value())
            elif p.kind == "int":
                out[k] = int(w.value())
            elif p.kind == "choice":
                out[k] = w.currentText()
            elif p.kind == "bool":
                out[k] = bool(w.isChecked())
            else:
                out[k] = w.text().strip()
        return out

    def set_values(self, vals: dict) -> None:
        for k, v in vals.items():
            if k not in self.w:
                continue
            p, w = self.w[k]
            try:
                if p.kind == "float":
                    w.setValue(float(v))
                elif p.kind == "int":
                    w.setValue(int(float(v)))
                elif p.kind == "choice":
                    i = w.findText(str(v))
                    if i >= 0:
                        w.setCurrentIndex(i)
                elif p.kind == "bool":
                    w.setChecked(v in (True, "true", "True", 1, "1"))
                else:
                    w.setText(str(v))
            except (TypeError, ValueError):
                pass

    def strategy(self, folder: str) -> Strategy:
        return self.cls(self.values(), folder)

    def apply_preset(self) -> None:
        name = self.preset.currentText()
        if name in self.cls.PRESETS:
            self.set_values(self.cls.PRESETS[name])

    def preset_name(self) -> str:
        return self.preset.currentText()

    def save(self, qs: QSettings) -> None:
        for k, v in self.values().items():
            qs.setValue(f"param/{k}", v)

    def load(self, qs: QSettings) -> None:
        vals = {}
        for k in self.w:
            v = qs.value(f"param/{k}")
            if v is not None:
                vals[k] = v
        self.set_values(vals)
