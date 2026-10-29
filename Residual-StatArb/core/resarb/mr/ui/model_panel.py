"""The model picker: a preset or every field of an MRModel by hand. The SAME model object drives the Backtest tab and the Live tab (the live loop locks this panel while it runs)."""
from __future__ import annotations

import json

from PyQt6.QtCore import QDate, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from ...ui.theme import QComboBox, QDateEdit, QSpinBox, YELLOW, muted, spin, strip
from .. import presets as PR
from ..models import MRModel

SIGNALS = ["ibs", "rsi2", "rsi3", "rsi5", "ret1", "ret2", "ret3", "ret5", "ret10", "z5", "z10", "z20", "gap", "intraday", "sma_dist"]
UNIVERSES = ["top50+etf", "top100+etf", "top50", "top100", "etf", "list:SPY,QQQ,IWM,DIA,TLT,GLD"]
MINUTES = [("15:30 (free delayed SIP)", 1530), ("15:45 (needs a real-time feed)", 1545), ("15:00", 1500), ("14:00", 1400)]
CUSTOM = "Custom (edit the fields)"


def _combo(items: list, editable: bool = False) -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    c.setEditable(editable)
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(4)
    return c


def _ispin(lo: int, hi: int, v: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(v)
    return s


class ModelPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        self._quiet = False
        root = QVBoxLayout(self)
        root.addWidget(strip("Pick a model. The backtest runs it, and the live engine trades exactly this model (same code: decide()). Long-only presets, no leverage: "
                             "at most 'slots' positions, each 1 / slots of the equity (minus the cash reserve).", YELLOW))
        g = QGroupBox("Model"); f = QFormLayout(g)
        self.preset = _combo(PR.preset_names() + [CUSTOM])
        self.preset.currentTextChanged.connect(self._preset)
        self.note = muted("")
        self.name = QLineEdit()
        f.addRow("Preset", self.preset); f.addRow(self.note); f.addRow("Name", self.name)
        root.addWidget(g)
        g = QGroupBox("Universe and timing"); f = QFormLayout(g)
        self.universe = _combo(UNIVERSES, True)
        self.universe.setToolTip("top50 / top100 = the N most liquid U.S. stocks of each month (survivorship-free); etf = 59 ETFs across assets; list:A,B,C = your own list; "
                                 "join parts with + (top50+etf)")
        self.minute = _combo([m[0] for m in MINUTES])
        self.minute.setToolTip("The decision uses the 15-minute bars that ended by this time; orders go into the closing auction (market-on-close, sent before 15:50).")
        self.direction = _combo(["long", "short"])
        f.addRow("Universe", self.universe); f.addRow("Decision time", self.minute); f.addRow("Direction", self.direction)
        root.addWidget(g)
        g = QGroupBox("Entry"); f = QFormLayout(g)
        self.entry_signal = _combo(SIGNALS, True)
        self.entry_op = _combo(["<", ">", "<=", ">="])
        self.entry_value = spin(-1000, 1000, 0.2, 4, 0.05)
        self.trend_n = _ispin(0, 400, 200); self.trend_n.setToolTip("> 0: a long needs the price above its N-session mean (0 = off)")
        self.entry2_signal = _combo(["(none)"] + SIGNALS, True)
        self.entry2_op = _combo(["<", ">", "<=", ">="])
        self.entry2_value = spin(-1000, 1000, 0.0, 4, 0.05)
        h = QHBoxLayout(); h.addWidget(self.entry_signal, 2); h.addWidget(self.entry_op, 1); h.addWidget(self.entry_value, 2)
        h2 = QHBoxLayout(); h2.addWidget(self.entry2_signal, 2); h2.addWidget(self.entry2_op, 1); h2.addWidget(self.entry2_value, 2)
        f.addRow("Enter when", h); f.addRow("…and (optional)", h2); f.addRow("Trend filter (days)", self.trend_n)
        root.addWidget(g)
        g = QGroupBox("Exit"); f = QFormLayout(g)
        self.hold = _combo(["rule", "overnight"]); self.hold.setToolTip("rule = the exit rules below (checked at each decision); overnight = sell at the next open")
        self.exit_signal = _combo(["(none)"] + SIGNALS, True)
        self.exit_op = _combo([">", "<", ">=", "<="])
        self.exit_value = spin(-1000, 1000, 0.7, 4, 0.05)
        self.exit_sma_n = _ispin(0, 200, 5); self.exit_sma_n.setToolTip("> 0: also leave when the price is above its N-session mean (0 = off)")
        self.max_hold = _ispin(1, 250, 10); self.max_hold.setToolTip("time stop in sessions")
        h = QHBoxLayout(); h.addWidget(self.exit_signal, 2); h.addWidget(self.exit_op, 1); h.addWidget(self.exit_value, 2)
        f.addRow("Holding", self.hold); f.addRow("Exit when", h); f.addRow("…or above N-day mean", self.exit_sma_n); f.addRow("Time stop (days)", self.max_hold)
        root.addWidget(g)
        g = QGroupBox("Capital allocation (no leverage)"); f = QFormLayout(g)
        self.slots = _ispin(1, 100, 10); self.slots.setToolTip("at most this many positions; each gets (1 - cash reserve) / slots of the equity")
        self.rank_by = _combo(["signal", "vol_low", "vol_high"])
        self.rank_by.setToolTip("More candidates than free slots: signal = the most extreme reading first; vol_low = the calmest names first; vol_high = the most volatile first")
        self.cash_reserve = spin(0.0, 0.2, 0.02, 3, 0.005)
        self.cash_reserve.setToolTip("kept in cash so a close above the decision price cannot push the book over 1x")
        self.sizing = _combo(["equal", "inv_vol"])
        self.sizing.setToolTip("equal = every slot the same $; inv_vol = a slot x min(max, vol target / the name's 20-day volatility): calm names (bonds, gold) get more $, "
                               "volatile ones less. The book never exceeds 1x either way.")
        self.vol_target = spin(0.001, 0.1, 0.015, 3, 0.001); self.vol_target.setToolTip("inv_vol: the daily volatility a full slot is sized for")
        self.max_mult = spin(1.0, 5.0, 2.0, 1, 0.5); self.max_mult.setToolTip("inv_vol: a calm name gets at most this many slots' worth")
        f.addRow("Slots (max positions)", self.slots); f.addRow("Rank candidates by", self.rank_by); f.addRow("Position sizing", self.sizing)
        h3 = QHBoxLayout(); h3.addWidget(self.vol_target); h3.addWidget(self.max_mult)
        f.addRow("inv_vol: target / max x", h3); f.addRow("Cash reserve", self.cash_reserve)
        f.addRow(muted("Max gross = 1.00x equity (fixed: no leverage)."))
        root.addWidget(g)
        g = QGroupBox("Costs and filters"); f = QFormLayout(g)
        self.cost = spin(0, 50, 1.0, 2, 0.5, " bp"); self.cost.setToolTip("per side, opening / closing auction")
        self.borrow = spin(0, 2000, 30, 0, 10, " bp/yr")
        self.min_price = spin(0, 1000, 5, 2, 1)
        f.addRow("Auction cost per side", self.cost); f.addRow("Borrow (shorts)", self.borrow); f.addRow("Min price $", self.min_price)
        root.addWidget(g)
        g = QGroupBox("Backtest"); f = QFormLayout(g)
        self.start = QDateEdit(); self.start.setCalendarPopup(True); self.start.setDisplayFormat("yyyy-MM-dd"); self.start.setDate(QDate(2016, 9, 1))
        self.split = QDateEdit(); self.split.setCalendarPopup(True); self.split.setDisplayFormat("yyyy-MM-dd"); self.split.setDate(QDate(2023, 1, 1))
        self.split.setToolTip("in-sample before, out-of-sample from this date - the presets were chosen on in-sample data only")
        self.capital = spin(1_000, 1e9, 100_000, 0, 10_000); self.capital.setPrefix("$ ")
        f.addRow("Start", self.start); f.addRow("Out-of-sample from", self.split); f.addRow("Capital", self.capital)
        root.addWidget(g)
        h = QHBoxLayout()
        b1 = QPushButton("Save model…"); b1.clicked.connect(self.save_file)
        b2 = QPushButton("Load model…"); b2.clicked.connect(self.load_file)
        h.addWidget(b1); h.addWidget(b2)
        root.addLayout(h)
        self.data_lbl = muted("")
        self.btn_update = QPushButton("Update data (daily + 15-minute bars)")
        root.addWidget(self.btn_update); root.addWidget(self.data_lbl)
        root.addStretch(1)
        for w in (self.name, ):
            w.textEdited.connect(self._edited)
        for w in (self.universe, self.minute, self.direction, self.entry_signal, self.entry_op, self.entry2_signal, self.entry2_op, self.hold, self.exit_signal, self.exit_op,
                  self.rank_by, self.sizing):
            w.currentTextChanged.connect(self._edited)
        for w in (self.entry_value, self.entry2_value, self.trend_n, self.exit_value, self.vol_target, self.max_mult, self.exit_sma_n, self.max_hold, self.slots, self.cash_reserve, self.cost, self.borrow, self.min_price):
            w.valueChanged.connect(self._edited)
        self._preset(self.preset.currentText())

    # ── model <-> fields ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def model(self) -> MRModel:
        mi = self.minute.currentIndex()
        ex = self.exit_signal.currentText().strip()
        e2 = self.entry2_signal.currentText().strip()
        return MRModel(name=self.name.text().strip() or "custom", universe=self.universe.currentText().strip(), minute=MINUTES[mi][1] if mi >= 0 else 1530,
                       direction=self.direction.currentText(), entry_signal=self.entry_signal.currentText().strip(), entry_op=self.entry_op.currentText(),
                       entry_value=float(self.entry_value.value()), entry2_signal="" if e2 in ("", "(none)") else e2, entry2_op=self.entry2_op.currentText(),
                       entry2_value=float(self.entry2_value.value()), trend_n=int(self.trend_n.value()), hold=self.hold.currentText(),
                       exit_signal="" if ex in ("", "(none)") else ex, exit_op=self.exit_op.currentText(), exit_value=float(self.exit_value.value()),
                       exit_sma_n=int(self.exit_sma_n.value()), max_hold=int(self.max_hold.value()), slots=int(self.slots.value()), rank_by=self.rank_by.currentText(),
                       sizing=self.sizing.currentText(), vol_target=float(self.vol_target.value()), max_slot_mult=float(self.max_mult.value()), max_gross=1.0, cash_reserve=float(self.cash_reserve.value()), cost_auction_bp=float(self.cost.value()), borrow_bp_yr=float(self.borrow.value()),
                       min_price=float(self.min_price.value()), note=self.note.text())

    def set_model(self, m: MRModel) -> None:
        self._quiet = True
        try:
            self.name.setText(m.name)
            self.universe.setCurrentText(m.universe)
            mins = [x[1] for x in MINUTES]
            if m.minute not in mins:
                MINUTES.append((f"{m.minute // 100}:{m.minute % 100:02d}", m.minute))
                self.minute.addItem(MINUTES[-1][0])
            self.minute.setCurrentIndex([x[1] for x in MINUTES].index(m.minute))
            self.direction.setCurrentText(m.direction)
            self.entry_signal.setCurrentText(m.entry_signal); self.entry_op.setCurrentText(m.entry_op); self.entry_value.setValue(m.entry_value)
            self.entry2_signal.setCurrentText(m.entry2_signal or "(none)"); self.entry2_op.setCurrentText(m.entry2_op); self.entry2_value.setValue(m.entry2_value)
            self.trend_n.setValue(m.trend_n); self.hold.setCurrentText(m.hold)
            self.exit_signal.setCurrentText(m.exit_signal or "(none)"); self.exit_op.setCurrentText(m.exit_op); self.exit_value.setValue(m.exit_value)
            self.exit_sma_n.setValue(m.exit_sma_n); self.max_hold.setValue(m.max_hold); self.slots.setValue(m.slots); self.rank_by.setCurrentText(m.rank_by)
            self.sizing.setCurrentText(m.sizing); self.vol_target.setValue(m.vol_target); self.max_mult.setValue(m.max_slot_mult)
            self.cash_reserve.setValue(m.cash_reserve); self.cost.setValue(m.cost_auction_bp); self.borrow.setValue(m.borrow_bp_yr); self.min_price.setValue(m.min_price)
            self.note.setText(m.note)
        finally:
            self._quiet = False
        self.changed.emit()

    def _preset(self, name: str) -> None:
        if name in PR.PRESETS:
            self.set_model(PR.get(name))

    def _edited(self, *_) -> None:
        if self._quiet:
            return
        if self.preset.currentText() != CUSTOM:
            p = self.preset.currentText()
            if p in PR.PRESETS and PR.get(p).to_dict() != {**self.model().to_dict(), "note": PR.get(p).note}:
                self.preset.blockSignals(True); self.preset.setCurrentText(CUSTOM); self.preset.blockSignals(False)
                self.note.setText("custom model - backtest it before trading it")
        self.changed.emit()

    def backtest_args(self) -> dict:
        return dict(start=self.start.date().toString("yyyy-MM-dd"), split=self.split.date().toString("yyyy-MM-dd"), capital=float(self.capital.value()))

    # ── files / persistence ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def save_file(self) -> None:
        import os
        path, _ = QFileDialog.getSaveFileName(self, "Save model", os.path.join(self.folder, "models", f"{self.model().name.replace(' ', '_').replace('<', 'lt')}.json"), "JSON (*.json)")
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.model().to_dict(), fh, indent=2)

    def load_file(self) -> None:
        import os
        path, _ = QFileDialog.getOpenFileName(self, "Load model", os.path.join(self.folder, "models"), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                m = MRModel.from_dict(json.load(fh))
        except Exception as e:                                                                # noqa: BLE001
            QMessageBox.warning(self, "Load model", f"Could not read that file: {e}")
            return
        self.preset.blockSignals(True); self.preset.setCurrentText(CUSTOM); self.preset.blockSignals(False)
        self.set_model(m)

    def save(self, qs) -> None:
        qs.setValue("model/preset", self.preset.currentText())
        qs.setValue("model/spec", json.dumps(self.model().to_dict()))
        for k, v in self.backtest_args().items():
            qs.setValue(f"bt/{k}", v)

    def load(self, qs) -> None:
        p = qs.value("model/preset")
        spec = qs.value("model/spec")
        try:
            if p and p in PR.PRESETS:
                self.preset.setCurrentText(p)
            elif spec:
                self.preset.blockSignals(True); self.preset.setCurrentText(CUSTOM); self.preset.blockSignals(False)
                self.set_model(MRModel.from_dict(json.loads(spec)))
            for k, w in (("start", self.start), ("split", self.split)):
                v = qs.value(f"bt/{k}")
                if v:
                    w.setDate(QDate.fromString(str(v), "yyyy-MM-dd"))
            v = qs.value("bt/capital")
            if v:
                self.capital.setValue(float(v))
        except (TypeError, ValueError):
            pass


def describe(m: MRModel) -> str:
    """One line a person can check before arming: everything that decides the trades."""
    ex = f"{m.exit_signal} {m.exit_op} {m.exit_value:g}" if m.exit_signal else "no signal exit"
    hold = "sell at the next open" if m.hold == "overnight" else f"exit {ex}" + (f" or price > {m.exit_sma_n}-day mean" if m.exit_sma_n else "") + f", time stop {m.max_hold}"
    return (f"{m.name}: {m.direction} {m.universe} · decide {m.minute // 100}:{m.minute % 100:02d} · enter {m.entry_signal} {m.entry_op} {m.entry_value:g}"
            + (f" and {m.entry2_signal} {m.entry2_op} {m.entry2_value:g}" if m.entry2_signal else "") + (f" above the {m.trend_n}-day mean" if m.trend_n else "") + f" · {hold} · {m.slots} slots ranked by {m.rank_by}"
            + (f", inverse-vol sizing ({m.vol_target:.1%}/day, max {m.max_slot_mult:g}x)" if m.sizing == "inv_vol" else "") + f" · cash reserve {m.cash_reserve:.1%} · "
            f"cost {m.cost_auction_bp:g} bp/side")
