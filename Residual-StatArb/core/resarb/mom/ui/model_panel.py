"""The model picker: a preset or every field of a MomModel by hand. The SAME model object drives the Backtest tab and the Live tab (the live loop locks this panel while it runs)."""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import QDate, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from ...ui.theme import YELLOW, QComboBox, QDateEdit, QSpinBox, muted, spin, strip
from .. import presets as PR
from ..ml import FIRST_YEAR
from ..models import MomModel

UNIVERSES = ["top150+etf", "top100+etf", "top50+etf", "top150", "etf"]
MINUTES = [("15:30 (free delayed SIP)", 1530), ("15:45 (needs the real-time SIP plan)", 1545)]
EVERY = [("30 minutes", 2), ("15 minutes", 1), ("60 minutes", 4)]
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
        root.addWidget(strip("Pick a book. The backtest runs it and the live engine trades exactly this model (same decision functions). Long-only, no leverage: the overnight "
                             "names are sold at the open before the intraday leg can buy, and the intraday position is sold in the closing auction that buys the overnight names.",
                             YELLOW))
        g = QGroupBox("Model"); f = QFormLayout(g)
        self.preset = _combo(PR.preset_names() + [CUSTOM])
        self.preset.currentTextChanged.connect(self._preset)
        self.note = muted("")
        self.name = QLineEdit()
        f.addRow("Preset", self.preset); f.addRow(self.note); f.addRow("Name", self.name)
        root.addWidget(g)
        g = QGroupBox("Overnight leg (bought at the close -> sold at the next open)"); f = QFormLayout(g)
        self.overnight = QCheckBox("on")
        self.universe = _combo(UNIVERSES, True)
        self.universe.setToolTip("topN = the N most liquid U.S. stocks of each month (point-in-time, survivorship-free); etf = 59 ETFs across assets; join parts with +")
        self.minute = _combo([m[0] for m in MINUTES])
        self.minute.setToolTip("The decision uses the 15-minute SIP bars that ended by this time; orders go into the closing auction (market-on-close, sent before 15:50).")
        self.k = _ispin(1, 100, 20); self.k.setToolTip("names bought each night; each gets (1 - cash reserve) / k of the capital")
        self.ranking = _combo(["rule", "rule+ml"])
        self.ranking.setToolTip("rule = today's move (last price / yesterday's close); rule+ml = the rank of that move + the rank of the walk-forward ML forecast of the overnight "
                                "return (one model per year, Backtest tab > Train ML models)")
        self.trend = QCheckBox("only when SPY's last close is above its 200-day mean")
        self.min_price = spin(0, 1000, 0, 2, 1)
        f.addRow("Overnight leg", self.overnight); f.addRow("Universe", self.universe); f.addRow("Decision time", self.minute); f.addRow("Names (k)", self.k)
        f.addRow("Rank by", self.ranking); f.addRow("Trend filter", self.trend); f.addRow("Min price $", self.min_price)
        root.addWidget(g)
        g = QGroupBox("Intraday leg (noise-area breakout, open -> close)"); f = QFormLayout(g)
        self.isym = _combo(["QQQ", "SPY", "(none)"], True)
        self.every = _combo([e[0] for e in EVERY])
        self.lookback = _ispin(2, 60, 14); self.lookback.setToolTip("sessions in the noise band (mean |close(slot) / open - 1| of each time of day)")
        self.trail = QCheckBox("sell below max(band, VWAP)")
        self.mult = spin(0.2, 5.0, 1.0, 2, 0.1); self.mult.setToolTip("band width multiplier")
        self.ifeed = _combo(["iex", "sip"])
        self.ifeed.setToolTip("decision prices and VWAP of the intraday leg: iex = the free real-time feed (the backtest reads the cached IEX bars, from 2024; the SIP bars stand in "
                              "before); sip = real-time SIP (paid plan in live)")
        f.addRow("Symbol", self.isym); f.addRow("Check every", self.every); f.addRow("Band lookback", self.lookback); f.addRow("Trailing exit", self.trail)
        f.addRow("Band multiplier", self.mult); f.addRow("Intraday feed", self.ifeed)
        root.addWidget(g)
        g = QGroupBox("Capital and costs (no leverage)"); f = QFormLayout(g)
        self.cash_reserve = spin(0.0, 0.2, 0.02, 3, 0.005)
        self.sector_cap = _ispin(0, 100, 0); self.sector_cap.setSpecialValueText("off"); self.sector_cap.setSuffix(" names per sector")
        self.sector_cap.setToolTip("At most this many overnight names per sector (Tech, Fin, Health, ... - the universe's 11 sectors; sector ETFs such as SMH / XLK count in "
                                   "theirs, other ETFs in none). A skipped name's place goes to the next-best name of another sector. Off = 0. Suggested below k = 10.")
        self.cash_reserve.setToolTip("kept in cash so a closing price above the decision price cannot push the book over 1x")
        self.cost = spin(0, 50, 1.0, 2, 0.5, " bp"); self.cost.setToolTip("per side, opening / closing auction")
        self.half = spin(0, 50, 0.5, 2, 0.25, " bp"); self.half.setToolTip("per side, intraday market orders")
        f.addRow("Cash reserve", self.cash_reserve); f.addRow("Sector cap", self.sector_cap); f.addRow("Auction cost per side", self.cost)
        f.addRow("Half spread (intraday)", self.half)
        f.addRow(muted("Max gross = 1.00x equity at every moment (fixed: no leverage)."))
        root.addWidget(g)
        g = QGroupBox("Backtest"); f = QFormLayout(g)
        self.start = QDateEdit(); self.start.setCalendarPopup(True); self.start.setDisplayFormat("yyyy-MM-dd"); self.start.setDate(QDate(2018, 1, 2))
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
        self.btn_update = QPushButton("Update data (daily + 15-minute SIP / IEX bars)")
        root.addWidget(self.btn_update); root.addWidget(self.data_lbl)
        root.addStretch(1)
        self.name.textEdited.connect(self._edited)
        for w in (self.universe, self.minute, self.ranking, self.isym, self.every, self.ifeed):
            w.currentTextChanged.connect(self._edited)
        for w in (self.k, self.min_price, self.lookback, self.mult, self.cash_reserve, self.sector_cap, self.cost, self.half):
            w.valueChanged.connect(self._edited)
        for w in (self.overnight, self.trend, self.trail):
            w.toggled.connect(self._edited)
        self._preset(self.preset.currentText())

    # ── model <-> fields ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def model(self) -> MomModel:
        mi, ei = self.minute.currentIndex(), self.every.currentIndex()
        isym = self.isym.currentText().strip().upper()
        return MomModel(name=self.name.text().strip() or "custom", universe=self.universe.currentText().strip(), minute=MINUTES[mi][1] if mi >= 0 else 1530,
                        overnight=self.overnight.isChecked(), k=int(self.k.value()), ranking=self.ranking.currentText(), trend_filter=self.trend.isChecked(),
                        min_price=float(self.min_price.value()), intraday_symbol="" if isym in ("", "(NONE)") else isym, intraday_every=EVERY[ei][1] if ei >= 0 else 2,
                        intraday_lookback=int(self.lookback.value()), intraday_trail=self.trail.isChecked(), intraday_mult=float(self.mult.value()),
                        intraday_feed=self.ifeed.currentText(), cash_reserve=float(self.cash_reserve.value()), sector_cap=int(self.sector_cap.value()), cost_auction_bp=float(self.cost.value()),
                        half_spread_bp=float(self.half.value()), note=self.note.text())

    def set_model(self, m: MomModel) -> None:
        self._quiet = True
        try:
            self.name.setText(m.name)
            self.overnight.setChecked(m.overnight)
            self.universe.setCurrentText(m.universe)
            mins = [x[1] for x in MINUTES]
            if m.minute not in mins:
                MINUTES.append((f"{m.minute // 100}:{m.minute % 100:02d}", m.minute))
                self.minute.addItem(MINUTES[-1][0])
            self.minute.setCurrentIndex([x[1] for x in MINUTES].index(m.minute))
            self.k.setValue(m.k); self.ranking.setCurrentText(m.ranking); self.trend.setChecked(m.trend_filter); self.min_price.setValue(m.min_price)
            self.isym.setCurrentText(m.intraday_symbol or "(none)")
            ev = [x[1] for x in EVERY]
            if m.intraday_every not in ev:
                EVERY.append((f"{15 * m.intraday_every} minutes", m.intraday_every))
                self.every.addItem(EVERY[-1][0])
            self.every.setCurrentIndex([x[1] for x in EVERY].index(m.intraday_every))
            self.lookback.setValue(m.intraday_lookback); self.trail.setChecked(m.intraday_trail); self.mult.setValue(m.intraday_mult); self.ifeed.setCurrentText(m.intraday_feed)
            self.cash_reserve.setValue(m.cash_reserve); self.sector_cap.setValue(m.sector_cap); self.cost.setValue(m.cost_auction_bp); self.half.setValue(m.half_spread_bp)
            self.note.setText(m.note)
            if m.uses_ml and self.start.date() < QDate(FIRST_YEAR, 1, 2):
                self.start.setDate(QDate(FIRST_YEAR, 1, 2))                                      # the first forecast year
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
        path, _ = QFileDialog.getSaveFileName(self, "Save model", os.path.join(self.folder, "models", f"{self.model().name.replace(' ', '_').replace('+', 'p')}.json"),
                                              "JSON (*.json)")
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.model().to_dict(), fh, indent=2)

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load model", os.path.join(self.folder, "models"), "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                m = MomModel.from_dict(json.load(fh))
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
        p = PR.RENAMED.get(p, p)
        spec = qs.value("model/spec")
        try:
            if p and p in PR.PRESETS:
                self.preset.setCurrentText(p)
            elif spec:
                self.preset.blockSignals(True); self.preset.setCurrentText(CUSTOM); self.preset.blockSignals(False)
                self.set_model(MomModel.from_dict(json.loads(spec)))
            for k, w in (("start", self.start), ("split", self.split)):
                v = qs.value(f"bt/{k}")
                if v:
                    w.setDate(QDate.fromString(str(v), "yyyy-MM-dd"))
            v = qs.value("bt/capital")
            if v:
                self.capital.setValue(float(v))
        except (TypeError, ValueError):
            pass


def describe(m: MomModel) -> str:
    """One line a person can check before arming: everything that decides the trades."""
    ex = f"MOC -> OPG ({m.cost_auction_bp:g} bp/auction side)"
    on = (f"overnight: {m.k} best of {m.universe} by {m.ranking} at {m.minute // 100}:{m.minute % 100:02d}" + (", SPY > 200d" if m.trend_filter else "")
          + (f", at most {m.sector_cap} per sector" if m.sector_cap > 0 else "")
          + f" · {ex}") if m.overnight else "overnight: off"
    it = (f"intraday: {m.intraday_symbol} noise band ({m.intraday_lookback}d, x{m.intraday_mult:g}) every {15 * m.intraday_every} min on {m.intraday_feed}"
          + (", trailing max(band, VWAP)" if m.intraday_trail else "")) if m.intraday_symbol else "intraday: off"
    return f"{m.name}: {on} · {it} · cash reserve {m.cash_reserve:.1%} · {m.half_spread_bp:g} bp half spread"
