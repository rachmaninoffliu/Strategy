"""Progress + go / no-go report of the pre-flight check. ``action`` records how the user chose to proceed: start | start_frac | dry."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit, QPushButton, QTableWidgetItem, QVBoxLayout
from PyQt6.QtCore import Qt
from datetime import datetime, timezone

from .params import Report
from .preflight import FAIL, INFO, PASS, SKIP, WARN
from .theme import GREEN, MUTED, RED, YELLOW, box_style, make_table, muted

ICON = {PASS: "✔ PASS", WARN: "⚠ WARN", FAIL: "✖ FAIL", INFO: "ℹ INFO", SKIP: "– SKIP"}


class PreflightDialog(QDialog):
    RANK = {FAIL: 0, WARN: 1, PASS: 2, INFO: 3, SKIP: 4}

    def __init__(self, parent, live: bool, armed: bool, then_start: bool, frac: float):
        super().__init__(parent)
        self.setWindowTitle("Pre-flight check")
        self.resize(1040, 700)
        self.live, self.armed, self.then_start, self.frac = live, armed, then_start, frac
        self.action: Optional[str] = None
        self.report: Optional[Report] = None
        self._go_action = "start"
        v = QVBoxLayout(self)
        self.title = QLabel("Running pre-flight checks…")
        self.title.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.verdict = QLabel(""); self.verdict.setWordWrap(True); self.verdict.hide()
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(130)
        self.table = make_table(["", "Check", "Result"])
        self.table.setWordWrap(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.hide()
        self.foot = muted("")
        row = QHBoxLayout()
        self.btn_copy = QPushButton("Copy report"); self.btn_copy.clicked.connect(self._copy); self.btn_copy.setEnabled(False)
        self.btn_dry = QPushButton("Run as a dry run instead"); self.btn_dry.clicked.connect(lambda: self._choose("dry")); self.btn_dry.hide()
        self.btn_frac = QPushButton("Use the suggested size instead"); self.btn_frac.clicked.connect(lambda: self._choose("start_frac")); self.btn_frac.hide()
        self.btn_go = QPushButton("…"); self.btn_go.setObjectName("go"); self.btn_go.clicked.connect(lambda: self._choose(self._go_action)); self.btn_go.hide()
        self.btn_close = QPushButton("Cancel" if then_start else "Close"); self.btn_close.clicked.connect(self.reject)
        row.addWidget(self.btn_copy); row.addStretch(1)
        for b in (self.btn_dry, self.btn_frac, self.btn_close):
            row.addWidget(b)
        for w in (self.title, self.verdict, self.log, self.table, self.foot):
            v.addWidget(w)
        v.addLayout(row)
        self.btn_go.setMinimumHeight(38)
        v.addWidget(self.btn_go)                       # the main action gets its own full-width row
        self.setMinimumSize(640, 460)

    def add_log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _choose(self, action: str) -> None:
        self.action = action
        self.accept()

    def _copy(self) -> None:
        if self.report:
            QApplication.clipboard().setText(self.report.text())

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.table.resizeRowsToContents()

    def set_report(self, rep: Report, saved: str = "") -> None:
        self.report = rep
        col = {"GO": GREEN, "CAUTION": YELLOW, "NO-GO": RED}[rep.verdict]
        self.title.setText(f"Pre-flight result — {datetime.fromtimestamp(rep.created, timezone.utc):%H:%M:%S} UTC")
        self.verdict.setText(rep.headline() + ("" if rep.armed else "\n(A dry run sends no orders, so a NO-GO only means the live engine could not do its job - it never blocks a dry run.)"))
        self.verdict.setStyleSheet(box_style(col, 0.16, pad=10) + " font-size: 14px; font-weight: 600;")
        self.verdict.show()
        self.log.hide()
        rows = sorted(enumerate(rep.checks), key=lambda t: (self.RANK.get(t[1].status, 9), t[0]))
        items = [r for _, r in rows]
        notes = list(rep.notes)
        self.table.setRowCount(len(items) + len(notes))
        colors = {PASS: GREEN, WARN: YELLOW, FAIL: RED, INFO: MUTED, SKIP: MUTED}
        for i, ck in enumerate(items):
            for j, txt in enumerate((ICON.get(ck.status, ck.status), f"{ck.group} · {ck.name}", ck.detail)):
                it = QTableWidgetItem(txt); it.setToolTip(txt)
                it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                if j == 0:
                    it.setForeground(QColor(colors.get(ck.status, MUTED)))
                    f = it.font(); f.setBold(True); it.setFont(f)
                self.table.setItem(i, j, it)
        for k, n in enumerate(notes):
            for j, txt in enumerate((ICON[INFO], "Note", n)):
                it = QTableWidgetItem(txt); it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop); it.setForeground(QColor(MUTED))
                self.table.setItem(len(items) + k, j, it)
        self.table.show()
        self.table.resizeColumnToContents(0); self.table.resizeColumnToContents(1)
        self.table.resizeRowsToContents()
        self.foot.setText(f"Report saved: {saved}" if saved else "")
        self.btn_copy.setEnabled(True)
        self._buttons(rep)

    def _buttons(self, rep: Report) -> None:
        if not self.then_start:
            return
        if not self.armed:                                              # a dry run is never blocked
            self.btn_go.setText("▶  Start dry run (orders shown, none sent)" + ("" if rep.verdict == "GO" else f"   — verdict: {rep.verdict}"))
            self._go_action = "start"
            self.btn_go.setEnabled(True)
            self.btn_go.show()
            return
        kind = "LIVE" if self.live else "PAPER"
        if rep.verdict == "GO":
            self.btn_go.setText("▶  Continue to LIVE confirmation" if self.live else "▶  Continue: start PAPER session"); self._go_action = "start"; self.btn_go.setEnabled(True)
        elif rep.verdict == "CAUTION":
            self.btn_go.setText(f"▶  Continue to {kind} with my settings ({self.frac:g}% of equity per trade)"); self._go_action = "start"; self.btn_go.setEnabled(True)
            if rep.suggested_frac is not None and self.frac > rep.suggested_frac + 1e-9:
                self.btn_frac.setText(f"Use {rep.suggested_frac:g}% per trade instead"); self.btn_frac.show()
            self.btn_dry.show()
        else:
            self.btn_go.setText(f"⛔  {kind} is blocked — fix the failed checks"); self.btn_go.setEnabled(False)
            self.btn_dry.show()
        self.btn_go.show()
