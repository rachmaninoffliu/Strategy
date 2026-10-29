"""Look and feel shared by every page: palette, style sheet, KPI cards, tables, scrolling pages and wheel-safe inputs."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPalette
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QComboBox as _QComboBox, QDateEdit as _QDateEdit, QDoubleSpinBox as _QDoubleSpinBox, QFrame,
                             QHeaderView, QLabel, QScrollArea, QSizePolicy, QSpinBox as _QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

# ── palette (same as the BTC 15-min trader and the arbitrage app) ────────────────
BG, PANEL, PANEL2, BORDER = "#0e1116", "#151a21", "#1b222b", "#2a3441"
TEXT, MUTED, ACCENT = "#d7dee8", "#8b98a9", "#3d8bfd"
GREEN, YELLOW, RED = "#2ea06b", "#d9a520", "#e05561"
# Direction colours are NOT P&L colours: long = green, short = red. A SOXL short is held as a LONG SOXS position, so colour is always derived from the strategy side.
COL_LONG, COL_SHORT, COL_FLAT = GREEN, RED, MUTED
CHECK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "check.svg").replace("\\", "/")      # white tick drawn inside a checked box

STYLE = f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 12px; color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; background: {BG}; }}
QTabBar::tab {{ background: {PANEL}; padding: 9px 22px; border: 1px solid {BORDER}; border-bottom: none; margin-right: 2px;
               color: {MUTED}; font-weight: 600; }}
QTabBar::tab:selected {{ background: {BG}; color: {TEXT}; border-top: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 14px; padding: 10px 8px 8px 8px; background: {PANEL}; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QLabel {{ background: transparent; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#warn {{ color: {YELLOW}; }}
QLabel#kpiValue {{ font-size: 18px; font-weight: 700; }}
QLabel#kpiName {{ color: {MUTED}; font-size: 11px; }}
QLabel#kpiSub {{ color: {MUTED}; font-size: 10px; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 6px;
    selection-background-color: {ACCENT}; min-height: 18px; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QDateEdit:disabled {{ color: #566172; background: #141a21; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QDateEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; selection-background-color: {ACCENT}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 14px; font-weight: 600; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #566172; border-color: #212a35; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton#primary:disabled {{ background: #27446d; color: #8fa6c8; }}
QPushButton#go {{ background: #1f6f4a; border-color: {GREEN}; color: white; }}
QPushButton#go:disabled {{ background: #1a3a2c; color: #7da892; border-color: #24503c; }}
QPushButton#danger {{ background: #7a2430; border-color: {RED}; color: white; }}
QPushButton#danger:disabled {{ background: #3a1d22; color: #a07a80; border-color: #55303a; }}
QPushButton#flat {{ background: transparent; border: none; color: {ACCENT}; padding: 2px 6px; }}
QCheckBox {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid #4b5a70; border-radius: 3px; background: {PANEL2}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("{CHECK}"); }}
QCheckBox::indicator:disabled {{ border-color: #2c3745; background: #141a21; }}
QCheckBox::indicator:checked:disabled {{ background: #27446d; border-color: #27446d; }}
QTableWidget {{ background: {PANEL}; alternate-background-color: #171d25; gridline-color: {BORDER}; border: 1px solid {BORDER};
    selection-background-color: #23406b; selection-color: white; }}
QHeaderView::section {{ background: {PANEL2}; color: {MUTED}; padding: 6px; border: none; border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER}; font-weight: 600; }}
QPlainTextEdit {{ background: #0b0e13; border: 1px solid {BORDER}; font-family: Consolas, "Cascadia Mono", monospace; font-size: 11px; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
QScrollBar:vertical {{ background: {BG}; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: {BG}; height: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #2f3b4b; border-radius: 5px; min-height: 28px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2f3b4b; border-radius: 5px; min-width: 28px; margin: 2px; }}
QScrollBar::handle:hover {{ background: #3d4d62; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QMenuBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {PANEL2}; }}
QMenu {{ background: {PANEL2}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 26px 6px 14px; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}
QToolTip {{ background: #1f2937; color: #f9fafb; border: 1px solid {BORDER}; padding: 4px; }}
QProgressBar {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; max-height: 12px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QTextBrowser {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 12px; font-size: 13px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()                                          # native-drawn parts (check marks, calendar pop-up, message boxes) follow the dark palette too
    for role, col in ((QPalette.ColorRole.Window, BG), (QPalette.ColorRole.WindowText, TEXT), (QPalette.ColorRole.Base, PANEL2), (QPalette.ColorRole.AlternateBase, PANEL),
                      (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.Button, PANEL2), (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
                      (QPalette.ColorRole.HighlightedText, "#ffffff"), (QPalette.ColorRole.ToolTipBase, "#1f2937"), (QPalette.ColorRole.ToolTipText, "#f9fafb"),
                      (QPalette.ColorRole.PlaceholderText, "#566172")):
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


# ── inputs that ignore a stray mouse wheel ────────────────────────────────────────
class _NoStrayWheel:
    """A spin box / drop-down / date field only reacts to the mouse wheel once you have clicked into it. Otherwise scrolling the settings panel silently changes whatever
    field is under the cursor (leverage per trade, factor, ...) - a real risk for a trading app. The ignored wheel event scrolls the panel instead."""

    def _quiet(self) -> None:
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, e) -> None:                      # noqa: N802 (Qt override)
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class QDoubleSpinBox(_NoStrayWheel, _QDoubleSpinBox):
    def __init__(self, *a):
        super().__init__(*a); self._quiet()


class QSpinBox(_NoStrayWheel, _QSpinBox):
    def __init__(self, *a):
        super().__init__(*a); self._quiet()


class QComboBox(_NoStrayWheel, _QComboBox):
    def __init__(self, *a):
        super().__init__(*a); self._quiet()


class QDateEdit(_NoStrayWheel, _QDateEdit):
    def __init__(self, *a):
        super().__init__(*a); self._quiet()


# ── formatting ────────────────────────────────────────────────────────────────────
def money(x: float, sign: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+,.2f}" if sign else f"{x:,.2f}"


def big_money(x: float, sign: bool = False) -> str:
    """$ with K / M / B / T suffixes: an equity curve that compounds to $767M would otherwise print as a wall of digits."""
    if x is None or not np.isfinite(x):
        return "—"
    s = "-" if x < 0 else ("+" if sign else "")
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"{s}${a / div:,.2f}{suf}"
    return f"{s}${a:,.0f}" if a >= 10_000 else f"{s}${a:,.2f}"


def pct(x: float, d: int = 2, sign: bool = False) -> str:
    """Fraction -> percent text (0.1234 -> 12.34%)."""
    if x is None or not np.isfinite(x):
        return "—"
    if abs(x) >= 100:                                        # +767,499% reads better with thousands separators and no decimals
        return f"{x * 100:+,.0f}%" if sign else f"{x * 100:,.0f}%"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def num(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{d}f}"


def rgba(color: str, alpha: float) -> str:
    c = QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def box_style(color: str, alpha: float = 0.14, bold: bool = False, pad: int = 8) -> str:
    """Style sheet of a tinted, bordered message strip (banner / notice / status)."""
    return f"background: {rgba(color, alpha)}; border: 1px solid {color}; border-radius: 6px; padding: {pad}px; color: {TEXT};" + (" font-weight: 700;" if bold else "")


def strip(text: str = "", color: str = ACCENT, alpha: float = 0.14, bold: bool = False, wrap: bool = True) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(wrap)
    lb.setStyleSheet(box_style(color, alpha, bold))
    lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lb


def muted(text: str = "", wrap: bool = True) -> QLabel:
    lb = QLabel(text)
    lb.setObjectName("muted")
    lb.setWordWrap(wrap)
    return lb


# ── small widgets ─────────────────────────────────────────────────────────────────
class ElideLabel(QLabel):
    """One-line label that shortens its text with '…' instead of forcing its parent wider (the full text is in the card's tooltip)."""

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:                   # noqa: N802 (Qt override)
        return QSize(20, super().minimumSizeHint().height())

    def paintEvent(self, e) -> None:                      # noqa: N802 (Qt override)
        p = QPainter(self)
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, max(self.width(), 1))
        self.style().drawItemText(p, self.rect(), int(self.alignment()) | int(Qt.AlignmentFlag.AlignVCenter), self.palette(), self.isEnabled(), text, self.foregroundRole())


class Kpi(QFrame):
    """A number card: name on top, big value, optional small line below."""

    def __init__(self, name: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        self.n = QLabel(name); self.n.setObjectName("kpiName")
        self.v = ElideLabel("—"); self.v.setObjectName("kpiValue")
        self.s = ElideLabel(""); self.s.setObjectName("kpiSub"); self.s.hide()
        for w in (self.n, self.v, self.s):
            lay.addWidget(w)

    def set(self, text: str, color: Optional[str] = None, sub: str = "") -> None:
        self.v.setText(text)
        self.v.setStyleSheet(f"color: {color};" if color else "")
        self.s.setText(sub)
        self.s.setVisible(bool(sub))
        self.setToolTip(f"{self.n.text()}: {text}" + (f"\n{sub}" if sub else ""))


def spin(lo: float, hi: float, val: float, dec: int = 2, step: float = 0.1, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    s.setGroupSeparatorShown(True)
    return s


def make_table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.setTextElideMode(Qt.TextElideMode.ElideNone)                         # never cut a cell's text with "..."
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)    # wide tables scroll sideways instead of clipping
    t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return t


def set_row(t: QTableWidget, r: int, vals: list, colors: Optional[dict] = None, align_left: tuple = ()) -> None:
    for c, v in enumerate(vals):
        it = QTableWidgetItem(str(v))
        it.setToolTip(str(v))
        left = c == 0 or c in align_left
        it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if left else Qt.AlignmentFlag.AlignRight) | Qt.AlignmentFlag.AlignVCenter)
        if colors and colors.get(c):
            it.setForeground(QColor(colors[c]))
        t.setItem(r, c, it)


class PageScroll(QScrollArea):
    """A scrolling page. When the window is smaller than the page needs it scrolls (both directions) instead of cutting text off.
    Unlike QScrollArea(widgetResizable=True) it sizes the page by its MINIMUM height for the current width: Qt would otherwise take the page's *preferred*
    height-for-width as a hard minimum as soon as one word-wrapped label is on it, forcing a scroll bar even on a big window."""

    def __init__(self, inner: QWidget, min_w: int = 0, min_h: int = 0):
        super().__init__()
        self._min = (min_w, min_h)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setWidget(inner)
        inner.installEventFilter(self)
        self.viewport().installEventFilter(self)
        self._fit()

    def eventFilter(self, obj, ev) -> bool:           # noqa: N802 (Qt override)
        if obj is self.widget() and ev.type() == QEvent.Type.LayoutRequest:
            QTimer.singleShot(0, self._fit)
        elif obj is self.viewport() and ev.type() == QEvent.Type.Resize:
            self._fit()
        return False

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._fit()

    def _fit(self) -> None:
        inner = self.widget()
        if inner is None:
            return
        vp = self.viewport().size()
        w = max(vp.width(), inner.minimumSizeHint().width(), self._min[0])
        lay = inner.layout()
        h_need = lay.minimumHeightForWidth(w) if (lay is not None and lay.hasHeightForWidth()) else inner.minimumSizeHint().height()
        h = max(vp.height(), h_need, self._min[1])
        if (inner.width(), inner.height()) != (w, h):
            inner.resize(w, h)


def scrollable(inner: QWidget, min_w: int = 0, min_h: int = 0) -> QScrollArea:
    """Wrap a page so that it scrolls instead of cutting text off (see PageScroll)."""
    return PageScroll(inner, min_w, min_h)
