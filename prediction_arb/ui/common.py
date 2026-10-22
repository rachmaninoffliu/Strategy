"""Shared UI plumbing: dark institutional theme, asyncio-in-QThread worker, sortable row model, APY badge."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from PyQt6.QtCore import (QAbstractTableModel, QModelIndex, QRectF, QSortFilterProxyModel, Qt, QThread,
                          pyqtSignal)
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPalette
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QLabel, QStyle,
                             QStyledItemDelegate, QStyleOptionViewItem, QTableView, QVBoxLayout)

from config import APY_GREEN, HURDLE_APY

# ── palette ───────────────────────────────────────────────────────────────────
BG = "#0e1116"
PANEL = "#151a21"
PANEL2 = "#1b222b"
BORDER = "#2a3441"
TEXT = "#d7dee8"
MUTED = "#8b98a9"
ACCENT = "#3d8bfd"
GREEN = "#2ea06b"
YELLOW = "#d9a520"
RED = "#e05561"
BLUE = "#4c9aff"

STYLE = f"""
* {{ font-family: "Segoe UI", "Inter", sans-serif; font-size: 12px; color: {TEXT}; }}
QMainWindow, QWidget {{ background: {BG}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; background: {BG}; }}
QTabBar::tab {{ background: {PANEL}; padding: 9px 20px; border: 1px solid {BORDER}; border-bottom: none;
                margin-right: 2px; color: {MUTED}; font-weight: 600; }}
QTabBar::tab:selected {{ background: {BG}; color: {TEXT}; border-top: 2px solid {ACCENT}; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 14px; padding: 10px 8px 8px 8px;
             background: {PANEL}; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}
QLabel {{ background: transparent; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#kpiValue {{ font-size: 20px; font-weight: 700; }}
QLabel#kpiName {{ color: {MUTED}; font-size: 11px; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit {{
    background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 6px;
    selection-background-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; selection-background-color: {ACCENT}; }}
QPushButton {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 6px 14px;
               font-weight: 600; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #566172; border-color: #212a35; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton#primary:disabled {{ background: #27446d; color: #8fa6c8; }}
QPushButton#danger {{ background: #4a2027; border-color: #7a3540; }}
QCheckBox {{ spacing: 6px; }}
QTableView, QTableWidget {{ background: {PANEL}; alternate-background-color: #171d25; gridline-color: {BORDER};
    border: 1px solid {BORDER}; selection-background-color: #23406b; selection-color: white; }}
QHeaderView::section {{ background: {PANEL2}; color: {MUTED}; padding: 6px; border: none;
    border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER}; font-weight: 600; }}
QProgressBar {{ background: {PANEL2}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QTextBrowser, QTextEdit, QPlainTextEdit {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; }}
QScrollBar:vertical {{ background: {BG}; width: 11px; }}
QScrollBar::handle:vertical {{ background: #34404f; border-radius: 5px; min-height: 24px; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; }}
QScrollBar::handle:horizontal {{ background: #34404f; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSplitter::handle {{ background: {BORDER}; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: 1px solid {BORDER}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QMenuBar {{ background: {PANEL}; }} QMenuBar::item:selected {{ background: {PANEL2}; }}
QMenu {{ background: {PANEL2}; border: 1px solid {BORDER}; }} QMenu::item:selected {{ background: {ACCENT}; }}
"""


def apply_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG))
    pal.setColor(QPalette.ColorRole.Base, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#171d25"))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(PANEL2))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def venue_label(venue, us_tradable: bool = True) -> str:
    """'Kalshi' | 'Poly US' | 'Poly INTL (not US)' - makes US-tradability visible wherever a venue is named."""
    name = getattr(venue, "value", str(venue))
    if name == "kalshi":
        return "Kalshi"
    return "Poly US" if us_tradable else "Poly INTL (not US)"


# ── formatting ────────────────────────────────────────────────────────────────
def fmt_apy(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x >= 99.99:
        return ">9,999%"
    return f"{x * 100:,.1f}%"


def fmt_pct(x: Optional[float], d: int = 2) -> str:
    return "-" if x is None else f"{x * 100:,.{d}f}%"


def fmt_usd(x: Optional[float], d: int = 2) -> str:
    if x is None:
        return "-"
    return f"-${abs(x):,.{d}f}" if x < 0 else f"${x:,.{d}f}"


def apy_color(apy: Optional[float]) -> tuple[QColor, QColor]:
    """(background, text) for an APY badge: green > 15 %, yellow 7-15 %, grey below the hurdle."""
    if apy is None:
        return QColor(PANEL2), QColor(MUTED)
    if apy > APY_GREEN:
        return QColor(GREEN), QColor("white")
    if apy >= HURDLE_APY:
        return QColor(YELLOW), QColor("#1a1400")
    return QColor("#39424f"), QColor(MUTED)


class ApyBadgeDelegate(QStyledItemDelegate):
    """Paints the APY cell as a coloured rounded badge (value read from UserRole)."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        raw = index.data(Qt.ItemDataRole.UserRole)
        txt = index.data(Qt.ItemDataRole.DisplayRole) or ""
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        bg, fg = apy_color(raw if isinstance(raw, (int, float)) else None)
        r = QRectF(option.rect).adjusted(6, 4, -6, -4)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(r, 9, 9)
        f = QFont(option.font)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(fg)
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, txt)
        painter.restore()


# ── generic table model ───────────────────────────────────────────────────────
@dataclass
class Col:
    header: str
    get: Callable[[Any], Any]                           # raw value (used for sorting)
    fmt: Callable[[Any], str] = str
    align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    fg: Optional[Callable[[Any], Optional[QColor]]] = None
    bg: Optional[Callable[[Any], Optional[QColor]]] = None
    tip: Optional[Callable[[Any], str]] = None
    numeric: bool = False


RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
CENTER = Qt.AlignmentFlag.AlignCenter


class RowTableModel(QAbstractTableModel):
    def __init__(self, cols: list[Col], parent=None):
        super().__init__(parent)
        self.cols = cols
        self.rows: list[Any] = []

    def set_rows(self, rows: list[Any]):
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def row_at(self, i: int) -> Any:
        return self.rows[i]

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.cols)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.cols[section].header
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = self.rows[index.row()], self.cols[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            try:
                return col.fmt(col.get(row))
            except Exception:
                return "-"
        if role == Qt.ItemDataRole.UserRole:                # stable, type-consistent sort key
            try:
                v = col.get(row)
            except Exception:
                v = None
            if col.numeric:
                return float(v) if isinstance(v, (int, float)) else -1e18
            return "" if v is None else str(v)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(col.align)
        if role == Qt.ItemDataRole.ForegroundRole and col.fg:
            c = col.fg(row)
            return QBrush(c) if c else None
        if role == Qt.ItemDataRole.BackgroundRole and col.bg:
            c = col.bg(row)
            return QBrush(c) if c else None
        if role == Qt.ItemDataRole.ToolTipRole and col.tip:
            return col.tip(row)
        return None


class SortProxy(QSortFilterProxyModel):
    """Sorts on UserRole; optional row predicate for filtering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortRole(Qt.ItemDataRole.UserRole)
        self.predicate: Optional[Callable[[Any], bool]] = None

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if self.predicate is None:
            return True
        return bool(self.predicate(self.sourceModel().row_at(source_row)))


# ── asyncio in a QThread ──────────────────────────────────────────────────────
class AsyncWorker(QThread):
    """Runs ``factory(progress_cb, cancel_event)`` (a coroutine) on a private event loop.

    Signals are queued to the GUI thread. ``cancel()`` cancels the running task (sessions close cleanly through
    ``async with``) and also sets the cooperative cancel event used by the progress callbacks.
    """
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, factory: Callable[[Callable[[int, str], None], asyncio.Event], Awaitable[Any]], parent=None):
        super().__init__(parent)
        self._factory = factory
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._evt: Optional[asyncio.Event] = None

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            self._evt = asyncio.Event()
            self._task = loop.create_task(self._factory(lambda p, m: self.progress.emit(int(p), str(m)), self._evt))
            result = loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            self.cancelled.emit()
        except Exception as e:                              # noqa: BLE001 - surface every failure to the UI
            from core.backtester import Cancelled
            if isinstance(e, Cancelled):
                self.cancelled.emit()
            else:
                self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.succeeded.emit(result)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
                self._loop = None

    def cancel(self):
        loop, task, evt = self._loop, self._task, self._evt
        if loop and task and not task.done():
            loop.call_soon_threadsafe(evt.set)
            loop.call_soon_threadsafe(task.cancel)


class DataTable(QTableView):
    """Read-only, sortable, row-selecting table with alternating rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(True)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(30)
        self.horizontalHeader().setStretchLastSection(False)
        self.setWordWrap(False)
        self.setShowGrid(False)


class KpiCard(QFrame):
    """Small stat tile: muted caption over a bold value."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        self.name = QLabel(name)
        self.name.setObjectName("kpiName")
        self.value = QLabel("-")
        self.value.setObjectName("kpiValue")
        lay.addWidget(self.name)
        lay.addWidget(self.value)

    def set(self, text: str, color: Optional[str] = None):
        self.value.setText(text)
        self.value.setStyleSheet(f"color:{color};" if color else "")


# ── matplotlib (dark) ─────────────────────────────────────────────────────────
def dark_canvas(width=8, height=6, dpi=100):
    """-> (Figure, FigureCanvasQTAgg). Figure is created without pyplot so tabs stay independent."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(width, height), dpi=dpi, facecolor=PANEL)
    return fig, FigureCanvasQTAgg(fig)


def style_ax(ax, title: str = "", ylabel: str = ""):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=BORDER, alpha=0.5, linewidth=0.6)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, loc="left", fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.xaxis.label.set_color(MUTED)
