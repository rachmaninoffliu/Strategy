"""A sortable, filterable table over a DataFrame (QTableView + model): the trade list can have tens of thousands of rows, too many for one QTableWidgetItem per cell."""
from __future__ import annotations

import re
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QRegularExpression, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

SORT = Qt.ItemDataRole.UserRole


class FrameModel(QAbstractTableModel):
    """``cols`` = [(header, frame column, formatter(value) -> str)]; ``color`` = optional fn(row Series, column) -> colour or None."""

    def __init__(self, cols: list, color: Optional[Callable] = None, left: tuple = ()):
        super().__init__()
        self.cols, self.color, self.left = cols, color, set(left)
        self.df = pd.DataFrame()

    def set_frame(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self._raw = {c: self.df[c].to_numpy() if c in self.df else np.full(len(self.df), None) for _, c, _ in self.cols}
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:                                         # noqa: N802 (Qt override)
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent=QModelIndex()) -> int:                                      # noqa: N802
        return 0 if parent.isValid() else len(self.cols)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):          # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.cols[section][0]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        h, c, fmt = self.cols[index.column()]
        v = self._raw[c][index.row()]
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.ToolTipRole:
            try:
                return fmt(v)
            except (TypeError, ValueError):
                return "—"
        if role == SORT:
            if v is None or (isinstance(v, float) and not np.isfinite(v)) or v is pd.NaT:
                return float("-inf")
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                return pd.Timestamp(v).value
            return v.item() if isinstance(v, np.generic) else v
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int((Qt.AlignmentFlag.AlignLeft if index.column() in self.left or index.column() == 0 else Qt.AlignmentFlag.AlignRight) | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole and self.color is not None:
            col = self.color(self.df.iloc[index.row()], c)
            return QColor(col) if col else None
        return None


class FrameView(QTableView):
    def __init__(self, cols: list, color: Optional[Callable] = None, left: tuple = (), filter_col: int = 1):
        super().__init__()
        self.src = FrameModel(cols, color, left)
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.src)
        self.proxy.setSortRole(SORT)
        self.proxy.setFilterKeyColumn(filter_col)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setModel(self.proxy)
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.setWordWrap(False)

    def set_frame(self, df: pd.DataFrame) -> None:
        self.src.set_frame(df)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.resizeColumnsToContents()

    def set_filter(self, text: str) -> None:
        """'AA' = tickers starting with AA; 'V, AAPL' (a comma) = exactly these tickers."""
        t = text.strip()
        if "," in t:
            names = [re.escape(x.strip()) for x in t.split(",") if x.strip()]
            pat = f"^({'|'.join(names)})$" if names else ""
        else:
            pat = f"^{re.escape(t)}" if t else ""
        self.proxy.setFilterRegularExpression(QRegularExpression(pat, QRegularExpression.PatternOption.CaseInsensitiveOption))


def f_money(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:+,.2f}"


def f_px(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:,.2f}"


def f_int(v) -> str:
    return "—" if v is None or not np.isfinite(float(v)) else f"{int(v):,}"


def f_pct(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:+.2f}%"


def f_num(v, d: int = 3) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def f_date(v) -> str:
    return "open" if v is None or v is pd.NaT or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) else f"{pd.Timestamp(v):%Y-%m-%d}"


def f_str(v) -> str:
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)
