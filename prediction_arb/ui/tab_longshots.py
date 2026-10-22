"""Tab 4 - Long-shot scanner: 'extremely unlikely' events (market-implied) that still trade sizable volume.

NOT an arbitrage tab. Selling a tail (buy NO) earns a small premium and loses ~100% of the stake if the event
happens; the detail panel shows that explicitly, plus the EV if the true probability is higher than the market's.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton,
                             QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from config import ScanSettings
from core.longshots import scenario_table
from core.models import LongShot, Venue
from core.scanner import LiveScanner, ScanResult
from database import Database
from ui.common import (ACCENT, GREEN, MUTED, RED, RIGHT, YELLOW, ApyBadgeDelegate, AsyncWorker, Col, DataTable,
                       RowTableModel, SortProxy, fmt_apy, fmt_pct, fmt_usd, venue_label)
from core.live_quote import fetch_live_book

VENUE = {Venue.KALSHI: "Kalshi", Venue.POLYMARKET: "Poly"}


def _short_flags(o: LongShot) -> str:
    return "; ".join(f.split(" (")[0] for f in o.flags)


def _name(o: LongShot) -> str:
    base = o.title if not o.outcome_label or o.outcome_label.lower() in o.title.lower() else f"{o.title} [{o.outcome_label}]"
    return f"{o.event_title} > {base}" if o.event_title and o.event_title not in o.title else base


def _tag_color(o: LongShot) -> Optional[QColor]:
    if "absurd" in o.tags:
        return QColor(GREEN)
    if "catastrophic" in o.tags:
        return QColor(RED)
    return None


COLS = [
    Col("Venue", lambda o: venue_label(o.venue, o.us_tradable), lambda v: v,
        fg=lambda o: None if o.us_tradable else QColor(RED), tip=lambda o: o.market_ref),
    Col("Event > Market", _name, lambda v: v, tip=lambda o: o.title),
    Col("P(YES)", lambda o: o.prob, lambda v: f"{v * 100:.2f}%", RIGHT, numeric=True),
    Col("YES bid / ask", lambda o: o.yes_ask if o.yes_ask is not None else -1.0, lambda v: "", RIGHT, numeric=True),
    Col("NO ask", lambda o: o.no_ask, lambda v: "n/a" if v is None else f"{v:.3f}", RIGHT, numeric=True),
    Col("Volume ($)", lambda o: o.volume_usd, lambda v: f"{v:,.0f}", RIGHT, numeric=True,
        tip=lambda o: "Kalshi: contracts x last price (estimate)" if o.venue is Venue.KALSHI else "Polymarket cash volume"),
    Col("24h Vol ($)", lambda o: o.volume_24h_usd, lambda v: f"{v:,.0f}", RIGHT, numeric=True),
    Col("OI ($)", lambda o: o.open_interest, lambda v: "-" if v <= 0 else f"{v:,.0f}", RIGHT, numeric=True,
        tip=lambda o: "Kalshi open interest (contracts x $1). Polymarket does not expose per-market OI in the listing."),
    Col("Days", lambda o: o.days, lambda v: "-" if v is None else f"{v:,.0f}", RIGHT, numeric=True),
    Col("NO net carry", lambda o: o.carry_net, fmt_pct, RIGHT, numeric=True,
        tip=lambda o: "Return from buying NO and holding to resolution IF it resolves NO, net of fees"),
    Col("Carry APY", lambda o: o.carry_apy, fmt_apy, RIGHT, numeric=True),
    Col("NO depth", lambda o: o.no_depth_usd, lambda v: "?" if v <= 0 else f"{v:,.0f}", RIGHT, numeric=True,
        tip=lambda o: "Known only where the listing carries sizes (Kalshi)"),
    Col("Tag", lambda o: ",".join(o.tags), lambda v: v, fg=_tag_color),
    Col("Flags", _short_flags, lambda v: v, tip=lambda o: "\n".join(o.flags)),
    Col("Same event elsewhere", lambda o: o.other_venue, lambda v: v),
]
APY_COL = 10
WIDTHS = [112, 320, 86, 100, 62, 98, 84, 92, 52, 80, 86, 74, 84, 230, 170]


def _bidask(o: LongShot) -> str:
    b = "-" if o.yes_bid is None else f"{o.yes_bid:.3f}"
    a = "-" if o.yes_ask is None else f"{o.yes_ask:.3f}"
    return f"{b} / {a}"


class LongShotTab(QWidget):
    status = pyqtSignal(str)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.worker: Optional[AsyncWorker] = None
        self.all: list[LongShot] = []

        root = QVBoxLayout(self)
        root.addWidget(self._filters())
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet(f"background:#3a1b20;border:1px solid {RED};border-radius:4px;padding:6px 10px;"
                                f"color:#ffb3ba;font-weight:600;")
        self.warn.hide()
        root.addWidget(self.warn)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.model = _Model(COLS)
        self.proxy = SortProxy(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.predicate = self._accept
        self.table = DataTable()
        self.table.setModel(self.proxy)
        self.table.setItemDelegateForColumn(APY_COL, ApyBadgeDelegate(self.table))
        self.table.sortByColumn(5, Qt.SortOrder.DescendingOrder)          # total volume: 24h is unknown on Polymarket US
        self.table.selectionModel().selectionChanged.connect(self._on_select)
        for c, w in enumerate(WIDTHS):
            self.table.setColumnWidth(c, w)
        split.addWidget(self.table)
        split.addWidget(self._detail())
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 2)
        split.setSizes([1180, 470])
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        bot = QHBoxLayout()
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setFixedWidth(260)
        self.lbl = QLabel("Press 'Scan long-shots' (about 30 s) - or run a Live scan, which fills this tab too.")
        self.lbl.setObjectName("muted")
        bot.addWidget(self.progress)
        bot.addWidget(self.lbl, 1)
        root.addLayout(bot)

    # ── filters ───────────────────────────────────────────────────────────────
    def _filters(self) -> QGroupBox:
        g = QGroupBox("Extremely-unlikely events with sizable volume  (market-implied probability, NOT true probability)")
        h = QGridLayout(g)

        def dspin(v, lo, hi, dec, suf="", step=None):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec); s.setValue(v); s.setSuffix(suf)
            s.setGroupSeparatorShown(True)
            if step:
                s.setSingleStep(step)
            s.valueChanged.connect(self._refilter)
            return s

        def lab(t):
            l = QLabel(t); l.setObjectName("muted"); return l

        self.s_prob = dspin(3.0, 0.1, 10.0, 2, " %", 0.25)
        self.s_minp = dspin(0.1, 0.0, 10.0, 2, " %", 0.05)
        self.s_vol = dspin(25_000, 0, 1e10, 0, " $", 5_000)
        self.s_vol.setToolTip("Minimum lifetime cash volume. Kalshi markets also qualify if open-interest notional "
                              "is at least 2x this value.")
        self.s_v24 = dspin(0, 0, 1e10, 0, " $", 1_000)
        self.s_dmin = dspin(0, 0, 36500, 0, " d", 1)
        self.s_dmax = dspin(0, 0, 36500, 0, " d (0=any)", 30)
        self.c_k = QCheckBox("Kalshi"); self.c_p = QCheckBox("Polymarket")
        for c in (self.c_k, self.c_p):
            c.setChecked(True); c.stateChanged.connect(self._refilter)
        self.c_no = QCheckBox("NO buyable only")
        self.c_od = QCheckBox("Hide overdue"); self.c_od.setChecked(True)
        self.c_ld = QCheckBox("Hide >1y")
        self.c_in = QCheckBox("Hide inactive")
        self.c_me = QCheckBox("Hide multi-outcome tails")
        for c in (self.c_no, self.c_od, self.c_ld, self.c_in, self.c_me):
            c.stateChanged.connect(self._refilter)
        self.tag = QComboBox(); self.tag.addItems(["Any tag", "Absurd only", "Catastrophic only", "Tagged (either)", "Untagged"])
        self.tag.currentIndexChanged.connect(self._refilter)
        self.q = QLineEdit(); self.q.setPlaceholderText("search title..."); self.q.textChanged.connect(self._refilter)
        self.kp = QSpinBox(); self.kp.setRange(1, 200); self.kp.setValue(60)
        self.pp = QSpinBox(); self.pp.setRange(1, 200); self.pp.setValue(60)
        self.src = QComboBox()
        self.src.addItem("Polymarket US (tradable in the US)", "us")
        self.src.addItem("International polymarket.com (NOT US-tradable)", "global")
        self.src.currentIndexChanged.connect(self._source_changed)
        self.c_us = QCheckBox("US-tradable only"); self.c_us.setChecked(True)
        self.c_us.stateChanged.connect(self._refilter)
        self.b_scan = QPushButton("Scan long-shots"); self.b_scan.setObjectName("primary")
        self.b_scan.clicked.connect(self.start_scan)
        self.b_cancel = QPushButton("Cancel"); self.b_cancel.setObjectName("danger"); self.b_cancel.setEnabled(False)
        self.b_cancel.clicked.connect(lambda: self.worker and self.worker.cancel())
        self.b_csv = QPushButton("Export CSV"); self.b_csv.clicked.connect(self._export)

        h.addWidget(lab("Implied P(YES) between"), 0, 0); h.addWidget(self.s_minp, 0, 1)
        h.addWidget(lab("and"), 0, 2); h.addWidget(self.s_prob, 0, 3)
        h.addWidget(lab("Min volume"), 0, 4); h.addWidget(self.s_vol, 0, 5)
        h.addWidget(lab("Min 24h vol"), 0, 6); h.addWidget(self.s_v24, 0, 7)
        h.addWidget(lab("Days"), 0, 8); h.addWidget(self.s_dmin, 0, 9); h.addWidget(self.s_dmax, 0, 10)
        h.addWidget(self.b_scan, 0, 11); h.addWidget(self.b_cancel, 0, 12)
        h.addWidget(self.c_k, 1, 0); h.addWidget(self.c_p, 1, 1)
        h.addWidget(self.c_no, 1, 2, 1, 2); h.addWidget(self.c_od, 1, 4); h.addWidget(self.c_ld, 1, 5)
        h.addWidget(self.c_in, 1, 6); h.addWidget(self.c_me, 1, 7, 1, 2)
        h.addWidget(self.tag, 1, 9); h.addWidget(self.q, 1, 10, 1, 2); h.addWidget(self.b_csv, 1, 12)
        h.addWidget(lab("Kalshi pages"), 2, 0); h.addWidget(self.kp, 2, 1)
        h.addWidget(lab("Polymarket pages"), 2, 2, 1, 2); h.addWidget(self.pp, 2, 4)
        h.addWidget(lab("Polymarket source"), 2, 5, 1, 2); h.addWidget(self.src, 2, 7, 1, 3)
        h.addWidget(self.c_us, 2, 10, 1, 2)
        return g

    def _source_changed(self):
        intl = self.src.currentData() == "global"
        self.c_us.setChecked(not intl)
        if intl:
            self.warn.setText("International polymarket.com selected: REFERENCE ONLY - it prohibits US persons and its "
                              "listing quotes are unreliable (up to 85c off the real book in our audit).")
            self.warn.show()

    # ── detail panel ──────────────────────────────────────────────────────────
    def _detail(self) -> QGroupBox:
        g = QGroupBox("Long-shot detail")
        g.setMinimumWidth(410)
        lay = QVBoxLayout(g)
        self.d_title = QLabel("Select a row")
        self.d_title.setWordWrap(True)
        self.d_title.setStyleSheet("font-size:14px;font-weight:700;")
        lay.addWidget(self.d_title)
        self.d_ref = QLabel("")
        self.d_ref.setObjectName("muted")
        self.d_ref.setWordWrap(True)
        self.d_ref.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.d_ref)
        grid = QGridLayout()
        self.kv: dict[str, QLabel] = {}
        names = ["Implied P(YES)", "NO ask", "Fee / unit", "Net carry", "Carry APY", "Days left", "Volume", "24h volume"]
        for i, n in enumerate(names):
            a = QLabel(n); a.setObjectName("muted")
            v = QLabel("-"); v.setStyleSheet("font-weight:700;")
            self.kv[n] = v
            grid.addWidget(a, i // 4 * 2, i % 4); grid.addWidget(v, i // 4 * 2 + 1, i % 4)
        lay.addLayout(grid)
        self.d_flags = QLabel("")
        self.d_flags.setWordWrap(True)
        self.d_flags.setStyleSheet(f"color:{YELLOW};")
        lay.addWidget(self.d_flags)
        lay.addWidget(QLabel("If the TRUE probability were higher than the market's (EV of buying NO per $1):"))
        self.scen = QTableWidget(0, 4)
        self.scen.setHorizontalHeaderLabels(["Multiple of implied", "True P(YES)", "P(lose stake)", "EV per $1"])
        self.scen.verticalHeader().setVisible(False)
        self.scen.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.scen.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.scen.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scen.setMinimumHeight(215)
        lay.addWidget(self.scen)
        b = QHBoxLayout()
        self.b_live = QPushButton("Refresh live book"); self.b_live.clicked.connect(self._refresh_book)
        self.b_live.setToolTip("Fetch this market's order book right now (Kalshi / Polymarket US)")
        self.b_open = QPushButton("Open market page"); self.b_open.clicked.connect(self._open)
        b.addWidget(self.b_live); b.addWidget(self.b_open)
        lay.addLayout(b)
        self.d_live = QLabel("")
        self.d_live.setWordWrap(True)
        self.d_live.setObjectName("muted")
        lay.addWidget(self.d_live)
        warn = QLabel("NOT arbitrage: buying NO earns the small carry if the event does not happen and loses ~100% of "
                      "the stake if it does. 'Extremely unlikely' here is the MARKET's view; catastrophic-tagged events "
                      "(war, nuclear, coups) are exactly where tails are fat.")
        warn.setWordWrap(True)
        warn.setStyleSheet(f"color:{RED};font-size:11px;")
        lay.addWidget(warn)
        self.rules = QTextBrowser()
        self.rules.setMinimumHeight(120)
        lay.addWidget(self.rules, 1)
        self._cur: Optional[LongShot] = None
        return g

    def _on_select(self, *_):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        o: LongShot = self.model.row_at(self.proxy.mapToSource(rows[0]).row())
        self._cur = o
        self.d_title.setText(_name(o))
        self.d_ref.setText(f"{venue_label(o.venue, o.us_tradable)}  |  id: {o.market_ref}")
        self.d_live.setText("Listing quote - press 'Refresh live book' before trading.")
        self.d_live.setStyleSheet("")
        kv = self.kv
        kv["Implied P(YES)"].setText(f"{o.prob * 100:.2f}%")
        kv["NO ask"].setText("n/a" if o.no_ask is None else f"{o.no_ask:.4f}")
        kv["Fee / unit"].setText("-" if o.no_ask is None else f"{o.fee:.4f}")
        kv["Net carry"].setText(fmt_pct(o.carry_net))
        kv["Carry APY"].setText(fmt_apy(o.carry_apy))
        kv["Days left"].setText("-" if o.days is None else f"{o.days:,.0f}")
        kv["Volume"].setText(("≥ " if not o.stats_exact else "") + fmt_usd(o.volume_usd, 0))
        kv["24h volume"].setText("n/a" if o.volume_24h_usd < 0 else fmt_usd(o.volume_24h_usd, 0))
        self.d_flags.setText("  |  ".join(o.flags + [f"tag: {t}" for t in o.tags]
                                          + ([f"also: {o.other_venue}"] if o.other_venue else [])))
        rows_ = scenario_table(o)
        self.scen.setRowCount(len(rows_))
        for r, s in enumerate(rows_):
            vals = [f"{s['multiple']:g}x", f"{s['true_prob'] * 100:.2f}%", f"{s['p_loss'] * 100:.2f}%",
                    f"{s['ev_pct'] * 100:+.2f}%"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(int(RIGHT))
                if c == 3:
                    it.setForeground(QColor(GREEN if s["ev_pct"] > 0 else RED))
                self.scen.setItem(r, c, it)
        self.rules.setHtml(f"<p style='color:{MUTED}'>{html.escape(o.rules[:1500] or '(no rules text in the listing)')}</p>")

    def _refresh_book(self):
        o = self._cur
        if o is None or (getattr(self, "_lw", None) and self._lw.isRunning()):
            return
        self.b_live.setEnabled(False)
        self.d_live.setText("Fetching live order book...")
        w = AsyncWorker(lambda prog, evt: fetch_live_book(o))
        w.succeeded.connect(lambda r: self._live_done(o, r))
        w.failed.connect(lambda e: (self.b_live.setEnabled(True), self.d_live.setText(f"Live refresh failed: {e}")))
        w.cancelled.connect(lambda: self.b_live.setEnabled(True))
        self._lw = w
        w.start()

    def _live_done(self, o: LongShot, r: Optional[dict]):
        self.b_live.setEnabled(True)
        if r is None:
            self.d_live.setText("Live book not available for this market (international Polymarket is not supported).")
            return
        if self._cur is not o:                                   # user moved on while we were fetching
            return
        fmt = lambda v: "-" if v is None else f"{v:.4f}"          # noqa: E731
        nl = r["no_ask"]
        if nl is None:
            msg = "NO is NOT buyable right now (no YES bid)."
        else:
            rate = (o.fee / (o.no_ask * (1 - o.no_ask))) if (o.no_ask and o.fee) else 0.07
            fee = rate * nl * (1 - nl)
            net = (1 - nl - fee) / (nl + fee)
            self.kv["NO ask"].setText(f"{nl:.4f}")
            self.kv["Fee / unit"].setText(f"{fee:.4f}")
            self.kv["Net carry"].setText(fmt_pct(net))
            d = (nl - o.no_ask) if o.no_ask is not None else 0.0
            msg = (f"NO ask listing {fmt(o.no_ask)} -> live {nl:.4f} ({d:+.4f}), size {r['no_ask_size']:,.0f}; "
                   f"net carry now {net * 100:.2f}%")
        self.d_live.setText(f"LIVE {r['ts']:%H:%M:%S} UTC | YES bid/ask {fmt(r['yes_bid'])} / {fmt(r['yes_ask'])} | {msg}")
        self.d_live.setStyleSheet(f"color:{GREEN};")

    def _open(self):
        if self._cur and self._cur.url:
            QDesktopServices.openUrl(QUrl(self._cur.url))

    # ── data ──────────────────────────────────────────────────────────────────
    def _accept(self, o: LongShot) -> bool:
        p = o.prob * 100
        if p > self.s_prob.value() + 1e-9 or p < self.s_minp.value() - 1e-9:
            return False
        if (o.venue is Venue.KALSHI and not self.c_k.isChecked()) or (o.venue is Venue.POLYMARKET and not self.c_p.isChecked()):
            return False
        if not (o.volume_usd >= self.s_vol.value() or o.open_interest >= 2 * self.s_vol.value()):
            return False
        if 0 <= o.volume_24h_usd < self.s_v24.value():          # < 0 = unknown (Polymarket US): never excluded
            return False
        if self.c_us.isChecked() and not o.us_tradable:
            return False
        if self.c_no.isChecked() and o.no_ask is None:
            return False
        flags = " ".join(o.flags)
        if self.c_od.isChecked() and "OVERDUE" in flags:
            return False
        if self.c_ld.isChecked() and "LONG-DATED" in flags:
            return False
        if self.c_in.isChecked() and "INACTIVE" in flags:
            return False
        if self.c_me.isChecked() and o.me_event:
            return False
        if self.s_dmin.value() > 0 and (o.days is None or o.days < self.s_dmin.value()):
            return False
        if self.s_dmax.value() > 0 and (o.days is None or o.days > self.s_dmax.value()):
            return False
        t = self.tag.currentIndex()
        if t == 1 and "absurd" not in o.tags or t == 2 and "catastrophic" not in o.tags:
            return False
        if t == 3 and not o.tags or t == 4 and o.tags:
            return False
        q = self.q.text().strip().lower()
        if q and q not in f"{o.event_title} {o.title} {o.outcome_label}".lower():
            return False
        return True

    def _refilter(self, *_):
        self.proxy.invalidateFilter()
        self._label()

    def _label(self):
        self.lbl.setText(f"{self.proxy.rowCount():,} shown of {len(self.all):,} collected "
                         f"(collected at implied P<=10%, volume>=$10k or OI>=$25k; re-scan to change that)")

    def set_results(self, longshots: list[LongShot], warnings: Optional[list[str]] = None):
        self.all = longshots
        self.model.set_rows(longshots)
        if warnings:
            self.warn.setText("DATA WARNING - results may be incomplete:  " + "   |   ".join(warnings))
            self.warn.show()
        else:
            self.warn.hide()
        self._label()

    def on_live_scan(self, res: ScanResult):
        """Reuse the markets a Live scan already downloaded."""
        self.set_results(res.longshots, res.warnings)

    # ── scanning ──────────────────────────────────────────────────────────────
    def start_scan(self):
        if self.worker and self.worker.isRunning():
            return
        sc = LiveScanner(ScanSettings(longshots_only=True, kalshi_max_pages=self.kp.value(),
                                      poly_max_pages=self.pp.value(), poly_source=self.src.currentData(),
                                      us_only=self.c_us.isChecked()), self.db)
        self.worker = AsyncWorker(lambda prog, evt: sc.scan_once(prog, evt))
        self.worker.progress.connect(lambda p, m: (self.progress.setValue(p), self.lbl.setText(m)))
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(self._fail)
        self.worker.cancelled.connect(lambda: self._end("Scan cancelled."))
        self.b_scan.setEnabled(False); self.b_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.worker.start()

    def _end(self, msg: str):
        self.b_scan.setEnabled(True); self.b_cancel.setEnabled(False)
        self.lbl.setText(msg)
        self.status.emit(msg)

    def _fail(self, err: str):
        self.warn.setText(f"SCAN FAILED - no data: {err}")
        self.warn.show()
        self._end(f"Scan failed: {err}")

    def _done(self, res: ScanResult):
        self.set_results(res.longshots, res.warnings)
        self.b_scan.setEnabled(True); self.b_cancel.setEnabled(False)
        self._label()
        self.status.emit(f"Long-shot scan: {len(res.longshots):,} collected in {res.stats.get('seconds')}s")

    def _export(self):
        rows = [self.model.row_at(self.proxy.mapToSource(self.proxy.index(r, 0)).row())
                for r in range(self.proxy.rowCount())]
        if not rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export long-shots", "longshots.csv", "CSV (*.csv)")
        if path:
            df = pd.DataFrame([{**o.model_dump(exclude={"rules", "flags", "tags"}), "venue": o.venue.value,
                                "flags": "; ".join(o.flags), "tags": ",".join(o.tags)} for o in rows])
            df.to_csv(path, index=False)
            self.status.emit(f"Exported {len(rows):,} long-shots to {path}")


class _Model(RowTableModel):
    """Row-dependent cells: combined bid/ask (3), volume with '>=' when only a lower bound (5), unknown 24h (6)."""

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            o = self.rows[index.row()]
            c = index.column()
            if c == 3:
                return _bidask(o)
            if c == 5:
                return ("\u2265 " if not o.stats_exact else "") + f"{o.volume_usd:,.0f}"
            if c == 6:
                return "n/a" if o.volume_24h_usd < 0 else f"{o.volume_24h_usd:,.0f}"
        return super().data(index, role)
