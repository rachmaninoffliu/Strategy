"""Tab 1 - Live arbitrage scanner: filters, streaming table, and the manual order ticket."""
from __future__ import annotations

import html
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
                             QSizePolicy, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser,
                             QVBoxLayout, QWidget)

from config import HURDLE_APY, ScanSettings
from core.models import ArbOpportunity, ArbType, TradeLog, Venue, utcnow
from core.scanner import LiveScanner, ScanResult
from database import Database
from ui.common import (ACCENT, GREEN, MUTED, RED, RIGHT, ApyBadgeDelegate, AsyncWorker, Col, DataTable,
                       RowTableModel, SortProxy, fmt_apy, fmt_pct, fmt_usd, venue_label)


def _vl(l) -> str:
    return venue_label(l.venue, l.us_tradable)


def leg_text(o: ArbOpportunity, i: int) -> str:
    """'Kalshi YES @0.4000 (avg 0.4120)': best ask first (what the venue's own page shows), then the fill average."""
    if i >= len(o.legs):
        return "-"
    l = o.legs[i]
    txt = f"{_vl(l)} {l.side.value} @{l.top_price:.4f}"
    if abs(l.price - l.top_price) >= 0.0005:
        txt += f" (avg {l.price:.4f})"
    if i == 1 and len(o.legs) > 2:
        txt += f"  (+{len(o.legs) - 2} more)"
    return txt


LIVE_COLS = [
    Col("Type", lambda o: o.arb_type.value, lambda v: v),
    Col("Event Name", lambda o: o.event_name, lambda v: v, tip=lambda o: "\n".join(o.notes)),
    Col("Leg 1  (Venue/Side/Price)", lambda o: leg_text(o, 0), lambda v: v,
        tip=lambda o: o.legs[0].title if o.legs else ""),
    Col("Leg 2  (Venue/Side/Price)", lambda o: leg_text(o, 1), lambda v: v,
        tip=lambda o: "\n".join(f"{_vl(l)} {l.side.value} ask {l.top_price:.4f} / avg {l.price:.4f}  {l.title[:80]}"
                                for l in o.legs[1:])),
    Col("Min Depth ($)", lambda o: o.min_depth_usd if o.depth_verified else None,
        lambda v: "n/a" if v is None else f"{v:,.0f}", RIGHT, numeric=True,
        fg=lambda o: None if o.depth_verified else QColor(MUTED),
        tip=lambda o: "L2-verified" if o.depth_verified else "Top-of-book only (L2 not fetched)"),
    Col("Total Cost", lambda o: o.total_cost, fmt_usd, RIGHT, numeric=True),
    Col("Guaranteed Payout", lambda o: o.guaranteed_payout, fmt_usd, RIGHT, numeric=True),
    Col("Net Return %", lambda o: o.net_return, fmt_pct, RIGHT, numeric=True),
    Col("Days to Settle", lambda o: o.days_to_settle, lambda v: f"{v:,.1f}", RIGHT, numeric=True),
    Col("APY %", lambda o: o.apy, fmt_apy, RIGHT, numeric=True),
    Col("Conf", lambda o: o.confidence, lambda v: f"{v:.2f}", RIGHT, numeric=True),
    Col("US", lambda o: o.us_tradable, lambda v: "yes" if v else "NOT US", numeric=True,
        fg=lambda o: QColor(GREEN) if o.us_tradable else QColor(RED),
        tip=lambda o: "All legs are tradable by US persons (Kalshi, Polymarket US)" if o.us_tradable else
        "Includes international polymarket.com, which prohibits US persons"),
]
APY_COL = 9
COL_WIDTHS = [64, 140, 160, 182, 80, 76, 120, 84, 92, 84, 40, 56]


class OrderTicket(QGroupBox):
    """Manual order ticket: copyable limit orders, deep links, resizing, rulebook side-by-side."""
    logged = pyqtSignal()

    def __init__(self, db: Database, parent=None):
        super().__init__("Manual Order Ticket", parent)
        self.db = db
        self.opp: Optional[ArbOpportunity] = None
        self.scanner: Optional[LiveScanner] = None
        self.cur: Optional[ArbOpportunity] = None
        self._busy = False

        lay = QVBoxLayout(self)
        self.title = QLabel("Select an opportunity")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-size:14px;font-weight:700;")
        self.sub = QLabel("")
        self.sub.setObjectName("muted")
        self.sub.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.sub)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size (units)"))
        self.units = QDoubleSpinBox()
        self.units.setDecimals(2)
        self.units.setRange(0, 1e9)
        self.units.setSingleStep(10)
        self.units.setGroupSeparatorShown(True)
        self.units.valueChanged.connect(self._recalc)
        size_row.addWidget(self.units, 1)
        self.btn_max = QPushButton("Max @ hurdle")
        self.btn_max.clicked.connect(self._reset_size)
        size_row.addWidget(self.btn_max)
        lay.addLayout(size_row)

        grid = QGridLayout()
        self.kv: dict[str, QLabel] = {}
        for i, name in enumerate(["Total cost", "Min payout", "Max payout", "Profit (min)", "Net return", "APY",
                                  "Days", "Fees"]):
            n = QLabel(name)
            n.setObjectName("muted")
            v = QLabel("-")
            v.setStyleSheet("font-weight:700;")
            self.kv[name] = v
            grid.addWidget(n, i // 4 * 2, i % 4)
            grid.addWidget(v, i // 4 * 2 + 1, i % 4)
        lay.addLayout(grid)

        self.us_warn = QLabel("NOT US-TRADABLE: this includes international polymarket.com, which prohibits US persons.")
        self.us_warn.setWordWrap(True)
        self.us_warn.setStyleSheet(f"background:#3a1b20;border:1px solid {RED};border-radius:4px;padding:5px 8px;"
                                   f"color:#ffb3ba;font-weight:600;")
        self.us_warn.hide()
        lay.addWidget(self.us_warn)

        self.legs = QTableWidget(0, 7)
        self.legs.setHorizontalHeaderLabels(["Action", "Market", "Best ask", "Limit", "Avg fill", "Link", ""])
        self.legs.verticalHeader().setVisible(False)
        self.legs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.legs.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        h = self.legs.horizontalHeader()
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (0, 2, 3, 4, 5, 6):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.legs.setMinimumHeight(120)
        lay.addWidget(self.legs, 1)

        self.b_copy = QPushButton("Copy all orders")
        self.b_copy.clicked.connect(self._copy_all)
        self.b_open = QPushButton("Open both pages")
        self.b_open.clicked.connect(self._open_all)
        self.b_log = QPushButton("Log as taken")
        self.b_log.setObjectName("primary")
        self.b_log.clicked.connect(self._log_taken)
        self.b_live = QPushButton("Refresh live quote")
        self.b_live.setToolTip("Re-fetch the order book of every leg RIGHT NOW and re-price at this size")
        self.b_live.clicked.connect(self._refresh_live)
        grid_b = QGridLayout()
        for i, b in enumerate((self.b_live, self.b_copy, self.b_open, self.b_log)):
            grid_b.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid_b)
        self.live_lbl = QLabel("")
        self.live_lbl.setWordWrap(True)
        self.live_lbl.setObjectName("muted")
        lay.addWidget(self.live_lbl)
        self._live_worker: Optional[AsyncWorker] = None

        self.rules = QTextBrowser()
        self.rules.setOpenExternalLinks(True)
        self.rules.setMinimumHeight(150)
        lay.addWidget(QLabel("Resolution rules - confirm both venues resolve identically before trading:"))
        lay.addWidget(self.rules, 1)
        warn = QLabel("Read-only tool: no orders are placed here. Legs are NOT atomic - a partial fill leaves "
                      "directional risk. Place the thinner leg first.")
        warn.setWordWrap(True)
        warn.setStyleSheet(f"color:{RED};font-size:11px;")
        lay.addWidget(warn)
        self._set_enabled(False)

    # ── state ─────────────────────────────────────────────────────────────────
    def _set_enabled(self, on: bool):
        for w in (self.units, self.btn_max, self.b_copy, self.b_open, self.b_log, self.b_live, self.legs):
            w.setEnabled(on)

    def show_opportunity(self, opp: Optional[ArbOpportunity], scanner: Optional[LiveScanner]):
        self.opp, self.scanner, self.cur = opp, scanner, opp
        if opp is None:
            self.title.setText("Select an opportunity")
            self.sub.setText("")
            self.legs.setRowCount(0)
            self.rules.clear()
            self._set_enabled(False)
            for v in self.kv.values():
                v.setText("-")
            return
        self._set_enabled(True)
        self.live_lbl.setText(f"Scan quote ({opp.detected_at:%H:%M:%S} UTC) - press 'Refresh live quote' before trading.")
        self.us_warn.setVisible(not opp.us_tradable)
        self.title.setText(opp.event_name)
        self.sub.setText(f"{opp.arb_type.value}  {opp.variant}  |  match confidence {opp.confidence:.2f}  |  "
                         f"{'L2-verified depth' if opp.depth_verified else 'top-of-book only'}")
        max_units = min((sum(s for _, s in l.ladder) for l in opp.legs if l.ladder), default=opp.size)
        self._busy = True
        self.units.setMaximum(max(max_units, opp.size))
        self.units.setValue(opp.size)
        self._busy = False
        self._render(opp)
        self._render_rules(opp)

    def _reset_size(self):
        if self.opp:
            self.units.setValue(self.opp.size)

    def _recalc(self):
        if self._busy or not self.opp or not self.scanner:
            return
        u = self.units.value()
        if u <= 0:
            return
        new = self.scanner.reprice(self.opp, u)
        if new is None:
            self.kv["Net return"].setText("below min order size")
            return
        self.cur = new
        self._render(new)

    # ── rendering ─────────────────────────────────────────────────────────────
    def _render(self, o: ArbOpportunity):
        kv = self.kv
        kv["Total cost"].setText(fmt_usd(o.total_cost))
        kv["Min payout"].setText(fmt_usd(o.guaranteed_payout))
        kv["Max payout"].setText(fmt_usd(o.max_payout))
        kv["Profit (min)"].setText(fmt_usd(o.profit))
        kv["Net return"].setText(fmt_pct(o.net_return))
        kv["APY"].setText(fmt_apy(o.apy))
        kv["Days"].setText(f"{o.days_to_settle:,.1f}")
        kv["Fees"].setText(fmt_usd(o.fees))
        kv["Profit (min)"].setStyleSheet(f"font-weight:700;color:{'#2ea06b' if o.profit > 0 else RED};")
        self.legs.setRowCount(len(o.legs))
        for r, l in enumerate(o.legs):
            action = f"BUY {l.side.value}"
            a_it = QTableWidgetItem(f"{_vl(l)}  {action}  x{l.size:,.0f}")
            if not l.us_tradable:
                a_it.setForeground(QColor(RED))
            self.legs.setItem(r, 0, a_it)
            it = QTableWidgetItem(l.title[:90])
            it.setToolTip(f"{l.title}\n{l.market_ref}")
            self.legs.setItem(r, 1, it)
            for c, txt in ((2, f"{l.top_price:.4f}"), (3, f"{l.limit_price:.4f}"), (4, f"{l.price:.4f}")):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(int(RIGHT))
                self.legs.setItem(r, c, item)
            link = QLabel(f'<a style="color:{ACCENT}" href="{html.escape(l.url)}">open</a>')
            link.setOpenExternalLinks(True)
            link.setToolTip(l.url)
            self.legs.setCellWidget(r, 5, link)
            b = QPushButton("Copy")
            b.setFixedWidth(54)
            b.clicked.connect(lambda _=False, leg=l: self._copy(self._leg_line(leg)))
            self.legs.setCellWidget(r, 6, b)
        self.legs.resizeRowsToContents()

    def _render_rules(self, o: ArbOpportunity):
        parts = []
        for l in o.legs:
            rules = html.escape(l.rules[:900] or "(no rules text returned by the API - open the market page)")
            parts.append(f"<p><b style='color:{ACCENT}'>[{_vl(l)}] {l.side.value}</b> "
                         f"{html.escape(l.title[:120])}<br><span style='color:{MUTED}'>{rules}</span></p>")
        notes = [n for n in o.notes if n and n.strip()]
        if notes:
            parts.append("<hr>" + "<br>".join(f"&bull; {html.escape(n)}" for n in notes))
        self.rules.setHtml("".join(parts))

    # ── actions ───────────────────────────────────────────────────────────────
    @staticmethod
    def _leg_line(l) -> str:
        return (f"{_vl(l)}: BUY {l.side.value}  size {l.size:,.2f}  best ask {l.top_price:.4f}  "
                f"limit {l.limit_price:.4f}  [{l.market_ref or l.market_id}]  {l.url}")

    def _copy(self, text: str):
        QApplication.clipboard().setText(text)

    def _copy_all(self):
        o = self.cur
        if o:
            head = f"{o.arb_type.value} {o.variant} | {o.event_name}\n"
            self._copy(head + "\n".join(self._leg_line(l) for l in o.legs))

    def _open_all(self):
        if self.cur:
            for l in self.cur.legs:
                if l.url:
                    QDesktopServices.openUrl(QUrl(l.url))

    def _refresh_live(self):
        """Re-fetch every leg's order book now; show scan-vs-live cost per unit so stale quotes are visible."""
        if not self.opp or not self.scanner or (self._live_worker and self._live_worker.isRunning()):
            return
        scanner, old, units = self.scanner, self.opp, self.units.value() or self.opp.size
        self.b_live.setEnabled(False)
        self.live_lbl.setText("Fetching live order books...")
        w = AsyncWorker(lambda prog, evt: scanner.refresh_quote(old, units))
        w.succeeded.connect(lambda new: self._live_done(old, new))
        w.failed.connect(lambda e: self._live_fail(f"Live refresh failed: {e}"))
        w.cancelled.connect(lambda: self._live_fail("Cancelled."))
        self._live_worker = w
        w.start()

    def _live_fail(self, msg: str):
        self.b_live.setEnabled(True)
        self.live_lbl.setText(msg)

    def _live_done(self, old: ArbOpportunity, new: Optional[ArbOpportunity]):
        self.b_live.setEnabled(True)
        if new is None:
            self.live_lbl.setText("Live refresh found no order book for at least one leg (market closed or illiquid).")
            return
        u_old, u_new = old.total_cost / old.size, new.total_cost / new.size
        self.opp, self.cur = new, new
        self._busy = True
        self.units.setValue(new.size)
        self._busy = False
        self._render(new)
        drift = u_new - u_old
        color = RED if drift > 0.0005 else GREEN
        self.live_lbl.setText(
            f"LIVE {new.detected_at:%H:%M:%S} UTC - all-in cost per unit: scan {u_old:.4f} -> live {u_new:.4f} "
            f"({drift:+.4f}); net return {fmt_pct(new.net_return)}, APY {fmt_apy(new.apy)}"
            + ("  - EDGE GONE" if new.net_return <= 0 else ""))
        self.live_lbl.setStyleSheet(f"color:{color};font-weight:600;")

    def _log_taken(self):
        o = self.cur
        if not o:
            return
        t = TradeLog(opp_key=o.key, arb_type=o.arb_type.value, event_name=o.event_name, opened_at=utcnow(),
                     settle_time=o.settle_time, size=o.size, cost=o.total_cost, fees=o.fees,
                     guaranteed_payout=o.guaranteed_payout, max_payout=o.max_payout,
                     net_return_locked=o.net_return, apy_at_entry=o.apy, confidence=o.confidence, source="manual",
                     legs=[{"venue": l.venue.value, "market": l.market_id, "side": l.side.value, "price": l.price,
                            "size": l.size, "title": l.title[:80]} for l in o.legs])
        self.db.add_trade("manual", t)
        QMessageBox.information(self, "Logged", "Position logged in 'Past Opportunities & PnL' (source: manual).")
        self.logged.emit()


class LiveTab(QWidget):
    status = pyqtSignal(str)
    scanned = pyqtSignal(object)
    manual_logged = pyqtSignal()

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.worker: Optional[AsyncWorker] = None
        self.scanner: Optional[LiveScanner] = None
        self.result: Optional[ScanResult] = None
        self._sel_key: Optional[str] = None

        root = QVBoxLayout(self)
        root.addWidget(self._build_filters())
        self.warn = QLabel()                                   # persistent data-quality banner (never auto-hides)
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet(f"background:#3a1b20;border:1px solid {RED};border-radius:4px;padding:6px 10px;"
                                f"color:#ffb3ba;font-weight:600;")
        self.warn.hide()
        root.addWidget(self.warn)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.model = RowTableModel(LIVE_COLS)
        self.proxy = SortProxy(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.predicate = self._accept
        self.table = DataTable()
        self.table.setModel(self.proxy)
        self.table.setItemDelegateForColumn(APY_COL, ApyBadgeDelegate(self.table))
        self.table.sortByColumn(APY_COL, Qt.SortOrder.DescendingOrder)
        self.table.selectionModel().selectionChanged.connect(self._on_select)
        split.addWidget(self.table)
        self.ticket = OrderTicket(db)
        self.ticket.logged.connect(self.manual_logged)
        self.ticket.setMinimumWidth(480)
        split.addWidget(self.ticket)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 3)
        split.setSizes([1160, 520])
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(260)
        self.lbl = QLabel("Ready. Press 'Scan now' - first scan downloads both venues (~30-90 s).")
        self.lbl.setObjectName("muted")
        bottom.addWidget(self.progress)
        bottom.addWidget(self.lbl, 1)
        root.addLayout(bottom)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.start_scan)

    # ── filters ───────────────────────────────────────────────────────────────
    def _build_filters(self) -> QGroupBox:
        g = QGroupBox("Scanner filters")
        h = QGridLayout(g)
        self.c_par = QCheckBox("I  Parity")
        self.c_mec = QCheckBox("II  MECE basket")
        self.c_sub = QCheckBox("III  Subset / ladder")
        for c in (self.c_par, self.c_mec, self.c_sub):
            c.setChecked(True)
            c.stateChanged.connect(self._refilter)
        self.s_apy = self._dspin(HURDLE_APY * 100, 0, 1e6, 1, " %")
        self.s_depth = self._dspin(25, 0, 1e9, 0, " $")
        self.s_conf = self._dspin(0.80, 0, 1, 2)
        self.s_vol = self._dspin(250, 0, 1e9, 0)
        self.c_l2 = QCheckBox("Fetch L2 depth")
        self.c_l2.setChecked(True)
        self.c_ver = QCheckBox("L2-verified only")
        self.c_ver.stateChanged.connect(self._refilter)
        for s in (self.s_apy, self.s_depth, self.s_conf):
            s.valueChanged.connect(self._refilter)
        self.s_kp = QSpinBox(); self.s_kp.setRange(1, 200); self.s_kp.setValue(60)
        self.s_pp = QSpinBox(); self.s_pp.setRange(1, 200); self.s_pp.setValue(60)
        self.src = QComboBox()
        self.src.addItem("Polymarket US (tradable in the US)", "us")
        self.src.addItem("International polymarket.com (NOT US-tradable)", "global")
        self.src.currentIndexChanged.connect(self._source_changed)
        self.c_us = QCheckBox("US-tradable only")
        self.c_us.setChecked(True)
        self.c_us.setToolTip("Drop any result that involves a market a US person cannot trade")
        self.c_us.stateChanged.connect(self._refilter)
        self.c_auto = QCheckBox("Auto-refresh")
        self.c_auto.stateChanged.connect(self._toggle_auto)
        self.s_ref = QSpinBox(); self.s_ref.setRange(30, 3600); self.s_ref.setValue(120); self.s_ref.setSuffix(" s")
        self.b_scan = QPushButton("Scan now")
        self.b_scan.setObjectName("primary")
        self.b_scan.clicked.connect(self.start_scan)
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.setObjectName("danger")
        self.b_cancel.setEnabled(False)
        self.b_cancel.clicked.connect(self.cancel_scan)

        def lab(t):
            l = QLabel(t)
            l.setObjectName("muted")
            return l
        h.addWidget(self.c_par, 0, 0); h.addWidget(self.c_mec, 0, 1); h.addWidget(self.c_sub, 0, 2)
        h.addWidget(lab("Min APY"), 0, 3); h.addWidget(self.s_apy, 0, 4)
        h.addWidget(lab("Min depth"), 0, 5); h.addWidget(self.s_depth, 0, 6)
        h.addWidget(lab("Min match conf."), 0, 7); h.addWidget(self.s_conf, 0, 8)
        h.addWidget(self.b_scan, 0, 9); h.addWidget(self.b_cancel, 0, 10)
        h.addWidget(lab("Min volume (contracts)"), 1, 0, 1, 1); h.addWidget(self.s_vol, 1, 1)
        h.addWidget(lab("Kalshi pages"), 1, 2); h.addWidget(self.s_kp, 1, 3)
        h.addWidget(lab("Polymarket pages"), 1, 4); h.addWidget(self.s_pp, 1, 5)
        h.addWidget(self.c_l2, 1, 6); h.addWidget(self.c_ver, 1, 7)
        h.addWidget(self.c_auto, 1, 9); h.addWidget(self.s_ref, 1, 10)
        h.addWidget(lab("Polymarket source"), 2, 0); h.addWidget(self.src, 2, 1, 1, 3); h.addWidget(self.c_us, 2, 4, 1, 2)
        return g

    def _source_changed(self):
        intl = self.src.currentData() == "global"
        self.c_us.setChecked(not intl)                     # reference mode shows international legs (flagged red)
        if intl:
            self.warn.setText("International polymarket.com selected: REFERENCE ONLY - it prohibits US persons and its "
                              "listing quotes are unreliable. Results below the next scan are flagged 'NOT US'.")
            self.warn.show()

    @staticmethod
    def _dspin(val, lo, hi, dec, suffix="") -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(val)
        s.setSuffix(suffix)
        s.setGroupSeparatorShown(True)
        return s

    def _accept(self, o: ArbOpportunity) -> bool:
        t = {ArbType.PARITY: self.c_par, ArbType.MECE: self.c_mec, ArbType.SUBSET: self.c_sub}[o.arb_type]
        if not t.isChecked():
            return False
        if o.apy * 100 < self.s_apy.value() or o.confidence < self.s_conf.value():
            return False
        if o.depth_verified and o.min_depth_usd < self.s_depth.value():
            return False
        if self.c_ver.isChecked() and not o.depth_verified:
            return False
        if self.c_us.isChecked() and not o.us_tradable:
            return False
        return True

    def _refilter(self):
        self.proxy.invalidateFilter()
        self._update_label()

    # ── scanning ──────────────────────────────────────────────────────────────
    def _settings(self) -> ScanSettings:
        inc = tuple(n for n, c in (("parity", self.c_par), ("mece", self.c_mec), ("subset", self.c_sub))
                    if c.isChecked())
        return ScanSettings(hurdle_apy=self.s_apy.value() / 100.0, min_depth_usd=self.s_depth.value(),
                            min_confidence=self.s_conf.value(), min_volume=self.s_vol.value(),
                            kalshi_max_pages=self.s_kp.value(), poly_max_pages=self.s_pp.value(),
                            use_l2_books=self.c_l2.isChecked(), include_types=inc or ("parity",),
                            poly_source=self.src.currentData(), us_only=self.c_us.isChecked())

    def start_scan(self):
        if self.worker and self.worker.isRunning():
            return
        self.scanner = LiveScanner(self._settings(), self.db)
        sc = self.scanner
        self.worker = AsyncWorker(lambda prog, evt: sc.scan_once(prog, evt))
        self.worker.progress.connect(self._on_progress)
        self.worker.succeeded.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.cancelled.connect(lambda: self._finish("Scan cancelled."))
        self.b_scan.setEnabled(False)
        self.b_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.worker.start()

    def cancel_scan(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()

    def _on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.lbl.setText(msg)

    def _finish(self, msg: str):
        self.b_scan.setEnabled(True)
        self.b_cancel.setEnabled(False)
        self.lbl.setText(msg)
        self.status.emit(msg)

    def _on_fail(self, err: str):
        self.warn.setText(f"SCAN FAILED - no data: {err}")
        self.warn.show()
        self._finish(f"Scan failed: {err}")

    def _on_done(self, res: ScanResult):
        self.result = res
        self.scanned.emit(res)
        self.set_opportunities(res.opportunities)
        s = res.stats
        if res.warnings:
            self.warn.setText("DATA WARNING - results may be incomplete:  " + "   |   ".join(res.warnings))
            self.warn.show()
        else:
            self.warn.hide()
        warn = ""
        rej = s.get("rejected", {})
        src = "Polymarket US" if s.get("poly_source", "us") == "us" else "Polymarket INTL"
        self._finish(f"Scan {s['seconds']}s: {s['kalshi_markets']:,} Kalshi + {s['poly_markets']:,} {src} markets, "
                     f"{s['pairs']} pairs, {s['subset_edges']:,} ladder edges, {s['baskets']} baskets -> "
                     f"{s['stage1_candidates']} screened, {s['actionable']} actionable. "
                     f"Rejected: {sum(rej.values())} (sanity gates){warn}")

    def set_opportunities(self, opps: list[ArbOpportunity]):
        sel = self._sel_key
        self.model.set_rows(opps)
        for c, w in enumerate(COL_WIDTHS):                     # keep the APY badge on-screen at 1560 px
            self.table.setColumnWidth(c, w)
        if sel:
            for r in range(self.proxy.rowCount()):
                o = self.proxy.mapToSource(self.proxy.index(r, 0)).row()
                if self.model.row_at(o).key == sel:
                    self.table.selectRow(r)
                    break
        self._update_label()

    def _update_label(self):
        n_all, n_vis = self.model.rowCount(), self.proxy.rowCount()
        if n_all and self.result is None:
            self.lbl.setText(f"{n_vis}/{n_all} shown")

    def _on_select(self, *_):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.ticket.show_opportunity(None, None)
            return
        o = self.model.row_at(self.proxy.mapToSource(rows[0]).row())
        self._sel_key = o.key
        self.ticket.show_opportunity(o, self.scanner)

    def _toggle_auto(self):
        if self.c_auto.isChecked():
            self.timer.start(self.s_ref.value() * 1000)
            self.start_scan()
        else:
            self.timer.stop()
