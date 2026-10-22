"""Tab 2 - Past 30-90 days of opportunities: realised / unrealised PnL and a locked-capital timeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget)

from core.backtester import BacktestResult, run_history
from core.models import Side, TradeLog, Venue, iso, utcnow
from core.pricing import annualize
from database import Database
from ui.common import (ACCENT, BLUE, GREEN, MUTED, RED, RIGHT, YELLOW, ApyBadgeDelegate, AsyncWorker, Col,
                       DataTable, KpiCard, RowTableModel, SortProxy, dark_canvas, fmt_apy, fmt_pct, fmt_usd, style_ax)

STATUS_COLOR = {"won": QColor(GREEN), "bonus": QColor(YELLOW), "loss": QColor(RED), "active": QColor(BLUE)}


@dataclass
class HistRow:
    t: TradeLog
    db_id: Optional[int] = None

    @property
    def resolved(self) -> bool:
        return self.t.realized_payout is not None

    @property
    def pnl(self) -> float:
        return self.t.pnl if self.resolved else self.t.unrealized

    @property
    def realized_apy(self) -> Optional[float]:
        if not self.resolved or self.t.cost <= 0:
            return None
        return annualize(self.t.pnl / self.t.cost, max(self.t.lockup_days, 0.25))

    @property
    def kind(self) -> str:
        s = self.t.status
        if s.startswith("Resolved - Won"):
            return "won"
        if "Bonus" in s:
            return "bonus"
        if "Loss" in s:
            return "loss"
        return "active"


def resolve_from_cache(t: TradeLog, db: Database) -> TradeLog:
    """Settle a logged (live / manual) position from cached settlement outcomes when all legs are known."""
    if t.realized_payout is not None or not t.legs:
        return t
    total = 0.0
    latest = None
    for l in t.legs:
        v = db.settlement(Venue(l["venue"]), l["market"])
        if v is None:
            return t
        total += float(l.get("size", t.size)) * (v if l["side"] == Side.YES.value else 1.0 - v)
        m = db.get_market(Venue(l["venue"]), l["market"])
        if m and (m.settled_ts or m.settle_time):
            ts = m.settled_ts or m.settle_time
            latest = ts if latest is None or ts > latest else latest
    t = t.model_copy()
    t.realized_payout = total
    t.released_at = latest
    if total > t.guaranteed_payout + 1e-6 and t.max_payout > t.guaranteed_payout:
        t.status = "Resolved - Upside Bonus Paid"
    elif total >= t.guaranteed_payout - 1e-6:
        t.status = "Resolved - Won"
    else:
        t.status = "Resolved - Loss (rule mismatch / non-exhaustive)"
    return t


def _st_color(r: HistRow) -> QColor:
    return STATUS_COLOR[r.kind]


COLS = [
    Col("Detected (entry)", lambda r: r.t.opened_at.timestamp(), lambda v: datetime.fromtimestamp(v, tz=timezone.utc)
        .strftime("%Y-%m-%d %H:%M"), numeric=True),
    Col("Type", lambda r: r.t.arb_type, lambda v: v),
    Col("Event", lambda r: r.t.event_name, lambda v: v, tip=lambda r: "\n".join(
        f"{l.get('venue', '')} {l.get('side', '')} @ {float(l.get('price', 0)):.4f}  {l.get('title', '')}"
        for l in r.t.legs)),
    Col("Status", lambda r: r.t.status, lambda v: v, fg=_st_color),
    Col("Cost", lambda r: r.t.cost, fmt_usd, RIGHT, numeric=True),
    Col("Guaranteed", lambda r: r.t.guaranteed_payout, fmt_usd, RIGHT, numeric=True),
    Col("PnL (realised / locked-in)", lambda r: r.pnl, fmt_usd, RIGHT, numeric=True,
        fg=lambda r: QColor(GREEN) if r.pnl > 0 else QColor(RED),
        tip=lambda r: "Realised" if r.resolved else "Unrealised: minimum payout - cost if held to settlement"),
    Col("Net Ret %", lambda r: r.t.net_return_locked, fmt_pct, RIGHT, numeric=True),
    Col("Entry APY", lambda r: r.t.apy_at_entry, fmt_apy, RIGHT, numeric=True),
    Col("Realised APY", lambda r: r.realized_apy, fmt_apy, RIGHT, numeric=True),
    Col("Lockup (d)", lambda r: r.t.lockup_days, lambda v: f"{v:,.1f}", RIGHT, numeric=True),
    Col("Settles", lambda r: r.t.settle_time.timestamp() if r.t.settle_time else None,
        lambda v: datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d") if v else "-", numeric=True),
    Col("Conf", lambda r: r.t.confidence, lambda v: f"{v:.2f}", RIGHT, numeric=True),
    Col("Source", lambda r: r.t.source, lambda v: v),
]


class HistoryTab(QWidget):
    status = pyqtSignal(str)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self.worker: Optional[AsyncWorker] = None
        self.rows: list[HistRow] = []

        root = QVBoxLayout(self)
        ctl = QGroupBox("Past opportunities")
        h = QHBoxLayout(ctl)
        self.days = QComboBox()
        for d in (30, 60, 90):
            self.days.addItem(f"Last {d} days", d)
        self.days.currentIndexChanged.connect(self.reload)
        self.hurdle = QDoubleSpinBox(); self.hurdle.setRange(0, 1e5); self.hurdle.setValue(7.0)
        self.hurdle.setSuffix(" % APY hurdle")
        self.size = QDoubleSpinBox(); self.size.setRange(10, 1e7); self.size.setValue(500); self.size.setPrefix("$ ")
        self.size.setGroupSeparatorShown(True)
        self.size.setToolTip("Notional per opportunity used when replaying history")
        self.live = QCheckBox("Include live-logged and manual")
        self.live.setChecked(True)
        self.live.stateChanged.connect(self.reload)
        self.b_run = QPushButton("Backfill from market data")
        self.b_run.setObjectName("primary")
        self.b_run.clicked.connect(self.start_backfill)
        self.b_cancel = QPushButton("Cancel")
        self.b_cancel.setObjectName("danger")
        self.b_cancel.setEnabled(False)
        self.b_cancel.clicked.connect(lambda: self.worker and self.worker.cancel())
        for w in (self.days, self.hurdle, QLabel("Size / opp."), self.size, self.live, self.b_run, self.b_cancel):
            h.addWidget(w)
        h.addStretch(1)
        root.addWidget(ctl)
        note = QLabel("NOT US-TRADABLE DATA: Polymarket price history and settled markets come from international polymarket.com (the only source with history), which prohibits US persons. These results show how Kalshi and Polymarket priced the same events - they are NOT a record of trades a US person could have made.")
        note.setWordWrap(True)
        note.setStyleSheet("background:#3a1b20;border:1px solid #e05561;border-radius:4px;padding:5px 8px;"
                           "color:#ffb3ba;font-weight:600;")
        root.addWidget(note)

        cards = QHBoxLayout()
        self.k = {n: KpiCard(n) for n in ("Total Past Opportunities", "Realized PnL ($)", "Capital Deployed ($)",
                                          "Average Realized APY (%)", "Active / Locked-in PnL", "Win rate")}
        for c in self.k.values():
            cards.addWidget(c)
        root.addLayout(cards)

        split = QSplitter(Qt.Orientation.Vertical)
        self.model = RowTableModel(COLS)
        self.proxy = SortProxy(self)
        self.proxy.setSourceModel(self.model)
        self.table = DataTable()
        self.table.setModel(self.proxy)
        self.table.setItemDelegateForColumn(8, ApyBadgeDelegate(self.table))
        self.table.setItemDelegateForColumn(9, ApyBadgeDelegate(self.table))
        self.table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        split.addWidget(self.table)
        self.fig, self.canvas = dark_canvas(9, 3.2)
        split.addWidget(self.canvas)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        bot = QHBoxLayout()
        self.progress = QProgressBar(); self.progress.setFixedWidth(260); self.progress.setRange(0, 100)
        self.lbl = QLabel("Loading cached history...")
        self.lbl.setObjectName("muted")
        bot.addWidget(self.progress)
        bot.addWidget(self.lbl, 1)
        root.addLayout(bot)
        self.reload()

    # ── data ──────────────────────────────────────────────────────────────────
    def reload(self, *_):
        days = self.days.currentData() or 30
        since = utcnow() - timedelta(days=days)
        rows: list[HistRow] = []
        srcs = ["backfill"] + (["manual"] if self.live.isChecked() else [])
        for tid, t in self.db.load_trades(since, srcs):
            rows.append(HistRow(resolve_from_cache(t, self.db) if t.source == "manual" else t, tid))
        if self.live.isChecked():
            for opp, first, _last in self.db.load_logged_opportunities(since):
                t = TradeLog(opp_key=opp.key, arb_type=opp.arb_type.value, event_name=opp.event_name,
                             opened_at=datetime.fromisoformat(first.replace("Z", "+00:00")),
                             settle_time=opp.settle_time, size=opp.size, cost=opp.total_cost, fees=opp.fees,
                             guaranteed_payout=opp.guaranteed_payout, max_payout=opp.max_payout,
                             net_return_locked=opp.net_return, apy_at_entry=opp.apy, confidence=opp.confidence,
                             source="live",
                             legs=[{"venue": l.venue.value, "market": l.market_id, "side": l.side.value,
                                    "price": l.price, "size": l.size, "title": l.title[:80]} for l in opp.legs])
                rows.append(HistRow(resolve_from_cache(t, self.db)))
        self.rows = rows
        self.model.set_rows(rows)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(2, 230)
        self.table.setColumnWidth(3, 215)
        self.table.setColumnWidth(8, 90)
        self.table.setColumnWidth(9, 90)
        self._update_kpis()
        self._draw()
        self.lbl.setText(f"{len(rows)} opportunities in the last {days} days"
                         + ("" if rows else " - press 'Backfill from market data' to replay history."))

    def _update_kpis(self):
        rows = self.rows
        res = [r for r in rows if r.resolved]
        act = [r for r in rows if not r.resolved]
        realized = sum(r.t.pnl for r in res)
        cost = sum(r.t.cost for r in rows)
        cap_days = sum(r.t.cost * max(r.t.lockup_days, 0.25) for r in res)     # capital-days deployed
        avg = (sum(r.t.pnl for r in res) / cap_days * 365.25) if cap_days > 0 else None
        wins = sum(1 for r in res if r.t.pnl > 0)
        self.k["Total Past Opportunities"].set(f"{len(rows):,}")
        self.k["Realized PnL ($)"].set(fmt_usd(realized), GREEN if realized > 0 else (RED if realized < 0 else None))
        self.k["Capital Deployed ($)"].set(fmt_usd(cost, 0))
        self.k["Average Realized APY (%)"].set(fmt_apy(avg) if avg is not None else "-")
        self.k["Average Realized APY (%)"].setToolTip("Resolved trades: total PnL / capital-days deployed x 365.25 (simple, capital-time weighted)")
        self.k["Active / Locked-in PnL"].set(f"{len(act)}  /  {fmt_usd(sum(r.t.unrealized for r in act), 0)}", BLUE)
        self.k["Win rate"].set(f"{100 * wins / len(res):.0f}%  ({wins}/{len(res)})" if res else "-")

    # ── chart: locked-capital timeline + cumulative PnL ───────────────────────
    def _draw(self):
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        ax2 = self.fig.add_subplot(1, 2, 2)
        style_ax(ax1, "Locked capital timeline", "$ locked")
        style_ax(ax2, "Cumulative PnL", "$")
        if self.rows:
            now = utcnow()
            t0 = min(r.t.opened_at for r in self.rows).replace(hour=0, minute=0, second=0, microsecond=0)
            idx = np.array([t0 + timedelta(days=i) for i in range(int((now - t0).days) + 2)])
            locked = np.zeros(len(idx))
            for r in self.rows:
                end = r.t.released_at or now
                locked += np.array([(r.t.opened_at <= d < end) * r.t.cost for d in idx])
            ax1.fill_between(idx, locked, step="post", color=ACCENT, alpha=0.35)
            ax1.step(idx, locked, where="post", color=ACCENT, linewidth=1.4)
            res = sorted((r for r in self.rows if r.resolved), key=lambda r: r.t.released_at or r.t.opened_at)
            if res:
                xs = [r.t.released_at or r.t.opened_at for r in res]
                ys = np.cumsum([r.t.pnl for r in res])
                ax2.step(xs, ys, where="post", color=GREEN, linewidth=1.6, label="realised")
            act_pnl = sum(r.t.unrealized for r in self.rows if not r.resolved)
            base = float(np.sum([r.t.pnl for r in self.rows if r.resolved]))
            ax2.axhline(base + act_pnl, color=BLUE, linestyle="--", linewidth=1, label="incl. locked-in (active)")
            ax2.legend(facecolor="#151a21", edgecolor="#2a3441", labelcolor="#d7dee8", fontsize=8, loc="upper left")
        else:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=MUTED)
        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ── backfill ──────────────────────────────────────────────────────────────
    def start_backfill(self):
        if self.worker and self.worker.isRunning():
            return
        days, hurdle, size = self.days.currentData(), self.hurdle.value() / 100.0, self.size.value()
        db = self.db
        self.worker = AsyncWorker(lambda prog, evt: run_history(days, db, prog, evt, hurdle=hurdle, trade_size=size))
        self.worker.progress.connect(lambda p, m: (self.progress.setValue(p), self.lbl.setText(m)))
        self.worker.succeeded.connect(self._done)
        self.worker.failed.connect(lambda e: self._end(f"Backfill failed: {e}"))
        self.worker.cancelled.connect(lambda: self._end("Backfill cancelled."))
        self.b_run.setEnabled(False)
        self.b_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.worker.start()

    def _end(self, msg: str):
        self.b_run.setEnabled(True)
        self.b_cancel.setEnabled(False)
        self.lbl.setText(msg)
        self.status.emit(msg)

    def _done(self, res: BacktestResult):
        self.db.replace_trades("backfill", res.trades)
        self.reload()
        st = res.stats
        self._end(f"Backfill done in {st.get('seconds', '?')}s: {len(res.trades)} hurdle-clearing opportunities "
                  f"({st.get('structures', 0)} structures, {st.get('bars_rejected_by_data_gate', 0)} bars rejected "
                  f"by the data-consistency gate).")
