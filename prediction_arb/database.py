"""SQLite cache: settled markets, candles, live opportunity log, positions and settlements."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Iterable, Optional

from config import DB_PATH
from core.models import (ArbOpportunity, Candle, Market, TradeLog, Venue, iso, parse_dt,
                         utcnow)

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    venue TEXT NOT NULL, market_id TEXT NOT NULL, event_id TEXT, status TEXT, result TEXT,
    close_ts INTEGER, settle_ts INTEGER, volume REAL, payload TEXT NOT NULL, updated_at TEXT,
    PRIMARY KEY (venue, market_id));
CREATE INDEX IF NOT EXISTS ix_markets_close ON markets (venue, status, close_ts);
CREATE TABLE IF NOT EXISTS settlements (
    venue TEXT NOT NULL, market_id TEXT NOT NULL, outcome_value REAL, settled_at TEXT, source TEXT,
    PRIMARY KEY (venue, market_id));
CREATE TABLE IF NOT EXISTS candles (
    venue TEXT NOT NULL, market_id TEXT NOT NULL, period INTEGER NOT NULL, ts INTEGER NOT NULL,
    yes_bid REAL, yes_ask REAL, price REAL, volume REAL,
    PRIMARY KEY (venue, market_id, period, ts));
CREATE TABLE IF NOT EXISTS candle_meta (
    venue TEXT NOT NULL, market_id TEXT NOT NULL, period INTEGER NOT NULL,
    start_ts INTEGER, end_ts INTEGER, degraded INTEGER DEFAULT 0, fetched_at TEXT, n INTEGER,
    PRIMARY KEY (venue, market_id, period));
CREATE TABLE IF NOT EXISTS opportunities (
    key TEXT PRIMARY KEY, arb_type TEXT, event_name TEXT, first_seen TEXT, last_seen TEXT,
    n_seen INTEGER DEFAULT 1, best_apy REAL, last_apy REAL, settle_time TEXT, payload TEXT,
    resolved INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS opp_snapshots (
    key TEXT NOT NULL, ts TEXT NOT NULL, apy REAL, cost REAL, payout REAL, size REAL);
CREATE INDEX IF NOT EXISTS ix_snap_key ON opp_snapshots (key, ts);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, opp_key TEXT, source TEXT, opened_at TEXT, status TEXT,
    cost REAL, realized_payout REAL, apy REAL, payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_trades_src ON trades (source, opened_at);
CREATE TABLE IF NOT EXISTS pair_cache (
    a_key TEXT NOT NULL, b_key TEXT NOT NULL, relation TEXT, confidence REAL, created_at TEXT,
    PRIMARY KEY (a_key, b_key));
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, config TEXT, metrics TEXT);
"""


class Database:
    """Thread-safe (single lock) wrapper over one SQLite file in WAL mode."""

    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()

    # ── markets / settlements ─────────────────────────────────────────────────
    def upsert_markets(self, markets: Iterable[Market]) -> int:
        rows = []
        now = iso(utcnow())
        for m in markets:
            rows.append((m.venue.value, m.market_id, m.event_id, m.status, m.result,
                         int(m.close_time.timestamp()) if m.close_time else None,
                         int(m.settle_time.timestamp()) if m.settle_time else None,
                         m.volume, m.model_dump_json(), now))
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO markets (venue, market_id, event_id, status, result, close_ts,"
                " settle_ts, volume, payload, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            srows = [(m.venue.value, m.market_id, m.settle_value, iso(m.settled_ts), "api")
                     for m in markets if m.settle_value is not None]
            self.conn.executemany(
                "INSERT OR REPLACE INTO settlements (venue, market_id, outcome_value, settled_at, source)"
                " VALUES (?,?,?,?,?)", srows)
            self.conn.commit()
        return len(rows)

    def load_markets(self, venue: Optional[Venue] = None, status: Optional[str] = None,
                     close_from: Optional[datetime] = None, close_to: Optional[datetime] = None,
                     min_volume: float = 0.0) -> list[Market]:
        q = "SELECT payload FROM markets WHERE 1=1"
        args: list = []
        if venue:
            q += " AND venue=?"; args.append(venue.value)
        if status:
            q += " AND status=?"; args.append(status)
        if close_from:
            q += " AND close_ts>=?"; args.append(int(close_from.timestamp()))
        if close_to:
            q += " AND close_ts<=?"; args.append(int(close_to.timestamp()))
        if min_volume:
            q += " AND volume>=?"; args.append(min_volume)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        return [Market.model_validate_json(r["payload"]) for r in rows]

    def get_market(self, venue: Venue, market_id: str) -> Optional[Market]:
        with self._lock:
            r = self.conn.execute("SELECT payload FROM markets WHERE venue=? AND market_id=?",
                                  (venue.value, market_id)).fetchone()
        return Market.model_validate_json(r["payload"]) if r else None

    def settlement(self, venue: Venue, market_id: str) -> Optional[float]:
        with self._lock:
            r = self.conn.execute("SELECT outcome_value FROM settlements WHERE venue=? AND market_id=?",
                                  (venue.value, market_id)).fetchone()
        return r["outcome_value"] if r else None

    # ── candles ───────────────────────────────────────────────────────────────
    def save_candles(self, venue: Venue, market_id: str, period: int, candles: list[Candle],
                     start_ts: int, end_ts: int, degraded: bool = False):
        rows = [(venue.value, market_id, period, c.ts, c.yes_bid, c.yes_ask, c.price, c.volume)
                for c in candles]
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO candles (venue, market_id, period, ts, yes_bid, yes_ask, price,"
                " volume) VALUES (?,?,?,?,?,?,?,?)", rows)
            self.conn.execute(
                "INSERT OR REPLACE INTO candle_meta (venue, market_id, period, start_ts, end_ts, degraded,"
                " fetched_at, n) VALUES (?,?,?,?,?,?,?,?)",
                (venue.value, market_id, period, start_ts, end_ts, int(degraded), iso(utcnow()), len(rows)))
            self.conn.commit()

    def candle_meta(self, venue: Venue, market_id: str, period: int) -> Optional[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM candle_meta WHERE venue=? AND market_id=? AND period=?",
                (venue.value, market_id, period)).fetchone()

    def load_candles(self, venue: Venue, market_id: str, period: int) -> list[Candle]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT ts, yes_bid, yes_ask, price, volume FROM candles WHERE venue=? AND market_id=?"
                " AND period=? ORDER BY ts", (venue.value, market_id, period)).fetchall()
        return [Candle(r["ts"], r["yes_bid"], r["yes_ask"], r["price"], r["volume"] or 0.0) for r in rows]

    # ── live opportunity log ──────────────────────────────────────────────────
    def log_opportunities(self, opps: Iterable[ArbOpportunity]):
        now = iso(utcnow())
        with self._lock:
            for o in opps:
                k = o.key
                payload = o.model_dump_json()
                row = self.conn.execute("SELECT best_apy FROM opportunities WHERE key=?", (k,)).fetchone()
                if row is None:
                    self.conn.execute(
                        "INSERT INTO opportunities (key, arb_type, event_name, first_seen, last_seen, n_seen,"
                        " best_apy, last_apy, settle_time, payload) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (k, o.arb_type.value, o.event_name, now, now, 1, o.apy, o.apy,
                         iso(o.settle_time), payload))
                else:
                    best = max(row["best_apy"] or 0.0, o.apy)
                    self.conn.execute(
                        "UPDATE opportunities SET last_seen=?, n_seen=n_seen+1, best_apy=?, last_apy=?,"
                        " payload=? WHERE key=?", (now, best, o.apy, payload, k))
                self.conn.execute(
                    "INSERT INTO opp_snapshots (key, ts, apy, cost, payout, size) VALUES (?,?,?,?,?,?)",
                    (k, now, o.apy, o.total_cost, o.guaranteed_payout, o.size))
            self.conn.commit()

    def load_logged_opportunities(self, since: datetime) -> list[tuple[ArbOpportunity, str, str]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT payload, first_seen, last_seen FROM opportunities WHERE first_seen>=?"
                " ORDER BY first_seen DESC", (iso(since),)).fetchall()
        return [(ArbOpportunity.model_validate_json(r["payload"]), r["first_seen"], r["last_seen"])
                for r in rows]

    # ── positions ─────────────────────────────────────────────────────────────
    def replace_trades(self, source: str, trades: list[TradeLog]):
        rows = [(t.opp_key, source, iso(t.opened_at), t.status, t.cost, t.realized_payout,
                 t.apy_at_entry, t.model_dump_json()) for t in trades]
        with self._lock:
            self.conn.execute("DELETE FROM trades WHERE source=?", (source,))
            self.conn.executemany(
                "INSERT INTO trades (opp_key, source, opened_at, status, cost, realized_payout, apy, payload)"
                " VALUES (?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()

    def add_trade(self, source: str, t: TradeLog):
        with self._lock:
            self.conn.execute(
                "INSERT INTO trades (opp_key, source, opened_at, status, cost, realized_payout, apy, payload)"
                " VALUES (?,?,?,?,?,?,?,?)", (t.opp_key, source, iso(t.opened_at), t.status, t.cost,
                                              t.realized_payout, t.apy_at_entry, t.model_dump_json()))
            self.conn.commit()

    def update_trade(self, trade_id: int, t: TradeLog):
        with self._lock:
            self.conn.execute(
                "UPDATE trades SET status=?, realized_payout=?, payload=? WHERE id=?",
                (t.status, t.realized_payout, t.model_dump_json(), trade_id))
            self.conn.commit()

    def load_trades(self, since: Optional[datetime] = None,
                    sources: Optional[Iterable[str]] = None) -> list[tuple[int, TradeLog]]:
        q = "SELECT id, payload FROM trades WHERE 1=1"
        args: list = []
        if since:
            q += " AND opened_at>=?"; args.append(iso(since))
        if sources:
            sl = list(sources)
            q += f" AND source IN ({','.join('?' * len(sl))})"; args += sl
        q += " ORDER BY opened_at DESC"
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        return [(r["id"], TradeLog.model_validate_json(r["payload"])) for r in rows]

    # ── misc ──────────────────────────────────────────────────────────────────
    def save_pairs(self, pairs: list[tuple[str, str, str, float]]):
        now = iso(utcnow())
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO pair_cache (a_key, b_key, relation, confidence, created_at)"
                " VALUES (?,?,?,?,?)", [(a, b, r, c, now) for a, b, r, c in pairs])
            self.conn.commit()

    def save_backtest_run(self, config: dict, metrics: dict):
        with self._lock:
            self.conn.execute("INSERT INTO backtest_runs (created_at, config, metrics) VALUES (?,?,?)",
                              (iso(utcnow()), json.dumps(config, default=str), json.dumps(metrics, default=str)))
            self.conn.commit()

    def stats(self) -> dict:
        with self._lock:
            out = {}
            for t in ("markets", "candles", "opportunities", "trades", "settlements", "backtest_runs"):
                out[t] = self.conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        return out

    def clear_cache(self, keep_trades: bool = True):
        with self._lock:
            for t in ("markets", "candles", "candle_meta", "settlements", "pair_cache"):
                self.conn.execute(f"DELETE FROM {t}")
            if not keep_trades:
                for t in ("opportunities", "opp_snapshots", "trades", "backtest_runs"):
                    self.conn.execute(f"DELETE FROM {t}")
            self.conn.commit()
            self.conn.execute("VACUUM")
