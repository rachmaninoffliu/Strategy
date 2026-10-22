"""Persistent trade ledger (Qt-free): every position the strategy opens, and how it settled, survives restarts.

One append-only file, ``data/live/ledger.jsonl`` (``<TRADER_STATE_DIR>/live/`` when that variable is set): one JSON event per line.
  {"ev": "open",   "id": "<mode>|<market>|<order id>", "ts": <fill time>, "mode": "live"|"paper", "venue": "kalshi"|"paper"|"polymarket", "session": ..., "market": <Kalshi ticker>, "window": <window open ts>, "side": ..., "entry": ..., "contracts": ..., "fee": ..., ...}
  {"ev": "settle", "id": ..., "ts": <settle time>, "outcome": "WIN"|"LOSS", "pnl": ..., "equity_after": ..., "recovered": bool}
  {"ev": "session", "id": "<session id>", "state": "start"|"alive"|"end", "ts": ..., "mode": ..., "venue": ..., "capital": ..., "cfg": {the settings the session trades with}, "why": ...}
A session writes "start" (with its settings) when it first ticks, "alive" once per 15-minute window and "end" when it stops or halts: that is what lets the live-vs-backtest reconcile know WHEN the app was trading and WITH WHICH settings.
A position is written the moment it is filled (not when it settles), so a position that is still open when the app is closed is not lost: ``reconcile`` settles it from Kalshi's result at the next start.
Older per-session CSVs (``session_*.csv``, written before the ledger existed and still written as a per-session backup) are merged in read-only; the ledger wins on duplicates.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from .data import DATA, SETTLE_DELAY, WINDOW


RETIRED_VENUES = ("polymarket",)        # trades of the retired venue are not part of the history (their files are kept in archive_polymarket/); ``include_retired=True`` still reads them


def ledger_dir() -> Path:
    return Path(os.environ.get("TRADER_STATE_DIR") or DATA) / "live"


def ledger_path() -> Path:
    return ledger_dir() / "ledger.jsonl"


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def trade_id(mode: str, market: str, order_id: str) -> str:
    return f"{mode}|{market}|{order_id}"


@dataclass
class LedgerState:
    settled: list = field(default_factory=list)     # trade rows in LiveEngine.trades format (+ session, recovered)
    open: list = field(default_factory=list)        # 'open' events that have not settled yet
    bad_lines: int = 0                              # unreadable lines skipped (a crash mid-write)
    legacy: int = 0                                 # settled rows that came only from the old per-session CSVs

    def count(self, mode: Optional[str] = None) -> int:
        return sum(1 for r in self.settled if mode is None or r.get("mode") == mode)


class Ledger:
    def __init__(self, path: Optional[Path] = None, legacy_dir: Optional[Path] = None):
        self.path = Path(path) if path else ledger_path()
        self.legacy_dir = Path(legacy_dir) if legacy_dir else self.path.parent

    # ── writing ──
    def _append(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(obj, separators=(",", ":"), default=float) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def record_open(self, *, mode: str, session: str, market: str, venue: str = "", window: int, leg: str, side: str, quoted: float, entry: float, contracts: int, stake: float, fee: float,
                    p_used: float, f_full: float, f_used: float, bankroll_before: float, order_id: str, entry_ts: float, regime: str = "", mult_used: float = float("nan"),
                    cap_used: float = float("nan"), note: str = "", requested: int = 0) -> str:
        tid = trade_id(mode, market, order_id or "none")
        self._append(dict(ev="open", id=tid, ts=float(entry_ts), mode=mode, venue=venue or ("kalshi" if mode == "live" else "paper"), session=session, market=market, window=int(window), leg=leg, side=side, quoted=float(quoted), entry=float(entry),
                          contracts=int(contracts), stake=float(stake), fee=float(fee), p_used=float(p_used), f_full=float(f_full), f_used=float(f_used), bankroll_before=float(bankroll_before),
                          order_id=order_id or "", regime=regime or "", mult_used=_num(mult_used), cap_used=_num(cap_used), note=note or "", requested=int(requested or contracts)))
        return tid

    def record_session(self, session: str, state: str, ts: float, *, mode: str = "", venue: str = "", capital: Optional[float] = None, cfg: Optional[dict] = None, why: str = "") -> None:
        ev = dict(ev="session", id=session, state=state, ts=float(ts), mode=mode, venue=venue)
        if capital is not None:
            ev["capital"] = float(capital)
        if cfg is not None:
            ev["cfg"] = cfg
        if why:
            ev["why"] = why
        self._append(ev)

    def record_settle(self, tid: str, *, settle_ts: float, outcome: str, pnl: float, equity_after: float, recovered: bool = False, account_bp: Optional[float] = None) -> None:
        """``account_bp``: the exchange's own buying power right after a LIVE settlement (audit trail: compare it with the ledger's equity)."""
        self._append(dict(ev="settle", id=tid, ts=float(settle_ts), outcome=outcome, pnl=float(pnl), equity_after=float(equity_after), recovered=bool(recovered), account_bp=_num(account_bp)))

    # ── reading ──
    def read_events(self) -> tuple[list, int]:
        out, bad = [], 0
        if not self.path.exists():
            return out, bad
        with open(self.path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if isinstance(ev, dict) and ev.get("ev") in ("open", "settle", "session") and ev.get("id"):
                        out.append(ev)
                    else:
                        bad += 1
                except ValueError:
                    bad += 1
        return out, bad

    def load(self, mode: Optional[str] = None, legacy: bool = True, include_retired: bool = False) -> LedgerState:
        events, bad = self.read_events()
        opens, settles = {}, {}
        for ev in events:
            if ev["ev"] != "session":
                (opens if ev["ev"] == "open" else settles).setdefault(ev["id"], ev)      # first event of each kind wins (a duplicate settle is harmless)
        st = LedgerState(bad_lines=bad)
        for tid, o in opens.items():
            if (mode and o.get("mode") != mode) or (not include_retired and o.get("venue") in RETIRED_VENUES):
                continue
            s = settles.get(tid)
            if s is None:
                st.open.append(o)
                continue
            st.settled.append(dict(entry_time=_fmt(o["ts"]), settle_time=_fmt(s["ts"]), market=o["market"], leg=o["leg"], side=o["side"], quoted=o["quoted"], entry=o["entry"],
                                   contracts=o["contracts"], stake=o["stake"], fee=o["fee"], outcome=s["outcome"], pnl=s["pnl"], p_used=o["p_used"], f_full=o["f_full"], f_used=o["f_used"],
                                   equity_after=s["equity_after"], regime=o.get("regime", ""), mult_used=_unnum(o.get("mult_used")), cap_used=_unnum(o.get("cap_used")),
                                   mode=o["mode"], order_id=o.get("order_id", ""), session=o.get("session", ""), recovered=bool(s.get("recovered", False)), requested=int(o.get("requested") or o["contracts"]),
                                   venue=o.get("venue") or ("kalshi" if o["mode"] == "live" else "paper")))
        if legacy:
            have = {(r["mode"], r["market"]) for r in st.settled} | {(o["mode"], o["market"]) for o in st.open}
            for r in self._legacy_rows():
                if (mode and r["mode"] != mode) or (r["mode"], r["market"]) in have or (not include_retired and r["venue"] in RETIRED_VENUES):
                    continue
                have.add((r["mode"], r["market"]))
                st.settled.append(r)
                st.legacy += 1
        st.settled.sort(key=lambda r: (r["settle_time"], r["entry_time"]))
        st.open.sort(key=lambda o: o["ts"])
        return st

    def sessions(self, mode: Optional[str] = None) -> list:
        """The sessions the ledger knows, oldest first: dict(session, mode, venue, start_ts, last_ts, end_ts or None, why, capital, cfg).  ``last_ts`` is the last sign of life (heartbeat, end or settlement),
        so a session that crashed still has a span.  Sessions that predate these records are not listed (their trades are, via ``load``)."""
        by: dict = {}
        for ev in self.read_events()[0]:
            if ev["ev"] != "session":
                continue
            s = by.setdefault(ev["id"], dict(session=ev["id"], mode=ev.get("mode", ""), venue=ev.get("venue", ""), start_ts=None, last_ts=None, end_ts=None, why="", capital=None, cfg=None))
            ts = float(ev["ts"])
            if ev.get("state") == "start" and s["start_ts"] is None:
                s.update(start_ts=ts, capital=ev.get("capital"), cfg=ev.get("cfg"), mode=ev.get("mode", s["mode"]), venue=ev.get("venue", s["venue"]))
            if ev.get("state") == "end":
                s["end_ts"], s["why"] = ts, ev.get("why", "")
            s["last_ts"] = max(ts, s["last_ts"] or 0.0)
        out = [s for s in by.values() if s["start_ts"] is not None and (not mode or s["mode"] == mode)]
        out.sort(key=lambda s: s["start_ts"])
        return out

    def _legacy_rows(self) -> Iterable[dict]:
        d = self.legacy_dir
        if not d.exists():
            return
        for f in sorted(d.glob("session_*.csv")):
            mode = "live" if f.stem.endswith("_live") else "paper"
            try:
                with open(f, newline="", encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        try:
                            yield dict(entry_time=r["entry_time"], settle_time=r["settle_time"], market=r["market"], leg=r["leg"], side=r["side"], quoted=float(r["quoted"]),
                                       entry=float(r["entry"]), contracts=int(float(r["contracts"])), stake=float(r["stake"]), fee=float(r["fee"]), outcome=r["outcome"], pnl=float(r["pnl"]),
                                       p_used=float(r["p_used"]), f_full=float(r["f_full"]), f_used=float(r["f_used"]), equity_after=float(r["equity_after"]), regime=r.get("regime", "") or "",
                                       mult_used=_unnum(r.get("mult_used")), cap_used=_unnum(r.get("cap_used")), mode=(r.get("mode") or mode), order_id=r.get("order_id", ""),
                                       session=f.stem.replace("session_", ""), recovered=False, venue=(r.get("venue") or ("polymarket" if mode == "live" else "paper")))
                        except (KeyError, ValueError, TypeError):
                            continue
            except OSError:
                continue

    def kalshi_order_ids(self) -> set:
        """Kalshi order ids of every LIVE position the ledger knows (open or settled)."""
        return {str(o.get("order_id", "")) for o in self.read_events()[0] if o["ev"] == "open" and o.get("mode") == "live" and (o.get("venue") or "kalshi") == "kalshi"}

    def unmatched_exchange_fills(self, fills: list) -> list:
        """Account fills in this strategy's markets (Kalshi ``GET /portfolio/fills``) that no LIVE ledger entry explains: an order a crashed session never recorded, or one placed outside this app.
        One row per order: {order_id, ticker, qty, time}."""
        known = self.kalshi_order_ids()
        out: dict = {}
        for f in fills:
            tk = str(f.get("ticker") or f.get("market_ticker") or "")
            oid = str(f.get("order_id", ""))
            if not tk.startswith("KXBTC15M-") or oid in known:
                continue
            o = out.setdefault(oid, dict(order_id=oid, ticker=tk, qty=0.0, time=str(f.get("created_time", ""))))
            try:
                o["qty"] += float(f.get("count_fp", f.get("count", 0)) or 0)
            except (TypeError, ValueError):
                pass
        return list(out.values())

    # ── recovery ──
    def reconcile(self, result_fn: Callable[[str], Optional[str]], now: Optional[float] = None, skip_sessions: Iterable[str] = ()) -> list:
        """Settle positions that were open when an earlier session ended: ask Kalshi for each closed window's result (``result_fn(ticker) -> 'yes' | 'no' | None``) and append the settle event.
        Positions of a session that is still running (``skip_sessions``) are left to its engine.  Returns the rows that were recovered."""
        now = time.time() if now is None else now
        skip = set(skip_sessions)
        got = []
        for o in self.load(legacy=False).open:
            if o.get("session") in skip or now < int(o["window"]) + WINDOW + SETTLE_DELAY:
                continue
            try:
                res = result_fn(o["market"])
            except Exception:                                    # noqa: BLE001  (offline: try again next time)
                res = None
            if res not in ("yes", "no"):
                continue
            win = (res == "yes") == (o["side"] == "YES")
            pnl = o["contracts"] * (1.0 if win else 0.0) - o["contracts"] * o["entry"] - o["fee"]
            self.record_settle(o["id"], settle_ts=int(o["window"]) + WINDOW, outcome="WIN" if win else "LOSS", pnl=pnl, equity_after=o["bankroll_before"] + pnl, recovered=True)
            got.append(dict(id=o["id"], market=o["market"], outcome="WIN" if win else "LOSS", pnl=pnl))
        return got


def _num(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x else None                              # JSON has no NaN


def _unnum(x) -> float:
    v = _num(x)                                               # '' / None / junk (old CSVs have no such columns) -> NaN
    return float("nan") if v is None else v


def pooled_frame(rows: list):
    """Trades of several sessions as ONE equity story: (DataFrame in backtest TRADE_COLS format (+ session / mode / recovered), starting capital).
    Equity is the first recorded trade's starting bankroll plus the cumulative P&L of the recorded trades (no deposits / withdrawals are known to the ledger)."""
    from .live import trades_frame
    t = trades_frame(rows)
    if t.empty:
        return t, 0.0
    t = t.sort_values(["settle_ts", "entry_ts"]).reset_index(drop=True)
    cap0 = float(t.bankroll_before.iloc[0])
    t["equity_after"] = cap0 + t.pnl.cumsum()
    t["bankroll_before"] = t.equity_after - t.pnl
    return t, cap0
