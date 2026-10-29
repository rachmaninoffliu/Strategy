"""
Live / paper engine for the mean-reversion models (no Qt): the backtest's decide() on the broker's positions, orders in the same auctions as the backtest.

Daily cycle (ET) - run() calls tick() every second:
  morning (from 08:45)          top up the official daily bars of the model's symbols; an 'overnight' model's positions whose holding ends today get a market-on-open (OPG)
                                sell (after the 09:28 OPG cut-off: a market sell at once, journaled as late)
  decision (close - lead + feed delay; 15:46 for the 15:30 snapshot on the free delayed SIP, 15:45 for 15:45 on a real-time feed)
                                today's 15-minute bars -> Market (daily history + today) -> Signals -> decide(model, today, member, broker positions, strategy capital)
                                -> exits and entries as market-on-close (CLS) orders, all before the 15:50 cut-off (12:50 on early-close days)
  after the close (+5 min)      the auction fills are read and journaled; the book (entry day, price, trade id per symbol) is rebuilt from the broker's positions

Backtest parity: the same Market class, the same Signals, the same decide(), the same sizing (whole shares at the decision-time price, 1 / slots of (1 - cash reserve) x capital), fills in
the same auctions. tests/test_mr.py replays a backtest session by session through this engine with a simulated broker and requires every order to equal the backtest's trade.
Live-only overlays (not in the backtest): the daily-loss halt and the drawdown stop - both off unless set; when one acts, the plan says so.

Safety (as the residual apps): PAPER + DRY RUN at start; ARMED needs the typed phrase "ARM PAPER", the project's own keys and an account no other strategy claimed (one engine per account,
operating-system lock); real money is disabled unless RESARB_ALLOW_LIVE=1; positions in symbols the model can never trade are reported and left alone; KILL cancels and flattens.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import data as D
from ..live import ARM_PHRASE, LIVE_PHRASE, Check, FileLock, Throttle, now_et  # noqa: F401  (ARM_PHRASE / LIVE_PHRASE re-exported for the GUI)
from . import engine as E
from .market import COLS, Market, _fetch_daily, decision_lead, load_daily, session_closes, update_daily
from .models import MRModel, Pos, Signals, decide, universe_parts

KEY, PREFIX = "MeanReversion", "MR"
NY = "America/New_York"
OPG_CUTOFF = pd.Timedelta(hours=9, minutes=28)                                                  # Alpaca refuses market-on-open orders from 09:28 to 19:00
FEEDS = ("sip_delayed", "sip+iex", "iex", "sip")


def default_feed(minute: int, current: str = "") -> str:
    """15:30 or earlier: the delayed SIP is exactly the backtest's data. Later: the free hybrid, unless the real-time SIP (paid) or IEX was chosen."""
    if int(minute) <= 1530:
        return "sip_delayed" if current in ("", "sip+iex") else current
    return "sip+iex" if current in ("", "sip_delayed") else current


@dataclass
class MRLiveSettings:
    paper: bool = True
    armed: bool = False
    capital: float = 0.0                 # > 0: the strategy trades this fixed amount; otherwise account equity x equity_fraction
    equity_fraction: float = 1.0
    feed: str = "sip_delayed"            # today's bars: sip_delayed (free plan, 16 min late: decisions at 15:30 or earlier, the backtest's data) | sip+iex (free: delayed SIP, the last
                                         # ~30 minutes from real-time IEX) | iex (real time, IEX prints only) | sip (real-time SIP plan, the backtest's data)
    decision_delay_s: float = 6.0        # wait after the decision bar's data is due
    moc_cutoff_min: float = 10.5         # last moment to send market-on-close orders, in minutes before the close (Alpaca: 15:50)
    daily_loss_halt: float = 0.0         # > 0: no new entries after this loss today (live-only overlay)
    dd_stop: float = 0.0                 # > 0: sell everything and stay flat after this drawdown from the high-water mark, until reset (live-only overlay)
    shared_keys_ok: bool = False         # allow an ARMED engine on keys that are not the project's own
    fill_wait_s: float = 30.0
    max_calls_per_min: int = 150
    history_days: int = 460              # calendar days of daily history behind the decision (200-session means need ~290)


@dataclass
class MRPlan:
    day: pd.Timestamp
    decided_at: pd.Timestamp             # clock time of the decision
    snapshot_to: pd.Timestamp            # decision time the snapshot stands for (bars that ended by then)
    capital: float
    equity: float
    positions: pd.DataFrame              # symbol, shares, entry_date, entry_px, last, value, pnl, age, exit
    candidates: pd.DataFrame             # symbol, signal, rank, taken, last
    orders: pd.DataFrame                 # symbol, side, qty, tif, ref_px, notional, why, ok, note, cid, signal, rank, kind
    notes: list = field(default_factory=list)
    halted: bool = False
    stopped: bool = False
    real: bool = True                    # False: a manual dry run away from the decision time


# ── broker adapter (the tests use a simulated one with the same methods) ─────────────────────────────────────────────────────────────────────────────
class AlpacaBroker:
    def __init__(self, paper: bool, calls_per_min: int = 150):
        self.paper = paper
        self.tc = D.trading_client(PREFIX, paper)
        self.dc = D.data_client(PREFIX)
        self._thr = Throttle(calls_per_min)

    def _call(self, fn, *a, **k):
        self._thr.wait()
        return fn(*a, **k)

    def account(self) -> dict:
        a = self._call(self.tc.get_account)
        return dict(number=str(a.account_number), status=str(getattr(a.status, "value", a.status)), equity=float(a.equity), last_equity=float(a.last_equity),
                    cash=float(a.cash), buying_power=float(a.buying_power), multiplier=float(a.multiplier or 1), shorting=bool(a.shorting_enabled),
                    blocked=bool(a.trading_blocked or a.account_blocked))

    def clock(self) -> dict:
        c = self._call(self.tc.get_clock)
        cv = lambda x: pd.Timestamp(x).tz_convert(NY).tz_localize(None)
        return dict(is_open=bool(c.is_open), next_open=cv(c.next_open), next_close=cv(c.next_close), ts=cv(c.timestamp))

    def positions(self) -> dict:
        return {p.symbol: dict(qty=int(float(p.qty)), avg_px=float(p.avg_entry_price or 0) or np.nan, mv=float(p.market_value or 0)) for p in self._call(self.tc.get_all_positions)}

    def submit(self, symbol: str, qty: int, side: str, tif: str, cid: str) -> dict:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        t = {"cls": TimeInForce.CLS, "opg": TimeInForce.OPG, "day": TimeInForce.DAY}[tif]
        o = self._call(self.tc.submit_order, MarketOrderRequest(symbol=symbol, qty=int(qty), side=OrderSide.BUY if side == "buy" else OrderSide.SELL, time_in_force=t,
                                                                client_order_id=cid))
        return dict(id=str(o.id), status=_status(o))

    @staticmethod
    def _row(o) -> dict:
        ts = lambda x: pd.Timestamp(x).tz_convert(NY).tz_localize(None) if x is not None else None
        return dict(id=str(o.id), cid=str(o.client_order_id or ""), symbol=o.symbol, side=str(getattr(o.side, "value", o.side)), qty=float(o.qty or 0), status=_status(o),
                    filled_qty=float(o.filled_qty or 0), filled_avg_price=float(o.filled_avg_price or 0) or np.nan, submitted_at=ts(o.submitted_at), filled_at=ts(o.filled_at),
                    tif=str(getattr(o.time_in_force, "value", o.time_in_force)))

    def orders(self, after: pd.Timestamp, status: str = "all") -> list:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        st = {"all": QueryOrderStatus.ALL, "open": QueryOrderStatus.OPEN, "closed": QueryOrderStatus.CLOSED}[status]
        aft = pd.Timestamp(after).tz_localize(NY).tz_convert("UTC").to_pydatetime()
        oo = self._call(self.tc.get_orders, GetOrdersRequest(status=st, after=aft, limit=500))
        return [self._row(o) for o in oo if str(o.client_order_id or "").startswith(PREFIX + "-")]

    def order_by_cid(self, cid: str) -> Optional[dict]:
        try:
            return self._row(self._call(self.tc.get_order_by_client_id, cid))
        except Exception:                                                                     # noqa: BLE001
            return None

    def cancel(self, order_id: str) -> None:
        self._call(self.tc.cancel_order_by_id, order_id)

    def bars(self, symbols: list, start: pd.Timestamp, end: pd.Timestamp, feed: str) -> dict:
        out = {}
        for i in range(0, len(symbols), 100):
            out.update(D.fetch_bars(self.dc, symbols[i:i + 100], start, end, feed="iex" if feed == "iex" else "sip"))
        return out

    def daily(self, symbols: list, start: pd.Timestamp) -> pd.DataFrame:
        frames = [_fetch_daily(self.dc, symbols[i:i + 200], start) for i in range(0, len(symbols), 200)]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)

    def assets(self) -> pd.DataFrame:
        return D.list_assets(PREFIX)


def _status(o) -> str:
    return str(getattr(getattr(o, "status", None), "value", getattr(o, "status", "?")))


FINAL = ("filled", "canceled", "rejected", "expired", "done_for_day")


# ── the engine ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class MRLiveEngine:
    KEY, PREFIX = KEY, PREFIX

    def __init__(self, model: MRModel, live: MRLiveSettings, folder: str, log: Callable[[str, str], None] = lambda m, k="info": print(m), broker=None,
                 clock_fn: Callable[[], pd.Timestamp] = now_et):
        self.m, self.live, self.folder, self.log = model, live, folder, log
        self.broker = broker
        self._injected = broker is not None
        self.now = clock_fn
        self.acct: dict = {}
        self._lock = threading.RLock()
        self._clk = (0.0, None)
        self.flock = FileLock(self.path("engine", "lock"))
        self.last_plan: Optional[MRPlan] = None

    # ── files ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return "paper" if self.live.paper else "live"

    def path(self, name: str, ext: str = "json") -> str:
        d = os.path.join(self.folder, "reports")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"mr_{name}_{self.mode}.{ext}")

    def load_state(self) -> dict:
        p = self.path("state")
        base = {"book": {}, "hwm": None, "day": None, "day_start_equity": None, "stopped": None, "pending": None, "attempt": 0, "prepared": None, "decided": None,
                "reconciled": None, "next_trade_id": 1, "model": None}
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return {**base, **json.load(f)}
            except (OSError, ValueError) as e:
                raise RuntimeError(f"state file {p} is unreadable ({e}) - fix or move it") from e
        return base

    def save_state(self, st: dict) -> None:
        p = self.path("state")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=1, default=str)
            f.flush()
            os.fsync(f.fileno())
        for _ in range(5):
            try:
                os.replace(tmp, p)
                return
            except PermissionError:
                time.sleep(0.2)
        os.replace(tmp, p)

    def journal(self, kind: str, **row) -> None:
        row = {"ts": str(self.now()), "kind": kind, "strategy": KEY, "model": self.m.name, "armed": bool(self.live.armed), **row}
        with open(self.path("journal", "jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def read_journal(self, kinds: Optional[tuple] = None) -> list:
        p = self.path("journal", "jsonl")
        if not os.path.exists(p):
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if kinds is None or r.get("kind") in kinds:
                    out.append(r)
        return out

    # ── broker ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def key_source(self) -> str:
        if self._injected:
            return getattr(self.broker, "keys", "own")
        return "own" if D.own_keys(PREFIX, self.live.paper) else "shared"

    def connect(self) -> dict:
        if not self.live.paper and os.environ.get("RESARB_ALLOW_LIVE") != "1":
            raise RuntimeError("real-money trading is disabled (Valid has not signed this model off). Set RESARB_ALLOW_LIVE=1 only after that.")
        if self.broker is None:
            self.broker = AlpacaBroker(self.live.paper, self.live.max_calls_per_min)
        t0 = time.time()
        a = self.broker.account()
        c = self.clock(0)
        self.acct = dict(paper=self.live.paper, **a, market_open=c["is_open"], next_open=c["next_open"], next_close=c["next_close"], latency_ms=int((time.time() - t0) * 1000),
                         keys=self.key_source(), key_src=("simulated broker" if self._injected else D.key_source(PREFIX, self.live.paper)))
        return self.acct

    def clock(self, max_age: float = 30.0) -> dict:
        t, c = self._clk
        if c is None or time.monotonic() - t > max_age or self._injected:
            c = self.broker.clock()
            self._clk = (time.monotonic(), c)
        return c

    # ── account ownership (shared registry with the residual apps: one strategy per account) ─────────────────────────────────────────────────────────
    def _registry_path(self) -> str:
        return os.path.join(D.DATA_DIR, "resarb_accounts.json")

    def _registry(self) -> dict:
        p = self._registry_path()
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def account_owner(self) -> Optional[str]:
        return self._registry().get(f"{self.mode}:{self.acct.get('number')}", {}).get("strategy")

    def claim_account(self) -> None:
        reg = self._registry()
        key = f"{self.mode}:{self.acct.get('number')}"
        owner = reg.get(key, {}).get("strategy")
        if owner and owner != KEY:
            raise RuntimeError(f"account {self.acct.get('number')} is claimed by {owner}: two strategies on one account would net and close each other's positions - stop {owner} "
                               f"and release it, or give {KEY} a separate paper account ({PREFIX}_ALPACA_API_KEY / _SECRET_KEY)")
        reg[key] = {"strategy": KEY, "since": reg.get(key, {}).get("since", str(self.now()))}
        with open(self._registry_path(), "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=1)

    def arm_problems(self) -> list:
        """Why this engine may NOT be armed now (empty = OK)."""
        out = []
        if not self.live.paper and os.environ.get("RESARB_ALLOW_LIVE") != "1":
            out.append("real money is disabled")
        if self.key_source() != "own" and not self.live.shared_keys_ok:
            out.append("no own keys: put 'Alpaca Key.txt' (Key ID / Secret Key) in the project folder - the ~/.claude.json keys of the Claude tools are refused for orders")
        out += self.model_problems()
        if self.acct:
            owner = self.account_owner()
            if owner and owner != KEY:
                out.append(f"account claimed by {owner}")
            if self.acct.get("blocked"):
                out.append("trading is blocked on the account")
            if self.m.direction == "short" and not self.acct.get("shorting"):
                out.append("shorting is disabled on the account")
        return out

    def model_problems(self) -> list:
        out = []
        if self.live.feed not in FEEDS:
            out.append(f"unknown feed {self.live.feed!r}")
        lead = decision_lead(self.m.minute)
        if self.data_delay() >= lead - pd.Timedelta(minutes=self.live.moc_cutoff_min):
            out.append(f"a {self.m.minute // 100}:{self.m.minute % 100:02d} decision cannot be made on the {self.live.feed} feed before the market-on-close cut-off - "
                       f"use 15:30 (or earlier) with the delayed SIP, or a real-time feed")
        if self.m.max_gross > 1.0 + 1e-9:
            out.append(f"max gross {self.m.max_gross:.2f} > 1: the model would use leverage")
        if self.m.direction != "long":
            out.append("short models are research-only: a short that moves against you grows the book above 1x equity (leverage), and the short side showed no edge")
        return out

    def data_delay(self) -> pd.Timedelta:
        """Time from the decision time (bars that ended by then) until those bars can be read."""
        d = pd.Timedelta(seconds=self.live.decision_delay_s)
        return d + (pd.Timedelta(minutes=D.SIP_DELAY_MIN) if self.live.feed == "sip_delayed" else pd.Timedelta(0))

    # ── calendar ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def session_close(self, day: pd.Timestamp) -> Optional[pd.Timestamp]:
        c = session_closes(day, day)
        return c.iloc[0] if len(c) else None

    def times(self, day: pd.Timestamp) -> Optional[dict]:
        """Today's schedule: snapshot time (the bars that ended by then), decision time (data readable), market-on-close cut-off, close. None = no session."""
        close = self.session_close(day)
        if close is None:
            return None
        snap = close - decision_lead(self.m.minute)
        return dict(snapshot=snap, decision=snap + self.data_delay(), cutoff=close - pd.Timedelta(minutes=self.live.moc_cutoff_min), close=close,
                    open=day + pd.Timedelta(hours=9, minutes=30))

    def last_session(self, now: pd.Timestamp) -> pd.Timestamp:
        c = session_closes(now.normalize() - pd.Timedelta(days=10), now.normalize())
        days = [d for d in c.index if d < now.normalize() or (d == now.normalize() and now >= d + pd.Timedelta(hours=9, minutes=45))]
        return days[-1]

    # ── universe / data ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def universe_symbols(self, day: pd.Timestamp, m: Optional[MRModel] = None) -> list:
        """The symbols the model may enter on ``day`` (plus SPY as the calendar anchor for a top-N universe)."""
        m = m or self.m
        out: list = []
        for part in universe_parts(m.universe):
            if not part.startswith("top"):
                out += E._part_symbols(part)
                continue
            n = int(part[3:] or 100)
            u = D.load_universe()
            if not len(u):
                raise RuntimeError("no stock universe: run 'Update data' first")
            mon = day.to_period("M").to_timestamp()
            mm = u[(u.month == mon) & (u["rank"] <= n)]
            if not len(mm):
                mm = u[(u.month == u[u.month <= mon].month.max()) & (u["rank"] <= n)]
            out += ["SPY"] + sorted(set(mm.symbol) - {"SPY"})
        return list(dict.fromkeys(out))

    def possible_symbols(self) -> set:
        """Everything this model could ever hold (KILL and the foreign-position check use it)."""
        try:
            return set(E.model_symbols(self.m))
        except Exception:                                                                     # noqa: BLE001
            return set(self.universe_symbols(self.now().normalize()))

    def history(self, syms: list, day: pd.Timestamp, notes: list) -> pd.DataFrame:
        """Official daily bars before ``day``: the cache, checked against the broker's latest adjusted bars - a split or dividend since the cache was written (e.g. an ex-date today)
        rescales the cached history, and sessions the cache misses are filled from the broker."""
        start = day - pd.Timedelta(days=self.live.history_days)
        h = load_daily(syms)
        h = h[(h.date >= start) & (h.date < day)]
        try:
            fresh = self.broker.daily(syms, day - pd.Timedelta(days=12))
        except Exception as e:                                                                # noqa: BLE001
            notes.append(f"recent daily bars unavailable ({type(e).__name__}) - the cache is used as it is")
            return h
        fresh = fresh[fresh.date < day] if len(fresh) else fresh
        if not len(fresh):
            return h
        frames, rescaled = [], []
        hs = dict(tuple(h.groupby("symbol"))) if len(h) else {}
        for s, f in fresh.groupby("symbol"):
            old = hs.pop(s, None)
            if old is None or not len(old):
                frames.append(f)
                continue
            common = old.set_index("date").close.reindex(f.date).dropna()
            if len(common):
                k = float(f.set_index("date").close.reindex(common.index).iloc[-1] / common.iloc[-1])
                if np.isfinite(k) and abs(k - 1) > 5e-4:
                    old = old.copy()
                    old[["open", "high", "low", "close"]] *= k
                    rescaled.append(f"{s} x{k:.4f}")
            frames.append(pd.concat([old[~old.date.isin(set(f.date))], f]))
        frames += list(hs.values())
        if rescaled:
            notes.append("adjusted history rewritten since the cache (split / dividend): " + ", ".join(rescaled[:10]) + (" ..." if len(rescaled) > 10 else ""))
        return pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

    def today_bars(self, syms: list, day: pd.Timestamp, now: pd.Timestamp) -> dict:
        start = day + pd.Timedelta(hours=9, minutes=30)
        if self.live.feed == "sip+iex":                                                       # complete delayed-SIP bars, then real-time IEX bars for the last ~30 minutes
            sip_end = now - pd.Timedelta(minutes=D.SIP_DELAY_MIN)
            sip = self.broker.bars(sorted(syms), start, sip_end, "sip") if sip_end > start else {}
            done_to = (sip_end - pd.Timedelta(minutes=D.BAR_MIN)).floor(f"{D.BAR_MIN}min")    # the last SIP bar start that is complete
            iex = self.broker.bars(sorted(syms), max(start, done_to + pd.Timedelta(minutes=D.BAR_MIN)), now, "iex")
            out = {}
            for s in syms:
                a, b = sip.get(s), iex.get(s)
                a = a[a.index <= done_to] if a is not None and len(a) else None
                b = b[b.index > done_to] if b is not None and len(b) else None
                parts = [x for x in (a, b) if x is not None and len(x)]
                if parts:
                    out[s] = pd.concat(parts) if len(parts) > 1 else parts[0]
            return out
        end = now - (pd.Timedelta(minutes=D.SIP_DELAY_MIN) if self.live.feed == "sip_delayed" else pd.Timedelta(0))
        if end <= start:
            return {}
        return self.broker.bars(sorted(syms), start, end, "iex" if self.live.feed == "iex" else "sip")

    def market(self, day: pd.Timestamp, now: pd.Timestamp, held: list, notes: list) -> tuple:
        syms = list(dict.fromkeys(self.universe_symbols(day) + [s for s in held if s in self.possible_symbols()]))
        hist = self.history(syms, day, notes)
        bars = self.today_bars(syms, day, now)
        if not bars:
            raise RuntimeError(f"no 15-minute bars for {day:%Y-%m-%d} yet ({self.live.feed} feed)")
        mk = Market(syms, minute=self.m.minute, start=day - pd.Timedelta(days=self.live.history_days), daily=hist, extra_bars=bars, hist_bars=False)
        t = len(mk.dates) - 1
        if t < 0 or mk.dates[t] != day:
            raise RuntimeError(f"today's session {day:%Y-%m-%d} is not in the data")
        return mk, t, bars

    # ── the book (entry day / price per held symbol) ─────────────────────────────────────────────────────────────────────────────────────────────────
    def held_positions(self, notes: list) -> tuple:
        """Broker positions this model may hold (dict symbol -> {qty, avg_px}) and the foreign ones (reported, never touched)."""
        pos = self.broker.positions()
        possible = self.possible_symbols()
        ours = {s: p for s, p in pos.items() if s in possible and p["qty"] != 0}
        foreign = {s: p for s, p in pos.items() if s not in possible and p["qty"] != 0}
        if foreign:
            notes.append(f"positions outside this model's symbols are left alone: { {s: p['qty'] for s, p in foreign.items()} }")
        return ours, foreign

    def to_pos(self, mk, ours: dict, book: dict, notes: list) -> dict:
        out = {}
        for s, p in ours.items():
            b = book.get(s) or {}
            if not b:
                notes.append(f"{s}: held without an entry record - treated as entered today (time stop counts from today)")
            ed = pd.Timestamp(b.get("entry_date") or mk.dates[-1])
            et = int(np.searchsorted(mk.dates, ed))
            et = min(et, len(mk.dates) - 1)
            px = float(b.get("entry_px") or p.get("avg_px") or np.nan)
            out[s] = Pos(symbol=s, shares=int(p["qty"]), entry_t=et, entry_px=px, exit_open_t=et + 1 if self.m.hold == "overnight" else None, trade_id=int(b.get("trade_id") or 0))
        return out

    # ── risk overlays ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _risk_state(self, st: dict, equity: float, notes: list) -> tuple:
        today = str(self.now().date())
        if st.get("day") != today:
            st["day"], st["day_start_equity"] = today, float(self.acct.get("last_equity") or equity)
        st["hwm"] = max(float(st.get("hwm") or 0.0), equity)
        halted = stopped = False
        d0 = float(st.get("day_start_equity") or equity)
        if self.live.daily_loss_halt > 0 and d0 > 0 and equity / d0 - 1 <= -self.live.daily_loss_halt:
            halted = True
            notes.append(f"DAILY-LOSS HALT (live overlay): equity {equity / d0 - 1:.2%} today (limit -{self.live.daily_loss_halt:.1%}) - no new entries today")
        if st.get("stopped"):
            stopped = True
            notes.append(f"DRAWDOWN STOP (live overlay) active since {st['stopped']} - everything is sold; reset the drawdown base to resume")
        elif self.live.dd_stop > 0 and st["hwm"] > 0 and equity / st["hwm"] - 1 <= -self.live.dd_stop:
            stopped = True
            st["stopped"] = str(self.now())
            notes.append(f"DRAWDOWN STOP (live overlay): equity {equity / st['hwm'] - 1:.1%} from the high-water mark (limit -{self.live.dd_stop:.0%}) - selling everything")
        return st, halted, stopped

    # ── planning ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def plan(self, now: Optional[pd.Timestamp] = None, force: bool = False) -> MRPlan:
        """The closing-auction decision of today's session at its decision time. ``force`` = a manual dry run at any time: it decides on the newest session's bars available now
        (before the decision time that is a partial-day snapshot - the plan says so)."""
        with self._lock:
            now = now or self.now()
            if self.broker is None:
                self.connect()
            notes: list = []
            day = now.normalize()
            tm = self.times(day)
            real = True
            if tm is None or now < tm["decision"] or now > tm["cutoff"]:
                if not force:
                    raise RuntimeError(f"not the decision time ({tm['decision']:%H:%M:%S} - {tm['cutoff']:%H:%M:%S} ET)" if tm else "no session today")
                real = False
                if tm is None or now < tm["open"] + pd.Timedelta(minutes=15):
                    day = self.last_session(now)
                    tm = self.times(day)
                now_eff = min(now, tm["decision"]) if day == now.normalize() else tm["decision"]
                notes.append(f"manual dry run: decided on the {day:%Y-%m-%d} bars available at {now_eff:%H:%M} ET"
                             + (" - a partial-day snapshot" if now_eff < tm["decision"] else "") + "; orders are only ever sent at the decision time")
            else:
                now_eff = now
            st = self.load_state()
            ours, foreign = self.held_positions(notes)
            mk, t, bars = self.market(day, now_eff if day == now.normalize() else tm["close"] + pd.Timedelta(hours=1), list(ours), notes)
            sig = Signals(mk)
            member = E.universe_mask(mk, self.m)[t]
            exp = tm["snapshot"] - pd.Timedelta(minutes=D.BAR_MIN)
            mem_syms = [s for s, j in mk.col.items() if member[j]]
            have = [s for s in mem_syms if s in bars and len(bars[s]) and (bars[s].index == exp).any()]
            cover = len(have) / max(len(mem_syms), 1)
            if real and cover < 0.8:
                raise RuntimeError(f"only {cover:.0%} of the universe has the {exp:%H:%M} bar - stale feed, no decision")
            if cover < 0.95:
                notes.append(f"{cover:.0%} of the universe has the {exp:%H:%M} bar; the others use their last earlier bar (as the backtest does when a bar is missing)")
            pos = self.to_pos(mk, ours, st.get("book") or {}, notes)
            acct = self.broker.account()
            self.acct.update(acct)
            equity = float(acct["equity"])
            last = sig.last[t]

            def mark(s, p):                                                                   # the backtest's decision-time value: snapshot price, else the previous close, else entry
                j = mk.col[s]
                for x in (last[j], mk.C[t - 1, j] if t > 0 else np.nan):
                    if np.isfinite(x):
                        return p.shares * x
                return p.shares * p.entry_px
            eq_snap = float(acct["cash"]) + sum(mark(s, p) for s, p in pos.items()) + sum(float(p.get("mv") or 0.0) for p in foreign.values())
            cap = self.live.capital if self.live.capital > 0 else eq_snap * self.live.equity_fraction
            st, halted, stopped = self._risk_state(st, equity, notes)
            if real:
                self.save_state(st)
            dec = decide(self.m, t, sig, member, pos, cap)
            notes += dec.notes
            exits = list(dec.exits)
            entries = list(dec.entries)
            if stopped:
                exits = [(s, "drawdown stop (live overlay)") for s in pos]
                entries = []
            elif halted and entries:
                entries = []
            rows = []
            n = int(st.get("attempt", 0)) + 1
            for s, why in exits:
                p = pos[s]
                rows.append(dict(symbol=s, side="sell" if p.shares > 0 else "buy", qty=abs(p.shares), tif="cls", ref_px=float(last[mk.col[s]]), why=why, ok=True, note="",
                                 cid=f"{PREFIX}-{day:%Y%m%d}-{s.replace('.', '')}-x{n}", signal=np.nan, rank=0, kind="exit"))
            assets = None
            if any(sh < 0 for _, sh, _, _ in entries):
                try:
                    assets = self.broker.assets().set_index("symbol")
                except Exception as e:                                                        # noqa: BLE001
                    notes.append(f"asset list unavailable ({type(e).__name__}) - new shorts are refused")
            for s, sh, sv, rank in entries:
                ok, note = True, ""
                if sh < 0:
                    a = assets.loc[s] if assets is not None and s in assets.index else None
                    if a is None or not bool(a.get("shortable", False)) or not bool(a.get("etb", False)):
                        ok, note = False, "not shortable / not easy-to-borrow"
                rows.append(dict(symbol=s, side="buy" if sh > 0 else "sell", qty=abs(sh), tif="cls", ref_px=float(last[mk.col[s]]), why=f"{self.m.entry_signal} {sv:.3f} - rank {rank}",
                                 ok=ok, note=note, cid=f"{PREFIX}-{day:%Y%m%d}-{s.replace('.', '')}-e{n}", signal=sv, rank=rank, kind="entry"))
            cols = ["symbol", "side", "qty", "tif", "ref_px", "notional", "why", "ok", "note", "cid", "signal", "rank", "kind"]
            orders = pd.DataFrame(rows, columns=[c for c in cols if c != "notional"])
            orders["notional"] = orders.qty * orders.ref_px
            orders = orders[cols]
            book = st.get("book") or {}
            prow = []
            for s, p in pos.items():
                j = mk.col[s]
                lx = float(last[j]) if np.isfinite(last[j]) else np.nan
                prow.append(dict(symbol=s, shares=p.shares, entry_date=(book.get(s) or {}).get("entry_date", ""), entry_px=p.entry_px, last=lx, value=p.shares * lx,
                                 pnl=p.shares * (lx - p.entry_px), age=t - p.entry_t, exit=next((w for x, w in exits if x == s), "")))
            cand = pd.DataFrame([dict(symbol=s, signal=v, rank=r, taken=tk, last=float(last[mk.col[s]])) for s, v, r, tk in dec.candidates],
                                columns=["symbol", "signal", "rank", "taken", "last"])
            plan = MRPlan(day=day, decided_at=now, snapshot_to=tm["snapshot"], capital=cap, equity=equity,
                          positions=pd.DataFrame(prow, columns=["symbol", "shares", "entry_date", "entry_px", "last", "value", "pnl", "age", "exit"]),
                          candidates=cand, orders=orders, notes=notes, halted=halted, stopped=stopped, real=real)
            self.last_plan = plan
            return plan

    # ── execution ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _guard(self) -> None:
        if not self.live.armed:
            raise RuntimeError("DRY RUN: the engine is not armed, nothing is sent")
        probs = self.arm_problems()
        if probs:
            raise RuntimeError("refusing to send: " + "; ".join(probs))
        if not self.flock.acquire():
            raise RuntimeError("another engine of this model holds the lock (GUI or headless runner) - stop it first")
        self.claim_account()

    def cancel_ours(self, day: pd.Timestamp, wait_s: float = 10.0) -> int:
        oo = self.broker.orders(day - pd.Timedelta(days=3), "open")
        for o in oo:
            try:
                self.broker.cancel(o["id"])
            except Exception as e:                                                            # noqa: BLE001
                self.log(f"cancel {o['symbol']} failed: {type(e).__name__}: {str(e)[:80]}", "warn")
        t0 = time.monotonic()
        while oo and time.monotonic() - t0 < wait_s:
            if not self._injected:
                time.sleep(1.0)
            oo = self.broker.orders(day - pd.Timedelta(days=3), "open")
        if oo:
            raise RuntimeError(f"{len(oo)} of this engine's orders are still open after cancelling - not trading on top of them")
        return len(oo)

    def submit(self, plan: MRPlan) -> list:
        """Send the plan's market-on-close orders (ARMED, at the decision time only)."""
        self._guard()
        if not plan.real:
            raise RuntimeError("a manual dry run is never sent")
        with self._lock:
            tm = self.times(plan.day)
            if self.now() > tm["cutoff"]:
                raise RuntimeError(f"past the market-on-close cut-off ({tm['cutoff']:%H:%M:%S} ET) - nothing sent")
            self.cancel_ours(plan.day)
            st = self.load_state()
            st["attempt"] = int(st.get("attempt", 0)) + 1
            todo = plan.orders[plan.orders.ok]
            st["pending"] = {"day": str(plan.day.date()), "orders": todo[["symbol", "side", "qty", "cid", "kind", "signal", "rank"]].to_dict("records")}
            self.save_state(st)                                                               # write-ahead: what is about to be sent
            self.journal("decision", **self._decision_row(plan), n_orders=len(todo))
            res = []
            for r in todo.itertuples():
                try:
                    o = self.broker.submit(r.symbol, int(r.qty), r.side, r.tif, r.cid)
                    res.append(dict(symbol=r.symbol, cid=r.cid, id=o["id"], status=o["status"], side=r.side, qty=int(r.qty), ref_px=r.ref_px, leg=r.kind))
                except Exception as e:                                                        # noqa: BLE001
                    code = getattr(e, "status_code", None)
                    if code is not None and 400 <= int(code) < 500 and int(code) not in (408, 429):
                        res.append(dict(symbol=r.symbol, cid=r.cid, status="rejected", side=r.side, qty=int(r.qty), ref_px=r.ref_px, leg=r.kind, note=f"{code}: {str(e)[:120]}"))
                    else:
                        f = self.broker.order_by_cid(r.cid)
                        res.append(dict(symbol=r.symbol, cid=r.cid, id=f["id"] if f else None, status=f["status"] if f else "unknown", side=r.side, qty=int(r.qty), ref_px=r.ref_px,
                                        leg=r.kind, note=f"ambiguous submit ({type(e).__name__}: {str(e)[:80]})"))
                self.journal("order", day=plan.day, **res[-1])
            st = self.load_state()
            st["last_submit"] = str(self.now())
            self.save_state(st)
            return res

    def _decision_row(self, plan: MRPlan) -> dict:
        """Everything a check against the backtest data needs: the positions held (with entry days), the capital, the candidates and the orders."""
        return dict(day=plan.day, snapshot_to=plan.snapshot_to, capital=plan.capital, equity=plan.equity, notes=plan.notes, halted=plan.halted, stopped=plan.stopped,
                    model_spec=self.m.to_dict(), feed=self.live.feed,
                    held={r.symbol: dict(shares=int(r.shares), entry_date=str(r.entry_date), entry_px=float(r.entry_px)) for r in plan.positions.itertuples()},
                    candidates=[dict(symbol=r.symbol, signal=float(r.signal), rank=int(r.rank), taken=bool(r.taken), last=float(r.last)) for r in plan.candidates.itertuples()],
                    orders=[dict(symbol=r.symbol, side=r.side, qty=int(r.qty), kind=r.kind, why=r.why, ok=bool(r.ok), ref_px=float(r.ref_px)) for r in plan.orders.itertuples()])

    def morning(self, now: Optional[pd.Timestamp] = None, refresh: bool = True) -> list:
        """Before the decision: top up the daily bars; an 'overnight' model sells the positions whose holding ends today (OPG before 09:28, a market order once the market is open)."""
        now = now or self.now()
        day = now.normalize()
        out = []
        if refresh and not self._injected:
            try:
                held = set(self.broker.positions()) & self.possible_symbols()
                update_daily(sorted(set(self.universe_symbols(day)) | held), log=lambda m: None, client=self.broker.dc)
            except Exception as e:                                                            # noqa: BLE001
                self.log(f"daily-bar top-up failed ({type(e).__name__}: {str(e)[:100]}) - the decision checks the latest bars itself", "warn")
        if self.m.hold != "overnight":
            return out
        st = self.load_state()
        book = st.get("book") or {}
        ours, _ = self.held_positions([])
        due = [s for s, p in ours.items() if str((book.get(s) or {}).get("entry_date", "")) < str(day.date())]
        if not due:
            return out
        tif = "opg" if now - day < OPG_CUTOFF else "day"
        tm = self.times(day)
        if tif == "day" and not (tm and tm["open"] <= now < tm["decision"]):
            return out                                                                         # before the open past 09:28 / after the decision: the closing auction takes them
        if not self.live.armed:
            self.log(f"DRY RUN: would sell {len(due)} overnight position(s) {'at the open (OPG)' if tif == 'opg' else 'now (late: the open was missed)'}: {', '.join(due)}", "info")
            return out
        self._guard()
        self.cancel_ours(day)
        n = int(st.get("attempt", 0)) + 1
        st["attempt"] = n
        self.save_state(st)
        for s in due:
            q = ours[s]["qty"]
            cid = f"{PREFIX}-{day:%Y%m%d}-{s.replace('.', '')}-o{n}"
            try:
                o = self.broker.submit(s, abs(q), "sell" if q > 0 else "buy", tif, cid)
                out.append(dict(symbol=s, cid=cid, status=o["status"], qty=abs(q), tif=tif))
            except Exception as e:                                                            # noqa: BLE001
                out.append(dict(symbol=s, cid=cid, status=f"FAILED {type(e).__name__}: {str(e)[:80]}", qty=abs(q), tif=tif))
            self.journal("order", day=day, leg="overnight exit", **out[-1])
        self.log(f"overnight exits sent ({tif}): " + ", ".join(f"{r['symbol']} {r['status']}" for r in out), "trade" if tif == "opg" else "warn")
        return out

    def reconcile(self, day: Optional[pd.Timestamp] = None) -> dict:
        """After the close: journal the day's fills, rebuild the book from the broker's positions (entry day / price / trade id), close finished trades in the journal."""
        with self._lock:
            day = (day or self.now()).normalize()
            st = self.load_state()
            book = dict(st.get("book") or {})
            pend = {o["cid"]: o for o in ((st.get("pending") or {}).get("orders") or [])}
            orders = [o for o in self.broker.orders(day - pd.Timedelta(days=1)) if o["cid"].startswith(f"{PREFIX}-{day:%Y%m%d}")]
            for o in orders:
                if o["status"] in FINAL or o["filled_qty"] > 0:
                    self.journal("fill", day=day, symbol=o["symbol"], cid=o["cid"], status=o["status"], side=o["side"], qty=o["qty"], filled_qty=o["filled_qty"],
                                 avg_px=o["filled_avg_price"], tif=o["tif"], submitted_at=o["submitted_at"], filled_at=o["filled_at"])
            ours, _ = self.held_positions([])
            leg = lambda o: o["cid"].split("-")[-1][:1]                                       # e = entry, x = exit at the close, o = overnight exit at the open
            for s in list(book):
                ex = [o for o in orders if o["symbol"] == s and o["filled_qty"] > 0 and leg(o) in ("x", "o")]
                if s not in ours or ex:                                                       # gone, or sold at the open and bought again at the close (a new trade)
                    b = book.pop(s)
                    px = ex[-1]["filled_avg_price"] if ex else np.nan
                    q = float(b.get("shares") or 0)
                    self.journal("trade_closed", symbol=s, trade_id=b.get("trade_id"), entry_date=b.get("entry_date"), entry_px=b.get("entry_px"), exit_date=str(day.date()),
                                 exit_px=px, shares=q, pnl=q * (px - float(b.get("entry_px") or np.nan)) if np.isfinite(px) else None, signal=b.get("signal"), rank=b.get("rank"))
            for s, p in ours.items():
                if s in book:
                    book[s]["shares"] = p["qty"]
                    continue
                buys = [o for o in orders if o["symbol"] == s and o["filled_qty"] > 0 and leg(o) == "e"]
                meta = next((pend[o["cid"]] for o in buys if o["cid"] in pend), {})
                tid = int(st.get("next_trade_id") or 1)
                st["next_trade_id"] = tid + 1
                if buys:
                    book[s] = dict(entry_date=str(day.date()), entry_px=float(buys[-1]["filled_avg_price"]), shares=p["qty"], trade_id=tid, signal=meta.get("signal"), rank=meta.get("rank"))
                else:
                    book[s] = dict(entry_date=str(self._entry_day(s, day).date()), entry_px=float(p.get("avg_px") or np.nan), shares=p["qty"], trade_id=tid)
                self.journal("trade_opened", symbol=s, **book[s])
            st["book"] = book
            st["pending"] = None
            st["reconciled"] = str(day.date())
            self.save_state(st)
            return book

    def _entry_day(self, s: str, day: pd.Timestamp) -> pd.Timestamp:
        """Entry day of a position the book does not know: the newest filled buy of ours in the last 30 days, else today."""
        try:
            oo = [o for o in self.broker.orders(day - pd.Timedelta(days=30)) if o["symbol"] == s and o["filled_qty"] > 0 and o["filled_at"] is not None]
            if oo:
                return pd.Timestamp(max(o["filled_at"] for o in oo)).normalize()
        except Exception:                                                                     # noqa: BLE001
            pass
        return day

    # ── the loop ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def tick(self, now: Optional[pd.Timestamp] = None, on_plan: Callable[[MRPlan], None] = lambda p: None) -> Optional[str]:
        """One pass of the daily cycle at clock time ``now``; returns what it did (or None)."""
        now = now or self.now()
        day = now.normalize()
        tm = self.times(day)
        st = self.load_state()
        dd = st.get("decided")
        if dd and st.get("reconciled") != dd and (dd < str(day.date()) or (tm is not None and now >= tm["close"] + pd.Timedelta(minutes=5))):
            self.reconcile(pd.Timestamp(dd))                                                  # also catches up a day whose close the app did not see
            self.log(f"after the {dd} close: book rebuilt ({len(self.load_state().get('book') or {})} position(s))", "ok")
            return "reconciled"
        if tm is None:
            return None
        if st.get("prepared") != str(day.date()) and now >= day + pd.Timedelta(hours=8, minutes=45) and now < tm["decision"]:
            self.morning(now)
            st = self.load_state()
            st["prepared"] = str(day.date())
            self.save_state(st)
            return "morning"
        if st.get("decided") != str(day.date()) and tm["decision"] <= now <= tm["cutoff"]:
            st["decided"] = str(day.date())                                                   # one decision a day, even if it fails half-way (no double orders)
            self.save_state(st)
            plan = self.plan(now)
            for n in plan.notes:
                self.log(n, "warn")
            o = plan.orders
            self.log(f"{now:%H:%M:%S} decision on the {tm['snapshot']:%H:%M} snapshot: {int((o.kind == 'exit').sum())} exit(s), {int((o.kind == 'entry').sum())} entr(ies), "
                     f"{len(plan.candidates)} candidate(s), capital ${plan.capital:,.0f}", "ok")
            if self.live.armed and len(o) and o.ok.any():
                res = self.submit(plan)
                bad = [r for r in res if r["status"] in ("rejected", "unknown")]
                self.log(f"sent {len(res) - len(bad)} / {len(res)} market-on-close orders" + (f", {len(bad)} rejected / unknown" if bad else ""), "warn" if bad else "trade")
            else:
                self.journal("plan_dry" if not self.live.armed else "plan_no_orders", **self._decision_row(plan), n_orders=len(o))
            on_plan(plan)
            return "decision"
        if st.get("decided") != str(day.date()) and tm["cutoff"] < now < tm["close"] and st.get("missed") != str(day.date()):
            st["missed"] = str(day.date())
            self.save_state(st)
            self.log(f"today's decision window ({tm['decision']:%H:%M:%S} - {tm['cutoff']:%H:%M:%S}) was missed - nothing traded today; the backtest would have traded", "err")
            self.journal("missed", day=day)
            return "missed"
        return None

    def run(self, stop: threading.Event, on_plan: Callable[[MRPlan], None] = lambda p: None) -> None:
        if self.broker is None:
            self.connect()
        if self.live.armed and not self.flock.acquire():
            raise RuntimeError("another engine of this model is running (lock held)")
        st = self.load_state()
        if st.get("model") and st["model"] != self.m.to_dict() and st.get("book"):
            self.log("the model differs from the one that opened the current positions - they are managed by the new model's exit rules from now on", "warn")
        st["model"] = self.m.to_dict()
        self.save_state(st)
        self.log(f"loop started ({'ARMED' if self.live.armed else 'DRY RUN'}, {self.mode}, model '{self.m.name}', feed {self.live.feed})", "ok")
        try:
            while not stop.is_set():
                try:
                    self.tick(on_plan=on_plan)
                except Exception as e:                                                        # noqa: BLE001
                    self.log(f"step error: {type(e).__name__}: {str(e)[:200]}", "err")
                    self.journal("error", msg=f"{type(e).__name__}: {str(e)[:300]}")
                stop.wait(1.0)
        finally:
            self.flock.release()
            self.log("loop stopped", "info")

    # ── manual actions ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def kill(self) -> list:
        """KILL: cancel this engine's open orders, then close every position in this model's symbols with market orders (OPG when the market is closed). Works in DRY RUN too."""
        if self.broker is None:
            self.connect()
        with self._lock:
            out = []
            now = self.now()
            try:
                self.cancel_ours(now.normalize())
            except Exception as e:                                                            # noqa: BLE001
                out.append(dict(symbol="*", status=f"cancel: {e}"))
            clk = self.clock(0)
            tif = "day" if clk["is_open"] else "opg"
            possible = self.possible_symbols()
            for s, p in self.broker.positions().items():
                q = p["qty"]
                if s not in possible or q == 0:
                    if q:
                        out.append(dict(symbol=s, qty=q, status="not this model's symbol - left alone"))
                    continue
                cid = f"{PREFIX}-KILL-{s.replace('.', '')}-{int(time.time()) % 1000000}"
                try:
                    self.broker.submit(s, abs(q), "sell" if q > 0 else "buy", tif, cid)
                    out.append(dict(symbol=s, qty=q, status=f"close sent ({tif})"))
                except Exception as e:                                                        # noqa: BLE001
                    out.append(dict(symbol=s, qty=q, status=f"FAILED {type(e).__name__}: {str(e)[:100]}"))
            self.journal("kill", actions=out)
            return out

    def reset_drawdown(self) -> float:
        st = self.load_state()
        eq = float(self.broker.account()["equity"])
        st["hwm"], st["stopped"] = eq, None
        self.save_state(st)
        return eq

    def preflight(self) -> list:
        out: list = []
        add = lambda n, l, m: out.append(Check(n, l, m))
        try:
            a = self.connect()
            add("broker", "ok", f"{'PAPER' if a['paper'] else 'LIVE'} {a['number'][:4]}**** · {a['status']} · equity ${a['equity']:,.0f} · buying power ${a['buying_power']:,.0f} · "
                                f"{a['latency_ms']} ms")
            add("keys", "ok" if a["keys"] == "own" else ("warn" if not self.live.armed else "fail"),
                f"from the {a['key_src']}" if a["keys"] == "own" else f"from the {a['key_src']} (not the project's key file) - fine for DRY RUN, refused when ARMED")
            owner = self.account_owner()
            add("account owner", "ok" if owner in (None, KEY) else "fail", f"claimed by {owner}" if owner else "not claimed yet (claimed on the first armed order)")
            if a["blocked"]:
                add("account", "fail", "trading is blocked")
            add("margin", "ok" if a["multiplier"] >= 2 else "warn",
                f"{a['multiplier']:.0f}x: market-on-close buys can be placed while the same auction's sells are pending (the book itself never exceeds 1x)" if a["multiplier"] >= 2 else
                "cash account (1x): buys placed while the same auction's sells are still pending may be refused for buying power")
            if self.m.direction == "short":
                add("shorting", "ok" if a["shorting"] else "fail", "enabled" if a["shorting"] else "disabled - this model sells short")
            add("market", "ok", f"{'open' if a['market_open'] else 'closed'} · next open {a['next_open']:%a %H:%M} · next close {a['next_close']:%a %H:%M} ET")
        except Exception as e:                                                                # noqa: BLE001
            add("broker", "fail", f"cannot reach Alpaca: {type(e).__name__}: {str(e)[:140]}")
            return out
        probs = self.model_problems()
        add("model", "ok" if not probs else "fail", f"'{self.m.name}': {self.m.entry_signal} {self.m.entry_op} {self.m.entry_value:g}, {self.m.slots} slots, rank by {self.m.rank_by}, "
                                                    f"decision {self.m.minute // 100}:{self.m.minute % 100:02d}" + (" - " + "; ".join(probs) if probs else ""))
        now = self.now()
        tm = self.times(now.normalize())
        if tm:
            add("schedule", "ok", f"today: snapshot {tm['snapshot']:%H:%M}, decision {tm['decision']:%H:%M:%S}, market-on-close cut-off {tm['cutoff']:%H:%M:%S}, close {tm['close']:%H:%M} ET")
        else:
            add("schedule", "ok", "no session today")
        d = load_daily(["SPY"])
        if len(d):
            prev = self.last_session(now - pd.Timedelta(days=1) if now.hour < 17 else now)
            age = (prev - d.date.max()).days
            add("daily history", "ok" if age <= 0 else "warn", f"cache through {d.date.max():%Y-%m-%d}" + ("" if age <= 0 else f" (the latest session is {prev:%Y-%m-%d}: the morning step tops it up; "
                                                                                                         f"the decision reads the newest bars itself)"))
        else:
            add("daily history", "fail", "no daily bars - Update data")
        if self.m.universe.startswith("top"):
            u = D.load_universe()
            mon = now.to_period("M").to_timestamp()
            add("universe", "ok" if len(u) and u.month.max() >= mon else "warn", f"members through {u.month.max():%Y-%m}" if len(u) else "no universe")
        delayed = now - pd.Timedelta(minutes=D.SIP_DELAY_MIN)
        feeds = {"sip_delayed": [("sip", delayed)], "iex": [("iex", now)], "sip": [("sip", now)], "sip+iex": [("sip", delayed), ("iex", now)]}.get(self.live.feed, [])
        for fd, end in feeds:
            try:
                t0 = time.time()
                f = self.broker.bars(["SPY"], end - pd.Timedelta(days=5), end, fd).get("SPY")
                add(f"live feed ({fd})", "ok" if f is not None and len(f) else "warn",
                    f"newest SPY bar {f.index.max():%m-%d %H:%M} ({time.time() - t0:.1f} s)" if f is not None and len(f) else "no bars")
            except Exception as e:                                                            # noqa: BLE001
                add(f"live feed ({fd})", "fail", f"{type(e).__name__}: {str(e)[:120]}" + (" - real-time SIP needs a paid data plan" if fd == "sip" and end == now else ""))
        st = self.load_state()
        add("state", "warn" if st.get("stopped") or st.get("pending") else "ok",
            ("DRAWDOWN STOP active; " if st.get("stopped") else "") + ("an unfinished submit is recorded (check the journal); " if st.get("pending") else "")
            + f"{len(st.get('book') or {})} position(s) in the book · high-water mark {st.get('hwm')}")
        probs = self.arm_problems()
        add("arming", "ok" if not probs else ("warn" if not self.live.armed else "fail"), "can be armed" if not probs else "; ".join(probs))
        return out

    # ── check against the backtest's data ────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def check_against_backtest(self, log: Callable[[str], None] = lambda s: None) -> pd.DataFrame:
        """For every journaled decision: rebuild the decision from the BACKTEST's data (cached daily bars + the cached SIP 15-minute bars of that day, i.e. what the backtest sees)
        with the positions and capital the live engine had, and compare the orders. Differences can only come from the data (feed, timing, adjustments) - not from the code."""
        rows = []
        for r in self.read_journal(("decision", "plan_dry", "plan_no_orders")):
            day = pd.Timestamp(r["day"]).normalize()
            m = MRModel.from_dict(r.get("model_spec") or self.m.to_dict())
            syms = list(dict.fromkeys(self.universe_symbols(day, m) + list(r.get("held") or {})))
            try:
                mk = Market(syms, minute=m.minute, start=day - pd.Timedelta(days=self.live.history_days), end=day + pd.Timedelta(days=1))
            except Exception as e:                                                            # noqa: BLE001
                log(f"{day:%Y-%m-%d}: cannot build the backtest data ({type(e).__name__})")
                continue
            if not len(mk.dates) or mk.dates[-1] != day or not np.isfinite(mk.sP[-1]).any():
                log(f"{day:%Y-%m-%d}: not in the cached data yet - update the data first")
                continue
            t = len(mk.dates) - 1
            pos = {}
            for s, h in (r.get("held") or {}).items():
                if s in mk.col:
                    ed = pd.Timestamp(h.get("entry_date") or day)
                    et = min(int(np.searchsorted(mk.dates, ed)), t)
                    pos[s] = Pos(symbol=s, shares=int(h["shares"]), entry_t=et, entry_px=float(h["entry_px"]), exit_open_t=et + 1 if m.hold == "overnight" else None)
            dec = decide(m, t, Signals(mk), E.universe_mask(mk, m)[t], pos, float(r["capital"]))
            bt = {(s, "exit"): (abs(pos[s].shares), "") for s, _ in dec.exits}
            bt.update({(s, "entry"): (abs(sh), f"rank {rk}") for s, sh, _, rk in dec.entries})
            lv = {(o["symbol"], o["kind"]): (int(o["qty"]), o.get("why", "")) for o in (r.get("orders") or [])}
            for k in sorted(set(bt) | set(lv)):
                b, l_ = bt.get(k), lv.get(k)
                rows.append(dict(day=day, symbol=k[0], kind=k[1], live_qty=l_[0] if l_ else 0, backtest_qty=b[0] if b else 0,
                                 match="same" if (b and l_ and b[0] == l_[0]) else ("qty differs" if b and l_ else ("live only" if l_ else "backtest only"))))
        return pd.DataFrame(rows, columns=["day", "symbol", "kind", "live_qty", "backtest_qty", "match"])
