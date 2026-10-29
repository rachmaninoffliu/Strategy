"""
Live / paper engine for the time-shared momentum book (no Qt): the backtest's decision functions on the broker's positions, orders at the backtest's times and auctions.

Daily cycle (ET) - run() calls tick() every second:
  morning (from 08:45)          top up the daily bars and the 15-minute SIP bars of the universe; check the year's ML model; last night's overnight names get market-on-open
                                (OPG) sells (after the 09:28 OPG cut-off: a market sell at once, journaled as late)
  intraday checks               at every check (bar end + a few seconds: 10:00:05, 10:30:05, ... 30 minutes before the close) today's bars of the intraday symbol from the
                                intraday feed (IEX) -> the noise band and noise_action() -> a market buy / sell
  overnight prefetch            ~10 minutes before the decision: today's SIP bars of the universe so far (the decision then reads only the last bar)
  overnight decision            15:46 for the 15:30 snapshot on the free delayed SIP: the intraday position (if still held) gets a market-on-close sell; then
                                MomMarket (daily history + cached SIP bars + today's bars) -> prepare() -> decide_overnight() -> market-on-close buys, all before the 15:50 cut-off
  after the close (+5 min)      fills journaled, the overnight book (entry day / price / trade id) rebuilt from the broker's positions, today's intraday round trips journaled

Auction orders (MomLiveSettings.auction_orders) - the only difference between the two choices is the order type; the model, its decisions and sizes are the same:
  auction  MOC / MOO orders, as the model (and its backtest)
  market   market DAY orders in their place (Alpaca's PAPER account does not run the auctions: it fills MOC / MOO orders at the quote in the last seconds, partly at
           random, and expires the rest - 2026-10-28: 6 of 20 MOC buys filled). The 15:46 decision is unchanged; its orders are queued and sent mkt_close_s (5) seconds
           before the close - the intraday sale first, then the exits, then the entries, in parallel: sells before buys; nothing in the last 2 seconds - and last
           night's names are sold with market orders sent mkt_open_s (-180) seconds from the open, i.e. 09:27:00, queued for the open (on a real Alpaca account an order
           received before 09:28 goes into the opening cross). The fills are compared with the backtest's auction prices (book_view, fill_report).
Orders that are not auction orders - the intraday checks' market orders - are the same under both.

Backtest parity: the same MomMarket, prepare(), decide_overnight(), noise_upper() / noise_action(), the same sizing (whole shares; (1 - reserve) of the capital) and the
same auctions. tests/test_mom.py replays a backtest session by session through this engine with a simulated broker - with auction orders and with market orders in their
place (filled there at the auction prices) - and requires every fill to equal a backtest trade and the final equity to match. Live-only overlays (not in the backtest): daily-loss halt and drawdown stop -
both off unless set.

Safety (as the other apps): PAPER + DRY RUN at start; ARMED needs the typed phrase "ARM PAPER", the project's own keys and an account no other strategy claimed (one engine
per account, operating-system lock); real money is disabled unless RESARB_ALLOW_LIVE=1; positions in symbols the book can never trade are left alone; KILL cancels and flattens.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import data as D
from ..live import ARM_PHRASE, LIVE_PHRASE, Check, FileLock, Throttle, now_et  # noqa: F401  (ARM_PHRASE / LIVE_PHRASE re-exported for the GUI)
from ..mr import engine as MRE
from ..mr.market import COLS, _fetch_daily, load_daily, update_daily
from . import engine as E
from . import ml
from . import quotes as Q
from .features import noise_sigma
from .market import MomMarket, closes, lead_bars
from .models import MomModel, check_slots, decide_overnight, intraday_shares, noise_action, noise_upper, slot_end

KEY, PREFIX = "MomentumBook", "MB"
NY = "America/New_York"
OPG_CUTOFF = pd.Timedelta(hours=9, minutes=28)                                                  # Alpaca refuses market-on-open orders from 09:28 to 19:00
FEEDS = ("sip_delayed", "sip")                                                                  # today's SIP bars for the overnight decision: free (16 minutes late) | real time (paid)
FINAL = ("filled", "canceled", "rejected", "expired", "done_for_day")
NOT_SENT = ("rejected", "unknown", "late", "disarmed", "killed")                                 # _send statuses of an order that did not (surely) go out
AUCTION_ORDERS = ("auction", "market")                                                          # auction: MOC / MOO, as the model; market: market DAY orders in their place
_OLD_ORDERS = {"auto": "auction", "emulate": "market", "model": "auction"}                      # names saved by earlier versions


@dataclass
class MomLiveSettings:
    paper: bool = True
    armed: bool = False
    capital: float = 0.0                 # > 0: the book trades this fixed amount; otherwise account equity x equity_fraction
    equity_fraction: float = 1.0
    feed: str = "sip_delayed"            # overnight decision bars: sip_delayed (free plan, decision 15:30 or earlier) | sip (real-time SIP plan)
    decision_delay_s: float = 6.0        # wait after the decision bar's data is due
    intraday_delay_s: float = 5.0        # wait after an intraday bar ends before reading it
    check_late_s: float = 240.0          # an intraday check more than this late is skipped (journaled as missed)
    moc_cutoff_min: float = 10.5         # last moment to send market-on-close orders, in minutes before the close (Alpaca: 15:50)
    auction_orders: str = "auction"      # auction: MOC / MOO orders, as the model | market: market DAY orders in their place, mkt_close_s before the close and
                                         # mkt_open_s from the open - paper and real money alike; only the model's auction orders are affected
    mkt_close_s: int = 5                 # 15:59:55: the intraday sale, exits and entries go out in parallel (~1 s for 20 orders); nothing in the last 2 s
    mkt_open_s: int = -180               # 09:27:00: queued for the open (before Alpaca's 09:28 cut-off for the opening cross on a real account)
    prefetch_min: float = 10.0           # read today's universe bars this many minutes before the decision
    daily_loss_halt: float = 0.0         # > 0: no new entries after this loss today (live-only overlay)
    dd_stop: float = 0.0                 # > 0: sell everything and stay flat after this drawdown from the high-water mark, until reset (live-only overlay)
    shared_keys_ok: bool = False
    max_calls_per_min: int = 150
    history_days: int = 460              # calendar days of daily history (252-session windows need ~370)
    bar_days: int = 45                   # calendar days of cached 15-minute bars read for the decision (20-session volume mean, 14-session noise band)
    refresh_data: bool = True            # morning top-up of the caches (the tests turn it off)


@dataclass
class MomPlan:
    day: pd.Timestamp
    decided_at: pd.Timestamp
    snapshot_to: pd.Timestamp
    capital: float
    equity: float
    positions: pd.DataFrame              # symbol, shares, entry_date, entry_px, last, value, pnl, exit
    candidates: pd.DataFrame             # symbol, score, rank, move, forecast, taken, last
    orders: pd.DataFrame                 # symbol, side, qty, tif, ref_px, notional, why, ok, note, cid, kind, score, rank, move, forecast
    notes: list = field(default_factory=list)
    halted: bool = False
    stopped: bool = False
    real: bool = True
    trend: bool = True
    ml: Optional[dict] = None
    shadow: Optional[pd.DataFrame] = None


# ── broker adapter (the tests use a simulated one with the same methods) ─────────────────────────────────────────────────────────────────────────────
class AlpacaBroker:
    def __init__(self, paper: bool, calls_per_min: int = 150, prefix: str = PREFIX):
        self.paper, self.prefix = paper, prefix
        self.tc = D.trading_client(prefix, paper)
        self.tc._retry = 0                                                                      # no silent resubmit seconds later (past the close it would trade next day)
        self.dc = D.data_client(prefix)
        self._thr = Throttle(calls_per_min)

    def _call(self, fn, *a, **k):
        self._thr.wait()
        return fn(*a, **k)

    def account(self) -> dict:
        a = self._call(self.tc.get_account)
        return dict(number=str(a.account_number), status=str(getattr(a.status, "value", a.status)), equity=float(a.equity), last_equity=float(a.last_equity),
                    cash=float(a.cash), buying_power=float(a.buying_power), multiplier=float(a.multiplier or 1), shorting=bool(a.shorting_enabled),
                    blocked=bool(a.trading_blocked or a.account_blocked), pdt=bool(getattr(a, "pattern_day_trader", False)),
                    daytrade_count=int(getattr(a, "daytrade_count", 0) or 0))

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
        return dict(id=str(o.id), status=_status(o), filled_qty=float(o.filled_qty or 0), filled_avg_price=float(o.filled_avg_price or 0) or np.nan)

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
        return [self._row(o) for o in oo if str(o.client_order_id or "").startswith(self.prefix + "-")]

    def orders_since(self, after: pd.Timestamp) -> list:
        """Every order of this book submitted after ``after`` (ET), oldest first - paged, 500 per call."""
        from alpaca.common.enums import Sort
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        out, seen = [], set()
        aft = pd.Timestamp(after).tz_localize(NY).tz_convert("UTC")
        while True:
            oo = self._call(self.tc.get_orders, GetOrdersRequest(status=QueryOrderStatus.ALL, after=aft.to_pydatetime(), limit=500, direction=Sort.ASC))
            for o in oo:
                if str(o.id) not in seen and str(o.client_order_id or "").startswith(self.prefix + "-"):
                    seen.add(str(o.id))
                    out.append(self._row(o))
            if len(oo) < 500:
                return out
            nxt = pd.Timestamp(oo[-1].submitted_at).tz_convert("UTC")
            if nxt <= aft:
                return out
            aft = nxt - pd.Timedelta(microseconds=1)

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

    def daily_raw(self, symbols: list, start: pd.Timestamp) -> pd.DataFrame:
        """Unadjusted official daily open / close - what fills (raw prices) are compared with."""
        return Q._raw_daily(self.dc, list(symbols), start, pd.Timestamp.now(tz="UTC"))


def _status(o) -> str:
    return str(getattr(getattr(o, "status", None), "value", getattr(o, "status", "?")))


def _cid_sym(s: str) -> str:
    return s.replace(".", "")


def _net(held: float, pending: Optional[dict]) -> int:
    """Shares still to close of a position ``held`` (long > 0, short < 0) after the book's open orders on it: sells reduce a long, buys a short; never past flat."""
    if not pending:
        return int(held)
    if held > 0:
        return int(max(held - pending.get("sell", 0.0), 0))
    return int(min(held + pending.get("buy", 0.0), 0))


def _open_offset(s: int) -> str:
    return f"{-s} s before the open (queued for it)" if s < 0 else f"{s} s after the open"


def _how(o: dict) -> str:
    """How an order of the book was executed, from its time in force and its client id (suffix o = opening exit, x = closing exit, e = entry, iC = intraday closing sale)."""
    if o["tif"] in ("opg", "cls"):
        return "opening auction" if o["tif"] == "opg" else "closing auction"
    leg = o["cid"].split("-")[-1]
    if leg[:1] in ("x", "e") or leg[:2] == "iC":
        return "market at the close"
    if leg[:1] == "o":
        t = o.get("submitted_at")
        late = t is not None and pd.notna(t) and pd.Timestamp(t) - pd.Timestamp(t).normalize() > pd.Timedelta(hours=9, minutes=35)
        return "market (late, after the open)" if late else "market at the open"
    return "market"


# ── the engine ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class MomLiveEngine:
    KEY, PREFIX = KEY, PREFIX

    def __init__(self, model: MomModel, live: MomLiveSettings, folder: str, log: Callable[[str, str], None] = lambda m, k="info": print(m), broker=None,
                 clock_fn: Callable[[], pd.Timestamp] = now_et, shadow: Optional[MomModel] = None, bars_fn=None):
        self.m, self.live, self.folder, self.log = model, live, folder, log
        self._own: Optional[dict] = None                                                        # own_shares() cache: the book's orders
        self.shadow = shadow
        self.broker = broker
        self._injected = broker is not None
        self.now = clock_fn
        self.bars_fn = bars_fn                                                                  # tests: the cached bars (else data.load_bars)
        self.acct: dict = {}
        self._lock = threading.RLock()
        self._jlock = threading.Lock()
        self._clk = (0.0, None)
        self.flock = FileLock(self.path("engine", "lock"))
        self.last_plan: Optional[MomPlan] = None
        self.last_check: Optional[dict] = None
        self._st: tuple = (None, None)                                                          # (state file stamp, state) - the last read / write
        self._hist: dict = {}                                                                   # day -> daily history (checked against the broker in the morning)
        self._pre: dict = {}                                                                    # day -> prefetched SIP bars of the universe

    # ── files ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return "paper" if self.live.paper else "live"

    @property
    def emulate(self) -> bool:
        """Market DAY orders in place of the model's MOC / MOO orders (the setting 'market'; paper and real money alike - see the module notes)."""
        return self.m.uses_auctions and _OLD_ORDERS.get(self.live.auction_orders, self.live.auction_orders) == "market"

    @property
    def close_tif(self) -> str:
        return "day" if self.emulate else "cls"

    def path(self, name: str, ext: str = "json") -> str:
        d = os.path.join(self.folder, "reports")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"mb_{name}_{self.mode}.{ext}")

    def load_state(self) -> dict:
        p = self.path("state")
        base = {"book": {}, "intraday": None, "checks": {}, "hwm": None, "day": None, "day_start_equity": None, "stopped": None, "pending": None, "attempt": 0,
                "prepared": None, "prefetched": None, "decided": None, "reconciled": None, "next_trade_id": 1, "model": None, "missed": None, "opened": None, "closed": None,
                "killed": None, "reconcile_failures": 0}
        try:
            stamp = os.stat(p).st_mtime_ns
        except OSError:
            return base
        if self._st[0] == stamp:                                                                # unchanged since this engine read / wrote it: no file read (tick runs every second)
            return copy.deepcopy(self._st[1])
        try:
            with open(p, encoding="utf-8") as f:
                st = {**base, **json.load(f)}
        except (OSError, ValueError) as e:
            raise RuntimeError(f"state file {p} is unreadable ({e}) - fix or move it") from e
        self._st = (stamp, copy.deepcopy(st))
        return st

    def save_state(self, st: dict) -> None:
        p = self.path("state")
        tmp = p + ".tmp"
        st = json.loads(json.dumps(st, default=str))                                            # what the file holds (and what a re-read would return)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        for i in range(6):
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                if i == 5:
                    raise
                time.sleep(0.2)
        self._st = (os.stat(p).st_mtime_ns, st)

    def journal(self, kind: str, **row) -> None:
        row = {"ts": str(self.now()), "kind": kind, "strategy": KEY, "model": self.m.name, "armed": bool(self.live.armed), **row}
        line = json.dumps(row, default=str) + "\n"
        with self._jlock, open(self.path("journal", "jsonl"), "a", encoding="utf-8") as f:     # the closing orders are sent from several threads
            f.write(line)

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
        """Connect to the account (read-only until ARMED: a real-money account can be viewed - positions, orders, history - but every order path refuses it unless
        RESARB_ALLOW_LIVE=1 and the live phrase was typed)."""
        if self.broker is None:
            self.broker = AlpacaBroker(self.live.paper, self.live.max_calls_per_min)
        t0 = time.time()
        a = self.broker.account()
        n0 = self.now()
        c = self.clock(0)
        n1 = self.now()
        ts = c.get("ts")
        off = (n0 + (n1 - n0) / 2 - pd.Timestamp(ts)).total_seconds() if ts is not None and pd.notna(ts) else None   # + = this PC's clock is ahead of Alpaca's
        self.acct = dict(paper=self.live.paper, **a, market_open=c["is_open"], next_open=c["next_open"], next_close=c["next_close"], latency_ms=int((time.time() - t0) * 1000),
                         keys=self.key_source(), key_src=("simulated broker" if self._injected else D.key_source(PREFIX, self.live.paper)), clock_offset_s=off)
        return self.acct

    def clock(self, max_age: float = 30.0) -> dict:
        t, c = self._clk
        if c is None or time.monotonic() - t > max_age or self._injected:
            c = self.broker.clock()
            self._clk = (time.monotonic(), c)
        return c

    # ── account ownership (shared registry with the residual and MR apps: one strategy per account) ─────────────────────────────────────────────────
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
        os.makedirs(D.DATA_DIR, exist_ok=True)
        with open(self._registry_path(), "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=1)

    def arm_problems(self) -> list:
        """Why this engine may NOT be armed now (empty = OK)."""
        out = []
        if not self.live.paper and os.environ.get("RESARB_ALLOW_LIVE") != "1":
            out.append("real money is disabled")
        if self.load_state().get("killed"):
            out.append(f"KILL is active (since {self.load_state()['killed'][:16]}): clear it with 'Reset drawdown base / clear KILL' to trade again")
        if self.key_source() != "own" and not self.live.shared_keys_ok:
            out.append("no own keys: put 'Alpaca Kay.txt' (Key ID / Secret Key) in the project folder - the ~/.claude.json keys of the Claude tools are refused for orders")
        out += self.model_problems()
        if self.m.uses_ml and ml.load_model(self.folder, self.m, self.now().year) is None:
            out.append(f"no ML model for {self.now().year}: Backtest tab > Train ML models")
        if self.acct:
            owner = self.account_owner()
            if owner and owner != KEY:
                out.append(f"account claimed by {owner}")
            if self.acct.get("blocked"):
                out.append("trading is blocked on the account")
            if self.m.intraday_symbol and float(self.acct.get("equity") or 0) < 25_000:
                out.append("the intraday leg day-trades: the pattern-day-trader rule needs at least $25,000 of equity in a margin account")
        return out

    def model_problems(self) -> list:
        out = []
        if self.live.feed not in FEEDS:
            out.append(f"unknown feed {self.live.feed!r}")
        try:
            lead = pd.Timedelta(minutes=D.BAR_MIN * lead_bars(self.m.minute))
        except ValueError as e:
            return [str(e)]
        if self.m.overnight and self.data_delay() >= lead - pd.Timedelta(minutes=self.live.moc_cutoff_min):
            out.append(f"a {self.m.minute // 100}:{self.m.minute % 100:02d} decision cannot be made on the {self.live.feed} feed before the market-on-close cut-off - "
                       f"use 15:30 with the delayed SIP, or the real-time SIP plan")
        if self.m.intraday_symbol and self.m.intraday_feed not in ("iex", "sip"):
            out.append(f"unknown intraday feed {self.m.intraday_feed!r}")
        if self.m.cash_reserve < 0 or self.m.k < 1:
            out.append("cash reserve must be >= 0 and k >= 1")
        if _OLD_ORDERS.get(self.live.auction_orders, self.live.auction_orders) not in AUCTION_ORDERS:
            out.append(f"unknown auction orders {self.live.auction_orders!r}")
        if not 3 <= self.live.mkt_close_s < 60 * self.live.moc_cutoff_min or not -900 <= self.live.mkt_open_s <= 1200:
            out.append("market orders in place of the auctions: the closing ones must go out after the decision window and at least 3 s before the close (nothing is sent "
                       "in the last 2 s), the opening sales between 15 minutes before and 20 minutes after the open (before the first intraday check)")
        return out

    def data_delay(self) -> pd.Timedelta:
        d = pd.Timedelta(seconds=self.live.decision_delay_s)
        return d + (pd.Timedelta(minutes=D.SIP_DELAY_MIN) if self.live.feed == "sip_delayed" else pd.Timedelta(0))

    # ── calendar ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def session_close(self, day: pd.Timestamp) -> Optional[pd.Timestamp]:
        c = closes(day, day)
        return c.iloc[0] if len(c) else None

    def times(self, day: pd.Timestamp) -> Optional[dict]:
        """Today's schedule: open, intraday checks [(slot, bar end, due)], snapshot, prefetch, decision, market-on-close cut-off, close, and the emulated auctions' order times
        (emu_open, emu_close). None = no session."""
        close = self.session_close(day)
        if close is None:
            return None
        nb = int((close - day - pd.Timedelta(hours=9, minutes=30)) / pd.Timedelta(minutes=D.BAR_MIN))
        snap = close - pd.Timedelta(minutes=D.BAR_MIN * lead_bars(self.m.minute))
        dec = snap + self.data_delay()
        checks = []
        if self.m.intraday_symbol:
            checks = [(s, slot_end(day, s), slot_end(day, s) + pd.Timedelta(seconds=self.live.intraday_delay_s)) for s in check_slots(nb, self.m.intraday_every)]
        op = day + pd.Timedelta(hours=9, minutes=30)
        return dict(open=op, checks=checks, snapshot=snap, prefetch=dec - pd.Timedelta(minutes=self.live.prefetch_min), decision=dec,
                    cutoff=close - pd.Timedelta(minutes=self.live.moc_cutoff_min), close=close, nb=nb, emu_open=op + pd.Timedelta(seconds=self.live.mkt_open_s),
                    emu_close=close - pd.Timedelta(seconds=self.live.mkt_close_s))

    def last_session(self, now: pd.Timestamp) -> pd.Timestamp:
        c = closes(now.normalize() - pd.Timedelta(days=10), now.normalize())
        days = [d for d in c.index if d < now.normalize() or (d == now.normalize() and now >= d + pd.Timedelta(hours=9, minutes=45))]
        return days[-1]

    # ── universe / data ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def universe_symbols(self, day: pd.Timestamp, m: Optional[MomModel] = None) -> list:
        """The symbols the book may hold overnight on ``day`` plus SPY (trend, market features) and the intraday symbol."""
        m = m or self.m
        out: list = []
        for part in [p.strip() for p in m.universe.split("+") if p.strip()]:
            if not part.startswith("top"):
                out += MRE._part_symbols(part)
                continue
            n = int(part[3:] or 100)
            u = D.load_universe()
            if not len(u):
                raise RuntimeError("no stock universe: run 'Update data' first")
            mon = day.to_period("M").to_timestamp()
            mm = u[(u.month == mon) & (u["rank"] <= n)]
            if not len(mm):
                mm = u[(u.month == u[u.month <= mon].month.max()) & (u["rank"] <= n)]
            out += sorted(set(mm.symbol))
        out += ["SPY"] + ([m.intraday_symbol] if m.intraday_symbol else [])
        return list(dict.fromkeys(out))

    def possible_symbols(self) -> set:
        try:
            return set(E.model_symbols(self.m))
        except Exception:                                                                     # noqa: BLE001
            return set(self.universe_symbols(self.now().normalize()))

    def history(self, syms: list, day: pd.Timestamp, notes: list) -> pd.DataFrame:
        """Official daily bars before ``day``: the cache, checked against the broker's latest adjusted bars (a split / dividend since the cache was written rescales the cached
        history; sessions the cache misses are filled from the broker). Kept in memory for the day."""
        key = (day, tuple(sorted(syms)))
        if key in self._hist:
            return self._hist[key]
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
            self._hist = {key: h}
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
        out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
        self._hist = {key: out}
        return out

    def _today_sip(self, syms: list, day: pd.Timestamp, now: pd.Timestamp) -> dict:
        """Today's complete SIP bars of ``syms`` readable at ``now``: the prefetched part plus a fresh read of the rest."""
        start = day + pd.Timedelta(hours=9, minutes=30)
        end = now - (pd.Timedelta(minutes=D.SIP_DELAY_MIN) if self.live.feed == "sip_delayed" else pd.Timedelta(0))
        if end <= start:
            return {}
        pre = self._pre.get(day) or {}
        have_to = min((b.index.max() for b in pre.values() if b is not None and len(b)), default=None) if pre else None
        frm = start if have_to is None or set(syms) - set(pre) else have_to
        new = self.broker.bars(sorted(syms), frm, end, "sip")
        out = {}
        for s in syms:
            a, b = pre.get(s), new.get(s)
            parts = [x for x in (a[a.index < frm] if a is not None and len(a) else None, b) if x is not None and len(x)]
            if parts:
                x = pd.concat(parts) if len(parts) > 1 else parts[0]
                x = x[~x.index.duplicated(keep="last")]
                out[s] = x[x.index + pd.Timedelta(minutes=D.BAR_MIN) <= end]                  # complete bars only
        return out

    def prefetch(self, now: Optional[pd.Timestamp] = None) -> int:
        """Read today's SIP bars of the universe so far (the decision then reads only the newest bars)."""
        now = now or self.now()
        day = now.normalize()
        syms = self.universe_symbols(day)
        self._pre = {day: self._today_sip(syms, day, now)}
        return sum(len(b) for b in self._pre[day].values())

    def _load_bars(self, s: str, feed: str) -> pd.DataFrame:
        return self.bars_fn(s, feed) if self.bars_fn else D.load_bars(s, feed)

    def overnight_market(self, day: pd.Timestamp, now: pd.Timestamp, held: list, notes: list) -> tuple:
        syms = list(dict.fromkeys(self.universe_symbols(day) + [s for s in held if s in self.possible_symbols()]))
        hist = self.history(syms, day, notes)
        bars = self._today_sip(syms, day, now)
        if not bars:
            raise RuntimeError(f"no 15-minute bars for {day:%Y-%m-%d} yet ({self.live.feed} feed)")
        mk = MomMarket(syms, minute=self.m.minute, start=day - pd.Timedelta(days=self.live.history_days), daily=hist, extra_bars=bars, hist_bars=True,
                       bars_from=day - pd.Timedelta(days=self.live.bar_days), intraday_symbol="", bars_fn=self.bars_fn)
        t = len(mk.dates) - 1
        if t < 0 or mk.dates[t] != day:
            raise RuntimeError(f"today's session {day:%Y-%m-%d} is not in the data")
        return mk, t, bars

    def intraday_market(self, day: pd.Timestamp, now: pd.Timestamp, notes: list) -> tuple:
        """The intraday symbol's MomMarket: daily history, the cached SIP bars of the last weeks (noise band) and today's bars from the intraday feed."""
        s = self.m.intraday_symbol
        hist = self._hist_for([s], day, notes)
        start = day + pd.Timedelta(hours=9, minutes=30)
        b = self.broker.bars([s], start, now, self.m.intraday_feed).get(s)
        if b is not None and len(b):
            b = b[b.index + pd.Timedelta(minutes=D.BAR_MIN) <= now]
        if b is None or not len(b):
            raise RuntimeError(f"no {self.m.intraday_feed} bars of {s} today yet")
        kw = dict(extra_bars={s: b}) if self.m.intraday_feed == "sip" else dict(feed_bars=b)
        mk = MomMarket([s], minute=self.m.minute, start=day - pd.Timedelta(days=self.live.history_days), daily=hist, hist_bars=True,
                       bars_from=day - pd.Timedelta(days=self.live.bar_days), intraday_symbol=s, intraday_feed=self.m.intraday_feed, bars_fn=self.bars_fn, **kw)
        t = len(mk.dates) - 1
        if t < 1 or mk.dates[t] != day:
            raise RuntimeError(f"today's session {day:%Y-%m-%d} is not in the {s} data")
        return mk, t

    def _hist_for(self, syms: list, day: pd.Timestamp, notes: list) -> pd.DataFrame:
        """The day's checked history restricted to ``syms`` when it is already in memory, else a fresh check."""
        for (d, ss), h in self._hist.items():
            if d == day and set(syms) <= set(ss):
                return h[h.symbol.isin(syms)]
        return self.history(syms, day, notes)

    # ── positions ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def own_shares(self) -> dict:
        """{symbol: shares this book's own orders hold}: its filled buys minus its filled sales (client id MB-..., from the broker's order history - paper and real money
        alike), in fill order and never below zero (a sale no buy of the book explains does not count). Cached: each call re-reads only the orders that can still change
        and the new ones; the first reads the history since the journal began (a year without a journal)."""
        now = self.now()
        c = self._own
        if c is None:
            jr = self.read_journal()
            since = (pd.Timestamp(jr[0]["ts"]).normalize() if jr else now.normalize() - pd.Timedelta(days=365)) - pd.Timedelta(days=3)
            c = self._own = {"orders": {}, "mark": since}
        open_ = [o["submitted_at"] for o in c["orders"].values() if o["status"] not in FINAL and o["submitted_at"] is not None]
        since = min([c["mark"]] + open_) - pd.Timedelta(minutes=1)
        get = getattr(self.broker, "orders_since", None)
        for o in (get(since) if get else self.broker.orders(since)):
            c["orders"][o["id"]] = o
        c["mark"] = now
        own: dict = {}
        key = lambda o: o["filled_at"] if o.get("filled_at") is not None and pd.notna(o["filled_at"]) else o["submitted_at"]
        for o in sorted((o for o in c["orders"].values() if o["filled_qty"] > 0), key=key):
            q = float(o["filled_qty"]) * (1 if o["side"] == "buy" else -1)
            own[o["symbol"]] = max(own.get(o["symbol"], 0.0) + q, 0.0)
        return {k: int(round(v)) for k, v in own.items() if v > 0}

    def held_positions(self, notes: list) -> tuple:
        """(overnight positions {symbol: {qty, avg_px, mv}}, the intraday symbol's shares, foreign positions): only the shares the book's own orders bought (own_shares,
        never more than the broker holds). Everything else - symbols the book can never trade, and shares in its symbols bought by hand or another tool - is foreign:
        reported, never sold, never part of the book's capital. The positions are read before the order history: a fill in between is never counted as the book's."""
        pos = self.broker.positions()
        own = self.own_shares()
        possible = self.possible_symbols()
        isym = self.m.intraday_symbol
        ours, iq, foreign = {}, 0, {}
        for s, p in pos.items():
            q = int(p["qty"])
            mine = min(q, own.get(s, 0)) if q > 0 and (s in possible or s == isym) else 0
            if mine and s == isym:
                iq = mine
            elif mine:
                ours[s] = {**p, "qty": mine, "mv": float(p.get("mv") or 0.0) * mine / q}
            if q != mine:
                foreign[s] = {**p, "qty": q - mine, "mv": float(p.get("mv") or 0.0) * (q - mine) / q if q else 0.0}
        if foreign:
            notes.append(f"positions that are not this book's are left alone: { {s: p['qty'] for s, p in foreign.items()} }")
        return ours, iq, foreign

    def capital(self, equity: float) -> float:
        """The book's capital from its own equity (cash + its positions; positions outside the book are not part of it): equity x fraction, or the fixed amount - never more
        than the equity (more would be leverage)."""
        if self.live.capital > 0:
            return min(self.live.capital, equity)
        return equity * min(self.live.equity_fraction, 1.0)

    # ── risk overlays ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _risk_state(self, st: dict, equity: float, notes: list, last_equity: Optional[float] = None) -> tuple:
        """``last_equity``: yesterday's closing equity from the same account read (the cached one can be a session old in a loop running for days)."""
        today = str(self.now().date())
        if st.get("day") != today:
            st["day"], st["day_start_equity"] = today, float(last_equity or self.acct.get("last_equity") or equity)
        st["hwm"] = max(float(st.get("hwm") or 0.0), equity)
        halted = stopped = False
        d0 = float(st.get("day_start_equity") or equity)
        if self.live.daily_loss_halt > 0 and d0 > 0 and equity / d0 - 1 <= -self.live.daily_loss_halt:
            halted = True
            notes.append(f"DAILY-LOSS HALT (live overlay): equity {equity / d0 - 1:.2%} today (limit -{self.live.daily_loss_halt:.1%}) - no new entries today")
        if st.get("stopped"):
            stopped = True
            notes.append(f"DRAWDOWN STOP (live overlay) active since {st['stopped']} - the book stays flat; reset the drawdown base to resume")
        elif self.live.dd_stop > 0 and st["hwm"] > 0 and equity / st["hwm"] - 1 <= -self.live.dd_stop:
            stopped = True
            st["stopped"] = str(self.now())
            notes.append(f"DRAWDOWN STOP (live overlay): equity {equity / st['hwm'] - 1:.1%} from the high-water mark (limit -{self.live.dd_stop:.0%}) - selling everything")
        return st, halted, stopped

    # ── guards ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _guard(self) -> None:
        if not self.live.armed:
            raise RuntimeError("DRY RUN: the engine is not armed, nothing is sent")
        if self.load_state().get("killed"):
            raise RuntimeError(f"KILL is active (since {self.load_state()['killed']}): nothing is sent until it is cleared ('Reset drawdown base / clear KILL')")
        if any("-KILL-" in o["cid"] for o in self.broker.orders(self.now().normalize() - pd.Timedelta(days=5), "open")):
            raise RuntimeError("KILL orders are still open (queued for the next open): nothing is sent until they are done")
        probs = self.arm_problems()
        if probs:
            raise RuntimeError("refusing to send: " + "; ".join(probs))
        if not self.flock.acquire():
            raise RuntimeError("another engine of this book holds the lock (GUI or headless runner) - stop it first")
        self.claim_account()

    def _send(self, symbol: str, qty: int, side: str, tif: str, cid: str, deadline: Optional[pd.Timestamp] = None, **meta) -> dict:
        """One order (ARMED callers only, after _guard). ``deadline``: not sent at or after it (status 'late') - a DAY order reaching Alpaca after the close would trade at
        the next open. The trading client does not retry submits itself (no resend seconds later)."""
        why = (("late", f"not sent: past {deadline:%H:%M:%S}") if deadline is not None and self.now() >= deadline else
               ("disarmed", "not sent: the engine was disarmed") if not self.live.armed else
               ("killed", "not sent: KILL is active") if self.load_state().get("killed") else None)   # re-checked per order: a KILL
        if why:                                                                                 # from another process stops a batch half-way
            r = dict(symbol=symbol, cid=cid, status=why[0], side=side, qty=int(qty), tif=tif, note=why[1], **meta)
            self.journal("order", day=self.now().normalize(), **r)
            return r
        try:
            o = self.broker.submit(symbol, int(qty), side, tif, cid)
            r = dict(symbol=symbol, cid=cid, id=o["id"], status=o["status"], side=side, qty=int(qty), tif=tif, **meta)
        except Exception as e:                                                                # noqa: BLE001
            code = getattr(e, "status_code", None)
            if code is not None and 400 <= int(code) < 500 and int(code) not in (408, 429):
                r = dict(symbol=symbol, cid=cid, status="rejected", side=side, qty=int(qty), tif=tif, note=f"{code}: {str(e)[:120]}", **meta)
            else:
                f = self.broker.order_by_cid(cid)
                r = dict(symbol=symbol, cid=cid, id=f["id"] if f else None, status=f["status"] if f else "unknown", side=side, qty=int(qty), tif=tif,
                         note=f"ambiguous submit ({type(e).__name__}: {str(e)[:80]})", **meta)
        self.journal("order", day=self.now().normalize(), **r)
        return r

    def _open_qty(self, day: pd.Timestamp) -> dict:
        """{symbol: {'sell': shares, 'buy': shares}} still to fill on every open order of this book - KILL orders and last sessions' included."""
        out: dict = {}
        for o in self.broker.orders(day - pd.Timedelta(days=5), "open"):
            d = out.setdefault(o["symbol"], {"sell": 0.0, "buy": 0.0})
            d[o["side"]] = d.get(o["side"], 0.0) + float(o["qty"]) - float(o.get("filled_qty") or 0.0)
        return out

    def _open_orders(self, day: pd.Timestamp, legs: tuple) -> list:
        """Our open orders of the given legs (cid suffix letters: o / e / x overnight, i intraday)."""
        return [o for o in self.broker.orders(day - pd.Timedelta(days=3), "open") if o["cid"].split("-")[-1][:1] in legs]

    def _cancel(self, oo: list, wait_s: float = 10.0) -> None:
        for o in oo:
            try:
                self.broker.cancel(o["id"])
            except Exception as e:                                                            # noqa: BLE001
                self.log(f"cancel {o['symbol']} failed: {type(e).__name__}: {str(e)[:80]}", "warn")
        ids = {o["id"] for o in oo}
        t0 = time.monotonic()
        while ids and time.monotonic() - t0 < wait_s:
            still = [o for o in self.broker.orders(self.now().normalize() - pd.Timedelta(days=3), "open") if o["id"] in ids]
            if not still:
                return
            if not self._injected:
                time.sleep(1.0)
            else:
                break
        still = [o for o in self.broker.orders(self.now().normalize() - pd.Timedelta(days=3), "open") if o["id"] in ids]
        if still:
            raise RuntimeError(f"{len(still)} of this engine's orders are still open after cancelling - not trading on top of them")

    # ── morning ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def morning(self, now: Optional[pd.Timestamp] = None) -> list:
        """Top up the caches; sell last night's overnight names in the opening auction (OPG before 09:28, a market order once the market is open). Emulating the auctions,
        the sale waits for the emulated open (the loop's open step)."""
        now = now or self.now()
        day = now.normalize()
        out = [] if self.emulate else self.overnight_exits(now)                                # first: the sale must not wait for the data top-up (OPG cut-off 09:28)
        tm = self.times(day)
        late = tm is not None and now > tm["decision"] - pd.Timedelta(minutes=30)               # started late: the decision reads the newest bars itself
        if self.live.refresh_data and not self._injected and not late:
            try:
                syms = sorted(set(self.universe_symbols(day)) | (set(self.broker.positions()) & self.possible_symbols()))
                update_daily(syms, log=lambda m: None, client=self.broker.dc)
                D.update_bars(syms, "sip", log=lambda m: None, client=self.broker.dc)
                if self.m.intraday_symbol and self.m.intraday_feed == "iex":
                    D.update_bars([self.m.intraday_symbol], "iex", start="2024-01-01", log=lambda m: None, client=self.broker.dc)
            except Exception as e:                                                            # noqa: BLE001
                self.log(f"data top-up failed ({type(e).__name__}: {str(e)[:100]}) - the decisions check the latest bars themselves", "warn")
        if self.m.uses_ml and ml.load_model(self.folder, self.m, day.year) is None:
            self.log(f"no ML model for {day.year}: the overnight leg cannot rank today - Backtest tab > Train ML models", "err")
        return out

    def overnight_exits(self, now: Optional[pd.Timestamp] = None, late_ok: bool = False) -> list:
        """Last night's overnight names (and an intraday position left from an earlier session - the leg is flat every night): OPG sells before 09:28; market sells from the
        market orders' open time until the decision (market DAY orders in place of the auctions; or late / what the opening auction left). ``late_ok``: the decision was
        missed - sold before the close (the backtest would have sold them at the open)."""
        now = now or self.now()
        day = now.normalize()
        out: list = []
        st = self.load_state()
        book = st.get("book") or {}
        pend = self._open_qty(day)                                                              # read BEFORE the positions: a sale filling in between is netted once, not twice
        ours, iq, _ = self.held_positions([])
        held = {s: ours[s]["qty"] for s in ours if str((book.get(s) or {}).get("entry_date", "")) < str(day.date())}
        if iq and self.m.intraday_symbol:
            held[self.m.intraday_symbol] = iq                                                   # before the first check every intraday position is a leftover
        qty = {s: _net(q, pend.get(s)) for s, q in held.items()}                                # minus the book's open orders (an opening auction still running, KILL)
        busy = [s for s, q in qty.items() if q == 0]
        if busy:
            self.log(f"already being sold by open orders: {', '.join(busy)}", "info")
        qty = {s: q for s, q in qty.items() if q}
        due = list(qty)
        if not due:
            return out
        tif = "opg" if now - day < OPG_CUTOFF and not self.emulate else "day"
        tm = self.times(day)
        if tm is None:
            return out
        first = min(tm["open"], tm["emu_open"]) if self.emulate else tm["open"]                 # a market sale may be sent before the open (queued for it)
        missed = late_ok and st.get("decided") != str(day.date()) and tm["cutoff"] < now < tm["close"] - pd.Timedelta(minutes=2)
        if tif == "day" and not (first <= now < tm["decision"] or missed):
            return out
        late = tif == "day" and not missed and now >= max(first, tm["open"]) + pd.Timedelta(minutes=2)
        how =("at the open (OPG)" if tif == "opg" else "now (the decision was missed: last night's book leaves before the close)" if missed else
               "now (late: after the open)" if late else "at the open (market orders" + (" queued for it)" if now < tm["open"] else ")"))
        if not self.live.armed:
            self.log(f"DRY RUN: would sell {len(due)} overnight position(s) {how}: {', '.join(due)}", "info")
            return out
        self._guard()
        st = self.load_state()                                                                  # nothing cancelled: an order still open is netted above, never sold twice
        n = int(st.get("attempt", 0)) + 1
        st["attempt"] = n
        self.save_state(st)
        for s in due:
            q = qty[s]
            out.append(self._send(s, abs(q), "sell" if q > 0 else "buy", tif, f"{PREFIX}-{day:%Y%m%d}-{_cid_sym(s)}-o{n}", leg="overnight exit",
                                  deadline=None if tif == "opg" else tm["close"] - pd.Timedelta(seconds=2)))
        self.log(f"overnight exits sent {how}: " + ", ".join(f"{r['symbol']} {r['status']}" for r in out), "warn" if late or missed else "trade")
        return out

    # ── intraday leg ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def intraday_check(self, slot: int, now: Optional[pd.Timestamp] = None, force: bool = False) -> dict:
        """The noise-area check of ``slot`` (bar ending at slot_end): returns the check row; ARMED and not ``force`` -> the market order is sent."""
        with self._lock:
            now = now or self.now()
            day = now.normalize()
            notes: list = []
            if self.broker is None:
                self.connect()
            mk, t = self.intraday_market(day, now, notes)
            s = self.m.intraday_symbol
            jq = mk.col[s]
            sig = noise_sigma(mk, self.m.intraday_lookback)[t]
            upper = noise_upper(mk.iq["op"][t], mk.C[t - 1, jq], sig, self.m.intraday_mult)
            px, vw = float(mk.iq["cl"][t, slot]), float(mk.iq["vwap"][t, slot])
            ours, iq_qty, _ = self.held_positions(notes)
            st = self.load_state()
            a0 = self.broker.account()
            st, halted, stopped = self._risk_state(st, float(a0["equity"]), notes, float(a0.get("last_equity") or 0) or None)
            if not force:
                self.save_state(st)                                                             # the high-water mark / drawdown stop follow the day, not only the decisions
            act = noise_action(iq_qty > 0, px, float(upper[slot]), vw, self.m.intraday_trail)
            row = dict(day=day, slot=slot, bar_end=slot_end(day, slot), at=now, price=px, band=float(upper[slot]), vwap=vw, held=iq_qty, action=act or "", qty=0,
                       src=str(mk.iq["src"][t]), notes=notes, open=float(mk.iq["op"][t]), prev_close=float(mk.C[t - 1, jq]), status="")
            if act == "buy" and (halted or stopped):
                row.update(action="", status="halted (live overlay)")
                act = None
            if act == "buy":
                acct = self.broker.account()
                prev = {}
                if ours:                                                                          # overnight names whose opening sale failed: the backtest marks them at yesterday's close
                    h = self._hist_for(list(ours), day, notes)
                    prev = h[h.date < day].groupby("symbol").close.last().to_dict() if len(h) else {}
                eq = float(acct["cash"]) + sum(p["qty"] * (prev[x] if np.isfinite(prev.get(x, np.nan)) else (p.get("avg_px") or 0.0)) for x, p in ours.items())
                row["qty"] = intraday_shares(self.m, self.capital(eq), float(acct["cash"]), px)
                row["capital"] = self.capital(eq)
            elif act == "sell":
                row["qty"] = iq_qty
            if act and row["qty"] > 0 and self.live.armed and not force:
                self._guard()
                st = self.load_state()
                st["attempt"] = int(st.get("attempt", 0)) + 1
                self.save_state(st)
                r = self._send(s, row["qty"], act, "day", f"{PREFIX}-{day:%Y%m%d}-{_cid_sym(s)}-i{'B' if act == 'buy' else 'S'}{slot:02d}{st['attempt']}", leg="intraday",
                               slot=slot, ref_px=px)
                row["status"] = r["status"]
                if act == "buy" and r["status"] not in ("rejected", "unknown"):
                    st = self.load_state()
                    st["intraday"] = dict(day=str(day.date()), entry_slot=slot, ref_px=px, qty=row["qty"], cid=r["cid"])
                    self.save_state(st)
            elif act and row["qty"] > 0:
                row["status"] = "dry run" if not self.live.armed else "manual check (not sent)"
            if not force:
                self.journal("check", **row)
            self.last_check = row
            return row

    def intraday_close(self, day: pd.Timestamp, tif: Optional[str] = None) -> Optional[dict]:
        """The intraday position still held after the last check: market-on-close sell (the backtest sells it in the closing auction); emulating the auctions, a market sell
        at the emulated close. None = nothing held (or DRY RUN); else the order row - a status in NOT_SENT means the sale did not go out (the caller then buys nothing)."""
        if not self.m.intraday_symbol:
            return None
        pend = self._open_qty(day)                                                              # before the position: an open sale is netted, never doubled
        _, iq_qty, _ = self.held_positions([])
        iq_qty = _net(iq_qty, pend.get(self.m.intraday_symbol))
        if iq_qty <= 0:
            return None
        tif = tif or self.close_tif
        how = "in the closing auction" if tif == "cls" else "with a market order"
        if not self.live.armed:
            self.log(f"DRY RUN: would sell {iq_qty} {self.m.intraday_symbol} {how}", "info")
            return None
        self._guard()
        tm = self.times(day)
        st = self.load_state()
        st["attempt"] = int(st.get("attempt", 0)) + 1
        self.save_state(st)
        r = self._send(self.m.intraday_symbol, iq_qty, "sell", tif, f"{PREFIX}-{day:%Y%m%d}-{_cid_sym(self.m.intraday_symbol)}-iC{st['attempt']}", leg="intraday close",
                       deadline=(tm["cutoff"] if tif == "cls" else tm["close"] - pd.Timedelta(seconds=2)) if tm else None)
        self.log(f"intraday {self.m.intraday_symbol}: {iq_qty} sold {how} ({r['status']})", "trade" if r["status"] not in NOT_SENT else "err")
        return r

    def _sellable_exits(self, day: pd.Timestamp, rows: list) -> list:
        """The decision's exit sales re-sized when they are sent: the position now minus the book's open orders on it (read first) - a sale that filled late (a halted
        opening sale resuming) is never sold again."""
        if not rows:
            return rows
        pend = self._open_qty(day)
        ours, _, _ = self.held_positions([])
        out = []
        for r in rows:
            q = min(int(r["qty"]), abs(_net(ours.get(r["symbol"], {}).get("qty", 0), pend.get(r["symbol"]))))
            if q > 0:
                out.append({**r, "qty": q})
            else:
                self.log(f"{r['symbol']}: nothing left to sell (already sold or being sold) - its exit is skipped", "info")
        return out

    def _trim_entries(self, entries: list, sells: list, drop_all: bool) -> list:
        """The buys that may still go out after the sells of the same close: none when the intraday position's sale failed (its value would be bought on top - up to 2x
        gross), and without the lowest-ranked buys worth the exits that failed. ``entries`` / ``sells``: dicts with symbol, qty, ref_px (sells: their status)."""
        if drop_all:
            if entries:
                self.log(f"the {self.m.intraday_symbol} sale did not go out: no overnight buys tonight ({len(entries)} dropped) - the book never exceeds its equity", "err")
                self.journal("entries_dropped", reason="intraday sale failed", dropped=[e["symbol"] for e in entries])
            return []
        failed = sum(float(s["qty"]) * float(s.get("ref_px") or 0.0) for s in sells if s.get("status") in NOT_SENT)
        if failed <= 0:
            return entries
        keep, cut, freed = list(entries), [], 0.0
        while keep and freed < failed:
            e = keep.pop()                                                                      # the lowest-ranked buy goes first
            cut.append(e["symbol"])
            freed += float(e["qty"]) * float(e.get("ref_px") or 0.0)
        self.log(f"exit sale(s) worth ${failed:,.0f} did not go out: {len(cut)} buy(s) dropped ({', '.join(cut)}) - the book never exceeds its equity", "err")
        self.journal("entries_dropped", reason="exit sale failed", dropped=cut)
        return keep

    # ── the overnight decision ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def plan(self, now: Optional[pd.Timestamp] = None, force: bool = False) -> MomPlan:
        """The overnight decision of today's session at its decision time. ``force`` = a manual dry run at any time (on the newest bars available - a partial-day snapshot
        before the decision time; the plan says so)."""
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
                if tm is None or now < tm["open"] + pd.Timedelta(minutes=45):
                    day = self.last_session(now - pd.Timedelta(minutes=45))
                    tm = self.times(day)
                now_eff = min(now, tm["decision"]) if day == now.normalize() else tm["close"] + pd.Timedelta(hours=1)
                notes.append(f"manual dry run: decided on the {day:%Y-%m-%d} bars available at {min(now_eff, tm['decision']):%H:%M} ET"
                             + (" - a partial-day snapshot" if now_eff < tm["decision"] else "") + "; orders are only ever sent at the decision time")
            else:
                now_eff = now
            st = self.load_state()
            ours, iq_qty, foreign = self.held_positions(notes)
            mk, t, bars = self.overnight_market(day, now_eff, list(ours), notes)
            exp = tm["snapshot"] - pd.Timedelta(minutes=D.BAR_MIN)
            P = E.prepare(self.m, mk, self.folder, rows=[t])
            U = P["U"][t]
            member = E.member_mask(mk, self.m)[t]
            mem = [s for s, j in mk.col.items() if member[j]]
            have = [s for s in mem if s in bars and len(bars[s]) and (bars[s].index == exp).any()]
            cover = len(have) / max(len(mem), 1)
            if real and cover < 0.8:
                raise RuntimeError(f"only {cover:.0%} of the universe has the {exp:%H:%M} bar - stale feed, no decision")
            if cover < 0.95:
                notes.append(f"{cover:.0%} of the universe has the {exp:%H:%M} bar; the others use their last earlier bar (as the backtest does when a bar is missing)")
            acct = self.broker.account()
            self.acct.update(acct)
            equity = float(acct["equity"])
            S = mk.S[t]

            def mark(s, q, avg):
                j = mk.col.get(s)
                if j is not None:
                    for x in (S[j], mk.C[t - 1, j] if t > 0 else np.nan):
                        if np.isfinite(x):
                            return q * x
                return q * (avg if np.isfinite(avg) else 0.0)
            eq_snap = float(acct["cash"]) + sum(mark(s, p["qty"], p.get("avg_px", np.nan)) for s, p in ours.items())    # the book's own equity at the decision prices
            if iq_qty:
                eq_snap += mark(self.m.intraday_symbol, iq_qty, np.nan)
            if foreign:
                notes.append(f"positions outside the book (${sum(float(p.get('mv') or 0.0) for p in foreign.values()):,.0f}) are not part of its capital")
            cap = self.capital(eq_snap)
            if self.live.capital > eq_snap:
                notes.append(f"fixed capital ${self.live.capital:,.0f} is more than the book's equity ${eq_snap:,.0f}: capped at the equity (no leverage)")
            st, halted, stopped = self._risk_state(st, equity, notes, float(acct.get("last_equity") or 0) or None)
            if real:
                self.save_state(st)
            pred = P["pred"][t] if P["pred"] is not None else None
            mlinfo = (P.get("ml_used") or {}).get(day.year)
            dec = decide_overnight(self.m, mk.syms, U, bool(P["trend"][t]), P["d45"][t], pred, S, {s: p["qty"] for s, p in ours.items()}, cap,
                                   sectors=E.sector_list(mk.syms) if self.m.sector_cap > 0 else None)
            notes += dec.notes
            exits, entries = list(dec.exits), list(dec.entries)
            if stopped:
                exits = [(s, "drawdown stop (live overlay)") for s in ours]
                entries = []
            elif halted:
                entries = []
            rows = []
            n = int(st.get("attempt", 0)) + 1
            for s, why in exits:
                q = ours[s]["qty"]
                j = mk.col.get(s)
                rows.append(dict(symbol=s, side="sell" if q > 0 else "buy", qty=abs(q), tif=self.close_tif, ref_px=float(S[j]) if j is not None else np.nan, why=why, ok=True, note="",
                                 cid=f"{PREFIX}-{day:%Y%m%d}-{_cid_sym(s)}-x{n}", kind="exit", score=np.nan, rank=0, move=np.nan, forecast=np.nan))
            for s, sh, sc, rk, mv, fc in entries:
                rows.append(dict(symbol=s, side="buy", qty=sh, tif=self.close_tif, ref_px=float(S[mk.col[s]]), why=f"{self.m.ranking} score {sc:.4f} - rank {rk}", ok=True, note="",
                                 cid=f"{PREFIX}-{day:%Y%m%d}-{_cid_sym(s)}-e{n}", kind="entry", score=sc, rank=rk, move=mv, forecast=fc))
            cols = ["symbol", "side", "qty", "tif", "ref_px", "notional", "why", "ok", "note", "cid", "kind", "score", "rank", "move", "forecast"]
            orders = pd.DataFrame(rows, columns=[c for c in cols if c != "notional"])
            orders["notional"] = orders.qty * orders.ref_px
            orders = orders[cols]
            book = st.get("book") or {}
            prow = []
            for s, p in ours.items():
                j = mk.col.get(s)
                lx = float(S[j]) if j is not None and np.isfinite(S[j]) else np.nan
                b = book.get(s) or {}
                ep = float(b.get("entry_px") or p.get("avg_px") or np.nan)
                prow.append(dict(symbol=s, shares=p["qty"], entry_date=b.get("entry_date", ""), entry_px=ep, last=lx, value=p["qty"] * lx, pnl=p["qty"] * (lx - ep),
                                 exit=next((w for x, w in exits if x == s), "kept (picked again)" if s in dec.keeps else "")))
            if iq_qty:
                j = mk.col.get(self.m.intraday_symbol)
                lx = float(S[j]) if j is not None else np.nan
                prow.append(dict(symbol=self.m.intraday_symbol, shares=iq_qty, entry_date=str(day.date()), entry_px=np.nan, last=lx, value=iq_qty * lx, pnl=np.nan,
                                 exit="intraday: market order at the close" if self.emulate else "intraday: closing auction"))
            cand = pd.DataFrame([dict(symbol=s, score=sc, rank=r, move=mv, forecast=fc, taken=tk, last=float(S[mk.col[s]])) for s, sc, r, mv, fc, tk in dec.candidates],
                                columns=["symbol", "score", "rank", "move", "forecast", "taken", "last"])
            shadow = None
            if self.shadow is not None:
                try:
                    shadow = self._shadow_picks(mk, t, P, cap)
                except Exception as e:                                                        # noqa: BLE001
                    notes.append(f"shadow model failed: {type(e).__name__}: {str(e)[:100]}")
            plan = MomPlan(day=day, decided_at=now, snapshot_to=tm["snapshot"], capital=cap, equity=equity,
                           positions=pd.DataFrame(prow, columns=["symbol", "shares", "entry_date", "entry_px", "last", "value", "pnl", "exit"]),
                           candidates=cand, orders=orders, notes=notes, halted=halted, stopped=stopped, real=real, trend=bool(P["trend"][t]), ml=mlinfo, shadow=shadow)
            self.last_plan = plan
            return plan

    def _shadow_picks(self, mk, t: int, P: dict, cap: float) -> pd.DataFrame:
        """The shadow model's overnight picks on the same data (journaled, never traded)."""
        sm = self.shadow
        pred = P["pred"][t] if P["pred"] is not None else None
        if sm.uses_ml and pred is None:
            pred = E.prepare(sm, mk, self.folder, rows=[t])["pred"][t]
        dec = decide_overnight(sm, mk.syms, P["U"][t], bool(P["trend"][t]), P["d45"][t], pred if sm.uses_ml else None, mk.S[t], {}, cap,
                               sectors=E.sector_list(mk.syms) if sm.sector_cap > 0 else None)
        return pd.DataFrame([dict(symbol=s, qty=sh, score=sc, rank=r, move=mv) for s, sh, sc, r, mv, _ in dec.entries], columns=["symbol", "qty", "score", "rank", "move"])

    def submit(self, plan: MomPlan, drop_entries: bool = False) -> list:
        """Send the plan's market-on-close orders (ARMED, at the decision time only): the exits, then the entries that may still go (_trim_entries; ``drop_entries`` = the
        intraday position's closing sale failed). Emulating the auctions they are queued (status 'queued') and close_orders() sends them as market orders at the close."""
        self._guard()
        if not plan.real:
            raise RuntimeError("a manual dry run is never sent")
        with self._lock:
            tm = self.times(plan.day)
            if self.now() > tm["cutoff"]:
                raise RuntimeError(f"past the market-on-close cut-off ({tm['cutoff']:%H:%M:%S} ET) - nothing sent")
            st = self.load_state()                                                              # nothing cancelled: the exits are netted against open orders when sent
            st["attempt"] = int(st.get("attempt", 0)) + 1
            todo = plan.orders[plan.orders.ok]
            st["pending"] = {"day": str(plan.day.date()), "sent": not self.emulate, "send_at": str(tm["emu_close"]) if self.emulate else None,
                             "orders": todo[["symbol", "side", "qty", "tif", "ref_px", "cid", "kind", "score", "rank", "move", "forecast"]].to_dict("records")}
            self.save_state(st)
            self.journal("decision", **self._decision_row(plan), n_orders=len(todo))
            if self.emulate:
                return [dict(symbol=r.symbol, cid=r.cid, status="queued", side=r.side, qty=int(r.qty), tif=r.tif, leg=r.kind) for r in todo.itertuples()]
            rows = todo.to_dict("records")
            send = lambda r: self._send(r["symbol"], int(r["qty"]), r["side"], r["tif"], r["cid"], leg=r["kind"], ref_px=r["ref_px"], deadline=tm["cutoff"])
            res = [send(r) for r in self._sellable_exits(plan.day, [r for r in rows if r["kind"] == "exit"])]
            res += [send(r) for r in self._trim_entries([r for r in rows if r["kind"] == "entry"], res, drop_entries)]
            st = self.load_state()
            st["last_submit"] = str(self.now())
            self.save_state(st)
            return res

    def close_orders(self, day: pd.Timestamp) -> list:
        """Emulated closing auction: the intraday position's sale, then the decision's queued orders as market orders - exits (sells) before entries (buys), so the book
        never holds more than its equity. Once per session (the loop's close step); nothing after the close, nothing after a KILL."""
        with self._lock:
            tm = self.times(day)
            if tm is None or not tm["emu_close"] <= self.now() < tm["close"]:
                raise RuntimeError("not the emulated close")
            out = []
            qfail = False
            try:
                r = self.intraday_close(day, tif="day")                                         # a market order (MOC is closed by now)
                if r:
                    out.append(r)
                    qfail = r["status"] in NOT_SENT
            except Exception as e:                                                            # noqa: BLE001
                qfail = True                                                                    # unknown whether the intraday position was sold: buy nothing on top of it
                self.log(f"intraday sale at the close failed: {type(e).__name__}: {str(e)[:150]}", "err")
            st = self.load_state()
            pend = st.get("pending") or {}
            if pend.get("day") != str(day.date()) or pend.get("sent", True) or not pend.get("orders"):  # no 'sent' flag = sent by an engine sending MOC orders
                return out
            if pend.get("killed") or st.get("killed"):
                self.log("queued closing orders dropped: KILL is active", "warn")
                return out
            sells = self._sellable_exits(day, [o for o in pend["orders"] if o["side"] == "sell"])   # re-sized now: a late-filling sale is never doubled
            buys = [o for o in pend["orders"] if o["side"] == "buy"]
            if not self.live.armed:
                self.log(f"DRY RUN: would send {len(sells) + len(buys)} queued market order(s) at the close", "info")
                return out
            self._guard()
            pend["sent"] = True                                                                 # one attempt (no double orders)
            st["pending"] = pend
            self.save_state(st)
            deadline = tm["close"] - pd.Timedelta(seconds=2)                                    # a day order reaching Alpaca after the close would trade at the NEXT open
            send = lambda o: self._send(o["symbol"], int(o["qty"]), o["side"], "day", o["cid"], leg=o["kind"], ref_px=o.get("ref_px"), deadline=deadline)
            workers = 1 if self._injected else 8                                                # seconds before the close: in parallel, the sells before the buys
            with ThreadPoolExecutor(max_workers=workers) as ex:
                rs = list(ex.map(send, sells))
                out += rs + list(ex.map(send, self._trim_entries(buys, rs, qfail)))
            late = [r for r in out if r["status"] == "late"]
            if late:
                self.log(f"the close came before every closing order went out: {len(late)} not sent ({', '.join(x['symbol'] for x in late)})", "err")
                self.journal("close_orders_cut", day=day, not_sent=[x["cid"] for x in late])
            st = self.load_state()
            st["last_submit"] = str(self.now())
            self.save_state(st)
            bad = [r for r in out if r["status"] in NOT_SENT]
            self.log(f"{self.now():%H:%M:%S} closing orders sent as market orders: {len(out) - len(bad)} / {len(out)}"
                     + (f", {len(bad)} rejected / unknown / late" if bad else ""), "warn" if bad else "trade")
            return out

    def _decision_row(self, plan: MomPlan) -> dict:
        """Everything a check against the backtest data needs: the positions held, the capital, the candidates and the orders."""
        return dict(day=plan.day, snapshot_to=plan.snapshot_to, capital=plan.capital, equity=plan.equity, notes=plan.notes, halted=plan.halted, stopped=plan.stopped,
                    trend=plan.trend, model_spec=self.m.to_dict(), feed=self.live.feed, ml=plan.ml,
                    held={r.symbol: dict(shares=int(r.shares), entry_date=str(r.entry_date)) for r in plan.positions.itertuples() if r.symbol != self.m.intraday_symbol},
                    candidates=[dict(symbol=r.symbol, score=float(r.score), rank=int(r.rank), taken=bool(r.taken)) for r in plan.candidates.head(40).itertuples()],
                    orders=[dict(symbol=r.symbol, side=r.side, qty=int(r.qty), kind=r.kind, why=r.why, ok=bool(r.ok), ref_px=float(r.ref_px)) for r in plan.orders.itertuples()],
                    shadow=(plan.shadow.to_dict("records") if plan.shadow is not None else None), shadow_spec=(self.shadow.to_dict() if self.shadow is not None else None))

    # ── after the close ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def reconcile(self, day: Optional[pd.Timestamp] = None) -> dict:
        """After the close: journal the day's fills and the finished trades of both legs, rebuild the overnight book from the broker's positions."""
        with self._lock:
            day = (day or self.now()).normalize()
            st = self.load_state()
            book = dict(st.get("book") or {})
            pend = {o["cid"]: o for o in ((st.get("pending") or {}).get("orders") or [])}
            if self.live.armed:                                                                 # a day order of the session still open would trade at the next open
                try:
                    left = [o for o in self.broker.orders(day - pd.Timedelta(days=1), "open") if o["cid"].startswith(f"{PREFIX}-{day:%Y%m%d}")]
                    if left:
                        self._cancel(left)
                        self.log(f"cancelled {len(left)} order(s) of {day:%Y-%m-%d} still open after its close: " + ", ".join(o["symbol"] for o in left), "warn")
                        self.journal("leftover_cancelled", day=day, cids=[o["cid"] for o in left])
                except Exception as e:                                                        # noqa: BLE001
                    self.log(f"could not cancel the session's leftover orders: {type(e).__name__}: {str(e)[:120]}", "err")
            orders = [o for o in self.broker.orders(day - pd.Timedelta(days=1)) if o["cid"].startswith(f"{PREFIX}-{day:%Y%m%d}")]
            for o in orders:
                if o["status"] in FINAL or o["filled_qty"] > 0:
                    self.journal("fill", day=day, symbol=o["symbol"], cid=o["cid"], status=o["status"], side=o["side"], qty=o["qty"], filled_qty=o["filled_qty"],
                                 avg_px=o["filled_avg_price"], tif=o["tif"], submitted_at=o["submitted_at"], filled_at=o["filled_at"])
            leg = lambda o: o["cid"].split("-")[-1][:1]
            filled = [o for o in orders if o["filled_qty"] > 0]
            ours, _, _ = self.held_positions([])
            for s in list(book):
                ex = [o for o in filled if o["symbol"] == s and leg(o) in ("x", "o")]
                if s not in ours or ex:
                    b = book.pop(s)
                    qx = sum(float(o["filled_qty"]) for o in ex)                                 # the shares actually sold (an OPG part and a market retry: their VWAP)
                    px = sum(float(o["filled_qty"]) * float(o["filled_avg_price"]) for o in ex) / qx if qx else np.nan
                    q = qx if ex else float(b.get("shares") or 0)
                    self.journal("trade_closed", leg="OVERNIGHT", symbol=s, trade_id=b.get("trade_id"), entry_date=b.get("entry_date"), entry_px=b.get("entry_px"),
                                 exit_date=str(day.date()), exit_px=px, exit_how=_how(ex[-1]) if ex else "?",
                                 shares=q, pnl=q * (px - float(b.get("entry_px") or np.nan)) if np.isfinite(px) else None, score=b.get("score"), rank=b.get("rank"))
            for s, p in ours.items():
                if s in book:
                    book[s]["shares"] = p["qty"]
                    continue
                buys = [o for o in filled if o["symbol"] == s and leg(o) == "e"]
                meta = next((pend[o["cid"]] for o in buys if o["cid"] in pend), {})
                tid = int(st.get("next_trade_id") or 1)
                st["next_trade_id"] = tid + 1
                if buys:
                    book[s] = dict(entry_date=str(day.date()), entry_px=float(buys[-1]["filled_avg_price"]), shares=p["qty"], trade_id=tid, score=meta.get("score"),
                                   rank=meta.get("rank"), move=meta.get("move"), forecast=meta.get("forecast"))
                else:
                    book[s] = dict(entry_date=str(day.date()), entry_px=float(p.get("avg_px") or np.nan), shares=p["qty"], trade_id=tid)
                self.journal("trade_opened", leg="OVERNIGHT", symbol=s, **book[s])
            # intraday round trips of the day (buys and sells of the intraday symbol, in time order)
            isym = self.m.intraday_symbol
            if isym:
                fl = sorted([o for o in filled if o["symbol"] == isym and leg(o) == "i"], key=lambda o: (o["filled_at"] or o["submitted_at"]))
                openq, entry = 0, None
                for o in fl:
                    if o["side"] == "buy":
                        entry, openq = o, openq + int(o["filled_qty"])
                    elif entry is not None:
                        tid = int(st.get("next_trade_id") or 1)
                        st["next_trade_id"] = tid + 1
                        q = int(o["filled_qty"])
                        self.journal("trade_closed", leg="INTRADAY", symbol=isym, trade_id=tid, entry_date=str(day.date()), entry_time=str(entry["filled_at"]),
                                     entry_px=entry["filled_avg_price"], exit_date=str(day.date()), exit_time=str(o["filled_at"]), exit_px=o["filled_avg_price"],
                                     exit_how=_how(o), shares=q, pnl=q * (o["filled_avg_price"] - entry["filled_avg_price"]))
                        openq -= q
                        entry = entry if openq > 0 else None
                st["intraday"] = None
            st["book"] = book
            st["pending"] = None
            st["reconciled"] = str(day.date())
            self.save_state(st)
            return book

    # ── the loop ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def tick(self, now: Optional[pd.Timestamp] = None, on_plan: Callable[[MomPlan], None] = lambda p: None, on_check: Callable[[dict], None] = lambda r: None) -> Optional[str]:
        """One pass of the daily cycle at clock time ``now``; returns what it did (or None)."""
        now = now or self.now()
        day = now.normalize()
        tm = self.times(day)
        steps = lambda s: s if self.live.armed else s.setdefault("dry_steps", {})               # a DRY loop never marks an armed loop's steps done
        st = self.load_state()
        dd = steps(st).get("decided")
        if dd and steps(st).get("reconciled") != dd and (dd < str(day.date()) or (tm is not None and now >= tm["close"] + pd.Timedelta(minutes=5))):
            if not self.live.armed:                                                           # a DRY loop leaves the book and the journal to the armed one
                steps(st)["reconciled"] = dd
                self.save_state(st)
                return "reconciled"
            try:
                self.reconcile(pd.Timestamp(dd))
            except Exception as e:                                                            # noqa: BLE001
                st = self.load_state()
                steps(st)["reconcile_failures"] = int(steps(st).get("reconcile_failures") or 0) + 1
                if steps(st)["reconcile_failures"] >= 3:                                               # never block the next session on it
                    steps(st)["reconciled"], steps(st)["reconcile_failures"] = dd, 0
                    self.log(f"the after-close step of {dd} failed 3 times ({type(e).__name__}: {str(e)[:120]}) - skipped; the next one rebuilds the book from the "
                             f"broker's positions", "err")
                self.save_state(st)
                self.journal("error", msg=f"reconcile {dd}: {type(e).__name__}: {str(e)[:300]}")
                return "reconcile failed"
            st = self.load_state()
            steps(st)["reconcile_failures"] = 0
            self.save_state(st)
            self.log(f"after the {dd} close: book rebuilt ({len(st.get('book') or {})} overnight position(s))", "ok")
            return "reconciled"
        if tm is None:
            return None
        if steps(st).get("prepared") != str(day.date()) and now >= day + pd.Timedelta(hours=8, minutes=45) and now < tm["decision"]:
            steps(st)["prepared"] = str(day.date())                                                  # one attempt a day (the OPG sales must not repeat every second)
            self.save_state(st)
            try:
                self.morning(now)
            except Exception as e:                                                            # noqa: BLE001
                self.log(f"morning step failed: {type(e).__name__}: {str(e)[:150]} - the open step retries the sales", "err")
                self.journal("error", msg=f"morning: {type(e).__name__}: {str(e)[:300]}")
            return "morning"
        open_at = tm["emu_open"] if self.emulate else tm["open"] + pd.Timedelta(seconds=60)    # auction orders: a market sale of what the opening auction left
        if steps(st).get("opened") != str(day.date()) and open_at <= now < tm["close"] - pd.Timedelta(minutes=2):
            steps(st)["opened"] = str(day.date())                                                    # one attempt a day (no double orders)
            self.save_state(st)
            self.overnight_exits(now, late_ok=True)
            return "open"
        done = set((steps(st).get("checks") or {}).get(str(day.date()), []))
        for slot, end, due in tm["checks"]:
            if slot in done or now < due or now >= tm["close"]:                                 # after the close a check is history, not a missed action
                continue
            st = self.load_state()
            ck = steps(st).get("checks") or {}
            ck = {str(day.date()): sorted(set(ck.get(str(day.date()), [])) | {slot})}                # one attempt per check (no double orders)
            steps(st)["checks"] = ck
            self.save_state(st)
            if (now - due).total_seconds() > self.live.check_late_s:
                self.log(f"intraday check {end:%H:%M} missed ({now:%H:%M:%S}) - skipped; the backtest would have checked", "err")
                self.journal("check_missed", day=day, slot=slot, at=now)
                return "check missed"
            row = self.intraday_check(slot, now)
            if row["action"]:
                self.log(f"{end:%H:%M} {self.m.intraday_symbol} {row['price']:.2f} vs band {row['band']:.2f} / VWAP {row['vwap']:.2f}: {row['action'].upper()} {row['qty']} "
                         f"({row['status']})", "trade" if row["status"] not in ("dry run", "rejected", "unknown") else "info")
            on_check(row)
            return "check"
        if steps(st).get("prefetched") != str(day.date()) and tm["prefetch"] <= now < tm["decision"] and self.m.overnight:
            steps(st)["prefetched"] = str(day.date())
            self.save_state(st)
            try:
                n = self.prefetch(now)
                self.log(f"prefetched {n:,} SIP bars of today", "info")
            except Exception as e:                                                            # noqa: BLE001
                self.log(f"prefetch failed ({type(e).__name__}) - the decision reads everything", "warn")
            return "prefetch"
        if steps(st).get("decided") != str(day.date()) and tm["decision"] <= now <= tm["cutoff"]:
            steps(st)["decided"] = str(day.date())                                                   # one decision a day, even if it fails half-way (no double orders)
            self.save_state(st)
            qfail = False
            if not self.emulate:                                                              # emulating the auction: sold at the emulated close
                try:
                    r = self.intraday_close(day)
                    qfail = bool(r) and r["status"] in NOT_SENT
                except Exception as e:                                                        # noqa: BLE001
                    qfail = True                                                                # unknown whether it was sold: buy nothing on top of it
                    self.log(f"intraday closing-auction sell failed: {type(e).__name__}: {str(e)[:150]}", "err")
            if not self.m.overnight:
                self.journal("plan_no_orders", day=day, note="overnight leg off")
                return "decision"
            plan = self.plan(now)
            for n in plan.notes:
                self.log(n, "warn")
            o = plan.orders
            self.log(f"{now:%H:%M:%S} decision on the {tm['snapshot']:%H:%M} snapshot: {int((o.kind == 'exit').sum())} exit(s), {int((o.kind == 'entry').sum())} entr(ies), "
                     f"capital ${plan.capital:,.0f}" + ("" if plan.trend else " - SPY below its 200-day mean: no overnight book"), "ok")
            if self.live.armed and len(o) and o.ok.any():
                res = self.submit(plan, drop_entries=qfail)
                bad = [r for r in res if r["status"] in NOT_SENT]
                if self.emulate:
                    self.log(f"{len(res)} closing order(s) queued: market orders at {tm['emu_close']:%H:%M:%S} (the paper account does not run the closing auction)", "trade")
                else:
                    self.log(f"sent {len(res) - len(bad)} / {len(res)} market-on-close orders" + (f", {len(bad)} rejected / unknown" if bad else ""), "warn" if bad else "trade")
            else:
                self.journal("plan_dry" if not self.live.armed else "plan_no_orders", **self._decision_row(plan), n_orders=len(o))
            on_plan(plan)
            return "decision"
        pend = st.get("pending") or {}
        queued = pend.get("day") == str(day.date()) and pend.get("sent") is False                # queued by a market-order decision (the setting changed since)
        if (self.emulate or queued) and steps(st).get("closed") != str(day.date()) and tm["emu_close"] <= now < tm["close"]:
            steps(st)["closed"] = str(day.date())                                                    # one attempt a day (no double orders)
            self.save_state(st)
            try:
                self.close_orders(day)
            except Exception as e:                                                            # noqa: BLE001
                self.log(f"closing orders failed: {type(e).__name__}: {str(e)[:150]}", "err")
                self.journal("error", msg=f"closing orders: {type(e).__name__}: {str(e)[:300]}")
            return "close orders"
        if steps(st).get("decided") != str(day.date()) and tm["cutoff"] < now < tm["close"] and steps(st).get("missed") != str(day.date()):
            steps(st)["missed"] = str(day.date())
            self.save_state(st)
            self.log(f"today's decision window ({tm['decision']:%H:%M:%S} - {tm['cutoff']:%H:%M:%S}) was missed - no overnight book tonight", "err")
            self.journal("missed", day=day)
            if not self.emulate:                                                              # auction orders: the intraday position would stay overnight - a market sale
                try:                                                                            # (market-order execution sells it at the close step)
                    self.intraday_close(day, tif="day")
                except Exception as e:                                                        # noqa: BLE001
                    self.log(f"intraday sale after the missed decision failed: {type(e).__name__}: {str(e)[:150]}", "err")
            return "missed"
        return None

    def run(self, stop: threading.Event, on_plan: Callable[[MomPlan], None] = lambda p: None, on_check: Callable[[dict], None] = lambda r: None) -> None:
        if self.broker is None:
            self.connect()
        if not self.flock.acquire():                                                           # armed or not: a loop marks the day's steps done in the shared state file
            raise RuntimeError("another loop of this book is running on this account (GUI or headless runner, armed or dry) - stop it first")
        st = self.load_state()
        if st.get("model") and st["model"] != self.m.to_dict() and st.get("book"):
            self.log("the model differs from the one that opened the current positions - they leave at the next open as usual", "warn")
        st["model"] = self.m.to_dict()
        self.save_state(st)
        self.log(f"loop started ({'ARMED' if self.live.armed else 'DRY RUN'}, {self.mode}, model '{self.m.name}', overnight bars {self.live.feed}, intraday "
                 f"{self.m.intraday_symbol or '-'} on {self.m.intraday_feed}, "
                 + (f"market DAY orders in place of the auctions: {self.live.mkt_close_s} s before the close and {_open_offset(self.live.mkt_open_s)}" if self.emulate
                    else "MOC / MOO orders" if self.m.uses_auctions else "no auction orders") + ")" + (f", shadow '{self.shadow.name}'" if self.shadow else ""), "ok")
        try:
            while not stop.is_set():
                try:
                    self.tick(on_plan=on_plan, on_check=on_check)
                except Exception as e:                                                        # noqa: BLE001
                    self.log(f"step error: {type(e).__name__}: {str(e)[:200]}", "err")
                    self.journal("error", msg=f"{type(e).__name__}: {str(e)[:300]}")
                stop.wait(1.0 - time.time() % 1.0 + 0.002)                                     # on the whole second: a step due at 15:59:55 starts then, not up to 1 s late
        finally:
            self.flock.release()
            self.log("loop stopped", "info")

    # ── manual actions ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def kill(self) -> list:
        """KILL: disarm, block every order path for the rest of the day, cancel this book's open orders, then close every position in its symbols with market (DAY) orders
        - at once when the market is open, at the next open when it is closed (OPG would be refused 09:28-09:30 and 16:00-19:00). Works in DRY RUN too; never on an
        account another strategy claimed, with borrowed keys, or on a real-money account while real money is disabled. A symbol whose open order could not be cancelled
        is not closed (both could fill: a short)."""
        if not self.live.paper and os.environ.get("RESARB_ALLOW_LIVE") != "1":
            raise RuntimeError("real money is disabled in code: KILL is refused on the live account - close its positions in Alpaca")
        if self.broker is None:
            self.connect()
        owner = self.account_owner()
        if owner and owner != KEY:
            raise RuntimeError(f"the account is claimed by {owner}: KILL would close that strategy's positions in the same names - refused")
        if self.key_source() != "own" and not self.live.shared_keys_ok:
            raise RuntimeError("not the project's own keys: KILL is refused (it could act on another account)")
        self.live.armed = False                                                                 # the loop sends nothing more
        with self._lock:
            out = []
            now = self.now()
            st = self.load_state()
            st["killed"] = str(now)                                                             # every order path refuses until a person clears it (_guard)
            pend = st.get("pending") or {}
            if pend and not pend.get("sent", True):                                             # queued closing orders (market-order execution) are dropped too
                pend["killed"] = True
                st["pending"] = pend
                out.append(dict(symbol="*", status=f"{len(pend.get('orders') or [])} queued closing order(s) dropped"))
            self.save_state(st)
            try:
                oo = self.broker.orders(now.normalize() - pd.Timedelta(days=3), "open")
                if oo:
                    self._cancel(oo)
            except Exception as e:                                                            # noqa: BLE001
                out.append(dict(symbol="*", status=f"cancel: {e}"))
            try:
                still = {o["symbol"] for o in self.broker.orders(now.normalize() - pd.Timedelta(days=3), "open")}
            except Exception as e:                                                            # noqa: BLE001
                out.append(dict(symbol="*", status=f"open orders unreadable ({type(e).__name__}) - nothing closed"))
                self.journal("kill", actions=out)
                return out
            try:
                ours, iq, foreign = self.held_positions([])
            except Exception as e:                                                            # noqa: BLE001
                out.append(dict(symbol="*", status=f"positions / order history unreadable ({type(e).__name__}) - nothing closed"))
                self.journal("kill", actions=out)
                return out
            for s, p in foreign.items():
                out.append(dict(symbol=s, qty=p["qty"], status="not this book's shares - left alone"))
            mine = {**{s: p["qty"] for s, p in ours.items()}, **({self.m.intraday_symbol: iq} if iq else {})}
            for s, q in mine.items():
                if s in still:
                    out.append(dict(symbol=s, qty=q, status="an order of it is still open - NOT closed (both could fill); cancel it in Alpaca and KILL again"))
                    continue
                cid = f"{PREFIX}-KILL-{_cid_sym(s)}-{int(time.time()) % 1000000}"
                try:
                    self.broker.submit(s, abs(q), "sell" if q > 0 else "buy", "day", cid)
                    out.append(dict(symbol=s, qty=q, status="close sent (market)"))
                except Exception as e:                                                        # noqa: BLE001
                    out.append(dict(symbol=s, qty=q, status=f"FAILED {type(e).__name__}: {str(e)[:100]}"))
            self.journal("kill", actions=out)
            return out

    def reset_drawdown(self) -> float:
        st = self.load_state()
        eq = float(self.broker.account()["equity"])
        st["hwm"], st["stopped"], st["killed"] = eq, None, None                                 # a new drawdown base, no drawdown stop, no KILL
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
            if self.m.intraday_symbol:
                ok = a["equity"] >= 25_000 and a["multiplier"] >= 2
                add("day trading", "ok" if ok else ("warn" if not self.live.armed else "fail"),
                    f"margin {a['multiplier']:.0f}x, equity ${a['equity']:,.0f}: the intraday leg's round trips are day trades - the pattern-day-trader rule needs a margin account "
                    f"with at least $25,000" + ("" if ok else " - NOT met"))
            add("margin", "ok" if a["multiplier"] >= 2 or self.emulate else "warn",
                "the closing buys can be placed while the same close's sales are pending (the book itself never exceeds 1x)" if a["multiplier"] >= 2 else
                ("cash account (1x): the emulated close sends the sells before the buys" if self.emulate else
                 "cash account (1x): buys placed while the same auction's sells are pending may be refused for buying power"))
            add("market", "ok", f"{'open' if a['market_open'] else 'closed'} · next open {a['next_open']:%a %H:%M} · next close {a['next_close']:%a %H:%M} ET")
        except Exception as e:                                                                # noqa: BLE001
            add("broker", "fail", f"cannot reach Alpaca: {type(e).__name__}: {str(e)[:140]}")
            return out
        probs = self.model_problems()
        add("model", "ok" if not probs else "fail", f"'{self.m.name}': {self.m.ranking}, k={self.m.k}, decision {self.m.minute // 100}:{self.m.minute % 100:02d}, intraday "
                                                    f"{self.m.intraday_symbol or '-'} on {self.m.intraday_feed}" + (" - " + "; ".join(probs) if probs else ""))
        now = self.now()
        if self.m.uses_ml:
            mo = ml.load_model(self.folder, self.m, now.year)
            add("ML model", "ok" if mo else "fail", f"{now.year}: fitted on {mo['info']['rows']:,} rows {mo['info']['first']} .. {mo['info']['last']} ({mo['info']['fitted']})"
                if mo else f"no model for {now.year} - Backtest tab > Train ML models")
        tm = self.times(now.normalize())
        if tm:
            ck = f"intraday checks {tm['checks'][0][1]:%H:%M} .. {tm['checks'][-1][1]:%H:%M} every {15 * self.m.intraday_every} min · " if tm["checks"] else ""
            add("schedule", "ok", ck + f"overnight snapshot {tm['snapshot']:%H:%M}, decision {tm['decision']:%H:%M:%S}, " + (
                f"closing market orders {tm['emu_close']:%H:%M:%S}, opening sells {tm['emu_open']:%H:%M:%S}" if self.emulate else f"MOC cut-off {tm['cutoff']:%H:%M:%S}")
                + f", close {tm['close']:%H:%M} ET")
        else:
            add("schedule", "ok", "no session today")
        if not self.m.uses_auctions:
            add("auction orders", "ok", "none: the model sends no market-on-close / market-on-open order")
        elif self.emulate:
            add("auction orders", "ok", f"market DAY orders in place of the model's MOC / MOO orders: {self.live.mkt_close_s} s before the close (the intraday sale, then the "
                f"exits, then the buys, in parallel) and {_open_offset(self.live.mkt_open_s)}. Same model, decisions and sizes; the fills are compared with the "
                f"backtest's auction prices (Positions + trade history: diff; Check live vs backtest: cost in bp)")
        else:
            add("auction orders", "warn", "market-on-close / market-on-open orders, as the model" + (
                " - the PAPER account does not run the auctions: it fills them at the quote, partly at random, and expires the rest (2026-10-28: 6 of 20 filled); "
                "choose market orders for paper" if self.live.paper else " - check with Alpaca that the account's routing sends them to the auctions"))
        off = (self.acct or {}).get("clock_offset_s")
        if off is not None and self.m.uses_auctions:
            add("PC clock", "ok" if abs(off) <= 1.0 else "warn", f"{off:+.1f} s against Alpaca's clock" + (
                "" if abs(off) <= 1.0 else f" - the orders timed to the second ({self.live.mkt_close_s} s before the close) would go out {abs(off):.1f} s "
                f"{'early' if off > 0 else 'late'}: sync the Windows clock (Settings > Time > Sync now)"))
        d = load_daily(["SPY"])
        if len(d):
            prev = self.last_session(now - pd.Timedelta(days=1) if now.hour < 17 else now)
            age = (prev - d.date.max()).days
            add("daily history", "ok" if age <= 0 else "warn", f"cache through {d.date.max():%Y-%m-%d}" + ("" if age <= 0 else " (the morning step tops it up)"))
        else:
            add("daily history", "fail", "no daily bars - Update data")
        b = D.load_bars("SPY", "sip")
        if len(b):
            add("15-minute bars", "ok" if (now.normalize() - b.index.max().normalize()).days <= 4 else "warn", f"SIP cache through {b.index.max():%Y-%m-%d %H:%M} (the morning step tops it up)")
        u = D.load_universe()
        mon = now.to_period("M").to_timestamp()
        add("universe", "ok" if len(u) and u.month.max() >= mon else "warn", f"members through {u.month.max():%Y-%m}" if len(u) else "no universe")
        delayed = now - pd.Timedelta(minutes=D.SIP_DELAY_MIN)
        feeds = [("sip", delayed if self.live.feed == "sip_delayed" else now)]
        if self.m.intraday_symbol and self.m.intraday_feed == "iex":
            feeds.append(("iex", now))
        for fd, end in feeds:
            sym = self.m.intraday_symbol if fd == "iex" else "SPY"
            try:
                t0 = time.time()
                f = self.broker.bars([sym], end - pd.Timedelta(days=5), end, fd).get(sym)
                add(f"live feed ({fd})", "ok" if f is not None and len(f) else "warn",
                    f"newest {sym} bar {f.index.max():%m-%d %H:%M} ({time.time() - t0:.1f} s)" if f is not None and len(f) else "no bars")
            except Exception as e:                                                            # noqa: BLE001
                add(f"live feed ({fd})", "fail", f"{type(e).__name__}: {str(e)[:120]}")
        st = self.load_state()
        add("state", "warn" if st.get("stopped") or st.get("pending") or st.get("killed") else "ok",
            ("KILL active - clear it to trade again; " if st.get("killed") else "") + ("DRAWDOWN STOP active; " if st.get("stopped") else "")
            + ("an unfinished submit is recorded (check the journal); " if st.get("pending") else "")
            + f"{len(st.get('book') or {})} overnight position(s) in the book · high-water mark {st.get('hwm')}")
        probs = self.arm_problems()
        add("arming", "ok" if not probs else ("warn" if not self.live.armed else "fail"), "can be armed" if not probs else "; ".join(probs))
        return out

    # ── checks against the backtest's data ───────────────────────────────────────────────────────────────────────────────────────────────────────────
    def check_against_backtest(self, log: Callable[[str], None] = lambda s: None) -> pd.DataFrame:
        """Rebuild every journaled overnight decision and intraday check from the BACKTEST's data (cached daily bars + cached 15-minute SIP / IEX bars of that day) with the
        positions and capital live had, and compare. Differences can only come from the data (feed, timing, adjustments) - not from the code."""
        rows = []
        for r in self.read_journal(("decision", "plan_dry", "plan_no_orders")):
            if "model_spec" not in r:
                continue
            day = pd.Timestamp(r["day"]).normalize()
            m = MomModel.from_dict(r["model_spec"])
            syms = list(dict.fromkeys(self.universe_symbols(day, m) + list(r.get("held") or {})))
            try:
                mk = MomMarket(syms, minute=m.minute, start=day - pd.Timedelta(days=self.live.history_days), end=day + pd.Timedelta(days=1),
                               bars_from=day - pd.Timedelta(days=self.live.bar_days), intraday_symbol="", bars_fn=self.bars_fn)
            except Exception as e:                                                            # noqa: BLE001
                log(f"{day:%Y-%m-%d}: cannot build the backtest data ({type(e).__name__})")
                continue
            if not len(mk.dates) or mk.dates[-1] != day or not np.isfinite(mk.S[-1]).any():
                log(f"{day:%Y-%m-%d}: not in the cached data yet - update the data first")
                continue
            t = len(mk.dates) - 1
            P = E.prepare(m, mk, self.folder, rows=[t])
            held = {s: int(h["shares"]) for s, h in (r.get("held") or {}).items()}
            dec = decide_overnight(m, mk.syms, P["U"][t], bool(P["trend"][t]), P["d45"][t], P["pred"][t] if P["pred"] is not None else None, mk.S[t], held, float(r["capital"]),
                                   sectors=E.sector_list(mk.syms) if m.sector_cap > 0 else None)
            bt = {(s, "exit"): abs(held[s]) for s, _ in dec.exits}
            bt.update({(s, "entry"): sh for s, sh, *_ in dec.entries})
            lv = {(o["symbol"], o["kind"]): int(o["qty"]) for o in (r.get("orders") or [])}
            for k in sorted(set(bt) | set(lv)):
                b, l_ = bt.get(k), lv.get(k)
                rows.append(dict(day=day, leg="overnight", what=k[1], symbol=k[0], live=l_ or 0, backtest=b or 0,
                                 match="same" if (b and l_ and b == l_) else ("qty differs" if b and l_ else ("live only" if l_ else "backtest only"))))
        for r in self.read_journal(("check",)):
            day = pd.Timestamp(r["day"]).normalize()
            s = self.m.intraday_symbol
            if not s:
                break
            try:
                mk = MomMarket([s], minute=self.m.minute, start=day - pd.Timedelta(days=self.live.history_days), end=day + pd.Timedelta(days=1),
                               bars_from=day - pd.Timedelta(days=self.live.bar_days), intraday_symbol=s, intraday_feed=self.m.intraday_feed, bars_fn=self.bars_fn)
            except Exception as e:                                                            # noqa: BLE001
                log(f"{day:%Y-%m-%d}: cannot build the backtest data ({type(e).__name__})")
                continue
            t = len(mk.dates) - 1
            if t < 1 or mk.dates[t] != day or mk.iq["src"][t] != self.m.intraday_feed:
                log(f"{day:%Y-%m-%d}: no cached {self.m.intraday_feed} bars of {s} for that day yet - update the data first")
                continue
            slot = int(r["slot"])
            upper = noise_upper(mk.iq["op"][t], mk.C[t - 1, mk.col[s]], noise_sigma(mk, self.m.intraday_lookback)[t], self.m.intraday_mult)
            act = noise_action(int(r.get("held") or 0) > 0, mk.iq["cl"][t, slot], upper[slot], mk.iq["vwap"][t, slot], self.m.intraday_trail) or ""
            rows.append(dict(day=day, leg="intraday", what=f"{slot_end(day, slot):%H:%M} check", symbol=s, live=r.get("action") or "-", backtest=act or "-",
                             match="same" if act == (r.get("action") or "") else "differs"))
        return pd.DataFrame(rows, columns=["day", "leg", "what", "symbol", "live", "backtest", "match"])

    def shadow_report(self) -> pd.DataFrame:
        """Per journaled decision with a shadow: the overnight return (close -> next official open, before costs) of the traded picks and of the shadow's picks, from the cached
        daily bars (update the data for the latest nights)."""
        rows = []
        dec = self.read_journal(("decision", "plan_dry", "plan_no_orders"))
        need = set()
        for r in dec:
            need |= {o["symbol"] for o in (r.get("orders") or []) if o.get("kind") == "entry"} | {x["symbol"] for x in (r.get("shadow") or [])}
        if not need:
            return pd.DataFrame(columns=["day", "traded", "traded_ret", "shadow", "shadow_ret", "common"])
        d = load_daily(sorted(need))
        C = d.pivot_table(index="date", columns="symbol", values="close")
        O = d.pivot_table(index="date", columns="symbol", values="open")
        for r in dec:
            if r.get("shadow") is None:
                continue
            day = pd.Timestamp(r["day"]).normalize()
            nxt = O.index[O.index > day]
            if not len(nxt) or day not in C.index:
                continue

            def night(names):
                if not names:
                    return 0.0                                                                  # no overnight book that night (SPY below its 200-day mean)
                v = [O.at[nxt[0], s] / C.at[day, s] - 1 for s in names if s in C.columns and np.isfinite(C.at[day, s]) and np.isfinite(O.at[nxt[0], s])]
                return float(np.mean(v)) if v else np.nan
            a = [o["symbol"] for o in (r.get("orders") or []) if o.get("kind") == "entry"]
            b = [x["symbol"] for x in r["shadow"]]
            rows.append(dict(day=day, traded=len(a), traded_ret=night(a), shadow=len(b), shadow_ret=night(b), common=len(set(a) & set(b))))
        return pd.DataFrame(rows, columns=["day", "traded", "traded_ret", "shadow", "shadow_ret", "common"])

    def _official_raw(self, syms: list, since: pd.Timestamp, notes: list) -> pd.DataFrame:
        """The official UNADJUSTED daily open / close (fills are raw prices; adjusted bars would show a later dividend or split as slippage). Without a broker connection
        the adjusted cache stands in (exact until the next split / ex-dividend date)."""
        if self.broker is not None:
            try:
                return (getattr(self.broker, "daily_raw", None) or self.broker.daily)(syms, since)
            except Exception as e:                                                            # noqa: BLE001
                notes.append(f"official raw prices from the broker unavailable ({type(e).__name__}) - the adjusted daily cache stands in")
        return load_daily(syms)

    def book_view(self, since: Optional[pd.Timestamp] = None) -> dict:
        """The book's open positions and trade history rebuilt from the BROKER's order history (paper or real money alike; the journal is not needed), every trade next to
        the backtest's prices for it - an overnight buy at the official close, its sale at the next official open (or the close), the intraday sale at the official close,
        an intraday market order at its check's bar close -/+ the half spread - and the backtest's P&L of the same shares after the model's costs (diff = live - backtest).
        Returns {positions, trades, notes, totals}."""
        if self.broker is None:
            self.connect()
        notes: list = []
        jr = self.read_journal()
        if since is None:
            first = pd.Timestamp(jr[0]["ts"]).normalize() if jr else self.now().normalize() - pd.Timedelta(days=365)
            since = first - pd.Timedelta(days=3)
        get = getattr(self.broker, "orders_since", None)
        orders = get(since) if get else self.broker.orders(since)
        fills = sorted([o for o in orders if o["filled_qty"] > 0], key=lambda o: o["filled_at"] or o["submitted_at"])
        checks = {(str(pd.Timestamp(r["day"]).date()), int(r["slot"])): float(r["price"]) for r in jr if r.get("kind") == "check"}
        isym = self.m.intraday_symbol
        c_open, c_close, c_iclose = self.m.auction_costs()
        h = self.m.half_spread_bp / 1e4
        syms = sorted({o["symbol"] for o in fills})
        ref: dict = {}
        if syms:
            d = self._official_raw(syms, since, notes)
            ref = {(s, pd.Timestamp(t).normalize()): (o, c) for s, t, o, c in zip(d.symbol, d.date, d.open, d.close)}

        def kind(o: dict) -> str:
            p = o["cid"].split("-")
            if len(p) > 1 and p[1] == "KILL":
                return "k"
            return "iC" if p[-1][:2] == "iC" else p[-1][:1]

        def bt(o: dict, k: str) -> tuple:
            """(the backtest's price of this fill, its cost per side): the auctions at the official open / close - with market DAY orders in their place the difference
            is what they cost"""
            day = pd.Timestamp(o["filled_at"] or o["submitted_at"]).normalize()
            op, cl = ref.get((o["symbol"], day), (np.nan, np.nan))
            if k == "o":
                return op, c_open
            if k in ("e", "x"):
                return cl, c_close
            if k == "iC":
                return cl, c_iclose
            if k == "i":
                px = checks.get((str(day.date()), int(o["cid"].split("-")[-1][2:4])), np.nan) if o["cid"].split("-")[-1][2:4].isdigit() else np.nan
                return (px * (1 + h) if o["side"] == "buy" else px * (1 - h)), 0.0
            return np.nan, 0.0

        lots: dict = {}
        trades = []
        for o in fills:
            k, s, q = kind(o), o["symbol"], float(o["filled_qty"])
            if o["side"] == "buy":
                lots.setdefault(s, []).append(dict(q=q, o=o, k=k))
                continue
            while q > 1e-9 and lots.get(s):
                lot = lots[s][0]
                x, e = min(q, lot["q"]), lot["o"]
                (er, ec), (xr, xc) = bt(e, lot["k"]), bt(o, k)
                live = x * (o["filled_avg_price"] - e["filled_avg_price"])
                btp = x * (xr - er) - x * er * ec - x * xr * xc
                trades.append(dict(leg="INTRADAY" if s == isym else "OVERNIGHT", symbol=s, shares=x, entry_at=e["filled_at"], entry_px=e["filled_avg_price"],
                                   entry_how=_how(e), bt_entry=er, exit_at=o["filled_at"], exit_px=o["filled_avg_price"], exit_how="KILL" if k == "k" else _how(o),
                                   bt_exit=xr, pnl=live, bt_pnl=btp, diff=live - btp, ordered=e["qty"], filled=e["filled_qty"]))
                lot["q"] -= x
                q -= x
                if lot["q"] <= 1e-9:
                    lots[s].pop(0)
            if q > 1e-9:
                notes.append(f"{s}: {q:g} shares sold that no buy since {since:%Y-%m-%d} explains (bought earlier or outside the book)")
        pos = self.broker.positions()
        possible = self.possible_symbols()
        prow = []
        for s, ls in lots.items():
            p = pos.get(s) or {}
            now_px = float(p["mv"]) / p["qty"] if p.get("qty") else np.nan
            for lot in ls:
                e = lot["o"]
                er, _ = bt(e, lot["k"])
                prow.append(dict(leg="INTRADAY" if s == isym else "OVERNIGHT", symbol=s, shares=lot["q"], ordered=e["qty"], filled=e["filled_qty"], entry_at=e["filled_at"],
                                 entry_px=e["filled_avg_price"], bt_entry=er, price=now_px, value=lot["q"] * now_px, pnl=lot["q"] * (now_px - e["filled_avg_price"]),
                                 status="filled" if e["filled_qty"] >= e["qty"] else f"partial: {e['filled_qty']:g} of {e['qty']:g} filled"))
        held = {s: sum(x["q"] for x in ls) for s, ls in lots.items()}
        for s, p in pos.items():
            if s not in possible and s != isym:
                prow.append(dict(leg="outside the book", symbol=s, shares=p["qty"], ordered=np.nan, filled=np.nan, entry_at=None, entry_px=p.get("avg_px", np.nan), bt_entry=np.nan,
                                 price=float(p["mv"]) / p["qty"] if p["qty"] else np.nan, value=float(p["mv"]), pnl=np.nan, status="not this book's - left alone"))
            elif abs(p["qty"] - held.get(s, 0)) > 1e-9:
                notes.append(f"{s}: the broker holds {p['qty']:g}, this book's order history explains {held.get(s, 0):g}")
        for s, q in held.items():
            if q > 1e-9 and s not in pos:
                notes.append(f"{s}: the order history leaves {q:g} shares open but the broker holds none")
        tcols = ["leg", "symbol", "shares", "entry_at", "entry_px", "entry_how", "bt_entry", "exit_at", "exit_px", "exit_how", "bt_exit", "pnl", "bt_pnl", "diff", "ordered",
                 "filled"]
        pcols = ["leg", "symbol", "shares", "ordered", "filled", "entry_at", "entry_px", "bt_entry", "price", "value", "pnl", "status"]
        tr = pd.DataFrame(trades, columns=tcols)
        ok = tr.bt_pnl.notna()
        what = {"e": "overnight buy (close)", "x": "overnight sale (close)", "o": "overnight sale (open)", "iC": f"{isym} sale (close)", "k": "KILL"}
        orows = [dict(submitted_at=o["submitted_at"], symbol=o["symbol"], side=o["side"], qty=o["qty"], order=("market-on-close" if o["tif"] == "cls" else
                                                                                                             "market-on-open" if o["tif"] == "opg" else "market (day)"),
                      leg=what.get(kind(o), f"{isym} {o['side']} (intraday check)" if kind(o) == "i" else "?"), status=o["status"], filled_qty=o["filled_qty"],
                      filled_avg_price=o["filled_avg_price"], filled_at=o["filled_at"], cid=o["cid"])
                 for o in sorted(orders, key=lambda o: o["submitted_at"] or pd.Timestamp(0), reverse=True)]
        ocols = ["submitted_at", "symbol", "side", "qty", "order", "leg", "status", "filled_qty", "filled_avg_price", "filled_at", "cid"]
        totals = dict(trades=len(tr), pnl=float(tr.pnl.sum()), bt_pnl=float(tr.bt_pnl[ok].sum()), diff=float(tr["diff"][ok].sum()), priced=int(ok.sum()),
                      open=len(prow), since=since, orders=len(orows), unfilled=sum(1 for o in orows if o["status"] in ("expired", "canceled", "rejected") and
                                                                                       not o["filled_qty"]))
        return dict(positions=pd.DataFrame(prow, columns=pcols), trades=tr, orders=pd.DataFrame(orows, columns=ocols), notes=notes, totals=totals)

    def fill_report(self) -> pd.DataFrame:
        """Every journaled auction-time order of the book (entries, exits at the close and at the open, the intraday sale at the close) against the price the backtest fills
        it at - the official close or open, from the cached daily bars (update the data for the latest sessions). cost_bp > 0 = worse than that price (the backtest adds its
        auction cost on top)."""
        cols = ["day", "what", "symbol", "side", "tif", "qty", "filled", "fill_px", "ref", "ref_px", "cost_bp"]
        what = {"e": "entry at the close", "x": "exit at the close", "o": "exit at the open", "iC": "intraday sale at the close"}
        want = []
        for r in self.read_journal(("fill",)):
            leg = str(r.get("cid", "")).split("-")[-1]
            k = "iC" if leg[:2] == "iC" else leg[:1]
            if k in what:
                want.append((r, k))
        if not want:
            return pd.DataFrame(columns=cols)
        d = self._official_raw(sorted({r["symbol"] for r, _ in want}), min(pd.Timestamp(r["day"]) for r, _ in want) - pd.Timedelta(days=3), [])
        ref = {(s, pd.Timestamp(t).normalize()): (o, c) for s, t, o, c in zip(d.symbol, d.date, d.open, d.close)}
        rows = []
        for r, k in want:
            day = pd.Timestamp(r["day"]).normalize()
            o, c = ref.get((r["symbol"], day), (np.nan, np.nan))
            rp = o if k == "o" else c
            fq, fp = float(r.get("filled_qty") or 0), float(r["avg_px"]) if r.get("avg_px") is not None else np.nan
            cost = (1 if r["side"] == "buy" else -1) * (fp / rp - 1) * 1e4 if fq > 0 and np.isfinite(fp) and np.isfinite(rp) and rp > 0 else np.nan
            rows.append(dict(day=day, what=what[k], symbol=r["symbol"], side=r["side"], tif=r.get("tif"), qty=float(r.get("qty") or 0), filled=fq, fill_px=fp,
                             ref="open" if k == "o" else "close", ref_px=rp, cost_bp=cost))
        return pd.DataFrame(rows, columns=cols)
