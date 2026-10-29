"""
Live / paper engine for the leveraged-ETF pairs book on Alpaca (no Qt).

Once per bar (5 minutes by default), a few seconds after the bar closes:
    fetch the latest bars  ->  signals for the newest bar (the SAME build_signals the backtest runs)  ->  decide_bar (the SAME state machine)  ->  target spread position per pair
    ->  whole-share orders for both legs  ->  (only when ARMED) market orders, fills confirmed, a failed leg unwinds the other one
Everything is flat before the close; nothing is held overnight.

Safety, in the spirit of the Supertrend and Stat-Arb apps:
  * starts on PAPER and DRY RUN. DRY RUN keeps a shadow book (its own state file) and sends nothing; ``armed`` must be switched on explicitly and a real-money account needs the typed phrase in the GUI
  * orders are tagged ``LP-<yyyymmdd>-<pair>-<n>``; the engine only touches the six ETFs in its own ledger and never the rest of the account
  * this account may not read real-time SIP data, only IEX: the signals are computed on the IEX feed (``feed='iex'``) - the fidelity test is in LEVPAIRS_REPORT.md
  * a stale feed holds (and then exits); a daily-loss halt flattens and blocks entries; KILL cancels the engine's orders and flattens its ledger; a hard flatten runs before the close
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

import levpairs_data as D
from levpairs_core import PairSignals, SigCfg, build_signals
from levpairs_sim import PAIR_ORDER, SimCfg, bar_gross, daily_sigma, decide_bar, entry_filters, filter_costs, leg_costs, unit_gross_leverage

NY = D.NY
PREFIX = "LP"
HISTORY_SESSIONS = 45


@dataclass
class LiveSettings:
    paper: bool = True
    armed: bool = False                          # False = DRY RUN: the shadow book runs, nothing is sent
    feed: str = "iex"                            # feed of the signals (this account: IEX only in real time)
    equity_fraction: float = 1.0
    capital_override: float = 0.0                # > 0: deploy this much instead of equity x fraction
    min_leg_usd: float = 200.0
    max_gross_x: float = 1.0                     # cap on total gross exposure of the open pairs, x min(deployed capital, account equity)
    max_gross_nav: float = 0.25                  # ... and as a share of account equity (RM: start at 25%)
    max_leg_nav: float = 0.10                    # no single leg above this share of account equity (the hedge leg is ~3x the other one)
    g_cap: float = 1.0                           # the volatility target saturates at its cap on this book (realised vol 0.7% vs 10%): the live gross multiplier never exceeds this
    max_trades_per_pair: int = 40                # round trips per pair per session
    daily_loss_halt: float = 0.01                # of deployed capital
    decision_delay_s: float = 4.0                # wait this long after a bar end for the bar to appear
    fill_wait_s: float = 15.0
    stale_bars: int = 3                          # this many consecutive missing bars in a leg = feed outage
    hard_flat_time: str = "15:52:00"             # flatten whatever is left, regardless of signals
    require_etb: bool = True
    accept_iex: bool = False                     # an ARMED session on IEX signals is refused unless this is set: IEX s-scores of these pairs correlate only 0.2-0.57 with the consolidated tape
    flatten_verify_s: float = 12.0               # a flatten re-reads the broker until the closing fills show up (the ledger is cleared only then)
    flatten_attempts: int = 6                    # ... and tries again this many times with a back-off
    flatten_backoff_s: float = 5.0
    flatten_until: str = "15:58:30"                # no more attempts today after this time (a market order after the close does not fill)
    flatten_join_s: float = 420.0                # a flatten / KILL that finds another flatten running waits this long for it: two closers on one ledger double-close
    kill_lock_wait_s: float = 5.0                # KILL waits this long for the step lock, then goes ahead without it (never beside another flatten)
    max_calls_per_min: int = 120                 # the flatten / reconcile paths never make more broker calls than this per minute (Alpaca allows 200 per account; 0 = no limit)
    flatten_poll_s: float = 1.0                  # a flatten re-reads the broker this often while it waits for the closing fills
    submit_timeout_s: float = 20.0               # both legs of an entry / exit are sent at the same moment; a submit call that has not returned after this long is treated as an unknown order
    lookup_retry_s: float = 1.0                  # an order whose submit result is ambiguous is looked up by its client id up to 3 times, this long apart
    resolve_unknown: bool = True                 # an entry with a leg in doubt is cleaned up in the same step: wait, cancel this engine's open orders, close what the two legs left (orphan flatten of their symbols)
    unknown_resolve_wait_s: float = 5.0          # ... after waiting this long for the late order to land
    flatten_orphans_eod: bool = False            # DEDICATED account only: at the hard flatten also close anything in the six ETFs the ledger does not explain
    lock_stale_s: float = 180.0                  # a heartbeat lock file younger than this means another engine is running
    sip_max_age_min: float = 5.0                 # with the SIP feed: the newest 1-minute bar of each leg must be this fresh (a non-real-time account sees data 16 minutes old)


@dataclass
class Check:
    name: str
    level: str                                   # ok | warn | fail
    msg: str


def pair_position_text(name: str, ps: dict) -> str:
    """'LONG TQQQ 357 sh / SHORT QQQ 107 sh' or 'FLAT' - which product is bought and which is sold right now."""
    if not ps.get("side"):
        return "FLAT"
    L, B = name.split("/")
    leg = lambda sym, q: f"{'LONG' if q > 0 else 'SHORT'} {sym} {abs(int(q)):,} sh"
    return leg(L, ps["shares_L"]) + " / " + leg(B, ps["shares_B"])


LOG_COLS = {
    "decisions": ["date", "j", "time", "wall", "pair", "mode", "s", "hl_min", "sig_eq_bp", "beta", "L_px", "B_px", "pos_before", "target", "pos_after", "position", "action", "shares_L", "shares_B", "fills", "note"],
    # the order journal: one row per order sent (entries, exits, unwinds, KILL / end-of-day closes, orphan closes), with the decision it belongs to and every timestamp we can see
    "fills": ["date", "time", "pair", "action", "trip", "j", "s", "decided_at", "symbol", "side", "qty", "px", "status", "ref", "slip_bp", "order_id", "client_order_id", "submitted_at", "filled_at",
              "t_submit_ns", "t_ack_ns", "t_seen_ns", "note"],
}


try:
    import msvcrt

    def _os_lock(fd: int) -> None:
        os.lseek(fd, 0, 0)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)                                           # OSError at once if another handle holds the byte

    def _os_unlock(fd: int) -> None:
        os.lseek(fd, 0, 0)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
except ImportError:                                                                      # pragma: no cover - not Windows
    import fcntl

    def _os_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _os_unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def clock_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


def state_path(mode: str) -> str:
    os.makedirs(D.REPORTS, exist_ok=True)
    return os.path.join(D.REPORTS, f"levpairs_state_{mode}.json")


def mode_name(live: LiveSettings) -> str:
    return "dry" if not live.armed else ("paper" if live.paper else "live")


# ── bar sources: the API, or the cached parquet (replays and tests) ──────────────────────────────────────────────────────────────────────────────────
class ApiSource:
    def __init__(self, feed: str):
        self.feed = feed
        self.client = D.data_client()

    def bars1(self, sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """1-minute RTH bars in [start, end) (ET naive). The SIP feed cannot be queried for the last 15 minutes on this account; IEX can."""
        e = pd.Timestamp(end).tz_localize(NY).tz_convert("UTC")
        if self.feed == "sip":
            e = min(e, pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16))
        b = D.fetch_1m(sym, self.feed, start=pd.Timestamp(start).tz_localize(NY).tz_convert("UTC"), end=e, client=self.client, log=lambda m: None)
        return b[b.index < pd.Timestamp(end)]


class ParquetSource:
    def __init__(self, feed: str):
        self.feed = feed
        self._c: dict = {}

    def bars1(self, sym: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if sym not in self._c:
            self._c[sym] = pd.read_parquet(D.path1(sym, self.feed))
        b = self._c[sym]
        return b[(b.index >= pd.Timestamp(start)) & (b.index < pd.Timestamp(end))]


# ── signals for the newest bar ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class PairView:
    name: str
    j: int
    s: float
    hl_min: float
    sig_eq_bp: float
    beta: float
    L_px: float
    B_px: float
    eps_j: float
    eps_gap_j: float
    eps_body_j: float
    miss_tail: int
    rB_day: float
    move_block: bool
    s_hist: np.ndarray
    fin: bool = True
    x_bp: float = float('nan')
    x_hist_bp: Optional[np.ndarray] = None
    L_prev: float = float('nan')                  # close of the previous bar (the shadow book prices the fill off it)
    B_prev: float = float('nan')
    L_open_j: float = float('nan')                # open of the bar that just ended
    B_open_j: float = float('nan')
    L_open_h: Optional[np.ndarray] = None         # the session so far (bars 0..j): lets the shadow book catch up bars that were missed in a restart or an outage
    L_close_h: Optional[np.ndarray] = None
    B_open_h: Optional[np.ndarray] = None
    B_close_h: Optional[np.ndarray] = None


def tf_bars(b1: pd.DataFrame, tf: int, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    r = D.resample(b1, tf)
    if end is not None and len(r):
        r = r[r.index + pd.Timedelta(minutes=tf) <= end]                                # only bars that have fully elapsed
    return r


def bar_index_now(date: pd.Timestamp, now: pd.Timestamp, tf: int) -> int:
    """Index of the newest bar that has completely elapsed at ``now`` (naive ET); -1 before the first bar ends."""
    mins = (now - (pd.Timestamp(date).normalize() + pd.Timedelta(minutes=D.OPEN_MIN))).total_seconds() / 60.0
    return int(mins // tf) - 1


def pair_views(source, pairs: list, sc: SigCfg, c: SimCfg, date: pd.Timestamp, now: pd.Timestamp, stale_bars: int = 3, history: Optional[dict] = None, sip_max_age_min: float = 5.0) -> tuple[dict, dict]:
    """Signals for every pair at the newest elapsed bar of ``date`` as seen at ``now``. ``history`` ({sym: 1-minute bars of earlier sessions}) is reused between calls; returns (views, history)."""
    tf = sc.tf
    j = bar_index_now(date, now, tf)
    nb = len(D.session_grid(tf))
    if j < 0 or j >= nb:
        return {}, history or {}
    day0 = pd.Timestamp(date).normalize()
    if history is None:
        history = {}
    views = {}
    for p in pairs:
        name = D.pair_name(p)
        frames, newest = {}, {}
        for sym in p:
            if sym not in history:
                history[sym] = source.bars1(sym, day0 - pd.Timedelta(days=int(HISTORY_SESSIONS * 1.6)), day0)
            today = source.bars1(sym, day0, now)
            frames[sym] = pd.concat([history[sym], today])
            newest[sym] = today.index.max() if len(today) else None
        L, B = (tf_bars(frames[s], tf, now) for s in p)
        pdata = D.pairdata_from_bars(p, L, B, tf, source.feed, partial_day=day0)
        if pdata.dates[-1] != day0 or len(pdata.dates) < 8:
            continue
        sig = build_signals(pdata, sc)
        d = len(pdata.dates) - 1
        miss = lambda m: int(np.argmax(~m[d, :j + 1][::-1])) if (~m[d, :j + 1]).any() else j + 1        # consecutive missing bars at the end
        dl = max(c.delay_bars, 0)
        jj = max(j - dl, 0)
        sg = daily_sigma(pdata.B_close)[d]
        rb = float(sig.rB_day[d, j])
        mb = bool(c.move_z > 0 and np.isfinite(sg) and abs(rb) > c.move_z * sg)
        old = source.feed == "sip" and any(newest[s] is None or (pd.Timestamp(now) - newest[s]) > pd.Timedelta(minutes=sip_max_age_min) for s in p)       # real-time SIP only: stale data is no signal
        views[name] = PairView(name, j, float(sig.s[d, jj]) if dl == 0 or j >= dl else float("nan"), float(sig.hl_min[d, jj]), float(sig.sig_eq_bp[d, jj]), float(sig.beta[d]), float(pdata.L_close[d, j]),
                               float(pdata.B_close[d, j]), float(sig.eps[d, j]), float(sig.eps_gap[d, j]), float(sig.eps_body[d, j]), max(miss(pdata.L_miss), miss(pdata.B_miss)), rb, mb, sig.s[d, :j + 1].copy(),
                               fin=bool(np.isfinite(sig.s[d, jj]) and max(miss(pdata.L_miss), miss(pdata.B_miss)) < stale_bars and not old), x_bp=float(sig.X[d, j]) * 1e4, x_hist_bp=sig.X[d, :j + 1] * 1e4,
                               L_prev=float(pdata.L_close[d, j - 1]) if j >= 1 else float('nan'), B_prev=float(pdata.B_close[d, j - 1]) if j >= 1 else float('nan'),
                               L_open_j=float(pdata.L_open[d, j]), B_open_j=float(pdata.B_open[d, j]), L_open_h=pdata.L_open[d, :j + 1].copy(), L_close_h=pdata.L_close[d, :j + 1].copy(),
                               B_open_h=pdata.B_open[d, :j + 1].copy(), B_close_h=pdata.B_close[d, :j + 1].copy())
    return views, history


# ── the engine ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class LiveEngine:
    def __init__(self, sc: SigCfg, c: SimCfg, live: LiveSettings, log: Callable[[str, str], None] = lambda m, k="info": print(m), source=None):
        self.sc, self.c, self.live, self.log = sc, c, live, log
        self.tc = None
        self.acct: dict = {}
        self.source = source
        self.history: dict = {}
        self.G = c.g_init
        self.deploy = 0.0
        self.halted = False
        self.views: dict = {}
        self._lock = threading.RLock()                                                   # step / flatten / KILL never run at the same time
        self._calls: deque = deque()                                                     # times of the recent broker calls (see _throttle)
        self._calls_lock = threading.Lock()
        self._xlock_held: Optional[str] = None
        self._xlock_fd: Optional[int] = None
        self._submit_threads: list = []                                                  # the submit threads that may still be alive (a hung call returns late)
        self._flatten_guard = threading.Lock()                                           # ... and only ONE flatten runs at a time, even when KILL goes ahead without the step lock
        self.is_replay = False
        self.kill_flag = threading.Event()                                               # set by KILL: the loop stops entering immediately
        self._lockfile: Optional[str] = None

    # ── broker ───────────────────────────────────────────────────────────────────
    def connect(self) -> dict:
        from alpaca.trading.client import TradingClient
        k, s = D.creds(self.live.paper)
        self.tc = D.harden(TradingClient(k, s, paper=self.live.paper))                  # alpaca-py sets no timeout: a stalled call would freeze the engine for ever
        t0 = time.time()
        a = self.tc.get_account()
        clk = self.tc.get_clock()
        self.acct = dict(paper=self.live.paper, number=str(a.account_number)[:4] + "****", status=str(getattr(a.status, "value", a.status)), equity=float(a.equity), cash=float(a.cash),
                         buying_power=float(a.buying_power), multiplier=str(a.multiplier), shorting=bool(a.shorting_enabled), blocked=bool(a.trading_blocked or a.account_blocked),
                         pattern_day_trader=bool(a.pattern_day_trader), daytrading_bp=float(getattr(a, "daytrading_buying_power", 0) or 0), market_open=bool(clk.is_open),
                         next_open=pd.Timestamp(clk.next_open), next_close=pd.Timestamp(clk.next_close), latency_ms=int((time.time() - t0) * 1000), last_equity=float(a.last_equity))
        ts_ = getattr(clk, "timestamp", None)
        if ts_ is not None:
            self.acct["clock_skew_s"] = float((pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts_).tz_convert("UTC")).total_seconds())      # PC minus broker, about half a round trip included
        return self.acct

    def _throttle(self) -> None:
        """Wait, if need be, so that no 60-second window holds more than ``max_calls_per_min`` broker calls made through the flatten / reconcile paths: a flatten that cannot close loops with a back-off
        and re-reads the broker for every symbol, and must not run into Alpaca's rate limit (HTTP 429) in the middle of an emergency."""
        lim = self.live.max_calls_per_min
        if lim <= 0:
            return
        while True:
            with self._calls_lock:
                now = time.monotonic()                                                   # not the wall clock: a PC clock stepped back must not stall a flatten for an hour
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < lim:
                    self._calls.append(now)
                    return
                wait = 60.0 - (now - self._calls[0])
            time.sleep(min(60.0, max(0.05, wait)))

    def broker_positions(self) -> dict:
        self._throttle()
        return {p.symbol: float(p.qty) for p in self.tc.get_all_positions() if p.symbol in D.SYMBOLS}

    def session_info(self, date) -> Optional[tuple]:
        from alpaca.trading.requests import GetCalendarRequest
        d = pd.Timestamp(date).date()
        cal = self.tc.get_calendar(GetCalendarRequest(start=d, end=d))
        return (pd.Timestamp(cal[0].open), pd.Timestamp(cal[0].close)) if cal and cal[0].date == d else None

    # ── state ────────────────────────────────────────────────────────────────────
    def blank_state(self, date) -> dict:
        return dict(date=str(pd.Timestamp(date).date()), last_j=-1, halted=False, start_equity=None, pairs={n: dict(pos=0, age=0, cool=0, stopped=False, n_trades=0, shares_L=0, shares_B=0, side=0,
                    entry_time=None, entry_s=None, N_L=0.0, pnl_usd=0.0, broken=False, side_prev=0, gross_usd=0.0, entry_px={}, realized_usd=0.0, ent_L=0.0, ent_B=0.0, attempts=0, trip=None) for n in PAIR_ORDER}, shadow_pnl=0.0, orders=0, pending=None)

    @staticmethod
    def _seed_entry(st: dict) -> None:
        """A state file written before the fixed-share shadow book has no entry prices of the open positions: take them from ``entry_px`` instead of 1.0 (which books a bogus first bar and can fake a loss halt)."""
        for name, ps in st["pairs"].items():
            ep = ps.get("entry_px") or {}
            if ps.get("side") and not (ps.get("ent_L") and ps.get("ent_B")) and len(ep) == 2:
                L, B = name.split("/")
                ps["ent_L"], ps["ent_B"] = float(ep[L]), float(ep[B])

    def load_state(self, date) -> dict:
        path = state_path(mode_name(self.live))
        if os.path.exists(path):
            try:
                st = json.load(open(path, encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise RuntimeError(f"state file {path} is unreadable ({e}); fix or move it - starting from an empty ledger would forget real positions") from e
            if st.get("date") == str(pd.Timestamp(date).date()):
                blank = self.blank_state(date)
                for k, v in blank["pairs"].items():                                    # states written by an older version get the new fields
                    st["pairs"].setdefault(k, v)
                    for f, dv in v.items():
                        st["pairs"][k].setdefault(f, dv)
                st.setdefault("pending", None)
                self._seed_entry(st)
                return st
            leftover = {k: v for k, v in st.get("pairs", {}).items() if v.get("side")}
            if leftover and self.live.armed:
                raise RuntimeError(f"the ledger still holds positions from {st.get('date')}: {list(leftover)}. Flatten them (KILL) or reconcile before a new session - nothing is meant to be held overnight")
        return self.blank_state(date)

    def save_state(self, st: dict) -> None:
        path = state_path(mode_name(self.live))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=1, default=str)
        for k in range(6):                                                               # the watchdog (or an editor) may have the file open for a moment: Windows refuses the replace
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if k == 5:
                    raise
                time.sleep(0.2)

    def append_log(self, name: str, rows: list) -> None:
        """Append rows to reports/levpairs_<name>_<mode>.csv. Decisions and the order journal have a FIXED column layout (rows of an entry carry more fields than rows of a hold, and a header written by the
        first row would not fit the later ones: pandas could not read the file); a log written with another layout is kept aside, never mixed. They are ALSO appended to a .jsonl twin: the CSV is what you open
        in Excel (which locks it on Windows, so the CSV append can fail) - the JSONL is the record, it is never lost to a lock and is what levpairs_compare.py reads."""
        if not rows:
            return
        base = os.path.join(D.REPORTS, f"levpairs_{name}_{mode_name(self.live)}")
        df, cols = pd.DataFrame(rows), LOG_COLS.get(name)
        if cols:
            df = df.reindex(columns=cols)
            try:
                with open(base + ".jsonl", "a", encoding="utf-8") as fh:
                    for rec in df.to_dict("records"):
                        rec = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in rec.items()}                  # numpy scalars would be written as text
                        fh.write(json.dumps({k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in rec.items()}, default=str) + "\n")
            except OSError as e:
                self.log(f"could not append to {os.path.basename(base)}.jsonl ({type(e).__name__}: {e})", "err")
        f = base + ".csv"
        try:
            if cols and os.path.exists(f):
                try:
                    head = pd.read_csv(f, nrows=0).columns.tolist()
                except Exception:                                                        # noqa: BLE001
                    head = None
                if head != cols:
                    os.replace(f, f"{f}.{pd.Timestamp.now():%Y%m%d%H%M%S}.old")
            df.to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        except OSError as e:                                                             # an open Excel window: the JSONL has the rows
            self.log(f"could not append to {os.path.basename(f)} ({type(e).__name__}) - is it open in Excel? the rows are in the .jsonl", "warn")

    def _journal(self, rows: list) -> None:
        try:
            self.append_log("fills", rows)
        except Exception as e:                                                           # noqa: BLE001 - the journal must never stop an order flow
            self.log(f"order journal write failed ({type(e).__name__}: {str(e)[:80]})", "err")

    # ── preflight ────────────────────────────────────────────────────────────────
    def preflight(self) -> list[Check]:
        out: list[Check] = []
        add = lambda n, l, m: out.append(Check(n, l, m))
        try:
            a = self.connect()
            add("broker", "ok", f"{'PAPER' if a['paper'] else 'LIVE'} account {a['number']} · {a['status']} · equity ${a['equity']:,.0f} · buying power ${a['buying_power']:,.0f} · {a['latency_ms']} ms")
            if a["blocked"]:
                add("account", "fail", "trading is blocked on this account")
            if not a["shorting"]:
                add("shorting", "fail", "shorting is not enabled (every trade shorts one leg)")
            if float(a["multiplier"] or 1) < 2:
                add("margin", "fail", f"account multiplier {a['multiplier']}x - the pair book needs a margin account (2x or 4x intraday)")
            if not a["paper"] and a["equity"] < 25_000:
                add("pdt", "fail", "equity < $25k on a margin account: the pattern-day-trader rule would block this strategy (it opens and closes positions every day)")
            if a["pattern_day_trader"]:
                add("pdt", "warn", "the account is flagged pattern day trader")
            if a.get("clock_skew_s") is not None:
                out.append(self._clock_check(a["clock_skew_s"]))
        except Exception as e:                                                         # noqa: BLE001
            add("broker", "fail", f"cannot reach Alpaca: {type(e).__name__}: {str(e)[:120]}")
            return out
        # assets
        try:
            bad = []
            for sym in D.SYMBOLS:
                ast = self.tc.get_asset(sym)
                if not ast.tradable or not ast.shortable or (self.live.require_etb and not ast.easy_to_borrow):
                    bad.append(f"{sym}(tradable={ast.tradable}, shortable={ast.shortable}, etb={ast.easy_to_borrow})")
            add("assets", "fail" if bad else "ok", ("not tradable / shortable / easy-to-borrow: " + ", ".join(bad)) if bad else "all six ETFs tradable, shortable and easy-to-borrow")
        except Exception as e:                                                         # noqa: BLE001
            add("assets", "warn", f"could not check the assets ({type(e).__name__}: {str(e)[:80]})")
        # data feed
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockLatestQuoteRequest
            cl = D.data_client()
            for feed in ("sip", "iex"):
                try:
                    cl.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols="SPY", feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX))
                    ok = True
                except Exception:                                                      # noqa: BLE001
                    ok = False
                if feed == "sip":
                    sip_ok = ok
                else:
                    iex_ok = ok
            lvl = "ok" if (self.live.feed == "sip" and sip_ok) else "fail" if (self.live.feed == "sip" or not iex_ok) else ("warn" if not self.live.armed or self.live.accept_iex else "fail")
            add("data feed", lvl, f"real-time SIP {'permitted' if sip_ok else 'NOT permitted'} · real-time IEX {'permitted' if iex_ok else 'not permitted'} · signals use the {self.live.feed.upper()} feed"
                + (" - IEX s-scores of these pairs correlate only 0.2-0.57 with the consolidated tape (levpairs_feed_test.py): fine for watching in DRY RUN, not for sending orders; ARMED needs 'accept IEX' ticked"
                   if self.live.feed == "iex" else ""))
        except Exception as e:                                                         # noqa: BLE001
            add("data feed", "warn", f"{type(e).__name__}: {str(e)[:100]}")
        try:
            src = self.source or ApiSource(self.live.feed)
            b = src.bars1("SPY", pd.Timestamp.now(tz=NY).tz_localize(None).normalize() - pd.Timedelta(days=6), pd.Timestamp.now(tz=NY).tz_localize(None) + pd.Timedelta(minutes=1))
            last = b.index.max() if len(b) else None
            add("bars", "ok" if last is not None else "fail", f"latest {self.live.feed.upper()} 1-minute bar of SPY: {last:%Y-%m-%d %H:%M}" if last is not None else "no recent bars returned")
        except Exception as e:                                                         # noqa: BLE001
            add("bars", "fail", f"{type(e).__name__}: {str(e)[:100]}")
        sp = os.path.exists(os.path.join(D.REPORTS, "spreads_levpairs.csv"))
        add("costs", "ok" if sp else "warn", "quoted-spread table present" if sp else "no spread sample yet - the cost model uses default spreads")
        try:
            st = self.load_state(clock_now().tz_localize(None))
            held = {k: (v["shares_L"], v["shares_B"]) for k, v in st["pairs"].items() if v.get("side")}
            add("state", "ok", f"mode {mode_name(self.live).upper()} · ledger: {len(held)} open pair(s)")
        except Exception as e:                                                         # noqa: BLE001
            add("state", "fail", str(e)[:160])
        if os.path.exists(os.path.join(D.REPORTS, "KILL")):
            add("kill file", "fail", "reports/KILL exists - remove it to trade")
        return out

    @staticmethod
    def _clock_check(skew: float) -> Check:
        lvl = "fail" if abs(skew) > 30 else "warn" if abs(skew) > 2 else "ok"
        return Check("clock", lvl, f"the PC clock is {skew:+.1f} s from the broker's (about half a round trip included): the 4 s decision delay, the order timestamps and the live-vs-backtest comparison rely on it"
                     + (" - sync the clock (Windows: Settings > Time & language > Sync now)" if lvl != "ok" else ""))

    # ── session ──────────────────────────────────────────────────────────────────
    def start_session(self, date, equity: Optional[float] = None) -> dict:
        d = pd.Timestamp(date).normalize()
        self.state = self.load_state(d)
        eq = equity if equity is not None else (self.acct.get("equity") if self.acct else None) or 100_000.0
        self.deploy = self.live.capital_override if self.live.capital_override > 0 else eq * self.live.equity_fraction
        if self.state.get("start_equity") is None:
            self.state["start_equity"] = float(eq)
        self.halted = bool(self.state.get("halted"))
        self.G = min(self.next_gross(), self.live.g_cap)
        self.history = {}
        self.save_state(self.state)
        self._write_run_record(d, eq)
        return self.state

    def _write_run_record(self, d: pd.Timestamp, equity: float) -> None:
        """reports/levpairs_run_<mode>_<date>.json: the exact configuration this session ran with (signal and simulation settings, live limits, capital, gross), so the comparison can rebuild the SAME backtest.
        A restart the same day is appended to ``starts`` instead of overwriting the first record."""
        f = os.path.join(D.REPORTS, f"levpairs_run_{mode_name(self.live)}_{d:%Y%m%d}.json")
        try:
            rec = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else dict(date=f"{d:%Y-%m-%d}", mode=mode_name(self.live), sc=asdict(self.sc), c=asdict(self.c), live=asdict(self.live), starts=[])
            rec["starts"].append(dict(at=str(pd.Timestamp.now(tz=NY)), equity=float(equity), deploy=float(self.deploy), G=float(self.G), feed=self.live.feed))
            json.dump(rec, open(f, "w", encoding="utf-8"), indent=1, default=str)
        except (OSError, ValueError) as e:
            self.log(f"could not write the run record ({type(e).__name__}: {e})", "warn")

    def next_gross(self) -> float:
        """Gross leverage for today from the volatility target, computed on the strategy's own backtest returns (cached SIP research data) - fixed ``g_init`` if that is not possible."""
        c = self.c
        if not c.vol_on:
            return float(c.g_init)
        try:
            import levpairs_bridge as BR
            panel = BR.load_panel(self.sc.tf, "sip", log=lambda m: None)
            sigs = BR.get_signals(panel, self.sc, "sip", log=lambda m: None)
            from levpairs_sim import simulate
            u = simulate(sigs, replace(c, start="2017-01-01")).daily["u"]
            return float(unit_gross_leverage(pd.concat([u, pd.Series([0.0], index=[u.index[-1] + pd.Timedelta(days=1)])]), c).iloc[-1])
        except Exception as e:                                                         # noqa: BLE001
            self.log(f"could not size from the volatility target ({type(e).__name__}: {str(e)[:80]}) - using the fixed gross {c.g_init:g}x", "warn")
            return float(c.g_init)

    def weights(self) -> dict:
        w = {k: self.c.pair_w[i] for i, k in enumerate(PAIR_ORDER) if i < len(self.c.pair_w) and self.c.pair_w[i] > 0}
        tot = sum(w.values()) or 1.0
        return {k: v / tot for k, v in w.items()}

    def active_pairs(self) -> list:
        w = self.weights()
        return [p for p in D.PAIRS if D.pair_name(p) in w]

    # ── account helpers ──────────────────────────────────────────────────────────
    def equity_now(self) -> float:
        return float((self.acct or {}).get("equity") or (getattr(self, "state", None) or {}).get("start_equity") or 100_000.0)

    def ledger_shares(self) -> dict:
        led = {}
        for name, ps in self.state["pairs"].items():
            if ps.get("side"):
                L, B = name.split("/")
                led[L], led[B] = ps["shares_L"], ps["shares_B"]
        return led

    def orphans(self) -> dict:
        """symbol -> (ledger shares, broker shares) for each of the six ETFs where they differ. Armed engines only; the broker is the truth."""
        if not self.live.armed or self.tc is None:
            return {}
        led, bp = self.ledger_shares(), self.broker_positions()
        return {s: (int(led.get(s, 0)), int(bp.get(s, 0))) for s in D.SYMBOLS if int(bp.get(s, 0)) != int(led.get(s, 0))}

    def lp_pnl(self) -> float:
        """Day P&L of THIS engine: ARMED = realised on its fills + open positions marked at the latest bar close (so another strategy in the account cannot mask or trigger the halt);
        DRY RUN = the shadow P&L."""
        tot = 0.0
        for name, ps in self.state["pairs"].items():
            if not self.live.armed:
                tot += ps.get("pnl_usd", 0.0)
                continue
            tot += ps.get("realized_usd", 0.0)
            v, ep = self.views.get(name), ps.get("entry_px") or {}
            if ps.get("side") and v is not None and len(ep) == 2:
                L, B = name.split("/")
                tot += ps["shares_L"] * (v.L_px - ep[L]) + ps["shares_B"] * (v.B_px - ep[B])
        return tot

    def loss_halt_check(self, shadow_pnl_usd: float) -> tuple[bool, str]:
        lim = self.live.daily_loss_halt
        if lim <= 0 or self.deploy <= 0:
            return False, ""
        pnl = self.lp_pnl() if (self.live.armed and getattr(self, "state", None)) else shadow_pnl_usd
        if not np.isfinite(pnl):                                                          # a NaN compares False with everything: it would let a real loss through
            return True, "the day P&L is not a number (a price or a state field is missing)"
        if pnl <= -lim * self.deploy:
            return True, f"day P&L ${pnl:,.0f} is below -{lim:.1%} of deployed capital"
        return False, ""

    def _honour_watchdog(self) -> None:
        """A watchdog that flattened this ledger while the engine was hung or asleep leaves reports/levpairs_watchdog_<mode>.json; a revived engine then reloads the (flat) ledger from disk instead of
        overwriting it with its stale one, and stops trading for the day."""
        f = os.path.join(D.REPORTS, f"levpairs_watchdog_{mode_name(self.live)}.json")
        try:
            if not os.path.exists(f) or getattr(self, "_wd_seen", None) == os.path.getmtime(f):
                return
            rec = json.load(open(f, encoding="utf-8"))
            self._wd_seen = os.path.getmtime(f)
            if rec.get("date") == self.state["date"]:
                st = self.load_state(pd.Timestamp(self.state["date"]))
                st["halted"] = True
                self.state.clear()                                                       # IN PLACE: the running step holds a reference to this very dict and would otherwise save its stale copy over the flat ledger
                self.state.update(st)
                self.halted = True
                self.save_state(self.state)
                self.log("the watchdog flattened this account while the engine was away: the ledger is reloaded from disk and the engine stops trading for today", "err")
        except (OSError, ValueError) as e:
            self.log(f"could not read the watchdog marker ({type(e).__name__})", "warn")

    def _book_shadow(self, ps: dict, v: PairView, j: int, last_j: int) -> None:
        """Shadow P&L of the bar(s) that ended since the last processed one, with the arithmetic of simulate_pair (``bar_gross``: fixed shares from the fill price; the position held before the fill earns
        close -> fill, the new one fill -> close). Normally that is one bar; after a restart or an outage the skipped bars are caught up one by one. A bar with a missing price books nothing - never a NaN."""
        hold, hp = ps["side"], ps.get("side_prev", 0)
        if j > 0 and (hold or hp):
            if v.L_close_h is not None and v.L_open_h is not None and len(v.L_close_h) == j + 1:
                bars = [(v.L_close_h[k - 1], v.L_close_h[k], v.L_open_h[k], v.B_close_h[k - 1], v.B_close_h[k], v.B_open_h[k]) for k in range(max(last_j + 1, 1), j + 1)]
            else:
                bars = [(v.L_prev, v.L_px, v.L_open_j, v.B_prev, v.B_px, v.B_open_j)]
            arr = lambda x: np.array([x], dtype=float)
            for Lc0, Lc1, Lo, Bc0, Bc1, Bo in bars:
                if not (np.all(np.isfinite([Lc0, Lc1, Lo, Bc0, Bc1, Bo, v.beta])) and np.isfinite(ps.get("N_L", 0.0))):
                    hp = hold
                    continue
                eL, eB = float(ps.get("ent_L") or 0.0), float(ps.get("ent_B") or 0.0)
                if hp and (eL <= 0 or eB <= 0):
                    eL, eB = float(Lc0), float(Bc0)                                       # no entry price in the state (an older file): value the position it holds from the previous close
                pnl, nL, nB = bar_gross(self.c.fill, np.array([hp]), np.array([hold]), arr(eL), arr(eB), v.beta, arr(Lc0), arr(Lc1), arr(Lo), arr(Bc0), arr(Bc1), arr(Bo))
                if np.isfinite(pnl[0]):
                    ps["pnl_usd"] = ps.get("pnl_usd", 0.0) + float(pnl[0]) * ps["N_L"]
                    ps["ent_L"], ps["ent_B"] = float(nL[0]), float(nB[0])
                hp = hold
        ps["side_prev"] = hold

    # ── one decision step ────────────────────────────────────────────────────────
    def step(self, date, now: pd.Timestamp, force_flat: bool = False) -> list[dict]:
        with self._lock:
            return self._step(date, now, force_flat)

    def _step(self, date, now: pd.Timestamp, force_flat: bool = False) -> list[dict]:
        """Process the newest elapsed bar (if it has not been processed yet). Returns the decision rows."""
        c, sc, live = self.c, self.sc, self.live
        self._honour_watchdog()
        st = self.state
        tf = sc.tf
        j = bar_index_now(date, now, tf)
        nb = len(D.session_grid(tf))
        if j < 0 or j >= nb or j <= st["last_j"]:
            return []
        src = self.source or ApiSource(live.feed)
        self.source = src
        if live.armed and self.tc is not None:
            try:
                self.acct["equity"] = float(self.tc.get_account().equity)
            except Exception:                                                            # noqa: BLE001
                pass
            try:                                                                         # the broker is the truth: a position the ledger does not explain blocks the pair
                orph = self.orphans()
                for sym, (lg, bk) in orph.items():
                    for name in PAIR_ORDER:
                        if sym in name.split("/") and not st["pairs"][name].get("broken"):
                            st["pairs"][name]["broken"] = True
                            self.log(f"RECONCILE: {sym} ledger {lg:+d} vs broker {bk:+d} - {name} is blocked (BROKEN) until the account is flat / reconciled", "err")
            except Exception as e:                                                       # noqa: BLE001
                self.log(f"reconcile failed ({type(e).__name__}: {str(e)[:80]})", "warn")
        views, self.history = pair_views(src, self.active_pairs(), sc, c, date, now, live.stale_bars, self.history, live.sip_max_age_min)
        self.views = views
        rows = []
        w = self.weights()
        eod_flat = j >= nb - 1 - int(math.ceil(c.flat_min / tf))
        for p in self.active_pairs():
            name = D.pair_name(p)
            v = views.get(name)
            ps = st["pairs"][name]
            row = dict(date=str(pd.Timestamp(date).date()), j=j, time=f"{pd.Timestamp(date).normalize() + pd.Timedelta(minutes=D.OPEN_MIN + (j + 1) * tf):%H:%M}", pair=name, mode=mode_name(live),
                       wall=pd.Timestamp.now(tz=NY).isoformat(timespec="milliseconds"))
            if v is None:
                row.update(note="no signal (not enough data)", pos_before=ps["side"], pos_after=ps["side"])
                rows.append(row)
                continue
            cL, cB = leg_costs(p, c)
            fL, fB = filter_costs(p, c)
            unit_cost = (cL + v.beta * cB) / 1e4
            hold = ps["side"]
            self._book_shadow(ps, v, j, st["last_j"])                                    # shadow P&L of the bar(s) that ended since the last processed one
            s2 = np.array([[v.s]]); hl2 = np.array([[v.hl_min]]); se2 = np.array([[v.sig_eq_bp]])
            hl_ok, edge_ok = entry_filters(s2, hl2, se2, np.array([v.beta]), c, sc.s_mode, fL, fB)
            pos = np.array([ps["pos"]], dtype=np.int8)
            gap = max(j - st["last_j"] - 1, 0) if st["last_j"] >= 0 else 0                # bars that passed without a decision (a restart, an outage) still age the position and run the cooldown
            age, cool = np.array([ps["age"] + (gap if ps["side"] else 0)]), np.array([max(ps["cool"] - gap, 0)])
            stopped = np.array([bool(ps["stopped"]) or (c.day_stop_bp > 0 and ps.get("pnl_usd", 0.0) / max(ps["N_L"], 1.0) * 1e4 <= -c.day_stop_bp and bool(hold))])
            blocked = self.halted or ps.get("broken", False) or force_flat or self.kill_flag.is_set() or os.path.exists(os.path.join(D.REPORTS, "KILL")) or ps["n_trades"] >= live.max_trades_per_pair
            fin = np.array([v.fin and not blocked])
            if blocked:                                                                  # an engine-level block behaves like a dead signal: exit and do not enter
                fin = np.array([False])
            pos, age, cool, reason = decide_bar(j, nb, tf, c, pos, age, cool, stopped, np.array([v.s]), fin, hl_ok[0], edge_ok[0], np.array([v.move_block]))
            tgt = int(pos[0])
            if eod_flat:
                tgt = 0
            row.update(s=v.s, hl_min=v.hl_min, sig_eq_bp=v.sig_eq_bp, beta=v.beta, L_px=v.L_px, B_px=v.B_px, pos_before=ps["side"], target=tgt, note=EXIT_TXT.get(int(reason[0]), "") if tgt == 0 and ps["side"] else "")
            if tgt != ps["side"]:
                done = self.trade_pair(p, name, ps, tgt, v, w[name], unit_cost, row)
                if not done:
                    tgt = ps["side"]
                    pos = np.array([ps["side"]], dtype=np.int8)
            ps["pos"], ps["age"], ps["cool"], ps["stopped"] = int(pos[0]) if tgt == int(pos[0]) else ps["side"], int(age[0]), int(cool[0]), bool(stopped[0])
            row["pos_after"] = ps["side"]
            row["position"] = pair_position_text(name, ps)
            rows.append(row)
        halt, why = self.loss_halt_check(sum(x.get("pnl_usd", 0.0) for x in st["pairs"].values()))
        if halt and not self.halted:
            self.halted = st["halted"] = True
            self.log(f"DAILY-LOSS HALT: {why} - flattening and blocking new entries", "err")
            for p in self.active_pairs():
                name = D.pair_name(p)
                ps, v = st["pairs"][name], views.get(name)
                r = next((x for x in rows if x["pair"] == name), None)
                if ps["side"] and v is not None and r is not None:
                    cL, cB = leg_costs(p, c)
                    if self.trade_pair(p, name, ps, 0, v, w[name], (cL + v.beta * cB) / 1e4, r):
                        ps["pos"], ps["age"] = 0, 0
                        r["pos_after"], r["note"], r["position"] = 0, "daily-loss halt", pair_position_text(name, ps)
        st["last_j"] = j
        st["shadow_pnl"] = sum(x.get("pnl_usd", 0.0) for x in st["pairs"].values())
        self.save_state(st)
        try:
            self.append_log("decisions", rows)
        except Exception as e:                                                           # noqa: BLE001 - a log that cannot be written must not undo a decided step
            self.log(f"decisions log write failed ({type(e).__name__}: {str(e)[:80]})", "err")
        return rows

    # ── trading a pair ───────────────────────────────────────────────────────────
    def size_pair(self, name: str, wt: float, v: PairView) -> tuple[Optional[tuple], str]:
        """Whole-share sizes for both legs, scaled down to fit the exposure caps (each leg valued at ITS OWN price). Returns (sizes or None, reason). The hedge leg is rounded to the nearest share, so the
        caps are re-checked with the rounded sizes and the size is stepped down until they hold: no leg or gross overshoot by a share."""
        live = self.live
        eq = self.equity_now()
        N_L = self.deploy * self.G * wt / (1.0 + v.beta)
        hedge = lambda n: int(round(v.beta * n * v.L_px / v.B_px))
        n_L = int(N_L // v.L_px)
        n_B = hedge(n_L)
        leg_cap = live.max_leg_nav * eq
        gross_cap = min(live.max_gross_nav * eq, live.max_gross_x * min(self.deploy, eq))
        room = gross_cap - sum(x.get("gross_usd", 0.0) for k, x in self.state["pairs"].items() if x["side"] and k != name)
        for _ in range(200):
            leg, gross = max(n_L * v.L_px, n_B * v.B_px), n_L * v.L_px + n_B * v.B_px
            if leg <= leg_cap and gross <= room:
                break
            f = min(leg_cap / leg if leg > 0 else 1.0, room / gross if gross > 0 else 1.0)
            if f <= 0.02:
                return None, f"entry skipped: the exposure caps leave no room (gross cap ${gross_cap:,.0f}, open ${gross_cap - room:,.0f}, leg cap ${leg_cap:,.0f})"
            n_L = min(int(n_L * f), n_L - 1)
            n_B = hedge(n_L)
            if n_L < 1:
                break
        else:
            return None, "entry skipped: the exposure caps cannot be met with whole shares"
        if n_L < 1 or n_B < 1 or n_L * v.L_px < live.min_leg_usd or n_B * v.B_px < live.min_leg_usd:
            return None, "entry skipped: legs below the minimum ticket or zero shares at this size"
        err = abs(n_B * v.B_px / (n_L * v.L_px) - v.beta) / v.beta
        if err > 0.10:
            return None, f"entry skipped: whole shares cannot hold the hedge ratio at this size (off by {err:.0%}); use more capital"
        return (n_L, n_B, n_L * v.L_px), ""

    def open_lp_orders(self, symbols) -> list:
        """Orders of this engine still open at the broker for these symbols (a retry must not stack a second order on top)."""
        try:
            self._throttle()
            return [o for o in self.tc.get_orders() if str(o.client_order_id).startswith(PREFIX + "-") and o.symbol in symbols]
        except Exception as e:                                                           # noqa: BLE001
            self.log(f"could not list open orders ({type(e).__name__}) - treating the symbols as busy", "warn")
            return [None]

    def _legs_text(self, legs: list, fills: dict, refs: dict) -> str:
        """'BUY 357 TQQQ @ 70.12  +  SELL 107 QQQ @ 612.40' from the legs that were SENT (symbol, shares, is_buy) and, when ARMED, the fill prices; a dry run (or a fill without a price) shows the signal bar's close and says so."""
        parts, est = [], False
        for sym, q, buy in legs:
            px = (fills.get(sym) or {}).get("px") if self.live.armed else None
            if not px:
                px, est = refs.get(sym), True
            parts.append(f"{'BUY' if buy else 'SELL'} {int(q):,} {sym} @ {float(px):,.2f}")
        tail = "  [dry run: prices are the signal bar's close, not fills]" if not self.live.armed else "  [no fill price reported: signal-bar close shown]" if est else ""
        return "  +  ".join(parts) + tail

    def trade_pair(self, pair: tuple, name: str, ps: dict, tgt: int, v: PairView, wt: float, unit_cost: float, row: dict) -> bool:
        """Move one pair from its ledger position to ``tgt`` (-1, 0, +1). Returns True when the ledger changed."""
        L, B = pair
        cur = ps["side"]
        if cur != 0 and tgt != 0 and cur != tgt:
            tgt = 0
        if cur == 0 and tgt != 0:                                                       # entry
            if self.kill_flag.is_set():
                row["note"] = "entry skipped: KILL was pressed"
                return False
            self._honour_watchdog()                                                      # an engine that woke up in the middle of a step: the watchdog may have flattened and halted it meanwhile
            if self.halted:
                row["note"] = "entry skipped: the account was flattened by the watchdog (or the loss halt tripped): no new risk today"
                return False
            sz, why = self.size_pair(name, wt, v)
            if sz is None:
                row["note"] = why
                return False
            n_L, n_B, N_L = sz
            if self.live.armed:
                bp, open_orders = self._pretrade_reads((L, B))
                if bp.get(L, 0) or bp.get(B, 0):
                    row["note"] = f"entry skipped: the broker already holds {L} {bp.get(L, 0):+.0f} / {B} {bp.get(B, 0):+.0f} - flatten or reconcile first"
                    return False
                if open_orders:
                    row["note"] = "entry skipped: an order of this engine is still open for these symbols"
                    return False
            legs = [(L, n_L, tgt > 0), (B, n_B, tgt < 0)]                                # long spread: buy L, sell B - both legs go out at the same moment
            self._honour_watchdog()                                                      # ... and again after the pre-trade reads, right before the orders
            if self.halted:
                row["note"] = "entry skipped: the account was flattened by the watchdog (or the loss halt tripped): no new risk today"
                return False
            ps["attempts"] = ps.get("attempts", 0) + 1                                   # one id per ATTEMPT: a failed entry must not share its id with the next one
            trip = f"{self.state['date']}-{name.replace('/', '')}-{ps['attempts']}"
            ctx = dict(action="ENTER", trip=trip, j=v.j, s=v.s, decided_at=row.get("wall"))
            fills = self.send(name, legs, row, refs={L: v.L_px, B: v.B_px}, ctx=ctx)
            if fills is None:
                self._clear_pending()
                return False
            ps.update(side=tgt, trip=trip, shares_L=n_L * (1 if tgt > 0 else -1), shares_B=n_B * (-1 if tgt > 0 else 1), entry_time=row["time"], entry_s=v.s, N_L=N_L, n_trades=ps["n_trades"] + 1,
                      gross_usd=n_L * v.L_px + n_B * v.B_px, entry_px={L: float(fills.get(L, {}).get("px") or v.L_px), B: float(fills.get(B, {}).get("px") or v.B_px)})
            ps["pnl_usd"] = ps.get("pnl_usd", 0.0) - 1.0 * unit_cost * N_L                 # shadow entry cost
            row.update(action=f"ENTER {'long' if tgt > 0 else 'short'} spread: " + self._legs_text(legs, fills, {L: v.L_px, B: v.B_px}), shares_L=ps["shares_L"], shares_B=ps["shares_B"], fills=fills)
            self._clear_pending()
            return True
        if cur != 0 and tgt == 0:                                                       # exit
            qL, qB = abs(ps["shares_L"]), abs(ps["shares_B"])
            if self.live.armed:                                                          # the broker is the truth: close what is really there, in the ledger's direction only
                bp, open_orders = self._pretrade_reads((L, B))
                qL = min(qL, abs(int(bp.get(L, 0)))) if np.sign(bp.get(L, 0)) == np.sign(ps["shares_L"]) else 0
                qB = min(qB, abs(int(bp.get(B, 0)))) if np.sign(bp.get(B, 0)) == np.sign(ps["shares_B"]) else 0
                if (qL, qB) != (abs(ps["shares_L"]), abs(ps["shares_B"])):
                    self.log(f"{name}: ledger {ps['shares_L']:+d}/{ps['shares_B']:+d} but the broker holds {bp.get(L, 0):+.0f}/{bp.get(B, 0):+.0f} - closing what is there", "warn")
                if open_orders:
                    row["note"] = "exit postponed: an order of this engine is still open for these symbols"
                    return False
            legs = [(s_, q_, b_) for s_, q_, b_ in ((L, qL, ps["shares_L"] < 0), (B, qB, ps["shares_B"] < 0)) if q_ > 0]
            ctx = dict(action="EXIT", trip=ps.get("trip") or f"{self.state['date']}-{name.replace('/', '')}-{ps.get('attempts') or ps['n_trades']}", j=v.j, s=v.s, decided_at=row.get("wall"))
            fills = self.send(name, legs, row, closing=True, refs={L: v.L_px, B: v.B_px}, ctx=ctx) if legs else {}
            if fills is None:
                self._clear_pending()
                return False
            ep = ps.get("entry_px") or {}
            if len(ep) == 2:                                                             # realised P&L of the fills (exit price = fill, else the signal price)
                px = {L: float(fills.get(L, {}).get("px") or v.L_px), B: float(fills.get(B, {}).get("px") or v.B_px)}
                ps["realized_usd"] = ps.get("realized_usd", 0.0) + ps["shares_L"] * (px[L] - ep[L]) + ps["shares_B"] * (px[B] - ep[B])
            ps["pnl_usd"] = ps.get("pnl_usd", 0.0) - 1.0 * unit_cost * ps["N_L"]
            what = self._legs_text(legs, fills, {L: v.L_px, B: v.B_px}) if legs else "nothing to close at the broker (the ledger was cleared)"
            if legs and (qL, qB) != (abs(ps["shares_L"]), abs(ps["shares_B"])):
                what += f"  (the ledger held {ps['shares_L']:+d} / {ps['shares_B']:+d})"
            row.update(action="EXIT: " + what, shares_L=-int(np.sign(ps["shares_L"])) * int(qL), shares_B=-int(np.sign(ps["shares_B"])) * int(qB), fills=fills)
            ps.update(side=0, shares_L=0, shares_B=0, entry_time=None, entry_s=None, gross_usd=0.0, entry_px={})
            self._clear_pending()
            return True
        return False

    def _live_submits(self) -> list:
        self._submit_threads = [t for t in self._submit_threads if t.is_alive()]
        return self._submit_threads

    def _resolve_unknown(self, name: str, symbols: list) -> None:
        """An entry whose leg is in doubt must not stay naked for hours: wait a few seconds for the late order to land, cancel this engine's open orders (an unknown order that is still open must not fill after the
        clean-up), then close what the two legs left at the broker (an orphan flatten of exactly these symbols). The pair stays BROKEN until someone has looked at it."""
        end = time.time() + self.live.unknown_resolve_wait_s
        for t in list(self._live_submits()):
            t.join(max(0.0, end - time.time()))
        time.sleep(max(0.0, min(1.0, end - time.time())))
        self.log(f"{name}: an order is in doubt - cancelling this engine's open orders and closing what the two legs left at the broker", "warn")
        try:
            self._cancel_lp_orders()
            self.flatten_orphans(only=set(symbols))
        except Exception as e:                                                           # noqa: BLE001
            self.log(f"{name}: the clean-up of the order in doubt failed ({type(e).__name__}: {str(e)[:80]}) - it stays on disk, close it by hand", "err")

    def _journal_close(self, action: str, pair: str, sym: str, qty: int, cid: str, order, t_submit_ns: int, err: Optional[str] = None) -> None:
        """Journal a close order sent by a flatten (KILL / end of day / session end) or the orphan flatten: qty > 0 means a long position is being sold."""
        self._journal([dict(date=self.state["date"], time=f"{pd.Timestamp.now(tz=NY):%H:%M:%S}", pair=pair, action=action, symbol=sym, side="SELL" if qty > 0 else "BUY", qty=abs(int(qty)),
                            status="rejected" if err else "sent", note=err, order_id=getattr(order, "id", None), client_order_id=cid, t_submit_ns=t_submit_ns, t_ack_ns=time.time_ns())])

    def _pretrade_reads(self, symbols) -> tuple[dict, list]:
        """The broker's positions and this engine's open orders, read at the same moment (two calls in parallel: the order must not wait for them one after the other)."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:
            fp, fo = ex.submit(self.broker_positions), ex.submit(self.open_lp_orders, symbols)
            return fp.result(), fo.result()

    def _clear_pending(self) -> None:
        """Forget the in-flight intent - unless one of its orders is UNKNOWN (it may be live at the broker): then it stays on disk, a restart refuses to start over it, and only a verified orphan flatten clears it."""
        if getattr(self, "state", None) is not None and self.state.get("pending") and not self.state["pending"].get("unknown"):
            self.state["pending"] = None
            self.save_state(self.state)

    @staticmethod
    def _ambiguous(e: Exception) -> bool:
        """True when the order may have reached the broker although the call raised: no HTTP answer at all (timeout, reset), 5xx, 408, 429, or the 'client_order_id must be unique' answer to alpaca-py's own retry of
        a POST that had in fact succeeded. A plain 4xx (insufficient buying power, not shortable ...) is a real rejection."""
        msg = str(e).lower()
        if "client_order_id must be unique" in msg or "client order id" in msg and "unique" in msg:
            return True
        code = getattr(e, "status_code", None)
        return code is None or code >= 500 or code in (408, 429)

    def _lookup_by_client_id(self, cid: str) -> tuple[str, Optional[object]]:
        """('found', order) | ('absent', None) | ('unknown', None): did the order with this client id reach the broker?"""
        try:
            self._throttle()
            return "found", self.tc.get_order_by_client_id(cid)
        except Exception as e:                                                           # noqa: BLE001
            code = getattr(e, "status_code", None)
            if code == 404 or "not found" in str(e).lower():
                return "absent", None
            return "unknown", None

    def _submit_all(self, reqs: list) -> list:
        """Send every leg at the SAME moment: one thread per leg, released together by a barrier, so the legs reach the broker within about a millisecond of each other instead of one round trip apart (the
        second leg used to wait for the first one's acknowledgement). reqs = [(symbol, qty, is_buy, client_order_id)]. Returns one dict per leg (order id or error, local submit / ack times in ns); a leg whose
        call has not returned after ``submit_timeout_s`` comes back as None (its order may be live: the caller treats it as unknown)."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        n = len(reqs)
        res: list = [None] * n
        gate = threading.Barrier(n) if n > 1 else None

        def one(i: int) -> None:
            sym, q, buy, cid = reqs[i]
            r = dict(sym=sym, cid=cid, oid=None, err=None, t_submit_ns=None, t_ack_ns=None)
            try:
                req = MarketOrderRequest(symbol=sym, qty=int(q), side=OrderSide.BUY if buy else OrderSide.SELL, time_in_force=TimeInForce.DAY, client_order_id=cid)
                if gate is not None:
                    gate.wait(timeout=5.0)                                               # a leg that cannot be built stops the others too: nothing is sent
                r["t_submit_ns"] = time.time_ns()
                o = self.tc.submit_order(req)
                r["t_ack_ns"] = time.time_ns()
                r["oid"] = o.id
            except Exception as e:                                                       # noqa: BLE001
                r["t_ack_ns"] = time.time_ns()
                r["err"] = f"{type(e).__name__}: {str(e)[:100]}"
                if self._ambiguous(e):                                                   # the call raised, but the order may exist (a retried POST, a timeout after the broker accepted it): ask - three times
                    for attempt in range(3):
                        found, o = self._lookup_by_client_id(cid)
                        if found != "unknown":
                            break
                        time.sleep(self.live.lookup_retry_s)
                    if found == "found":
                        r["oid"], r["err"], r["recovered"] = o.id, None, True
                    elif found == "unknown":
                        r["unknown"] = True
                        r["err"] += "; the order could not be looked up by its client id"
            res[i] = r
        ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(n)]
        self._submit_threads = [t for t in self._submit_threads if t.is_alive()] + ths
        for t in ths:
            t.start()
        end = time.time() + self.live.submit_timeout_s
        for t in ths:
            t.join(max(0.0, end - time.time()))
        return res

    def send(self, name: str, legs: list, row: dict, closing: bool = False, refs: Optional[dict] = None, ctx: Optional[dict] = None) -> Optional[dict]:
        """legs = [(symbol, qty, is_buy)]. DRY RUN: nothing is sent (the shadow book assumes the fills of the backtest). ARMED: market orders for BOTH legs sent at the same moment, the intent written to the state
        file BEFORE the first order (a crash in between is found on restart), both legs confirmed, a failed leg unwinds the other (a KILL pressed while an entry is in flight unwinds it). Every order - also a
        rejected or unknown one and every unwind - is journaled with its decision context and timestamps. Returns the fills, or None when the pair is in doubt (it is then marked BROKEN)."""
        if not self.live.armed:
            return {s: dict(qty=q, side="BUY" if b else "SELL", status="dry run") for s, q, b in legs}
        if self.live.feed == "iex" and not self.live.accept_iex:
            raise RuntimeError("ARMED on the IEX feed is refused: the IEX s-scores of these pairs correlate only 0.2-0.57 with the consolidated-tape scores the strategy was tested on (reports/feed_test.log)")
        if self.tc is None:
            self.connect()
        if self.acct.get("blocked"):
            raise RuntimeError("trading is blocked on the account")
        ctx = ctx or {}
        out, sent, meta = {}, [], {}
        st = self.state
        st["pending"] = dict(pair=name, closing=closing, legs=[[s, int(q), bool(b)] for s, q, b in legs], ts=str(pd.Timestamp.now()))
        self.save_state(st)                                                              # write-ahead: if the process dies from here on, the next start sees an order in flight
        reqs = []
        for sym, q, buy in legs:
            st["orders"] = st.get("orders", 0) + 1
            cid = f"{PREFIX}-{pd.Timestamp(st['date']):%Y%m%d}-{name.replace('/', '')}-{int(time.time() * 1000)}-{st['orders']}"
            reqs.append((sym, int(q), bool(buy), cid[:120]))
        if self.kill_flag.is_set() and not closing:                                      # KILL before anything was sent: send nothing
            for sym, q, buy, cid in reqs:
                out[sym] = dict(qty=q, status="rejected", note="KILL was pressed")
            results = [None] * len(reqs)
        else:
            results = self._submit_all(reqs)
        for (sym, q, buy, cid), r in zip(reqs, results):
            meta[sym] = dict(order_id=(r or {}).get("oid"), client_order_id=cid, t_submit_ns=(r or {}).get("t_submit_ns"), t_ack_ns=(r or {}).get("t_ack_ns"), side="BUY" if buy else "SELL")
            if r is None:
                if sym not in out:
                    out[sym] = dict(qty=0, status="unknown", note="the submit call did not return - the order may be live")
                continue
            if r.get("unknown"):
                out[sym] = dict(qty=0, status="unknown", note=r["err"])
            elif r["err"]:
                out[sym] = dict(qty=q, status="rejected", note=r["err"])
            else:
                if r.get("recovered"):
                    self.log(f"{name} {sym}: the submit call raised but the order exists at the broker (found by its client id): continuing with it", "warn")
                sent.append((sym, q, buy, r["oid"]))
        t0 = time.time()
        pending, errs = {s: oid for s, q, b, oid in sent}, {}
        while pending and time.time() - t0 < self.live.fill_wait_s:
            self._heartbeat()
            for sym, oid in list(pending.items()):
                try:
                    o = self.tc.get_order_by_id(oid)
                except Exception as e:                                                   # noqa: BLE001 - an API error while polling must not leave the ledger blind
                    errs[sym] = errs.get(sym, 0) + 1
                    if errs[sym] >= 3:
                        out[sym] = dict(qty=0, status="unknown", note=f"{type(e).__name__}: {str(e)[:80]}")
                        pending.pop(sym)
                    continue
                stt = str(getattr(o.status, "value", o.status))
                if stt in ("filled", "canceled", "expired", "rejected"):
                    out[sym] = dict(qty=float(o.filled_qty or 0), side=str(getattr(o.side, "value", o.side)), status=stt, px=float(o.filled_avg_price) if o.filled_avg_price else None,
                                    submitted_at=str(getattr(o, "submitted_at", "") or ""), filled_at=str(getattr(o, "filled_at", "") or ""), t_seen_ns=time.time_ns())
                    pending.pop(sym)
            if pending:
                time.sleep(0.4)
        for sym in pending:
            try:
                self.tc.cancel_order_by_id(pending[sym])
            except Exception:                                                            # noqa: BLE001
                pass
            try:
                o = self.tc.get_order_by_id(pending[sym])
                out[sym] = dict(qty=float(o.filled_qty or 0), status="timeout", px=float(o.filled_avg_price) if o.filled_avg_price else None,
                                submitted_at=str(getattr(o, "submitted_at", "") or ""), filled_at=str(getattr(o, "filled_at", "") or ""), t_seen_ns=time.time_ns())   # a partial fill is unwound with the rest
            except Exception:                                                            # noqa: BLE001
                out[sym] = dict(qty=0, status="unknown", note="timeout and the order could not be read back")
        ok = all(o.get("status") == "filled" for o in out.values()) and len(out) == len(legs)
        killed_in_flight = self.kill_flag.is_set() and not closing and ok
        if killed_in_flight:
            ok = False                                                                   # both legs filled but KILL was pressed meanwhile: the entry is undone
            row["note"] = "KILL was pressed while the entry was in flight: unwinding it"
        # the journal: every order, whatever became of it (nothing in here may raise: an unwind may still be needed below)
        jrows = []
        try:
            jrows = self._journal_rows(st, name, legs, out, meta, row, ctx, refs, closing)
        except Exception as e:                                                           # noqa: BLE001
            self.log(f"could not build the journal rows ({type(e).__name__}: {str(e)[:80]})", "err")
        self._journal(jrows)
        if not ok:
            return self._send_failed(st, name, legs, sent, out, row, ctx, closing, killed_in_flight)
        return out

    def _journal_rows(self, st, name, legs, out, meta, row, ctx, refs, closing) -> list:
        jrows = []
        for sym, q, buy in legs:
            o = out.get(sym, {})
            r = dict(date=st["date"], time=row.get("time"), pair=name, action=ctx.get("action", "CLOSE" if closing else "ENTER"), trip=ctx.get("trip"), j=ctx.get("j"), s=ctx.get("s"),
                     decided_at=ctx.get("decided_at"), symbol=sym, side=meta.get(sym, {}).get("side", "BUY" if buy else "SELL"), qty=o.get("qty", q), px=o.get("px"), status=o.get("status"), note=o.get("note"),
                     order_id=meta.get(sym, {}).get("order_id"), client_order_id=meta.get(sym, {}).get("client_order_id"), submitted_at=o.get("submitted_at"), filled_at=o.get("filled_at"),
                     t_submit_ns=meta.get(sym, {}).get("t_submit_ns"), t_ack_ns=meta.get(sym, {}).get("t_ack_ns"), t_seen_ns=o.get("t_seen_ns"))
            ref, px = (refs or {}).get(sym), o.get("px")
            if ref and px:                                                               # slippage against the bar close the signal was computed on, positive = worse than the signal price
                r["ref"], r["slip_bp"] = ref, (1 if r["side"].lower().startswith("b") else -1) * (px - ref) / ref * 1e4
            elif ref:
                r["ref"] = ref
            jrows.append(r)
        return jrows

    def _send_failed(self, st, name, legs, sent, out, row, ctx, closing, killed_in_flight):
        if True:
            unknown = any(o.get("status") == "unknown" for o in out.values())
            self.log(f"{name}: order problem {out} - " + ("a leg's state is UNKNOWN: nothing is unwound (unwinding one leg would leave the other naked); reconcile and use 'flatten unexplained positions'" if unknown
                                                          else "unwinding any filled leg"), "err")
            unw = [(sym, int(float(out.get(sym, {}).get("qty") or 0)), not buy) for sym, q, buy, oid in sent if float(out.get(sym, {}).get("qty") or 0) > 0 and not closing and not unknown]
            if unw:
                ureqs = [(sym, q, buy, f"{PREFIX}-UNW-{sym}-{int(time.time() * 1000)}") for sym, q, buy in unw]
                for (sym, q, buy, cid), r in zip(ureqs, self._submit_all(ureqs)):        # both unwinds at the same moment too
                    failed = r is None or r["err"]
                    self._journal([dict(date=st["date"], time=row.get("time"), pair=name, action="UNWIND", trip=ctx.get("trip"), j=ctx.get("j"), symbol=sym, side="BUY" if buy else "SELL", qty=q,
                                        status="rejected" if failed else "sent", note=(r or {}).get("err") if r else "submit call did not return", order_id=(r or {}).get("oid"), client_order_id=cid,
                                        t_submit_ns=(r or {}).get("t_submit_ns"), t_ack_ns=(r or {}).get("t_ack_ns"))])
                    if failed:
                        self.log(f"UNWIND of {sym} FAILED ({(r or {}).get('err', 'no answer')}): close it by hand", "err")
                        st.setdefault("alarms", []).append(f"unwind of {sym} failed at {pd.Timestamp.now():%H:%M:%S}")
            if any(o.get("status") == "unknown" for o in out.values()):
                st.setdefault("alarms", []).append(f"{name}: order state unknown at {pd.Timestamp.now():%H:%M:%S} - reconcile")
                if st.get("pending"):
                    st["pending"]["unknown"] = True                                      # kept on disk: a restart refuses to start over an order that may be live
                    self.save_state(st)
            self.state["pairs"][name]["broken"] = True
            row["note"] = row.get("note") if killed_in_flight else f"orders failed: {out}"
            if unknown and not closing and self.live.resolve_unknown:
                self._resolve_unknown(name, [s_ for s_, q, b in legs])
            return None

    # ── emergency / end of day ───────────────────────────────────────────────────
    def _latest_state(self) -> dict:
        """The newest ledger of this mode whatever its date (KILL must work on yesterday's leftovers too)."""
        path = state_path(mode_name(self.live))
        if os.path.exists(path):
            try:
                st = json.load(open(path, encoding="utf-8"))
                blank = self.blank_state(st.get("date", clock_now().tz_localize(None)))
                for k, v in blank["pairs"].items():
                    st["pairs"].setdefault(k, v)
                    for f, dv in v.items():
                        st["pairs"][k].setdefault(f, dv)
                self._seed_entry(st)
                return st
            except (OSError, ValueError) as e:
                raise RuntimeError(f"state file {path} is unreadable ({e}); fix or move it") from e
        return self.blank_state(clock_now().tz_localize(None))

    def flatten(self, reason: str = "KILL", persist: Optional[bool] = None) -> list[dict]:
        """``persist`` (default: the end-of-day and session-end flattens) keeps retrying with the back-off until the last useful moment of the session (flatten_until) instead of a fixed number of attempts.
        Only ONE flatten runs at a time: KILL goes ahead of a hung step after ``kill_lock_wait_s`` but never beside another flatten (two closers on one ledger sold the same shares twice); a second caller
        waits for the running one, then makes one more pass over what is left."""
        if reason.upper().startswith("KILL"):
            self.kill_flag.set()                                                          # a step in flight stops entering at once (trade_pair checks the flag)
        if persist is None:
            persist = reason in ("end of day", "session end")
        if not self._flatten_guard.acquire(blocking=False):
            self.log(f"{reason}: a flatten is already running - not starting a second closer beside it, waiting for it to finish", "warn")
            if not self._flatten_guard.acquire(timeout=self.live.flatten_join_s):
                self.log(f"{reason}: the running flatten did not finish within {self.live.flatten_join_s:.0f}s - this call sent nothing", "err")
                return [dict(symbol="*", status="a flatten is already running - this call sent nothing")]
        try:
            if not self._xlock_acquire(self.live.flatten_join_s):
                self.log(f"{reason}: another PROCESS is flattening this account (reports/levpairs_flatten_{mode_name(self.live)}.lock) - this call sent nothing", "err")
                return [dict(symbol="*", status="a flatten is running in another process - this call sent nothing")]
            try:
                got = self._lock.acquire(timeout=self.live.kill_lock_wait_s)              # KILL must not wait forever behind a hung step
                try:
                    return self._flatten(reason, persist)
                finally:
                    if got:
                        self._lock.release()
            finally:
                self._xlock_release()
        finally:
            self._flatten_guard.release()

    def _closes_allowed(self) -> tuple[bool, str]:
        """Close orders go out only while the market is open: a market order sent after the close queues for the next open and can flip a position that was closed in the meantime. The broker's clock decides
        (it knows holidays and early closes); if it cannot be read the local clock must say a weekday between 09:30:00 and 15:59:30 ET."""
        try:
            self._throttle()
            if not bool(self.tc.get_clock().is_open):
                return False, "the market is closed - a market order now would queue for the next open and could flip a position closed meanwhile"
            return True, ""
        except Exception as e:                                                           # noqa: BLE001
            now = clock_now()
            t = now.strftime("%H:%M:%S")
            if now.weekday() < 5 and "09:30:00" <= t <= "15:59:30":
                self.log(f"could not read the market clock ({type(e).__name__}) - the local time {t} ET is inside weekday regular hours, sending the close", "warn")
                return True, ""
            return False, f"could not read the market clock ({type(e).__name__}) and the local time {t} ET is outside weekday regular hours (09:30:00-15:59:30) - a market order now could queue for the next open"

    def _cancel_lp_orders(self, prefix: str = PREFIX + "-") -> int:
        n = 0
        try:
            self._throttle()
            for o in self.tc.get_orders():
                if str(o.client_order_id).startswith(prefix):
                    try:
                        self._throttle()
                        self.tc.cancel_order_by_id(o.id)
                        n += 1
                    except Exception:                                                    # noqa: BLE001
                        pass
        except Exception:                                                                # noqa: BLE001
            pass
        return n

    def _past_cutoff(self) -> bool:
        """True once the session of the ledger is today's and the clock is past the last useful moment to close (market orders after the close do not fill)."""
        st = getattr(self, "state", None) or {}
        now = clock_now()
        return str(now.date()) == st.get("date") and now.strftime("%H:%M:%S") > self.live.flatten_until

    def _lp_open_symbols(self) -> set:
        self._throttle()
        return {o.symbol for o in self.tc.get_orders() if str(o.client_order_id).startswith(PREFIX + "-")}

    def _flatten(self, reason: str, persist: bool = False) -> list[dict]:
        """Cancel the engine's open orders and close every ledger position, retrying with a back-off, then RE-READ THE BROKER: a pair leaves the ledger only when its closing fills are visible, otherwise it stays,
        marked BROKEN, and an alarm is raised (DRY RUN: clears the shadow ledger). Never touches anything outside the six ETFs of the ledger, never closes more than the broker holds (the position and the open
        orders of each symbol are re-read right before its order), never stacks a second order on a symbol that still has one open, never sends while the market is closed."""
        st = getattr(self, "state", None) or self._latest_state()
        res = []
        if not self.live.armed:
            for name, ps in st["pairs"].items():
                if ps.get("side"):
                    res += [dict(symbol=s, qty=-q, status="shadow position cleared") for s, q in zip(name.split("/"), (ps["shares_L"], ps["shares_B"]))]
                    ps.update(side=0, shares_L=0, shares_B=0, pos=0, age=0, entry_time=None, gross_usd=0.0, entry_px={})
            st["pending"] = None
            self.state = st
            self.save_state(st)
            self.log(f"{reason}: {len(res)} actions", "warn")
            return res
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        self.state = st
        if self.tc is None:
            try:
                self.connect()
            except Exception as e:                                                       # noqa: BLE001
                self.log(f"{reason}: cannot connect ({type(e).__name__}) - NOTHING WAS CLOSED, close by hand", "err")
                return [dict(symbol="*", status=f"FAILED to connect: {type(e).__name__}")]
        attempt = -1
        while True:
            attempt += 1
            todo = [n for n, ps in st["pairs"].items() if ps.get("side")]
            if not todo:
                break
            if attempt and self._past_cutoff():
                break
            if attempt >= max(1, self.live.flatten_attempts) and not (persist and str(clock_now().date()) == st.get("date")):
                break
            if attempt:
                time.sleep(self.live.flatten_backoff_s)
            self._heartbeat()
            ok, why = self._closes_allowed()
            if not ok:
                res.append(dict(symbol="*", status=why))
                self._cancel_lp_orders()                                                 # nothing of ours may queue for the open
                self.log(f"{reason}: {why} - nothing sent, the ledger keeps the pair(s); press KILL again during market hours", "err")
                break
            try:
                self._throttle()
                for o in self.tc.get_orders():
                    if str(o.client_order_id).startswith(PREFIX + "-"):
                        try:
                            self._throttle()
                            self.tc.cancel_order_by_id(o.id)
                        except Exception:                                                # noqa: BLE001
                            pass
            except Exception as e:                                                       # noqa: BLE001
                res.append(dict(symbol="*", status=f"attempt {attempt + 1}: {type(e).__name__}: {str(e)[:60]}"))
                continue
            expected, owner = {}, {}
            for name in todo:
                ps = st["pairs"][name]
                L, B = name.split("/")
                for sym, q in ((L, ps["shares_L"]), (B, ps["shares_B"])):
                    owner[sym] = name
                    try:                                                                 # re-read right before the order: the position may have changed since the top of the attempt
                        busy_now = sym in self._lp_open_symbols()
                        bq = int(self.broker_positions().get(sym, 0))
                    except Exception as e:                                               # noqa: BLE001
                        expected[sym] = None
                        res.append(dict(symbol=sym, qty=0, status=f"could not re-read the broker: {type(e).__name__}"))
                        continue
                    if busy_now:
                        expected[sym] = None
                        res.append(dict(symbol=sym, qty=0, status="an order of this engine is still open - not stacking another"))
                        continue
                    qq = int(math.copysign(min(abs(q), abs(bq)), q)) if bq and np.sign(bq) == np.sign(q) else 0
                    expected[sym] = bq - qq
                    if qq == 0:
                        res.append(dict(symbol=sym, qty=0, status="nothing to close at the broker"))
                        continue
                    cid, t_sub = f"{PREFIX}-KILL-{sym}-{int(time.time() * 1000)}", time.time_ns()
                    try:
                        self._throttle()
                        o = self.tc.submit_order(MarketOrderRequest(symbol=sym, qty=abs(qq), side=OrderSide.SELL if qq > 0 else OrderSide.BUY, time_in_force=TimeInForce.DAY, client_order_id=cid))
                        self._journal_close(reason, name, sym, qq, cid, o, t_sub)
                        res.append(dict(symbol=sym, qty=-qq, status="sent"))
                    except Exception as e:                                               # noqa: BLE001
                        self._journal_close(reason, name, sym, qq, cid, None, t_sub, err=f"{type(e).__name__}: {str(e)[:80]}")
                        res.append(dict(symbol=sym, qty=-qq, status=f"FAILED {type(e).__name__}: {str(e)[:80]}"))
            unresolved = {s for s, e in expected.items() if e is None}
            waiting = {s for s, e in expected.items() if e is not None}
            deadline = time.time() + self.live.flatten_verify_s
            while waiting and time.time() < deadline:                                    # verify against the broker
                self._heartbeat()
                try:
                    now_pos = self.broker_positions()
                    waiting = {s for s in waiting if int(now_pos.get(s, 0)) != expected[s]}
                except Exception:                                                        # noqa: BLE001
                    pass
                if waiting:
                    time.sleep(self.live.flatten_poll_s)
            bad = {owner[s] for s in (unresolved | waiting)}
            for name in todo:
                if name not in bad:
                    st["pairs"][name].update(side=0, shares_L=0, shares_B=0, pos=0, age=0, entry_time=None, gross_usd=0.0, entry_px={})
            if not bad:
                break
        left = [n for n, ps in st["pairs"].items() if ps.get("side")]
        for name in left:
            st["pairs"][name]["broken"] = True
            st.setdefault("alarms", []).append(f"{name} NOT FLAT after {reason} at {pd.Timestamp.now():%H:%M:%S}")
            self.log(f"{reason}: {name} is NOT confirmed flat at the broker - it stays in the ledger (BROKEN): close it by hand", "err")
        if left and (self._past_cutoff() or not self._closes_allowed()[0]):
            self._cancel_lp_orders()                                                     # a close that did not fill must not queue for the next open
        st["pending"] = None if (not left and not (st.get("pending") or {}).get("unknown")) else st.get("pending")
        self.state = st
        self.save_state(st)
        self.log(f"{reason}: {len(res)} actions" + (f", {len(left)} pair(s) NOT flat" if left else ", broker confirmed flat"), "err" if left else "warn")
        return res

    def _excess(self, orph: dict) -> dict:
        """Shares to TRADE to remove what the ledger does not explain - never to move toward the ledger. Same sign and the broker holds more: only the excess. Opposite sign, or the ledger holds nothing:
        everything the broker holds. Broker flat, or holding no more than the ledger: nothing (a position that is missing is not recreated)."""
        out = {}
        for sym, (lg, bk) in orph.items():
            if bk == 0:
                continue
            if lg != 0 and np.sign(bk) == np.sign(lg):
                if abs(bk) > abs(lg):
                    out[sym] = bk - lg
            else:
                out[sym] = bk
        return out

    def flatten_orphans(self, only: Optional[set] = None) -> list[dict]:
        """Close positions in the six ETFs that the ledger does not explain (the broker's excess over the ledger, see _excess), without stacking on an open order, verified against the broker. For a DEDICATED
        account only: it closes anything held in these six symbols beyond the ledger, whoever put it there. It never opens or enlarges a position, never runs beside a flatten, and sends nothing while the
        market is closed or after the last useful moment of the session."""
        if not self.live.armed:
            return []
        if not self._flatten_guard.acquire(blocking=False):
            self.log("unexplained positions: a flatten is running - not starting a second closer beside it", "warn")
            return [dict(symbol="*", status="a flatten is running - nothing sent")]
        try:
            if not self._xlock_acquire(5.0):
                self.log("unexplained positions: another process is flattening this account - nothing sent", "warn")
                return [dict(symbol="*", status="a flatten is running in another process - nothing sent")]
            try:
                return self._flatten_orphans(only)
            finally:
                self._xlock_release()
        finally:
            self._flatten_guard.release()

    def _flatten_orphans(self, only: Optional[set] = None) -> list[dict]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        res = []
        if not self._lock.acquire(timeout=self.live.kill_lock_wait_s):                   # not an emergency: a decision step is running, and an order may be in flight (the first leg of an entry is
            self.log("unexplained positions: a decision step is running (an order may be in flight) - nothing sent, press again between bars", "warn")      # on the way in; closing it now leaves the second as a naked hedge)
            return [dict(symbol="*", status="a step is running - nothing sent")]
        got = True
        try:
            if getattr(self, "state", None) is None:
                self.state = self._latest_state()
            if self.tc is None:
                self.connect()
            for attempt in range(max(1, self.live.flatten_attempts)):
                self._heartbeat()
                if self._past_cutoff():
                    res.append(dict(symbol="*", status="past the last useful moment of the session - nothing sent"))
                    break
                ok, why = self._closes_allowed()
                if not ok:
                    res.append(dict(symbol="*", status=why))
                    self.log(f"unexplained positions: {why} - nothing sent", "err")
                    break
                try:
                    todo = {k: v for k, v in self._excess(self.orphans()).items() if only is None or k in only}
                    if not todo:
                        break
                except Exception as e:                                                   # noqa: BLE001
                    res.append(dict(symbol="*", status=f"attempt {attempt + 1}: {type(e).__name__}"))
                    time.sleep(self.live.flatten_backoff_s)
                    continue
                expected = {}
                for sym in list(todo):
                    try:                                                                 # re-read right before each order
                        held = int(self.broker_positions().get(sym, 0))
                        qq = self._excess({sym: (int(self.ledger_shares().get(sym, 0)), held)}).get(sym, 0)
                        busy = sym in self._lp_open_symbols()
                    except Exception as e:                                               # noqa: BLE001
                        res.append(dict(symbol=sym, qty=0, status=f"could not re-read the broker: {type(e).__name__}"))
                        continue
                    if not qq:
                        continue
                    if busy:
                        res.append(dict(symbol=sym, qty=0, status="an order of this engine is still open - not stacking another"))
                        continue
                    expected[sym] = held - int(qq)
                    cid, t_sub = f"{PREFIX}-ORPH-{sym}-{int(time.time() * 1000)}", time.time_ns()
                    try:
                        self._throttle()
                        o = self.tc.submit_order(MarketOrderRequest(symbol=sym, qty=abs(int(qq)), side=OrderSide.SELL if qq > 0 else OrderSide.BUY, time_in_force=TimeInForce.DAY, client_order_id=cid))
                        self._journal_close("ORPHAN", "-", sym, int(qq), cid, o, t_sub)
                        res.append(dict(symbol=sym, qty=-int(qq), status="sent"))
                    except Exception as e:                                               # noqa: BLE001
                        self._journal_close("ORPHAN", "-", sym, int(qq), cid, None, t_sub, err=f"{type(e).__name__}: {str(e)[:80]}")
                        res.append(dict(symbol=sym, qty=-int(qq), status=f"FAILED {type(e).__name__}: {str(e)[:80]}"))
                        expected.pop(sym, None)
                deadline = time.time() + self.live.flatten_verify_s
                while expected and time.time() < deadline:
                    self._heartbeat()
                    try:
                        now_pos = self.broker_positions()
                        expected = {s: e for s, e in expected.items() if int(now_pos.get(s, 0)) != e}
                    except Exception:                                                    # noqa: BLE001
                        pass
                    if expected:
                        time.sleep(self.live.flatten_poll_s)
                if not {k for k in self._excess(self.orphans()) if only is None or k in only}:
                    break
                time.sleep(self.live.flatten_backoff_s)
            self._cancel_lp_orders(PREFIX + "-ORPH-")                                    # a close that did not fill must not queue for the next open
            left = {k: v for k, v in self._excess(self.orphans()).items() if only is None or k in only}
            if not left and not self._live_submits():                                    # nothing unexplained is left at the broker (in the symbols asked for) and no submit call is still out: whatever order was in flight is accounted for
                self.state["pending"] = None
            self.save_state(self.state)
            self.log("unexplained positions closed" if not left else f"unexplained positions REMAIN: {left}", "warn" if not left else "err")
        finally:
            if got:
                self._lock.release()
        return res

    def sync_broker_day(self, date, write: bool = True) -> dict:
        """After the close: the BROKER's record of the day - every order of this engine (client id LP-...) with its timestamps and fill, and the execution-level fills (account activities). The journal can be
        incomplete (a crash, an unknown leg, an order placed by the flatten of another process); this record is what the comparison trusts. Read-only. Returns {'orders': DataFrame, 'fills': DataFrame}."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        if self.tc is None:
            self.connect()
        d0 = pd.Timestamp(date).normalize().tz_localize(NY)
        kw = dict(status=QueryOrderStatus.ALL, after=d0.tz_convert("UTC").to_pydatetime(), until=(d0 + pd.Timedelta(days=1)).tz_convert("UTC").to_pydatetime(), limit=500)
        try:
            from alpaca.common.enums import Sort
            kw["direction"] = Sort.ASC
        except Exception:                                                                # noqa: BLE001
            pass
        val = lambda x: getattr(x, "value", x)
        txt = lambda x: None if x is None else str(x)
        self._throttle()
        listed = self.tc.get_orders(GetOrdersRequest(**kw))
        orders = [o for o in listed if str(o.client_order_id).startswith(PREFIX + "-")]
        orows = [dict(order_id=str(o.id), client_order_id=str(o.client_order_id), symbol=o.symbol, side=str(val(o.side)), qty=txt(o.qty), filled_qty=txt(o.filled_qty), filled_avg_price=txt(o.filled_avg_price),
                      status=str(val(o.status)), type=str(val(getattr(o, "order_type", getattr(o, "type", "")))), submitted_at=txt(getattr(o, "submitted_at", None)), filled_at=txt(getattr(o, "filled_at", None)),
                      created_at=txt(getattr(o, "created_at", None))) for o in orders]
        frows = []
        try:                                                                              # this alpaca-py has no method for activities: the REST endpoint through the client's own GET
            ids = {str(o.id) for o in orders}
            self._throttle()
            for a in self.tc.get("/account/activities/FILL", {"date": f"{d0:%Y-%m-%d}"}) or []:
                if str(a.get("order_id", "")) in ids:
                    frows.append(dict(order_id=str(a.get("order_id")), symbol=a.get("symbol"), side=a.get("side"), qty=txt(a.get("qty")), price=txt(a.get("price")), transaction_time=txt(a.get("transaction_time")),
                                      type=a.get("type"), cum_qty=txt(a.get("cum_qty")), leaves_qty=txt(a.get("leaves_qty"))))
        except Exception as e:                                                           # noqa: BLE001 - the order list alone is already the record
            self.log(f"could not read the fill activities ({type(e).__name__}: {str(e)[:80]}) - using the order list only", "warn")
        out = {"orders": pd.DataFrame(orows), "fills": pd.DataFrame(frows)}
        if write:
            for k, df in out.items():
                df.to_csv(os.path.join(D.REPORTS, f"levpairs_broker_{k}_{mode_name(self.live)}_{d0:%Y%m%d}.csv"), index=False)
        if len(listed) >= 500:
            self.log("the broker returned 500 orders for the day - the list may be cut off", "warn")
        if len(frows) >= 100:
            self.log("100 fill activities: the broker's first page - the list may be cut off (the comparison uses the order list)", "warn")
        return out

    def reconcile(self) -> pd.DataFrame:
        st = self.state if hasattr(self, "state") else self._latest_state()
        self.state = st
        led = self.ledger_shares()
        broker = self.broker_positions() if self.tc is not None else {}
        rows = [dict(symbol=s, ledger=led.get(s, 0), broker=int(broker.get(s, 0)), diff=int(broker.get(s, 0)) - led.get(s, 0)) for s in D.SYMBOLS]
        return pd.DataFrame(rows)

    def reset_broken(self, force: bool = False) -> None:
        """Re-enable pairs after a failure. An ARMED engine first reads the broker (it must be connected): if the broker still disagrees with the ledger the flags stay."""
        if not self._lock.acquire(timeout=self.live.kill_lock_wait_s):
            raise RuntimeError("a decision step is running - try again between bars")
        try:
            if self.live.armed and not force:
                if self.tc is None:
                    raise RuntimeError("cannot clear the BROKEN flags before the broker has been read: connect first (the flags are cleared only when the broker and the ledger agree)")
                orph = self.orphans()
                if orph:
                    raise RuntimeError(f"cannot clear the BROKEN flags: the broker and the ledger still differ {orph}. Flatten (KILL) or fix it by hand first")
            for ps in self.state["pairs"].values():
                ps["broken"] = False
            self.save_state(self.state)
        finally:
            self._lock.release()

    # ── single-instance lock ─────────────────────────────────────────────────────
    def _acquire_lock(self) -> None:
        path = os.path.join(D.REPORTS, f"levpairs_engine_{mode_name(self.live)}.lock")
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)               # atomic: two simultaneous starts cannot both win
                os.write(fd, f"{os.getpid()} {pd.Timestamp.now()}".encode())
                os.close(fd)
                self._lockfile = path
                return
            except FileExistsError:
                age = time.time() - os.path.getmtime(path) if os.path.exists(path) else 1e9
                if age < self.live.lock_stale_s:
                    raise RuntimeError(f"another {mode_name(self.live).upper()} engine looks to be running (heartbeat {age:.0f}s ago: {path}). Two engines on one ledger would double-send orders")
                try:
                    os.remove(path)                                                      # a dead engine's stale lock
                except OSError:
                    pass
        raise RuntimeError(f"could not take the engine lock {path}")

    def _heartbeat(self) -> None:
        if self._lockfile and os.path.exists(self._lockfile):
            os.utime(self._lockfile, None)

    # ── one closer at a time, across PROCESSES (the in-process guard cannot see the watchdog) ────────────────────────
    def _xlock_acquire(self, timeout: float) -> bool:
        """An OPERATING-SYSTEM byte-range lock on reports/levpairs_flatten_<mode>.lock (msvcrt.locking on Windows, flock elsewhere): it is released by the OS when the holder dies, so there is no 'is this lock stale?'
        guess, no deleting a file another process may hold, no race between two processes that find the same stale file. Waits up to ``timeout`` seconds."""
        path = os.path.join(D.REPORTS, f"levpairs_flatten_{mode_name(self.live)}.lock")
        end = time.time() + timeout
        while True:
            fd = None
            try:
                fd = os.open(path, os.O_CREAT | os.O_RDWR)
                _os_lock(fd)
                self._xlock_fd, self._xlock_held = fd, path
                try:
                    os.write(fd, f"{os.getpid()} {pd.Timestamp.now()}".encode())
                except OSError:
                    pass
                return True
            except OSError:                                                              # another process (or another engine here) holds it
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                if time.time() >= end:
                    return False
                time.sleep(0.25)

    def _xlock_release(self) -> None:
        if self._xlock_fd is not None:
            try:
                _os_unlock(self._xlock_fd)
            except OSError:
                pass
            try:
                os.close(self._xlock_fd)
            except OSError:
                pass
            try:
                os.remove(self._xlock_held)                                              # best effort: a waiting process may have the file open
            except OSError:
                pass
        self._xlock_fd = self._xlock_held = None

    def _release_lock(self) -> None:
        if self._lockfile and os.path.exists(self._lockfile):
            os.remove(self._lockfile)
        self._lockfile = None

    # ── a whole session ──────────────────────────────────────────────────────────
    def run_session(self, should_stop: Callable[[], bool] = lambda: False, on_rows: Callable[[list], None] = lambda rows: None) -> None:
        """Blocking loop for one session (the GUI runs it in a worker thread): wait for each bar, decide, hard-flatten before the close, stop. However it ends (hard flatten, Stop, an error) an ARMED
        engine flattens its ledger on the way out."""
        if self.tc is None:
            self.connect()
        today = clock_now().tz_localize(None).normalize()
        info = self.session_info(today)
        if info is None:
            self.log(f"{today.date()} is not a trading day", "warn")
            return
        if info[1].hour != 16:
            self.log(f"{today.date()} closes early ({info[1]:%H:%M}): the full-session grid does not apply, no trading today", "warn")
            return
        if self.live.armed and self.live.feed == "iex" and not self.live.accept_iex:
            raise RuntimeError("ARMED on the IEX feed is refused: the IEX s-scores of these pairs correlate only 0.2-0.57 with the consolidated-tape scores the strategy was tested on "
                               "(reports/feed_test.log). Run DRY RUN, get real-time SIP data, or accept IEX knowingly.")
        if self.live.armed:                                                              # pre-flight is a gate, not advice
            fails = [c for c in self.preflight() if c.level == "fail"]
            if fails:
                raise RuntimeError("pre-flight failed: " + "; ".join(f"{c.name}: {c.msg}" for c in fails))
        self._acquire_lock()
        try:
            self.start_session(today)
            if self.live.armed:
                pend = self.state.get("pending")
                orph = self.orphans()
                if pend or orph:
                    raise RuntimeError(f"refusing to start: an order was in flight when the engine last stopped ({pend}) or the broker holds positions the ledger does not explain {orph}. Flatten or reconcile first")
            tf = self.sc.tf
            self.log(f"session {today.date()} · {mode_name(self.live).upper()} · feed {self.live.feed.upper()} · {tf}-minute bars · deployed ${self.deploy:,.0f} × gross {self.G:.2f}", "ok")
            nb = len(D.session_grid(tf))
            hard = pd.Timestamp(f"{today:%Y-%m-%d} {self.live.hard_flat_time}")
            while not should_stop():
                now = clock_now().tz_localize(None)
                self._heartbeat()
                if now >= hard:                                                          # checked first: a stalled loop must still flatten
                    if any(p.get("side") for p in self.state["pairs"].values()):
                        self.log("hard flatten before the close", "warn")
                        self.flatten("end of day")
                    if self.live.armed and self.live.flatten_orphans_eod:
                        self.flatten_orphans()
                    break
                j = bar_index_now(today, now - pd.Timedelta(seconds=self.live.decision_delay_s), tf)
                if 0 <= j < nb and j > self.state["last_j"]:
                    try:
                        rows = self.step(today, now)
                        on_rows(rows)
                    except Exception as e:                                               # noqa: BLE001
                        self.log(f"step failed: {type(e).__name__}: {str(e)[:140]}", "err")
                        self.state["last_j"] = j
                time.sleep(1.0)
        finally:
            try:
                self._leave_session()
            except Exception as e:                                                       # noqa: BLE001
                self.log(f"FINAL FLATTEN FAILED ({type(e).__name__}: {str(e)[:100]}): check the account and close by hand", "err")
            finally:
                self._release_lock()
        self.finish_session(today)

    def _leave_session(self) -> None:
        """However a session ends (hard flatten, Stop, Ctrl+C, an error): close the ledger, and - when an entry was in flight (the intent is still on disk) - whatever that entry left at the broker, which the
        ledger does not know about."""
        if getattr(self, "state", None) is not None and any(p.get("side") for p in self.state["pairs"].values()):
            self.log("leaving with open pairs: flattening the ledger first", "warn")
            self.flatten("session end")
        if self.live.armed and getattr(self, "state", None) is not None and self.state.get("pending"):
            syms = {leg[0] for leg in (self.state["pending"].get("legs") or [])}
            self.log("leaving with an order in flight (Ctrl+C or an error during an entry): closing whatever it left at the broker", "warn")
            end = time.time() + self.live.submit_timeout_s
            for t in list(self._live_submits()):                                         # a submit call that is still out (a slow leg) lands later: wait for it
                t.join(max(0.0, end - time.time()))
            time.sleep(min(2.0, self.live.fill_wait_s))                                  # let the orders that were just sent arrive
            self.flatten_orphans(only=syms or None)                                      # only the legs of THIS entry: another position in the six ETFs is not ours to close here
            if self._live_submits() and self.state.get("pending"):
                self.state["pending"]["unknown"] = True                                  # a call is still out after the wait: the intent stays on disk
                self.save_state(self.state)

    def finish_session(self, date) -> None:
        st = self.state
        pnl = sum(x.get("pnl_usd", 0.0) for x in st["pairs"].values())
        self.append_log("daily", [dict(date=st["date"], shadow_pnl_usd=pnl, shadow_pnl_pct=pnl / max(self.deploy, 1.0), trades=sum(x["n_trades"] for x in st["pairs"].values()), G=self.G,
                                       halted=bool(self.halted), start_equity=st.get("start_equity"), lp_pnl_usd=self.lp_pnl())])
        real = ""
        if self.live.armed and self.tc is not None and st.get("start_equity"):
            try:
                eq = float(self.tc.get_account().equity)
                real = f" · account equity change ${eq - float(st['start_equity']):+,.0f} · this engine's fills ${self.lp_pnl():+,.0f}"
            except Exception:                                                            # noqa: BLE001
                pass
        self.log(f"session over · shadow P&L ${pnl:,.0f} ({pnl / max(self.deploy, 1.0):+.2%} of deployed capital){real}", "ok")


EXIT_TXT = {1: "target", 2: "stop", 3: "time stop", 4: "end of day", 5: "no signal", 6: "day stop"}


# ── replay of a past session through the same step function (no network with the parquet source) ───────────────────────────────────────────────────
def replay_session(date, sc: SigCfg, c: SimCfg, live: Optional[LiveSettings] = None, source=None, equity: float = 100_000.0, log=lambda m, k="info": None, start_state: bool = True) -> tuple[LiveEngine, pd.DataFrame]:
    """Walk through the bars of a past session one decision at a time, the way the live loop would, in DRY RUN (nothing is sent). Returns the engine and the decision table."""
    live = replace(live or LiveSettings(), armed=False)
    src = source or ParquetSource(live.feed)
    eng = LiveEngine(sc, c, live, log=log, source=src)
    eng.is_replay = True                                                                 # the GUI must never route KILL / flatten / reconcile to this unarmed, broker-less book
    d = pd.Timestamp(date).normalize()
    eng.state = eng.blank_state(d)
    eng.deploy = live.capital_override if live.capital_override > 0 else equity * live.equity_fraction
    eng.state["start_equity"] = equity
    eng.G = min(c.g_init if not c.vol_on else eng.next_gross(), live.g_cap)
    eng.history = {}
    tf = sc.tf
    nb = len(D.session_grid(tf))
    rows = []
    for j in range(nb):
        now = d + pd.Timedelta(minutes=D.OPEN_MIN + (j + 1) * tf, seconds=live.decision_delay_s)
        eng.save_state = lambda st: None                                                # a replay never writes the live state file
        eng.append_log = lambda name, r: None
        rows += eng.step(d, now)
    return eng, pd.DataFrame(rows)
