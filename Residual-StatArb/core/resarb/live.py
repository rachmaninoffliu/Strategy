"""
Live / paper engine for the residual strategies on Alpaca (no Qt).

Every 15 minutes (a little after the bar boundary) the engine
    1. fetches today's 15-minute bars from the live feed (IEX in real time, or SIP 15 minutes late) and appends them to the cached history,
    2. rebuilds betas, residuals and the strategy's score - the SAME code as the backtest - for the data bar the backtest would use (bar k-1-lag for a trade at bar k's open),
    3. if that bar is a decision bar: cancels its own open orders, reads the broker's positions, calls portfolio.decide() with them (the backtest calls it with its simulated holdings),
       adds the ETF hedge, turns weights into whole shares and sends only the differences (closing / reducing orders first),
    4. journals the decision, every order and every fill (reports/journal_<mode>.jsonl).

Target-based: nothing is remembered between steps except the high-water mark and the day's start equity; the broker's positions are the truth. An order whose answer was lost is looked
up by its client id; if it later fills, the next step simply trades from the new position.

Safety
  * starts on PAPER and DRY RUN; ARMED needs the typed phrase "ARM PAPER" (real money is disabled unless RESARB_ALLOW_LIVE=1 and the phrase "ARM LIVE MONEY")
  * an ARMED engine needs the project's own keys (the key file "Alpaca Key.txt" in the project folder, or <PREFIX>_ALPACA_API_KEY / _SECRET_KEY for a separate account - never
    the ~/.claude.json fallback) and an account no other residual strategy has claimed
  * one engine per strategy and account at a time (an operating-system file lock)
  * it only ever trades its universe and the hedge ETFs; other symbols in the account are reported, never touched
  * shorts only in shortable, easy-to-borrow names; whole shares; a minimum ticket; a gross cap; at most max_orders per rebalance
  * daily-loss halt (no new risk), drawdown stop from the high-water mark (flatten and stay flat until reset), KILL (cancel + flatten), stale-data guard, HTTP timeouts, API throttle
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import data as D
from . import strategy as S
from .portfolio import approach, decide, exposures, hedge_etf, project

NY = "America/New_York"
ARM_PHRASE = "ARM PAPER"
LIVE_PHRASE = "ARM LIVE MONEY"


@dataclass
class LiveSettings:
    paper: bool = True
    armed: bool = False
    capital: float = 0.0                 # > 0: trade this much; otherwise equity x equity_fraction
    equity_fraction: float = 0.9
    max_gross_x: float = 1.9             # (stocks + ETFs) gross / capital, hard cap - Reg-T allows 2x overnight
    min_trade_usd: float = 100.0
    max_orders: int = 150
    daily_loss_halt: float = 0.03
    dd_stop: float = 0.15
    require_etb: bool = True
    shared_keys_ok: bool = False         # allow an ARMED engine on keys that are not this strategy's own (knowingly)
    fill_wait_s: float = 40.0
    bar_delay_s: float = 25.0            # seconds after the 15-minute boundary before a step runs
    late_limit_s: float = 300.0          # a step that starts later than this after its boundary is skipped (the backtest fills at the bar open)
    max_calls_per_min: int = 150


@dataclass
class Check:
    name: str
    level: str                           # ok | warn | fail
    msg: str


@dataclass
class Plan:
    data_bar: pd.Timestamp               # last bar the decision saw
    trade_bar: pd.Timestamp              # bar whose open the orders aim at
    decision: bool                       # is the data bar a decision bar (or the end-of-day flatten)?
    eod: bool
    capital: float
    equity: float
    book: pd.DataFrame                   # symbol, kind, score, target_w, px, cur_qty, tgt_qty
    orders: pd.DataFrame                 # symbol, side, qty, ref_px, notional, reason, ok, note, cid
    expo: dict
    notes: list = field(default_factory=list)
    halted: bool = False
    stopped: bool = False


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY).tz_localize(None)


def bar_start(t: pd.Timestamp) -> pd.Timestamp:
    return t.floor(f"{D.BAR_MIN}min")


def data_bar_for(k_ts: pd.Timestamp, lag: int, history: pd.DatetimeIndex) -> pd.Timestamp:
    """The data bar a trade at bar ``k_ts``'s open decides on: 1 + lag bars earlier in the trading calendar (the backtest's k-1-lag). Inside the session that is plain clock arithmetic;
    across the overnight it counts back through the previous session's bars in ``history``."""
    back = 1 + int(lag)
    sess_open = k_ts.normalize() + pd.Timedelta(hours=9, minutes=30)
    n_today = int((k_ts - sess_open) / pd.Timedelta(minutes=D.BAR_MIN))
    if back <= n_today:
        return k_ts - pd.Timedelta(minutes=D.BAR_MIN * back)
    prev = history[history < sess_open]
    m = back - n_today
    if len(prev) < m:
        raise RuntimeError("no previous session in the history")
    return prev[-m]


class Throttle:
    def __init__(self, per_min: int):
        self.per_min, self.calls, self.lock = per_min, deque(), threading.Lock()

    def wait(self) -> None:
        with self.lock:
            while True:
                t = time.monotonic()
                while self.calls and t - self.calls[0] > 60:
                    self.calls.popleft()
                if len(self.calls) < self.per_min:
                    self.calls.append(t)
                    return
                time.sleep(max(0.05, 60 - (t - self.calls[0]) + 0.01))


class FileLock:
    """Operating-system lock on a file: one armed engine per strategy and mode; released by the OS if the process dies."""

    def __init__(self, path: str):
        self.path, self.fh = path, None

    def acquire(self) -> bool:
        if self.fh is not None:
            return True
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self.fh = fh
        return True

    def release(self) -> None:
        if self.fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self.fh.close()
        self.fh = None


def _status(o) -> str:
    return str(getattr(getattr(o, "status", None), "value", getattr(o, "status", "?")))


class LiveEngine:
    def __init__(self, strat: S.Strategy, live: LiveSettings, log: Callable[[str, str], None] = lambda m, k="info": print(m)):
        self.s, self.live, self.log = strat, live, log
        self.tc = None
        self.dc = None
        self.acct: dict = {}
        self._lock = threading.RLock()
        self._thr = Throttle(live.max_calls_per_min)
        self._hist: dict = {}                # symbol -> cached SIP frame (loaded once per day)
        self._hist_day = None
        self._iex_scale: dict = {}
        self._clock = (0.0, None)
        self.flock = FileLock(self.path("engine", "lock"))
        self.last_plan: Optional[Plan] = None
        self.last_step_key = None

    # ── files ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return "paper" if self.live.paper else "live"

    def path(self, name: str, ext: str = "json") -> str:
        d = os.path.join(self.s.folder, "reports")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{name}_{self.mode}.{ext}")

    def load_state(self) -> dict:
        p = self.path("state")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError) as e:
                raise RuntimeError(f"state file {p} is unreadable ({e}) - fix or move it") from e
        return {"hwm": None, "day": None, "day_start_equity": None, "stopped": None, "pending": None, "attempt": 0}

    def save_state(self, st: dict) -> None:
        p = self.path("state")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:                                          # closed (flushed) BEFORE the rename
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
        row = {"ts": str(now_et()), "kind": kind, "strategy": self.s.KEY, "armed": bool(self.live.armed), **row}
        with open(self.path("journal", "jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    # ── broker ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def _call(self, fn, *a, **k):
        self._thr.wait()
        return fn(*a, **k)

    def key_source(self) -> str:
        return "own" if D.own_keys(self.s.PREFIX, self.live.paper) else "shared"

    def connect(self) -> dict:
        if not self.live.paper and os.environ.get("RESARB_ALLOW_LIVE") != "1":
            raise RuntimeError("real-money trading is disabled (Valid has not signed off any of these strategies). Set RESARB_ALLOW_LIVE=1 only after that.")
        self.tc = D.trading_client(self.s.PREFIX, self.live.paper)
        self.dc = D.data_client(self.s.PREFIX)
        t0 = time.time()
        a = self._call(self.tc.get_account)
        clk = self._call(self.tc.get_clock)
        self._clock = (time.monotonic(), clk)
        self.acct = dict(paper=self.live.paper, number=str(a.account_number), status=str(getattr(a.status, "value", a.status)), equity=float(a.equity), last_equity=float(a.last_equity),
                         cash=float(a.cash), buying_power=float(a.buying_power), multiplier=str(a.multiplier), shorting=bool(a.shorting_enabled),
                         blocked=bool(a.trading_blocked or a.account_blocked), pdt=bool(a.pattern_day_trader), market_open=bool(clk.is_open),
                         next_open=pd.Timestamp(clk.next_open).tz_convert(NY).tz_localize(None), next_close=pd.Timestamp(clk.next_close).tz_convert(NY).tz_localize(None),
                         latency_ms=int((time.time() - t0) * 1000), keys=self.key_source(), key_src=D.key_source(self.s.PREFIX, self.live.paper))
        return self.acct

    def clock(self, max_age: float = 30.0):
        t, c = self._clock
        if c is None or time.monotonic() - t > max_age:
            c = self._call(self.tc.get_clock)
            self._clock = (time.monotonic(), c)
        return c

    def refresh_account(self) -> dict:
        a = self._call(self.tc.get_account)
        self.acct.update(equity=float(a.equity), last_equity=float(a.last_equity), buying_power=float(a.buying_power), blocked=bool(a.trading_blocked or a.account_blocked))
        return self.acct

    def positions(self) -> dict:
        return {p.symbol: int(float(p.qty)) for p in self._call(self.tc.get_all_positions)}

    def our_open_orders(self) -> list:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        oo = self._call(self.tc.get_orders, GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        return [o for o in oo if str(o.client_order_id or "").startswith(self.s.PREFIX + "-")]

    def cancel_ours(self, wait_s: float = 10.0) -> int:
        oo = self.our_open_orders()
        for o in oo:
            try:
                self._call(self.tc.cancel_order_by_id, o.id)
            except Exception as e:                                                         # noqa: BLE001
                self.log(f"cancel {o.symbol} failed: {type(e).__name__}: {str(e)[:80]}", "warn")
        t0 = time.monotonic()
        while oo and time.monotonic() - t0 < wait_s:
            time.sleep(1.0)
            oo = self.our_open_orders()
        if oo:
            raise RuntimeError(f"{len(oo)} of this engine's orders are still open after cancelling - not trading on top of them")
        return len(oo)

    # ── account ownership ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
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
        p = self._registry_path()
        reg = self._registry()
        key = f"{self.mode}:{self.acct.get('number')}"
        owner = reg.get(key, {}).get("strategy")
        if owner and owner != self.s.KEY:
            raise RuntimeError(f"account {self.acct.get('number')} is claimed by {owner}: two strategies on one account would net and close each other's positions - "
                               f"stop {owner} first, or give {self.s.KEY} a separate paper account ({self.s.PREFIX}_ALPACA_API_KEY / _SECRET_KEY)")
        reg[key] = {"strategy": self.s.KEY, "since": reg.get(key, {}).get("since", str(now_et()))}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=1)

    def arm_problems(self) -> list:
        """Why this engine may NOT be armed now (empty = OK)."""
        out = []
        if not self.live.paper and os.environ.get("RESARB_ALLOW_LIVE") != "1":
            out.append("real money is disabled")
        if self.key_source() != "own" and not self.live.shared_keys_ok:
            out.append("no own keys: put 'Alpaca Key.txt' (Key ID / Secret Key) in the project folder - the ~/.claude.json keys of the Claude tools are refused for orders")
        if self.acct:
            owner = self.account_owner()
            if owner and owner != self.s.KEY:
                out.append(f"account claimed by {owner}")
            if self.acct.get("blocked"):
                out.append("trading is blocked on the account")
            if not self.acct.get("shorting"):
                out.append("shorting is disabled on the account")
        return out

    # ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def symbols_now(self, month: Optional[pd.Timestamp] = None) -> list:
        u = D.load_universe()
        n = int(self.s.v["universe_n"])
        month = (month or now_et()).to_period("M").to_timestamp()
        m = u[(u.month == month) & (u["rank"] <= n)]
        if not len(m) and len(u):
            m = u[(u.month == u.month.max()) & (u["rank"] <= n)]
        return list(m.sort_values("rank").symbol)

    def strategy_symbols(self) -> set:
        """Everything this engine may trade: this month's and last month's members (to close leavers) and the hedge ETFs."""
        now = now_et()
        prev = (now.to_period("M") - 1).to_timestamp()
        return set(self.symbols_now()) | set(self.symbols_now(prev)) | set(D.ETFS)

    def prepare(self, refresh: bool = True, now: Optional[pd.Timestamp] = None) -> None:
        """Before the open: top up the SIP cache of the members + ETFs (history up to 15 minutes ago) and load it."""
        now = now or now_et()
        syms = sorted(self.strategy_symbols())
        if refresh:
            self.log(f"topping up the SIP cache for {len(syms)} symbols ...", "info")
            D.update_bars(syms, "sip", workers=4, log=lambda m: None, client=self.dc or D.data_client(self.s.PREFIX))
        start = now.normalize() - pd.Timedelta(days=int(self.s.warmup_days() * 1.5) + 10)
        self._hist = {}
        for s in syms:
            f = D.load_bars(s, "sip")
            self._hist[s] = f[(f.index >= start) & (f.index < now.normalize())] if len(f) else f       # history = sessions before today; today comes from the live feed
        self._hist_day = now.normalize()
        self._iex_scale = {}
        if self.s.v["feed"] == "iex" and self.dc is not None:
            self._iex_scale = self.iex_volume_scale(now)

    def iex_volume_scale(self, now: pd.Timestamp, days: int = 10) -> dict:
        """IEX carries a few % of each stock's volume: today's IEX volumes are scaled by each symbol's SIP / IEX volume ratio over the last sessions, so that volume features
        (volume surprise) compare like with like."""
        out = {}
        try:
            t0 = now.normalize() - pd.Timedelta(days=int(days * 1.6) + 3)
            syms = sorted(self._hist)
            iex = {}
            for i in range(0, len(syms), 100):
                iex.update(D.fetch_bars(self.dc, syms[i:i + 100], t0, now.normalize(), feed="iex"))
            for s in syms:
                h, x = self._hist.get(s), iex.get(s)
                if h is None or x is None or not len(h) or not len(x):
                    continue
                common = h.index.intersection(x.index)
                vi = float(x.volume.reindex(common).sum())
                if vi > 0:
                    out[s] = float(np.clip(h.volume.reindex(common).sum() / vi, 3.0, 1000.0))
            if out:
                med = float(np.median(list(out.values())))
                out = {s: out.get(s, med) for s in syms}
                self.log(f"IEX volume scale: median x{med:.0f} (IEX = {1 / med:.1%} of SIP volume)", "info")
        except Exception as e:                                                             # noqa: BLE001
            self.log(f"IEX volume scale unavailable ({type(e).__name__}) - volume features use raw IEX volume", "warn")
        return out

    def live_bars(self, now: pd.Timestamp) -> dict:
        """Today's complete bars from the live feed (history before today always comes from the SIP cache)."""
        feed = "iex" if self.s.v["feed"] == "iex" else "sip"
        end = now if self.s.v["feed"] != "sip_delayed" else now - pd.Timedelta(minutes=15, seconds=5)        # the free plan serves SIP data older than 15 minutes
        start = now.normalize() + pd.Timedelta(hours=9, minutes=30)
        if end <= start:
            return {}
        syms = sorted(self._hist)
        out = {}
        for i in range(0, len(syms), 100):
            out.update(D.fetch_bars(self.dc, syms[i:i + 100], start, end, feed=feed))
        cut = bar_start(end)
        out = {s: f[f.index < cut] for s, f in out.items()}                                 # only COMPLETE bars
        if feed == "iex" and getattr(self, "_iex_scale", None):
            for s, f in out.items():
                k = self._iex_scale.get(s)
                if k and len(f):
                    f = f.copy()
                    f["volume"] = f["volume"] * k
                    if "trades" in f:
                        f["trades"] = f["trades"] * k
                    out[s] = f
        return out

    def context(self, now: pd.Timestamp) -> S.Context:
        if self._hist_day is None or self._hist_day != now.normalize():
            self.prepare(refresh=False, now=now)
        extra = self.live_bars(now)
        frames = {}
        for s, f in self._hist.items():
            x = extra.get(s)
            frames[s] = pd.concat([f[f.index < x.index.min()], x]) if x is not None and len(x) else f
        u = D.load_universe()
        n = int(self.s.v["universe_n"])
        syms = sorted(set(self._hist) - set(D.ETFS))
        p = D.Panel(frames, list(D.ETFS) + syms)
        member_d = D.membership_mask(p, n, u)
        from . import factors as F
        res = F.build(p, self.s.factor_cfg(), member_d)
        member = member_d[p.day] & np.isfinite(res.e_cc) & np.isfinite(p.open)
        return S.Context(p=p, res=res, member=member, member_d=member_d, lag=self.s.lag)

    # ── planning ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def plan(self, now: Optional[pd.Timestamp] = None, force: bool = False) -> Plan:
        """What the engine would do at the bar opening at ``now`` (rounded down to 15 minutes). ``force`` = decide even if the data bar is not a decision bar (a manual dry run)."""
        with self._lock:
            now = now or now_et()
            if self.tc is None:
                self.connect()
            k_ts = bar_start(now)
            lag = self.s.lag
            ctx = self.context(now)
            p = ctx.p
            notes: list = []
            prev_days = p.dates[p.dates < now.normalize()]
            if not len(prev_days) or (now.normalize() - prev_days[-1]).days > 4:
                raise RuntimeError(f"the SIP history ends {prev_days[-1]:%Y-%m-%d} - top it up (Update data / restart the loop before the open)" if len(prev_days) else "no history")
            sess_open = k_ts.normalize() + pd.Timedelta(hours=9, minutes=30)
            in_session = k_ts.weekday() < 5 and sess_open <= k_ts <= sess_open + pd.Timedelta(hours=6, minutes=15)
            if force and not in_session:                                                     # a manual dry run outside the session: decide on the newest complete bar
                want_td = p.ts[-1]
                notes.append(f"outside the session: planned on the newest bar ({want_td:%Y-%m-%d %H:%M}); the orders would go out at the next open")
            else:
                want_td = data_bar_for(k_ts, lag, p.ts)
            td = int(np.searchsorted(p.ts, want_td, side="right") - 1)
            if td < 0 or p.ts[td] != want_td:
                raise RuntimeError(f"the data bar {want_td:%Y-%m-%d %H:%M} is not in the feed yet (newest: {p.ts[-1]:%H:%M}) - no decision")
            m = p.col[D.MARKET]
            have = np.isfinite(p.close[td])
            mem_d = ctx.member_d[p.day[td]]
            frac = float(have[mem_d].mean()) if mem_d.any() else 0.0
            if frac < 0.8:
                raise RuntimeError(f"only {frac:.0%} of the members have a bar at {p.ts[td]:%H:%M} - stale feed, no decision")
            clk = self.clock()
            next_close = pd.Timestamp(clk.next_close).tz_convert(NY).tz_localize(None)
            eod = bool(self.s.v["flat_eod"]) and clk.is_open and (next_close - k_ts) <= pd.Timedelta(minutes=D.BAR_MIN)
            sched = self.s.decision_bars(p)
            is_dec = bool(sched[td]) or eod or force
            sc = (self.s.live_score(ctx) if hasattr(self.s, "live_score") else self.s.score(ctx))[td]
            acct = self.refresh_account()
            equity = float(acct["equity"])
            cap = self.live.capital if self.live.capital > 0 else equity * self.live.equity_fraction
            pos = self.positions()
            allowed = self.strategy_symbols()
            foreign = {s: q for s, q in pos.items() if s not in allowed}
            if foreign:
                u = D.load_universe()
                ours = set(u[u["rank"] <= int(self.s.v["universe_n"])].symbol)
                old = {s: q for s, q in foreign.items() if s in ours}
                notes.append(f"positions outside this month's symbols are left alone: {foreign}" + (f" - {sorted(old)} were members earlier: close them with KILL or by hand" if old else ""))
            px = p.close[td]
            cur_q = np.array([pos.get(s, 0) for s in p.syms], float)
            with np.errstate(invalid="ignore"):
                held_w = np.where(np.isfinite(px), cur_q * px / cap, 0.0)
            st = self.load_state()
            st, halted, stopped = self._risk_state(st, equity, notes)
            self.save_state(st)
            if stopped and (cur_q != 0).any() and not is_dec:                                   # a stop flattens at the next bar, not at the next scheduled (e.g. monthly) decision
                is_dec = True
                notes.append("drawdown stop: flattening at this bar (outside the rebalance schedule)")
            res = ctx.res
            is_etf = np.array([s in D.ETFS for s in p.syms])
            sidx = np.where(~is_etf)[0]
            tradable = np.isfinite(px) & np.isfinite(p.open[td])
            pc = self.s.port_cfg()
            if eod or stopped:
                tgt_s = np.zeros(len(sidx))
            else:
                tgt_s = decide(sc[sidx], ctx.member[td, sidx], held_w[sidx], pc, res.sector_col[sidx], tradable[sidx])
                tgt_s = approach(held_w[sidx], tgt_s, pc.trade_rate, pc.gross, tradable[sidx])
            full = np.zeros(len(p.syms))
            full[sidx] = tgt_s
            if pc.hedge == "project" and not (eod or stopped):
                full = project(full, res, p.day[td])
            tgt = full.copy()
            h = np.zeros(len(res.etf_idx))
            held_r = full
            if pc.hedge == "etf" and not (eod or stopped):
                with np.errstate(invalid="ignore", divide="ignore"):                              # hedge the WHOLE shares that will be held, not the unrounded weights
                    held_r = np.where(np.isfinite(px) & (px > 0), np.trunc(np.where(tradable, full, held_w) * cap / px) * px / cap, full)
                held_r[is_etf] = 0.0
                h = hedge_etf(held_r, res, p.day[td])
                tgt[res.etf_idx] += h
            tgt = np.where(tradable, tgt, held_w)
            book, orders = self._orders(p, tgt, cur_q, px, cap, sc, is_etf, halted, notes, allowed, k_ts, st)
            expo = exposures(held_r, h, res, p.day[td])
            plan = Plan(data_bar=p.ts[td], trade_bar=k_ts, decision=is_dec, eod=eod, capital=cap, equity=equity, book=book, orders=orders if is_dec else orders.iloc[0:0], expo=expo,
                        notes=notes, halted=halted, stopped=stopped)
            self.last_plan = plan
            return plan

    def _risk_state(self, st: dict, equity: float, notes: list) -> tuple:
        today = str(now_et().date())
        if st.get("day") != today:
            st["day"], st["day_start_equity"] = today, float(self.acct.get("last_equity") or equity)
        st["hwm"] = max(float(st.get("hwm") or 0.0), equity)
        halted = stopped = False
        d0 = float(st.get("day_start_equity") or equity)
        if self.live.daily_loss_halt > 0 and d0 > 0 and equity / d0 - 1 <= -self.live.daily_loss_halt:
            halted = True
            notes.append(f"DAILY-LOSS HALT: equity {equity / d0 - 1:.2%} today (limit -{self.live.daily_loss_halt:.1%}) - only reducing orders")
        if st.get("stopped"):
            stopped = True
            notes.append(f"DRAWDOWN STOP active since {st['stopped']} - the target is flat; reset the drawdown base to resume")
        elif self.live.dd_stop > 0 and st["hwm"] > 0 and equity / st["hwm"] - 1 <= -self.live.dd_stop:
            stopped = True
            st["stopped"] = str(now_et())
            notes.append(f"DRAWDOWN STOP: equity {equity / st['hwm'] - 1:.1%} from the high-water mark (limit -{self.live.dd_stop:.0%}) - flattening")
        return st, halted, stopped

    def _orders(self, p, tgt_w, cur_q, px, cap, sc, is_etf, halted, notes, allowed, k_ts, st) -> tuple:
        live = self.live
        assets = None
        try:
            assets = D.list_assets(self.s.PREFIX).set_index("symbol")
        except Exception as e:                                                             # noqa: BLE001
            notes.append(f"asset list unavailable ({type(e).__name__}) - new shorts are refused")
        with np.errstate(invalid="ignore", divide="ignore"):
            tq = np.where(np.isfinite(px) & (px > 0), np.trunc(tgt_w * cap / px), cur_q)
        gross = float(np.nansum(np.abs(tq) * np.nan_to_num(px)) / cap) if cap > 0 else 0.0
        gmax = min(float(live.max_gross_x), float(self.s.v.get("max_gross", live.max_gross_x)))       # the backtest applies the strategy's cap too
        if gross > gmax:
            notes.append(f"target gross {gross:.2f}x > cap {gmax:.2f}x - scaled down (the backtest does the same)")
            tq = np.trunc(tq * gmax / gross)
        rows, book = [], []
        n = int(st.get("attempt", 0)) + 1
        for j, s in enumerate(p.syms):
            c, t = int(cur_q[j]), int(tq[j])
            if c == 0 and t == 0 and not tgt_w[j]:
                continue
            book.append(dict(symbol=s, kind="hedge ETF" if is_etf[j] else "stock", score=float(sc[j]) if np.isfinite(sc[j]) else np.nan, target_w=float(tgt_w[j]), px=float(px[j]),
                             cur_qty=c, tgt_qty=t))
            if t == c:
                continue
            if s not in allowed:
                continue
            ok, note = True, ""
            reducing = (t == 0) or (np.sign(t) == np.sign(c) and abs(t) < abs(c))
            if halted and not reducing:
                ok, note = False, "daily-loss halt: no new risk"
            elif c != 0 and t != 0 and np.sign(c) != np.sign(t):
                t, note = 0, "flip split: closing now, re-opening at the next decision"
                reducing = True
            if ok and t < 0 and abs(t) > abs(min(c, 0)) and not is_etf[j] and live.require_etb:
                a = assets.loc[s] if assets is not None and s in assets.index else None
                if a is None or not bool(a.get("shortable", False)) or not bool(a.get("etb", False)):
                    ok, note = False, "not shortable / not easy-to-borrow"
            qty = abs(t - c)
            notional = qty * float(px[j]) if np.isfinite(px[j]) else np.nan
            if ok and not reducing and np.isfinite(notional) and notional < live.min_trade_usd:
                continue
            rows.append(dict(symbol=s, side="buy" if t > c else "sell", qty=qty, cur_qty=c, tgt_qty=t, ref_px=float(px[j]), notional=notional, reason="reduce" if reducing else "add",
                             ok=ok, note=note, cid=f"{self.s.PREFIX}-{k_ts:%Y%m%d%H%M}-{s.replace('.', '')}-{n}"))
        cols = ["symbol", "side", "qty", "cur_qty", "tgt_qty", "ref_px", "notional", "reason", "ok", "note", "cid"]
        o = pd.DataFrame(rows, columns=cols)
        if len(o) > live.max_orders:
            notes.append(f"{len(o)} orders > max {live.max_orders}: reducing orders and the largest others are kept")
            red = o[o.reason == "reduce"]
            rest = o[o.reason != "reduce"].sort_values("notional", ascending=False).head(max(live.max_orders - len(red), 0))
            o = pd.concat([red, rest], ignore_index=True)
        o = o.assign(_k=o.reason.ne("reduce").astype(int)).sort_values(["_k", "notional"], ascending=[True, False]).drop(columns="_k").reset_index(drop=True)
        return pd.DataFrame(book), o

    # ── execution ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def submit(self, plan: Plan) -> list:
        """Send the plan's orders (ARMED only). Cancels this engine's open orders first; reducing orders go out before the others."""
        if not self.live.armed:
            raise RuntimeError("DRY RUN: the engine is not armed, nothing is sent")
        probs = self.arm_problems()
        if probs:
            raise RuntimeError("refusing to send: " + "; ".join(probs))
        if not self.flock.acquire():
            raise RuntimeError("another engine of this strategy holds the lock (GUI or headless runner) - stop it first")
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        with self._lock:
            self.claim_account()
            clk = self.clock(5)
            if not clk.is_open:
                raise RuntimeError("the market is closed")
            self.cancel_ours()
            st = self.load_state()
            st["attempt"] = int(st.get("attempt", 0)) + 1
            todo = plan.orders[plan.orders.ok].copy()
            st["pending"] = {"trade_bar": str(plan.trade_bar), "orders": todo[["symbol", "side", "qty", "cid"]].to_dict("records")}
            self.save_state(st)                                                             # write-ahead: what is about to be sent
            self.journal("decision", **self._decision_row(plan), n_orders=len(todo))
            results = []
            t_sub = pd.Timestamp.now(tz="UTC")
            for r in todo.itertuples():
                t0 = time.time_ns()
                try:
                    o = self._call(self.tc.submit_order, MarketOrderRequest(symbol=r.symbol, qty=int(r.qty), side=OrderSide.BUY if r.side == "buy" else OrderSide.SELL,
                                                                             time_in_force=TimeInForce.DAY, client_order_id=r.cid))
                    results.append(dict(symbol=r.symbol, cid=r.cid, id=str(o.id), status=_status(o), side=r.side, qty=int(r.qty), ref_px=r.ref_px, t_ns=time.time_ns() - t0))
                except Exception as e:                                                     # noqa: BLE001
                    code = getattr(e, "status_code", None)
                    if code is not None and 400 <= int(code) < 500 and int(code) not in (408, 429):
                        results.append(dict(symbol=r.symbol, cid=r.cid, status="rejected", side=r.side, qty=int(r.qty), ref_px=r.ref_px, note=f"{code}: {str(e)[:120]}"))
                    else:
                        found = self._lookup(r.cid)
                        results.append(dict(symbol=r.symbol, cid=r.cid, id=str(found.id) if found else None, status=_status(found) if found else "unknown", side=r.side,
                                            qty=int(r.qty), ref_px=r.ref_px, note=f"ambiguous submit ({type(e).__name__}: {str(e)[:80]})"))
                self.journal("order", data_bar=plan.data_bar, trade_bar=plan.trade_bar, **results[-1])
            st["pending"] = None
            st["last_submit"] = str(now_et())
            self.save_state(st)
            self._await_fills(results, t_sub, plan)
            return results

    @staticmethod
    def _decision_row(plan: Plan) -> dict:
        """Everything the comparison needs to rebuild the backtest's decision for the same bar: targets, the positions held, the prices used, capital."""
        b = plan.book
        return dict(data_bar=plan.data_bar, trade_bar=plan.trade_bar, capital=plan.capital, equity=plan.equity, expo=plan.expo, notes=plan.notes, eod=plan.eod,
                    halted=plan.halted, stopped=plan.stopped,
                    targets={r.symbol: r.target_w for r in b.itertuples()} if len(b) else {}, held={r.symbol: r.cur_qty for r in b.itertuples() if r.cur_qty} if len(b) else {},
                    px={r.symbol: r.px for r in b.itertuples()} if len(b) else {}, scores={r.symbol: r.score for r in b.itertuples() if np.isfinite(r.score)} if len(b) else {})

    def _lookup(self, cid: str):
        for _ in range(3):
            try:
                return self._call(self.tc.get_order_by_client_id, cid)
            except Exception:                                                              # noqa: BLE001
                time.sleep(1.0)
        return None

    def _await_fills(self, results: list, t_sub: pd.Timestamp, plan: Plan) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        want = {r["cid"] for r in results if r.get("status") not in ("rejected",)}
        done: dict = {}
        t0 = time.monotonic()
        first = True
        while want - set(done) and (first or time.monotonic() - t0 < self.live.fill_wait_s):
            if not first:
                time.sleep(4.0)
            first = False
            try:
                oo = self._call(self.tc.get_orders, GetOrdersRequest(status=QueryOrderStatus.ALL, after=(t_sub - pd.Timedelta(seconds=30)).to_pydatetime(), limit=500))
            except Exception as e:                                                         # noqa: BLE001
                self.log(f"order status read failed: {type(e).__name__}", "warn")
                continue
            for o in oo:
                cid = str(o.client_order_id or "")
                if cid in want and _status(o) in ("filled", "canceled", "rejected", "expired", "done_for_day"):
                    done[cid] = o
        for r in results:
            o = done.get(r["cid"])
            if o is None:
                continue
            fq, fp = float(o.filled_qty or 0), float(o.filled_avg_price or 0) or np.nan
            slip = (fp / r["ref_px"] - 1) * 1e4 * (1 if r["side"] == "buy" else -1) if np.isfinite(fp) and r.get("ref_px") else np.nan
            self.journal("fill", trade_bar=plan.trade_bar, symbol=r["symbol"], cid=r["cid"], status=_status(o), side=r["side"], qty=r["qty"], filled_qty=fq, avg_px=fp, ref_px=r["ref_px"],
                         slip_vs_ref_bp=slip, submitted_at=str(o.submitted_at), filled_at=str(o.filled_at))
        missing = want - set(done)
        if missing:
            self.log(f"{len(missing)} order(s) not final after {self.live.fill_wait_s:.0f} s - the next step cancels them and trades from the real positions", "warn")

    # ── the loop ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def step(self, now: Optional[pd.Timestamp] = None) -> Optional[Plan]:
        """One bar: plan; if it is a decision bar and the engine is ARMED, send. Returns the plan (or None outside the session)."""
        now = now or now_et()
        clk = self.clock()
        if not clk.is_open:
            return None
        k_ts = bar_start(now)
        late = (now - k_ts).total_seconds()
        if late > self.live.late_limit_s:
            self.log(f"step for {k_ts:%H:%M} is {late:.0f} s late - skipped (the backtest trades at the bar open)", "warn")
            return None
        plan = self.plan(now)
        for n in plan.notes:
            self.log(n, "warn")
        if not plan.decision:
            self.log(f"{k_ts:%H:%M} no rebalance at this bar (data {plan.data_bar:%H:%M}; equity ${plan.equity:,.0f})", "info")
            return plan
        o = plan.orders
        self.log(f"{k_ts:%H:%M} decision on {plan.data_bar:%H:%M} data: {int((plan.book.kind == 'stock').sum()) if len(plan.book) else 0} names, {int(o.ok.sum()) if len(o) else 0} orders"
                 + (" (EOD flatten)" if plan.eod else ""), "ok")
        if self.live.armed and len(o) and o.ok.any():
            res = self.submit(plan)
            bad = [r for r in res if r["status"] in ("rejected", "unknown")]
            self.log(f"sent {len(res) - len(bad)} / {len(res)} orders" + (f", {len(bad)} rejected / unknown" if bad else ""), "warn" if bad else "trade")
        else:
            self.journal("plan_dry" if not self.live.armed else "plan_no_orders", **self._decision_row(plan), n_orders=len(o))
        return plan

    def run(self, stop: threading.Event, on_plan: Callable[[Plan], None] = lambda p: None) -> None:
        """Loop until ``stop``: prepare before the open, one step after every 15-minute boundary, a summary after the close."""
        if self.tc is None:
            self.connect()
        if self.live.armed and not self.flock.acquire():
            raise RuntimeError("another engine of this strategy is running (lock held)")
        self.log(f"loop started ({'ARMED' if self.live.armed else 'DRY RUN'}, {self.mode}, feed {self.s.v['feed']}, lag {self.s.lag})", "ok")
        prepared = None
        try:
            while not stop.is_set():
                now = now_et()
                try:
                    clk = self.clock(60)
                    if prepared != now.date() and (clk.is_open or (now.hour, now.minute) >= (8, 45)) and now.weekday() < 5:
                        self.prepare(refresh=True, now=now)
                        prepared = now.date()
                    if clk.is_open:
                        k_ts = bar_start(now)
                        if (now - k_ts).total_seconds() >= self.live.bar_delay_s and self.last_step_key != k_ts:
                            self.last_step_key = k_ts
                            plan = self.step(now)
                            if plan is not None:
                                on_plan(plan)
                except Exception as e:                                                     # noqa: BLE001
                    self.log(f"step error: {type(e).__name__}: {str(e)[:200]}", "err")
                    self.journal("error", msg=f"{type(e).__name__}: {str(e)[:300]}")
                stop.wait(1.0)
        finally:
            self.flock.release()
            self.log("loop stopped", "info")

    # ── manual actions ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def kill(self) -> list:
        """KILL: cancel this engine's open orders, then close every position in this strategy's symbols with market orders. Works in DRY RUN too (an emergency action)."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        if self.tc is None:
            self.connect()
        with self._lock:
            out = []
            try:
                self.cancel_ours()
            except Exception as e:                                                         # noqa: BLE001
                out.append(dict(symbol="*", status=f"cancel: {e}"))
            u = D.load_universe()
            allowed = set(u[u["rank"] <= int(self.s.v["universe_n"])].symbol) | set(D.ETFS) | self.strategy_symbols()     # anything this strategy could ever have bought
            clk = self.clock(1)
            tif = TimeInForce.DAY if clk.is_open else TimeInForce.OPG
            for s, q in self.positions().items():
                if s not in allowed or q == 0:
                    if q:
                        out.append(dict(symbol=s, qty=q, status="not this strategy's symbol - left alone"))
                    continue
                cid = f"{self.s.PREFIX}-KILL-{s.replace('.', '')}-{int(time.time()) % 1000000}"
                try:
                    self._call(self.tc.submit_order, MarketOrderRequest(symbol=s, qty=abs(q), side=OrderSide.SELL if q > 0 else OrderSide.BUY, time_in_force=tif, client_order_id=cid))
                    out.append(dict(symbol=s, qty=q, status="close sent"))
                except Exception as e:                                                     # noqa: BLE001
                    out.append(dict(symbol=s, qty=q, status=f"FAILED {type(e).__name__}: {str(e)[:100]}"))
            self.journal("kill", actions=out)
            return out

    def reset_drawdown(self) -> float:
        st = self.load_state()
        eq = float(self.refresh_account()["equity"])
        st["hwm"], st["stopped"] = eq, None
        self.save_state(st)
        return eq

    def preflight(self) -> list:
        out: list = []
        add = lambda n, l, m: out.append(Check(n, l, m))
        try:
            a = self.connect()
            add("broker", "ok", f"{'PAPER' if a['paper'] else 'LIVE'} {a['number'][:4]}**** · {a['status']} · equity ${a['equity']:,.0f} · buying power ${a['buying_power']:,.0f} · {a['latency_ms']} ms")
            add("keys", "ok" if a["keys"] == "own" else ("warn" if not self.live.armed else "fail"),
                f"from the {a['key_src']}" if a["keys"] == "own" else f"from the {a['key_src']} (not the project's key file) - fine for DRY RUN, refused when ARMED")
            owner = self.account_owner()
            add("account owner", "ok" if owner in (None, self.s.KEY) else "fail", f"claimed by {owner}" if owner else "not claimed yet (claimed on the first armed order)")
            if a["blocked"]:
                add("account", "fail", "trading is blocked")
            add("shorting", "ok" if a["shorting"] else "fail", "shorting enabled" if a["shorting"] else "shorting disabled - the book needs it")
            if a["equity"] < 25_000:
                add("PDT", "warn" if str(self.s.v["rebalance"]) in ("daily", "weekly", "monthly") else "fail",
                    "equity < $25,000: intraday round trips are limited by the pattern-day-trader rule")
            add("market", "ok", f"{'open' if a['market_open'] else 'closed'} · next open {a['next_open']:%a %H:%M} · next close {a['next_close']:%a %H:%M} ET")
        except Exception as e:                                                             # noqa: BLE001
            add("broker", "fail", f"cannot reach Alpaca: {type(e).__name__}: {str(e)[:140]}")
            return out
        u = D.load_universe()
        mon = now_et().to_period("M").to_timestamp()
        add("universe", "ok" if len(u) and u.month.max() >= mon else "warn", f"members through {u.month.max():%Y-%m}" if len(u) else "no universe")
        spy = D.load_bars(D.MARKET)
        if len(spy):
            age = (now_et() - spy.index.max()).days
            add("history", "ok" if age <= 4 else "warn", f"SIP cache through {spy.index.max():%Y-%m-%d %H:%M} ({age} d)")
        try:
            t0 = time.time()
            feed = "iex" if self.s.v["feed"] == "iex" else "sip"
            end = now_et() if feed == "iex" else now_et() - pd.Timedelta(minutes=D.SIP_DELAY_MIN)
            f = D.fetch_bars(self.dc, [D.MARKET], end - pd.Timedelta(days=5), end, feed=feed).get(D.MARKET)
            add("live feed", "ok" if f is not None and len(f) else "warn", f"{self.s.v['feed']}: newest SPY bar {f.index.max():%m-%d %H:%M} ({time.time() - t0:.1f} s)" if f is not None and len(f) else "no bars")
        except Exception as e:                                                             # noqa: BLE001
            add("live feed", "fail", f"{type(e).__name__}: {str(e)[:120]}")
        if hasattr(self.s, "model_info"):
            mi = self.s.model_info()
            if not mi:
                add("model", "fail", "no trained model - press 'Train model'")
            else:
                age = (now_et() - pd.Timestamp(mi["data_end"])).days
                limit = {"monthly": 31, "quarterly": 92, "yearly": 366}.get(str(self.s.v.get("retrain")), 92)
                same = mi.get("features") == self.s.feature_list()
                add("model", "ok" if age <= limit and same else "warn",
                    f"trained {mi['trained'][:16]} on data to {mi['data_end'][:16]} ({age} d old, retrain every {limit} d)" + ("" if same else " - FEATURES DIFFER from the settings: train again"))
        st = self.load_state()
        add("state", "warn" if st.get("stopped") or st.get("pending") else "ok",
            ("DRAWDOWN STOP active; " if st.get("stopped") else "") + ("an unfinished submit is recorded (check the journal); " if st.get("pending") else "") + f"high-water mark {st.get('hwm')}")
        probs = self.arm_problems()
        add("arming", "ok" if not probs else ("warn" if not self.live.armed else "fail"), "can be armed" if not probs else "; ".join(probs))
        return out
