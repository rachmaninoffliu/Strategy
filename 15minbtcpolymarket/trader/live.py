"""Live trading engine (no Qt): decides at the strategy's decision instants, sizes with Kelly from the strategy's historical
win probability (per leg / UTC hour), routes orders through a Broker (PaperBroker simulates; KalshiBroker sends real IOC orders to Kalshi),
settles and tracks equity. Fail-safe: any unresolved order state halts trading instead of guessing.  One venue: Kalshi supplies the signal AND takes the order."""
from __future__ import annotations

import csv
import dataclasses
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .adaptive import VOLUME_FALLBACK, AdaptiveStats, AutoConfig, decide, resolve_volume, snapshot
from .data import SETTLE_DELAY, WINDOW, extract_signals
from .broker import Broker, BrokerError, Fill, InsufficientFunds, NotPlaced, PaperBroker      # noqa: F401  (re-exported: tests and the GUI import them from here)
from .probmodel import SignalStats
from .strategy import MIN_HISTORY, StrategyParams, evaluate_quote_all, size_bet, throttled_sizing


@dataclass
class LiveConfig:
    capital: float = 1000.0
    params: StrategyParams = field(default_factory=StrategyParams)
    fee_model: str = "Kalshi"
    kelly_mult: float = 0.25
    max_frac: float = 0.02
    min_edge: float = 0.04                   # see BacktestConfig.min_edge
    max_contracts: int = 2000
    mode: str = "paper"                      # 'paper' | 'live'
    paper_slippage: float = 0.02
    max_slippage: float = 0.03               # live: the IOC order's limit price = quoted ask + this (it only caps what an order can pay; sizing uses paper_slippage)
    prob_source: str = "history"             # 'history' | 'manual'
    manual_p: float = 0.25
    lookback_days: int = 0                   # history window for p (0 = all local history)
    halflife_days: float = 0.0               # >0: exponentially decaying memory instead of the hard window
    dd_throttle: bool = True                 # scale the Kelly multiplier down as the drawdown below the session peak deepens (strategy.drawdown_throttle)
    core_hours: tuple = ()                   # 'by_hour' mode: hours with an informative prior (smaller shrinkage core_k); 0 = off
    core_k: float = 0.0
    near_k: float = 0.0                      # hours within near_width of a core hour use this lighter shrinkage (see SignalStats); 0 = off
    near_width: int = 2
    p_model: str = "overall"                 # 'overall' | 'by_hour'
    p_basis: str = "edge"                    # 'edge' (price-adjusted) | 'win_rate'
    hour_k: float = 40.0
    prior_n: float = 20.0
    daily_loss_limit: float = 0.10           # pause until next UTC day if equity falls this much from the day's start
    max_drawdown_stop: float = 0.25          # halt if equity falls this far below its peak
    min_history: int = MIN_HISTORY           # refuse to trade a leg whose probability rests on fewer historical signals (history mode only)
    auto_fund: bool = False                  # LIVE: before orders, move cash from your other Kalshi exchange shards onto the shard BTC 15m orders are funded from (what the Kalshi app does for you); never moves money out of the account
    fund_max: float = 0.0                    # ... keeping at most this much there (0 = the session's equity)
    grace_s: float = 25.0                    # how long after a decision instant we still try to evaluate it
    auto: Optional[AutoConfig] = None        # Auto mode (trader.adaptive): regime-aware memory, evidence-aware Kelly, equity ladder, liquidity cap; replaces kelly_mult / max_frac / max_contracts / halflife


def config_to_dict(cfg: LiveConfig) -> dict:
    """The session's settings as plain JSON data (stored in the ledger's session record, so a later reconcile can rebuild exactly what the session traded with)."""
    return dataclasses.asdict(cfg)


def _tuples(x):
    return tuple(_tuples(v) for v in x) if isinstance(x, (list, tuple)) else x


def config_from_dict(d: dict) -> LiveConfig:
    """Inverse of ``config_to_dict`` (unknown keys from a newer/older version are ignored)."""
    known = {f.name for f in dataclasses.fields(LiveConfig)}
    kw = {k: _tuples(v) for k, v in d.items() if k in known and k not in ("params", "auto")}
    pk = {f.name for f in dataclasses.fields(StrategyParams)}
    kw["params"] = StrategyParams(**{k: _tuples(v) for k, v in (d.get("params") or {}).items() if k in pk})
    ak = {f.name for f in dataclasses.fields(AutoConfig)}
    kw["auto"] = AutoConfig(**{k: _tuples(v) for k, v in d["auto"].items() if k in ak}) if d.get("auto") else None
    return LiveConfig(**kw)


def trades_frame(rows: list) -> pd.DataFrame:
    """Session trades (LiveEngine.trades) in the backtest's TRADE_COLS format, so the very same metrics code (metrics.compute_metrics) and tables apply to live and paper sessions."""
    from .backtest import TRADE_COLS
    recs = []
    for r in rows:
        et = pd.Timestamp(r["entry_time"], tz="UTC")
        st = pd.Timestamp(r["settle_time"], tz="UTC")
        recs.append(dict(entry_ts=int(et.timestamp()), settle_ts=int(st.timestamp()), entry_time=et.to_pydatetime(), market=r["market"], leg=r["leg"], hour=et.hour,
                         side=r["side"], quoted=r["quoted"], entry=r["entry"], contracts=r["contracts"], stake=r["stake"], fee=r["fee"], outcome=r["outcome"],
                         win=int(r["outcome"] == "WIN"), pnl=r["pnl"], ret_on_stake=r["pnl"] / r["stake"] if r["stake"] else 0.0, p_used=r["p_used"], f_full=r["f_full"],
                         f_used=r["f_used"], bankroll_before=r["equity_after"] - r["pnl"], equity_after=r["equity_after"], regime=r.get("regime", ""),
                         mult_used=r.get("mult_used", float("nan")), cap_used=r.get("cap_used", float("nan"))))
    extras = [c for c in ("session", "mode", "recovered", "venue") if any(c in r for r in rows)]          # ledger rows say which session / mode a trade belongs to
    for rec, r in zip(recs, rows):
        for c in extras:
            rec[c] = r.get(c, False if c == "recovered" else "")
    t = pd.DataFrame(recs, columns=TRADE_COLS + extras)
    return t.sort_values("settle_ts").reset_index(drop=True) if len(t) else t


def auto_active(cfg: LiveConfig) -> bool:
    """Auto mode needs the learned (history) probability, like the backtest's expanding mode."""
    return cfg.auto is not None and cfg.auto.enabled and cfg.prob_source == "history"


def build_stats(rows: pd.DataFrame, cfg: LiveConfig):
    """Probability model from the local history (all hours: the leg/hour estimate needs them)."""
    sig = extract_signals(rows, cfg.params)
    if auto_active(cfg):
        return AdaptiveStats.from_signals(sig, cfg.auto, mode=cfg.p_model, hour_k=cfg.hour_k, prior_n=cfg.prior_n, basis=cfg.p_basis, core_hours=cfg.core_hours, core_k=cfg.core_k, near_k=cfg.near_k, near_width=cfg.near_width)
    return SignalStats.from_signals(sig, mode=cfg.p_model, hour_k=cfg.hour_k, prior_n=cfg.prior_n, lookback_days=cfg.lookback_days, basis=cfg.p_basis, halflife_days=cfg.halflife_days, core_hours=cfg.core_hours, core_k=cfg.core_k, near_k=cfg.near_k, near_width=cfg.near_width)


@dataclass
class Position:
    window: int
    leg: str
    ticker: str
    side: str
    qty: int
    price: float
    fee: float
    p_used: float
    f_full: float
    f_used: float
    entry_ts: float
    order_id: str
    bankroll_before: float
    regime: str = ""
    mult: float = 0.0
    cap: float = 0.0
    tid: str = ""                      # ledger id of this trade
    requested: int = 0                 # contracts the engine asked for (after sizing and the depth guard); qty is what was filled
    note: str = ""                     # raw fill details from the broker (audit)


TRADE_FIELDS = ["entry_time", "settle_time", "market", "leg", "side", "quoted", "entry", "contracts", "stake", "fee", "outcome", "pnl",
                "p_used", "f_full", "f_used", "equity_after", "regime", "mult_used", "cap_used", "mode", "order_id", "venue"]


class LiveEngine:
    def __init__(self, cfg: LiveConfig, quotes, broker: Broker, stats,
                 emit: Callable[[str, dict], None], clock: Callable[[], float] = time.time,
                 log_dir: Optional[Path] = None, ledger=None, session_id: Optional[str] = None):
        self.cfg, self.quotes, self.broker, self.stats, self.emit, self.clock = cfg, quotes, broker, stats, emit, clock
        self.ledger = ledger                                  # trader.ledger.Ledger: every fill and settlement is appended to it so history survives restarts
        self.session_id = session_id or f"{datetime.now():%Y%m%d_%H%M%S}_{cfg.mode}"
        self._beat: Optional[int] = None                      # window of the last session heartbeat written to the ledger
        self._fund_window: Optional[int] = None               # window of the last auto-fund top-up
        self._session_closed = False
        self.venue = getattr(broker, "venue", "kalshi" if broker.mode == "live" else "paper")
        self.auto_on = auto_active(cfg) and isinstance(stats, AdaptiveStats)
        self.equity = float(cfg.capital)
        self.peak = float(cfg.capital)
        self.halted = False
        self.halt_reason = ""
        self.paused_until = 0.0
        self.day_key = None
        self.day_start_equity = float(cfg.capital)
        self.trades: list[dict] = []
        self._evaluated: set[tuple[int, int]] = set()
        self._leg_done: dict[int, set] = {}
        self._traded: set[int] = set()
        self._vol_seen: list[float] = []                  # recent last-minute volume readings (Auto liquidity cap fallback)
        self._shadow: dict[tuple[int, str], dict] = {}    # (window, leg) -> signal awaiting its result
        self._positions: dict[int, Position] = {}
        self._last_poll: dict[tuple[int, str], float] = {}
        self._csv: Optional[Path] = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._csv = log_dir / f"session_{self.session_id}.csv"
            with open(self._csv, "w", newline="") as fh:
                csv.DictWriter(fh, TRADE_FIELDS).writeheader()

    # ── helpers ───────────────────────────────────────────────────────────────
    def log(self, msg: str) -> None:
        self.emit("log", {"msg": f"{datetime.fromtimestamp(self.clock(), timezone.utc):%H:%M:%S}Z  {msg}"})

    def halt(self, reason: str) -> None:
        if not self.halted:
            self.halted, self.halt_reason = True, reason
            self.log(f"HALTED: {reason}")
            self.emit("halt", {"reason": reason})
            self.push_status()
            self._session_event("end", why=f"HALTED: {reason}")

    def _session_event(self, state: str, why: str = "") -> None:
        """Session record in the ledger: 'start' (with the settings), 'alive' once per window, 'end'.  A ledger problem never stops trading."""
        if self.ledger is None or self._session_closed:
            return
        try:
            first = state == "start"
            self.ledger.record_session(self.session_id, state, self.clock(), mode=self.broker.mode, venue=self.venue, why=why,
                                       capital=self.cfg.capital if first else None, cfg=config_to_dict(self.cfg) if first else None)
            self._session_closed = state == "end"
        except Exception as e:                                # noqa: BLE001
            self._session_closed = True                       # report once, then leave the ledger alone
            self.log(f"⚠ COULD NOT WRITE THE SESSION RECORD ({type(e).__name__}: {e})")

    def _heartbeat(self, now: float) -> None:
        w = int(now // WINDOW)
        if w != self._beat:
            first, self._beat = self._beat is None, w
            self._session_event("start" if first else "alive")

    def close(self, why: str = "stopped") -> None:
        """The session is over (the app stops it, or it halted): records the end so the ledger knows when this session was trading."""
        self._session_event("end", why=why)

    def p_for(self, leg: str, hour: int, cost: float, now: float) -> tuple[float, int]:
        """(win probability, number of historical signals behind it)."""
        if self.cfg.prob_source == "manual":
            return float(self.cfg.manual_p), 10**9
        p, n, _nh = self.stats.p(leg, hour, cost, now)
        return p, n

    VOLUME_FALLBACK = VOLUME_FALLBACK

    def _minute_volume(self, ticker: str, window: int, minute: int) -> Optional[float]:
        """Contracts traded in the last completed minute (liquidity cap input).  A failed read falls back to the median of the recent readings (or a small default) - never to 'unlimited'."""
        fn = getattr(self.quotes, "minute_volume", None)
        if fn is None:
            return None                                       # a quote source without volume (tests, custom feeds): only the hard ceiling applies
        try:
            v = fn(ticker, window, minute)
        except Exception:
            v = None
        had_history = bool(self._vol_seen)
        est, measured = resolve_volume(v, self._vol_seen)                 # the same rule the backtest applies
        if not measured:
            self.log(f"  no volume reading for the last minute - assuming {est:,.0f} contracts traded ({'median of recent readings' if had_history else 'default'})")
        return est

    def auto_snapshot(self, now: float) -> Optional[dict]:
        if not self.auto_on:
            return None
        leg = self.cfg.params.legs[0]
        hour = datetime.fromtimestamp(now, timezone.utc).hour
        try:
            return snapshot(self.cfg.auto, self.stats, leg, hour, 0.16 if leg == "DOG" else 0.92, now, self.equity, self.peak, self.cfg.fee_model, self.cfg.paper_slippage, self.cfg.min_edge)
        except Exception:
            return None

    def p_now(self, now: float) -> tuple[float, int]:
        """Headline probability for the status bar: first enabled leg, current UTC hour, a typical price."""
        leg = self.cfg.params.legs[0]
        hour = datetime.fromtimestamp(now, timezone.utc).hour
        return self.p_for(leg, hour, 0.16 if leg == "DOG" else 0.92, now)

    def _ledger_open(self, pos: "Position", quoted: float) -> str:
        """Record the fill the moment it happens (a position still open when the app closes is then not lost).  A ledger problem is reported loudly but never stops trading."""
        if self.ledger is None:
            return ""
        try:
            return self.ledger.record_open(mode=self.broker.mode, venue=self.venue, session=self.session_id, market=pos.ticker, window=pos.window, leg=pos.leg, side=pos.side, quoted=quoted, entry=pos.price,
                                           contracts=pos.qty, stake=pos.qty * pos.price + pos.fee, fee=pos.fee, p_used=pos.p_used, f_full=pos.f_full, f_used=pos.f_used,
                                           bankroll_before=pos.bankroll_before, order_id=pos.order_id, entry_ts=pos.entry_ts, regime=pos.regime, mult_used=pos.mult, cap_used=pos.cap, note=pos.note,
                                           requested=pos.requested)
        except Exception as e:                                # noqa: BLE001
            self.log(f"⚠ COULD NOT WRITE THE TRADE LEDGER ({type(e).__name__}: {e}) - this trade is only in memory and the session CSV")
            return ""

    def _ledger_settle(self, pos: "Position", row: dict, now: float, account_bp: Optional[float] = None) -> None:
        if self.ledger is None or not pos.tid:
            return
        try:
            self.ledger.record_settle(pos.tid, settle_ts=now, outcome=row["outcome"], pnl=row["pnl"], equity_after=row["equity_after"], account_bp=account_bp)
        except Exception as e:                                # noqa: BLE001
            self.log(f"⚠ COULD NOT WRITE THE TRADE LEDGER ({type(e).__name__}: {e}) - the settlement is only in memory and the session CSV")

    def _window_done(self, window: int) -> bool:
        return set(self.cfg.params.legs) <= self._leg_done.get(window, set())

    def next_decision_in(self, now: float) -> Optional[float]:
        """Seconds until the next decision instant inside a strategy hour (None if hours is empty)."""
        base = int(now // WINDOW) * WINDOW
        for k in range(0, 24 * 4 + 2):
            w = base + k * WINDOW
            if datetime.fromtimestamp(w, timezone.utc).hour in self.cfg.params.hours and not self._window_done(w):
                for m in self.cfg.params.minutes:
                    t = w + 60 * m
                    if t + self.cfg.grace_s > now and (w, m) not in self._evaluated:
                        return max(0.0, t - now)
        return None

    def sleep_hint(self, now: float) -> float:
        """How long the driver loop may sleep before it must tick again: coarse while nothing is due, fine (10-50 ms) just before a decision instant so the quote is read right at the minute mark -
        the backtest's candle close IS the quote at that mark."""
        nxt = self.next_decision_in(now)
        if nxt is None or nxt > 3.0:
            return 0.25
        return max(0.005, min(0.05, nxt))

    def push_status(self) -> None:
        now = self.clock()
        p, n = self.p_now(now)
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        self.emit("status", dict(
            equity=self.equity, pnl=self.equity - self.cfg.capital, peak=self.peak,
            drawdown=(self.equity - self.peak) / self.peak if self.peak else 0.0, trades=len(self.trades), wins=wins,
            p=p, n=n, halted=self.halted, halt_reason=self.halt_reason, mode=self.cfg.mode,
            paused=now < self.paused_until, next_in=self.next_decision_in(now),
            open_positions=len(self._positions), auto=self.auto_snapshot(now)))

    # ── main loop step ────────────────────────────────────────────────────────
    def tick(self) -> None:
        now = self.clock()
        self._heartbeat(now)
        self._settle_due(now)
        if self.halted:
            return
        self._risk_checks(now)
        window = int(now // WINDOW) * WINDOW
        sec = now - window
        hour = datetime.fromtimestamp(window, timezone.utc).hour
        self._prefund(now, window, sec)
        if hour in self.cfg.params.hours and not self._window_done(window):
            for m in self.cfg.params.minutes:
                if 60 * m <= sec < 60 * m + self.cfg.grace_s and (window, m) not in self._evaluated:
                    self._on_decision(window, m, hour, now)
                    if self._window_done(window) or self.halted:
                        break

    # ── auto-fund (live only) ──
    def _fund_target(self) -> float:
        return min(self.equity, self.cfg.fund_max) if self.cfg.fund_max > 0 else self.equity

    def ensure_cash(self, why: str, wait_s: float = 15.0, amount: Optional[float] = None) -> bool:
        """Top the trading shard up to the session's bankroll - or move exactly ``amount`` dollars - from the account's other shards.  True if cash arrived.  A problem here is reported, never fatal: the order path still answers for itself."""
        fund = getattr(self.broker, "fund", None)
        if fund is None or self.broker.mode != "live":
            return False
        try:
            if amount is None:
                bp = self.broker.buying_power()
                need = self._fund_target()
                if bp is None or bp >= need - 1.0:
                    return False
                amount = need - bp
            r = fund(amount, wait_s=wait_s)
        except Exception as e:                                # noqa: BLE001
            msg = f"auto-fund ({why}) failed: {e}"
            self.log(f"  ⚠ {msg}")
            self.emit("alert", {"title": "Auto-fund failed", "msg": msg})
            self._session_event("alert", why=msg[:300])
            return False
        if r["moved"] <= 0:
            return False
        src = ", ".join(f"${m:,.2f} from shard {s}" for s, m in r["moves"])
        self.log(f"  auto-fund ({why}): moved {src} onto the trading shard -> ${r['balance']:,.2f} there" + ("" if r["arrived"] else " (accepted by Kalshi, not visible yet - it completes asynchronously)"))
        return bool(r["arrived"])

    def _prefund(self, now: float, window: int, sec: float) -> None:
        """Keep the trading shard funded OUTSIDE the decision path: at session start and once early in each window (never inside the decision second, where a transfer would only add delay)."""
        if not self.cfg.auto_fund or self.broker.mode != "live" or self.halted or window == self._fund_window:
            return
        first = self._fund_window is None
        if first and 650 <= sec < 60 * max(self.cfg.params.minutes) + self.cfg.grace_s + 5:
            return                                            # session started right at a decision instant: leave the order path alone, the fallback in the order path covers it
        if not first and not 30 <= sec < 300:
            return
        self._fund_window = window
        self.ensure_cash("session start" if first else "top-up before the window")

    def _risk_checks(self, now: float) -> None:
        day = datetime.fromtimestamp(now, timezone.utc).date()
        if day != self.day_key:
            self.day_key, self.day_start_equity = day, self.equity
        if self.cfg.max_drawdown_stop and self.peak > 0 and self.equity <= self.peak * (1 - self.cfg.max_drawdown_stop):           # 0 = off, as in the backtest
            self.halt(f"max drawdown stop hit ({(self.equity / self.peak - 1) * 100:.1f}% from peak)")
        elif self.cfg.daily_loss_limit and now >= self.paused_until and self.equity <= self.day_start_equity * (1 - self.cfg.daily_loss_limit):
            nxt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() + 86400
            self.paused_until = nxt
            self.log(f"daily loss limit hit ({(self.equity / self.day_start_equity - 1) * 100:.1f}%) - no new trades until next UTC day")

    def _on_decision(self, window: int, minute: int, hour: int, now: float) -> None:
        ticker = self.quotes.window_ticker(window)
        if not ticker:
            return
        t_read = time.perf_counter()
        book_fn = getattr(self.quotes, "book", None)
        book = book_fn(ticker) if book_fn is not None else None             # the order book: one request gives the signal quote AND the depth the order is checked against
        q = book.top() if book is not None else self.quotes.quote(ticker)
        read_ms = (time.perf_counter() - t_read) * 1000.0
        if q is None:
            return                                        # transient: retry on the next tick while inside the grace period
        self._evaluated.add((window, minute))
        bid, ask = q
        done = self._leg_done.setdefault(window, set())
        cands = [s for s in evaluate_quote_all(bid, ask, minute, hour, self.cfg.params) if s.leg not in done]
        if not cands:
            self.log(f"{ticker} min {minute}: no signal (YES bid/ask {bid:.2f}/{ask:.2f})")
            return
        slip = self.cfg.paper_slippage                    # the SAME assumed slippage as the backtest and paper mode sizes the bet; max_slippage only sets the live order's limit price
        bankroll = self.equity
        if self.broker.mode == "live":
            try:
                bp = self.broker.buying_power()
            except Exception as e:
                self.halt(f"cannot read buying power ({type(e).__name__})")
                return
            if bp is not None:
                bankroll = min(bankroll, bp)
                if bp < 1.0 and self.cfg.auto_fund and self.ensure_cash("no cash at the decision instant", wait_s=8.0):
                    try:
                        bp = self.broker.buying_power()
                        bankroll = min(self.equity, bp)
                    except Exception:                         # noqa: BLE001
                        pass
                if bp < 1.0:                                  # cannot buy even one contract: say so loudly instead of quietly sizing to nothing
                    msg = (f"the cash available to this market is ${bp:,.2f} - no order can be placed. Kalshi funds an order only from the exchange shard its market trades on; "
                           f"check the account box / pre-flight for where your cash is and move it there.")
                    self.log(f"  ⚠ {msg}")
                    self.emit("alert", {"title": "No cash available for this market", "msg": msg})
                    self._session_event("alert", why=msg[:300])
        manual = self.cfg.prob_source == "manual"
        sized = []
        vol, vol_read = None, False
        for s in cands:
            done.add(s.leg)
            self._shadow[(window, s.leg)] = dict(ticker=ticker, side=s.side, cost=s.cost, minute=minute, hour=hour)
            pred = self.stats.predict(s.leg, hour, s.cost, now) if self.auto_on else None
            p, n = (pred.p_used, pred.n) if pred is not None else self.p_for(s.leg, hour, s.cost, now)
            self.log(f"{ticker} min {minute}: SIGNAL {s.leg} buy {s.side} @ {s.cost:.2f} (YES {bid:.2f}/{ask:.2f}); p={p * 100:.1f}% (n={n if n < 10**8 else 'manual'})")
            if not manual and (n < self.cfg.min_history or not np.isfinite(p)):
                self.log(f"  {s.leg}: only {n} historical signals (<{self.cfg.min_history}) - not trading it")
                continue
            regime, mult_u, cap_u = "", 0.0, 0.0
            if pred is not None:
                if not vol_read:
                    vol, vol_read = self._minute_volume(ticker, window, minute), True
                dec = decide(self.cfg.auto, self.stats, s.leg, hour, s.cost, now, self.equity, self.peak, bankroll, vol, self.cfg.fee_model, slip, self.cfg.min_edge, pred)
                sz, regime, mult_u, cap_u = dec.sizing, dec.regime, dec.mult, dec.cap
                liq = f"{dec.liq_cap:,} (last-minute volume {vol:,.0f})" if dec.liq_cap is not None else "ceiling (no volume reading)"
                self.log(f"  AUTO: regime {regime}; edge fast {pred.d_fast * 100:+.1f}c / slow {pred.d_slow * 100:+.1f}c / mix {pred.d_mix * 100:+.1f}c +/-{pred.se * 100:.1f}c; weights {'/'.join(f'{w * 100:.0f}%' for w in pred.weights)}; "
                         f"Kelly x{mult_u:.3f} (ladder tier {dec.tier + 1}), stake cap {cap_u * 100:.2f}%, max contracts {dec.max_contracts:,} [liquidity {liq}]")
            else:
                px = min(s.cost + slip, 0.99)
                mult_u, cap_u = throttled_sizing(self.cfg.kelly_mult, self.cfg.max_frac, self.equity, self.peak, self.cfg.dd_throttle)
                sz = size_bet(bankroll, p, px, self.cfg.fee_model, mult_u, cap_u, self.cfg.max_contracts, self.cfg.min_edge)
            if sz.contracts < 1:
                self.log(f"  {s.leg}: Kelly no bet ({sz.reason}); p={p * 100:.1f}% vs all-in cost {sz.cost_all * 100:.1f}c")
                continue
            sized.append((sz.edge, s, p, sz, regime, mult_u, cap_u))
        if not sized:
            return
        if window in self._traded:
            self.log("  already traded this window - extra signals tracked only")
            return
        if now < self.paused_until:
            self.log("  paused (daily loss limit) - signals tracked, not traded")
            return
        _edge, s, p, sz, regime, mult_u, cap_u = max(sized, key=lambda x: x[0])
        max_cost = min(s.cost + self.cfg.max_slippage, 0.99)
        contracts = sz.contracts
        depth_note = ""
        if book is not None:                              # execution quality: never send more than the book can fill inside the limit
            fillable, avg = book.walk(s.side, contracts, max_cost)
            if fillable < 1:
                self.log(f"  {s.leg}: no liquidity at or below {max_cost:.2f} for {s.side} ({self._fmt_asks(book, s.side)}) - flat")
                return
            if fillable + 1e-9 < contracts:
                self.log(f"  {s.leg}: book fills only {int(fillable)} of {contracts} contracts at or below {max_cost:.2f} - sending {int(fillable)}")
                contracts = int(fillable)
                _fl2, avg = book.walk(s.side, contracts, max_cost)
            depth_note = f"book {bid:.2f}/{ask:.2f} expected avg {avg:.4f} read {read_ms:.0f}ms"
        self.log(f"  {s.leg}: Kelly full {sz.f_full * 100:.1f}% -> using {sz.f_used * 100:.2f}% of ${bankroll:,.0f}: {contracts} contracts, max {max_cost:.2f}")
        fill, funded = None, False
        while True:
            try:
                fill = self.broker.buy(ticker, s.side, contracts, max_cost, s.cost)
                break
            except InsufficientFunds as e:
                if self.cfg.auto_fund and not funded and self.clock() - now < 15.0:
                    funded = True                             # what the Kalshi app does: move the cash, then send the order (same IOC limit: it cannot pay more than before)
                    if self.ensure_cash("order rejected for cash", wait_s=8.0, amount=contracts * max_cost * 1.02 + 0.5):
                        continue
                self.log(f"  ⚠ ORDER NOT PLACED: {e}")
                self.emit("alert", {"title": "Order rejected: not enough cash", "msg": str(e)})
                self._session_event("alert", why=str(e)[:300])
                return                                        # certain not to have been placed: the session keeps running (and trades again once the cash is where Kalshi looks for it)
            except NotPlaced as e:
                self.log(f"  order not placed: {e} - flat")
                return
            except BrokerError as e:
                self.halt(f"order problem: {e}")
                return
            except Exception as e:                           # anything unexpected around an order: stop, never guess
                self.halt(f"unexpected order error ({type(e).__name__})")
                return
        if fill is None or fill.qty < 1:
            self.log("  no fill (IOC found nothing at the limit) - flat")
            return
        self._traded.add(window)
        self._positions[window] = Position(window, s.leg, ticker, s.side, fill.qty, fill.price, fill.fee, p, sz.f_full, sz.f_used,
                                           now, fill.order_id, self.equity, regime, mult_u, cap_u)
        self._positions[window].note = (fill.note + (" | " if fill.note and depth_note else "") + depth_note)[:400]
        self._positions[window].requested = contracts
        self._positions[window].tid = self._ledger_open(self._positions[window], s.cost)
        self.log(f"  FILLED {fill.qty} {s.side} @ {fill.price:.3f} (fee ${fill.fee:.2f}) [{self.broker.mode}]")
        self.emit("trade_open", dict(market=ticker, side=s.side, qty=fill.qty, price=fill.price))
        self.push_status()

    @staticmethod
    def _fmt_asks(book, side: str) -> str:
        lv = book.asks(side)[:3]
        return ", ".join(f"{q:g}@{p:.2f}" for p, q in lv) or "empty side"

    def _settle_due(self, now: float) -> None:
        for window, leg in sorted(self._shadow):
            if now < window + WINDOW + SETTLE_DELAY:
                continue
            if now - self._last_poll.get((window, leg), 0) < 3.0:
                continue
            self._last_poll[(window, leg)] = now
            sh = self._shadow[(window, leg)]
            pos = self._positions.get(window)
            res = self.quotes.result(sh["ticker"])
            if res is None:
                if now > window + WINDOW + 20 * 60 and pos is not None and pos.leg == leg:
                    self.halt(f"{sh['ticker']} still unresolved 20 min after close with an open position")
                continue
            win = (res == "yes") == (sh["side"] == "YES")
            if self.auto_on:
                self.stats.add_live(leg, sh["hour"], window, win, sh["cost"], minute=sh.get("minute", 12))
            else:
                self.stats.add_live(leg, sh["hour"], window, win, sh["cost"])
            del self._shadow[(window, leg)]
            if pos is None or pos.leg != leg:
                self.log(f"{sh['ticker']} resolved {res.upper()} - untraded {leg} signal {'WON' if win else 'lost'}")
                self.push_status()
                continue
            del self._positions[window]
            pnl = pos.qty * (1.0 if win else 0.0) - pos.qty * pos.price - pos.fee
            self.equity += pnl
            self.peak = max(self.peak, self.equity)
            row = dict(entry_time=datetime.fromtimestamp(pos.entry_ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                       settle_time=datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                       market=pos.ticker, leg=pos.leg, side=pos.side, quoted=sh["cost"], entry=pos.price, contracts=pos.qty,
                       stake=pos.qty * pos.price + pos.fee, fee=pos.fee, outcome="WIN" if win else "LOSS", pnl=pnl,
                       p_used=pos.p_used, f_full=pos.f_full, f_used=pos.f_used, equity_after=self.equity,
                       regime=pos.regime, mult_used=pos.mult, cap_used=pos.cap, mode=self.broker.mode, order_id=pos.order_id, venue=self.venue)
            self.trades.append(row)
            bp = None
            if self.broker.mode == "live":                    # reconcile: the real account figure goes next to the session's own P&L (and into the ledger as an audit trail)
                try:
                    bp = self.broker.buying_power()
                except Exception:                            # noqa: BLE001
                    bp = None
            self._ledger_settle(pos, row, now, bp)
            if self._csv:
                with open(self._csv, "a", newline="") as fh:
                    csv.DictWriter(fh, TRADE_FIELDS).writerow(row)
            self.log(f"{pos.ticker} resolved {res.upper()} -> {pos.leg} {'WIN' if win else 'LOSS'} {pnl:+.2f}  equity ${self.equity:,.2f}")
            self.emit("trade_close", row)
            if self.broker.mode == "live":
                self.log(f"  account buying power now ${bp:,.2f} (session equity ${self.equity:,.2f})" if bp is not None else "  (could not read account buying power for reconciliation)")
            self.push_status()

    def kill(self, reason: str = "kill switch") -> None:
        try:
            self.broker.cancel_all()
        except Exception as e:
            self.log(f"cancel-all failed ({type(e).__name__}) - check open orders on Kalshi")
        self.halt(reason)
