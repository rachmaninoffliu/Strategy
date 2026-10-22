"""Historical simulation: universe download + cache, cross-venue matching, signal generation, fill modelling,
capital-lockup accounting and institutional performance metrics.

Design rules (all deliberate, all reported in ``BacktestResult.assumptions``):
* Decision at bar *t*, fill at the quotes observed *d* bars later (``exec_delay_bars``, default 1) - no trading on a
  price the desk could not have acted on.
* Entry filter uses the **scheduled** settlement time (+ venue lag); capital is released at the **actual** settlement
  time. The realised payout of every leg comes from the leg's **actual** outcome, so a resolution mismatch between the
  venues shows up as a loss instead of being assumed away.
* Kalshi history gives bid/ask closes; Polymarket history is trade/mid price only -> synthetic half-spread applied.
* No historical L2 exists on either venue -> depth is an assumption (``assumed_depth_usd`` per structure).
"""
from __future__ import annotations

import asyncio
import bisect
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import DAYS_PER_YEAR, MIN_DAYS, FeeConfig
from connectors import mapper
from connectors.kalshi_client import KalshiClient
from connectors.poly_client import PolyClient
from core.models import ArbType, Candle, Market, Relation, Side, TradeLog, Venue, iso, utcnow
from core.pricing import annualize, annualize_arr, chunk_fee, kelly_fraction, unit_fee
from database import Database

ProgressCB = Optional[Callable[[int, str], None]]


# ── configuration / results ───────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    start: datetime = field(default_factory=lambda: utcnow() - timedelta(days=180))
    end: datetime = field(default_factory=utcnow)
    initial_capital: float = 10_000.0
    max_capital_per_trade: float = 1_000.0
    max_concurrent_capital: Optional[float] = None       # C_max; None -> initial capital
    hurdle_apy: float = 0.07
    fees: FeeConfig = field(default_factory=FeeConfig)
    slippage_bps: float = 25.0
    sizing: str = "fixed"                                # fixed | kelly
    kelly_mult: float = 0.25
    basis_risk_prob: float = 0.02                        # P(total loss) from resolution mismatch, cross-venue legs
    exec_delay_bars: int = 1
    bar_minutes: int = 1440                              # 1440 (daily) or 60 (hourly)
    min_match_confidence: float = 0.80
    min_volume: float = 500.0
    assumed_depth_usd: float = 500.0
    poly_half_spread: float = 0.005
    max_stale_bars: int = 3
    risk_free: float = 0.05
    types: tuple = ("parity", "subset", "mece")
    max_structures: int = 300
    max_mece_events: int = 60
    kalshi_max_pages: int = 60
    poly_max_pages: int = 60
    kalshi_lag_h: float = 2.0
    poly_lag_h: float = 4.0
    date_tol_hours: float = 24.0
    subset_tol_hours: float = 2.0
    min_days: float = MIN_DAYS
    consistency_tol: float = 0.10          # data gate: sum of leg MID prices must be >= (1-tol) x payout
    max_net_return: float = 0.30           # reject 'edges' above this (definition / data mismatch)
    spike_threshold: float = 0.12          # Polymarket price-only history: drop isolated prints this far off
    include_open: bool = False                           # history mode: also markets still open
    unconstrained: bool = False                          # history mode: log every signal, no capital limit

    @property
    def c_max(self) -> float:
        return self.max_concurrent_capital if self.max_concurrent_capital else self.initial_capital


@dataclass
class Structure:
    sid: str
    arb_type: ArbType
    name: str
    legs: list[tuple[Market, Side]]
    min_pu: float
    max_pu: float
    confidence: float
    variant: str = ""

    @property
    def cross_venue(self) -> bool:
        return len({m.venue for m, _ in self.legs}) > 1


@dataclass
class Signal:
    st: Structure
    t_fill: int                    # epoch seconds
    prices: list[float]            # per-leg fill price incl. slippage
    unit_cost: float               # per unit incl. continuous fee approximation
    net_return: float
    apy: float
    sched_settle: int
    release: Optional[int]         # actual capital release (None while any leg is still open)
    payout_pu: Optional[float]     # realised payout per unit (None while open)


@dataclass
class BacktestResult:
    cfg: BacktestConfig
    trades: list[TradeLog] = field(default_factory=list)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    locked: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    drawdown: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ── helpers ───────────────────────────────────────────────────────────────────
def _ts(dt: Optional[datetime]) -> Optional[int]:
    return int(dt.timestamp()) if dt else None


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class Cancelled(Exception):
    pass


class Backtester:
    def __init__(self, cfg: BacktestConfig, db: Optional[Database] = None, progress: ProgressCB = None,
                 cancel: Optional[asyncio.Event] = None):
        self.cfg = cfg
        self.db = db
        self._progress = progress
        self._cancel = cancel
        self.assumptions: list[str] = []
        self.stats: dict = {}
        self._degraded: set[str] = set()

    def _p(self, pct: int, msg: str):
        if self._progress:
            self._progress(pct, msg)
        if self._cancel and self._cancel.is_set():
            raise Cancelled()

    # ══ pipeline ══════════════════════════════════════════════════════════════
    async def run(self) -> BacktestResult:
        cfg = self.cfg
        t0 = time.time()
        now = utcnow()
        cfg.end = min(cfg.end, now)
        async with KalshiClient() as kc, PolyClient() as pc:
            kal, pol = await self._load_universe(kc, pc)
            self._p(22, f"Matching {len(kal)} Kalshi x {len(pol)} Polymarket contracts...")
            structs = await self._build_structures(kc, kal, pol)
            self._p(30, f"{len(structs)} candidate structures; downloading price history...")
            series = await self._load_series(kc, pc, structs)
        self._p(82, "Scanning history for hurdle-clearing entries...")
        signals = self._signals(structs, series, now)
        self._p(90, f"{len(signals)} signals; simulating portfolio...")
        res = self._simulate(signals, now)
        res.assumptions = self._assumptions()
        res.stats.update(self.stats)
        res.stats.update({"signals": len(signals), "structures": len(structs),
                          "seconds": round(time.time() - t0, 1)})
        if self.db:
            try:
                self.db.save_backtest_run({"start": iso(cfg.start), "end": iso(cfg.end),
                                           "capital": cfg.initial_capital, "hurdle": cfg.hurdle_apy,
                                           "sizing": cfg.sizing, "slippage_bps": cfg.slippage_bps},
                                          res.metrics)
            except Exception:
                pass
        self._p(100, f"Done in {res.stats['seconds']}s: {len(res.trades)} trades")
        return res

    # ── 1. universe ───────────────────────────────────────────────────────────
    async def _load_universe(self, kc: KalshiClient, pc: PolyClient):
        cfg = self.cfg
        k_task = kc.get_settled_markets(cfg.start, cfg.end, cfg.kalshi_max_pages, cfg.min_volume,
                                        lambda m: self._p(8, m))
        p_task = pc.get_closed_events(cfg.start, cfg.end, cfg.poly_max_pages, cfg.min_volume,
                                      lambda m: self._p(14, m))
        res = await asyncio.gather(k_task, p_task, return_exceptions=True)
        out = []
        for name, r in zip(("Kalshi", "Polymarket"), res):
            if isinstance(r, Cancelled):
                raise r
            if isinstance(r, Exception):
                self.assumptions.append(f"WARNING: {name} settled-market download failed ({r}); results are partial.")
                out.append([])
            else:
                out.append(r)
        kal, pol = out
        self.assumptions += [f"WARNING: {w}" for w in pc.warnings]
        if not kal or not pol:
            self.assumptions.append(f"WARNING: {'Kalshi' if not kal else 'Polymarket'} returned no settled markets - "
                                    f"no cross-venue structures can be formed; results are not meaningful.")
        if cfg.include_open:
            self._p(18, "Adding currently-open markets...")
            ko, po = await asyncio.gather(kc.get_open_events(30), pc.get_open_events(20), return_exceptions=True)
            if not isinstance(ko, Exception):
                kal += [m for m in ko if m.status == "open"]
            if not isinstance(po, Exception):
                pol += [m for m in po if m.status == "open"]
        if self.db:
            self.db.upsert_markets(kal + pol)
        self.stats.update({"kalshi_universe": len(kal), "poly_universe": len(pol)})
        return kal, pol

    # ── 2. structures ─────────────────────────────────────────────────────────
    async def _build_structures(self, kc: KalshiClient, kal: list[Market], pol: list[Market]) -> list[Structure]:
        cfg = self.cfg
        structs: list[Structure] = []
        k_ok = [m for m in kal if m.volume >= cfg.min_volume]
        p_ok = [m for m in pol if m.volume >= cfg.min_volume and m.yes_token]
        if "parity" in cfg.types:
            pairs = mapper.match_pairs(k_ok, p_ok, cfg.date_tol_hours, cfg.min_match_confidence)
            for pr in pairs:
                combos = ([(Side.YES, Side.NO), (Side.NO, Side.YES)] if pr.relation is Relation.EQUIVALENT
                          else [(Side.YES, Side.YES), (Side.NO, Side.NO)])
                for sk, sp in combos:
                    structs.append(Structure(f"P|{pr.a.key}|{sk.value}|{pr.b.key}|{sp.value}", ArbType.PARITY,
                                             f"{pr.a.title or pr.a.event_title}  <->  {pr.b.title}",
                                             [(pr.a, sk), (pr.b, sp)], 1.0, 1.0, pr.confidence,
                                             f"K {sk.value} + P {sp.value}"))
            self.stats["pairs"] = len(pairs)
        if "subset" in cfg.types:
            edges = mapper.build_subset_edges(k_ok + p_ok, cfg.subset_tol_hours)
            edges = [e for e in edges if e.confidence >= cfg.min_match_confidence]
            for e in edges:
                structs.append(Structure(f"S|{e.outer.key}|{e.inner.key}", ArbType.SUBSET,
                                         f"{(e.outer.event_title or e.outer.title)[:60]}: "
                                         f"{e.inner.outcome_label or e.inner.title} ⊆ "
                                         f"{e.outer.outcome_label or e.outer.title}",
                                         [(e.outer, Side.YES), (e.inner, Side.NO)], 1.0, 2.0, e.confidence,
                                         "buy YES(B) + NO(A)"))
            self.stats["subset_edges"] = len(edges)
        if "mece" in cfg.types:
            structs += await self._mece_structures(kc, k_ok, pol)
        # keep the most promising (confidence x liquidity) to bound the number of history downloads
        def prio(s: Structure) -> float:
            return s.confidence * math.log1p(min(m.volume for m, _ in s.legs))
        if len(structs) > cfg.max_structures:
            # round-robin across arbitrage types so 17k ladder edges cannot crowd out parity / MECE baskets
            buckets: dict[ArbType, list[Structure]] = {}
            for st in sorted(structs, key=prio, reverse=True):
                buckets.setdefault(st.arb_type, []).append(st)
            picked: list[Structure] = []
            i = 0
            while len(picked) < cfg.max_structures and any(buckets.values()):
                for lst in buckets.values():
                    if i < len(lst) and len(picked) < cfg.max_structures:
                        picked.append(lst[i])
                i += 1
            self.assumptions.append(f"Structures truncated to {cfg.max_structures} of {len(structs)} (round-robin "
                                    f"across types, ranked by confidence x liquidity).")
            structs = picked
        return structs

    async def _mece_structures(self, kc: KalshiClient, k_ok: list[Market], pol: list[Market]) -> list[Structure]:
        cfg = self.cfg
        baskets: list[mapper.Basket] = []
        # Kalshi: event metadata (mutually_exclusive flag + complete outcome list) via /events/{ticker}
        by_ev: dict[str, float] = {}
        n_ev: dict[str, int] = {}
        for m in k_ok:
            by_ev[m.event_id] = by_ev.get(m.event_id, 0.0) + m.volume
            n_ev[m.event_id] = n_ev.get(m.event_id, 0) + 1
        cands = sorted((e for e in by_ev if n_ev[e] >= 2), key=lambda e: -by_ev[e])[: cfg.max_mece_events]
        sem = asyncio.Semaphore(6)

        async def one(e: str):
            async with sem:
                try:
                    return await kc.get_event_full(e)
                except Exception:
                    return None

        got = await asyncio.gather(*(one(e) for e in cands))
        k_all: list[Market] = []
        for g in got:
            if g:
                k_all += g[1]
        baskets += mapper.kalshi_baskets(k_all, 60, historical=True)
        baskets += mapper.poly_baskets(pol, 60, historical=True)
        out: list[Structure] = []
        for bk in baskets:
            n = len(bk.outcomes)
            for side in (Side.YES, Side.NO):
                if side is Side.YES and not bk.exhaustive:
                    continue
                legs = [(cands_[0], side) for cands_ in bk.outcomes]
                if any(m.settle_value is None and m.status == "settled" for m, _ in legs):
                    continue
                if side is Side.YES:
                    mn, mx, var = 1.0, 1.0, "buy-all-YES"
                else:
                    mn = float(n - 1)
                    mx = float(n - 1) if bk.exhaustive else float(n)
                    var = "buy-all-NO"
                out.append(Structure(f"M|{bk.scope}|{legs[0][0].event_id}|{side.value}", ArbType.MECE,
                                     bk.name[:90], legs, mn, mx, bk.confidence, var))
        self.stats["mece_baskets"] = len(baskets)
        return out

    # ── 3. history ────────────────────────────────────────────────────────────
    async def _load_series(self, kc: KalshiClient, pc: PolyClient, structs: list[Structure]) -> dict[str, list[Candle]]:
        cfg = self.cfg
        period = cfg.bar_minutes * 60
        need: dict[str, Market] = {}
        for s in structs:
            for m, _ in s.legs:
                need[m.key] = m
        keys = list(need)
        out: dict[str, list[Candle]] = {}
        done = 0
        sem = asyncio.Semaphore(8)
        now_ts = int(utcnow().timestamp())

        async def one(k: str):
            nonlocal done
            m = need[k]
            async with sem:
                self._p(30 + int(50 * done / max(len(keys), 1)), f"History {done}/{len(keys)}: {m.market_id[:40]}")
                lo = max(_ts(cfg.start) or 0, _ts(m.open_time) or 0)
                hi = min(_ts(m.close_time) or now_ts, now_ts)
                lo = lo // period * period
                if hi <= lo:
                    out[k] = []
                    done += 1
                    return
                cached = self.db.candle_meta(m.venue, m.market_id, period) if self.db else None
                if cached and cached["start_ts"] <= lo + period and cached["end_ts"] >= hi - period:
                    out[k] = self.db.load_candles(m.venue, m.market_id, period)
                    if cached["degraded"]:
                        self._degraded.add(k)
                else:
                    try:
                        if m.venue is Venue.KALSHI:
                            cs = await kc.get_candlesticks(m, _dt(lo), _dt(hi), cfg.bar_minutes)
                            deg = False
                        else:
                            cs, deg = await pc.get_price_history(m.yes_token, _dt(lo), _dt(hi), cfg.bar_minutes)
                        if deg:
                            self._degraded.add(k)
                        if self.db and cs:
                            self.db.save_candles(m.venue, m.market_id, period, cs, lo, hi, deg)
                        out[k] = cs
                    except Cancelled:
                        raise
                    except Exception:
                        out[k] = []
                done += 1

        results = await asyncio.gather(*(one(k) for k in keys), return_exceptions=True)
        for r in results:
            if isinstance(r, Cancelled):
                raise r
        self.stats["markets_with_history"] = sum(1 for v in out.values() if v)
        self.stats["markets_requested"] = len(keys)
        return out

    # ── 4. signals ────────────────────────────────────────────────────────────
    @staticmethod
    def _despike(cs: list[Candle], thr: float) -> list[Candle]:
        """Drop isolated Polymarket prints that jump away from BOTH neighbours in opposite directions.
        Only removes data (so it can only remove candidate trades, never create one)."""
        if thr <= 0 or len(cs) < 3:
            return cs
        px = [c.price for c in cs]
        keep = [True] * len(cs)
        for i in range(1, len(cs) - 1):
            a, b, c = px[i - 1], px[i], px[i + 1]
            if a is None or b is None or c is None:
                continue
            if (b - a) * (c - b) < 0 and abs(b - a) > thr and abs(b - c) > thr:
                keep[i] = False
        return [c for c, k in zip(cs, keep) if k]

    def _leg_quotes(self, m: Market, side: Side, cs: list[Candle], grid: np.ndarray):
        """-> (ask, mid): price paid to BUY ``side`` and its mid, on each grid time (NaN if no fresh / valid quote)."""
        cfg = self.cfg
        nan = np.full(grid.shape, np.nan)
        if not cs:
            return nan, nan
        if m.venue is Venue.POLYMARKET:
            cs = self._despike(cs, cfg.spike_threshold)
        ts = np.array([c.ts for c in cs], dtype=float)
        idx = np.searchsorted(ts, grid, side="right") - 1
        ok = idx >= 0
        idx_c = np.clip(idx, 0, len(cs) - 1)
        fresh = ok & ((grid - ts[idx_c]) <= cfg.max_stale_bars * cfg.bar_minutes * 60)
        if m.venue is Venue.KALSHI:
            yb = np.array([np.nan if c.yes_bid is None else c.yes_bid for c in cs])[idx_c]
            ya = np.array([np.nan if c.yes_ask is None else c.yes_ask for c in cs])[idx_c]
            bad = ~((yb > 0.0) & (ya < 1.0) & (ya >= yb))          # empty book / crossed / end-of-life quotes
            yb = np.where(bad, np.nan, yb)
            ya = np.where(bad, np.nan, ya)
            mid_yes = 0.5 * (yb + ya)
        else:                                                       # price-only history -> synthetic spread
            pr = np.array([np.nan if c.price is None else c.price for c in cs])[idx_c]
            hs = cfg.poly_half_spread
            ya, yb, mid_yes = pr + hs, pr - hs, pr
        ask = ya if side is Side.YES else 1.0 - yb
        mid = mid_yes if side is Side.YES else 1.0 - mid_yes
        ask = np.where(fresh, ask, np.nan) * (1.0 + cfg.slippage_bps / 1e4)
        mid = np.where(fresh, mid, np.nan)
        good = (ask >= 0.001) & (ask <= 0.999)
        return np.where(good, ask, np.nan), np.where(good, mid, np.nan)

    def _sched_settle(self, st: Structure) -> int:
        cfg = self.cfg
        ts = []
        for m, _ in st.legs:
            base = m.settle_time or m.close_time
            if base:
                lag = cfg.kalshi_lag_h if m.venue is Venue.KALSHI else cfg.poly_lag_h
                ts.append(base.timestamp() + lag * 3600)
        return int(max(ts)) if ts else 0

    def _signals(self, structs: list[Structure], series: dict[str, list[Candle]], now: datetime) -> list[Signal]:
        cfg = self.cfg
        period = cfg.bar_minutes * 60
        now_ts = int(now.timestamp())
        out: list[Signal] = []
        n_no_data = n_gate = 0
        for st in structs:
            if any(not series.get(m.key) for m, _ in st.legs):
                n_no_data += 1
                continue
            sched = self._sched_settle(st)
            if not sched:
                continue
            opens = [_ts(m.open_time) or 0 for m, _ in st.legs]
            closes = [_ts(m.close_time) or now_ts for m, _ in st.legs]
            t_lo = max(max(opens), _ts(cfg.start) or 0)
            t_hi = min(min(closes), now_ts) - period            # never enter inside the final bar
            t_lo = -(-t_lo // period) * period
            if t_hi <= t_lo:
                continue
            grid = np.arange(t_lo, t_hi + 1, period, dtype=float)
            qs = [self._leg_quotes(m, sd, series[m.key], grid) for m, sd in st.legs]
            asks = np.vstack([q[0] for q in qs])
            mids = np.vstack([q[1] for q in qs])
            valid = ~np.isnan(asks).any(axis=0)
            if not valid.any():
                continue
            # DATA GATE: legs of a riskless structure must have mid prices summing to ~ the payout. A far lower sum
            # means one leg's quote is stale / a bad print (or the set is not what we think), not an arbitrage.
            with np.errstate(invalid="ignore"):
                consistent = mids.sum(axis=0) >= st.min_pu * (1.0 - cfg.consistency_tol)
            n_gate += int((valid & ~consistent).sum())
            valid &= consistent
            fee = np.zeros_like(asks)
            for i, (m, sd) in enumerate(st.legs):
                fee[i] = unit_fee(m, asks[i], cfg.fees)
            unit_cost = (asks + fee).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                nr = (st.min_pu - unit_cost) / unit_cost
            days = np.maximum((sched - grid) / 86400.0, cfg.min_days)
            apy = annualize_arr(nr, days)
            ok = valid & (apy >= cfg.hurdle_apy) & np.isfinite(apy) & (nr <= cfg.max_net_return)
            d = max(cfg.exec_delay_bars, 0)
            fill_ok = ok if d == 0 else np.concatenate([ok[:-d] & ok[d:], np.zeros(d, dtype=bool)])
            hit = np.flatnonzero(fill_ok)
            if hit.size == 0:
                continue
            j = int(hit[0]) + d
            if j >= len(grid):
                continue
            releases = [(m.settled_ts or m.settle_time) for m, _ in st.legs]
            all_settled = all(m.status == "settled" and m.settle_value is not None for m, _ in st.legs)
            release = _ts(max(r for r in releases if r)) if (all_settled and all(releases)) else None
            payout_pu = None
            if all_settled:
                payout_pu = sum(m.leg_payout(sd) for m, sd in st.legs)
            elif any(m.status == "settled" for m, _ in st.legs):
                pass                                                # partially settled: still active
            out.append(Signal(st, int(grid[j]), [float(asks[i, j]) for i in range(len(st.legs))],
                              float(unit_cost[j]), float(nr[j]), float(apy[j]), sched, release, payout_pu))
        self.stats["no_history"] = n_no_data
        self.stats["bars_rejected_by_data_gate"] = n_gate
        return sorted(out, key=lambda s: (s.t_fill, -s.apy))

    # ── 5. portfolio accounting ───────────────────────────────────────────────
    def _simulate(self, signals: list[Signal], now: datetime) -> BacktestResult:
        cfg = self.cfg
        cash = cfg.initial_capital
        locked_cost = 0.0
        open_pos: list[tuple[int, float, float, TradeLog]] = []      # (release_ts, cost, payout, trade)
        trades: list[TradeLog] = []
        events: list[tuple[int, float, float]] = []                 # (ts, cash, locked) after each state change
        skipped = {"capital": 0, "min_size": 0, "kelly_zero": 0, "concurrency": 0}
        now_ts = int(now.timestamp())
        t_start = _ts(cfg.start)

        def release_due(upto: int):
            nonlocal cash, locked_cost
            open_pos.sort(key=lambda x: x[0])
            while open_pos and open_pos[0][0] <= upto:
                rts, cost, payout, tr = open_pos.pop(0)
                cash += payout
                locked_cost -= cost
                events.append((rts, cash, locked_cost))

        events.append((t_start, cash, 0.0))
        for sg in signals:
            release_due(sg.t_fill)
            st = sg.st
            equity = cash + locked_cost
            if cfg.unconstrained:
                budget = cfg.max_capital_per_trade
            else:
                avail = min(cash, cfg.c_max - locked_cost)
                if avail <= 0:
                    skipped["capital" if cash <= 0 else "concurrency"] += 1
                    continue
                if cfg.sizing == "kelly":
                    lp = cfg.basis_risk_prob if st.cross_venue else 0.0
                    f = kelly_fraction(sg.net_return, lp) * cfg.kelly_mult
                    if f <= 0:
                        skipped["kelly_zero"] += 1
                        continue
                    budget = min(f * equity, cfg.max_capital_per_trade, avail)
                else:
                    budget = min(cfg.max_capital_per_trade, avail)
            budget = min(budget, cfg.assumed_depth_usd * len(st.legs) if cfg.assumed_depth_usd else budget)
            units = math.floor(budget / sg.unit_cost)
            min_units = max(m.min_order_size for m, _ in st.legs)
            if units < min_units:
                skipped["min_size"] += 1
                continue
            # exact (cent-rounded) fees at the chosen size; shrink until it fits the budget
            while units >= min_units:
                fees_tot = sum(chunk_fee(m, units, sg.prices[i], cfg.fees) for i, (m, _) in enumerate(st.legs))
                cost = units * sum(sg.prices) + fees_tot
                if cost <= budget + 1e-9:
                    break
                units -= max(1, int(units * 0.01))
            if units < min_units:
                skipped["min_size"] += 1
                continue
            guaranteed = units * st.min_pu
            realized = None if sg.payout_pu is None else units * sg.payout_pu
            status = "Active / Pending Settlement"
            if realized is not None:
                if realized > guaranteed + 1e-6 and st.max_pu > st.min_pu:
                    status = "Resolved - Upside Bonus Paid"
                elif realized >= guaranteed - 1e-6:
                    status = "Resolved - Won"
                else:
                    status = "Resolved - Loss (rule mismatch / non-exhaustive)"
            tr = TradeLog(
                opp_key=st.sid, arb_type=st.arb_type.value, event_name=st.name, opened_at=_dt(sg.t_fill),
                settle_time=_dt(sg.sched_settle), released_at=_dt(sg.release) if sg.release else None,
                size=units, cost=cost, fees=fees_tot, guaranteed_payout=guaranteed,
                max_payout=units * st.max_pu, realized_payout=realized, status=status,
                net_return_locked=(guaranteed - cost) / cost if cost else 0.0, apy_at_entry=sg.apy,
                confidence=st.confidence, source="backtest",
                legs=[{"venue": m.venue.value, "market": m.market_id, "side": sd.value, "price": sg.prices[i],
                       "title": m.title[:80]} for i, (m, sd) in enumerate(st.legs)])
            trades.append(tr)
            if not cfg.unconstrained:
                cash -= cost
                locked_cost += cost
                events.append((sg.t_fill, cash, locked_cost))
                if sg.release is not None:
                    open_pos.append((sg.release, cost, realized, tr))
                else:                                               # still open at the end of the test
                    open_pos.append((10 ** 12, cost, guaranteed, tr))
        release_due(now_ts if not cfg.unconstrained else 0)
        res = BacktestResult(cfg=cfg, trades=trades)
        res.stats = {"skipped": skipped}
        if cfg.unconstrained:
            self._locked_series_from_trades(res)
            return res
        self._finish(res, events, open_pos, now_ts)
        return res

    # ── 6. series + metrics ───────────────────────────────────────────────────
    def _locked_series_from_trades(self, res: BacktestResult):
        """Locked-capital timeline for the history tab (no portfolio constraint)."""
        if not res.trades:
            return
        t0 = min(t.opened_at for t in res.trades).replace(hour=0, minute=0, second=0, microsecond=0)
        idx = pd.date_range(t0, utcnow(), freq="D", tz="UTC")
        arr = np.zeros(len(idx))
        for t in res.trades:
            end = t.released_at or t.settle_time or utcnow()
            arr += ((idx >= t.opened_at) & (idx < end)) * t.cost
        res.locked = pd.Series(arr, index=idx)

    def _finish(self, res: BacktestResult, events: list, open_pos: list, now_ts: int):
        cfg = self.cfg
        ev = sorted(events, key=lambda x: x[0])
        ts = np.array([e[0] for e in ev], dtype=float)
        cash = np.array([e[1] for e in ev])
        lock = np.array([e[2] for e in ev])
        start = _ts(cfg.start)
        last_real = max([t.released_at.timestamp() for t in res.trades if t.released_at] + [ts.max()])
        end = int(min(max(_ts(cfg.end), last_real), now_ts))
        idx = pd.date_range(_dt(start).replace(hour=0, minute=0, second=0, microsecond=0),
                            _dt(end), freq="D", tz="UTC")
        grid = np.array([i.timestamp() for i in idx])
        k = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
        equity = cash[k] + lock[k]                                  # realised basis: cash + capital at cost
        res.equity = pd.Series(equity, index=idx)
        res.locked = pd.Series(lock[k], index=idx)
        years = np.arange(len(idx)) / DAYS_PER_YEAR
        res.benchmark = pd.Series(cfg.initial_capital * (1.0 + cfg.risk_free) ** years, index=idx)
        peak = np.maximum.accumulate(equity)
        res.drawdown = pd.Series(equity / peak - 1.0, index=idx)
        res.metrics = self._metrics(res, idx)

    def _metrics(self, res: BacktestResult, idx: pd.DatetimeIndex) -> dict:
        cfg = self.cfg
        eq = res.equity.values
        tr = [t for t in res.trades if t.realized_payout is not None]
        pnl = [t.pnl for t in tr]
        days = max((idx[-1] - idx[0]).days, 1)
        end_eq = float(eq[-1]) if len(eq) else cfg.initial_capital
        net = end_eq - cfg.initial_capital
        # open positions carry their locked-in value in the total-return view
        unreal = sum(t.guaranteed_payout - t.cost for t in res.trades if t.realized_payout is None)
        cagr = (end_eq / cfg.initial_capital) ** (DAYS_PER_YEAR / days) - 1.0 if end_eq > 0 else -1.0
        rets = pd.Series(eq).pct_change().dropna().values
        rf_d = cfg.risk_free / 365.0
        sd = rets.std(ddof=1) if len(rets) > 2 else 0.0
        sharpe = float((rets.mean() - rf_d) / sd * math.sqrt(365.0)) if (sd > 0 and res.trades) else float("nan")
        dn = np.minimum(rets - rf_d, 0.0)
        dd = math.sqrt((dn ** 2).mean()) if len(dn) else 0.0
        sortino = float((rets.mean() - rf_d) / dd * math.sqrt(365.0)) if (dd > 0 and res.trades) else float("nan")
        peak = np.maximum.accumulate(eq)
        max_dd_usd = float((peak - eq).max()) if len(eq) else 0.0
        max_dd_pct = float((eq / peak - 1.0).min()) if len(eq) else 0.0
        lock_days = [t.lockup_days for t in res.trades]
        deployed_days = sum(t.cost * t.lockup_days for t in res.trades)
        total_pnl = sum(pnl)
        avg_locked = float(res.locked.mean()) if len(res.locked) else 0.0
        return {
            "Total Net PnL ($)": round(net, 2),
            "Realised PnL ($)": round(total_pnl, 2),
            "Unrealised Locked-in PnL ($)": round(unreal, 2),
            "Return on Capital (%)": round(100 * net / cfg.initial_capital, 3),
            "CAGR / APY (%)": round(100 * cagr, 3),
            "Max Drawdown (%)": round(100 * max_dd_pct, 3),
            "Max Drawdown ($)": round(max_dd_usd, 2),
            "Sharpe Ratio": None if math.isnan(sharpe) else round(sharpe, 2),
            "Sortino Ratio": None if math.isnan(sortino) else round(sortino, 2),
            "Win Rate (%)": round(100 * sum(1 for p in pnl if p > 0) / len(pnl), 2) if pnl else None,
            "Trades": len(res.trades),
            "Resolved Trades": len(tr),
            "Avg Lockup (days)": round(float(np.mean(lock_days)), 2) if lock_days else 0.0,
            "Capital Utilisation (%)": round(100 * avg_locked / cfg.initial_capital, 2),
            "Return on Deployed Capital (APY %)":
                round(100 * total_pnl / deployed_days * DAYS_PER_YEAR, 2) if deployed_days > 0 else None,
            "Fees Paid ($)": round(sum(t.fees for t in res.trades), 2),
            "Avg Net Return / Trade (%)": round(100 * float(np.mean([t.net_return_locked for t in res.trades])), 3)
            if res.trades else None,
            "Losing Trades (rule mismatch)": sum(1 for t in tr if t.status.startswith("Resolved - Loss")),
        }

    def _assumptions(self) -> list[str]:
        cfg = self.cfg
        a = ["NOT US-TRADABLE DATA: Polymarket price history and settled markets come from international polymarket.com (the only source with history), which prohibits US persons. These results show how Kalshi and Polymarket priced the same events - they are NOT a record of trades a US person could have made."] + list(self.assumptions)
        a.append(f"Kalshi history: end-of-bar bid/ask closes ({cfg.bar_minutes}-min bars). Polymarket history is "
                 f"trade/mid price only: synthetic half-spread of {cfg.poly_half_spread * 100:.2f}c applied to each side.")
        a.append(f"Data gates: sum of leg mid-prices must be >= {1 - cfg.consistency_tol:.0%} of the payout; isolated "
                 f"Polymarket prints >{cfg.spike_threshold:.2f} off both neighbours are dropped; edges above "
                 f"{cfg.max_net_return:.0%} net are rejected as definition/data mismatches.")
        a.append(f"No historical L2 depth exists on either venue: fill size capped at ${cfg.assumed_depth_usd:,.0f} "
                 f"per leg (assumption, not observed liquidity).")
        a.append(f"Execution: decision at bar t, fill at the quotes {cfg.exec_delay_bars} bar(s) later, plus "
                 f"{cfg.slippage_bps:.0f} bps slippage on every leg.")
        a.append(f"Fees: Kalshi taker = ceil(cent) of {cfg.fees.kalshi_coef} x C x P x (1-P); Polymarket "
                 f"{cfg.fees.poly_mode} mode (per-market feeSchedule when fee-enabled); +{cfg.fees.extra_bps:.0f} bps extra.")
        a.append("Entry APY uses the SCHEDULED settlement time (+ venue lag); capital is released at the ACTUAL "
                 "settlement time. Realised payouts use each leg's actual outcome (mismatches count as losses).")
        a.append("Equity is on a realised basis (cash + capital at cost): profit is recognised at settlement, so "
                 "Sharpe/Sortino reflect lumpy realisation, not mark-to-market.")
        if self._degraded:
            a.append(f"DEGRADED DATA: {len(self._degraded)} Polymarket series fell back to end-of-day prices because "
                     f"the requested intraday fidelity was refused.")
        if cfg.sizing == "kelly":
            a.append(f"Sizing: {cfg.kelly_mult:.2f} x Kelly with resolution-mismatch loss probability "
                     f"{cfg.basis_risk_prob:.1%} on cross-venue structures (0% on single-venue).")
        else:
            a.append(f"Sizing: fixed ${cfg.max_capital_per_trade:,.0f} per trade, C_max ${cfg.c_max:,.0f}.")
        a.append("Universe is limited by volume filter, page caps and structure caps; matched-pair coverage is "
                 "heuristic (no manual rulebook review). Survivorship: only markets that reached settlement.")
        return a


# ── convenience entry points ──────────────────────────────────────────────────
async def run_backtest(cfg: BacktestConfig, db: Optional[Database] = None, progress: ProgressCB = None,
                       cancel: Optional[asyncio.Event] = None) -> BacktestResult:
    return await Backtester(cfg, db, progress, cancel).run()


async def run_history(days: int, db: Optional[Database] = None, progress: ProgressCB = None,
                      cancel: Optional[asyncio.Event] = None, hurdle: float = 0.07,
                      trade_size: float = 500.0, **kw) -> BacktestResult:
    """Past-``days`` opportunity log: every hurdle-clearing structure, resolved or still active, unconstrained."""
    now = utcnow()
    cfg = BacktestConfig(start=now - timedelta(days=days), end=now, hurdle_apy=hurdle,
                         max_capital_per_trade=trade_size, include_open=True, unconstrained=True,
                         exec_delay_bars=1, **kw)
    return await Backtester(cfg, db, progress, cancel).run()
