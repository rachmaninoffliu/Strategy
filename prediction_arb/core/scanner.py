"""Live scan service: fetch both venues -> map -> detect -> L2 refine -> log.

One ``scan_once`` call is a self-contained async job (own HTTP sessions), so the GUI can run it in a worker
thread with its own event loop.

Polymarket source (``ScanSettings.poly_source``):
* ``"us"``     (default) Polymarket US - the CFTC-regulated exchange US persons may trade (gateway.polymarket.us);
* ``"global"`` international polymarket.com - NOT tradable by US persons; every leg on it is flagged and, with
  ``us_only`` (default), dropped from the results.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import ScanSettings
from connectors import mapper
from connectors.kalshi_client import KalshiClient
from connectors.poly_client import PolyClient
from connectors.polyus_client import PolyUSClient
from core.arb_detector import ArbDetector, Candidate, Recipe
from core.longshots import find_longshots, implied_prob
from core.models import ArbOpportunity, Market, OrderBook, Venue, utcnow
from core.pricing import LegSpec  # noqa: F401  (re-export for callers)
from database import Database

ProgressCB = Optional[Callable[[int, str], None]]
US_STATS_CAP = 600          # exact volume / OI is fetched (1 call each) for at most this many US long-shot candidates


@dataclass
class ScanResult:
    opportunities: list[ArbOpportunity] = field(default_factory=list)
    recipes: dict[str, Recipe] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    longshots: list = field(default_factory=list)


class LiveScanner:
    def __init__(self, settings: Optional[ScanSettings] = None, db: Optional[Database] = None):
        self.s = settings or ScanSettings()
        self.db = db
        self.detector = ArbDetector(self.s)
        self.last: Optional[ScanResult] = None

    async def scan_once(self, progress: ProgressCB = None, cancel: Optional[asyncio.Event] = None) -> ScanResult:
        t0 = time.time()
        res = ScanResult()
        self.detector.rejected.clear()
        us_src = self.s.poly_source == "us"
        poly_name = "Polymarket US" if us_src else "Polymarket (international)"

        def step(pct: int, msg: str):
            if progress:
                progress(pct, msg)
            if cancel and cancel.is_set():
                raise asyncio.CancelledError()

        PolyCls = PolyUSClient if us_src else PolyClient
        async with KalshiClient() as kc, PolyCls() as pc:
            step(3, f"Downloading open markets (Kalshi + {poly_name})...")
            k_task = asyncio.create_task(kc.get_open_events(self.s.kalshi_max_pages, lambda m: step(10, m)))
            if us_src:
                floors = ((self.s.longshot.min_volume_usd, 100_000.0) if self.s.longshots_only
                          else (self.s.us_min_volume_usd, self.s.longshot.min_volume_usd, 100_000.0))
                p_task = asyncio.create_task(pc.get_open_events(self.s.poly_max_pages, lambda m: step(20, m), floors))
            else:
                p_task = asyncio.create_task(pc.get_open_events(self.s.poly_max_pages, lambda m: step(20, m)))
            results = await asyncio.gather(k_task, p_task, return_exceptions=True)
            kal, pol = [], []
            for name, r in zip(("Kalshi", poly_name), results):
                if isinstance(r, asyncio.CancelledError):
                    raise r
                if isinstance(r, Exception):
                    res.warnings.append(f"{name} fetch failed: {r}")
                else:
                    (kal if name == "Kalshi" else pol).extend(r)
            res.warnings += pc.warnings
            if not kal and not pol:
                raise RuntimeError("; ".join(res.warnings) or "no market data returned")
            if not kal or not pol:
                res.warnings.insert(0, f"{'Kalshi' if not kal else poly_name} returned NO markets - cross-venue "
                                       f"arbitrage (Type I and cross-venue II/III) cannot be detected in this scan")
            if not us_src:
                res.warnings.insert(0, "Polymarket INTERNATIONAL data: polymarket.com is NOT available to US persons - "
                                       "anything involving it is reference only, not tradable from the US")

            # Polymarket US lists no volume: exact cash volume / open interest for the few long-shot candidates
            if us_src and pol:
                lc = self.s.longshot
                cand = [m for m in pol if m.volume_usd >= lc.min_volume_usd
                        and (implied_prob(m) is not None and lc.min_prob <= implied_prob(m) <= lc.max_prob)]
                cand.sort(key=lambda m: -m.volume_usd)
                if cand:
                    step(50, f"Fetching exact volume / open interest for {min(len(cand), US_STATS_CAP)} "
                             f"Polymarket US long-shot candidates...")
                    await pc.enrich_stats(cand[:US_STATS_CAP], progress=lambda m: step(52, m))
                    for m in cand[:US_STATS_CAP]:
                        m.stats_exact = True

            if self.s.longshots_only:                       # fast path: no matching, no arbitrage detection
                step(60, f"Screening {len(kal) + len(pol):,} markets for long-shots...")
                res.longshots = self._ls(kal + pol, utcnow(), None)
                res.stats = {"seconds": round(time.time() - t0, 1), "kalshi_markets": len(kal),
                             "poly_markets": len(pol), "longshots": len(res.longshots), "poly_source": self.s.poly_source,
                             "requests": kc.request_count + pc.request_count}
                self.last = res
                step(100, f"Done in {res.stats['seconds']}s: {len(res.longshots)} long-shots")
                return res

            step(45, f"Mapping {len(kal)} Kalshi x {len(pol)} {poly_name} markets...")
            # Baskets need COMPLETE events (all outcomes, all statuses); pairs / ladders only need liquid open markets.
            kb = mapper.kalshi_baskets(kal, self.s.max_basket_legs)
            pb = mapper.poly_baskets(pol, self.s.max_basket_legs)
            cross = mapper.match_baskets(kb, pb, self.s.date_tol_hours)
            kal_o = [m for m in kal if m.status == "open" and m.volume >= self.s.min_volume]
            pol_o = [m for m in pol if m.status == "open" and m.volume >= self.s.min_volume]
            pairs = mapper.match_pairs(kal_o, pol_o, self.s.date_tol_hours, self.s.min_confidence)
            edges = mapper.build_subset_edges(kal_o + pol_o, self.s.subset_tol_hours)
            baskets = kb + pb + cross

            step(60, f"Screening {len(pairs)} pairs, {len(edges)} ladder edges, {len(baskets)} baskets...")
            now = utcnow()
            pair_index = {}
            for pr in pairs:                                  # lets the long-shot screen show the same event elsewhere
                pair_index[pr.a.key] = pr.b
                pair_index[pr.b.key] = pr.a
            res.longshots = self._ls(kal + pol, now, pair_index)
            cands = self.detector.scan(pairs, edges, baskets, now)
            n_stage1 = len(cands)

            if self.s.use_l2_books and cands:
                step(70, f"Refining {min(n_stage1, self.s.max_book_candidates)} candidates with L2 books...")

                async def fetch_book(m: Market) -> Optional[OrderBook]:
                    if m.venue is Venue.KALSHI:
                        return await kc.get_orderbook(m.market_id)
                    return await pc.get_market_book(m)

                cands = await self.detector.refine(cands, fetch_book, now, lambda m: step(85, m))
            else:
                cands = [c for c in cands if c.opp.min_depth_usd >= self.s.min_depth_usd or not c.opp.depth_verified]

        n_non_us = sum(1 for c in cands if not c.opp.us_tradable)
        if self.s.us_only:
            cands = [c for c in cands if c.opp.us_tradable]
        res.opportunities = [c.opp for c in cands]
        res.recipes = {c.opp.key: c.recipe for c in cands}
        res.stats = {
            "kalshi_markets": len(kal_o), "poly_markets": len(pol_o), "pairs": len(pairs), "subset_edges": len(edges),
            "baskets": len(baskets), "stage1_candidates": n_stage1, "actionable": len(cands),
            "longshots": len(res.longshots), "poly_source": self.s.poly_source, "non_us_dropped": n_non_us if self.s.us_only else 0,
            "rejected": dict(self.detector.rejected), "requests": kc.request_count + pc.request_count,
            "seconds": round(time.time() - t0, 1)}
        if self.db and res.opportunities:
            try:
                self.db.log_opportunities(res.opportunities)
            except Exception as e:                                # logging must never break a scan
                res.warnings.append(f"DB log failed: {e}")
        self.last = res
        step(100, f"Done in {res.stats['seconds']}s: {len(cands)} actionable "
                  f"({n_stage1} passed the top-of-book screen)")
        return res

    def _ls(self, markets: list[Market], now, pair_index) -> list:
        out = find_longshots(markets, self.s.longshot, self.s.fees, now, pair_index)
        return [o for o in out if o.us_tradable] if self.s.us_only else out

    # ── ticket helpers ────────────────────────────────────────────────────────
    def reprice(self, opp: ArbOpportunity, units: Optional[float] = None) -> Optional[ArbOpportunity]:
        """Re-walk the stored ladders for a manual size (order-ticket recalculation)."""
        r = self.last.recipes.get(opp.key) if self.last else None
        if r is None:
            return None
        ladders = [l.ladder for l in opp.legs]
        return self.detector.price(r, ladders, utcnow(), verified=opp.depth_verified,
                                   max_units=units if units else float("inf"), hurdle=-0.999)

    async def refresh_quote(self, opp: ArbOpportunity, units: Optional[float] = None) -> Optional[ArbOpportunity]:
        """Fetch FRESH L2 books for every leg right now and re-price at ``units`` (no hurdle: show what it costs)."""
        r = self.last.recipes.get(opp.key) if self.last else None
        if r is None:
            return None
        async with KalshiClient() as kc, PolyUSClient() as pus, PolyClient() as pg:
            async def fb(m: Market) -> Optional[OrderBook]:
                if m.venue is Venue.KALSHI:
                    return await kc.get_orderbook(m.market_id)
                return await (pus if m.us_tradable else pg).get_market_book(m)

            books = await asyncio.gather(*(fb(m) for m, _ in r.legs), return_exceptions=True)
        ladders = []
        for (m, sd), b in zip(r.legs, books):
            if isinstance(b, Exception) or b is None or not b.asks(sd):
                return None
            ladders.append(b.asks(sd))
        return self.detector.price(r, ladders, utcnow(), verified=True,
                                   max_units=units if units else opp.size, hurdle=-0.999)
