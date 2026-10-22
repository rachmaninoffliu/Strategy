"""Arbitrage mathematical verification engine (Types I, II, III) with two-stage depth refinement.

Stage 1 (``scan``)   : top-of-book screen from the listing endpoints (cheap, all structures).
Stage 2 (``refine``) : replace top-of-book with real L2 ladders for the best candidates and re-walk the book
                       so size / cost / APY reflect what can actually be filled above the hurdle.

Payoff algebra (every unit of every leg pays $1 on a win):
  Type I  parity   : YES_v1 + NO_v2 on equivalent questions              -> exactly $1 in every state
                     YES_v1 + YES_v2 on logically opposite questions      -> exactly $1
  Type II MECE     : all YES of an exhaustive mutually-exclusive set      -> exactly $1
                     all NO  of a mutually-exclusive set                  -> at least n-1 (n if none occurs)
  Type III subset  : A implies B; buy YES_B and NO_A                      -> $1 if A, $2 if B and not A, $1 if not B
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from config import ScanSettings
from connectors.mapper import Basket, MarketPair, SubsetEdge
from core.models import (ArbLeg, ArbOpportunity, ArbType, Market, OrderBook, Relation, Side, Venue,
                         utcnow)
from core.pricing import LegSpec, days_to_settle, walk_legs

UNKNOWN_SIZE = 1_000.0            # placeholder depth while only top-of-book (no size) is known


@dataclass
class Recipe:
    """Everything needed to (re)price an opportunity against any ladder."""
    arb_type: ArbType
    name: str
    legs: list[tuple[Market, Side]]
    min_pu: float
    max_pu: float
    variant: str = ""
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    recipe: Recipe
    opp: ArbOpportunity


class ArbDetector:
    def __init__(self, settings: Optional[ScanSettings] = None):
        self.s = settings or ScanSettings()
        self.rejected: dict[str, int] = defaultdict(int)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _lag(self, m: Market) -> float:
        return (self.s.kalshi_lag_h if m.venue is Venue.KALSHI else self.s.poly_lag_h) * 3600.0

    def _settle(self, markets: list[Market]) -> Optional[datetime]:
        """Capital is free only when the *last* leg has settled (plus venue dispute / payout lag)."""
        ts = [m.settle_time.timestamp() + self._lag(m) for m in markets if m.settle_time]
        if not ts:
            return None
        return datetime.fromtimestamp(max(ts), tz=timezone.utc)

    def _ok_market(self, m: Market) -> bool:
        return m.status == "open" and m.volume >= self.s.min_volume

    @staticmethod
    def _top_ladder(m: Market, side: Side) -> list[tuple[float, float]]:
        ask = m.ask(side)
        if ask is None or not (0.0 < ask < 1.0):
            return []
        size = m.ask_size(side)
        return [(ask, size if size > 0 else UNKNOWN_SIZE)]

    def price(self, r: Recipe, ladders: list[list[tuple[float, float]]], now: datetime, *,
              verified: bool, max_units: float = float("inf"), max_cost: float = float("inf"),
              hurdle: Optional[float] = None) -> Optional[ArbOpportunity]:
        """Walk the ladders and build an :class:`ArbOpportunity` (None when nothing clears the hurdle)."""
        markets = [m for m, _ in r.legs]
        settle = self._settle(markets)
        days = days_to_settle(settle, now, self.s.min_days)
        specs = [LegSpec(m, sd, lad) for (m, sd), lad in zip(r.legs, ladders)]
        min_units = max((m.min_order_size for m in markets), default=1.0)
        res = walk_legs(specs, min_unit_payout=r.min_pu, max_unit_payout=r.max_pu, days=days,
                        hurdle=self.s.hurdle_apy if hurdle is None else hurdle, fees=self.s.fees,
                        max_units=max_units, max_cost=max_cost, min_units=min_units)
        if res is None:
            return None
        legs = []
        for i, (m, sd) in enumerate(r.legs):
            legs.append(ArbLeg(
                venue=m.venue, market_id=m.market_id, title=m.title or m.event_title,
                side=sd, price=res.avg_prices[i], limit_price=res.limit_prices[i], top_price=res.top_prices[i],
                size=res.units, fee=res.leg_fees[i], url=m.url,
                token_id=m.yes_token if sd is Side.YES else m.no_token, ladder=ladders[i],
                depth_usd=res.depth_usd[i], rules=m.rules, us_tradable=m.us_tradable, market_ref=m.market_id))
        notes = list(r.notes)
        if r.arb_type is ArbType.PARITY and res.net_return > 0.08:
            notes.insert(0, f"WARNING: a {res.net_return:.0%} edge between two liquid venues is unusual - in practice "
                            f"this almost always means the two markets are NOT the same event. Verify teams / dates / rules.")
        # depth is only meaningful for legs whose size is actually known
        known = [l.depth_usd for l, lad in zip(legs, ladders) if verified or lad[0][1] != UNKNOWN_SIZE]
        return ArbOpportunity(
            arb_type=r.arb_type, event_name=r.name, legs=legs, size=res.units, total_cost=res.total_cost,
            fees=res.fees, guaranteed_payout=res.min_payout, max_payout=res.max_payout,
            net_return=res.net_return, days_to_settle=days, apy=res.apy,
            min_depth_usd=min(known) if known else 0.0, settle_time=settle, detected_at=now,
            confidence=r.confidence, depth_verified=verified, variant=r.variant, notes=notes,
            unit_min_payout=r.min_pu, unit_max_payout=r.max_pu)

    def _candidate(self, r: Recipe, now: datetime) -> Optional[Candidate]:
        ladders = [self._top_ladder(m, sd) for m, sd in r.legs]
        if any(not l for l in ladders):
            return None
        opp = self.price(r, ladders, now, verified=False)
        if opp and opp.net_return > self.s.max_net_return:
            self.rejected["implausible_edge"] += 1          # almost always a definition / data mismatch
            return None
        return Candidate(r, opp) if opp else None

    # ── Type I ────────────────────────────────────────────────────────────────
    def detect_parity(self, pairs: list[MarketPair], now: datetime) -> list[Candidate]:
        out: list[Candidate] = []
        for p in pairs:
            if p.confidence < self.s.min_confidence or not (self._ok_market(p.a) and self._ok_market(p.b)):
                continue
            k, q = p.a, p.b
            combos = ([(Side.YES, Side.NO), (Side.NO, Side.YES)] if p.relation is Relation.EQUIVALENT
                      else [(Side.YES, Side.YES), (Side.NO, Side.NO)])
            notes = ["match: " + "; ".join(p.reasons),
                     "Guarantee holds only if BOTH rulebooks resolve identically - verify before trading"]
            best: Optional[Candidate] = None
            for sk, sq in combos:
                r = Recipe(ArbType.PARITY, f"{k.title or k.event_title}  <->  {q.title}",
                           [(k, sk), (q, sq)], 1.0, 1.0, variant=f"K {sk.value} + P {sq.value}",
                           confidence=p.confidence, notes=notes)
                c = self._candidate(r, now)
                if c and (best is None or c.opp.apy > best.opp.apy):
                    best = c
            if best:
                out.append(best)
        return out

    # ── Type III ──────────────────────────────────────────────────────────────
    def detect_subset(self, edges: list[SubsetEdge], now: datetime) -> list[Candidate]:
        out: list[Candidate] = []
        for e in edges:
            a, b = e.inner, e.outer
            if e.confidence < self.s.min_confidence or not (self._ok_market(a) and self._ok_market(b)):
                continue
            # quick reject: need bid_A > ask_B  <=>  ask_B + ask_NO_A < 1
            if b.yes_ask is None or a.no_ask is None or b.yes_ask + a.no_ask >= 1.0:
                continue
            notes = ["; ".join(e.reasons),
                     f"A = {a.title or a.event_title} {a.outcome_label}".strip(),
                     "Payout: $1 if A, $2 if B and not A, $1 if neither (upside bonus)"]
            if a.venue != b.venue:
                notes.append("Cross-venue ladder: confirm identical price source and observation time")
            r = Recipe(ArbType.SUBSET, f"{(b.event_title or b.title)[:70]}: {a.outcome_label or a.title} ⊆ "
                       f"{b.outcome_label or b.title}", [(b, Side.YES), (a, Side.NO)], 1.0, 2.0,
                       variant="buy YES(B) + NO(A)", confidence=e.confidence, notes=notes)
            c = self._candidate(r, now)
            if c:
                out.append(c)
        return out

    # ── Type II ───────────────────────────────────────────────────────────────
    @staticmethod
    def _mid(m: Market) -> Optional[float]:
        if m.yes_bid is not None and m.yes_ask is not None:
            return 0.5 * (m.yes_bid + m.yes_ask)
        return m.yes_ask if m.yes_ask is not None else m.yes_bid

    def detect_mece(self, baskets: list[Basket], now: datetime) -> list[Candidate]:
        out: list[Candidate] = []
        for bk in baskets:
            n = len(bk.outcomes)
            if n < 2 or n > self.s.max_basket_legs or bk.confidence < self.s.min_confidence:
                continue
            if sum(m.volume for cands in bk.outcomes for m in cands) < self.s.min_volume:
                self.rejected["basket_illiquid"] += 1
                continue
            for side in (Side.YES, Side.NO):
                if side is Side.YES and not bk.exhaustive:
                    continue                                   # buy-all-YES needs an exhaustive set
                legs: list[tuple[Market, Side]] = []
                ok = True
                for cands in bk.outcomes:
                    live = [m for m in cands if m.status == "open" and m.ask(side) is not None]
                    if not live:
                        ok = False
                        break
                    legs.append((min(live, key=lambda m: m.ask(side)), side))
                if not ok:
                    continue
                # Sanity: the YES mids of a complete mutually-exclusive set must sum to ~1 (never > ~1.1).
                mids = [self._mid(cands[0]) for cands in bk.outcomes]
                if all(x is not None for x in mids):
                    tot = sum(mids)
                    lo = self.s.mece_mid_lo if bk.exhaustive else 0.0
                    if not (lo <= tot <= self.s.mece_mid_hi):
                        self.rejected["mece_mid_sum"] += 1
                        continue
                if side is Side.YES:
                    min_pu, max_pu, variant = 1.0, 1.0, "buy-all-YES"
                else:
                    min_pu = float(n - 1)
                    max_pu = float(n - 1) if bk.exhaustive else float(n)
                    variant = "buy-all-NO"
                notes = [f"{bk.scope} basket, {n} outcomes; {bk.exhaustive_reason}", *bk.notes]
                r = Recipe(ArbType.MECE, bk.name[:90], legs, min_pu, max_pu, variant=variant,
                           confidence=bk.confidence, notes=notes)
                c = self._candidate(r, now)
                if c:
                    out.append(c)
        return out

    # ── orchestration ─────────────────────────────────────────────────────────
    def scan(self, pairs: list[MarketPair], edges: list[SubsetEdge], baskets: list[Basket],
             now: Optional[datetime] = None) -> list[Candidate]:
        now = now or utcnow()
        inc = self.s.include_types
        cands: list[Candidate] = []
        if "parity" in inc:
            cands += self.detect_parity(pairs, now)
        if "subset" in inc:
            cands += self.detect_subset(edges, now)
        if "mece" in inc:
            cands += self.detect_mece(baskets, now)
        best: dict[str, Candidate] = {}
        for c in cands:
            k = c.opp.key
            if k not in best or c.opp.apy > best[k].opp.apy:
                best[k] = c
        return sorted(best.values(), key=lambda c: -c.opp.apy)

    async def refine(self, cands: list[Candidate], fetch_book: Callable[[Market], Awaitable[Optional[OrderBook]]],
                     now: Optional[datetime] = None,
                     progress: Optional[Callable[[str], None]] = None) -> list[Candidate]:
        """Re-price the top candidates against real L2 ladders. Candidates whose books vanish are dropped."""
        import asyncio
        now = now or utcnow()
        top = cands[: self.s.max_book_candidates]
        need: dict[str, Market] = {}
        for c in top:
            for m, _ in c.recipe.legs:
                need[m.key] = m
        books: dict[str, Optional[OrderBook]] = {}

        async def get(m: Market):
            try:
                books[m.key] = await fetch_book(m)
            except Exception:                                   # network / 4xx on one book must not kill the scan
                books[m.key] = None

        await asyncio.gather(*(get(m) for m in need.values()))
        if progress:
            progress(f"Fetched {sum(b is not None for b in books.values())}/{len(books)} L2 books")
        out: list[Candidate] = []
        for c in top:
            r = c.recipe
            ladders, ok = [], True
            for m, sd in r.legs:
                ob = books.get(m.key)
                lad = ob.asks(sd) if ob else []
                if not lad:
                    ok = False
                    break
                ladders.append(lad)
            if not ok:
                continue
            opp = self.price(r, ladders, now, verified=True)
            if opp and opp.net_return > self.s.max_net_return:
                self.rejected["implausible_edge"] += 1
                continue
            if opp and opp.min_depth_usd >= self.s.min_depth_usd:
                out.append(Candidate(r, opp))
        return sorted(out, key=lambda c: -c.opp.apy)
