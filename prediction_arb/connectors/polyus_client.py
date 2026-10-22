"""Polymarket US client - the Polymarket product that US persons may lawfully trade.

International polymarket.com prohibits US persons; Polymarket US (QCX LLC, CFTC-regulated, launched Dec 2025) is a
separate exchange with its own order books, a subset of markets and its own fee schedule. Public market data needs no key:

* ``GET https://gateway.polymarket.us/v1/events``            events with nested markets (+ quotes)
* ``GET https://gateway.polymarket.us/v1/markets``           markets (server-side ``volumeNumMin`` filter works)
* ``GET https://gateway.polymarket.us/v1/markets/{slug}/book``  L2 book + stats (notionalTraded, openInterest, ...)

Model: every market is ONE instrument with a *long* side (the quoted one) and a *short* side. Quotes describe the long
side, so  YES = long, NO = short  and  NO ask = 1 - best long bid  (identical to Kalshi's mirror).
Fees (docs.polymarket.us/fees): taker = 0.0695 * C * p * (1-p), rounded to the nearest cent (half-even).
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Callable, Optional

from config import POLY_US_GATEWAY, POLY_US_RPS, POLY_US_WEB
from connectors.http_base import ApiError, HttpClient
from core.models import Market, OrderBook, Venue, parse_dt, utcnow

ProgressCB = Optional[Callable[[str], None]]
DEFAULT_FEE_COEF = 0.0695
VOLUME_TIERS = (100_000.0, 10_000.0, 1_000.0)     # server-side market-volume floors, highest first


def _f(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _quote(q) -> Optional[float]:
    """{'value': '0.1790', 'currency': 'USD'} -> 0.179 (None when absent or not a tradable price)."""
    v = _f(q.get("value")) if isinstance(q, dict) else _f(q)
    return v if (v is not None and 0.0 < v < 1.0) else None


_GAME_TYPES = ("moneyline", "spreads", "totals", "props", "drawable_outcome")
_LINE_RE = re.compile(r"-(\d+)pt(\d+)$")
_SLUG_DATE_RE = re.compile(r"(20\d\d)-(\d\d)-(\d\d)")
GAME_SETTLE_LAG = timedelta(hours=3)                 # a game market settles a few hours after the game starts


def event_url(ev: dict) -> str:
    """polymarket.us: sports -> /sports/{league}/{event-slug}; everything else -> /event/{event-slug} (both verified)."""
    slug = ev.get("slug")
    if not slug:
        return POLY_US_WEB
    league = (ev.get("primaryTag") or {}).get("slug")
    # /sports/{league}/{slug} is for GAMES (two teams); season futures live at /event/{slug} (the other form redirects)
    if (ev.get("category") or "").lower() == "sports" and league and len(ev.get("teams") or []) == 2:
        return f"{POLY_US_WEB}/sports/{league}/{slug}"
    return f"{POLY_US_WEB}/event/{slug}"


def event_title(ev: dict) -> str:
    """Full team names when available ('Atlanta Falcons vs. New Orleans Saints'): display titles use abbreviations."""
    teams = [t.get("name") for t in (ev.get("teams") or []) if isinstance(t, dict) and t.get("name")]
    if len(teams) == 2:
        return f"{teams[0]} vs. {teams[1]}"
    return ev.get("title") or ""


def markets_from_us(m: dict, ev: dict, now: Optional[datetime] = None) -> list[Market]:
    """Normalise one US market into 1 binary Market (Yes/No) or 2 (A / B outcomes: long side and its complement)."""
    if m.get("hidden") or m.get("closed") or m.get("active") is False:
        return []
    if (m.get("status") or "") != "MARKET_STATUS_OPEN":
        return []
    sides = m.get("marketSides") or []
    long_s = next((s for s in sides if s.get("long")), None)
    short_s = next((s for s in sides if not s.get("long")), None)
    if long_s is None or short_s is None:
        return []
    ldesc, sdesc = str(long_s.get("description") or ""), str(short_s.get("description") or "")
    is_yes_no = ldesc.lower() == "yes" and sdesc.lower() == "no"
    bid, ask = _quote(m.get("bestBidQuote")), _quote(m.get("bestAskQuote"))
    now = now or utcnow()
    end = parse_dt(m.get("endDate")) or parse_dt(ev.get("endDate"))
    slug = m.get("slug") or ""
    mtype = (m.get("marketType") or m.get("sportsMarketType") or "").lower()
    date_tokens: list[str] = []
    gst = parse_dt(m.get("gameStartTime"))
    if mtype in _GAME_TYPES:
        if gst:                       # endDate of a game market is the expiry DEADLINE (weeks later), not the game
            end = gst + GAME_SETTLE_LAG
        dm = _SLUG_DATE_RE.search(slug)
        if dm:
            date_tokens = [f"{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"]
    # structural strike ladder: totals 'Over'/'Under' with the line encoded in the slug (…-total-60pt5)
    lm = _LINE_RE.search(slug) if mtype == "totals" else None
    line = float(f"{lm.group(1)}.{lm.group(2)}") if lm else None
    ladder = slug[: lm.start()] if lm else ""
    common = dict(
        venue=Venue.POLYMARKET, event_id=str(ev.get("id", "")), series_id=str(ev.get("slug", "")),
        event_title=event_title(ev), title=m.get("question") or "", rules=(m.get("description") or "")[:2500],
        url=event_url(ev), category=str(m.get("category") or ev.get("category") or ""), status="open",
        open_time=parse_dt(m.get("startDate")), close_time=end, settle_time=end,
        min_order_size=_f(m.get("minimumTradeQty")) or 1.0, yes_token=slug,
        fees_enabled=True, fee_rate=_f(m.get("feeCoefficient")) or DEFAULT_FEE_COEF, fee_exponent=1.0,
        fee_round_cents=True, us_tradable=True, volume_24h_usd=-1.0, quote_ts=now, stats_exact=False)
    kind = {"moneyline": "moneyline", "spreads": "spread", "totals": "total", "props": "prop", "futures": "future",
            "drawable_outcome": "moneyline"}.get(mtype, "")
    def _abbr(side) -> list[str]:
        t = (side or {}).get("team") or {}
        a = (t.get("displayAbbreviation") or t.get("abbreviation") or "").lower()
        return [a] if (a and kind == "moneyline") else []

    league = ((ev.get("primaryTag") or {}).get("slug") or "").lower() if kind == "moneyline" else ""
    common.update(settle_time=end, close_time=end, date_tokens=date_tokens, kind=kind, league=league)
    over_under = ldesc.lower() == "over" and sdesc.lower() == "under"
    long_title = f"{ldesc} wins" if mtype == "moneyline" else common["title"]
    short_title = f"{sdesc} wins" if mtype == "moneyline" else common["title"]
    base = {k: v for k, v in common.items() if k != "title"}
    base["rules"] = ((common["title"] + " ") if mtype == "moneyline" else "") + common["rules"]
    out = [Market(market_id=slug, title=long_title, outcome_label=(m.get("title") or ldesc) if is_yes_no else ldesc,
                  yes_bid=bid, yes_ask=ask, no_bid=None if ask is None else 1.0 - ask,
                  no_ask=None if bid is None else 1.0 - bid, subject_ids=_abbr(long_s),
                  **({"direction": "gt", "strike": line, "ladder_key": ladder} if (over_under and line is not None) else {}),
                  **base)]
    if not is_yes_no:                                    # complement: YES = the short side (e.g. the other team)
        c_bid = None if ask is None else 1.0 - ask
        c_ask = None if bid is None else 1.0 - bid
        out.append(Market(market_id=slug + "#short", title=short_title, outcome_label=sdesc, yes_bid=c_bid,
                          yes_ask=c_ask, no_bid=bid, no_ask=ask, subject_ids=_abbr(short_s),
                          **({"direction": "lt", "strike": line, "ladder_key": ladder}
                             if (over_under and line is not None) else {}), **base))
    return [mk for mk in out if mk.yes_bid is not None or mk.yes_ask is not None]


def book_from_us(d: dict, slug: str, complement: bool = False) -> Optional[tuple[OrderBook, dict]]:
    """Parse /book into an OrderBook (+ stats dict). ``complement`` swaps long/short for the '#short' market."""
    md = (d or {}).get("marketData")
    if not md:
        return None
    bids = sorted(((_f((l.get("px") or {}).get("value")), _f(l.get("qty"))) for l in md.get("bids") or []),
                  key=lambda x: -(x[0] or 0))
    offers = sorted(((_f((l.get("px") or {}).get("value")), _f(l.get("qty"))) for l in md.get("offers") or []),
                    key=lambda x: (x[0] or 0))
    bids = [(p, q) for p, q in bids if p is not None and q and 0 < p < 1]
    offers = [(p, q) for p, q in offers if p is not None and q and 0 < p < 1]
    yes_asks, yes_bids = offers, bids
    no_asks = sorted(((round(1.0 - p, 6), q) for p, q in bids), key=lambda x: x[0])      # NO ask = 1 - YES bid
    no_bids = sorted(((round(1.0 - p, 6), q) for p, q in offers), key=lambda x: -x[0])
    if complement:
        yes_asks, yes_bids, no_asks, no_bids = no_asks, no_bids, yes_asks, yes_bids
    st = md.get("stats") or {}
    stats = {"notional": _f((st.get("notionalTraded") or {}).get("value")), "shares": _f(st.get("sharesTraded")),
             "oi": _f(st.get("openInterest")), "last": _f((st.get("lastTradePx") or {}).get("value"))}
    return OrderBook(venue=Venue.POLYMARKET, market_id=slug + ("#short" if complement else ""), yes_asks=yes_asks,
                     no_asks=no_asks, yes_bids=yes_bids, no_bids=no_bids), stats


class PolyUSClient(HttpClient):
    def __init__(self, session=None, rps: float = POLY_US_RPS):
        super().__init__(POLY_US_GATEWAY, rps, session)
        self.warnings: list[str] = []

    # ── listings ──────────────────────────────────────────────────────────────
    async def _volume_tiers(self, floors, progress: ProgressCB) -> dict[str, float]:
        """slug -> highest volume floor it passes (server-side ``volumeNumMin`` filter; the listing has no volume)."""
        tier: dict[str, float] = {}
        for floor in sorted(floors, reverse=True):
            off = 0
            while True:
                try:
                    d = await self.get_json("/v1/markets", {"active": "true", "closed": "false", "limit": 500,
                                                            "offset": off, "volumeNumMin": floor})
                except ApiError as e:
                    self.warnings.append(f"Polymarket US volume filter ${floor:,.0f} failed (HTTP {e.status})")
                    break
                batch = d.get("markets") if isinstance(d, dict) else None
                if not batch:
                    break
                for m in batch:
                    tier.setdefault(m.get("slug", ""), floor)
                off += len(batch)
            if progress:
                progress(f"Polymarket US: {len(tier):,} markets with volume >= ${floor:,.0f}")
        return tier

    async def get_open_events(self, max_pages: int = 60, progress: ProgressCB = None,
                              volume_floors=(1_000.0, 10_000.0, 100_000.0)) -> list[Market]:
        """All open US markets (via events, so event title / league / URL are known).

        ``volume_usd`` is a LOWER BOUND (the highest floor the market passes); exact figures come from
        :meth:`enrich_stats` for the few markets that need them.
        """
        tier = await self._volume_tiers(volume_floors, progress) if volume_floors else {}
        now = utcnow()
        out: list[Market] = []
        off = 0
        for page in range(max_pages):
            try:
                d = await self.get_json("/v1/events", {"active": "true", "closed": "false", "limit": 100, "offset": off})
            except ApiError as e:
                self.warnings.append(f"Polymarket US events page {page + 1} failed (HTTP {e.status}: "
                                     f"{e.body[:120]}); continuing with {len(out)} markets")
                break
            evs = d.get("events") if isinstance(d, dict) else None
            if not evs:
                break
            for ev in evs:
                if ev.get("hidden"):
                    continue
                for m in ev.get("markets") or []:
                    for mk in markets_from_us(m, ev, now):
                        v = tier.get(m.get("slug", ""), 0.0)
                        mk.volume_usd = v
                        mk.volume = v
                        out.append(mk)
            off += len(evs)
            if progress:
                progress(f"Polymarket US events: page {page + 1}, {len(out):,} markets")
        return out

    # ── per-market data ───────────────────────────────────────────────────────
    async def get_market_book(self, market: Market) -> Optional[OrderBook]:
        slug = market.market_id.split("#")[0]
        d = await self.get_json(f"/v1/markets/{slug}/book", allow_status=(404, 400))
        parsed = book_from_us(d, slug, complement=market.market_id.endswith("#short"))
        return parsed[0] if parsed else None

    async def enrich_stats(self, markets: list[Market], concurrency: int = 8,
                           progress: ProgressCB = None) -> int:
        """Replace lower-bound volume with exact cash volume / open interest / last price (and refresh the quote)."""
        by_slug: dict[str, list[Market]] = {}
        for m in markets:
            by_slug.setdefault(m.market_id.split("#")[0], []).append(m)
        sem = asyncio.Semaphore(concurrency)
        done = 0

        async def one(slug: str, ms: list[Market]):
            nonlocal done
            async with sem:
                try:
                    d = await self.get_json(f"/v1/markets/{slug}/book", allow_status=(404, 400))
                except (ApiError, asyncio.TimeoutError):
                    return
            parsed = book_from_us(d, slug)
            if not parsed:
                return
            book, st = parsed
            for m in ms:
                short = m.market_id.endswith("#short")
                m.stats_exact = True
                if st["notional"] is not None:
                    m.volume_usd = m.volume = st["notional"]
                if st["oi"] is not None:
                    m.open_interest = st["oi"]
                if st["last"] is not None:
                    m.last_price = (1.0 - st["last"]) if short else st["last"]
                bid_l = book.yes_bids[0][0] if book.yes_bids else None
                ask_l = book.yes_asks[0][0] if book.yes_asks else None
                m.yes_bid, m.yes_ask = ((None if ask_l is None else round(1 - ask_l, 6),
                                         None if bid_l is None else round(1 - bid_l, 6)) if short else (bid_l, ask_l))
                m.no_bid = None if m.yes_ask is None else 1.0 - m.yes_ask
                m.no_ask = None if m.yes_bid is None else 1.0 - m.yes_bid
                m.no_ask_size = (book.yes_bids[0][1] if (book.yes_bids and not short) else
                                 (book.yes_asks[0][1] if (book.yes_asks and short) else 0.0))
                m.quote_ts = utcnow()
            done += 1
            if progress and done % 50 == 0:
                progress(f"Polymarket US stats: {done}/{len(by_slug)}")

        await asyncio.gather(*(one(s, ms) for s, ms in by_slug.items()))
        return done
