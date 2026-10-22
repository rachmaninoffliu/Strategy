"""Asynchronous public REST client for Kalshi (live + historical partitions).

Notes from probing the live API (2026-09):
* prices arrive as ``*_dollars`` strings and sizes as ``*_fp`` strings (legacy integer cents are still parsed);
* data settled before ``/historical/cutoff`` lives under ``/historical/*`` (newest-first, no time filters);
* multivariate ("MVE") combo markets flood listings -> ``mve_filter=exclude``.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from config import KALSHI_BASE, KALSHI_RPS, KALSHI_WEB
from connectors.http_base import ApiError, HttpClient
from core.models import Candle, Market, OrderBook, Venue, parse_dt, utcnow

ProgressCB = Optional[Callable[[str], None]]


def _f(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _dollars(d: dict, name: str) -> Optional[float]:
    """Read ``<name>_dollars`` (preferred) or legacy integer-cent ``<name>`` and return decimal odds."""
    v = _f(d.get(name + "_dollars"))
    if v is not None:
        return v
    c = _f(d.get(name))
    return None if c is None else c / 100.0


def _px(d: dict, yb: Optional[float], ya: Optional[float]) -> float:
    """Representative price used to turn Kalshi contract volume into an (estimated) cash volume."""
    lp = _dollars(d, "last_price")
    if lp:
        return lp
    if yb is not None and ya is not None:
        return 0.5 * (yb + ya)
    return ya if ya is not None else (yb or 0.0)


def _kind_of(series: str) -> str:
    """Kalshi game series are named KX<LEAGUE>GAME / ...MATCH (moneyline), ...SPREAD, ...TOTAL."""
    s = (series or "").upper()
    if s.endswith(("GAME", "MATCH")):
        return "moneyline"
    if s.endswith("SPREAD"):
        return "spread"
    if s.endswith("TOTAL"):
        return "total"
    return ""


def _league_of(series: str) -> str:
    s = re.sub(r"^KX", "", (series or "").upper())
    return re.sub(r"(GAME|MATCH)$", "", s).lower()


def _series_of(event_ticker: str) -> str:
    return event_ticker.split("-")[0] if event_ticker else ""


_STRIKE_DIR = {"greater": "gt", "greater_or_equal": "gt", "less": "lt", "less_or_equal": "lt",
               "between": "between"}


def market_from_json(d: dict, event: Optional[dict] = None) -> Market:
    """Normalise one Kalshi market JSON object (prices -> decimal odds 0..1)."""
    event = event or {}
    tk = d.get("ticker", "")
    ev_tk = d.get("event_ticker", "") or event.get("event_ticker", "")
    series = event.get("series_ticker") or _series_of(ev_tk)

    yb, ya = _dollars(d, "yes_bid"), _dollars(d, "yes_ask")
    nb, na = _dollars(d, "no_bid"), _dollars(d, "no_ask")
    ybs = _f(d.get("yes_bid_size_fp")) or 0.0
    yas = _f(d.get("yes_ask_size_fp")) or 0.0
    # Empty side of the book shows up as bid 0 / ask 1 with zero size.
    if yb is not None and (yb <= 0.0 or ybs <= 0.0):
        yb = None
    if ya is not None and (ya >= 1.0 or yas <= 0.0):
        ya = None
    # NO side is the mirror of the YES book.
    nb = None if ya is None else 1.0 - ya
    na = None if yb is None else 1.0 - yb
    nbs, nas = yas, ybs

    result = (d.get("result") or "").lower() or None
    sval = _f(d.get("settlement_value_dollars"))
    if result == "yes":
        sval = 1.0 if sval is None else sval
    elif result == "no":
        sval = 0.0 if sval is None else sval
    status_raw = (d.get("status") or "").lower()
    if result in ("yes", "no") or status_raw in ("settled", "finalized"):
        status = "settled"
        if result not in ("yes", "no") and sval is None:
            result = "void"
    elif status_raw in ("active", "open"):
        status = "open"
    else:
        status = "closed"

    strike_type = (d.get("strike_type") or "").lower()
    direction = _STRIKE_DIR.get(strike_type, "")
    floor_s, cap_s = _f(d.get("floor_strike")), _f(d.get("cap_strike"))
    strike = strike_hi = None
    if direction == "gt":
        strike = floor_s
    elif direction == "lt":
        strike = cap_s
    elif direction == "between":
        strike, strike_hi = floor_s, cap_s
    if strike is not None and abs(strike - round(strike)) < 0.015:   # 73099.99 -> 73100
        strike = float(round(strike))
    if strike_hi is not None and abs(strike_hi - round(strike_hi)) < 0.015:
        strike_hi = float(round(strike_hi))

    expected = parse_dt(d.get("expected_expiration_time"))
    close_t = parse_dt(d.get("close_time"))
    settle = expected or parse_dt(d.get("expiration_time")) or close_t
    rules = " ".join(x for x in (d.get("rules_primary"), d.get("rules_secondary")) if x)[:2500]
    return Market(
        venue=Venue.KALSHI, market_id=tk, event_id=ev_tk, series_id=series,
        event_title=event.get("title", "") or "", title=d.get("title", "") or "",
        outcome_label=d.get("yes_sub_title", "") or "", rules=rules,
        url=f"{KALSHI_WEB}/{ev_tk}" if ev_tk else "https://kalshi.com",      # /events/{EVENT_TICKER} -> that event
        quote_ts=utcnow(), kind=_kind_of(series),
        subject_ids=[tk.rsplit("-", 1)[-1].lower()] if (_kind_of(series) == "moneyline" and "-" in tk) else [],
        league=_league_of(series) if _kind_of(series) == "moneyline" else "",
        category=event.get("category", "") or "", status=status, result=result, settle_value=sval,
        yes_bid=yb, yes_ask=ya, no_bid=nb, no_ask=na,
        yes_bid_size=ybs, yes_ask_size=yas, no_bid_size=nbs, no_ask_size=nas,
        last_price=_dollars(d, "last_price"), volume=_f(d.get("volume_fp")) or _f(d.get("volume")) or 0.0,
        volume_usd=(_f(d.get("volume_fp")) or _f(d.get("volume")) or 0.0) * _px(d, yb, ya),
        volume_24h_usd=(_f(d.get("volume_24h_fp")) or 0.0) * _px(d, yb, ya),
        liquidity=_f(d.get("liquidity_dollars")) or 0.0, open_interest=_f(d.get("open_interest_fp")) or 0.0,
        open_time=parse_dt(d.get("open_time")), close_time=close_t, settle_time=settle,
        settled_ts=parse_dt(d.get("settlement_ts")),
        mutually_exclusive=bool(event.get("mutually_exclusive", False)),
        direction=direction, strike=strike, strike_hi=strike_hi)


def _candle_from_json(c: dict) -> Optional[Candle]:
    ts = c.get("end_period_ts")
    if ts is None:
        return None

    def pick(block, key):
        if not isinstance(block, dict):
            return None
        v = block.get(key + "_dollars", block.get(key))
        return _f(v)

    yb = pick(c.get("yes_bid"), "close")
    ya = pick(c.get("yes_ask"), "close")
    pr = c.get("price") or {}
    price = pick(pr, "close")
    if price is None:
        price = pick(pr, "mean")
    vol = _f(c.get("volume_fp", c.get("volume"))) or 0.0
    return Candle(int(ts), yb, ya, price, vol)


class KalshiClient(HttpClient):
    def __init__(self, session=None, rps: float = KALSHI_RPS):
        super().__init__(KALSHI_BASE, rps, session)
        self._cutoff: Optional[datetime] = None

    # ── listing ───────────────────────────────────────────────────────────────
    async def get_cutoff(self) -> Optional[datetime]:
        if self._cutoff is None:
            try:
                d = await self.get_json("/historical/cutoff")
                self._cutoff = parse_dt(d.get("market_settled_ts"))
            except ApiError:
                self._cutoff = None
        return self._cutoff

    async def get_markets(self, status: str = "open", limit: int = 1000, max_pages: int = 5,
                          progress: ProgressCB = None, historical: bool = False, **filters) -> list[Market]:
        """GET /markets?status=...&limit=1000 (cursor-paginated). MVE combo markets are excluded."""
        path = "/historical/markets" if historical else "/markets"
        out: list[Market] = []
        cursor = None
        for page in range(max_pages):
            params = {"limit": limit, "cursor": cursor, "mve_filter": "exclude", **filters}
            if not historical:
                params["status"] = status
            d = await self.get_json(path, params)
            batch = d.get("markets", [])
            out.extend(market_from_json(m) for m in batch if not m.get("mve_collection_ticker"))
            if progress:
                progress(f"Kalshi {status}{' (hist)' if historical else ''}: page {page + 1}, {len(out)} markets")
            cursor = d.get("cursor")
            if not cursor or not batch:
                break
        return out

    async def get_open_events(self, max_pages: int = 30, progress: ProgressCB = None) -> list[Market]:
        """Open events with nested markets (carries ``mutually_exclusive`` + category per event)."""
        out: list[Market] = []
        cursor = None
        for page in range(max_pages):
            d = await self.get_json("/events", {"status": "open", "limit": 200, "with_nested_markets": "true",
                                                "mve_filter": "exclude", "cursor": cursor})
            evs = d.get("events", [])
            for e in evs:
                if str(e.get("event_ticker", "")).startswith("KXMVE"):
                    continue
                for m in e.get("markets", []) or []:
                    if m.get("mve_collection_ticker"):
                        continue
                    out.append(market_from_json(m, e))      # all statuses: baskets need complete events
            if progress:
                progress(f"Kalshi events: page {page + 1}, {len(out)} open markets")
            cursor = d.get("cursor")
            if not cursor or not evs:
                break
        return out

    async def get_settled_markets(self, start: datetime, end: datetime, max_pages: int = 60,
                                  min_volume: float = 0.0, progress: ProgressCB = None) -> list[Market]:
        """Settled markets whose trading closed in [start, end] (live partition + historical partition)."""
        cutoff = await self.get_cutoff()
        s_ts, e_ts = int(start.timestamp()), int(end.timestamp())
        out: dict[str, Market] = {}

        def keep(mk: Market) -> bool:
            return (mk.close_time is not None and start <= mk.close_time <= end
                    and mk.status == "settled" and mk.volume >= min_volume)

        # live partition (server-side close-time filter)
        cursor = None
        for page in range(max_pages):
            d = await self.get_json("/markets", {"status": "settled", "limit": 1000, "mve_filter": "exclude",
                                                 "min_close_ts": s_ts, "max_close_ts": e_ts, "cursor": cursor})
            batch = d.get("markets", [])
            for m in batch:
                if m.get("mve_collection_ticker"):
                    continue
                mk = market_from_json(m)
                if keep(mk):
                    out[mk.market_id] = mk
            if progress:
                progress(f"Kalshi settled (live): page {page + 1}, kept {len(out)}")
            cursor = d.get("cursor")
            if not cursor or not batch:
                break
        # historical partition: newest-first, no server filter -> page back until before ``start``
        if cutoff is None or start < cutoff:
            cursor = None
            for page in range(max_pages * 3):
                d = await self.get_json("/historical/markets", {"limit": 1000, "mve_filter": "exclude",
                                                                "cursor": cursor})
                batch = d.get("markets", [])
                oldest = None
                for m in batch:
                    if m.get("mve_collection_ticker"):
                        continue
                    mk = market_from_json(m)
                    if mk.close_time and (oldest is None or mk.close_time < oldest):
                        oldest = mk.close_time
                    if keep(mk):
                        out[mk.market_id] = mk
                if progress:
                    progress(f"Kalshi settled (historical): page {page + 1}, kept {len(out)}")
                cursor = d.get("cursor")
                if not cursor or not batch or (oldest is not None and oldest < start):
                    break
        return list(out.values())

    async def get_event_meta(self, event_ticker: str) -> Optional[dict]:
        d = await self.get_json(f"/events/{event_ticker}", allow_status=(404, 400))
        return (d or {}).get("event")

    async def get_event_full(self, event_ticker: str) -> Optional[tuple[dict, list[Market]]]:
        """One event with ALL of its markets (any status). None if the event is archived / unknown."""
        d = await self.get_json(f"/events/{event_ticker}", {"with_nested_markets": "true"},
                                allow_status=(404, 400))
        if not d or "event" not in d:
            return None
        ev = d["event"]
        raw = d.get("markets") or ev.get("markets") or []
        return ev, [market_from_json(m, ev) for m in raw if not m.get("mve_collection_ticker")]

    async def get_market(self, ticker: str) -> Optional[Market]:
        d = await self.get_json(f"/markets/{ticker}", allow_status=(404,))
        if not d or "market" not in d:
            return None
        return market_from_json(d["market"])

    # ── history ───────────────────────────────────────────────────────────────
    async def get_candlesticks(self, market: Market, start: datetime, end: datetime,
                               period_minutes: int = 1440) -> list[Candle]:
        """OHLC bars with yes bid/ask closes. Tries the live route first (or historical for old markets)."""
        cutoff = await self.get_cutoff()
        ref = market.settled_ts or market.close_time
        old = bool(cutoff and ref and ref < cutoff)
        series = market.series_id or _series_of(market.event_id)
        span = period_minutes * 60 * 1500                      # <= 1500 bars per request
        routes = [f"/historical/markets/{market.market_id}/candlesticks",
                  f"/series/{series}/markets/{market.market_id}/candlesticks"]
        if not old:
            routes.reverse()
        s, e = int(start.timestamp()), int(end.timestamp())
        out: dict[int, Candle] = {}
        for route in routes:
            ok = True
            t = s
            while t < e:
                t2 = min(t + span, e)
                d = await self.get_json(route, {"start_ts": t, "end_ts": t2, "period_interval": period_minutes},
                                        allow_status=(400, 404))
                if d is None:
                    ok = False
                    break
                for c in d.get("candlesticks", []):
                    cd = _candle_from_json(c)
                    if cd:
                        out[cd.ts] = cd
                t = t2
            if ok and out:
                break
        return [out[k] for k in sorted(out)]

    async def get_orderbook(self, ticker: str) -> Optional[OrderBook]:
        d = await self.get_json(f"/markets/{ticker}/orderbook", allow_status=(404,))
        if not d:
            return None
        ob = d.get("orderbook_fp") or {}
        if ob:
            yes = [(float(p), float(s)) for p, s in ob.get("yes_dollars", [])]
            no = [(float(p), float(s)) for p, s in ob.get("no_dollars", [])]
        else:                                                    # legacy integer-cent format
            legacy = d.get("orderbook") or {}
            yes = [(p / 100.0, float(s)) for p, s in (legacy.get("yes") or [])]
            no = [(p / 100.0, float(s)) for p, s in (legacy.get("no") or [])]
        yes = [(p, s) for p, s in yes if s > 0]
        no = [(p, s) for p, s in no if s > 0]
        yes_bids = sorted(yes, key=lambda x: -x[0])
        no_bids = sorted(no, key=lambda x: -x[0])
        # A NO bid at p is a YES ask at 1-p (and vice-versa).
        yes_asks = sorted(((round(1.0 - p, 6), s) for p, s in no_bids), key=lambda x: x[0])
        no_asks = sorted(((round(1.0 - p, 6), s) for p, s in yes_bids), key=lambda x: x[0])
        return OrderBook(venue=Venue.KALSHI, market_id=ticker, yes_asks=yes_asks, no_asks=no_asks,
                         yes_bids=yes_bids, no_bids=no_bids)
