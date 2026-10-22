"""Asynchronous public REST client for Polymarket (Gamma metadata + CLOB books / price history).

Notes from probing the live API (2026-09):
* Gamma pages cap at 100 events regardless of ``limit`` -> paginate by ``offset`` until empty;
* ``bestBid``/``bestAsk`` on a Gamma market refer to outcome-0 (the YES token);
* many markets are fee-enabled: ``feeSchedule = {rate, exponent, takerOnly}``;
* CLOB ``/book`` returns asks sorted worst-first -> always sort explicitly.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Callable, Optional

from config import POLY_CLOB, POLY_CLOB_RPS, POLY_GAMMA, POLY_GAMMA_RPS, POLY_WEB
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


def _json_list(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip():
        try:
            out = json.loads(v)
            return out if isinstance(out, list) else []
        except json.JSONDecodeError:
            return []
    return []


def markets_from_gamma(m: dict, ev: Optional[dict] = None) -> list[Market]:
    """Normalise one Gamma market into 1 binary Market (Yes/No) or 2 virtual ones (A vs B outcomes)."""
    ev = ev or {}
    outcomes = [str(o) for o in _json_list(m.get("outcomes"))]
    prices = [_f(p) for p in _json_list(m.get("outcomePrices"))]
    tokens = [str(t) for t in _json_list(m.get("clobTokenIds"))]
    if len(outcomes) != 2 or len(tokens) != 2:
        return []
    cond = m.get("conditionId") or str(m.get("id", ""))
    best_bid, best_ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
    is_yes_no = [o.lower() for o in outcomes] == ["yes", "no"]

    closed = bool(m.get("closed"))
    accepting = bool(m.get("acceptingOrders", True)) and bool(m.get("enableOrderBook", True))
    uma = (m.get("umaResolutionStatus") or "").lower()
    settled = closed and (uma == "resolved" or (prices and prices[0] is not None
                                                 and prices[0] in (0.0, 1.0, 0.5)))
    status = "settled" if settled else ("closed" if (closed or not accepting or not m.get("active", True))
                                        else "open")
    fs = m.get("feeSchedule") or {}
    fees_enabled = bool(m.get("feesEnabled")) and bool(fs)
    end = parse_dt(m.get("endDate")) or parse_dt(ev.get("endDate"))
    ev_slug, m_slug = ev.get("slug", ""), m.get("slug", "")
    url = f"{POLY_WEB}/{ev_slug or m_slug}"                      # event page (polymarket.com: NOT for US persons)
    rules = " ".join(x for x in (m.get("description") or ev.get("description"),
                                 m.get("resolutionSource") or ev.get("resolutionSource")) if x)[:2500]
    out: list[Market] = []
    for i in range(2 if not is_yes_no else 1):
        if is_yes_no:
            yb, ya = best_bid, best_ask
            yes_tok, no_tok = tokens[0], tokens[1]
            label = m.get("groupItemTitle") or ""
            title = m.get("question", "")
            sv_price = prices[0] if prices else None
            mid = cond
        else:                                            # virtual market: "does <outcome i> win?"
            if i == 0:
                yb, ya = best_bid, best_ask
            else:
                yb = None if best_ask is None else 1.0 - best_ask
                ya = None if best_bid is None else 1.0 - best_bid
            yes_tok, no_tok = tokens[i], tokens[1 - i]
            label = outcomes[i]
            title = f"{m.get('question', '')} - {outcomes[i]}"
            sv_price = prices[i] if len(prices) > i else None
            mid = f"{cond}:{i}"
        if yb is not None and yb <= 0.0:
            yb = None
        if ya is not None and ya >= 1.0:
            ya = None
        sval = sv_price if (status == "settled" and sv_price is not None) else None
        result = None
        if sval is not None:
            result = "yes" if sval >= 0.999 else ("no" if sval <= 0.001 else "void")
        out.append(Market(
            venue=Venue.POLYMARKET, market_id=mid, event_id=str(ev.get("id", "") or cond),
            series_id=str(ev.get("slug", "")), event_title=ev.get("title", "") or "", title=title,
            outcome_label=label, rules=rules, url=url,
            category=str((ev.get("tags") or [{}])[0].get("label", "")) if ev.get("tags") else "",
            status=status, result=result, settle_value=sval,
            yes_bid=yb, yes_ask=ya, no_bid=None if ya is None else 1.0 - ya,
            no_ask=None if yb is None else 1.0 - yb,
            last_price=(None if _f(m.get("lastTradePrice")) is None else
                        (_f(m.get("lastTradePrice")) if (is_yes_no or i == 0) else 1.0 - _f(m.get("lastTradePrice")))),
            volume=_f(m.get("volumeNum")) or _f(m.get("volume")) or 0.0,
            volume_usd=_f(m.get("volumeNum")) or _f(m.get("volume")) or 0.0,
            volume_24h_usd=_f(m.get("volume24hr")) or 0.0,
            liquidity=_f(m.get("liquidityNum")) or _f(m.get("liquidity")) or 0.0,
            min_order_size=_f(m.get("orderMinSize")) or 5.0,
            open_time=parse_dt(m.get("startDate")), close_time=end, settle_time=end,
            settled_ts=parse_dt(m.get("closedTime")) if settled else None,
            yes_token=yes_tok, no_token=no_tok,
            neg_risk=bool(m.get("negRisk") or ev.get("negRisk")),
            neg_risk_augmented=bool(ev.get("negRiskAugmented")),
            us_tradable=False, quote_ts=utcnow(),
            fees_enabled=fees_enabled, fee_rate=_f(fs.get("rate")) or 0.0,
            fee_exponent=_f(fs.get("exponent")) or 1.0))
    return out


def _levels(raw) -> list[tuple[float, float]]:
    out = []
    for lv in raw or []:
        p, s = _f(lv.get("price")), _f(lv.get("size"))
        if p is not None and s is not None and s > 0:
            out.append((p, s))
    return out


class PolyClient(HttpClient):
    """Gamma REST client; CLOB calls go through an internal second limiter."""

    def __init__(self, session=None, rps: float = POLY_GAMMA_RPS):
        super().__init__(POLY_GAMMA, rps, session)
        from connectors.http_base import RateLimiter
        self._clob_limiter = RateLimiter(POLY_CLOB_RPS)
        self.warnings: list[str] = []                       # surfaced by the scanner / UI (never swallowed)

    # ── Gamma ─────────────────────────────────────────────────────────────────
    async def _events(self, params: dict, max_pages: int, progress: ProgressCB, label: str,
                      stop_below_volume: Optional[float] = None) -> list[dict]:
        """Cursor pagination via /events/keyset (plain ``offset`` is rejected beyond ~2,500 rows with HTTP 422).

        A failure on a later page keeps the events already fetched and records a warning instead of discarding all.
        """
        out: list[dict] = []
        cursor: Optional[str] = None
        for page in range(max_pages):
            try:
                d = await self.get_json("/events/keyset", {**params, "limit": 100, "after_cursor": cursor})
            except ApiError as e:
                self.warnings.append(f"Polymarket {label}: page {page + 1} failed (HTTP {e.status}: "
                                     f"{e.body[:120]}); continuing with {len(out)} events")
                break
            batch = d.get("events") if isinstance(d, dict) else None
            if not batch:
                break
            out.extend(batch)
            if progress:
                progress(f"Polymarket {label}: page {page + 1}, {len(out)} events")
            if stop_below_volume is not None and (_f(batch[-1].get("volume")) or 0.0) < stop_below_volume:
                break
            cursor = d.get("next_cursor")
            if not cursor:
                break
        return out

    async def get_open_events(self, max_pages: int = 30, progress: ProgressCB = None) -> list[Market]:
        """GET /events?closed=false — most active first (24h volume)."""
        evs = await self._events({"closed": "false", "active": "true", "order": "volume24hr",
                                  "ascending": "false"}, max_pages, progress, "open")
        out: list[Market] = []
        for e in evs:
            for m in e.get("markets", []) or []:
                out.extend(markets_from_gamma(m, e))    # all statuses: baskets need complete events
        return out

    async def get_closed_events(self, start: datetime, end: datetime, max_pages: int = 60,
                                min_volume: float = 0.0, progress: ProgressCB = None) -> list[Market]:
        """GET /events?closed=true — resolved markets ending in [start, end], highest volume first."""
        evs = await self._events(
            {"closed": "true", "order": "volume", "ascending": "false",
             "end_date_min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "end_date_max": end.strftime("%Y-%m-%dT%H:%M:%SZ")},
            max_pages, progress, "closed", stop_below_volume=min_volume or None)
        out: list[Market] = []
        for e in evs:
            for m in e.get("markets", []) or []:
                for mk in markets_from_gamma(m, e):
                    if mk.status == "settled":          # event volume already >= min_volume; keep every outcome
                        out.append(mk)
        return out

    async def get_market_by_condition(self, condition_id: str) -> Optional[Market]:
        cid = condition_id.split(":")[0]
        d = await self.get_json("/markets", {"condition_ids": cid}, allow_status=(404, 422))
        if not d:
            return None
        found = markets_from_gamma(d[0])
        for mk in found:
            if mk.market_id == condition_id:
                return mk
        return found[0] if found else None

    # ── CLOB ──────────────────────────────────────────────────────────────────
    async def _clob(self, path: str, params: dict, allow_status: tuple = ()):
        await self._clob_limiter.wait()
        return await self.get_json(f"{POLY_CLOB}{path}", params, allow_status=allow_status)

    async def get_book(self, token_id: str) -> Optional[tuple[list, list]]:
        """GET /book?token_id=... -> (asks ascending, bids descending); None if the book is missing."""
        d = await self._clob("/book", {"token_id": token_id}, allow_status=(404, 400))
        if not d:
            return None
        asks = sorted(_levels(d.get("asks")), key=lambda x: x[0])
        bids = sorted(_levels(d.get("bids")), key=lambda x: -x[0])
        return asks, bids

    async def get_market_book(self, market: Market) -> Optional[OrderBook]:
        """Fetch both outcome tokens' books and merge into one YES/NO :class:`OrderBook`."""
        if not market.yes_token:
            return None
        yes, no = await asyncio.gather(self.get_book(market.yes_token),
                                       self.get_book(market.no_token) if market.no_token else asyncio.sleep(0))
        if yes is None:
            return None
        ob = OrderBook(venue=Venue.POLYMARKET, market_id=market.market_id,
                       yes_asks=yes[0], yes_bids=yes[1])
        if isinstance(no, tuple):
            ob.no_asks, ob.no_bids = no
        else:                                             # fall back to the mirrored YES book
            ob.no_asks = sorted(((round(1 - p, 6), s) for p, s in yes[1]), key=lambda x: x[0])
            ob.no_bids = sorted(((round(1 - p, 6), s) for p, s in yes[0]), key=lambda x: -x[0])
        return ob

    async def get_price_history(self, token_id: str, start: datetime, end: datetime,
                                fidelity_min: int = 1440) -> tuple[list[Candle], bool]:
        """Price series for one outcome token. Returns (candles, degraded).

        ``degraded`` is True when the requested fidelity was refused and we fell back to end-of-day
        prices (``interval=max&fidelity=1440``). Polymarket history has no bid/ask -> ``price`` only.
        """
        s, e = int(start.timestamp()), int(end.timestamp())
        step = 13 * 86400                                      # API rejects windows longer than ~14 days
        pts: dict[int, float] = {}
        degraded = False
        t = s
        try:
            if fidelity_min >= 1440:                           # daily: one request over the full lifetime
                d = await self._clob("/prices-history", {"market": token_id, "interval": "max",
                                                         "fidelity": 1440}, allow_status=(400,))
                for p in (d or {}).get("history", []):
                    if s <= int(p["t"]) <= e:
                        pts[int(p["t"])] = float(p["p"])
                t = e
            while t < e:
                t2 = min(t + step, e)
                d = await self._clob("/prices-history", {"market": token_id, "startTs": t, "endTs": t2,
                                                         "fidelity": fidelity_min}, allow_status=(400,))
                if d is None:
                    raise ApiError(400, "prices-history", "range refused")
                for p in d.get("history", []):
                    pts[int(p["t"])] = float(p["p"])
                t = t2
        except ApiError:
            degraded = fidelity_min < 1440
            pts.clear()
            d = await self._clob("/prices-history", {"market": token_id, "interval": "max", "fidelity": 1440},
                                 allow_status=(400,))
            for p in (d or {}).get("history", []):
                if s <= int(p["t"]) <= e:
                    pts[int(p["t"])] = float(p["p"])
        return [Candle(ts, None, None, pts[ts], 0.0) for ts in sorted(pts)], degraded
