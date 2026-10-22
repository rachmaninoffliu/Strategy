"""Fresh top-of-book for ONE market right now (used by the long-shot detail panel's 'Refresh live book')."""
from __future__ import annotations

from typing import Optional

from connectors.kalshi_client import KalshiClient
from connectors.polyus_client import PolyUSClient
from core.models import LongShot, Market, Venue, utcnow


async def fetch_live_book(ls: LongShot) -> Optional[dict]:
    """-> {'yes_bid','yes_ask','no_ask','no_ask_size','ts'} from the venue's own order book, or None.

    Supported for the US-tradable venues (Kalshi, Polymarket US). International polymarket.com is not supported
    here on purpose: it is not tradable from the US and its listing quotes are the unreliable ones.
    """
    mk = Market(venue=ls.venue, market_id=ls.market_id)
    if ls.venue is Venue.KALSHI:
        async with KalshiClient() as k:
            ob = await k.get_orderbook(ls.market_id)
    elif ls.us_tradable:
        async with PolyUSClient() as p:
            ob = await p.get_market_book(mk)
    else:
        return None
    if ob is None:
        return None
    return {
        "yes_bid": ob.yes_bids[0][0] if ob.yes_bids else None,
        "yes_ask": ob.yes_asks[0][0] if ob.yes_asks else None,
        "no_ask": ob.no_asks[0][0] if ob.no_asks else None,
        "no_ask_size": ob.no_asks[0][1] if ob.no_asks else 0.0,
        "ts": utcnow(),
    }
