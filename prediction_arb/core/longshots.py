"""Long-shot screen: events priced as 'extremely unlikely' that still trade real volume.

This is NOT arbitrage. Buying NO on a p-probability event earns ~p if it does not happen and loses ~100% of the
stake if it does. The screen therefore reports, per market:
  * the market-implied probability (bid/ask mid) - never a claim about the TRUE probability;
  * the carry of selling the tail (buy NO, hold to resolution), net of fees, annualised, vs T-bills;
  * flags that separate real opportunities from traps (no bid, overdue, long-dated, fee drag, thin book, wide spread,
    mutually-exclusive-event tail, inactive);
  * 'absurd' (supernatural / physically implausible) vs 'catastrophic' (nuclear, coup, default...) text tags - the
    latter are unlikely but NOT comfortable to be short.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from config import FeeConfig, LongShotConfig, MIN_DAYS
from core.models import LongShot, Market, Venue
from core.pricing import annualize, unit_fee

ABSURD_RE = re.compile(
    r"\b(aliens?|ufo|extraterrestrial|jesus|second coming|rapture|zombie|vampire|bigfoot|loch ness|time travel|"
    r"flat earth|immortal\w*|resurrect\w*|human(?:s)? (?:on|land\w* on) mars|visit mars|land on mars|mars colon\w*|"
    r"faster[- ]than[- ]light|perpetual motion|cure for aging|teleport\w*)\b", re.I)
CATASTROPHIC_RE = re.compile(
    r"\b(nuclear|nuke[sd]?|detonat\w*|world war|ww3|third world war|assassinat\w*|coup|invad\w*|invasion|pandemic|"
    r"asteroid|meteor\w*|extinction|civil war|martial law|default on|collaps\w*|blockade|annex\w*|"
    r"declare(?:s|d)? war)\b", re.I)


def _prob(m: Market) -> Optional[float]:
    """Market-implied P(YES): bid/ask mid; falls back to last trade, then the ask (an upper bound)."""
    if m.yes_bid is not None and m.yes_ask is not None:
        return 0.5 * (m.yes_bid + m.yes_ask)
    lp = m.last_price
    if m.yes_bid is not None:                        # bid only: YES is worth AT LEAST the bid
        return max(m.yes_bid, lp) if lp is not None else m.yes_bid
    if m.yes_ask is not None:                        # ask only: YES is worth AT MOST the ask
        return min(m.yes_ask, lp) if lp else m.yes_ask
    return lp


implied_prob = _prob


def find_longshots(markets: list[Market], cfg: LongShotConfig, fees: FeeConfig, now: datetime,
                   pair_index: Optional[dict[str, Market]] = None) -> list[LongShot]:
    """Screen ``markets`` (any statuses) and return long-shots sorted by 24h cash volume, then total volume."""
    out: list[LongShot] = []
    for m in markets:
        if m.status != "open":
            continue
        p = _prob(m)
        if p is None or not (cfg.min_prob <= p <= cfg.max_prob):
            continue
        if not (m.volume_usd >= cfg.min_volume_usd or m.open_interest >= cfg.min_open_interest):
            continue

        days = None if m.settle_time is None else (m.settle_time - now).total_seconds() / 86400.0
        no_ask = m.no_ask if (m.yes_bid is not None and m.no_ask is not None and 0.0 < m.no_ask < 0.999) else None
        fee = carry_g = carry_n = apy = None
        depth = 0.0
        flags: list[str] = []
        if no_ask is None:
            flags.append("NO UNAVAILABLE (no YES bid)")
        else:
            fee = unit_fee(m, no_ask, fees)
            carry_g = (1.0 - no_ask) / no_ask
            carry_n = (1.0 - no_ask - fee) / (no_ask + fee)
            if days is not None and days > 0 and carry_n > 0:
                apy = annualize(carry_n, max(days, MIN_DAYS))
            depth = m.no_ask_size * no_ask if m.no_ask_size > 0 else 0.0
            if carry_g > 0 and fee > 0.35 * (1.0 - no_ask):
                flags.append("FEE DRAG")
            if m.venue is Venue.KALSHI and 0 < depth < 100:
                flags.append("THIN NO BOOK")
            if apy is not None and apy < cfg.risk_free:
                flags.append("CARRY < T-BILL")
        if days is not None and days < 0:
            flags.append("OVERDUE (past end date, still open)")
        elif days is not None and days > 365:
            flags.append("LONG-DATED (>1y)")
        if m.yes_bid is not None and m.yes_ask is not None and (m.yes_ask - m.yes_bid) > max(0.02, p):
            flags.append("WIDE SPREAD")
        if 0 <= m.volume_24h_usd < 100:                  # < 0 = unknown (Polymarket US lists no 24h volume)
            flags.append("INACTIVE (24h vol < $100)")
        if not m.us_tradable:
            flags.append("NOT US-TRADABLE (international Polymarket)")
        if not m.stats_exact:
            flags.append("VOLUME IS A LOWER BOUND")
        me = bool(m.mutually_exclusive or m.neg_risk)
        if me:
            flags.append("MULTI-OUTCOME TAIL")

        text = f"{m.event_title} {m.title} {m.outcome_label}"
        tags = []
        if ABSURD_RE.search(text):
            tags.append("absurd")
        if CATASTROPHIC_RE.search(text):
            tags.append("catastrophic")

        other = ""
        if pair_index and m.key in pair_index:
            o = pair_index[m.key]
            ob = "-" if o.yes_bid is None else f"{o.yes_bid:.3f}"
            oa = "-" if o.yes_ask is None else f"{o.yes_ask:.3f}"
            other = f"{'Kalshi' if o.venue is Venue.KALSHI else 'Polymarket'} YES {ob}/{oa}"

        out.append(LongShot(
            venue=m.venue, market_id=m.market_id, title=m.title, event_title=m.event_title,
            outcome_label=m.outcome_label, url=m.url, rules=m.rules, prob=p, yes_bid=m.yes_bid, yes_ask=m.yes_ask,
            no_ask=no_ask, no_ask_size=m.no_ask_size, no_depth_usd=depth, volume_usd=m.volume_usd,
            volume_24h_usd=m.volume_24h_usd, open_interest=m.open_interest, liquidity=m.liquidity,
            settle_time=m.settle_time, days=days, fee=fee or 0.0, carry_gross=carry_g, carry_net=carry_n,
            carry_apy=apy, flags=flags, tags=tags, other_venue=other, me_event=me,
            us_tradable=m.us_tradable, market_ref=m.market_id, stats_exact=m.stats_exact))
    out.sort(key=lambda x: (-max(x.volume_24h_usd, 0.0), -x.volume_usd))
    return out


def scenario_table(ls: LongShot, multiples=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)) -> list[dict]:
    """EV of buying NO at the ask if the TRUE probability were ``m x`` the market-implied one.

    EV per $1 staked = ((1 - q) - cost) / cost, cost = NO ask + fee. Loses the whole stake with probability q.
    """
    if ls.no_ask is None:
        return []
    cost = ls.no_ask + ls.fee
    rows = []
    for k in multiples:
        q = min(ls.prob * k, 0.999)
        ev = ((1.0 - q) - cost) / cost
        rows.append({"true_prob": q, "multiple": k, "p_loss": q, "ev_pct": ev,
                     "win_pnl": (1.0 - cost) / cost, "lose_pnl": -1.0})
    return rows
