"""Pydantic schemas: Market, OrderBook, ArbOpportunity, TradeLog (+ time helpers)."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import NamedTuple, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── time helpers ──────────────────────────────────────────────────────────────
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


_FRAC = re.compile(r"(\.\d{6})\d+")
_TZ_SHORT = re.compile(r"([+-]\d{2})$")


def parse_dt(v) -> Optional[datetime]:
    """Parse ISO strings / epoch seconds / epoch ms into an aware UTC datetime."""
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        x = float(v)
        if x > 1e12:
            x /= 1000.0
        return datetime.fromtimestamp(x, tz=timezone.utc)
    s = str(v).strip()
    if not s:
        return None
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = _FRAC.sub(r"\1", s)
    if _TZ_SHORT.search(s) and "T" in s:
        s += ":00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def iso(dt: Optional[datetime]) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else ""


# ── enums ─────────────────────────────────────────────────────────────────────
class Venue(str, Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class Side(str, Enum):
    YES = "YES"
    NO = "NO"

    @property
    def other(self) -> "Side":
        return Side.NO if self is Side.YES else Side.YES


class ArbType(str, Enum):
    PARITY = "I-Parity"
    MECE = "II-MECE"
    SUBSET = "III-Subset"


class Relation(str, Enum):
    EQUIVALENT = "equivalent"    # same question
    INVERSE = "inverse"          # A is the logical negation of B


# ── market / book ─────────────────────────────────────────────────────────────
class Market(BaseModel):
    """A single binary contract on one venue (prices are decimal odds 0..1)."""
    model_config = ConfigDict(extra="ignore")

    venue: Venue
    market_id: str
    event_id: str = ""
    series_id: str = ""
    event_title: str = ""
    title: str = ""
    outcome_label: str = ""
    rules: str = ""
    url: str = ""
    category: str = ""

    status: str = "open"                 # open | closed | settled
    result: Optional[str] = None         # yes | no | void | None
    settle_value: Optional[float] = None  # YES-leg payout at settlement (1 / 0 / 0.5)

    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    no_bid_size: float = 0.0
    no_ask_size: float = 0.0
    last_price: Optional[float] = None
    volume: float = 0.0
    liquidity: float = 0.0
    open_interest: float = 0.0
    volume_usd: float = 0.0              # cash volume (Kalshi: contracts x last price, an ESTIMATE)
    volume_24h_usd: float = 0.0          # < 0 means "unknown" (Polymarket US lists no 24h volume)
    us_tradable: bool = True             # Kalshi + Polymarket US = True; international polymarket.com = False
    fee_round_cents: bool = False        # Polymarket US rounds each fee to the nearest cent (banker's rounding)
    quote_ts: Optional[datetime] = None  # when the bid/ask on this object was observed
    stats_exact: bool = True             # False: volume_usd is a server-side LOWER BOUND, not the traded amount
    ladder_key: str = ""                 # exchange-native ladder id (Polymarket US: slug minus the line); beats text
    kind: str = ""                       # moneyline | spread | total | prop | future | "" (exchange-native market type)
    subject_ids: list[str] = Field(default_factory=list)   # exchange-native subject keys (team abbreviation), lowercase
    league: str = ""                     # exchange-native league key, lowercase (nfl, nba, mlb, nhl, cfb ...)
    min_order_size: float = 1.0

    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None    # trading stops
    settle_time: Optional[datetime] = None   # scheduled / expected resolution
    settled_ts: Optional[datetime] = None    # actual resolution (ex-post only)

    yes_token: str = ""
    no_token: str = ""
    neg_risk: bool = False
    neg_risk_augmented: bool = False
    mutually_exclusive: bool = False
    fees_enabled: bool = False
    fee_rate: float = 0.0
    fee_exponent: float = 1.0

    # semantic features (filled by connectors.mapper.enrich)
    underlying: str = ""
    direction: str = ""          # gt | lt | between | ""
    strike: Optional[float] = None
    strike_hi: Optional[float] = None
    style: str = ""              # terminal | touch | ""
    date_tokens: list[str] = Field(default_factory=list)
    hour_token: Optional[int] = None
    tokens: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.venue.value}:{self.market_id}"

    def ask(self, side: Side) -> Optional[float]:
        return self.yes_ask if side is Side.YES else self.no_ask

    def bid(self, side: Side) -> Optional[float]:
        return self.yes_bid if side is Side.YES else self.no_bid

    def ask_size(self, side: Side) -> float:
        return self.yes_ask_size if side is Side.YES else self.no_ask_size

    def leg_payout(self, side: Side) -> Optional[float]:
        """Realised payout of one unit of ``side`` once the market has settled."""
        if self.settle_value is None:
            return None
        return self.settle_value if side is Side.YES else 1.0 - self.settle_value


Ladder = list[tuple[float, float]]   # (price, size), best price first


class OrderBook(BaseModel):
    venue: Venue
    market_id: str
    yes_asks: Ladder = Field(default_factory=list)   # ascending price
    no_asks: Ladder = Field(default_factory=list)
    yes_bids: Ladder = Field(default_factory=list)   # descending price
    no_bids: Ladder = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=utcnow)

    def asks(self, side: Side) -> Ladder:
        return self.yes_asks if side is Side.YES else self.no_asks


class Candle(NamedTuple):
    """Bar quote: ``ts`` is the bar END (epoch seconds). Prices are decimal odds for the YES token."""
    ts: int
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    price: Optional[float]
    volume: float = 0.0


# ── arbitrage objects ─────────────────────────────────────────────────────────
class ArbLeg(BaseModel):
    venue: Venue
    market_id: str
    title: str
    side: Side
    price: float                 # volume-weighted average fill price
    limit_price: float           # worst level touched (use as limit price)
    top_price: float             # best ask
    size: float                  # contracts / shares
    fee: float = 0.0
    url: str = ""
    token_id: str = ""
    ladder: Ladder = Field(default_factory=list)
    depth_usd: float = 0.0       # $ available at levels that clear the hurdle
    rules: str = ""
    us_tradable: bool = True
    market_ref: str = ""         # human-readable identifier: Kalshi market ticker / Polymarket US slug


class ArbOpportunity(BaseModel):
    arb_type: ArbType
    event_name: str
    legs: list[ArbLeg]
    size: float                          # units bought of each leg
    total_cost: float                    # $ incl. fees
    fees: float
    guaranteed_payout: float             # $ minimum payout at ``size``
    max_payout: float                    # $ best-case payout (upside bonus)
    net_return: float                    # (payout - cost) / cost
    days_to_settle: float
    apy: float
    min_depth_usd: float
    settle_time: Optional[datetime] = None
    detected_at: datetime = Field(default_factory=utcnow)
    confidence: float = 1.0
    depth_verified: bool = False
    variant: str = ""                    # e.g. "buy-all-YES", "buy-all-NO"
    notes: list[str] = Field(default_factory=list)
    unit_min_payout: float = 1.0
    unit_max_payout: float = 1.0

    @property
    def us_tradable(self) -> bool:
        return all(l.us_tradable for l in self.legs)

    @property
    def key(self) -> str:
        raw = self.arb_type.value + "|" + "|".join(
            sorted(f"{l.venue.value}:{l.market_id}:{l.side.value}" for l in self.legs))
        return hashlib.sha1(raw.encode()).hexdigest()[:14]

    @property
    def profit(self) -> float:
        return self.guaranteed_payout - self.total_cost


class LongShot(BaseModel):
    """A market whose YES is priced as 'extremely unlikely' yet trades real volume (NOT an arbitrage)."""
    venue: Venue
    market_id: str
    title: str
    event_title: str = ""
    outcome_label: str = ""
    url: str = ""
    rules: str = ""
    prob: float                              # market-implied P(YES): bid/ask mid
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_ask: Optional[float] = None           # None when there is no YES bid (NO cannot be bought below $1)
    no_ask_size: float = 0.0
    no_depth_usd: float = 0.0                # known only where the listing carries sizes (Kalshi)
    volume_usd: float = 0.0
    volume_24h_usd: float = 0.0
    open_interest: float = 0.0               # notional $ (Kalshi contracts x $1)
    liquidity: float = 0.0
    settle_time: Optional[datetime] = None
    days: Optional[float] = None
    fee: float = 0.0
    carry_gross: Optional[float] = None      # return from buying NO and holding to resolution (if it resolves NO)
    carry_net: Optional[float] = None
    carry_apy: Optional[float] = None
    flags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)   # 'absurd' | 'catastrophic'
    other_venue: str = ""
    me_event: bool = False
    us_tradable: bool = True
    market_ref: str = ""
    stats_exact: bool = True                 # False: volume is a server-side lower bound, not the exact figure

    @property
    def key(self) -> str:
        return f"{self.venue.value}:{self.market_id}"


class TradeLog(BaseModel):
    """One (simulated or logged) position from entry to resolution."""
    opp_key: str
    arb_type: str
    event_name: str
    opened_at: datetime
    settle_time: Optional[datetime] = None      # scheduled
    released_at: Optional[datetime] = None      # actual capital release
    size: float = 0.0
    cost: float = 0.0
    fees: float = 0.0
    guaranteed_payout: float = 0.0
    max_payout: float = 0.0
    realized_payout: Optional[float] = None
    status: str = "Active / Pending Settlement"
    net_return_locked: float = 0.0
    apy_at_entry: float = 0.0
    confidence: float = 1.0
    source: str = "backfill"                    # live | backfill | backtest | manual
    legs: list[dict] = Field(default_factory=list)

    @property
    def pnl(self) -> Optional[float]:
        if self.realized_payout is None:
            return None
        return self.realized_payout - self.cost

    @property
    def unrealized(self) -> float:
        """Locked-in profit if held to settlement (minimum payout - cost)."""
        return self.guaranteed_payout - self.cost

    @property
    def lockup_days(self) -> float:
        end = self.released_at or self.settle_time
        if not end:
            return 0.0
        return max((end - self.opened_at).total_seconds() / 86400.0, 0.0)
