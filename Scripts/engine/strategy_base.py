"""
Module 4: Strategy Base Class & Order Types
============================================
Defines the abstract interface every strategy must implement, and the
order-type value objects the backtester consumes.

Order hierarchy:
    Order (base)
      +-- MarketOrder
      +-- LimitOrder
      +-- StopOrder
      +-- TrailingStopOrder
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Order Types
# ═══════════════════════════════════════════════════════════════════════════════

class Side(Enum):
    """Trade direction."""
    BUY = auto()
    SELL = auto()


@dataclass(frozen=True)
class Order:
    """Base order — every order carries these fields.

    Attributes:
        symbol: Ticker to trade.
        side: BUY or SELL.
        quantity: Number of shares (0 = use target-weight sizing).
        tag: Optional free-text label for the trade ledger.
    """
    symbol: str
    side: Side
    quantity: int = 0
    tag: str = ""


@dataclass(frozen=True)
class MarketOrder(Order):
    """Executes at the open of the next bar."""
    pass


@dataclass(frozen=True)
class LimitOrder(Order):
    """Executes at *limit_price* if the next bar's range allows.

    - BUY limit: fills if bar low <= limit_price  (fill at limit_price).
    - SELL limit: fills if bar high >= limit_price (fill at limit_price).

    Attributes:
        limit_price: The limit price for this order.
    """
    limit_price: float = 0.0


@dataclass(frozen=True)
class StopOrder(Order):
    """Becomes a market order when the stop level is breached.

    - BUY stop (breakout):  triggers if bar high >= stop_price.
    - SELL stop (stop-loss): triggers if bar low <= stop_price.

    Gap handling: if the bar opens *through* the stop, the fill price is
    the bar open (worst-case gap fill).

    Attributes:
        stop_price: The stop trigger price.
    """
    stop_price: float = 0.0


@dataclass(frozen=True)
class TrailingStopOrder(Order):
    """A sell-side stop that dynamically ratchets up with price.

    The stop trails by either a fixed dollar offset OR a percentage offset
    from the highest price observed since the order was placed.

    Attributes:
        trail_offset: Fixed dollar amount below the high-water mark.
        trail_pct: Percentage (0-1) below the high-water mark.
            Exactly one of trail_offset / trail_pct should be set.
        high_water_mark: Internal tracker — set/updated by the backtester.
    """
    trail_offset: Optional[float] = None
    trail_pct: Optional[float] = None
    high_water_mark: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Base Class
# ═══════════════════════════════════════════════════════════════════════════════

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Every concrete strategy must implement at least one of the two signal
    generation methods.  The backtester calls them in this order each bar:

        1. generate_signals()  ->  target_weights DataFrame
        2. generate_orders()   ->  list of Order objects

    A strategy may use weights-based allocation, discrete orders, or both.
    """

    @abstractmethod
    def generate_signals(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Produce a DataFrame of target portfolio weights.

        Args:
            data_dict: Mapping of symbol -> OHLCV DataFrame (all bars
                up to and including the *current* bar).

        Returns:
            DataFrame indexed by timestamp with one column per symbol.
            Values are target portfolio weights (e.g., 0.33 = 33% of equity).
            A weight of 0.0 means "close the position".
        """
        ...

    @abstractmethod
    def generate_orders(
        self, data_dict: dict[str, pd.DataFrame]
    ) -> list[Order]:
        """Emit discrete order objects for the backtester.

        Called once per bar *after* generate_signals().  Allows strategies
        to place limit, stop, and trailing-stop orders in addition to
        (or instead of) weight-based allocation.

        Args:
            data_dict: Same as generate_signals.

        Returns:
            A (possibly empty) list of Order objects.
        """
        ...
