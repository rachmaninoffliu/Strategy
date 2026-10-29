"""
Backtest Engine Package
=======================
Provides data fetching, order types, strategy base class,
backtesting engine, and performance analytics.
"""

from engine.data_engine import DataFetcher
from engine.strategy_base import (
    BaseStrategy,
    LimitOrder,
    MarketOrder,
    Order,
    Side,
    StopOrder,
    TrailingStopOrder,
)
from engine.backtester import DynamicBacktester, ExecutionModel, FixedBpsModel
from engine.metrics import QuantMetrics, MetricsReport

__all__ = [
    "DataFetcher",
    "BaseStrategy",
    "LimitOrder",
    "MarketOrder",
    "Order",
    "Side",
    "StopOrder",
    "TrailingStopOrder",
    "DynamicBacktester",
    "ExecutionModel",
    "FixedBpsModel",
    "QuantMetrics",
    "MetricsReport",
]
