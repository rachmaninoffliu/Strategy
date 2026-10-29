"""
Module 2: Dynamic Backtest Engine
==================================
Event-driven backtester with multi-asset support, advanced order types
(Market, Limit, Stop, Trailing Stop), session-gap handling, dynamic
position sizing via target weights, and pluggable friction models.

Architecture:
    DynamicBacktester
      +-- ExecutionModel (abstract)  ->  FixedBpsModel (default)
      +-- _OrderBook (internal pending-order manager)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from engine.strategy_base import (
    BaseStrategy,
    LimitOrder,
    MarketOrder,
    Order,
    Side,
    StopOrder,
    TrailingStopOrder,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Execution / Friction Models
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionModel(ABC):
    """Abstract friction model applied to every fill."""

    @abstractmethod
    def compute_cost(self, fill_price: float, quantity: int) -> float:
        """Return the total dollar cost of executing *quantity* shares at *fill_price*.

        This cost is subtracted from portfolio cash on each fill.
        """
        ...


class FixedBpsModel(ExecutionModel):
    """Charges a fixed basis-point spread per trade (each side).

    Args:
        bps: One-way cost in basis points (e.g. 5 = 0.05%).
    """

    def __init__(self, bps: float = 5.0) -> None:
        self._frac = bps / 10_000

    def compute_cost(self, fill_price: float, quantity: int) -> float:
        return abs(fill_price * quantity) * self._frac


# ═══════════════════════════════════════════════════════════════════════════════
# Fill Record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Fill:
    """Immutable record of a single execution."""
    timestamp: pd.Timestamp
    symbol: str
    side: str          # "BUY" or "SELL"
    quantity: int
    fill_price: float
    cost: float
    tag: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Pending Order Wrapper (mutable for trailing stop updates)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _PendingOrder:
    """Wraps an Order and tracks mutable state (e.g. trailing HWM)."""
    order: Order
    hwm: float = 0.0   # high-water mark for trailing stops


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Backtester
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicBacktester:
    """Event-driven multi-asset backtesting engine.

    Args:
        starting_capital: Initial cash balance.
        execution_model: Friction model applied to fills.
    """

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        execution_model: Optional[ExecutionModel] = None,
    ) -> None:
        self._starting_capital = starting_capital
        self._exec_model = execution_model or FixedBpsModel(bps=5)

        # Portfolio state
        self._cash: float = 0.0
        self._positions: dict[str, int] = {}       # symbol -> shares
        self._pending: list[_PendingOrder] = []     # open orders
        self._fills: list[Fill] = []                # execution log

        # Equity tracking
        self._equity_records: list[dict] = []

    # ── public entry point ───────────────────────────────────────────────

    def run(
        self,
        strategy: BaseStrategy,
        data_dict: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute a full backtest.

        Args:
            strategy: Concrete BaseStrategy implementation.
            data_dict: Symbol -> OHLCV DataFrame (all same length / aligned).

        Returns:
            Tuple of (equity_curve, drawdown_curve, trade_ledger) DataFrames.
        """
        self._reset()

        # Build a unified sorted timestamp index across all symbols
        all_timestamps = sorted(
            set().union(*(df.index for df in data_dict.values()))
        )
        logger.info(
            "Backtest: %d bars, %d symbols, $%.0f capital",
            len(all_timestamps), len(data_dict), self._starting_capital,
        )

        # Pre-compute searchsorted positions for O(n log n) instead of O(n²)
        sym_positions: dict[str, np.ndarray] = {}
        for sym, df in data_dict.items():
            sym_positions[sym] = df.index.searchsorted(all_timestamps, side="right")

        for bar_idx, ts in enumerate(all_timestamps):
            # Slice data up to current bar using pre-computed positions
            current_data: dict[str, pd.DataFrame] = {}
            current_bars: dict[str, pd.Series] = {}
            for sym, df in data_dict.items():
                pos = sym_positions[sym][bar_idx]
                if pos == 0:
                    continue
                current_data[sym] = df.iloc[:pos]
                current_bars[sym] = df.iloc[pos - 1]

            if not current_bars:
                continue

            # ── 1. Process pending orders against THIS bar ───────────
            fills_before = len(self._fills)
            self._process_pending_orders(current_bars, ts)

            # Notify strategy of new fills so it can update state
            for fill in self._fills[fills_before:]:
                if hasattr(strategy, "on_fill"):
                    strategy.on_fill(fill.symbol, fill.side)

            # ── 2. Ask strategy for signals / orders ─────────────────
            # Expose position state to strategy
            if hasattr(strategy, "_backtester_positions"):
                strategy._backtester_positions = dict(self._positions)
                strategy._backtester_equity = self._cash + sum(
                    shares * current_bars[s]["close"]
                    for s, shares in self._positions.items()
                    if s in current_bars
                )

            target_weights = strategy.generate_signals(current_data)
            discrete_orders = strategy.generate_orders(current_data)

            # ── 3. Convert target weights into MarketOrders for NEXT bar
            if target_weights is not None and not target_weights.empty:
                weight_orders = self._weights_to_orders(
                    target_weights, current_bars, ts
                )
                discrete_orders = (discrete_orders or []) + weight_orders

            # ── 4. Queue new orders as pending (execute on next bar) ─
            for order in (discrete_orders or []):
                if isinstance(order, MarketOrder):
                    # Market orders fill at next bar's open — queue them
                    self._pending.append(_PendingOrder(order=order))
                elif isinstance(order, TrailingStopOrder):
                    # Initialize HWM with current price
                    price_now = current_bars.get(order.symbol)
                    hwm = price_now["close"] if price_now is not None else 0
                    self._pending.append(_PendingOrder(order=order, hwm=hwm))
                else:
                    self._pending.append(_PendingOrder(order=order))

            # ── 5. Mark-to-market equity ─────────────────────────────
            mtm = self._cash
            for sym, shares in self._positions.items():
                if sym in current_bars:
                    mtm += shares * current_bars[sym]["close"]
            self._equity_records.append({"timestamp": ts, "equity": mtm})

        return self._build_output()

    # ── order processing ─────────────────────────────────────────────────

    def _process_pending_orders(
        self, bars: dict[str, pd.Series], ts: pd.Timestamp
    ) -> None:
        """Attempt to fill every pending order against the current bar."""
        still_pending: list[_PendingOrder] = []

        for po in self._pending:
            order = po.order
            bar = bars.get(order.symbol)
            if bar is None:
                still_pending.append(po)
                continue

            fill_price = self._try_fill(po, bar)
            if fill_price is not None:
                self._execute_fill(order, fill_price, ts)
            else:
                # Update trailing-stop HWM even if not filled
                if isinstance(order, TrailingStopOrder):
                    po.hwm = max(po.hwm, bar["high"])
                still_pending.append(po)

        self._pending = still_pending

    def _try_fill(self, po: _PendingOrder, bar: pd.Series) -> Optional[float]:
        """Determine fill price for an order against a single bar.

        Returns fill price, or None if not filled.
        """
        order = po.order
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]

        # ── MarketOrder ──────────────────────────────────────────────
        if isinstance(order, MarketOrder):
            return o

        # ── LimitOrder ───────────────────────────────────────────────
        if isinstance(order, LimitOrder):
            lp = order.limit_price
            if order.side == Side.BUY and l <= lp:
                # If the bar opens below limit, fill at the (better) open
                return min(o, lp)
            if order.side == Side.SELL and h >= lp:
                return max(o, lp)
            return None

        # ── StopOrder ────────────────────────────────────────────────
        if isinstance(order, StopOrder):
            sp = order.stop_price
            if order.side == Side.BUY and h >= sp:
                # Gap-up: open is already above stop -> fill at open
                return max(o, sp)
            if order.side == Side.SELL and l <= sp:
                # Gap-down: open is already below stop -> fill at open
                return min(o, sp)
            return None

        # ── TrailingStopOrder (sell-side only) ───────────────────────
        if isinstance(order, TrailingStopOrder):
            # Update HWM with this bar's high
            po.hwm = max(po.hwm, h)

            # Compute trailing stop level
            if order.trail_pct is not None:
                stop_level = po.hwm * (1 - order.trail_pct)
            elif order.trail_offset is not None:
                stop_level = po.hwm - order.trail_offset
            else:
                return None

            if l <= stop_level:
                # Gap handling: if open is below stop level, fill at open
                return min(o, stop_level)
            return None

        return None

    def _execute_fill(
        self, order: Order, fill_price: float, ts: pd.Timestamp
    ) -> None:
        """Record a fill and update portfolio state."""
        qty = order.quantity
        cost = self._exec_model.compute_cost(fill_price, qty)

        if order.side == Side.BUY:
            self._cash -= (fill_price * qty + cost)
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) + qty
            )
        else:  # SELL
            self._cash += (fill_price * qty - cost)
            self._positions[order.symbol] = (
                self._positions.get(order.symbol, 0) - qty
            )
            # Clean up zero positions
            if self._positions[order.symbol] == 0:
                del self._positions[order.symbol]

        self._fills.append(Fill(
            timestamp=ts,
            symbol=order.symbol,
            side=order.side.name,
            quantity=qty,
            fill_price=fill_price,
            cost=cost,
            tag=order.tag,
        ))

    # ── weight-based sizing ──────────────────────────────────────────────

    def _weights_to_orders(
        self,
        weights_df: pd.DataFrame,
        bars: dict[str, pd.Series],
        ts: pd.Timestamp,
    ) -> list[MarketOrder]:
        """Convert the latest row of target weights into MarketOrders.

        Uses current portfolio equity and prices to compute share deltas.
        """
        if weights_df.empty:
            return []

        # Use the last row of the weights frame
        last_row = weights_df.iloc[-1]
        orders: list[MarketOrder] = []

        # Current equity
        equity = self._cash
        for sym, shares in self._positions.items():
            if sym in bars:
                equity += shares * bars[sym]["close"]

        for sym in last_row.index:
            target_w = last_row[sym]
            if pd.isna(target_w) or sym not in bars:
                continue

            price = bars[sym]["close"]
            if price <= 0:
                continue

            target_shares = int((equity * target_w) / price)
            current_shares = self._positions.get(sym, 0)
            delta = target_shares - current_shares

            if delta > 0:
                orders.append(MarketOrder(
                    symbol=sym, side=Side.BUY, quantity=delta, tag="rebalance"
                ))
            elif delta < 0:
                orders.append(MarketOrder(
                    symbol=sym, side=Side.SELL, quantity=abs(delta), tag="rebalance"
                ))

        return orders

    # ── output builders ──────────────────────────────────────────────────

    def _build_output(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Package backtest results into DataFrames."""
        # Equity curve
        eq_df = pd.DataFrame(self._equity_records).set_index("timestamp")
        # Collapse to daily (end-of-day equity)
        eq_daily = eq_df.resample("D").last().dropna()

        # Drawdown curve
        peak = eq_daily["equity"].cummax()
        dd = (eq_daily["equity"] - peak) / peak * 100
        dd_df = dd.to_frame(name="drawdown")

        # Trade ledger
        if self._fills:
            ledger = pd.DataFrame([
                {
                    "timestamp": f.timestamp,
                    "symbol": f.symbol,
                    "side": f.side,
                    "quantity": f.quantity,
                    "fill_price": f.fill_price,
                    "cost": f.cost,
                    "tag": f.tag,
                }
                for f in self._fills
            ])
        else:
            ledger = pd.DataFrame(columns=[
                "timestamp", "symbol", "side", "quantity",
                "fill_price", "cost", "tag",
            ])

        return eq_daily, dd_df, ledger

    def _reset(self) -> None:
        """Reset all state for a fresh run."""
        self._cash = self._starting_capital
        self._positions = {}
        self._pending = []
        self._fills = []
        self._equity_records = []
