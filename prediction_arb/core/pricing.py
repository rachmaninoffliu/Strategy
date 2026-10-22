"""Fee models, APY / hurdle maths, book-walking and Kelly sizing (pure functions)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import APY_EXP_CAP, DAYS_PER_YEAR, MIN_DAYS, FeeConfig
from core.models import Ladder, Market, Side, Venue


# ── time value ────────────────────────────────────────────────────────────────
def days_to_settle(settle: Optional[datetime], now: datetime, min_days: float = MIN_DAYS) -> float:
    """D = max((T_settle - T_now) / 86400, 0.25)."""
    if settle is None:
        return 365.0
    return max((settle - now).total_seconds() / 86400.0, min_days)


def annualize(net_return: float, days: float) -> float:
    """APY = (1 + R)^(365.25 / D) - 1, computed in log space (overflow-safe)."""
    if net_return <= -1.0:
        return -1.0
    x = (DAYS_PER_YEAR / max(days, 1e-9)) * math.log1p(net_return)
    return math.expm1(min(x, APY_EXP_CAP))


# ── fees ──────────────────────────────────────────────────────────────────────
def kalshi_taker_fee(contracts: float, price: float, coef: float = 0.07) -> float:
    """Kalshi taker fee: ceil-to-next-cent( coef * C * P * (1 - P) )."""
    raw = coef * contracts * price * (1.0 - price)
    if raw <= 0:
        return 0.0
    return math.ceil(round(raw * 100.0, 6)) / 100.0


def poly_taker_fee(shares: float, price: float, rate: float, exponent: float = 1.0) -> float:
    """Polymarket taker fee (fee-enabled markets): C * rate * (P * (1-P)) ** exponent."""
    if rate <= 0:
        return 0.0
    return shares * rate * (price * (1.0 - price)) ** exponent


def poly_us_fee(contracts: float, price: float, coef: float = 0.0695) -> float:
    """Polymarket US taker fee: Theta * C * p * (1-p), rounded to the nearest cent, ties to even (docs.polymarket.us/fees).

    Exact decimal arithmetic: in binary floats 6.255 is 6.2549999..., which would round the wrong way.
    """
    from decimal import ROUND_HALF_EVEN, Decimal
    c, p = Decimal(repr(float(contracts))), Decimal(repr(float(price)))
    raw = Decimal(repr(float(coef))) * c * p * (Decimal(1) - p)
    return float(raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def unit_fee(market: Market, price: float, fees: FeeConfig) -> float:
    """Continuous per-unit fee (no cent rounding) used while walking the book."""
    f = fees.extra_bps / 1e4 * price
    if market.venue is Venue.KALSHI:
        f += fees.kalshi_coef * price * (1.0 - price)
    else:
        if fees.poly_mode == "schedule" and market.fees_enabled:
            f += market.fee_rate * (price * (1.0 - price)) ** market.fee_exponent
        elif fees.poly_mode == "flat":
            f += fees.poly_flat_bps / 1e4 * price
    return f


def chunk_fee(market: Market, qty: float, price: float, fees: FeeConfig) -> float:
    """Exact fee for one fill (Kalshi rounds each fill up to the next cent)."""
    f = fees.extra_bps / 1e4 * price * qty
    if market.venue is Venue.KALSHI:
        f += kalshi_taker_fee(qty, price, fees.kalshi_coef)
    else:
        if fees.poly_mode == "schedule" and market.fees_enabled:
            if market.fee_round_cents and market.fee_exponent == 1.0:
                f += poly_us_fee(qty, price, market.fee_rate)             # Polymarket US: nearest cent, half-even
            else:
                f += poly_taker_fee(qty, price, market.fee_rate, market.fee_exponent)
        elif fees.poly_mode == "flat":
            f += fees.poly_flat_bps / 1e4 * price * qty
    return f


# ── book walking ──────────────────────────────────────────────────────────────
@dataclass
class LegSpec:
    market: Market
    side: Side
    ladder: Ladder                       # ascending asks [(price, size), ...]


@dataclass
class WalkResult:
    units: float
    cost_ex_fees: float
    fees: float
    total_cost: float
    min_payout: float                    # $ at ``units``
    max_payout: float
    avg_prices: list[float]
    limit_prices: list[float]
    top_prices: list[float]
    leg_fees: list[float]
    depth_usd: list[float]               # $ consumed per leg at the cleared levels
    net_return: float
    apy: float
    chunks: list = field(default_factory=list)


def _finalize(legs: list[LegSpec], chunks: list, min_pu: float, max_pu: float,
              days: float, fees: FeeConfig) -> Optional[WalkResult]:
    units = sum(q for q, _ in chunks)
    if units <= 1e-9:
        return None
    n = len(legs)
    cost = [0.0] * n
    lf = [0.0] * n
    depth = [0.0] * n
    limit = [0.0] * n
    top = [legs[i].ladder[0][0] for i in range(n)]
    for q, prices in chunks:
        for i in range(n):
            cost[i] += q * prices[i]
            depth[i] += q * prices[i]
            lf[i] += chunk_fee(legs[i].market, q, prices[i], fees)
            limit[i] = max(limit[i], prices[i])
    total_ex = sum(cost)
    fee_total = sum(lf)
    total = total_ex + fee_total
    payout = min_pu * units
    nr = (payout - total) / total if total > 0 else 0.0
    return WalkResult(
        units=units, cost_ex_fees=total_ex, fees=fee_total, total_cost=total,
        min_payout=payout, max_payout=max_pu * units,
        avg_prices=[cost[i] / units for i in range(n)], limit_prices=limit, top_prices=top,
        leg_fees=lf, depth_usd=depth, net_return=nr, apy=annualize(nr, days), chunks=chunks)


def walk_legs(legs: list[LegSpec], *, min_unit_payout: float, max_unit_payout: Optional[float] = None,
              days: float, hurdle: float, fees: FeeConfig,
              max_units: float = math.inf, max_cost: float = math.inf,
              min_units: float = 0.0) -> Optional[WalkResult]:
    """Buy ``units`` of every leg, consuming ask levels while the *marginal* APY clears the hurdle.

    All legs are bought in equal quantity (each unit of every leg pays $1 on a win). Ladders are
    ascending in price, so marginal cost is non-decreasing and we can stop at the first miss.
    """
    n = len(legs)
    if n == 0 or any(not l.ladder for l in legs):
        return None
    max_pu = max_unit_payout if max_unit_payout is not None else min_unit_payout
    ptr = [0] * n
    remaining = [legs[i].ladder[0][1] for i in range(n)]
    chunks: list = []
    filled = 0.0
    spent = 0.0
    while True:
        prices = [legs[i].ladder[ptr[i]][0] for i in range(n)]
        if any(p <= 0 or p >= 1 for p in prices):
            break
        uc = sum(prices) + sum(unit_fee(legs[i].market, prices[i], fees) for i in range(n))
        if uc <= 0:
            break
        if annualize((min_unit_payout - uc) / uc, days) < hurdle:
            break
        qty = min(remaining)
        qty = min(qty, max_units - filled)
        if math.isfinite(max_cost):
            qty = min(qty, (max_cost - spent) / uc)
        if qty <= 1e-9:
            break
        chunks.append((qty, prices))
        filled += qty
        spent += qty * uc
        exhausted = False
        for i in range(n):
            remaining[i] -= qty
            if remaining[i] <= 1e-9:
                ptr[i] += 1
                if ptr[i] >= len(legs[i].ladder):
                    exhausted = True
                else:
                    remaining[i] = legs[i].ladder[ptr[i]][1]
        if exhausted:
            break
    # Exact (cent-rounded) fees can push a thin trade under the hurdle: trim trailing chunks.
    while chunks:
        res = _finalize(legs, chunks, min_unit_payout, max_pu, days, fees)
        if res is None:
            return None
        if res.apy >= hurdle and res.units + 1e-9 >= min_units:
            return res
        chunks = chunks[:-1]
    return None


# ── sizing ────────────────────────────────────────────────────────────────────
def kelly_fraction(net_return: float, loss_prob: float) -> float:
    """Kelly stake fraction for a bet returning ``net_return`` w.p. (1-p) and losing everything w.p. p.

    f* = ((1-p) * R - p) / R.  A pure arb (p = 0) gives f* = 1; resolution-mismatch risk shrinks it and
    makes thin edges un-tradeable (f* <= 0).
    """
    if net_return <= 0:
        return 0.0
    f = ((1.0 - loss_prob) * net_return - loss_prob) / net_return
    return max(0.0, min(1.0, f))


def annualize_arr(net_return, days):
    """Vectorised :func:`annualize` (numpy arrays in, array out; NaN where the input is NaN / <= -1)."""
    import numpy as np
    r = np.asarray(net_return, dtype=float)
    d = np.maximum(np.asarray(days, dtype=float), 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (DAYS_PER_YEAR / d) * np.log1p(np.where(r > -1.0, r, np.nan))
        return np.expm1(np.minimum(x, APY_EXP_CAP))
