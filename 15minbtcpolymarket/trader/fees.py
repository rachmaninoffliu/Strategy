"""Kalshi taker fee, per order, rounded UP to the next cent exactly as the exchange does:  fee = ceil_cent(0.07 * contracts * p * (1 - p)).  Kalshi is the only venue (signal AND execution)."""
from __future__ import annotations

import math

KALSHI_COEF = 0.07

FEE_MODELS = ("Kalshi",)


def kalshi_fee_total(contracts: float, price: float) -> float:
    return math.ceil(KALSHI_COEF * contracts * price * (1 - price) * 100 - 1e-9) / 100


def fee_total(model: str, contracts: float, price: float) -> float:
    """Fee in dollars for one taker order of ``contracts`` at ``price`` (0..1).  ``model`` is kept so call sites state which schedule they assume; only 'Kalshi' exists."""
    if model not in FEE_MODELS:
        raise ValueError(f"unknown fee model {model!r} (this app trades on Kalshi only)")
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0.0
    return kalshi_fee_total(contracts, price)


def fee_per_contract(model: str, price: float, ref_contracts: int = 100) -> float:
    """Per-contract fee for a reference order size (used for the Kelly cost, before the exact size is known)."""
    return fee_total(model, ref_contracts, price) / ref_contracts
