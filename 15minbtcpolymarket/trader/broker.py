"""Order-routing types shared by the live engine, the paper broker and the Kalshi broker (no network, no Qt)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .fees import fee_total


class BrokerError(Exception):
    """Order state could not be determined (or the order was refused for a reason we do not understand): callers must stop trading rather than guess."""


class NotPlaced(Exception):
    """The order was definitely NOT placed (e.g. rate limited before it reached the matching engine): the signal is skipped, trading continues."""


class InsufficientFunds(NotPlaced):
    """Kalshi refused the order because the cash available to THIS market is too small (cash on another exchange shard does not count).  Nothing was placed; the session keeps running and tells you."""


@dataclass
class Fill:
    qty: int
    price: float                 # volume-weighted price per contract of the SIDE we bought (the YES price for a YES buy, the NO price for a NO buy)
    fee: float                   # total fee in dollars for the whole order
    order_id: str
    state: str = ""
    note: str = ""               # raw exchange details kept for audit (ledger)


class Broker(Protocol):
    mode: str                    # 'paper' | 'live'
    venue: str                   # 'paper' | 'kalshi'

    def buying_power(self) -> Optional[float]: ...
    def buy(self, ticker: str, side: str, contracts: int, max_cost: float, quoted_cost: float) -> Optional[Fill]: ...
    def cancel_all(self) -> None: ...


class PaperBroker:
    """Simulated fills at quote + assumed slippage (the same rule the backtest uses); sends nothing anywhere."""
    mode = "paper"
    venue = "paper"

    def __init__(self, fee_model: str, slippage: float):
        self.fee_model, self.slip = fee_model, slippage

    def buying_power(self) -> Optional[float]:
        return None

    def buy(self, ticker: str, side: str, contracts: int, max_cost: float, quoted_cost: float) -> Optional[Fill]:
        px = min(quoted_cost + self.slip, 0.99)
        return Fill(qty=contracts, price=px, fee=fee_total(self.fee_model, contracts, px), order_id="paper", state="PAPER")

    def cancel_all(self) -> None:
        pass
