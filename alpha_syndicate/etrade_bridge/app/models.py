from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class WebhookPayload(BaseModel):
    """Strict contract for the TradingView -> bridge JSON body."""

    ticker: str = Field(min_length=1, max_length=12)
    action: Literal["buy", "sell", "BUY", "SELL"]
    quantity: int = Field(gt=0, le=1_000_000)
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    limit_price: Optional[float] = Field(default=None, gt=0)

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("action")
    @classmethod
    def _upper_action(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _limit_needs_price(self) -> "WebhookPayload":
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("limit_price is required when order_type=LIMIT")
        return self
