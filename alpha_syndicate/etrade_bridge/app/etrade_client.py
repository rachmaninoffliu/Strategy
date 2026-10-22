from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from requests_oauthlib import OAuth1Session

from .config import Settings
from .token_store import TokenStore

log = logging.getLogger(__name__)


class ETradeAuthError(RuntimeError):
    """Token missing, expired beyond renewal, or signature rejected."""


class ETradeOrderError(RuntimeError):
    """Preview or place rejected by E*TRADE (margin, ticker, validation, etc.)."""


class ETradeClient:
    """Sync wrapper around E*TRADE REST endpoints. Drive from FastAPI via asyncio.to_thread."""

    def __init__(self, settings: Settings, store: TokenStore) -> None:
        self._settings = settings
        self._store = store
        self._session: Optional[OAuth1Session] = None

    # ------------------------------------------------------------------ session
    def _build_session(self) -> OAuth1Session:
        tokens = self._store.load()
        if tokens is None:
            raise ETradeAuthError(
                "No cached E*TRADE tokens. Run `python auth_handshake.py` first."
            )
        return OAuth1Session(
            client_key=self._settings.etrade_consumer_key,
            client_secret=self._settings.etrade_consumer_secret,
            resource_owner_key=tokens.oauth_token,
            resource_owner_secret=tokens.oauth_token_secret,
            signature_method="HMAC-SHA1",
        )

    def _require_session(self) -> OAuth1Session:
        if self._session is None:
            self._session = self._build_session()
        return self._session

    def reload_session(self) -> None:
        """Drop the cached OAuth1Session so the next call re-reads tokens from disk."""
        self._session = None

    def renew(self) -> None:
        """Reactivate an idle access token.

        E*TRADE deactivates access tokens after ~2 hours of inactivity; this endpoint
        reactivates them with no PIN required. Tokens still die at midnight US/Eastern,
        at which point a full handshake (auth_handshake.py) must run again.
        """
        sess = self._require_session()
        url = f"{self._settings.oauth_base}/oauth/renew_access_token"
        resp = sess.get(url, timeout=15)
        if resp.status_code >= 400:
            raise ETradeAuthError(
                f"Renewal failed [{resp.status_code}]: {resp.text[:300]}"
            )
        self._store.touch_renewed()
        log.info("E*TRADE access token renewed")

    # ------------------------------------------------------------------- orders
    def place_equity_order(
        self,
        *,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
    ) -> dict[str, Any]:
        """Two-step preview -> place. Returns the parsed PlaceOrderResponse JSON."""
        acct = self._settings.etrade_account_id_key
        client_order_id = uuid.uuid4().hex[:20]  # E*TRADE limit: 20 chars

        preview_body = self._build_order_body(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        preview_url = f"{self._settings.api_base}/v1/accounts/{acct}/orders/preview"
        preview_resp = self._post_json(preview_url, {"PreviewOrderRequest": preview_body})

        preview_ids = self._extract_preview_ids(preview_resp)
        if not preview_ids:
            raise ETradeOrderError(f"PreviewOrder returned no previewIds: {preview_resp}")

        place_body = dict(preview_body)
        place_body["PreviewIds"] = [{"previewId": pid} for pid in preview_ids]

        place_url = f"{self._settings.api_base}/v1/accounts/{acct}/orders/place"
        place_resp = self._post_json(place_url, {"PlaceOrderRequest": place_body})
        log.info(
            "order placed: %s %s %s qty=%s order_type=%s previewIds=%s",
            action, quantity, symbol, order_type, order_type, preview_ids,
        )
        return place_resp

    # --------------------------------------------------------------- internals
    @staticmethod
    def _build_order_body(
        *,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float],
        client_order_id: str,
    ) -> dict[str, Any]:
        order: dict[str, Any] = {
            "allOrNone": False,
            "priceType": order_type,             # MARKET | LIMIT
            "orderTerm": "GOOD_FOR_DAY",
            "marketSession": "REGULAR",
            "Instrument": [
                {
                    "Product": {"securityType": "EQ", "symbol": symbol},
                    "orderAction": action,        # BUY | SELL
                    "quantityType": "QUANTITY",
                    "quantity": quantity,
                }
            ],
        }
        if order_type == "LIMIT":
            if limit_price is None:
                raise ETradeOrderError("limit_price is required for LIMIT orders")
            order["limitPrice"] = limit_price

        return {
            "orderType": "EQ",
            "clientOrderId": client_order_id,
            "Order": [order],
        }

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        sess = self._require_session()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        resp = sess.post(url, json=body, headers=headers, timeout=20)
        if resp.status_code == 401:
            # Most likely an idle/expired token. Surface clearly so the renewer can react.
            raise ETradeAuthError(
                f"Unauthorized from {url}: {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise ETradeOrderError(
                f"E*TRADE rejected [{resp.status_code}] {url}: {resp.text[:500]}"
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise ETradeOrderError(f"Non-JSON response from {url}: {resp.text[:300]}") from exc

    @staticmethod
    def _extract_preview_ids(preview_resp: dict[str, Any]) -> list[int]:
        root = preview_resp.get("PreviewOrderResponse", preview_resp)
        ids = root.get("PreviewIds") or []
        return [item["previewId"] for item in ids if "previewId" in item]
