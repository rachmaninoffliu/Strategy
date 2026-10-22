from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from .config import get_settings
from .etrade_client import ETradeAuthError, ETradeClient, ETradeOrderError
from .models import WebhookPayload
from .scheduler import TokenRenewer
from .token_store import TokenStore

log = logging.getLogger("etrade_bridge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    store = TokenStore(settings.token_store_file)
    client = ETradeClient(settings, store)
    renewer = TokenRenewer(client, settings.renewal_interval_minutes)

    app.state.settings = settings
    app.state.etrade = client
    app.state.renewer = renewer

    renewer.start()
    log.info("bridge online: env=%s api=%s", settings.etrade_env, settings.api_base)
    try:
        yield
    finally:
        await renewer.stop()
        log.info("bridge offline")


app = FastAPI(
    title="TradingView -> E*TRADE Bridge",
    version="1.0.0",
    lifespan=lifespan,
)


def _verify_secret(provided: Optional[str]) -> None:
    expected = get_settings().webhook_secret_passphrase
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_secret")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook/etrade")
async def etrade_webhook(
    request: Request,
    payload: WebhookPayload,
    secret: Optional[str] = Query(default=None),
) -> JSONResponse:
    # Accept secret via ?secret=... OR X-Webhook-Secret header. Constant-time compare.
    header_secret = request.headers.get("X-Webhook-Secret")
    _verify_secret(secret or header_secret)

    client: ETradeClient = request.app.state.etrade
    log.info(
        "webhook accepted: %s %s qty=%s type=%s",
        payload.action, payload.ticker, payload.quantity, payload.order_type,
    )

    try:
        result = await asyncio.to_thread(
            client.place_equity_order,
            symbol=payload.ticker,
            action=payload.action,
            quantity=payload.quantity,
            order_type=payload.order_type,
            limit_price=payload.limit_price,
        )
    except ETradeAuthError as exc:
        log.error("etrade auth error: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": "auth", "detail": str(exc)},
        )
    except ETradeOrderError as exc:
        log.error("etrade order rejected: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"ok": False, "error": "order_rejected", "detail": str(exc)},
        )
    except Exception as exc:
        log.exception("unexpected error placing order")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "internal", "detail": repr(exc)},
        )

    return JSONResponse(status_code=200, content={"ok": True, "result": result})
