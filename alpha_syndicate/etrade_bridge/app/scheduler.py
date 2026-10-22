from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .etrade_client import ETradeAuthError, ETradeClient

log = logging.getLogger(__name__)


class TokenRenewer:
    """Background asyncio task that periodically reactivates the E*TRADE access token.

    Strategy: renew once at startup (fail-fast if tokens are missing/dead), then
    every `interval_minutes`. Default 90m sits well inside E*TRADE's 2-hour idle
    deactivation window.
    """

    def __init__(self, client: ETradeClient, interval_minutes: int) -> None:
        self._client = client
        self._interval_sec = max(60, interval_minutes * 60)
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="etrade-token-renewer")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        await self._try_renew()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_sec)
            except asyncio.TimeoutError:
                await self._try_renew()

    async def _try_renew(self) -> None:
        try:
            await asyncio.to_thread(self._client.renew)
        except ETradeAuthError as exc:
            log.error("token renewal failed (handshake likely required): %s", exc)
        except Exception:
            log.exception("unexpected error during token renewal")
