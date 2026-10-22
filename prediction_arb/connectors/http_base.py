"""Shared async HTTP plumbing: token-bucket rate limiter, retry/back-off, JSON GET."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import aiohttp

from config import CONCURRENCY, HTTP_TIMEOUT_S, MAX_RETRIES, USER_AGENT


class ApiError(Exception):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status
        self.url = url
        self.body = body


class RateLimiter:
    """Simple pacing limiter: at most ``rps`` request starts per second."""

    def __init__(self, rps: float):
        self.interval = 1.0 / max(rps, 0.1)
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            await asyncio.sleep(start - now)


class HttpClient:
    """Base for the venue clients. Use as ``async with Client() as c``."""

    def __init__(self, base_url: str, rps: float, session: Optional[aiohttp.ClientSession] = None,
                 concurrency: int = CONCURRENCY):
        self.base_url = base_url.rstrip("/")
        self._limiter = RateLimiter(rps)
        self._sem = asyncio.Semaphore(concurrency)
        self._session = session
        self._own_session = session is None
        self.request_count = 0
        self.retry_count = 0

    async def __aenter__(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S),
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        return self

    async def __aexit__(self, *exc):
        if self._own_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def get_json(self, path: str, params: Optional[dict] = None, *,
                       allow_status: tuple = ()) -> Any:
        """GET with pacing, retries on 429/5xx/network errors. ``allow_status`` returns None for those codes."""
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        params = {k: v for k, v in (params or {}).items() if v is not None}
        delay = 1.0
        last: Exception = RuntimeError("unreachable")
        for attempt in range(MAX_RETRIES + 1):
            await self._limiter.wait()
            try:
                async with self._sem:
                    self.request_count += 1
                    async with self._session.get(url, params=params) as r:
                        if r.status == 200:
                            return await r.json(content_type=None)
                        body = await r.text()
                        if r.status in allow_status:
                            return None
                        if r.status == 429 or r.status >= 500:
                            ra = r.headers.get("Retry-After")
                            wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else delay
                            last = ApiError(r.status, url, body)
                            self.retry_count += 1
                            await asyncio.sleep(min(wait, 30.0))
                            delay = min(delay * 2, 30.0)
                            continue
                        raise ApiError(r.status, url, body)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last = e
                self.retry_count += 1
                await asyncio.sleep(min(delay, 30.0))
                delay = min(delay * 2, 30.0)
        raise last
