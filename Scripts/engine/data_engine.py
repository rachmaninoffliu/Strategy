"""
Module 1: Data Engine
=====================
Robust data pipeline for fetching, caching, and serving intraday market data
via the Alpaca Markets API.

Features:
    - Single-ticker or multi-ticker fetching
    - Split/dividend adjustment (adjustment='all')
    - UTC -> America/New_York timezone conversion
    - Extended-hours toggle (RTH vs. pre/post market)
    - Local parquet caching to avoid redundant API calls
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path
from typing import Union

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and caches intraday bar data from Alpaca.

    Args:
        api_key: Alpaca API key.
        secret_key: Alpaca secret key.
        cache_dir: Directory for parquet file cache.
        feed: Alpaca data feed ('iex' or 'sip').
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        cache_dir: Union[str, Path] = "cache",
        feed: str = "iex",
    ) -> None:
        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._feed = feed

    # ── public API ───────────────────────────────────────────────────────

    def fetch(
        self,
        symbols: Union[str, list[str]],
        start: Union[str, dt.datetime],
        end: Union[str, dt.datetime],
        timeframe_minutes: int = 15,
        include_extended_hours: bool = False,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV bars for one or more symbols.

        Args:
            symbols: A single ticker string or list of tickers.
            start: Start date/datetime (inclusive).
            end: End date/datetime (inclusive).
            timeframe_minutes: Bar size in minutes (e.g. 1, 5, 15, 60).
            include_extended_hours: If True, include pre/post market
                (04:00-20:00 ET). If False, only RTH (09:30-16:00 ET).
            use_cache: If True, check local parquet cache before calling API.

        Returns:
            Dict mapping each symbol to its OHLCV DataFrame, indexed by
            timezone-aware Eastern timestamps.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = [s.upper() for s in symbols]

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            cache_key = self._cache_key(
                sym, start_dt, end_dt, timeframe_minutes, include_extended_hours
            )
            cache_path = self._cache_dir / f"{cache_key}.parquet"

            if use_cache and cache_path.exists():
                logger.info("Cache hit: %s", cache_path.name)
                df = pd.read_parquet(cache_path)
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(
                    "America/New_York"
                )
                result[sym] = df
                continue

            logger.info("Fetching %s from Alpaca (%s -> %s) ...", sym, start_dt.date(), end_dt.date())
            df = self._fetch_symbol(sym, start_dt, end_dt, timeframe_minutes)

            if df.empty:
                logger.warning("No data returned for %s", sym)
                result[sym] = df
                continue

            # Timezone conversion: UTC -> Eastern
            df.index = df.index.tz_convert("America/New_York")

            # Extended hours filter
            if not include_extended_hours:
                df = df.between_time("09:30", "15:59")

            df = df[~df.index.duplicated(keep="first")].sort_index()

            # Cache to parquet
            df.to_parquet(cache_path)
            logger.info("Cached %s: %d bars -> %s", sym, len(df), cache_path.name)

            result[sym] = df

        return result

    # ── private helpers ──────────────────────────────────────────────────

    def _fetch_symbol(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tf_minutes: int,
    ) -> pd.DataFrame:
        """Fetch bars in chunked 6-month windows to stay within API limits."""
        frames: list[pd.DataFrame] = []
        cursor = start.to_pydatetime()
        end_dt = end.to_pydatetime()

        while cursor < end_dt:
            chunk_end = min(cursor + dt.timedelta(days=180), end_dt)
            try:
                if tf_minutes >= 60:
                    tf = TimeFrame(tf_minutes // 60, TimeFrame.Hour.unit)
                else:
                    tf = TimeFrame(tf_minutes, TimeFrame.Minute.unit)
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=cursor,
                    end=chunk_end,
                    adjustment="all",
                    feed=self._feed,
                    limit=None,
                )
                bars = self._client.get_stock_bars(request)
                df = bars.df
                if not df.empty:
                    # Flatten MultiIndex if present
                    if isinstance(df.index, pd.MultiIndex):
                        df = df.reset_index(level=0, drop=True)
                    frames.append(df)
            except Exception as exc:
                logger.error("API error for %s (%s->%s): %s", symbol, cursor.date(), chunk_end.date(), exc)

            cursor = chunk_end

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames)
        combined.index.name = "timestamp"
        return combined

    @staticmethod
    def _cache_key(
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tf: int,
        extended: bool,
    ) -> str:
        """Generate a deterministic cache filename."""
        raw = f"{symbol}_{start.date()}_{end.date()}_{tf}m_ext{extended}"
        h = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"{symbol}_{tf}m_{h}"
