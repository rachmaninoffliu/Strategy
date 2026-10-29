"""
Data layer for the intraday session-convergence study (Demir et al. 2022 translated to US cash equities).

* 5-minute regular-hours bars from Alpaca (SIP feed, split-adjusted) for the candidate products, cached per symbol in data/bars5_<SYM>.parquet
* quoted spreads sampled from Alpaca historical quotes (the cost side of the study)
Credentials come from the project's key file "Alpaca Key.txt" in the project folder, like every app in this project (alpaca_keys.py; paper keys; data requests are read-only).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
START = "2016-01-04"
NY = "America/New_York"

CANDIDATES = {
    "index ETF": ["SPY", "QQQ", "IWM", "DIA"],
    "sector / industry ETF": ["XLF", "XLE", "XLK", "SMH", "XBI", "GLD", "TLT"],
    "leveraged ETF": ["TQQQ", "SQQQ", "SOXL", "UPRO"],
    "mega-cap stock": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "AMD"],
}
ALL = [s for v in CANDIDATES.values() for s in v]


def client():
    from alpaca.data.historical import StockHistoricalDataClient

    import alpaca_keys
    return StockHistoricalDataClient(*alpaca_keys.creds(True))


def fetch_bars(sym: str, tf_min: int = 5, log=print, feed: str = "sip", start: str | pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """5-minute regular-hours bars. feed 'sip' = consolidated tape (the research data), 'iex' = the only feed this account may query in real time."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    c = client()
    end = end if end is not None else pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
    frames, t0 = [], pd.Timestamp(start or START, tz="UTC")
    while t0 < end:                                                                    # one year per request keeps each answer small
        t1 = min(t0 + pd.Timedelta(days=366), end)
        for attempt in range(5):
            try:
                df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(tf_min, TimeFrameUnit.Minute), start=t0.to_pydatetime(), end=t1.to_pydatetime(),
                                                       adjustment=Adjustment.SPLIT, feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP)).df
                if len(df):
                    frames.append(df.reset_index(level=0, drop=True))
                break
            except Exception as e:                                                     # noqa: BLE001
                time.sleep(2 * (attempt + 1))
                if attempt == 4:
                    log(f"  {sym} {t0:%Y}: failed ({type(e).__name__}: {str(e)[:80]})")
        t0 = t1
    if not frames:
        return pd.DataFrame()
    b = pd.concat(frames)
    b = b[~b.index.duplicated()].sort_index()
    et = b.index.tz_convert(NY)
    keep = (et.time >= pd.Timestamp("09:30").time()) & (et.time < pd.Timestamp("16:00").time())
    b = b[keep]
    b.index = et[keep].tz_localize(None)
    return b[["open", "high", "low", "close", "volume", "vwap", "trade_count"]]


def bars_path(sym: str, feed: str = "sip") -> str:
    return os.path.join(DATA, f"bars5_{sym}.parquet" if feed == "sip" else f"bars5_{sym}_{feed}.parquet")


def load_bars(sym: str, feed: str = "sip") -> pd.DataFrame:
    return pd.read_parquet(bars_path(sym, feed))


def download_all(symbols=None, log=print):
    os.makedirs(DATA, exist_ok=True)
    todo = [s for s in (symbols or ALL) if not os.path.exists(bars_path(s))]
    log(f"downloading {len(todo)} symbols ...")

    def one(s):
        b = fetch_bars(s, 5, log)
        if len(b):
            b.to_parquet(bars_path(s))
        return s, len(b)
    with ThreadPoolExecutor(4) as ex:
        for s, n in ex.map(one, todo):
            log(f"  {s}: {n:,} bars")


if __name__ == "__main__":
    download_all()


# ── daily official closes: the settlement price (the closing auction print an MOC order would get) ─────────────────────────────────────────────────────────────
def daily_path() -> str:
    return os.path.join(DATA, "daily_closes.parquet")


def download_daily(symbols=None, log=print) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    c = client()
    syms = list(symbols or ALL)
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
    df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day, start=pd.Timestamp(START, tz="UTC").to_pydatetime(), end=end.to_pydatetime(),
                                           adjustment=Adjustment.SPLIT, feed=DataFeed.SIP)).df.reset_index()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    out = df[["symbol", "date", "open", "high", "low", "close", "volume"]]
    os.makedirs(DATA, exist_ok=True)
    out.to_parquet(daily_path())
    log(f"daily bars: {out.symbol.nunique()} symbols, {len(out):,} rows")
    return out


def load_daily() -> pd.DataFrame:
    return pd.read_parquet(daily_path())
