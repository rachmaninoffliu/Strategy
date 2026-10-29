"""
Data layer for the leveraged-ETF pairs strategy: TQQQ / QQQ, SOXL / SMH, SPXL / SPY.

* 1-minute regular-hours bars from Alpaca (SIP, split-adjusted, 2016-01 on), cached per symbol in data/bars1_<SYM>.parquet; every other time frame is resampled from them
  (data/bars<tf>_<SYM>.parquet). The IEX feed is cached the same way (…_iex.parquet): it is the only real-time feed this account may read, so the live engine and the fidelity test use it.
* ``load_pairdata(pair, tf)`` gives day x bar matrices for both legs on a complete session grid (a bar with no trade is carried forward and flagged).
* quoted spreads sampled from SIP quotes (the cost side of the study), see ``sample_spreads``.
Credentials come from the project's key file "Alpaca Key.txt" in the project folder, like every other app in this project (alpaca_keys.py); data requests are read-only.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
NY = "America/New_York"
START = "2016-01-04"
SYMBOLS = ["TQQQ", "QQQ", "SOXL", "SMH", "SPXL", "SPY"]
PAIRS = [("TQQQ", "QQQ"), ("SOXL", "SMH"), ("SPXL", "SPY")]            # (leveraged 3x ETF, 1x ETF)
NOMINAL_BETA = 3.0
TIMEFRAMES = (1, 5, 15, 30)
OPEN_MIN, CLOSE_MIN = 9 * 60 + 30, 16 * 60                              # regular session, minutes after midnight ET
COLS = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]


def pair_name(pair: tuple) -> str:
    return f"{pair[0]}/{pair[1]}"


# ── credentials / clients ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def creds(paper: bool = True) -> tuple[str, str]:
    """Alpaca keys: the project's key file "Alpaca Key.txt" in the project folder, then ALPACA_API_KEY / ALPACA_SECRET_KEY, then ~/.claude.json (alpaca_keys.py)."""
    import alpaca_keys
    return alpaca_keys.creds(paper)


def own_keys() -> bool:
    """True when the Alpaca keys are the project's own - the key file or the ALPACA_API_KEY / ALPACA_SECRET_KEY variables - and not the ~/.claude.json fallback of the Claude tools."""
    import alpaca_keys
    return alpaca_keys.source(True) in ("file", "env")


def harden(client, connect: float = 5.0, read: float = 20.0):
    """alpaca-py makes its HTTP calls without a timeout: one stalled connection would freeze the engine (or a flatten) for ever. Give the client's session a default (connect, read) timeout."""
    import requests
    from requests.adapters import HTTPAdapter

    class TimeoutAdapter(HTTPAdapter):
        def send(self, request, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (connect, read)
            return super().send(request, **kwargs)
    sess = getattr(client, "_session", None)
    if isinstance(sess, requests.Session):
        sess.mount("https://", TimeoutAdapter())
        sess.mount("http://", TimeoutAdapter())
    return client


def data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    k, s = creds(True)
    return harden(StockHistoricalDataClient(k, s))


# ── download ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def fetch_1m(sym: str, feed: str = "sip", start=None, end=None, client=None, log: Callable[[str], None] = print) -> pd.DataFrame:
    """1-minute regular-hours bars, quarter by quarter. Index = bar start in ET (naive). ``end`` defaults to 16 minutes ago (the SIP delay rule)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    c = client or data_client()
    utc = lambda x: pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x).tz_convert("UTC")
    end = utc(end) if end is not None else pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
    t0 = utc(start or START)
    frames = []
    while t0 < end:
        t1 = min(t0 + pd.Timedelta(days=92), end)
        for attempt in range(6):
            try:
                df = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, start=t0.to_pydatetime(), end=t1.to_pydatetime(), adjustment=Adjustment.SPLIT,
                                                       feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP)).df
                if len(df):
                    frames.append(df.reset_index(level=0, drop=True))
                break
            except KeyError:
                break
            except Exception as e:                                                      # noqa: BLE001
                time.sleep(2 * (attempt + 1))
                if attempt == 5:
                    log(f"  {sym} {t0:%Y-%m}: failed ({type(e).__name__}: {str(e)[:80]})")
        t0 = t1
    if not frames:
        return pd.DataFrame(columns=COLS)
    b = pd.concat(frames)
    b = b[~b.index.duplicated()].sort_index()
    et = b.index.tz_convert(NY)
    mod = et.hour * 60 + et.minute
    keep = (mod >= OPEN_MIN) & (mod < CLOSE_MIN)
    b = b[keep]
    b.index = et[keep].tz_localize(None)
    return b[COLS].astype("float64")


def path1(sym: str, feed: str = "sip") -> str:
    return os.path.join(DATA, f"bars1_{sym}.parquet" if feed == "sip" else f"bars1_{sym}_{feed}.parquet")


def pathtf(sym: str, tf: int, feed: str = "sip") -> str:
    return os.path.join(DATA, f"bars{tf}_{sym}.parquet" if feed == "sip" else f"bars{tf}_{sym}_{feed}.parquet")


def download_all(symbols=None, feed: str = "sip", log: Callable[[str], None] = print) -> None:
    os.makedirs(DATA, exist_ok=True)
    todo = [s for s in (symbols or SYMBOLS) if not os.path.exists(path1(s, feed))]
    log(f"downloading {len(todo)} symbols ({feed}) ...")
    c = data_client()

    def one(s):
        t = time.time()
        b = fetch_1m(s, feed, client=c, start=START if feed == "sip" else "2020-07-27", log=log)
        if len(b):
            b.to_parquet(path1(s, feed))
        return s, len(b), round(time.time() - t)
    with ThreadPoolExecutor(3) as ex:
        for s, n, sec in ex.map(one, todo):
            log(f"  {s}: {n:,} 1-minute bars in {sec}s")


def update_cache(symbols=None, feeds=("sip",), log: Callable[[str], None] = print) -> None:
    """Top the cached 1-minute bars up to 16 minutes ago (read-only requests) and drop the stale resampled files."""
    c = data_client()
    for feed in feeds:
        for s in symbols or SYMBOLS:
            f = path1(s, feed)
            old = pd.read_parquet(f) if os.path.exists(f) else None
            start = (old.index.max().normalize() + pd.Timedelta(days=1)) if old is not None and len(old) else pd.Timestamp(START)
            add = fetch_1m(s, feed, start=start, client=c, log=log)
            if len(add):
                new = pd.concat([old, add]) if old is not None else add
                new = new[~new.index.duplicated()].sort_index()
                new.to_parquet(f)
                for tf in TIMEFRAMES:
                    if os.path.exists(pathtf(s, tf, feed)):
                        os.remove(pathtf(s, tf, feed))
            log(f"{s} {feed}: +{len(add):,} bars" + (f" through {add.index.max():%Y-%m-%d %H:%M}" if len(add) else ""))


# ── time frames ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def resample(b1: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Bars of ``tf`` minutes aligned to 09:30 (bar start index). Volume-weighted VWAP."""
    if tf == 1:
        return b1
    idx = b1.index
    day = idx.normalize()
    mins = ((idx - day) / pd.Timedelta(minutes=1)).to_numpy() - OPEN_MIN
    key = day + pd.to_timedelta(OPEN_MIN + (mins // tf) * tf, unit="m")
    d = b1.assign(_pv=b1.vwap * b1.volume)
    g = d.groupby(key)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), _pv=("_pv", "sum"), trade_count=("trade_count", "sum"))
    out["vwap"] = (out._pv / out.volume.replace(0, np.nan)).fillna(out.close)
    out.index.name = None
    return out[COLS]


def load_bars(sym: str, tf: int = 5, feed: str = "sip") -> pd.DataFrame:
    f = pathtf(sym, tf, feed)
    if tf == 1:
        return pd.read_parquet(path1(sym, feed))
    if os.path.exists(f) and os.path.getmtime(f) >= os.path.getmtime(path1(sym, feed)):
        return pd.read_parquet(f)
    r = resample(pd.read_parquet(path1(sym, feed)), tf)
    r.to_parquet(f)
    return r


# ── pair panels ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def session_grid(tf: int) -> np.ndarray:
    """Bar start offsets in minutes from 09:30 for a full session."""
    return np.arange(0, CLOSE_MIN - OPEN_MIN, tf)


@dataclass
class PairData:
    pair: tuple
    tf: int
    feed: str
    dates: pd.DatetimeIndex
    nb: int                                   # bars per session
    L_open: np.ndarray                        # (days, bars)
    L_close: np.ndarray
    B_open: np.ndarray
    B_close: np.ndarray
    L_vol: np.ndarray
    B_vol: np.ndarray
    L_miss: np.ndarray                        # bool: no trade in the bar, price carried forward
    B_miss: np.ndarray
    notes: list = field(default_factory=list)

    def day_index(self, date) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(date).normalize()))

    def last_date(self) -> pd.Timestamp:
        return self.dates[-1]


def _matrix(b: pd.DataFrame, days: pd.DatetimeIndex, nb: int, tf: int):
    """Day x bar matrices on the full session grid. Missing bars: close carried forward from the previous bar (first bar: previous close), volume 0."""
    grid = session_grid(tf)
    idx = (days.values[:, None] + (OPEN_MIN + grid)[None, :].astype("timedelta64[m]")).ravel()
    a = b.reindex(pd.DatetimeIndex(idx))
    miss = a.close.isna().to_numpy().reshape(len(days), nb)
    close = a.close.ffill()
    # the series is chronological across days, so the first bar of a day inherits the previous day's last close if it has no trade
    opn = a.open.where(a.open.notna(), close)
    return (opn.to_numpy().reshape(len(days), nb), close.to_numpy().reshape(len(days), nb), a.volume.fillna(0.0).to_numpy().reshape(len(days), nb), miss)


def pairdata_from_bars(pair: tuple, L: pd.DataFrame, B: pd.DataFrame, tf: int, feed: str = "sip", start=None, end=None, max_missing: float = 0.5, partial_day=None) -> PairData:
    """Both legs on one session grid from bars of ``tf`` minutes. A day is used only if it is a full session for both legs (early-close days drop out) and neither leg is missing more than
    ``max_missing`` of its bars (IEX bars of a thin ETF can be); smaller gaps are carried forward and flagged in ``*_miss``. ``partial_day`` (live) is a session that has not finished:
    it is kept with whatever bars exist, the rest of its grid is carried forward."""
    nb = len(session_grid(tf))
    last_start = pd.Timedelta(minutes=OPEN_MIN + session_grid(tf)[-1])

    def sessions(b):
        d = b.index.normalize()
        has_last = pd.Series(b.index == (d + last_start), index=b.index).groupby(d).any()
        n = b.groupby(d).size()
        return n.index[(n.to_numpy() >= 0.5 * nb) & has_last.reindex(n.index).to_numpy()]
    days = sessions(L).intersection(sessions(B))
    if partial_day is not None:
        pdy = pd.Timestamp(partial_day).normalize()
        days = days[days < pdy].append(pd.DatetimeIndex([pdy]))
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    lo, lc, lv, lm = _matrix(L, days, nb, tf)
    bo, bc, bv, bm = _matrix(B, days, nb, tf)
    ok = (lm.mean(axis=1) <= max_missing) & (bm.mean(axis=1) <= max_missing) & np.isfinite(lc).all(axis=1) & np.isfinite(bc).all(axis=1)
    if partial_day is not None:
        ok[-1] = True
    notes = [f"{len(days) - int(ok.sum())} sessions dropped (too many missing bars)"] if (~ok).any() else []
    keep = np.where(ok)[0]
    return PairData(pair, tf, feed, pd.DatetimeIndex(days[keep]), nb, lo[keep], lc[keep], bo[keep], bc[keep], lv[keep], bv[keep], lm[keep], bm[keep], notes)


def load_pairdata(pair: tuple, tf: int = 5, feed: str = "sip", start=None, end=None, max_missing: float = 0.5) -> PairData:
    L, B = (load_bars(s, tf, feed) for s in pair)
    return pairdata_from_bars(pair, L, B, tf, feed, start, end, max_missing)


# ── spreads (cost side) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
WINDOWS = [("09:45", "09:46"), ("10:30", "10:31"), ("12:00", "12:01"), ("13:30", "13:31"), ("15:00", "15:01"), ("15:45", "15:46")]


def recent_sessions(n: int = 3) -> list[str]:
    b = pd.read_parquet(path1("SPY"))
    return [str(d.date()) for d in b.index.normalize().unique()[-n - 1:-1]]               # complete sessions only


def sample_spreads(days: Optional[list] = None, symbols=None, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Median relative quoted spread (bp) from SIP quotes in 1-minute windows through the session on a few recent days -> reports/spreads_levpairs.csv (symbol x window)."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest
    days = days or recent_sessions(3)
    c = data_client()

    def one(sym):
        rows = []
        for d in days:
            for a, b in WINDOWS:
                t0, t1 = pd.Timestamp(f"{d} {a}", tz=NY).tz_convert("UTC"), pd.Timestamp(f"{d} {b}", tz=NY).tz_convert("UTC")
                q = None
                for attempt in range(4):
                    try:
                        q = c.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym, start=t0.to_pydatetime(), end=t1.to_pydatetime(), feed=DataFeed.SIP, limit=100000)).df
                        break
                    except Exception:                                                   # noqa: BLE001
                        time.sleep(2)
                if q is None or not len(q):
                    continue
                q = q.reset_index(level=0, drop=True)
                q = q[(q.ask_price > 0) & (q.bid_price > 0) & (q.ask_price > q.bid_price)]
                if len(q):
                    mid = (q.ask_price + q.bid_price) / 2
                    rows.append(dict(symbol=sym, day=d, window=a, n=len(q), spread_bp=float(((q.ask_price - q.bid_price) / mid * 1e4).median()), px=float(mid.median())))
        return rows
    out = []
    with ThreadPoolExecutor(3) as ex:
        for r in ex.map(one, symbols or SYMBOLS):
            out += r
    df = pd.DataFrame(out)
    os.makedirs(REPORTS, exist_ok=True)
    df.to_csv(os.path.join(REPORTS, "spreads_levpairs_raw.csv"), index=False)
    s = df.groupby(["symbol", "window"]).spread_bp.median().unstack("window")
    s["all"] = df.groupby("symbol").spread_bp.median()
    s.to_csv(os.path.join(REPORTS, "spreads_levpairs.csv"))
    log(s.round(2).to_string())
    return s


def load_spreads() -> pd.DataFrame:
    """symbol -> median full quoted spread in bp over all windows (fallbacks if the sample has not been taken yet)."""
    f = os.path.join(REPORTS, "spreads_levpairs.csv")
    if os.path.exists(f):
        return pd.read_csv(f).set_index("symbol")
    return pd.DataFrame({"all": {"TQQQ": 1.4, "QQQ": 0.3, "SOXL": 2.6, "SMH": 2.3, "SPXL": 2.0, "SPY": 0.3}})


def half_spread_bp(sym: str, tf: int = 5) -> float:
    s = load_spreads()
    return float(s.loc[sym, "all"]) / 2.0 if sym in s.index else 1.0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "iex":
        download_all(feed="iex")
    elif len(sys.argv) > 1 and sys.argv[1] == "spreads":
        sample_spreads()
    else:
        download_all()
