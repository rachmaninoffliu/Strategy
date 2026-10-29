"""Local historical data (Kalshi KXBTC15M 1-minute bid/ask candles + outcomes) -> strategy signals."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .strategy import StrategyParams

BASE = Path(__file__).resolve().parent.parent          # the project root (folder containing collect.py, data/, trader/)
DATA = BASE / "data"
WINDOW = 900
SETTLE_DELAY = 20.0            # Kalshi posts a window's result ~7 s after the close; the live engine looks for it this long after (and the backtest's day-start rule mirrors it)


def load_rows(data_dir: Path = DATA) -> pd.DataFrame:
    """One row per (market, decision minute): ticker, open_ts, minute, bid, ask, yes. Candle CLOSE quote = quote at open+60*minute."""
    m = pd.read_csv(data_dir / "markets.csv", usecols=["ticker", "open_ts", "close_ts", "result"])
    m["yes"] = (m.result == "yes").astype(int)
    c = pd.read_csv(data_dir / "candles.csv", usecols=["ticker", "minute", "bid_c", "ask_c", "volume"], on_bad_lines="skip")
    c = c[(c.minute >= 1) & (c.minute <= 14)]
    d = c.merge(m[["ticker", "open_ts", "yes"]], on="ticker", how="inner")
    d = d.rename(columns={"bid_c": "bid", "ask_c": "ask", "volume": "vol"})           # vol = contracts traded in that one-minute candle (NaN before Jun 2026: the historical partition has none)
    d["open_ts"] = d.open_ts.astype("int64")
    return d.sort_values(["open_ts", "minute"]).reset_index(drop=True)


SIG_COLS = ["ticker", "open_ts", "close_ts", "minute", "hour", "leg", "side", "cost", "bid", "ask", "yes", "win", "vol"]


def extract_signals(rows: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """All qualifying signals under ``p``: one per (market, leg) = that leg's first qualifying decision minute, in time order.
    Columns: ticker, open_ts, close_ts, minute, hour, leg, side, cost, bid, ask, yes, win."""
    if rows.empty:
        return pd.DataFrame(columns=SIG_COLS)
    d = rows[rows.minute.isin(p.minutes)].copy()
    if "vol" not in d:
        d["vol"] = np.nan
    d["hour"] = pd.to_datetime(d.open_ts, unit="s", utc=True).dt.hour
    d = d[d.hour.isin(p.hours)]
    d = d[(d.bid > 0) & (d.ask < 1) & (d.ask >= d.bid) & ((d.ask - d.bid) <= p.max_spread + 1e-9)]
    mid = (d.bid + d.ask) / 2
    yes_dog = mid < 0.5
    out = []
    if "DOG" in p.legs:
        x = d.copy()
        x["leg"] = "DOG"
        x["side"] = np.where(yes_dog, "YES", "NO")
        x["cost"] = np.where(yes_dog, x.ask, 1.0 - x.bid).round(6)
        out.append(x[(x.cost >= p.cost_min - 1e-9) & (x.cost <= p.cost_max + 1e-9)])
    if "FAV" in p.legs:
        x = d.copy()
        x["leg"] = "FAV"
        x["side"] = np.where(yes_dog, "NO", "YES")
        x["cost"] = np.where(yes_dog, 1.0 - x.bid, x.ask).round(6)
        out.append(x[(x.cost >= p.fav_min - 1e-9) & (x.cost <= p.fav_max + 1e-9)])
    if not out:
        return pd.DataFrame(columns=SIG_COLS)
    s = pd.concat(out)
    if s.empty:
        return pd.DataFrame(columns=SIG_COLS)
    s["win"] = np.where(s.side == "YES", s.yes, 1 - s.yes)
    s = s.sort_values(["ticker", "leg", "minute"]).drop_duplicates(["ticker", "leg"])         # the leg's FIRST qualifying minute, row-wise (a NaN volume must not borrow a later minute's)
    s["close_ts"] = s.open_ts + WINDOW
    return s.sort_values(["open_ts", "minute", "leg"]).reset_index(drop=True)[SIG_COLS]


def data_range(rows: pd.DataFrame) -> tuple[Optional[datetime], Optional[datetime]]:
    if rows.empty:
        return None, None
    return (datetime.fromtimestamp(int(rows.open_ts.min()), timezone.utc),
            datetime.fromtimestamp(int(rows.open_ts.max()) + WINDOW, timezone.utc))


def coverage(rows: pd.DataFrame, p: StrategyParams, start_ts: int, end_ts: int) -> tuple[int, int]:
    """(markets in range that have candles, markets in range per the markets file) restricted to the strategy hours."""
    m = pd.read_csv(DATA / "markets.csv", usecols=["ticker", "open_ts"])
    m = m[(m.open_ts >= start_ts) & (m.open_ts < end_ts)]
    m = m[pd.to_datetime(m.open_ts, unit="s", utc=True).dt.hour.isin(p.hours)]
    have = set(rows.ticker.unique()) if "ticker" in rows else set()
    return int(m.ticker.isin(have).sum()), int(len(m))


REFRESH_LAG_S = 20 * 60          # the newest window is older than this: a newer one has probably settled (a window closes every 15 min)
REFRESH_GAP_S = 5 * 60           # ... but do not hit the API again within this many seconds of the last attempt


def needs_refresh(rows: pd.DataFrame, now: float, last_attempt_ts: float = 0.0) -> bool:
    """Should the local data be refreshed before it is used (backtest, session start)?  Pure policy, no I/O."""
    if rows.empty:
        return True
    lag = now - data_range(rows)[1].timestamp()
    return lag > REFRESH_LAG_S and now - last_attempt_ts > REFRESH_GAP_S


def update_data(log: Callable[[str], None], rps: Optional[float] = None) -> bool:
    """Refresh markets + candles (+ the newest windows' candles via gapfill) via collect.py. Streams output to ``log``.  ``rps`` = requests per second the download may use (default 5): the app passes a low
    value while a live session is running, because the download and the session share Kalshi's public rate limit."""
    env = dict(os.environ, **({"COLLECT_RPS": str(rps)} if rps else {}))
    last = int(pd.read_csv(DATA / "markets.csv", usecols=["open_ts"]).open_ts.max())
    since = str(last - 2 * 86400)
    ok = True
    for args in (["markets"], ["batch", since], ["gapfill", "3"]):
        log(f"$ python collect.py {' '.join(args)}")
        proc = subprocess.Popen([sys.executable, "collect.py", *args], cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)
        ok &= proc.wait() == 0
    return ok


EARLIEST_MARKET = "2025-12-15"          # Kalshi's KXBTC15M series launched on this date: no BTC 15-minute history exists before it


def backfill_plan(hours) -> int:
    """How many markets (in the given UTC hours) have no candle data yet, i.e. what 'Download older history' would fetch."""
    m = pd.read_csv(DATA / "markets.csv", usecols=["ticker", "open_ts"])
    m = m[pd.to_datetime(m.open_ts, unit="s", utc=True).dt.hour.isin(list(hours))]
    have = set(pd.read_csv(DATA / "candles.csv", usecols=["ticker"], on_bad_lines="skip").ticker.unique())
    return int((~m.ticker.isin(have)).sum())


def backfill_history(hours, log: Callable[[str], None], rps: Optional[float] = None) -> bool:
    """Download 1-minute candles for every market in ``hours`` back to the series launch (public API, ~1.5 markets/s)."""
    end = (datetime.now(timezone.utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [sys.executable, "collect_hours.py", EARLIEST_MARKET, end, ",".join(str(h) for h in sorted(hours))]
    log("$ python " + " ".join(cmd[1:]))
    proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=dict(os.environ, **({"COLLECT_RPS": str(rps)} if rps else {})))
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
    return proc.wait() == 0
