"""Collect Kalshi KXBTC15M history (read-only public API) + BTC-USD spot proxies.

Outputs (under ./data):
  markets.csv   one row per settled 15-min market (strike = BRTI 60s avg at open, expiration_value = BRTI 60s avg at close)
  candles.csv   1-minute yes bid/ask/price candles per market (minute index 1..15 since open)
  cb_1m.csv     Coinbase BTC-USD 1-minute candles covering the market window (spot proxy, BRTI constituent)

Usage:  python collect.py markets | markets_full | batch [since_ts] | gapfill [days] | candles | spot | all
"""
from __future__ import annotations

import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

B = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTC15M"
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

_lock = threading.Lock()
_last = [0.0]
_pause_until = [0.0]
RPS = 5.0          # unauthenticated Kalshi read limit bites above ~6 rps (HTTP 429); stay under it


def _throttle() -> None:
    with _lock:
        now = time.time()
        wait = max(_last[0] + 1.0 / RPS, _pause_until[0]) - now
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def get(path: str, params: dict | None = None, retries: int = 8):
    for i in range(retries):
        _throttle()
        try:
            r = requests.get(B + path, params=params, timeout=25)
        except requests.RequestException:
            time.sleep(1 + i)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (400, 404):
            return None
        if r.status_code == 429:
            _pause_until[0] = time.time() + 2.0 + i     # shared pause: every thread backs off together
            continue
        time.sleep(1 + i)
    return None


def ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return ""


MARKET_FIELDS = ["ticker", "open_ts", "close_ts", "strike", "expiration_value", "result", "volume", "open_interest"]


def collect_markets(full: bool = False) -> None:
    """Merge Kalshi's settled markets into data/markets.csv.  Incremental by default: the live partition (newest ~6k markets) is re-read, the historical partition only when the file is empty or ``full``.
    Existing rows are never dropped, so a network hiccup half-way can no longer truncate the file."""
    rows: dict[str, dict] = {}
    path_csv = DATA / "markets.csv"
    if path_csv.exists():
        with open(path_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["ticker"]] = r
    have_history = len(rows) > 1000

    def add(m: dict) -> None:
        if m.get("result") not in ("yes", "no") or not m.get("floor_strike"):
            return
        rows[m["ticker"]] = dict(ticker=m["ticker"], open_ts=ts(m["open_time"]), close_ts=ts(m["close_time"]),
                                 strike=f(m["floor_strike"]), expiration_value=f(m.get("expiration_value")),
                                 result=m["result"], volume=f(m.get("volume_fp")),
                                 open_interest=f(m.get("open_interest_fp")))

    for path, extra in (("/markets", {"status": "settled"}), ("/historical/markets", {})):
        if path == "/historical/markets" and have_history and not full:
            continue
        cur = None
        for page in range(200):
            d = get(path, {"series_ticker": SERIES, "limit": 1000, "cursor": cur, **extra})
            if not d or not d.get("markets"):
                break
            for m in d["markets"]:
                add(m)
            cur = d.get("cursor")
            print(f"{path} page {page}: total {len(rows)}", flush=True)
            if not cur:
                break
    out = sorted(rows.values(), key=lambda r: int(r["open_ts"]))
    tmp = DATA / "markets.csv.tmp"
    with open(tmp, "w", newline="") as fh:                              # write-then-rename: readers never see a half-written file
        w = csv.DictWriter(fh, MARKET_FIELDS)
        w.writeheader()
        w.writerows(out)
    tmp.replace(path_csv)
    print("markets:", len(out), datetime.fromtimestamp(int(out[0]["open_ts"]), timezone.utc), "->",
          datetime.fromtimestamp(int(out[-1]["close_ts"]), timezone.utc))


CANDLE_FIELDS = ["ticker", "minute", "end_ts", "bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c",
                 "px_o", "px_h", "px_l", "px_c", "volume"]


def _blk(c: dict, key: str, k: str):
    b = c.get(key) or {}
    return f(b.get(k + "_dollars", b.get(k)))


def _rows(t: str, o: int, cds: list) -> list:
    out = []
    for cd in cds:
        e = int(cd["end_period_ts"])
        out.append([t, round((e - o) / 60), e,
                    _blk(cd, "yes_bid", "open"), _blk(cd, "yes_bid", "high"), _blk(cd, "yes_bid", "low"), _blk(cd, "yes_bid", "close"),
                    _blk(cd, "yes_ask", "open"), _blk(cd, "yes_ask", "high"), _blk(cd, "yes_ask", "low"), _blk(cd, "yes_ask", "close"),
                    _blk(cd, "price", "open"), _blk(cd, "price", "high"), _blk(cd, "price", "low"), _blk(cd, "price", "close"),
                    f(cd.get("volume_fp"))])
    return out


def fetch_candles(row: dict):
    t = row["ticker"]
    o, c = int(row["open_ts"]), int(row["close_ts"])
    for route in (f"/series/{SERIES}/markets/{t}/candlesticks", f"/historical/markets/{t}/candlesticks"):
        d = get(route, {"start_ts": o, "end_ts": c, "period_interval": 1})
        if d and d.get("candlesticks"):
            return _rows(t, o, d["candlesticks"])
    return None


def fetch_batch(chunk: list[dict]):
    """GET /markets/candlesticks: <=10,000 candles = markets x span(min), so ~24 contiguous 15-min markets per call.
    Serves the live partition only (settled after /historical/cutoff); older chunks come back empty."""
    s = min(int(r["open_ts"]) for r in chunk)
    e = max(int(r["close_ts"]) for r in chunk)
    d = get("/markets/candlesticks", {"market_tickers": ",".join(r["ticker"] for r in chunk),
                                      "start_ts": s, "end_ts": e, "period_interval": 1})
    opens = {r["ticker"]: int(r["open_ts"]) for r in chunk}
    out, got = [], set()
    for z in (d or {}).get("markets", []):
        t = z["market_ticker"]
        if z.get("candlesticks"):
            out.extend(_rows(t, opens[t], z["candlesticks"]))
            got.add(t)
    return out, [r for r in chunk if r["ticker"] not in got]


def collect_candles_batch(since_ts: int = 0, chunk_n: int = 24) -> None:
    """Fast path for the live partition. Writes data/candles.csv (append, resumable)."""
    with open(DATA / "markets.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["open_ts"]) >= since_ts]
    path = DATA / "candles.csv"
    done: set[str] = set()
    if path.exists():
        with open(path) as fh:
            done = {r["ticker"] for r in csv.DictReader(fh)}
    todo = sorted((r for r in rows if r["ticker"] not in done), key=lambda r: -int(r["open_ts"]))
    chunks = [todo[i:i + chunk_n] for i in range(0, len(todo), chunk_n)]
    print(f"batch candles: {len(done)} done, {len(todo)} todo in {len(chunks)} chunks", flush=True)
    new = not path.exists()
    leftover: list[dict] = []
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(CANDLE_FIELDS)
        for i, ch in enumerate(chunks):
            ch = sorted(ch, key=lambda r: int(r["open_ts"]))
            out, miss = fetch_batch(ch)
            w.writerows(out)
            leftover.extend(miss)
            if i % 10 == 0:
                fh.flush()
                print(f"  chunk {i + 1}/{len(chunks)}  rows {len(out)}  leftover {len(leftover)}", flush=True)
    with open(DATA / "leftover.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, MARKET_FIELDS)
        w.writeheader()
        w.writerows(leftover)
    print("batch done; leftover (historical partition / empty):", len(leftover))


def collect_gapfill(days: float = 3.0) -> None:
    """Candles for markets that settled within the last ``days`` days and are still missing.  The batch route serves the newest windows up to ~2 h late; the per-market route has a window's
    candles within a minute or two of its close, so this is what makes the newest evening show up.  A market is only stored once its late minutes (>= 13) exist."""
    now = time.time()
    with open(DATA / "markets.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["close_ts"]) <= now - 90 and int(r["open_ts"]) >= now - days * 86400]
    path = DATA / "candles.csv"
    done: set[str] = set()
    if path.exists():
        with open(path) as fh:
            done = {r["ticker"] for r in csv.DictReader(fh)}
    todo = sorted((r for r in rows if r["ticker"] not in done), key=lambda r: -int(r["open_ts"]))
    print(f"gapfill: {len(todo)} recently settled markets without candles", flush=True)
    if not todo:
        return
    new = not path.exists()
    got = skipped = 0
    with open(path, "a", newline="") as fh, ThreadPoolExecutor(3) as ex:
        w = csv.writer(fh)
        if new:
            w.writerow(CANDLE_FIELDS)
        for row, res in zip(todo, ex.map(fetch_candles, todo)):
            if res and max(int(x[1]) for x in res) >= 13:
                w.writerows(res)
                got += 1
            else:
                skipped += 1
    print(f"gapfill done: {got} markets added, {skipped} not available yet", flush=True)


def collect_candles() -> None:
    with open(DATA / "markets.csv") as fh:
        rows = list(csv.DictReader(fh))
    path = DATA / "candles.csv"
    done: set[str] = set()
    if path.exists():
        with open(path) as fh:
            done = {r["ticker"] for r in csv.DictReader(fh)}
    todo = [r for r in rows if r["ticker"] not in done]
    todo.sort(key=lambda r: -int(r["open_ts"]))          # newest first: the liquid regime is what matters
    print(f"candles: {len(done)} markets done, {len(todo)} to fetch", flush=True)
    new = not path.exists()
    n = miss = 0
    with open(path, "a", newline="") as fh, ThreadPoolExecutor(4) as ex:
        w = csv.writer(fh)
        if new:
            w.writerow(CANDLE_FIELDS)
        for res in ex.map(fetch_candles, todo):
            n += 1
            if res is None:
                miss += 1
            else:
                w.writerows(res)
            if n % 100 == 0:
                fh.flush()
                print(f"  {n}/{len(todo)} fetched, {miss} empty", flush=True)
    print("candles done; empty:", miss)


def collect_spot() -> None:
    """Coinbase BTC-USD 1-minute candles; incremental (appends from the last stored bar)."""
    with open(DATA / "markets.csv") as fh:
        rows = list(csv.DictReader(fh))
    path = DATA / "cb_1m.csv"
    got: dict[int, list] = {}
    if path.exists():
        with open(path) as fh:
            for r in csv.DictReader(fh):
                got[int(float(r["ts"]))] = [float(r["ts"]), r["low"], r["high"], r["open"], r["close"], r["volume"]]
    start = (max(got) - 3600) if got else int(rows[0]["open_ts"]) - 3600
    end = int(time.time())
    t = start
    step = 300 * 60
    while t < end:
        t2 = min(t + step, end)
        for i in range(5):
            r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                             params={"granularity": 60, "start": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                                     "end": datetime.fromtimestamp(t2, timezone.utc).isoformat()}, timeout=25)
            if r.status_code == 200:
                for c in r.json():
                    got[int(c[0])] = c      # [time, low, high, open, close, volume]
                break
            time.sleep(1 + i)
        time.sleep(0.12)
        t = t2
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "low", "high", "open", "close", "volume"])
        for k in sorted(got):
            w.writerow(got[k])
    print("spot rows:", len(got))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("markets", "all", "markets_full"):
        collect_markets(full=what in ("all", "markets_full"))
    if what == "gapfill":
        collect_gapfill(days=float(sys.argv[2]) if len(sys.argv) > 2 else 3.0)
    if what == "batch":
        collect_candles_batch(since_ts=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    if what in ("candles", "all"):
        collect_candles()
    if what in ("spot", "all"):
        collect_spot()
