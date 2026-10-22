"""Download the Kalshi trade tape for a sample of settled 15-min markets (read-only public API).

Usage: python tape.py SERIES N [since=YYYY-MM-DD]   -> data/tape/<SERIES>.csv  (resumable)
Columns: ticker, ts_ms, taker_side, yes_price, count
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from collect import get

DATA = Path(__file__).parent / "data"
OUT = DATA / "tape"
OUT.mkdir(exist_ok=True)


def market_list(series: str) -> pd.DataFrame:
    p = DATA / "markets.csv" if series == "KXBTC15M" else DATA / "multi" / f"{series}_markets.csv"
    return pd.read_csv(p)


def fetch_tape(ticker: str, open_ts: int, close_ts: int, max_pages: int = 80):
    out, cur = [], None
    for _ in range(max_pages):
        d = get("/markets/trades", {"ticker": ticker, "limit": 1000, "cursor": cur,
                                    "min_ts": open_ts, "max_ts": close_ts})
        if not d or not d.get("trades"):
            break
        for t in d["trades"]:
            ms = int(datetime.fromisoformat(t["created_time"].replace("Z", "+00:00")).timestamp() * 1000)
            out.append((ticker, ms, t["taker_side"], float(t["yes_price_dollars"]), float(t["count_fp"])))
        cur = d.get("cursor")
        if not cur:
            break
    return out


def main(series: str, n: int, since: str = "2026-09-10") -> None:
    m = market_list(series)
    m = m[m.open_ts >= int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp())].sort_values("open_ts")
    idx = np.linspace(0, len(m) - 1, min(n, len(m))).round().astype(int)
    m = m.iloc[idx]
    path = OUT / f"{series}.csv"
    done = set()
    if path.exists():
        done = set(pd.read_csv(path, usecols=["ticker"], on_bad_lines="skip").ticker.unique())
    todo = m[~m.ticker.isin(done)]
    print(f"{series}: {len(done)} done, {len(todo)} to fetch", flush=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["ticker", "ts_ms", "taker_side", "yes_price", "count"])
        for i, r in enumerate(todo.itertuples()):
            w.writerows(fetch_tape(r.ticker, int(r.open_ts), int(r.close_ts)))
            if i % 25 == 0:
                fh.flush()
                print(f"  {series} {i}/{len(todo)}", flush=True)
    print(f"{series}: TAPE_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "2026-09-10")
