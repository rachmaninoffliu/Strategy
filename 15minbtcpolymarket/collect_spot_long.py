"""Resumable backfill of Coinbase BTC-USD 1-minute candles (public API, same source as cb_1m.csv).

    python collect_spot_long.py 2024-01-01 [end_date] [out_file]   # default end = first bar of data/cb_1m.csv

Appends to data/cb_1m_long.csv (ts, low, high, open, close, volume); re-running continues from the last stored bar.
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
OUT = DATA / "cb_1m_long.csv"
STEP = 300 * 60


def first_bar_of_main() -> int:
    with open(DATA / "cb_1m.csv") as fh:
        next(fh)
        return int(float(next(fh).split(",")[0]))


def main(start_date: str, end_date: str | None = None, out: str = "cb_1m_long.csv") -> None:
    global OUT
    OUT = DATA / out
    t0 = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
    end = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp()) if end_date else first_bar_of_main() + 3600
    last = t0
    if OUT.exists():
        with open(OUT) as fh:
            for row in csv.reader(fh):
                if row and row[0] != "ts":
                    last = max(last, int(float(row[0])))
        last = max(t0, last - 3600)
    new_file = not OUT.exists()
    t, n_req, empty = last, 0, 0
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["ts", "low", "high", "open", "close", "volume"])
        while t < end:
            t2 = min(t + STEP, end)
            ok = False
            for i in range(6):
                try:
                    r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                                     params={"granularity": 60, "start": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                                             "end": datetime.fromtimestamp(t2, timezone.utc).isoformat()}, timeout=25)
                except requests.RequestException:
                    time.sleep(1 + i)
                    continue
                if r.status_code == 200:
                    rows = sorted(r.json(), key=lambda c: c[0])
                    w.writerows(rows)
                    empty += (len(rows) == 0)
                    ok = True
                    break
                time.sleep(1 + 2 * i)
            if not ok:
                print("giving up at", datetime.fromtimestamp(t, timezone.utc), flush=True)
                break
            t = t2
            n_req += 1
            if n_req % 100 == 0:
                fh.flush()
                print(f"{datetime.fromtimestamp(t, timezone.utc):%Y-%m-%d %H:%M} ({empty} empty chunks)", flush=True)
            time.sleep(0.11)
    print("done", datetime.fromtimestamp(t, timezone.utc), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2024-01-01", sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else "cb_1m_long.csv")
