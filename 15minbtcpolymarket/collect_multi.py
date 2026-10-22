"""Collect Kalshi 15-minute crypto up/down series other than BTC (live partition: settled after 2026-08-06).

Outputs data/multi/<SERIES>_markets.csv and <SERIES>_candles.csv (same schema as the BTC files).
Calibration-type analyses need only quotes + outcomes, so no spot data is pulled for these.
Usage: python collect_multi.py [SERIES ...]   (default: all siblings)
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from collect import CANDLE_FIELDS, MARKET_FIELDS, fetch_batch, f, get, ts

SIBLINGS = ["KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M", "KXBNB15M", "KXHYPE15M", "KXADA15M", "KXBCH15M",
            "KXZEC15M", "KXNEAR15M", "KXTON15M"]
OUT = Path(__file__).parent / "data" / "multi"
OUT.mkdir(parents=True, exist_ok=True)
SINCE = int(datetime(2026, 8, 6, tzinfo=timezone.utc).timestamp())


def collect_series(series: str, since_ts: int = SINCE) -> None:
    mpath, cpath = OUT / f"{series}_markets.csv", OUT / f"{series}_candles.csv"
    rows: dict[str, dict] = {}
    cur = None
    for _ in range(60):
        d = get("/markets", {"series_ticker": series, "status": "settled", "limit": 1000, "cursor": cur,
                             "min_close_ts": since_ts})
        if not d or not d.get("markets"):
            break
        for m in d["markets"]:
            if m.get("result") in ("yes", "no") and m.get("floor_strike"):
                rows[m["ticker"]] = dict(ticker=m["ticker"], open_ts=ts(m["open_time"]), close_ts=ts(m["close_time"]),
                                         strike=f(m["floor_strike"]), expiration_value=f(m.get("expiration_value")),
                                         result=m["result"], volume=f(m.get("volume_fp")),
                                         open_interest=f(m.get("open_interest_fp")))
        cur = d.get("cursor")
        if not cur:
            break
    mk = sorted(rows.values(), key=lambda r: r["open_ts"])
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, MARKET_FIELDS)
        w.writeheader()
        w.writerows(mk)
    print(f"{series}: {len(mk)} settled markets", flush=True)
    if not mk:
        return
    chunks = [mk[i:i + 24] for i in range(0, len(mk), 24)]
    n_rows = 0
    with open(cpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CANDLE_FIELDS)
        for i, ch in enumerate(chunks):
            out, _miss = fetch_batch(ch)
            w.writerows(out)
            n_rows += len(out)
            if i % 40 == 0:
                fh.flush()
    print(f"{series}: candle rows {n_rows}", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or SIBLINGS):
        collect_series(s)
    print("MULTI_DONE")
