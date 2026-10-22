"""Parallel-safe variant of collect_hours.py: this process handles every N-th missing market and writes to its OWN file (data/candles_part_<k>.csv),
so several processes never interleave lines in candles.csv.  Merge afterwards with:  python collect_hours_part.py merge

    python collect_hours_part.py 2025-12-15 2026-06-01 0,1,2,...,19  K N        # K = 0..N-1
    python collect_hours_part.py merge
"""
import csv
import glob
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from collect import CANDLE_FIELDS, DATA, SERIES, _rows, get


def fetch_hist_first(row: dict):
    """Old markets live in the historical partition: ask it first (1 request instead of 2 per market)."""
    t, o, c = row["ticker"], int(row["open_ts"]), int(row["close_ts"])
    for route in (f"/historical/markets/{t}/candlesticks", f"/series/{SERIES}/markets/{t}/candlesticks"):
        d = get(route, {"start_ts": o, "end_ts": c, "period_interval": 1})
        if d and d.get("candlesticks"):
            return _rows(t, o, d["candlesticks"])
    return None


def merge() -> None:
    parts = sorted(glob.glob(str(DATA / "candles_part_*.csv")))
    have = set()
    n_new = 0
    with open(DATA / "candles.csv", "a", newline="") as out:
        w = csv.writer(out)
        for p in parts:
            with open(p, newline="") as fh:
                for row in csv.reader(fh):
                    if len(row) == len(CANDLE_FIELDS) and row[0] != "ticker":
                        w.writerow(row)
                        n_new += 1
    print(f"merged {n_new} candle rows from {len(parts)} part files")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge()
        sys.exit(0)
    start = int(pd.Timestamp(sys.argv[1], tz="UTC").timestamp())
    end = int(pd.Timestamp(sys.argv[2], tz="UTC").timestamp())
    hours = [int(h) for h in sys.argv[3].split(",")]
    k, n = int(sys.argv[4]), int(sys.argv[5])
    m = pd.read_csv(DATA / "markets.csv")
    m = m[(m.open_ts >= start) & (m.open_ts < end)]
    m = m[pd.to_datetime(m.open_ts, unit="s", utc=True).dt.hour.isin(hours)]
    done = set(pd.read_csv(DATA / "candles.csv", usecols=["ticker"], on_bad_lines="skip").ticker.unique())
    todo = [r for r in m.sort_values("open_ts").to_dict("records") if r["ticker"] not in done]
    if len(sys.argv) > 6:                                  # random sample of whole days (every ~1/frac-th), in random order so any prefix is representative
        import random
        frac = int(sys.argv[6])
        days = sorted({r["open_ts"] // 86400 for r in todo})
        rnd = random.Random(11)
        keep = set(d for d in days if rnd.random() < 1.0 / frac)
        todo = [r for r in todo if r["open_ts"] // 86400 in keep]
        order = {d: i for i, d in enumerate(rnd.sample(sorted(keep), len(keep)))}
        todo.sort(key=lambda r: (order[r["open_ts"] // 86400], r["open_ts"]))
    todo = todo[k::n]
    print(f"part {k}/{n}: {len(todo)} markets to backfill", flush=True)
    cnt = 0
    with open(DATA / f"candles_part_{k}.csv", "a", newline="") as fh, ThreadPoolExecutor(8) as ex:
        w = csv.writer(fh)
        for res in ex.map(fetch_hist_first, todo):
            cnt += 1
            if res:
                w.writerows(res)
            if cnt % 100 == 0:
                fh.flush()
                print(f"  {cnt}/{len(todo)}", flush=True)
    print("PART_DONE", k, flush=True)
