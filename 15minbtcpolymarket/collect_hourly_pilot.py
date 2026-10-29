"""Pilot data for Kalshi's hourly BTC 'above/below' ladders (KXBTCD): the quote of every near-the-money strike at 60 / 30 / 15 / 5 minutes before expiry, plus the outcome.

    python collect_hourly_pilot.py [days=14]      # -> data/hourly_pilot.csv   (public API, ~2 calls per hourly event, rate limited)

One event per hour (close_time on the hour; the event ticker uses US Eastern time).  Strikes within +-2.5% of the Coinbase spot at T-60 min are kept (they carry the volume).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

B = "https://external-api.kalshi.com/trade-api/v2"
D = Path(__file__).parent / "data"
ET = ZoneInfo("America/New_York")


def get(path: str, **p) -> dict:
    for i in range(6):
        time.sleep(0.28)
        try:
            r = requests.get(B + path, params=p, timeout=30)
        except requests.RequestException:
            time.sleep(1 + i)
            continue
        if r.status_code == 429:
            time.sleep(2 + i * 1.5)
            continue
        return r.json() if r.status_code == 200 else {}
    return {}


def spot() -> pd.Series:
    parts = [pd.read_csv(D / f, usecols=["ts", "close"]) for f in ("cb_1m.csv", "cb_1m_recent.csv")]
    s = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
    return s.set_index(s.ts.astype("int64") // 60 * 60).close


def main(days: int = 14) -> None:
    sp = spot()
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    out = []
    t = end
    n_ev = 0
    while t > end - timedelta(days=days):
        et = t.astimezone(ET)
        ev = f"KXBTCD-{et:%y%b%d}".upper() + f"{et.hour:02d}"
        close_ts = int(t.timestamp())
        s60 = sp.get(close_ts - 3600 - 60)
        if s60 is None or np.isnan(s60):
            t -= timedelta(hours=1)
            continue
        ms = get("/markets", event_ticker=ev, limit=300).get("markets", [])
        keep = [m for m in ms if m.get("floor_strike") and abs(np.log(m["floor_strike"] / s60)) <= 0.025 and m.get("result") in ("yes", "no")]
        if keep:
            tick = ",".join(m["ticker"] for m in keep[:60])
            c = get("/markets/candlesticks", market_tickers=tick, start_ts=close_ts - 3900, end_ts=close_ts, period_interval=1).get("markets", [])
            by = {m["ticker"]: m for m in keep}
            for cm in c:
                m = by.get(cm.get("market_ticker") or cm.get("ticker"))
                if m is None:
                    continue
                cs = {x["end_period_ts"]: x for x in cm.get("candlesticks", [])}
                for off in (60, 30, 15, 5):
                    x = cs.get(close_ts - off * 60)
                    if not x or not x.get("yes_bid") or not x.get("yes_ask"):
                        continue
                    try:
                        out.append(dict(event=ev, ticker=m["ticker"], close_ts=close_ts, strike=float(m["floor_strike"]), off=off, bid=float(x["yes_bid"]["close_dollars"]), ask=float(x["yes_ask"]["close_dollars"]),
                                        vol=float(x.get("volume_fp") or 0), oi=float(x.get("open_interest_fp") or 0), yes=int(m["result"] == "yes"), spot=float(sp.get(close_ts - off * 60 - 60, np.nan))))
                    except (KeyError, TypeError, ValueError):
                        pass
        n_ev += 1
        if n_ev % 24 == 0:
            print(f"{et:%Y-%m-%d} events {n_ev} rows {len(out)}", flush=True)
            pd.DataFrame(out).to_csv(D / "hourly_pilot.csv", index=False)
        t -= timedelta(hours=1)
    pd.DataFrame(out).to_csv(D / "hourly_pilot.csv", index=False)
    print("done", n_ev, "events", len(out), "rows")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
