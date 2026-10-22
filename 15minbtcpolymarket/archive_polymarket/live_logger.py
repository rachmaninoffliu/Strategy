"""READ-ONLY live logger: Kalshi KXBTC15M book + Polymarket US btc-updown-15m book + Coinbase spot, ~1 Hz.

Never places orders. Purpose: (1) measure Kalshi<->Polymarket-US cross-venue gaps net of fees,
(2) measure quote latency vs spot, (3) give a live out-of-sample set to validate any backtest claim.

Usage:  python live_logger.py [minutes=60]       -> data/live_YYYYmmdd_HHMM.csv
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

K = "https://api.elections.kalshi.com/trade-api/v2"
G = "https://gateway.polymarket.us"
CB = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
DATA = Path(__file__).parent / "data"
S = requests.Session()


def jget(url, params=None, timeout=6):
    try:
        r = S.get(url, params=params, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def kalshi_market(ws: datetime):
    d = jget(K + "/markets", {"series_ticker": "KXBTC15M", "status": "open", "limit": 4})
    for m in (d or {}).get("markets", []):
        if m["open_time"].startswith(ws.strftime("%Y-%m-%dT%H:%M")):
            return m
    return None


def kalshi_book(ticker: str):
    d = jget(K + f"/markets/{ticker}/orderbook")
    ob = (d or {}).get("orderbook_fp") or {}
    yb = [(float(p), float(s)) for p, s in ob.get("yes_dollars", []) if float(s) > 0]
    nb = [(float(p), float(s)) for p, s in ob.get("no_dollars", []) if float(s) > 0]
    if not yb and not nb:
        return None
    ybid = max(yb) if yb else (None, 0)
    nbid = max(nb) if nb else (None, 0)
    return dict(k_ybid=ybid[0], k_ybid_sz=ybid[1], k_yask=None if nbid[0] is None else round(1 - nbid[0], 4), k_yask_sz=nbid[1])


_poly_pause = [0.0]


def poly_book(slug: str):
    if time.time() < _poly_pause[0]:
        return None
    try:
        r = S.get(G + f"/v1/markets/{slug}/book", timeout=6)
    except Exception:
        return None
    if r.status_code == 429:
        _poly_pause[0] = time.time() + 5.0          # gateway limit is stricter than Kalshi's: back off
        return None
    d = r.json() if r.status_code == 200 else None
    md = (d or {}).get("marketData")
    if not md:
        return None
    bids = [(float(x["px"]["value"]), float(x["qty"])) for x in md.get("bids", [])]
    offs = [(float(x["px"]["value"]), float(x["qty"])) for x in md.get("offers", [])]
    bb = max(bids) if bids else (None, 0)
    ba = min(offs) if offs else (None, 0)
    return dict(p_ybid=bb[0], p_ybid_sz=bb[1], p_yask=ba[0], p_yask_sz=ba[1], p_t=round(time.time(), 3))


def coinbase():
    d = jget(CB, timeout=4)
    if not d:
        return None
    return dict(cb_px=float(d["price"]), cb_bid=float(d["bid"]), cb_ask=float(d["ask"]))


FIELDS = ["t", "ws", "kticker", "strike", "k_ybid", "k_ybid_sz", "k_yask", "k_yask_sz", "p_ybid", "p_ybid_sz", "p_yask",
          "p_yask_sz", "p_t", "cb_px", "cb_bid", "cb_ask"]


def main(minutes: float) -> None:
    out = DATA / f"live_{datetime.now():%Y%m%d_%H%M}.csv"
    end = time.time() + minutes * 60
    cur_ws, km, strike = None, None, None
    last_poly: dict = {}
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        while time.time() < end:
            t0 = time.time()
            now = datetime.now(timezone.utc)
            ws = now.replace(minute=now.minute // 15 * 15, second=0, microsecond=0)
            if ws != cur_ws or km is None:
                km = kalshi_market(ws)
                if km:
                    cur_ws, strike = ws, km.get("floor_strike")
            if km:
                row = dict(t=round(t0, 3), ws=int(ws.timestamp()), kticker=km["ticker"], strike=strike)
                if n % 2 == 0:       # Polymarket US polled every 2s; last good quote carried with its own timestamp p_t
                    pb = poly_book(f"cpc-btc-updown-15m-{ws:%Y-%m-%d-%H%M}z")
                    if pb:
                        last_poly = pb
                for part in (kalshi_book(km["ticker"]), last_poly if last_poly.get("p_t", 0) > t0 - 20 else {}, coinbase()):
                    if part:
                        row.update(part)
                w.writerow({k: row.get(k) for k in FIELDS})
                n += 1
                if n % 30 == 0:
                    fh.flush()
                    print(f"{datetime.now():%H:%M:%S} rows {n}  {km['ticker']}  k {row.get('k_ybid')}/{row.get('k_yask')}  "
                          f"p {row.get('p_ybid')}/{row.get('p_yask')}  cb {row.get('cb_px')}", flush=True)
            time.sleep(max(0.0, 1.0 - (time.time() - t0)))
    print("done", out, n)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 60)
