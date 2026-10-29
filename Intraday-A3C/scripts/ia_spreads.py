"""Quoted-spread sample for every candidate: 1-minute windows at 10:30, 13:00 and 15:30 ET on three recent days, SIP quotes. Writes reports/spreads.csv (relative spread in bp, medians)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import ia_data as D
from concurrent.futures import ThreadPoolExecutor

DAYS = ["2026-10-14", "2026-10-20", "2026-10-22"]
WINS = [("10:30", "10:31"), ("13:00", "13:01"), ("15:30", "15:31")]

def one(sym):
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest
    c = D.client()
    rows = []
    for d in DAYS:
        for a, b in WINS:
            t0, t1 = pd.Timestamp(f"{d} {a}", tz=D.NY).tz_convert("UTC"), pd.Timestamp(f"{d} {b}", tz=D.NY).tz_convert("UTC")
            for attempt in range(4):
                try:
                    q = c.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym, start=t0.to_pydatetime(), end=t1.to_pydatetime(), feed=DataFeed.SIP, limit=100000)).df
                    break
                except Exception as e:
                    q = None; time.sleep(2)
            if q is None or not len(q):
                continue
            q = q.reset_index(level=0, drop=True)
            q = q[(q.ask_price > 0) & (q.bid_price > 0) & (q.ask_price > q.bid_price)]
            if not len(q):
                continue
            mid = (q.ask_price + q.bid_price) / 2
            rel = (q.ask_price - q.bid_price) / mid * 1e4
            rows.append(dict(symbol=sym, day=d, win=a, n=len(q), spread_bp_med=float(rel.median()), spread_bp_mean=float(rel.mean()), px=float(mid.median()), cents=float((q.ask_price - q.bid_price).median() * 100)))
    return rows

if __name__ == "__main__":
    out = []
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(one, D.ALL):
            out += r
            if r:
                print(r[0]["symbol"], "ok", len(r), flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(D.REPORTS, "spreads_raw.csv"), index=False)
    s = df.groupby("symbol").agg(spread_bp=("spread_bp_med", "median"), spread_cents=("cents", "median"), px=("px", "median"), quotes_per_min=("n", "median")).sort_values("spread_bp")
    s.to_csv(os.path.join(D.REPORTS, "spreads.csv"))
    print(s.round(2).to_string())
