"""
DA: what does a market order really cost in these stocks? Median SIP quoted half-spread (bp) of the top-100 members and the hedge ETFs at 10:00, 12:00 and 15:45 ET on sample days.
Writes reports/r0_spreads.csv and prints the distribution - the input for the backtest's per-side cost.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from resarb import data as D  # noqa: E402


def main():
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest
    c = D.data_client()
    u = D.load_universe()
    days = sys.argv[1].split(",") if len(sys.argv) > 1 else ["2024-02-14", "2024-06-12", "2024-10-16", "2025-03-12", "2025-08-13", "2026-01-14", "2026-06-10", "2026-10-14"]
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    rows = []
    for d in days:
        mo = pd.Timestamp(d).to_period("M").to_timestamp()
        syms = list(u[(u.month == mo) & (u["rank"] <= 100)].symbol) + D.ETFS
        for hm in ("10:00", "12:00", "15:45"):
            t0 = pd.Timestamp(f"{d} {hm}", tz=D.NY)
            for i in range(0, len(syms), 60):
                ch = syms[i:i + 60]
                try:
                    q = c.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=ch, start=t0.to_pydatetime(), end=(t0 + pd.Timedelta(seconds=3)).to_pydatetime(), feed=DataFeed.SIP,
                                                              limit=2_000_000)).df
                except Exception as e:                                                       # noqa: BLE001
                    print(d, hm, "failed", type(e).__name__, str(e)[:100])
                    continue
                if not len(q):
                    continue
                q = q.reset_index()
                q = q[(q.bid_price > 0) & (q.ask_price > q.bid_price)]
                q["hs_bp"] = (q.ask_price - q.bid_price) / (q.ask_price + q.bid_price) * 1e4
                g = q.groupby("symbol").hs_bp.median()
                for s, v in g.items():
                    rows.append(dict(day=d, time=hm, symbol=s, half_spread_bp=float(v), etf=s in D.ETFS))
            print(d, hm, len(rows), flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(HERE), "research", "reports")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"r0_spreads{tag}.csv"), index=False)
    st = df[~df.etf]
    per = st.groupby("symbol").half_spread_bp.median()
    print("\nstocks: median half-spread per symbol (bp):", per.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict())
    print("stocks by time of day (median of symbols):", st.groupby("time").half_spread_bp.median().round(2).to_dict())
    print("stocks by year:", st.assign(y=st.day.str[:4]).groupby("y").half_spread_bp.median().round(2).to_dict())
    print("ETFs:", df[df.etf].groupby("symbol").half_spread_bp.median().round(2).to_dict())
    print("dollar-weighted? mean of symbol medians:", round(per.mean(), 2))


if __name__ == "__main__":
    main()
