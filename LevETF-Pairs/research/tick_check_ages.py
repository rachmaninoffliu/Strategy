"""What does an order sent right after a 5-minute bar closes actually get?  One recent session, SIP ticks.
For each 5-minute boundary b and symbol: how old is the last trade of the bar (the 'close' of the bar), how soon is the first trade of the next bar (its 'open'), and how far is the last trade from the
quote mid at b (where a marketable order would really fill)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\LevETF-Pairs\scripts")
os.chdir(r"C:\Users\xinyu\PycharmProjects\pythonProject\LevETF-Pairs\scripts")
import levpairs_data as D
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockQuotesRequest, StockTradesRequest

cl = D.data_client()
day = "2026-10-23"
t0, t1 = pd.Timestamp(f"{day} 10:00", tz="America/New_York"), pd.Timestamp(f"{day} 12:00", tz="America/New_York")
bounds = pd.date_range(t0 + pd.Timedelta(minutes=5), t1, freq="5min")
rows = []
for sym in D.SYMBOLS:
    tr = cl.get_stock_trades(StockTradesRequest(symbol_or_symbols=sym, start=t0.tz_convert("UTC"), end=(t1 + pd.Timedelta(minutes=1)).tz_convert("UTC"), feed=DataFeed.SIP)).data.get(sym, [])
    ts = pd.DatetimeIndex([x.timestamp for x in tr]).tz_convert("America/New_York")
    px = np.array([x.price for x in tr], float)
    print(f"{sym}: {len(tr):,} trades", flush=True)
    for b in bounds:
        i = ts.searchsorted(b, side="left")                      # first trade at / after the boundary
        if i == 0 or i >= len(ts):
            continue
        last_t, first_t = ts[i - 1], ts[i]
        q = cl.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym, start=(b - pd.Timedelta(seconds=30)).tz_convert("UTC"), end=(b + pd.Timedelta(seconds=3)).tz_convert("UTC"), feed=DataFeed.SIP)).data.get(sym, [])
        qt = pd.DatetimeIndex([x.timestamp for x in q]).tz_convert("America/New_York")
        if not len(q):
            continue
        k0 = qt.searchsorted(b, side="right") - 1                 # the quote in force at the boundary
        k1 = qt.searchsorted(b + pd.Timedelta(seconds=1), side="right") - 1      # the quote an order arriving one second later meets
        if k0 < 0:
            continue
        mid0 = (q[k0].bid_price + q[k0].ask_price) / 2
        mid1 = (q[k1].bid_price + q[k1].ask_price) / 2
        if q[k0].bid_price <= 0 or q[k0].ask_price <= 0:
            continue
        rows.append(dict(sym=sym, b=b, last_age_s=(b - last_t).total_seconds(), first_wait_s=(first_t - b).total_seconds(), close_vs_mid_bp=(px[i - 1] / mid0 - 1) * 1e4,
                         open_vs_mid1_bp=(px[i] / mid1 - 1) * 1e4, mid_move_1s_bp=(mid1 / mid0 - 1) * 1e4, half_spread_bp=(q[k0].ask_price - q[k0].bid_price) / 2 / mid0 * 1e4))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(os.getcwd()), "reports", "tick_check_2026-10-23.csv"), index=False)
g = df.groupby("sym")
out = pd.DataFrame({
    "boundaries": g.size(),
    "age of last trade (s) median": g.last_age_s.median(), "p90": g.last_age_s.quantile(0.9),
    "wait to first trade (s) median": g.first_wait_s.median(), "p90 ": g.first_wait_s.quantile(0.9),
    "|close - mid| bp median": g.close_vs_mid_bp.apply(lambda x: x.abs().median()), "|close - mid| p90": g.close_vs_mid_bp.apply(lambda x: x.abs().quantile(0.9)),
    "half-spread bp median": g.half_spread_bp.median(),
    "|mid move in 1 s| bp median": g.mid_move_1s_bp.apply(lambda x: x.abs().median()),
}).round(2)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.to_string())
