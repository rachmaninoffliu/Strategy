"""At the instant a 5-minute bar closes, how far is the SPREAD built from the bar's closing prices from the spread of the quote mids (where an order would actually trade)?  One session, SIP ticks, 3 pairs."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\LevETF-Pairs\scripts")
os.chdir(r"C:\Users\xinyu\PycharmProjects\pythonProject\LevETF-Pairs\scripts")
import levpairs_data as D
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockQuotesRequest, StockTradesRequest

NYTZ = "America/New_York"
cl = D.data_client()
day = "2026-10-23"
t0, t1 = pd.Timestamp(f"{day} 10:00", tz=NYTZ), pd.Timestamp(f"{day} 14:00", tz=NYTZ)
bounds = pd.date_range(t0 + pd.Timedelta(minutes=5), t1, freq="5min")
bars = {s: pd.read_parquet(D.path1(s, "sip")) for s in D.SYMBOLS}


def ticks(sym):
    tr = cl.get_stock_trades(StockTradesRequest(symbol_or_symbols=sym, start=t0.tz_convert("UTC"), end=(t1 + pd.Timedelta(minutes=1)).tz_convert("UTC"), feed=DataFeed.SIP)).data.get(sym, [])
    return pd.DatetimeIndex([x.timestamp for x in tr]).tz_convert(NYTZ), np.array([x.price for x in tr], float)


def mids(sym, b):
    q = cl.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=sym, start=(b - pd.Timedelta(seconds=30)).tz_convert("UTC"), end=(b + pd.Timedelta(seconds=3)).tz_convert("UTC"), feed=DataFeed.SIP)).data.get(sym, [])
    if not q:
        return None
    qt = pd.DatetimeIndex([x.timestamp for x in q]).tz_convert(NYTZ)
    out = []
    for dt in (0.0, 1.0):
        k = qt.searchsorted(b + pd.Timedelta(seconds=dt), side="right") - 1
        if k < 0 or q[k].bid_price <= 0 or q[k].ask_price <= 0:
            return None
        out.append((q[k].bid_price + q[k].ask_price) / 2)
    return out


tick = {s: ticks(s) for s in D.SYMBOLS}
print({s: len(v[0]) for s, v in tick.items()}, flush=True)
rows = []
for L, B in D.PAIRS:
    for b in bounds:
        bn = b.tz_localize(None)
        try:
            CL, CB = bars[L].loc[bn - pd.Timedelta(minutes=1), "close"], bars[B].loc[bn - pd.Timedelta(minutes=1), "close"]
            OL, OB = bars[L].loc[bn, "open"], bars[B].loc[bn, "open"]
        except KeyError:
            continue
        mL, mB = mids(L, b), mids(B, b)
        if mL is None or mB is None:
            continue
        sp = lambda a_l, a_b, b_l, b_b: ((b_l / a_l - 1) - 3.0 * (b_b / a_b - 1)) * 1e4              # move of the spread (bp of the 3x leg) from price pair a to price pair b
        ages = []
        for s in (L, B):
            ts, px = tick[s]
            i = ts.searchsorted(b, side="left")
            ages.append((b - ts[i - 1]).total_seconds())
        rows.append(dict(pair=f"{L}/{B}", b=b, close_to_mid_at_b=sp(CL, CB, mL[0], mB[0]), close_to_open=sp(CL, CB, OL, OB), close_to_mid_1s=sp(CL, CB, mL[1], mB[1]), mid_move_1s=sp(mL[0], mB[0], mL[1], mB[1]),
                         age_L=ages[0], age_B=ages[1], bar_close_is_last_tick_L=bool(abs(tick[L][1][tick[L][0].searchsorted(b, side="left") - 1] - CL) < 1e-9)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(os.getcwd()), "reports", "tick_check2_2026-10-23.csv"), index=False)
f = lambda x: pd.Series({"mean |x|": x.abs().mean(), "p90 |x|": x.abs().quantile(0.9), "max |x|": x.abs().max()})
for col in ("close_to_mid_at_b", "close_to_open", "close_to_mid_1s", "mid_move_1s"):
    print(f"\n{col}  (bp of the 3x leg, |.|)")
    print(df.groupby("pair")[col].apply(lambda x: f(x)).unstack().round(2).to_string())
print("\nbar close equals the last tick before the boundary:", df.groupby("pair").bar_close_is_last_tick_L.mean().round(2).to_dict())
print("age of the last print of the 3x leg (s) median / p90:", df.groupby("pair").age_L.agg(["median", lambda x: x.quantile(0.9)]).round(2).to_dict())
print("rows", len(df))
