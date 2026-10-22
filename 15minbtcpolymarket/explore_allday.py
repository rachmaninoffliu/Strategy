"""Where is the edge across the whole day? EV per contract of underdog / favourite legs by UTC hour and decision minute,
after Polymarket US taker fees and slippage. One trade per market per leg (first qualifying minute)."""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.data import load_rows
from trader.fees import fee_per_contract

SLIP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
FEE = "Polymarket US"
rows = load_rows()
rows = rows[rows.minute.between(1, 13)]
rows = rows[(rows.bid > 0) & (rows.ask < 1) & (rows.ask >= rows.bid) & ((rows.ask - rows.bid) <= 0.06)].copy()
mid = (rows.bid + rows.ask) / 2
yes_dog = mid < 0.5
rows["dog_side"] = np.where(yes_dog, "YES", "NO")
rows["dog_cost"] = np.where(yes_dog, rows.ask, 1 - rows.bid)
rows["fav_cost"] = np.where(yes_dog, 1 - rows.bid + 0, rows.ask)             # favourite = the other side
rows["fav_cost"] = np.where(yes_dog, 1 - rows.bid, rows.ask)
# the favourite side is NO when YES is the dog: NO ask = 1 - yes_bid ; the favourite is YES when YES is not the dog: ask
rows["fav_cost"] = np.where(yes_dog, 1 - rows.bid, rows.ask)
rows["dog_win"] = np.where(yes_dog, rows.yes, 1 - rows.yes)
rows["fav_win"] = 1 - rows.dog_win
# favourite side cost: if YES is the dog (mid<0.5) the favourite is NO with ask = 1 - yes_bid ... which equals dog_cost's complement? no: NO ask = 1 - bid
rows["fav_cost"] = np.where(yes_dog, 1 - rows.bid, rows.ask)
rows["fav_cost"] = np.where(yes_dog, 1 - rows.bid, rows.ask)
rows["hr"] = pd.to_datetime(rows.open_ts, unit="s", utc=True).dt.hour


def leg(df: pd.DataFrame, cost_col: str, win_col: str, lo: float, hi: float, minutes) -> pd.DataFrame:
    d = df[df.minute.isin(minutes) & (df[cost_col] >= lo) & (df[cost_col] <= hi)]
    d = d.sort_values(["ticker", "minute"]).groupby("ticker", as_index=False).first()
    px = np.minimum(d[cost_col] + SLIP, 0.99)
    d["pnl"] = d[win_col] - px - px.apply(lambda c: fee_per_contract(FEE, c))
    d["win"] = d[win_col]
    d["price"] = px
    return d


def summarize(d: pd.DataFrame, key) -> pd.DataFrame:
    g = d.groupby(key).agg(n=("pnl", "size"), win=("win", "mean"), price=("price", "mean"), ev_c=("pnl", lambda x: x.mean() * 100),
                           t=("pnl", lambda x: x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan))
    return g.round(3)


pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)
d0, d1 = pd.to_datetime(rows.open_ts.min(), unit="s"), pd.to_datetime(rows.open_ts.max(), unit="s")
print(f"data {d0.date()} -> {d1.date()}   slippage {SLIP*100:.0f}c   fee {FEE}\n")

print("=== UNDERDOG (cost 5-30c), minutes 12-13: by UTC hour ===")
dog = leg(rows, "dog_cost", "dog_win", 0.05, 0.30, [12, 13])
print(summarize(dog, "hr").T.to_string())
print("\n=== UNDERDOG, all hours pooled, by FIRST decision minute available (single-minute rules) ===")
res = []
for m in range(2, 14):
    x = leg(rows, "dog_cost", "dog_win", 0.05, 0.30, [m])
    res.append((m, len(x), x.win.mean(), x.price.mean(), x.pnl.mean() * 100, x.pnl.mean() / x.pnl.std() * np.sqrt(len(x))))
print(pd.DataFrame(res, columns=["minute", "n", "win", "price", "ev_c", "t"]).round(3).T.to_string())

print("\n=== FAVOURITE (cost 85-97c): by decision minute, all hours pooled ===")
res = []
for m in range(1, 14):
    x = leg(rows, "fav_cost", "fav_win", 0.85, 0.97, [m])
    res.append((m, len(x), x.win.mean(), x.price.mean(), x.pnl.mean() * 100, x.pnl.mean() / x.pnl.std() * np.sqrt(len(x))))
print(pd.DataFrame(res, columns=["minute", "n", "win", "price", "ev_c", "t"]).round(3).T.to_string())

print("\n=== FAVOURITE (cost 85-97c), minutes 3-11, by UTC hour ===")
fav = leg(rows, "fav_cost", "fav_win", 0.85, 0.97, range(3, 12))
print(summarize(fav, "hr").T.to_string())
print("\npooled favourite: n=%d win=%.3f ev=%.2fc" % (len(fav), fav.win.mean(), fav.pnl.mean() * 100))
