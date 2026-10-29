"""Calibration map: for every (decision minute, price bucket) of the cheap side (the side priced below 50c) and of the expensive side, gross EV = win - ask and net EV after fee + 0.5c slippage, pooled over all hours Jun1 -> now."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from trader.data import load_rows
from trader.fees import fee_per_contract
rows = load_rows()
rows = rows[rows.open_ts >= int(pd.Timestamp("2026-06-01", tz="UTC").timestamp())].copy()
rows = rows[(rows.bid > 0) & (rows.ask < 1) & (rows.ask >= rows.bid) & ((rows.ask - rows.bid) <= 0.06 + 1e-9)]
rows["mid"] = (rows.bid + rows.ask) / 2
yes_dog = rows.mid < 0.5
rows["side"] = np.where(yes_dog, "dog", "fav")
# dog = YES when mid<0.5: cost = ask, win = yes ; fav = NO then: cost = 1 - bid, win = 1 - yes.   (and the mirror when mid >= 0.5)
dog_cost = np.where(yes_dog, rows.ask, 1 - rows.bid)
dog_win = np.where(yes_dog, rows.yes, 1 - rows.yes)
fav_cost = np.where(yes_dog, 1 - rows.bid, rows.ask)
fav_win = 1 - dog_win
SLIP = 0.005
def fee(c): return np.array([fee_per_contract("Kalshi", float(x)) for x in np.minimum(c + SLIP, 0.99)])
bins = [0.0, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5]
rows["dog_cost"], rows["dog_win"] = dog_cost, dog_win
d = rows[(rows.dog_cost > 0)]
d["bucket"] = pd.cut(d.dog_cost, bins)
d["fee"] = fee(d.dog_cost.values)
d["gross"] = d.dog_win - d.dog_cost
d["net"] = d.dog_win - np.minimum(d.dog_cost + SLIP, 0.99) - d.fee
d["period"] = np.where(d.open_ts < pd.Timestamp("2026-08-24", tz="UTC").timestamp(), "dead", np.where(d.open_ts < pd.Timestamp("2026-09-28", tz="UTC").timestamp(), "hot", "recent"))
print("DOG side (price < 50c), pooled over all hours Jun1-now, one observation per (window, minute); slippage 0.5c\n")
for m in (9, 10, 11, 12, 13, 14):
    x = d[d.minute == m]
    g = x.groupby("bucket").agg(n=("net", "size"), price=("dog_cost", "mean"), win=("dog_win", "mean"), gross=("gross", "mean"), net=("net", "mean"), se=("net", lambda v: v.std() / np.sqrt(len(v))))
    g["price"] *= 100; g["win"] *= 100; g["gross"] *= 100; g["net"] *= 100; g["se"] *= 100; g["t"] = g.net / g.se
    print(f"--- minute {m}"); print(g.round(2).to_string())
