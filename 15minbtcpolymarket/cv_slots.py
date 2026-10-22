import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import cv_longspot as lsp, cv_mechanism as mech
from trader.data import load_rows
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
s = lsp.load_spot_all(); w = lsp.build_windows(s, minutes=(12, 13)); g = mech.fit_g(w)
w["g"] = 0.5
for mn in g:
    i = w.minute == mn; w.loc[i, "g"] = g[mn].predict(w.loc[i, "z"].values)
dog = w.g < 0.5
w["g_dog"] = np.where(dog, w.g, 1 - w.g); w["win"] = np.where(dog, w.yes_proxy, 1 - w.yes_proxy); w["exc"] = w.win - w.g_dog
dt = pd.to_datetime(w.open_ts, unit="s", utc=True)
w["slot"] = dt.dt.strftime("%H:%M"); w["ym"] = dt.dt.strftime("%Y-%m"); w["dow"] = dt.dt.dayofweek
x = w[(w.minute == 12) & (w.g_dog >= 0.12) & (w.g_dog <= 0.38) & (w.open_ts >= lsp.CUT) & w.hour.isin([20, 21, 22, 23])]
print("=== since 2026-08-24: dog excess by 15-minute slot (window OPEN time, UTC), minute 12, dogs priced 12-38c ===")
t = x.groupby("slot").agg(n=("exc", "size"), win=("win", "mean"), priced=("g_dog", "mean"), excess_c=("exc", lambda v: v.mean() * 100))
print(t.round(2).T.to_string())
print("\n=== by weekday (0=Mon), 22-23 UTC only ===")
y = x[x.hour.isin([22, 23])]
print(y.groupby("dow").agg(n=("exc", "size"), win=("win", "mean"), excess_c=("exc", lambda v: v.mean() * 100)).round(2).T.to_string())
print("\n=== by week, 22-23 UTC: n dogs / win / excess ===")
y2 = y.assign(wk=pd.to_datetime(y.open_ts, unit="s", utc=True).dt.to_period("W-SUN").dt.start_time.dt.strftime("%m-%d"))
print(y2.groupby("wk").agg(n=("exc", "size"), win=("win", "mean"), excess_c=("exc", lambda v: v.mean() * 100)).round(2).T.to_string())
# market repricing: real dog price minus g, by month, 22-23 and others
q = load_rows(); q = q[q.minute.isin([12]) & (q.bid > 0) & (q.ask < 1) & (q.ask >= q.bid) & ((q.ask - q.bid) <= 0.06 + 1e-9)].copy()
m = q.merge(w[w.minute == 12][["open_ts", "g", "ym", "hour"]], on="open_ts")
m["mid"] = (m.bid + m.ask) / 2; d = m.mid < 0.5
m["dog_real"] = np.where(d, m.ask, 1 - m.bid); m["g_dog"] = np.where(d, m.g, 1 - m.g)
m = m[(m.g_dog >= 0.12) & (m.g_dog <= 0.38)]
m["gap_c"] = (m.dog_real - m.g_dog) * 100
m["grp"] = np.where(m.hour.isin([22, 23]), "22-23", "other")
print("\n=== REAL dog ask minus pricing function g (cents), by month: has the market been repricing? ===")
print(m.groupby(["ym", "grp"]).gap_c.mean().unstack().round(2).T.to_string())
