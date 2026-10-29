import sys, pickle, os
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
from scipy.stats import rankdata
import statarb_data as D

panel = D.load_panel()
sec = np.array([sorted(set(panel.sector.values)).index(panel.sector[c]) for c in panel.close.columns])
Cc = panel.close.to_numpy(float); O = panel.open.to_numpy(float)
T, N = Cc.shape
dates = panel.dates
r1 = np.full((T, N), np.nan); r1[1:] = Cc[1:] / Cc[:-1] - 1
r1[np.abs(r1) > 0.45] = np.nan

def sec_rel(x):
    y = x.copy()
    for k in np.unique(sec):
        m = sec == k
        y[:, m] = x[:, m] - np.nanmedian(x[:, m], axis=1, keepdims=True)
    return y

R = sec_rel(r1)                       # sector-relative daily returns
cum = np.nancumsum(np.nan_to_num(R), axis=0)    # sector-relative log-ish cumulative (additive approx)
def window_sum(a, b):                 # sum of R over [t-b, t-a]  (a<=b)  -> cum[t-a]-cum[t-b-1]
    out = np.full((T, N), np.nan)
    for t in range(b + 1, T):
        out[t] = cum[t - a] - cum[t - b - 1]
    return out
buckets = {"d0": (0, 0), "d1-4": (1, 4), "d5-9": (5, 9), "d10-20": (10, 20), "d21-62 (skip 1m)": (21, 62), "d63-126": (63, 126), "d127-252": (127, 252), "12-1m (21..252)": (21, 252)}
X = {k: window_sum(*v) for k, v in buckets.items()}
vol = pd.DataFrame(R).rolling(60, min_periods=40).std().to_numpy()

def fwd(h):
    out = np.full((T, N), np.nan)
    for d in range(T - 1 - h):
        out[d] = O[d + 1 + h] / O[d + 1] - 1
    out[np.abs(out) > 0.9] = np.nan
    return sec_rel(out)
FW = {5: fwd(5), 21: fwd(21)}
adv = pd.DataFrame(Cc * panel.volume.to_numpy(float)).rolling(20, min_periods=15).median().to_numpy()
univ = (Cc >= 5) & (adv >= 5e7)
t0 = np.searchsorted(dates.values, np.datetime64("2017-06-01"))
rows = []
for h in (5, 21):
    for k, a in X.items():
        ics = []
        for t in range(t0, T - h - 2, 3):
            m = univ[t] & np.isfinite(a[t]) & np.isfinite(FW[h][t])
            if m.sum() > 100:
                ics.append(np.corrcoef(rankdata(a[t][m]), rankdata(FW[h][t][m]))[0, 1])
        ics = np.array(ics)
        # overlap-aware t: effective n ~ n * 3 / max(1, h)   (sampled every 3 days, forward window h)
        neff = len(ics) * 3 / max(3, h)
        rows.append((h, k, ics.mean(), ics.mean() / ics.std() * np.sqrt(neff)))
print(pd.DataFrame(rows, columns=["fwd days", "lookback bucket (sector-relative)", "mean IC", "t (overlap-adj)"]).round(4).to_string(index=False))
