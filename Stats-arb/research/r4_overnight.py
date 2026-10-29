import sys, pickle, os
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
from scipy.stats import rankdata
import statarb_data as D

SP = os.path.dirname(os.path.abspath(__file__))
panel = D.load_panel()
sig = pickle.load(open(os.path.join(SP, "sig_pca15.pkl"), "rb"))
O = panel.open.to_numpy(float); Cc = panel.close.to_numpy(float)
T, N = O.shape
sec = sig.sector_idx
dates = sig.dates

def demean(x):
    y = x.copy()
    for k in np.unique(sec):
        m = sec == k
        y[:, m] = x[:, m] - np.nanmedian(x[:, m], axis=1, keepdims=True)
    return y

on = np.full((T, N), np.nan); on[:-1] = O[1:] / Cc[:-1] - 1          # overnight gap decided at close d, earned to open d+1
on[np.abs(on) > 0.45] = np.nan
ond = demean(on)
r1 = np.full((T, N), np.nan); r1[1:] = Cc[1:] / Cc[:-1] - 1; r1[np.abs(r1) > .45] = np.nan
intraday = Cc / O - 1; intraday[np.abs(intraday) > .45] = np.nan
gap_today = np.full((T, N), np.nan); gap_today[1:] = O[1:] / Cc[:-1] - 1
r5 = np.full((T, N), np.nan); r5[5:] = Cc[5:] / Cc[:-5] - 1
r20 = np.full((T, N), np.nan); r20[20:] = Cc[20:] / Cc[:-20] - 1
sd60 = pd.DataFrame(r1).rolling(60, min_periods=40).std().shift(0).to_numpy()
cands = {
    "-s (OU, PCA15 residual 60d)": -sig.s,
    "-1d return (sector-demeaned)": -demean(r1),
    "-1d return / vol": -demean(r1) / sd60,
    "-intraday O->C (demeaned)": -demean(intraday),
    "-overnight gap today": -demean(gap_today),
    "-5d return (demeaned)": -demean(r5),
    "-20d return (demeaned)": -demean(r20),
}
base = sig.univ & np.isfinite(ond)
since = pd.Timestamp("2017-06-01")

def ic_series(a):
    out = {}
    for d in range(T):
        if dates[d] < since:
            continue
        m = base[d] & np.isfinite(a[d])
        if m.sum() > 100:
            out[dates[d]] = np.corrcoef(rankdata(a[d][m]), rankdata(ond[d][m]))[0, 1]
    return pd.Series(out)

def spread_bp(a, q=0.1):
    out = {}
    for d in range(T):
        if dates[d] < since:
            continue
        m = base[d] & np.isfinite(a[d])
        if m.sum() > 100:
            x, y = a[d][m], ond[d][m]
            lo, hi = np.quantile(x, [q, 1 - q])
            out[dates[d]] = (y[x >= hi].mean() - y[x <= lo].mean()) * 1e4
    return pd.Series(out)

tab, yr = [], {}
for name, a in cands.items():
    ic = ic_series(a)
    sp = spread_bp(a)
    tab.append((name, ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)), sp.mean(), sp.mean() / sp.std() * np.sqrt(252)))
    yr[name] = ic.groupby(ic.index.year).mean()
print("Predicting the OVERNIGHT gap  C[d] -> O[d+1]  (sector-demeaned), 2017-06 .. 2026-10, liquid universe, signals use the official close")
print(pd.DataFrame(tab, columns=["signal", "mean IC", "t", "decile spread bp/day", "spread Sharpe(ann, gross)"]).round(3).to_string(index=False))
print("\nIC by year"); print(pd.DataFrame(yr).round(4).to_string())
