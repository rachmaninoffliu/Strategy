import sys, pickle, os
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
from scipy.stats import rankdata
import statarb_data as D

SP = os.path.dirname(os.path.abspath(__file__))
panel = D.load_panel()
sig = pickle.load(open(os.path.join(SP, f"sig_{sys.argv[1] if len(sys.argv)>1 else 'pca15'}.pkl"), "rb"))
O = panel.open.to_numpy(float)
T, N = O.shape
ro = np.full((T, N), np.nan); ro[:-1] = O[1:] / O[:-1] - 1
bad = np.abs(ro) > 0.45; ro[bad] = np.nan
# forward returns starting at the open of d+1 : 1-day (to open d+2), 3-day, 5-day
def fwd(h):
    out = np.full((T, N), np.nan)
    for d in range(T - 1 - h):
        out[d] = O[d + 1 + h] / O[d + 1] - 1
    out[np.abs(out) > 0.9] = np.nan
    return out
F = {1: fwd(1), 3: fwd(3), 5: fwd(5), 10: fwd(10)}
sec = sig.sector_idx

def demean(x):  # cross-sectional, within sector
    y = x.copy()
    for k in np.unique(sec):
        m = sec == k
        y[:, m] = x[:, m] - np.nanmedian(x[:, m], axis=1, keepdims=True)
    return y
FD = {h: demean(v) for h, v in F.items()}

def ic_series(sig_arr, fwd_arr, mask):
    out = np.full(T, np.nan)
    for d in range(T):
        m = mask[d] & np.isfinite(sig_arr[d]) & np.isfinite(fwd_arr[d])
        if m.sum() > 60:
            a, b = rankdata(sig_arr[d][m]), rankdata(fwd_arr[d][m])
            out[d] = np.corrcoef(a, b)[0, 1]
    return pd.Series(out, index=sig.dates)

s = -sig.s  # long when s very negative
base = sig.univ
filt = base & ~sig.news & ~sig.post_earn & ~sig.earn_soon
ev = base & (sig.news | sig.post_earn | sig.earn_soon)
rows = []
for name, mask in (("all", base), ("clean (no news/earn)", filt), ("event days", ev)):
    for h in (1, 3, 5, 10):
        ic = ic_series(s, FD[h], mask).dropna()
        ic = ic[ic.index >= "2017-06-01"]
        rows.append((name, h, ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)), len(ic)))
print(pd.DataFrame(rows, columns=["subset", "horizon_d", "mean_IC", "t_stat", "n"]).round(4).to_string(index=False))
ic = ic_series(s, FD[3], filt).dropna()
print("\nIC (3d fwd, clean subset) by year:")
print(ic.groupby(ic.index.year).agg(["mean", "count"]).assign(t=lambda d: d["mean"] / ic.groupby(ic.index.year).std() * np.sqrt(d["count"])).round(4).to_string())
# long-short decile spread on clean subset, 3d horizon, by year
sp = []
for d in range(T):
    m = filt[d] & np.isfinite(s[d]) & np.isfinite(FD[3][d])
    if m.sum() > 100:
        x, y = s[d][m], FD[3][d][m]
        q = np.quantile(x, [0.1, 0.9])
        sp.append((sig.dates[d], y[x >= q[1]].mean() - y[x <= q[0]].mean()))
sp = pd.Series(dict(sp))
sp = sp[sp.index >= "2017-06-01"]
print("\ntop-minus-bottom decile 3d fwd spread (bp) by year:")
print((sp.groupby(sp.index.year).mean() * 1e4).round(1).to_string())
print("share of name-days flagged: news %.3f  post_earn %.3f  earn_soon %.3f" % (sig.news[base].mean(), sig.post_earn[base].mean(), sig.earn_soon[base].mean()))
