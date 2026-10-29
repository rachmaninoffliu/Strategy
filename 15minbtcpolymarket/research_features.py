import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from research_ds import build
ds = build()
ds["log_vol"] = np.log1p(ds.vol_dec.fillna(0))
ds["weekend"] = ds.dow.isin([5, 6]).astype(float)
feats = ["vol_state30", "vol_state120", "move_state", "vr360", "vr1440", "vr4320", "z_dog", "dog_move3", "dog_move6", "prev_abs", "spread", "log_vol"]
def slope_t(x, y):
    x = (x - x.mean()) / x.std()
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    # HC1 robust se
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * res ** 2) @ X @ XtXi * len(y) / (len(y) - 2)
    return b[1] * 100, b[1] / np.sqrt(V[1, 1])
print(f"all-hours dog signals: n={len(ds)}, baseline net EV {ds.net.mean() * 100:+.2f}c.  Slope = change in net EV (c) per 1 sd of the feature (HC robust t).  Bonferroni bar for 14 tests: |t| > 2.9\n")
print(f"{'feature':14s} {'ALL slope(t)':>16s} {'dead':>16s} {'hot':>16s} {'recent':>16s}   top-quintile EV | bottom-quintile EV")
rows = []
for f in feats + ["weekend"]:
    d = ds.dropna(subset=[f])
    out = []
    for lab, sub in [("ALL", d)] + [(p, d[d.period == p]) for p in sorted(d.period.unique())]:
        s, t = slope_t(sub[f].values.astype(float), sub.net.values)
        out.append(f"{s:+6.2f}c ({t:+5.2f})")
    q = pd.qcut(d[f], 5, duplicates="drop", labels=False) if f != "weekend" else d[f]
    top = d[q == q.max()].net.mean() * 100; bot = d[q == q.min()].net.mean() * 100
    print(f"{f:14s} {out[0]:>16s} {out[1]:>16s} {out[2]:>16s} {out[3]:>16s}   {top:+6.2f}c | {bot:+6.2f}c")
