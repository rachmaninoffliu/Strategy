"""Is there a SPOT-based real-time indicator of the 22-23 UTC regime?  Evening (20-24 UTC) variance ratio / vol of the previous 3, 7, 14 days vs the core-hour dog EV that followed."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from research_ds import build, load_spot
ds = build()
core = ds[ds.hour.isin([22, 23])].copy()
s = load_spot()
lc = np.log(s.close.values.astype(float)); idx = pd.to_datetime(s.index, unit="s", utc=True)
r1 = pd.Series(np.r_[0.0, np.diff(lc)], index=idx); r5 = pd.Series(lc, index=idx).diff(5)
eve = (idx.hour >= 20)
day = idx.floor("D")
g = pd.DataFrame({"r1sq": r1.pow(2).where(eve), "r5sq": (r5.pow(2) / 5).where(eve), "r1abs": r1.abs().where(eve), "day": day}).groupby("day").mean()
g["vr_eve"] = g.r5sq / g.r1sq
g["vol_eve"] = np.sqrt(g.r1sq)
for N in (3, 7, 14):
    g[f"vr_eve{N}"] = g.vr_eve.rolling(N).mean().shift(1)                  # previous N evenings only
    g[f"vol_eve{N}"] = g.vol_eve.rolling(N).mean().shift(1)
    g[f"volrel_eve{N}"] = g[f"vol_eve{N}"] / g.vol_eve.rolling(60).median().shift(1)
core["day"] = pd.to_datetime(core.open_ts, unit="s", utc=True).dt.floor("D")
core = core.join(g[[c for c in g.columns if c.endswith(("3", "7", "14"))]], on="day").dropna(subset=["vr_eve7"])
def slope_t(x, y):
    x = (x - x.mean()) / x.std(); X = np.c_[np.ones(len(x)), x]; b = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ b
    XtXi = np.linalg.inv(X.T @ X); V = XtXi @ (X.T * res ** 2) @ X @ XtXi * len(y) / (len(y) - 2)
    return b[1] * 100, b[1] / np.sqrt(V[1, 1])
print(f"core-hour (22-23 UTC) dog signals, Jun1-now: n={len(core)}, mean net EV {core.net.mean() * 100:+.2f}c  (slippage 2c)")
for f in ("vr_eve3", "vr_eve7", "vr_eve14", "volrel_eve3", "volrel_eve7", "volrel_eve14"):
    out = []
    for lab, sub in [("ALL", core)] + [(p, core[core.period == p]) for p in sorted(core.period.unique())]:
        sl, t = slope_t(sub[f].values.astype(float), sub.net.values); out.append(f"{lab[:3]} {sl:+6.2f}c ({t:+5.2f})")
    print(f"  {f:13s} " + " | ".join(out))
print("(6 tests; Bonferroni bar |t| > 2.6)")
# does the evening variance ratio look different in the hot weeks at all?
wk = core.assign(wk=pd.to_datetime(core.open_ts, unit="s", utc=True).dt.to_period("W-SUN").dt.start_time).groupby("wk").agg(n=("net", "size"), ev=("net", lambda v: v.mean() * 100), vr7=("vr_eve7", "mean"), volrel7=("volrel_eve7", "mean"))
print("\nweekly core EV vs previous-7-evening variance ratio / relative vol:"); print(wk.round(2).to_string()); print("corr(ev, vr7) =", round(wk.ev.corr(wk.vr7), 2), " corr(ev, volrel7) =", round(wk.ev.corr(wk.volrel7), 2), " weeks:", len(wk))
