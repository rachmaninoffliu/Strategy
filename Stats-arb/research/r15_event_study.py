import sys
from rlib import *

universe = sys.argv[1] if len(sys.argv) > 1 else "core"
panel = panel_for(universe)
earn = D.load_earnings()
er = D.earnings_reaction_matrix(panel, earn)
O = panel.open.to_numpy(float); Cc = panel.close.to_numpy(float)
T, N = O.shape
dates = panel.dates
secs = sorted(set(panel.sector.values)); sec = np.array([secs.index(panel.sector[c]) for c in panel.stocks])
ro = np.full((T, N), np.nan); ro[:-1] = O[1:] / O[:-1] - 1
ro[np.abs(ro) > 0.6] = np.nan
# sector-relative (median of the sector that day), so the market and sector moves drop out
rel = ro.copy()
for k in np.unique(sec):
    m = sec == k
    rel[:, m] = ro[:, m] - np.nanmedian(ro[:, m], axis=1, keepdims=True)
adv = pd.DataFrame(Cc * panel.volume.to_numpy(float)).rolling(20, min_periods=15).median().to_numpy()
ok = (Cc >= 5) & (adv >= 2e7)
t0 = np.searchsorted(dates.values, np.datetime64("2017-06-01"))
evs = [(d, j) for d, j in zip(*np.where(er)) if d >= t0 and d < T - 25 and ok[d - 1, j]]
print(f"{universe}: {len(evs)} earnings events since 2017-06 among liquid names")
K = range(-10, 21)
mat = np.full((len(evs), len(K)), np.nan)
for i, (d, j) in enumerate(evs):
    for c, k in enumerate(K):
        if 0 <= d + k < T:
            mat[i, c] = rel[d + k, j]
mean = np.nanmean(mat, axis=0) * 1e4
se = np.nanstd(mat, axis=0) / np.sqrt(np.sum(np.isfinite(mat), axis=0)) * 1e4
print("event day k (period open(r+k)->open(r+k+1); the earnings GAP is in k=-1)   mean sector-relative return, bp   t")
for c, k in enumerate(K):
    print(f"  k={k:+3d}  {mean[c]:7.1f}   t={mean[c]/se[c]:5.1f}")
pre = np.nansum(mat[:, [list(K).index(k) for k in range(-6, -1)]], axis=1)
print(f"\nrun-up k=-6..-2 : mean {np.nanmean(pre)*1e4:.1f} bp  t {np.nanmean(pre)/np.nanstd(pre)*np.sqrt(np.isfinite(pre).sum()):.1f}")
# drift conditional on the reaction (gap) sign
gap = mat[:, list(K).index(-1)]
post = np.nansum(mat[:, [list(K).index(k) for k in range(0, 10)]], axis=1)
for lab, m in (("positive reaction (top tercile)", gap > np.nanquantile(gap, 2 / 3)), ("negative reaction (bottom tercile)", gap < np.nanquantile(gap, 1 / 3))):
    v = post[m]
    print(f"drift k=0..9 after a {lab}: mean {np.nanmean(v)*1e4:+.1f} bp  t {np.nanmean(v)/np.nanstd(v)*np.sqrt(np.isfinite(v).sum()):.1f}")
