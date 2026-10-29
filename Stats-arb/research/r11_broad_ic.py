import sys
from rlib import *
from scipy.stats import rankdata

if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "broad"
    variant = sys.argv[2] if len(sys.argv) > 2 else "pca15"
    min_adv = float(sys.argv[3]) if len(sys.argv) > 3 else 1e7
    panel = panel_for(universe)
    cfg = replace(VARIANTS[variant], min_adv=min_adv)
    name = f"{variant}_{int(min_adv/1e6)}m_noearn"
    sig = sig_for(universe, name, cfg, panel, earn=False)
    O = panel.open.to_numpy(float); Cc = panel.close.to_numpy(float)
    T, N = O.shape
    sec = sig.sector_idx
    def demean(x):
        y = x.copy()
        for k in np.unique(sec):
            m = sec == k
            y[:, m] = x[:, m] - np.nanmedian(x[:, m], axis=1, keepdims=True)
        return y
    def fwd(h):
        out = np.full((T, N), np.nan)
        for d in range(T - 1 - h):
            out[d] = O[d + 1 + h] / O[d + 1] - 1
        out[np.abs(out) > 0.9] = np.nan
        return demean(out)
    ovn = np.full((T, N), np.nan); ovn[:-1] = O[1:] / Cc[:-1] - 1; ovn[np.abs(ovn) > .45] = np.nan
    F = {"1d": fwd(1), "3d": fwd(3), "5d": fwd(5), "ovn": demean(ovn)}
    s = -sig.s
    base = sig.univ & np.isfinite(sig.s)
    since = np.searchsorted(sig.dates.values, np.datetime64("2017-06-01"))
    print(f"{universe} {variant} min_adv {min_adv/1e6:.0f}M: avg tradable names/day {base[since:].sum(1).mean():.0f}")
    for hname, Fh in F.items():
        ics = []
        for d in range(since, T):
            m = base[d] & np.isfinite(Fh[d])
            if m.sum() > 100:
                ics.append(np.corrcoef(rankdata(s[d][m]), rankdata(Fh[d][m]))[0, 1])
        ics = np.array(ics)
        print(f"  IC(-s vs fwd {hname}): {ics.mean():+.4f}  t {ics.mean()/ics.std()*np.sqrt(len(ics)):5.2f}")
    # fade return by |s| bucket and by ADV tier (3d)
    a = np.abs(sig.s)
    direction = -np.sign(sig.s)
    adv = sig.adv
    mask0 = np.zeros((T, N), bool); mask0[since:] = True; mask0 &= base
    rows = []
    for lab, lo, hi in (("1.25-1.75", 1.25, 1.75), ("1.75-2.5", 1.75, 2.5), (">2.5", 2.5, 99)):
        for tier, alo, ahi in (("ADV>200M", 2e8, 1e12), ("50-200M", 5e7, 2e8), ("20-50M", 2e7, 5e7), ("8-20M", 7e6, 2e7)):
            m = mask0 & (a >= lo) & (a < hi) & (adv >= alo) & (adv < ahi)
            v = (direction * F["3d"])[m]; v = v[np.isfinite(v)]
            if len(v) > 300:
                rows.append((lab, tier, len(v), v.mean() * 1e4, v.mean() / v.std() * np.sqrt(len(v) / 3), (direction * F["1d"])[m][np.isfinite((direction * F["1d"])[m])].mean() * 1e4))
    print(pd.DataFrame(rows, columns=["|s|", "ADV tier", "n", "3d fade bp", "t(adj)", "1d fade bp"]).round(1).to_string(index=False))
