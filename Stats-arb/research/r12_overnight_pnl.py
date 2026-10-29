import sys
from rlib import *
from scipy.stats import rankdata

def run(universe, name):
    panel = panel_for(universe)
    sig = sig_for(universe, name, VARIANTS["pca15"] if name.startswith("pca15") else None, panel, earn=False) if False else pickle.load(open(os.path.join(SP, f"sig_{universe}_{name}.pkl"), "rb"))
    O = panel.open.to_numpy(float); Cc = panel.close.to_numpy(float)
    T, N = O.shape
    sec = sig.sector_idx
    def demean(x):
        y = x.copy()
        for k in np.unique(sec):
            m = sec == k
            y[:, m] = x[:, m] - np.nanmedian(x[:, m], axis=1, keepdims=True)
        return y
    ovn = np.full((T, N), np.nan); ovn[:-1] = O[1:] / Cc[:-1] - 1; ovn[np.abs(ovn) > .45] = np.nan
    r1 = np.full((T, N), np.nan); r1[1:] = Cc[1:] / Cc[:-1] - 1; r1[np.abs(r1) > .45] = np.nan
    intr = Cc / O - 1; intr[np.abs(intr) > .45] = np.nan
    vol60 = pd.DataFrame(r1).rolling(60, min_periods=40).std().to_numpy()
    sigs = {"-s (OU)": -sig.s, "-1d ret": -demean(r1), "-1d ret/vol": -demean(r1) / vol60, "-intraday/vol": -demean(intr) / vol60}
    adv = sig.adv
    base = sig.univ & np.isfinite(ovn)
    since = np.searchsorted(sig.dates.values, np.datetime64("2017-06-01"))
    print(f"\n=== {universe}/{name}  tradable names/day {base[since:].sum(1).mean():.0f} ===")
    print(f"{'signal':16s} {'construction':26s} gross bp/day  Sharpe(gross)   net@0.5bp/side  net@1bp  net@2bp   (Sharpe, ann.)")
    for sname, a in sigs.items():
        for cons, fn in (("rank-weighted L/S", "rank"), ("top/bottom decile", "dec"), ("top/bottom 2% (extreme)", "ext")):
            pnl = []
            for d in range(since, T - 1):
                m = base[d] & np.isfinite(a[d])
                if m.sum() < 150:
                    continue
                x, g = a[d][m], ovn[d][m]
                if fn == "rank":
                    w = rankdata(x) - (m.sum() + 1) / 2
                    w = w / np.abs(w).sum()
                else:
                    q = 0.1 if fn == "dec" else 0.02
                    lo, hi = np.quantile(x, [q, 1 - q])
                    w = np.where(x >= hi, 1.0, np.where(x <= lo, -1.0, 0.0))
                    w = w / np.abs(w).sum()
                pnl.append((sig.dates[d], float(w @ g)))
            p = pd.Series(dict(pnl))
            sh = lambda r: r.mean() / r.std() * np.sqrt(252)
            print(f"{sname:16s} {cons:26s} {p.mean()*1e4:8.2f}      {sh(p):6.2f}         {sh(p-2*0.5e-4):6.2f}      {sh(p-2*1e-4):6.2f}   {sh(p-2*2e-4):6.2f}")

if __name__ == "__main__":
    run("core", "pca15") if os.path.exists(os.path.join(SP, "sig_core_pca15.pkl")) else None
    run("broad", "pca15_10m_noearn")
