import sys, pickle, os
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
import statarb_data as D

SP = os.path.dirname(os.path.abspath(__file__))
panel = D.load_panel()
sig = pickle.load(open(os.path.join(SP, f"sig_{sys.argv[1] if len(sys.argv) > 1 else 'pca15'}.pkl"), "rb"))
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

def fwd(h):
    out = np.full((T, N), np.nan)
    for d in range(T - 1 - h):
        out[d] = O[d + 1 + h] / O[d + 1] - 1
    out[np.abs(out) > 0.9] = np.nan
    return demean(out)

F1, F3, F5 = fwd(1), fwd(3), fwd(5)
s = sig.s
vr = sig.vol_ratio / np.nanmedian(sig.vol_ratio, axis=1, keepdims=True)    # idiosyncratic volume ratio
since = np.searchsorted(dates.values, np.datetime64("2017-06-01"))
mask = np.zeros((T, N), bool); mask[since:] = True
mask &= sig.univ & np.isfinite(s)
direction = -np.sign(s)

def table(title, groups):
    print("\n" + title)
    rows = []
    for name, g in groups:
        m = mask & g
        n = m.sum()
        if n < 200:
            continue
        r = [name, n]
        for F in (F1, F3, F5):
            v = (direction * F)[m]
            v = v[np.isfinite(v)]
            r += [v.mean() * 1e4, v.mean() / v.std() * np.sqrt(len(v))]
        rows.append(r)
    print(pd.DataFrame(rows, columns=["bucket", "n", "1d bp", "t", "3d bp", "t", "5d bp", "t"]).round(1).to_string(index=False))

a = np.abs(s)
table("Fade trade return (bp, sector-demeaned) by |s| bucket", [("|s|<0.5", a < .5), ("0.5-1.0", (a >= .5) & (a < 1)), ("1.0-1.25", (a >= 1) & (a < 1.25)), ("1.25-1.5", (a >= 1.25) & (a < 1.5)),
      ("1.5-2.0", (a >= 1.5) & (a < 2)), ("2.0-2.5", (a >= 2) & (a < 2.5)), ("2.5-3.5", (a >= 2.5) & (a < 3.5)), (">3.5", a >= 3.5)])
big = a >= 1.25
table("Among |s|>=1.25: by idiosyncratic volume ratio of the signal day", [("vr<0.8", big & (vr < .8)), ("0.8-1.0", big & (vr >= .8) & (vr < 1)), ("1.0-1.3", big & (vr >= 1) & (vr < 1.3)),
      ("1.3-2.0", big & (vr >= 1.3) & (vr < 2)), (">=2.0", big & (vr >= 2))])
table("Among |s|>=1.25: by flag", [("news", big & sig.news), ("post_earn", big & sig.post_earn), ("earn_soon only", big & sig.earn_soon & ~sig.news & ~sig.post_earn),
      ("clean", big & ~sig.news & ~sig.post_earn & ~sig.earn_soon)])
rv = sig.regime.copy()
hi = (rv["spy_rv21"] > rv["spy_rv21"].rolling(252, min_periods=100).median()).to_numpy()[:, None] & np.ones((1, N), bool)
table("Among |s|>=1.25: by market vol regime (SPY rv21 vs trailing-year median)", [("high vol", big & hi), ("low vol", big & ~hi)])
ab = np.abs(sig.b_spy)
table("Among |s|>=1.25: by side", [("long (s<-1.25)", big & (s < 0)), ("short (s>1.25)", big & (s > 0))])
