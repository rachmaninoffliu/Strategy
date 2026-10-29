import sys, pickle, os
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
from scipy.stats import rankdata
import statarb_data as D

SP = os.path.dirname(os.path.abspath(__file__))
panel = D.load_panel()
sig = pickle.load(open(os.path.join(SP, f"sig_{sys.argv[1] if len(sys.argv)>1 else 'pca15'}.pkl"), "rb"))
O = panel.open.to_numpy(float); Cc = panel.close.to_numpy(float)
T, N = O.shape
sec = sig.sector_idx

def shift_ratio(A, B, k_a, k_b):
    """A[d+k_a]/B[d+k_b]-1, nan when out of range"""
    out = np.full((T, N), np.nan)
    for d in range(T):
        if d + k_a < T and d + k_b < T:
            out[d] = A[d + k_a] / B[d + k_b] - 1
    out[np.abs(out) > 0.9] = np.nan
    return out

def demean(x):
    y = x.copy()
    for k in np.unique(sec):
        m = sec == k
        y[:, m] = x[:, m] - np.nanmedian(x[:, m], axis=1, keepdims=True)
    return y

def ic(sig_arr, fwd_arr, mask, since="2017-06-01"):
    out = []
    for d in range(T):
        if sig.dates[d] < pd.Timestamp(since):
            continue
        m = mask[d] & np.isfinite(sig_arr[d]) & np.isfinite(fwd_arr[d])
        if m.sum() > 60:
            out.append(np.corrcoef(rankdata(sig_arr[d][m]), rankdata(fwd_arr[d][m]))[0, 1])
    out = np.array(out)
    return out.mean(), out.mean() / out.std() * np.sqrt(len(out)), len(out)

s = -sig.s
base = sig.univ
rows = []
fw = {
    "C[d]->C[d+1]   (fill at signal close)": demean(shift_ratio(Cc, Cc, 1, 0)),
    "C[d]->C[d+3]": demean(shift_ratio(Cc, Cc, 3, 0)),
    "C[d]->C[d+5]": demean(shift_ratio(Cc, Cc, 5, 0)),
    "C[d]->O[d+1]   (overnight gap only)": demean(shift_ratio(O, Cc, 1, 0)),
    "O[d+1]->C[d+1] (next-day intraday)": demean(shift_ratio(Cc, O, 1, 1)),
    "O[d+1]->O[d+2] (my backtest, 1d)": demean(shift_ratio(O, O, 2, 1)),
    "O[d+1]->O[d+4] (3d)": demean(shift_ratio(O, O, 4, 1)),
    "O[d+1]->O[d+6] (5d)": demean(shift_ratio(O, O, 6, 1)),
}
for name, F in fw.items():
    m, t, n = ic(s, F, base)
    rows.append((name, m, t))
print("ALL liquid names, IC of -s vs sector-demeaned forward return")
print(pd.DataFrame(rows, columns=["forward window", "mean IC", "t"]).round(4).to_string(index=False))

print("\nsubsets (O[d+1]->O[d+4], 3d):")
F3 = fw["O[d+1]->O[d+4] (3d)"]
for name, mask in (("news only", base & sig.news), ("post_earn only", base & sig.post_earn), ("earn_soon only", base & sig.earn_soon & ~sig.news & ~sig.post_earn),
                   ("clean", base & ~sig.news & ~sig.post_earn & ~sig.earn_soon), ("|s|>1.25", base & (np.abs(sig.s) > 1.25)), ("|s|>2", base & (np.abs(sig.s) > 2.0))):
    m, t, n = ic(s, F3, mask)
    print(f"  {name:18s} IC {m:+.4f}  t {t:5.2f}  days {n}  frac {mask.sum()/base.sum():.3f}")
