"""Sizing-free test of the three EV-regime rules: flat 1 contract per signal, rule evaluated point-in-time at each signal's entry. Does 'passes the filter' predict a higher net EV?"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from trader.data import load_rows, extract_signals, data_range
from trader.strategy import StrategyParams
from trader.fees import fee_per_contract
from trader.evfilter import TrailingEVStats
rows = load_rows()
p = StrategyParams(hours=tuple(range(24)), minutes=(12, 13), cost_min=0.15, cost_max=0.35, legs=("DOG",))
sig = extract_signals(rows, p).reset_index(drop=True)
px = np.minimum(sig.cost.values + 0.02, 0.99)
sig["net"] = sig.win.values - px - np.array([fee_per_contract("Kalshi", float(c)) for c in px])
sig["entry_ts"] = sig.open_ts + 60 * sig.minute
rules = {"FILTER-14 (14d>10c)": ((14, 0.10),), "FILTER-28 (28d>5c)": ((28, 0.05),), "FILTER-BOTH": ((14, 0.10), (28, 0.05))}
T0 = int(pd.Timestamp("2026-06-01", tz="UTC").timestamp())
S = {n: TrailingEVStats.from_signals(sig, r, 8, 0.02, "Kalshi") for n, r in rules.items()}
cut = sig[sig.open_ts >= T0].copy()
for n, st in S.items():
    cut[n] = [st.passes(int(h), float(ts)) for h, ts in zip(cut.hour.values, cut.entry_ts.values)]
periods = {"Jun1-Aug23 (dead)": (T0, int(pd.Timestamp("2026-08-24", tz="UTC").timestamp())), "Aug24-Sep27 (hot)": (int(pd.Timestamp("2026-08-24", tz="UTC").timestamp()), int(pd.Timestamp("2026-09-28", tz="UTC").timestamp())),
           "Sep28-now": (int(pd.Timestamp("2026-09-28", tz="UTC").timestamp()), 10**12), "Jun1-now": (T0, 10**12), "last 14d": (int(data_range(rows)[1].timestamp()) - 14 * 86400, 10**12)}
def tstat(a, b):
    if len(a) < 3 or len(b) < 3: return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print("flat 1-contract net EV (c) of signals that PASS vs FAIL the rule at entry; t = pass minus fail\n")
for name in rules:
    print("==", name)
    for pn, (a, b) in periods.items():
        x = cut[(cut.open_ts >= a) & (cut.open_ts < b)]
        pa, fa = x[x[name]].net, x[~x[name]].net
        print(f"   {pn:20s} pass n={len(pa):4d} EV {pa.mean() * 100:+6.2f}c (se {pa.std() / np.sqrt(max(len(pa), 1)) * 100:4.1f}) | fail n={len(fa):4d} EV {fa.mean() * 100:+6.2f}c | all n={len(x):4d} EV {x.net.mean() * 100:+6.2f}c | diff t={tstat(pa, fa):+.2f}")
# are the passing signals just the core hours?
print("\nshare of passing signals in hours 22-23 (Jun1-now):", {n: round(float(cut[cut[n]].hour.isin([22, 23]).mean()) * 100, 0) for n in rules})
print("signals passing per day (Jun1-now), mean:", {n: round(float(cut[n].sum() / ((cut.open_ts.max() - T0) / 86400)), 1) for n in rules})
# excluding 22-23: does the rule still add anything in the other 22 hours?
oth = cut[~cut.hour.isin([22, 23])]
print("\nOTHER 22 hours only (Jun1-now):")
for name in rules:
    pa, fa = oth[oth[name]].net, oth[~oth[name]].net
    print(f"   {name:22s} pass n={len(pa):4d} EV {pa.mean() * 100:+6.2f}c | fail n={len(fa):4d} EV {fa.mean() * 100:+6.2f}c | diff t={tstat(pa, fa):+.2f}")
