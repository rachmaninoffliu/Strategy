"""Does the radar's shrunk estimate or the raw trailing EV predict an hour's NEXT-H-day net EV better?  Panel of (evaluation day T, UTC hour h), point-in-time, no look-ahead."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from trader.data import load_rows, extract_signals
from trader.strategy import StrategyParams
from trader.probmodel import SignalStats
from trader.fees import fee_per_contract
from trader import radar

rows = load_rows()
p = StrategyParams(hours=tuple(range(24)), minutes=(12, 13), cost_min=0.15, cost_max=0.35, legs=("DOG",))
sig = extract_signals(rows, p).reset_index(drop=True)
slip = 0.02
px = np.minimum(sig.cost.values + slip, 0.99)
sig["net"] = sig.win.values - px - np.array([fee_per_contract("Kalshi", float(c)) for c in px])
typ = 0.25
pxt = typ + slip
cost_all = pxt + fee_per_contract("Kalshi", pxt)
end_ts = int(sig.close_ts.max())
H = int(sys.argv[1]) if len(sys.argv) > 1 else 14
import os
T0 = int(pd.Timestamp(os.environ.get("T0", "2026-03-01"), tz="UTC").timestamp())
Ts = list(range(T0, end_ts - H * 86400, 86400))
CORE = {22, 23}; NEAR = {20, 21, 0, 1}

def stats_for(k, hl, core_k=0.0, near_k=0.0, core=()):
    return SignalStats.from_signals(sig, mode="by_hour", hour_k=k, prior_n=20.0, halflife_days=hl, basis="edge", core_hours=core, core_k=core_k, near_k=near_k, near_width=2)

combos = {"app default (k=100 scout, hl 14)": (100.0, 14.0), "k=30, hl 14": (30.0, 14.0), "k=10, hl 14": (10.0, 14.0), "k=30, hl 7": (30.0, 7.0), "k=30, hl 30": (30.0, 30.0), "k=5, hl 14": (5.0, 14.0)}
S = {name: stats_for(k, hl) for name, (k, hl) in combos.items()}

recs = []
for T in Ts:
    s_ok = sig[sig.close_ts <= T]
    w28 = s_ok[s_ok.open_ts >= T - 28 * 86400]
    w14 = s_ok[s_ok.open_ts >= T - 14 * 86400]
    fw = sig[(sig.open_ts >= T) & (sig.close_ts <= T + H * 86400)]
    for h in range(24):
        f = fw[fw.hour == h]
        if len(f) < 4:
            continue
        a, b = w28[w28.hour == h], w14[w14.hour == h]
        if len(a) < 10:
            continue
        r = dict(T=T, h=h, n28=len(a), raw28=a.net.mean(), raw14=b.net.mean() if len(b) >= 4 else np.nan, fwd=f.net.mean(), nf=len(f))
        for name, st in S.items():
            r[name] = st.p("DOG", h, typ, T)[0] - cost_all
        recs.append(r)
P = pd.DataFrame(recs)
print(f"panel: {len(P)} (day,hour) points, forward window {H} d, evaluation days {len(Ts)}, hours with data")
def report(sub, title):
    print(f"\n=== {title}: {len(sub)} points; mean forward net EV {sub.fwd.mean() * 100:+.2f}c; mean raw28 {sub.raw28.mean() * 100:+.2f}c")
    print(f"{'predictor':36s} {'corr':>6s} {'slope':>6s} {'MSE':>8s}   (MSE of predicting forward EV, in c^2; lower = better)")
    cols = ["raw28", "raw14"] + list(combos)
    base = ((sub.fwd - sub.fwd.mean()) ** 2).mean() * 1e4
    print(f"{'(constant: mean forward EV)':36s} {'':6s} {'':6s} {base:8.1f}")
    for c in cols:
        d = sub.dropna(subset=[c])
        x = d[c].values; y = d.fwd.values
        corr = np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan
        slope = np.polyfit(x, y, 1)[0] if x.std() > 0 else np.nan
        mse = ((y - x) ** 2).mean() * 1e4
        print(f"{c:36s} {corr:6.3f} {slope:6.2f} {mse:8.1f}")
report(P, "all hours")
sc = P[~P.h.isin(CORE | NEAR)]
report(sc, "far scout hours (not 20-23, 0-1)")
# calibration: bucket by raw28 and by app estimate -> realised forward EV
for feat in ("raw28", "app default (k=100 scout, hl 14)"):
    q = pd.qcut(sc[feat], 5, duplicates="drop")
    g = sc.groupby(q).agg(n=("fwd", "size"), feat=(feat, "mean"), fwd=("fwd", "mean"))
    print(f"\nscout hours, quintiles of {feat}:  predicted -> realised next-{H}d net EV")
    print((g[["feat", "fwd"]] * 100).round(2).assign(n=g.n).to_string())
# high-raw cases specifically (the user's question): hours with raw28 >= +8c AND raw14 >= +8c
hi = sc[(sc.raw28 >= 0.08) & (sc.raw14 >= 0.08)]
print(f"\nscout hours with raw28 >= +8c and raw14 >= +8c: {len(hi)} points | their mean raw28 {hi.raw28.mean() * 100:+.1f}c raw14 {hi.raw14.mean() * 100:+.1f}c | engine estimate {hi['app default (k=100 scout, hl 14)'].mean() * 100:+.1f}c | REALISED next-{H}d EV {hi.fwd.mean() * 100:+.1f}c (all scout hours: {sc.fwd.mean() * 100:+.1f}c)")
