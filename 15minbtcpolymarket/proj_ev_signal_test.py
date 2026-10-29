import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from datetime import date, timedelta
from trader.data import load_rows, extract_signals, data_range
from trader.strategy import StrategyParams
from trader.fees import fee_per_contract
from trader.live import build_stats
from trader.presets import preset_config
from trader.replay import live_config_from
rows = load_rows()
end_dt = data_range(rows)[1]
now = end_dt.timestamp()
lc = live_config_from(preset_config("PROJ-EV", date(2026, 10, 1), date(2026, 10, 22)))
st = build_stats(rows, lc)
p = lc.params
sig = extract_signals(rows, p).reset_index(drop=True)
px = np.minimum(sig.cost.values + 0.02, 0.99)
fee = np.array([fee_per_contract("Kalshi", float(c)) for c in px])
sig["net"] = sig.win.values - px - fee
sig["entry_ts"] = sig.open_ts + 60 * sig.minute
T0 = int(pd.Timestamp("2026-06-01", tz="UTC").timestamp())
c = sig[sig.open_ts >= T0].copy()
pp = np.array([st.p("DOG", int(h), float(co), float(ts))[0] for h, co, ts in zip(c.hour.values, c.cost.values, c.entry_ts.values)])
pxc = np.minimum(c.cost.values + 0.02, 0.99)
edge = pp - (pxc + np.array([fee_per_contract("Kalshi", float(x)) for x in pxc]))
c["proj_edge"] = edge
c["pass"] = edge > 0
periods = {"Jun1-Aug23 (dead)": (T0, int(pd.Timestamp("2026-08-24", tz="UTC").timestamp())), "Aug24-Sep27 (hot)": (int(pd.Timestamp("2026-08-24", tz="UTC").timestamp()), int(pd.Timestamp("2026-09-28", tz="UTC").timestamp())),
           "Sep28-now": (int(pd.Timestamp("2026-09-28", tz="UTC").timestamp()), 10**12), "last 14d": (int(now) - 14 * 86400, 10**12), "Jun1-now": (T0, 10**12)}
def tstat(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 2 and len(b) > 2 else np.nan
print("flat 1-contract net EV of signals whose projected EV is > 0 (PASS) vs <= 0 (FAIL) at entry")
for pn, (a, b) in periods.items():
    x = c[(c.open_ts >= a) & (c.open_ts < b)]
    pa, fa = x[x["pass"]].net, x[~x["pass"]].net
    print(f"   {pn:20s} pass n={len(pa):4d} EV {pa.mean() * 100:+6.2f}c (se {pa.std() / np.sqrt(max(len(pa), 1)) * 100:4.1f}) | fail n={len(fa):4d} EV {fa.mean() * 100:+6.2f}c | diff t={tstat(pa, fa):+.2f} | mean projected edge of passes {x[x['pass']].proj_edge.mean() * 100:+.2f}c")
oth = c[~c.hour.isin([22, 23])]
pa, fa = oth[oth["pass"]].net, oth[~oth["pass"]].net
print(f"   OTHER 22 hours Jun1-now: pass n={len(pa)} EV {pa.mean() * 100:+.2f}c | fail n={len(fa)} EV {fa.mean() * 100:+.2f}c | diff t={tstat(pa, fa):+.2f}; share of passes in 22-23: {c[c['pass']].hour.isin([22, 23]).mean() * 100:.0f}%")
print(f"\nsignals passing per day (Jun1-now): {c['pass'].sum() / ((c.open_ts.max() - T0) / 86400):.1f}")
# right now
typ = 0.25; pxt = typ + 0.02; ca = pxt + fee_per_contract("Kalshi", pxt)
rows_now = []
for h in range(24):
    pr, n, nh = st.p("DOG", h, typ, now)
    rows_now.append((h, (pr - ca) * 100))
pos = [(h, e) for h, e in rows_now if e > 0]
print("\nprojected EV per hour NOW (net, at a 25c signal), data through", end_dt)
print("  positive:", ", ".join(f"{h:02d}h {e:+.2f}c" for h, e in pos) or "none")
print("  least negative:", ", ".join(f"{h:02d}h {e:+.2f}c" for h, e in sorted(rows_now, key=lambda x: -x[1])[:6]))
