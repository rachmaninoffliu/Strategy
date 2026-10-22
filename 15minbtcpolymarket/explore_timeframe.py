"""Was 22-23 UTC / minute 12-13 a sensible choice?  Systematic scan on the complete data (Jun 1 -> now), gate-independent:
EV per signal = win - (price + slippage) - Polymarket US fee, one signal per market, underdog leg.
  1. every single UTC hour           2. every decision minute inside 22-23 UTC and, for contrast, all other hours
  3. sliding windows of 1/2/3/4/6 hours   4. permutation test: how often does chance alone produce a window this good?
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.data import extract_signals, load_rows
from trader.fees import fee_per_contract
from trader.strategy import ALL_HOURS, StrategyParams

SLIP = 0.02
BAND = (0.15, 0.35)
rows = load_rows()
pd.set_option("display.width", 230)
pd.set_option("display.max_rows", 200)


def sigs(minutes, band=BAND, hours=ALL_HOURS):
    s = extract_signals(rows, StrategyParams(hours=hours, minutes=tuple(minutes), cost_min=band[0], cost_max=band[1]))
    px = np.minimum(s.cost + SLIP, 0.99)
    s["pnl"] = s.win - px - px.apply(lambda c: fee_per_contract("Polymarket US", c))
    s["date"] = pd.to_datetime(s.open_ts, unit="s", utc=True).dt.date
    return s


def stat(x):
    n = len(x)
    if n < 5:
        return dict(n=n, win=np.nan, price=np.nan, ev_c=np.nan, t=np.nan)
    return dict(n=n, win=x.win.mean(), price=x.cost.mean(), ev_c=x.pnl.mean() * 100, t=x.pnl.mean() / x.pnl.std() * np.sqrt(n))


# ── 1. every UTC hour (minutes 12-13, band 15-35c) ────────────────────────────
s = sigs([12, 13])
h = pd.DataFrame({hr: stat(g) for hr, g in s.groupby("hour")}).T
print("=== 1. EV per signal by UTC hour of window open (decision minutes 12-13, underdog 15-35c, 2c slippage) ===")
print(h.round(3).T.to_string())
print(f"hours with EV > 0: {(h.ev_c > 0).sum()} of 24 | with t > 2: {(h.t > 2).sum()} | with t < -2: {(h.t < -2).sum()}")

# ── 2. every decision minute ──────────────────────────────────────────────────
print("\n=== 2. EV per signal by DECISION MINUTE (single-minute rule), 22-23 UTC vs all other hours ===")
res = []
for m in range(2, 15):
    a = sigs([m])
    inn, out = a[a.hour.isin([22, 23])], a[~a.hour.isin([22, 23])]
    si, so = stat(inn), stat(out)
    res.append((m, si["n"], si["win"], si["price"], si["ev_c"], si["t"], so["n"], so["ev_c"], so["t"]))
print(pd.DataFrame(res, columns=["minute", "n(22-23)", "win", "price", "EV c", "t", "n(other)", "EV c other", "t other"]).round(2).to_string(index=False))

# ── 3. sliding windows of different lengths (cyclic over the day) ─────────────
print("\n=== 3. Best contiguous UTC-hour windows (minutes 12-13, 15-35c): EV per signal ===")
by_hour = {hr: g for hr, g in s.groupby("hour")}
out = []
for L in (1, 2, 3, 4, 6):
    for start in range(24):
        hrs = [(start + k) % 24 for k in range(L)]
        x = pd.concat([by_hour[hh] for hh in hrs if hh in by_hour])
        st = stat(x)
        out.append((L, f"{hrs[0]:02d}-{hrs[-1]:02d}", st["n"], st["win"], st["ev_c"], st["t"]))
W = pd.DataFrame(out, columns=["len_h", "hours", "n", "win", "EV c", "t"])
for L in (1, 2, 3, 4, 6):
    print(f"\nlength {L}h - top 4 by EV (and where 22-23 ranks):")
    d = W[W.len_h == L].sort_values("EV c", ascending=False)
    print(d.head(4).round(2).to_string(index=False))
    if L == 2:
        print(f"   rank of 22-23 among 24 two-hour windows: {list(d.hours).index('22-23') + 1}")

# ── 4. permutation test for 'hot window found by searching' ───────────────────
print("\n=== 4. Permutation test: if hour-of-day carried no information, how often would the BEST 2-hour window look this good? ===")
obs_best = W[W.len_h == 2]["EV c"].max()
obs_2223 = float(W[(W.len_h == 2) & (W.hours == "22-23")]["EV c"].iloc[0])
rng = np.random.default_rng(0)
arr_h, arr_pnl, arr_date = s.hour.values, s.pnl.values, s.date.values
dates = pd.unique(arr_date)
idx_by_date = {d: np.where(arr_date == d)[0] for d in dates}
best = []
for _ in range(1000):
    shift = {d: int(rng.integers(0, 24)) for d in dates}                         # cyclic shift of the hour labels within each day
    hh = np.empty_like(arr_h)
    for d, ix in idx_by_date.items():
        hh[ix] = (arr_h[ix] + shift[d]) % 24
    means = []
    for start in range(24):
        sel = (hh == start) | (hh == (start + 1) % 24)
        if sel.sum() >= 20:
            means.append(arr_pnl[sel].mean() * 100)
    best.append(max(means))
best = np.array(best)
print(f"observed best 2-hour window EV = {obs_best:.2f}c (22-23: {obs_2223:.2f}c)")
print(f"under shuffled hour labels the best 2-hour window EV: median {np.median(best):.2f}c, 95th pct {np.percentile(best, 95):.2f}c, 99th pct {np.percentile(best, 99):.2f}c")
print(f"P(chance gives a window >= observed best) = {(best >= obs_best).mean():.3f}")

# ── 5. is 22-23 stable, or only the burst?  (EV by 3 periods, 22-23 vs neighbours) ─
print("\n=== 5. Stability of the hot window vs its neighbours (EV c / n) ===")
P = lambda d: pd.Timestamp(d, tz="UTC").timestamp()
per = [("Jun1-Aug23", P("2026-06-01"), P("2026-08-24")), ("Aug24-Sep27", P("2026-08-24"), P("2026-09-28")), ("Sep28-now", P("2026-09-28"), 1e12)]
rows5 = []
for name, hrs in (("18-19", [18, 19]), ("20-21", [20, 21]), ("22-23", [22, 23]), ("00-01", [0, 1]), ("02-03", [2, 3]), ("all other hours", [h2 for h2 in range(24) if h2 not in (22, 23)])):
    x = s[s.hour.isin(hrs)]
    rec = {"hours": name}
    for lab, a, b in per:
        y = x[(x.open_ts >= a) & (x.open_ts < b)]
        rec[lab] = f"{y.pnl.mean() * 100 if len(y) else float('nan'):+.1f}c / {len(y)}"
    rows5.append(rec)
print(pd.DataFrame(rows5).to_string(index=False))
