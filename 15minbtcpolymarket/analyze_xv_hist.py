"""Kalshi vs Polymarket-US minute-level comparison on ~2 weeks of overlapping history.

Q1  how big/persistent is the cross-venue price gap at minute resolution?
Q2  does the gap predict the outcome (is one venue a better predictor / does the spread carry information)?
Q3  which venue leads (does the gap close by Kalshi moving, or by Poly moving)?
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

from panel import build_panel

pd.set_option("display.width", 250)
K = build_panel()
K = K[(K.bid_c > 0) & (K.ask_c < 1) & (K.ask_c >= K.bid_c)].copy()
K["mid_k"] = (K.bid_c + K.ask_c) / 2
K["ts"] = K.open_ts + 60 * K.minute
H = pd.read_csv("data/poly_hist.csv")
H = H.drop_duplicates(["ws", "ts"])
H["ts"] = H.ts.round(-1) + 60                 # Poly label = START of the 1-min bucket, value = price at its END (verified: sd 14c -> 4.7c at +60s)
H = H.rename(columns={"ws": "open_ts", "long": "p_long", "short": "p_short"})
D = K.merge(H[["open_ts", "ts", "p_long", "p_short"]], on=["open_ts", "ts"], how="inner")
D = D[(D.p_long > 0.0) & (D.p_long < 1.0)]
print("matched rows", len(D), "markets", D.ticker.nunique(), pd.to_datetime(D.open_ts.min(), unit="s").date(), "->", pd.to_datetime(D.open_ts.max(), unit="s").date())

cl = lambda s: s.clip(0.01, 0.99)
lg = lambda s: np.log(cl(s) / (1 - cl(s)))
D["gap"] = D.p_long - D.mid_k
print("\n== Q1  gap = Poly long price - Kalshi mid (cents) ==")
print(f"mean {D.gap.mean()*100:+.2f}  sd {D.gap.std()*100:.2f}  |gap|>2c: {(D.gap.abs()>0.02).mean():.1%}   >3c: {(D.gap.abs()>0.03).mean():.1%}   >5c: {(D.gap.abs()>0.05).mean():.1%}")
D["zone"] = pd.cut(D.mid_k, [0, .1, .3, .7, .9, 1.0])
print(D.groupby("zone", observed=True).gap.agg(n="size", mean=lambda s: s.mean() * 100, sd=lambda s: s.std() * 100, p_gt3=lambda s: (s.abs() > 0.03).mean()).round(3))
print("by minute:"); print(D.groupby("minute").gap.agg(n="size", mean=lambda s: s.mean() * 100, sd=lambda s: s.std() * 100).round(3).T.to_string())

print("\n== Q2  outcome prediction (cluster-robust by market) ==")
D["lk"] = lg(D.mid_k); D["lp"] = lg(D.p_long); D["d"] = D.lp - D.lk
def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
g = D.ticker.astype("category").cat.codes
for name, cols in (("Kalshi only", ["lk"]), ("Poly only", ["lp"]), ("Kalshi + (Poly-Kalshi)", ["lk", "d"])):
    r = sm.Logit(D.yes, sm.add_constant(D[cols])).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    print(f"{name:24s} in-sample LL {ll(D.yes, r.predict(sm.add_constant(D[cols]))):.5f}  " + "  ".join(f"{k}={v:.3f}(t={r.tvalues[k]:.1f})" for k, v in r.params.items() if k != "const"))
print("raw mids:  LL(Kalshi mid) %.5f   LL(Poly long) %.5f   LL(avg) %.5f" % (ll(D.yes, D.mid_k), ll(D.yes, D.p_long), ll(D.yes, (D.mid_k + D.p_long) / 2)))

print("\n== Q3  who leads?  next-minute change of each venue regressed on today's gap ==")
D = D.sort_values(["ticker", "minute"])
D["dk_next"] = D.groupby("ticker").mid_k.shift(-1) - D.mid_k
D["dp_next"] = D.groupby("ticker").p_long.shift(-1) - D.p_long
x = D.dropna(subset=["dk_next", "dp_next"])
x = x[x.mid_k.between(0.1, 0.9)]
for name, y in (("Kalshi next-min change", x.dk_next), ("Poly next-min change", x.dp_next)):
    r = sm.OLS(y, sm.add_constant(x[["gap"]])).fit(cov_type="cluster", cov_kwds={"groups": x.ticker.astype("category").cat.codes})
    print(f"{name:24s} beta on gap = {r.params['gap']:+.3f} (t={r.tvalues['gap']:.1f})   [+ => this venue moves toward the other => it LAGS]" if name.startswith("Kalshi") else
          f"{name:24s} beta on gap = {r.params['gap']:+.3f} (t={r.tvalues['gap']:.1f})   [- => this venue moves toward the other => it LAGS]")
