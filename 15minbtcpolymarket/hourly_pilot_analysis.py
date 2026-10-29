"""Hourly BTC ladder pilot: is the market calibrated, and does a spot-vol model find edge after the fee?  Standard errors are CLUSTERED by event (all strikes of one hour share one BTC outcome)."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from math import erf, sqrt
from trader.fees import fee_per_contract
D = "data/"
h = pd.read_csv(D + "hourly_pilot.csv")
sp = pd.concat([pd.read_csv(D + f, usecols=["ts", "close"]) for f in ("cb_1m.csv", "cb_1m_recent.csv")]).drop_duplicates("ts").sort_values("ts")
sp = sp.set_index(sp.ts.astype("int64") // 60 * 60).close
grid = pd.RangeIndex(int(sp.index.min()), int(sp.index.max()) + 60, 60)
sp = sp.reindex(grid).ffill()
lc = np.log(sp.values.astype(float)); r1 = pd.Series(np.r_[0.0, np.diff(lc)], index=sp.index)
sig120 = r1.rolling(120).std(); sig30 = r1.rolling(30).std(); sig1440 = r1.rolling(1440).std()
obs_ts = (h.close_ts.values - h.off.values * 60 - 60)
h["sig120"] = sig120.reindex(obs_ts).values; h["sig30"] = sig30.reindex(obs_ts).values; h["sig1440"] = sig1440.reindex(obs_ts).values
h = h.dropna(subset=["spot", "sig120", "bid", "ask"])
h = h[(h.ask >= h.bid) & ((h.ask - h.bid) <= 0.05) & (h.ask > 0) & (h.bid >= 0)].copy()
h["mid"] = (h.bid + h.ask) / 2
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))
SLIP = 0.005
def fee(c): return np.array([fee_per_contract("Kalshi", float(x)) for x in np.minimum(c + SLIP, 0.99)])
h["fee_yes"] = fee(h.ask.values); h["fee_no"] = fee((1 - h.bid).values)
h["net_yes"] = h.yes - np.minimum(h.ask + SLIP, 0.99) - h.fee_yes
h["net_no"] = (1 - h.yes) - np.minimum(1 - h.bid + SLIP, 0.99) - h.fee_no
h["d"] = np.log(h.spot / h.strike)
def cluster_t(df, col):
    g = df.groupby("event")[col].agg(["sum", "size"])
    m = df[col].mean(); n = len(df)
    # cluster-robust se of the mean
    resid = df[col] - m
    s = resid.groupby(df.event).sum()
    se = np.sqrt((s ** 2).sum()) / n
    return m, se, m / se if se > 0 else np.nan
print(f"observations {len(h)}, events {h.event.nunique()}, offsets {sorted(h.off.unique())}, period {pd.to_datetime(h.close_ts.min(), unit='s')} -> {pd.to_datetime(h.close_ts.max(), unit='s')}")
print("\n(A) calibration: buy YES at the ask / buy NO at 1-bid, by YES-price bucket and minutes to expiry (net of fee + 0.5c slippage; cluster-robust t)")
h["pb"] = pd.cut(h.mid, [0, .03, .08, .2, .4, .6, .8, .92, .97, 1])
rows = []
for off in sorted(h.off.unique()):
    x = h[h.off == off]
    for b, g in x.groupby("pb"):
        if len(g) < 30: continue
        my, sy, ty = cluster_t(g, "net_yes"); mn, sn, tn = cluster_t(g, "net_no")
        rows.append(dict(off=off, bucket=str(b), n=len(g), mid=g.mid.mean() * 100, win_yes=g.yes.mean() * 100, buyYES_c=my * 100, t_yes=ty, buyNO_c=mn * 100, t_no=tn))
A = pd.DataFrame(rows)
print(A.round(2).to_string(index=False))
print("\n(B) model: p_fair = Phi(ln(S/K) / (sigma_120 * sqrt(minutes))); take the better side when its model edge (before fee) exceeds a hurdle")
for off in sorted(h.off.unique()):
    x = h[h.off == off].copy()
    x["pf"] = Phi((x.d / (x.sig120 * np.sqrt(off))).values)
    x["e_yes"] = x.pf - x.ask; x["e_no"] = x.bid - x.pf
    x["side"] = np.where(x.e_yes >= x.e_no, "yes", "no"); x["edge"] = np.maximum(x.e_yes, x.e_no)
    x["net"] = np.where(x.side == "yes", x.net_yes, x.net_no)
    for hur in (0.0, 0.03, 0.06, 0.10):
        g = x[x.edge > hur]
        if len(g) < 40: continue
        m, se, t = cluster_t(g, "net")
        print(f"  off {off:2d}m  edge>{hur * 100:>2.0f}c: n={len(g):5d} events={g.event.nunique():3d}  net EV {m * 100:+6.2f}c  (cluster se {se * 100:.2f}, t {t:+.2f})  mean model edge {g.edge.mean() * 100:.1f}c  mean price {np.where(g.side == 'yes', g.ask, 1 - g.bid).mean() * 100:.0f}c")
print("\n(C) calibration of the model itself: mean(yes - p_fair) by p_fair bucket, off=60/30")
for off in (60, 30):
    x = h[h.off == off].copy(); x["pf"] = Phi((x.d / (x.sig120 * np.sqrt(off))).values)
    x["pfb"] = pd.cut(x.pf, [0, .05, .2, .4, .6, .8, .95, 1])
    g = x.groupby("pfb").agg(n=("yes", "size"), pf=("pf", "mean"), mid=("mid", "mean"), win=("yes", "mean"))
    print(f"  off {off}m:"); print((g[["pf", "mid", "win"]] * 100).round(1).assign(n=g.n).to_string())
