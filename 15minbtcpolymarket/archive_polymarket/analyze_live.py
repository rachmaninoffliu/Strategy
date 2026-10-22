"""Analyse live_*.csv: (1) Kalshi<->Polymarket-US cross-venue gap NET of fees, (2) persistence, (3) spot->quote lag."""
import glob
import sys

import numpy as np
import pandas as pd

from strategy import fee_pc

pd.set_option("display.width", 250)
files = sorted(glob.glob("data/live_*.csv"))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d = d.dropna(subset=["k_ybid", "k_yask", "cb_px"]).copy()
d["minute_in"] = ((d.t - d.ws) // 60).astype(int)
print("ticks", len(d), "windows", d.ws.nunique(), "span", round((d.t.max() - d.t.min()) / 60, 1), "min")
print("poly coverage:", d.p_ybid.notna().mean().round(3), "| poly quote age median (s):", round((d.t - d.p_t).median(), 2))


def poly_fee(p):                  # Polymarket US taker: 0.0695 * p * (1-p)  (per contract, continuous)
    return 0.0695 * p * (1 - p)


x = d.dropna(subset=["p_ybid", "p_yask"]).copy()
x = x[(x.t - x.p_t) <= 3.0]       # require a fresh Polymarket quote
# Lock-in cost of a $1 payout (buy YES on A at ask + buy NO on B at 1-bid_B), all taker, fees included
x["cost_KyesPno"] = x.k_yask + fee_pc(x.k_yask) + (1 - x.p_ybid) + poly_fee(1 - x.p_ybid)
x["cost_PyesKno"] = x.p_yask + poly_fee(x.p_yask) + (1 - x.k_ybid) + fee_pc(1 - x.k_ybid)
x["net"] = 1 - np.minimum(x.cost_KyesPno, x.cost_PyesKno)
x["gross"] = 1 - np.minimum(x.k_yask + 1 - x.p_ybid, x.p_yask + 1 - x.k_ybid)
x["dir"] = np.where(x.cost_KyesPno <= x.cost_PyesKno, "K-yes/P-no", "P-yes/K-no")
x["mid_k"] = (x.k_ybid + x.k_yask) / 2
x["cap"] = np.where(x.dir == "K-yes/P-no", np.minimum(x.k_yask_sz, x.p_ybid_sz), np.minimum(x.p_yask_sz, x.k_ybid_sz))
print("\n== cross-venue lock-in, TAKER on both legs ==")
print(f"gross edge before fees: mean {x.gross.mean()*100:+.2f}c  p90 {x.gross.quantile(.9)*100:+.2f}c  max {x.gross.max()*100:+.2f}c")
print(f"net after both fees   : mean {x.net.mean()*100:+.2f}c  p90 {x.net.quantile(.9)*100:+.2f}c  max {x.net.max()*100:+.2f}c")
pos = x[x.net > 0]
print(f"ticks with net>0: {len(pos)} / {len(x)} = {len(pos)/len(x):.2%}")
if len(pos):
    print(pos.groupby("dir").agg(n=("net", "size"), net_c=("net", lambda s: s.mean() * 100), cap=("cap", "median"), mid=("mid_k", "mean")).round(3))
    # persistence: consecutive-tick runs (same window) with net>0
    x = x.sort_values("t")
    x["run"] = ((x.net > 0) != (x.net.shift() > 0)) | (x.ws != x.ws.shift())
    x["rid"] = x.run.cumsum()
    runs = x[x.net > 0].groupby("rid").agg(len=("net", "size"), net_c=("net", lambda s: s.mean() * 100), cap=("cap", "min"), mid=("mid_k", "mean"), t=("t", "min"))
    print("\nrun lengths (ticks~seconds) of net>0 episodes:", runs.len.describe().round(2).to_dict())
    print("episodes >=2 ticks:", (runs.len >= 2).sum(), " >=3 ticks:", (runs.len >= 3).sum())
    print(runs.sort_values("net_c", ascending=False).head(10).round(3))
print("\nnet edge by quoted-price zone (Kalshi mid):")
x["zone"] = pd.cut(x.mid_k, [0, .1, .3, .7, .9, 1.0])
print(x.groupby("zone", observed=True).agg(n=("net", "size"), net_c=("net", lambda s: s.mean() * 100), frac_pos=("net", lambda s: (s > 0).mean()),
                                           gross_c=("gross", lambda s: s.mean() * 100)).round(3))

# (3) lead-lag: Coinbase 1s log-return vs Kalshi mid change (1 Hz grid, within window)
d = d.sort_values("t")
d["sec"] = d.t.round().astype(int)
g = d.drop_duplicates("sec").set_index("sec")
g["mid"] = (g.k_ybid + g.k_yask) / 2
g["lcb"] = np.log(g.cb_px)
g["ws2"] = g.ws
rows = []
for k in range(0, 9):
    a = g.lcb.diff()
    b = g.mid.diff().shift(-k)                 # Kalshi change k seconds AFTER the spot move
    ok = (g.ws2 == g.ws2.shift(-k)) & a.notna() & b.notna() & (g.mid.between(0.1, 0.9))
    rows.append((k, ok.sum(), np.corrcoef(a[ok], b[ok])[0, 1]))
print("\nspot->Kalshi lead-lag (corr of Coinbase 1s return with Kalshi mid change k seconds later, mid in 0.1-0.9):")
print(pd.DataFrame(rows, columns=["lag_s", "n", "corr"]).round(4).to_string(index=False))

# (4) latency-adjusted execution: signal on tick i (net > thr), fill on the quotes `lag` seconds later, same direction
print("\n== latency-adjusted cross-venue arb (signal at t, execute at quotes t+lag, same direction, TAKER both legs) ==")
x = x.sort_values("t").reset_index(drop=True)
x["sec"] = x.t.round().astype(int)
by = x.drop_duplicates("sec").set_index("sec")
for lag in (1, 2, 3, 5):
    for thr in (0.003, 0.006):
        sig = by[(by.net > thr)]
        idx = sig.index + lag
        ok = idx.isin(by.index)
        s0 = sig[ok]; s1 = by.loc[idx[ok]]
        same_win = (s0.ws.values == s1.ws.values)
        s0, s1 = s0[same_win], s1[same_win]
        cost = np.where(s0.dir.values == "K-yes/P-no", s1.cost_KyesPno.values, s1.cost_PyesKno.values)
        net_exec = 1 - cost
        cap = np.where(s0.dir.values == "K-yes/P-no", np.minimum(s1.k_yask_sz.values, s1.p_ybid_sz.values), np.minimum(s1.p_yask_sz.values, s1.k_ybid_sz.values))
        cap = np.minimum(cap, s0.cap.values)
        usd = net_exec * np.minimum(cap, 500)
        hours = (d.t.max() - d.t.min()) / 3600
        print(f"lag {lag}s thr {thr*100:.1f}c: signals {len(s0):5d} | exec net mean {net_exec.mean()*100:+.2f}c | P(net_exec>0) {np.mean(net_exec>0):.2%} | "
              f"median cap {np.median(cap):6.0f} | $ per hr if 500-ct max/signal-tick-dedup~: {usd.sum()/hours/ max(1,np.median(pd.Series(s0.index.values).diff().dropna().clip(1,30)) ) :8.1f}")
