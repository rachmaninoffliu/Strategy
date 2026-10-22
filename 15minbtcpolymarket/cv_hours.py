"""Construct-validity check of the synthetic world: for each UTC hour, compare the REAL Kalshi underdog EV with the synthetic one (same windows, price = g + half spread),
Dec 15 2025 -> Aug 23 2026 (no regime) and Aug 24 -> now.  If they diverge, g is missing something the real market prices (e.g. intraday vol seasonality)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import cv_longspot as lsp, cv_mechanism as mech
from trader.data import load_rows
from trader.fees import fee_per_contract

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
s = lsp.load_spot_all(); w = lsp.build_windows(s, minutes=(12, 13)); g = mech.fit_g(w)
w["g"] = 0.5
for mn in g:
    i = w.minute == mn; w.loc[i, "g"] = g[mn].predict(w.loc[i, "z"].values)
q = load_rows(); q = q[q.minute.isin([12, 13]) & (q.bid > 0) & (q.ask < 1) & (q.ask >= q.bid) & ((q.ask - q.bid) <= 0.06 + 1e-9)].copy()
m = q.merge(w[["open_ts", "minute", "g", "yes_proxy", "z"]], on=["open_ts", "minute"])
m["mid"] = (m.bid + m.ask) / 2
dog = m.mid < 0.5
m["cost_real"] = np.where(dog, m.ask, 1 - m.bid)
m["g_dog"] = np.where(dog, m.g, 1 - m.g)
m["win"] = np.where(dog, m.yes, 1 - m.yes)
m["win_p"] = np.where(dog, m.yes_proxy, 1 - m.yes_proxy)
m["cost_syn"] = m.g_dog + 0.005
m["hour"] = pd.to_datetime(m.open_ts, unit="s", utc=True).dt.hour
m = m[(m.cost_real >= 0.05) & (m.cost_real <= 0.30) & (m.cost_syn >= 0.05) & (m.cost_syn <= 0.30)].sort_values(["open_ts", "minute"]).drop_duplicates("ticker")
fee = lambda c: np.array([fee_per_contract("Polymarket US", float(x)) for x in np.round(np.minimum(c + 0.02, 0.99), 3)])
m["ev_real"] = m.win - np.minimum(m.cost_real + 0.02, 0.99) - fee(m.cost_real.values)
m["ev_syn"] = m.win_p - np.minimum(m.cost_syn + 0.02, 0.99) - fee(m.cost_syn.values)
m["price_gap_c"] = (m.cost_real - m.cost_syn) * 100
for lab, a, b in (("Dec15-Aug23 (no regime)", 0, lsp.CUT), ("Aug24-now", lsp.CUT, 1e12)):
    x = m[(m.open_ts >= a) & (m.open_ts < b)]
    t = x.groupby("hour").agg(n=("ev_real", "size"), real_price_minus_g_c=("price_gap_c", "mean"), real_ev_c=("ev_real", lambda v: v.mean() * 100), syn_ev_c=("ev_syn", lambda v: v.mean() * 100))
    t["diff_c"] = t.syn_ev_c - t.real_ev_c
    print(f"\n=== {lab}: underdogs 5-30c, minutes 12/13, per UTC hour (real quotes vs synthetic g-pricing) ===")
    print(t.round(1).T.to_string())
    print(f"   all hours: real {x.ev_real.mean()*100:+.2f}c  synthetic {x.ev_syn.mean()*100:+.2f}c | corr over hours (real,syn) = {np.corrcoef(t.real_ev_c, t.syn_ev_c)[0,1]:+.2f}")
x = m[(m.open_ts < lsp.CUT) & m.hour.isin([13, 14])]
print(f"\nhours 13-14, Dec15-Aug23: real EV {x.ev_real.mean()*100:+.2f}c (t {x.ev_real.mean()/x.ev_real.std()*np.sqrt(len(x)):+.2f}, n={len(x)}) vs synthetic {x.ev_syn.mean()*100:+.2f}c")
