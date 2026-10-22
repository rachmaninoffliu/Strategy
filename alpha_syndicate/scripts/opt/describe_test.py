"""Descriptive only (frozen params, no selection): yearly returns + Sharpe significance on Test."""
import numpy as np, pandas as pd, json, math
from sim import *
P = json.load(open(r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt\frozen_params.json"))["params"]
TW = ("2022-01-01", "2026-10-06")
for a, (tr, px, src) in {"TQQQ": ("TQQQ", "QQQ", "tv"), "SPXL": ("SPXL", "SPY", None)}.items():
    m = Market(tr, px, src="yf", traded_src=src)
    mt, rets = run_one(m, TW, MODE_DIP, P["rsiLen"], P["rsiEntry"], P["exitLen"], P["trendLen"], P["maxHold"], P["dipTrend"], series=True)
    inwin, _ = m.window(TW)
    s = pd.Series(rets[inwin], index=m.dates[inwin])
    cl = pd.Series(m.cl, index=m.dates); b = cl[inwin].pct_change().fillna(0)
    y = pd.DataFrame({"strategy": (1 + s).groupby(s.index.year).prod() - 1, "buy&hold": (1 + b).groupby(b.index.year).prod() - 1})
    T = len(s) / 252; sr = mt["sharpe"]; se = math.sqrt((1 + 0.5 * sr * sr) / T)
    print(f"\n{a}: Test {T:.2f}y Sharpe {sr:.2f} SE {se:.2f} t={sr/se:.2f}")
    print(y.applymap(lambda v: f"{v:+.1%}").to_string())
    eq = (1 + s).cumprod(); pk = eq.cummax(); dd = (eq / pk - 1)
    print("worst DD date", dd.idxmin().date(), f"{dd.min():.1%}", "| longest underwater (days):", int((dd < 0).astype(int).groupby((dd >= 0).cumsum()).sum().max()))
