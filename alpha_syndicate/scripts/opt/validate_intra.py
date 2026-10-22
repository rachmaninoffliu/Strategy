"""Validate the intraday replica against the TradingView 1h run captured earlier (TQQQ, Dip defaults, Train, HTF=D)."""
from intra import *
m = IMarket("TQQQ", "QQQ", "60")
print("bars", len(m.op), "first", m.t["dt"].iloc[0], "last", m.t["dt"].iloc[-1], "| bars/day check:", m.t.groupby("key").size().value_counts().head(4).to_dict())
mt = run_one(m, TRAIN, MODE_DIP, 2, 20, 5, 200, 10, True)
tv = dict(trades=1324, win=685/1324, cagr=-0.228, mdd=0.963, sharpe=-0.77, sortino=-0.94, calmar=-0.24, pf=0.788, corr=0.51, exposure=0.258)
print("TQQQ 1h Train Dip defaults (HTF D, hold 10d = 70 bars)")
for k, v in tv.items():
    print(f"  {k:9s} replica {mt[k]:10.4f}   TV {v:10.4f}")
print("  final equity multiple replica", round(mt['final'], 5), " TV tester: net -95,344.33 on 100k -> 0.04656")
