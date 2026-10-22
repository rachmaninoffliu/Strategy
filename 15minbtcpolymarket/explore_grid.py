"""Selectivity / window / price-band exploration (same data, so read as exploration, not validation).
Usage: python explore_grid.py [start] [slippage_cents]"""
import sys
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import ALL_HOURS, StrategyParams

rows = load_rows()
lo, hi = data_range(rows)
start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else lo.date()
slip = float(sys.argv[2]) / 100 if len(sys.argv) > 2 else 0.02
HOURS = {"22-23": (22, 23), "20-23": (20, 21, 22, 23), "ALL": ALL_HOURS}
out = []


def run(label, **kw):
    params = kw.pop("params")
    cfg = BacktestConfig(start=start, end=hi.date(), params=params, slippage=slip, **kw)
    m = run_backtest(rows, cfg).metrics
    if m["trades"] == 0:
        out.append((label, 0, 0, 0, 0, 0, 0, 0))
        return
    out.append((label, m["trades"], m["trades_per_day"], m["win_rate"], m["total_return"] * 100, m["max_drawdown"] * 100, m["sharpe"], m["max_consec_losses"]))


for hname, hrs in HOURS.items():
    for pm in ("overall", "by_hour"):
        if hname == "22-23" and pm == "by_hour":
            continue
        for me in (0.005, 0.02, 0.04):
            run(f"{hname:5s} dog .05-.30 {pm:7s} minEdge {me*100:.1f}c", params=StrategyParams(hours=hrs), p_model=pm, min_edge=me)
for hname in ("22-23", "20-23"):
    hrs = HOURS[hname]
    for lb in (30, 60, 90):
        run(f"{hname:5s} dog overall lookback {lb}d", params=StrategyParams(hours=hrs), lookback_days=lb)
    run(f"{hname:5s} dog minutes 10-13", params=StrategyParams(hours=hrs, minutes=(10, 11, 12, 13)))
    run(f"{hname:5s} dog minutes 8-13", params=StrategyParams(hours=hrs, minutes=(8, 9, 10, 11, 12, 13)))
    for cmin, cmax in ((0.10, 0.30), (0.15, 0.35), (0.20, 0.40), (0.15, 0.45)):
        run(f"{hname:5s} dog {cmin:.2f}-{cmax:.2f} (higher win-rate band)", params=StrategyParams(hours=hrs, cost_min=cmin, cost_max=cmax))
T = pd.DataFrame(out, columns=["variant", "trades", "per_day", "win", "return%", "maxDD%", "sharpe", "maxLossStreak"])
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
print(f"{start} -> {hi.date()}  slippage {slip*100:.0f}c  p basis edge, quarter-Kelly, 2% cap, Polymarket US fees\n")
print(T.round(2).to_string(index=False))
