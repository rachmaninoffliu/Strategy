"""Stress the candidate presets: by period (pre-burst / burst / post-burst) and by slippage. Same-data exploration."""
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import ALL_HOURS, StrategyParams

rows = load_rows()
lo, hi = data_range(rows)
start = date(2026, 7, 20)
P = lambda d: pd.Timestamp(d, tz="UTC").timestamp()
PERIODS = [("Jul20-Aug23", P("2026-07-20"), P("2026-08-24")), ("Aug24-Sep27 (burst)", P("2026-08-24"), P("2026-09-28")), ("Sep28-now", P("2026-09-28"), 1e12)]
PRESETS = {
    "R1  22-23h dog 5-30c": (StrategyParams(), "overall"),
    "R2  22-23h dog 15-35c": (StrategyParams(cost_min=0.15, cost_max=0.35), "overall"),
    "R2b 22-23h dog 10-30c": (StrategyParams(cost_min=0.10, cost_max=0.30), "overall"),
    "R3  20-23h dog 5-30c by-hour": (StrategyParams(hours=(20, 21, 22, 23)), "by_hour"),
    "R3b 20-23h dog 15-35c by-hour": (StrategyParams(hours=(20, 21, 22, 23), cost_min=0.15, cost_max=0.35), "by_hour"),
    "ALL-DAY dog 5-30c by-hour": (StrategyParams(hours=ALL_HOURS), "by_hour"),
    "ALL-DAY dog 15-35c by-hour": (StrategyParams(hours=ALL_HOURS, cost_min=0.15, cost_max=0.35), "by_hour"),
}
pd.set_option("display.width", 220)
print("=== by period (2c slippage) : trades / win% / P&L$ on $10k start ===")
rows_out = []
for name, (params, pm) in PRESETS.items():
    r = run_backtest(rows, BacktestConfig(start=start, end=hi.date(), params=params, p_model=pm, slippage=0.02))
    t = r.trades
    rec = {"preset": name}
    for lab, a, b in PERIODS:
        x = t[(t.settle_ts >= a) & (t.settle_ts < b)]
        rec[lab] = f"{len(x):3d} / {x.win.mean()*100 if len(x) else 0:3.0f}% / {x.pnl.sum() if len(x) else 0:+8.0f}"
    rec["total"] = f"{len(t)} / {r.metrics.get('win_rate', 0)*100:.0f}% / Sharpe {r.metrics.get('sharpe', float('nan')):.1f}"
    rows_out.append(rec)
print(pd.DataFrame(rows_out).to_string(index=False))
print("\n=== slippage sensitivity (Sharpe / return% / trades) ===")
res = []
for name, (params, pm) in PRESETS.items():
    rec = {"preset": name}
    for s in (0.0, 0.01, 0.02, 0.03, 0.04):
        m = run_backtest(rows, BacktestConfig(start=start, end=hi.date(), params=params, p_model=pm, slippage=s)).metrics
        rec[f"{s*100:.0f}c"] = f"{m.get('sharpe', float('nan')):5.1f} / {m['total_return']*100:+6.0f}% / {m['trades']:3d}" if m["trades"] else "no trades"
    res.append(rec)
print(pd.DataFrame(res).to_string(index=False))
