"""Can a broader, adaptive design (any hour, hierarchical shrinkage, 4c hurdle) beat a single hard-wired cell (R2 = 22-23 UTC)?
Real Kalshi quotes, the real engine.  `start` is the first day; 'dead' = start .. 2026-08-23 (no regime), 'live' = whole range to now."""
import itertools
import sys
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

START = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 6, 1)
rows = load_rows()
_, hi = data_range(rows)
pd.set_option("display.width", 220)
ALL = tuple(range(24))
V = {
    "R2 22-23 (baseline)": (StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35), dict(p_model="overall")),
    "20-23 by-hour": (StrategyParams(hours=(20, 21, 22, 23), cost_min=.15, cost_max=.35), dict(p_model="by_hour")),
    "all-day overall": (StrategyParams(hours=ALL, cost_min=.15, cost_max=.35), dict(p_model="overall")),
    "all-day by-hour k=40": (StrategyParams(hours=ALL, cost_min=.15, cost_max=.35), dict(p_model="by_hour", hour_k=40)),
    "all-day by-hour k=15": (StrategyParams(hours=ALL, cost_min=.15, cost_max=.35), dict(p_model="by_hour", hour_k=15)),
    "all-day by-hour k=100": (StrategyParams(hours=ALL, cost_min=.15, cost_max=.35), dict(p_model="by_hour", hour_k=100)),
    "all-day by-hour 5-35c": (StrategyParams(hours=ALL, cost_min=.05, cost_max=.35), dict(p_model="by_hour", hour_k=40)),
    "all-day by-hour +fav": (StrategyParams(hours=ALL, cost_min=.15, cost_max=.35, legs=("DOG", "FAV")), dict(p_model="by_hour", hour_k=40)),
}
out = []
for (nm, (p, kw)), me, lb in itertools.product(V.items(), (0.02, 0.04), (45, 90)):
    if nm.startswith("R2") and lb == 90:
        continue
    cfg = dict(params=p, min_edge=me, lookback_days=lb, **kw)
    dead = run_backtest(rows, BacktestConfig(start=START, end=date(2026, 8, 23), **cfg))
    full = run_backtest(rows, BacktestConfig(start=START, end=hi.date(), **cfg))
    t = full.trades
    live = t[t.entry_ts >= pd.Timestamp("2026-08-24", tz="UTC").timestamp()]
    m = full.metrics
    out.append(dict(variant=nm, min_edge_c=me * 100, lookback=lb, dead_trades=dead.metrics["trades"], dead_ret=dead.metrics.get("total_return", 0) * 100,
                    live_trades=len(live), live_pnl_k=live.pnl.sum() / 1000, full_ret=m.get("total_return", 0) * 100, sharpe=m.get("sharpe", float("nan")), maxdd=m.get("max_drawdown", 0) * 100))
D = pd.DataFrame(out)
print(f"start {START} -> {hi:%Y-%m-%d}\n")
print(D.round(1).to_string(index=False))
