"""Learner memory design for R2 on the real 10 months: hard window vs exponential decay.  Dead = Dec 15 .. Aug 23, live = Aug 24 -> now.
Entry speed = first trade date; capture = P&L from Aug 24 on; costs = losses before it."""
import itertools
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

rows = load_rows()
_, hi = data_range(rows)
R2 = StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35)
CUT = pd.Timestamp("2026-08-24", tz="UTC").timestamp()
designs = [("window 45d", dict(lookback_days=45)), ("window 30d", dict(lookback_days=30)), ("window 21d", dict(lookback_days=21))]
designs += [(f"half-life {h}d", dict(halflife_days=float(h))) for h in (5, 7, 10, 14, 21, 30)]
out = []
for (nm, kw), pn, me in itertools.product(designs, (10, 20), (0.02, 0.04)):
    cfg = dict(params=R2, prior_n=pn, min_edge=me, **kw)
    dead = run_backtest(rows, BacktestConfig(start=date(2025, 12, 15), end=date(2026, 8, 23), **cfg))
    full = run_backtest(rows, BacktestConfig(start=date(2025, 12, 15), end=hi.date(), **cfg))
    t = full.trades
    live = t[t.entry_ts >= CUT]
    first = pd.to_datetime(live.entry_ts.min(), unit="s", utc=True).strftime("%m-%d") if len(live) else "-"
    m = full.metrics
    out.append(dict(memory=nm, prior_n=pn, min_edge_c=me * 100, dead_trades=dead.metrics["trades"], dead_ret=dead.metrics.get("total_return", 0) * 100, first_live=first,
                    live_trades=len(live), full_ret=m.get("total_return", 0) * 100, sharpe=m.get("sharpe", float("nan")), maxdd=m.get("max_drawdown", 0) * 100))
D = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(D.round(1).to_string(index=False))
