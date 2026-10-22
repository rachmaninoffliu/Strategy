"""Real history (Jun 1 -> now, GUI-like safety rules): fixed sizings vs Auto, after the drawdown-throttle-scales-the-cap fix.  Also the same on the 10-month range (Dec 15 -> now)."""
import warnings
from dataclasses import replace
from datetime import date

warnings.filterwarnings("ignore")
import pandas as pd

import trader.strategy as STRAT
from trader.adaptive import AutoConfig
from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

rows = load_rows()
lo, hi = data_range(rows)
ALL = StrategyParams(hours=tuple(range(24)), cost_min=.15, cost_max=.35)
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, compounding=True, p_basis="edge",
           p_model="by_hour", hour_k=100.0, core_hours=(22, 23), core_k=10.0)
AGGR = ((0.0, 1.0, 0.05), (25000.0, 0.5, 0.03), (100000.0, 0.35, 0.02), (500000.0, 0.2, 0.01))
base = AutoConfig()
DES = {
    "fixed 0.25x/2%/2k  [old throttle]": (dict(auto=None, kelly_mult=.25, max_frac=.02, max_contracts=2000, halflife_days=14.0), False),
    "fixed 0.25x/2%/2k": (dict(auto=None, kelly_mult=.25, max_frac=.02, max_contracts=2000, halflife_days=14.0), True),
    "fixed 0.5x/3%/20k": (dict(auto=None, kelly_mult=.5, max_frac=.03, max_contracts=20000, halflife_days=14.0), True),
    "fixed 1.0x/5%/200k (your sizing) [old throttle]": (dict(auto=None, kelly_mult=1.0, max_frac=.05, max_contracts=200000, halflife_days=14.0), False),
    "fixed 1.0x/5%/200k (your sizing)": (dict(auto=None, kelly_mult=1.0, max_frac=.05, max_contracts=200000, halflife_days=14.0), True),
    "AUTO default (z=.25, ladder 0.5x/3% start)": (dict(auto=replace(base, z=0.25)), True),
    "AUTO z=.5": (dict(auto=replace(base, z=0.5)), True),
    "AUTO aggressive ladder (1.0x/5% start)": (dict(auto=replace(base, z=0.25, ladder=AGGR)), True),
}
for label, start in (("Jun 1 -> now", date(2026, 6, 1)), ("Dec 15 -> now (10 months, hours 0-19 only 1/3 of days)", lo.date())):
    out = []
    for nm, (kw, tc) in DES.items():
        STRAT.THROTTLE_SCALES_CAP = tc
        r = run_backtest(rows, BacktestConfig(start=start, end=hi.date(), params=ALL, **GUI, **kw))
        m, t = r.metrics, r.trades
        out.append(dict(design=nm, ret_pct=m["total_return"] * 100, sharpe=m["sharpe"], maxdd_pct=m["max_drawdown"] * 100, calmar=m.get("calmar", float("nan")), trades=m["trades"], win_pct=m["win_rate"] * 100,
                        max_contracts=int(t.contracts.max()) if len(t) else 0, halted=int(m.get("windows_halted", 0) > 0)))
    print("\n== " + label)
    print(pd.DataFrame(out).round(1).to_string(index=False))
