"""Compare strategy variants honestly: all-day vs 22-23 UTC, underdog vs favourite legs, overall vs hour-adaptive probability.
Usage: python explore_variants.py [start YYYY-MM-DD] [slippage_cents]"""
import sys
import warnings
from dataclasses import replace
from datetime import date

warnings.filterwarnings("ignore")
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import ALL_HOURS, StrategyParams

start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
slip = float(sys.argv[2]) / 100 if len(sys.argv) > 2 else 0.02
rows = load_rows()
lo, hi = data_range(rows)
start = start or lo.date()
days = (hi.date() - start).days

R1 = StrategyParams()
ALL_DOG = StrategyParams(hours=ALL_HOURS)
ALL_DOGFAV = StrategyParams(hours=ALL_HOURS, legs=("DOG", "FAV"))
ALL_FAV = StrategyParams(hours=ALL_HOURS, legs=("FAV",))
ALL_DOG_LATE = StrategyParams(hours=ALL_HOURS, minutes=(10, 11, 12, 13))

BASIS = sys.argv[3] if len(sys.argv) > 3 else "edge"
VARIANTS = [
    ("R1: 22-23h dog, overall p", R1, "overall", 40),
    ("22-23h dog, by-hour p", R1, "by_hour", 40),
    ("ALL-DAY dog, overall p", ALL_DOG, "overall", 40),
    ("ALL-DAY dog, by-hour p k=20", ALL_DOG, "by_hour", 20),
    ("ALL-DAY dog, by-hour p k=40", ALL_DOG, "by_hour", 40),
    ("ALL-DAY dog, by-hour p k=80", ALL_DOG, "by_hour", 80),
    ("ALL-DAY dog min10-13, by-hour k=40", ALL_DOG_LATE, "by_hour", 40),
    ("ALL-DAY fav only, overall p", ALL_FAV, "overall", 40),
    ("ALL-DAY fav only, by-hour p", ALL_FAV, "by_hour", 40),
    ("ALL-DAY dog+fav, by-hour p", ALL_DOGFAV, "by_hour", 40),
]
out = []
for name, params, pm, k in VARIANTS:
    cfg = BacktestConfig(start=start, end=hi.date(), params=params, p_model=pm, hour_k=k, slippage=slip, p_basis=BASIS)
    r = run_backtest(rows, cfg)
    m = r.metrics
    if m["trades"] == 0:
        out.append((name, 0, 0, 0, 0, 0, 0, 0, 0))
        continue
    legs = r.trades.leg.value_counts().to_dict()
    out.append((name, m["trades"], m["trades_per_day"], m["win_rate"], m["total_return"] * 100, m["max_drawdown"] * 100, m["sharpe"],
                m["max_consec_losses"], m["profit_factor"]))
T = pd.DataFrame(out, columns=["variant", "trades", "per_day", "win", "return%", "maxDD%", "sharpe", "maxLossStreak", "profitF"])
pd.set_option("display.width", 200)
print(f"{start} -> {hi.date()} ({days} days)  slippage {slip*100:.0f}c  p basis: {BASIS}  quarter-Kelly, 2% cap, Polymarket US fees\n")
print(T.round(2).to_string(index=False))
