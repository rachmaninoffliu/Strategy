"""R1 (underdog 5-30c) vs R2 (15-35c) vs a combined band, across periods, slippage and learning windows."""
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

rows = load_rows()
lo, hi = data_range(rows)
V = {"R1  5-30c": StrategyParams(hours=(22, 23)),
     "R2 15-35c": StrategyParams(hours=(22, 23), cost_min=0.15, cost_max=0.35),
     "R0  5-35c": StrategyParams(hours=(22, 23), cost_min=0.05, cost_max=0.35)}
pd.set_option("display.width", 230)


def run(p, start, **kw):
    return run_backtest(rows, BacktestConfig(start=start, end=hi.date(), params=p, **kw))


print(f"data through {hi}\n")
print("=== A. Sharpe / return / maxDD by START date x learning window (2c slippage, default sizing) ===")
out = []
for start in (date(2026, 6, 1), date(2026, 7, 20), date(2026, 8, 24), date(2026, 9, 28)):
    for lb in (45, 0):
        rec = {"start": str(start), "learner": f"{lb or 'all'}d"}
        for n, p in V.items():
            m = run(p, start, lookback_days=lb).metrics
            rec[n] = f"{m['sharpe']:.2f} / {m['total_return']*100:+.0f}% / {m['max_drawdown']*100:.0f}% ({m['trades']})" if m["trades"] else "no trades"
        out.append(rec)
print(pd.DataFrame(out).to_string(index=False))

print("\n=== B. Slippage sensitivity, Jun 1 -> now, 45d learner: Sharpe / return ===")
out = []
for n, p in V.items():
    rec = {"variant": n}
    for sl in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
        m = run(p, date(2026, 6, 1), lookback_days=45, slippage=sl).metrics
        rec[f"{sl*100:.0f}c"] = f"{m['sharpe']:.1f} / {m['total_return']*100:+.0f}%"
    out.append(rec)
print(pd.DataFrame(out).to_string(index=False))

print("\n=== C. Where does each variant's profit come from? (Jun 1 -> now, 45d, 2c) ===")
for n, p in V.items():
    r = run(p, date(2026, 6, 1), lookback_days=45)
    t = r.trades.copy()
    t["pb"] = pd.cut(t.entry, [0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    g = t.groupby("pb", observed=True).agg(n=("pnl", "size"), win=("win", "mean"), pnl=("pnl", "sum"))
    top5 = t.pnl.nlargest(5).sum() / t.pnl.sum() if t.pnl.sum() > 0 else float("nan")
    m = r.metrics
    print(f"\n{n}: {m['trades']} trades, win {m['win_rate']*100:.0f}%, avg price paid {t.entry.mean():.3f}, payoff {m['payoff_ratio']:.1f}x, max loss streak {m['max_consec_losses']}, "
          f"top-5 trades = {top5*100:.0f}% of profit, worst trade ${m['worst_trade']:,.0f}, best ${m['best_trade']:,.0f}")
    print(g.round(2).T.to_string())

print("\n=== D. First half vs second half of the burst era (Aug 24 -> now) split at Sep 28, 45d learner, 2c: trades / win / P&L$ ===")
P = lambda d: pd.Timestamp(d, tz="UTC").timestamp()
for n, p in V.items():
    t = run(p, date(2026, 6, 1), lookback_days=45).trades
    s = []
    for lab, a, b in (("Jun-Aug23", P("2026-06-01"), P("2026-08-24")), ("Aug24-Sep27", P("2026-08-24"), P("2026-09-28")), ("Sep28-now", P("2026-09-28"), 1e12)):
        x = t[(t.settle_ts >= a) & (t.settle_ts < b)]
        s.append(f"{lab}: {len(x)}/{x.win.mean()*100 if len(x) else 0:.0f}%/{x.pnl.sum():+,.0f}")
    print(f"  {n}: " + " | ".join(s))
