"""Full-history test (Dec 15 2025 -> now, hours 20-23 available all the way back): month-by-month raw edge and full backtests."""
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, extract_signals, load_rows
from trader.fees import fee_per_contract
from trader.strategy import StrategyParams

rows = load_rows()
lo, hi = data_range(rows)
print(f"data: {lo.date()} -> {hi}\n")
mk = pd.read_csv("data/markets.csv", usecols=["ticker", "open_ts", "volume"])
vol = mk.set_index("ticker").volume
pd.set_option("display.width", 230)


def raw(params, label):
    s = extract_signals(rows, params)
    px = np.minimum(s.cost + 0.02, 0.99)
    s["pnl"] = s.win - px - px.apply(lambda c: fee_per_contract("Polymarket US", c))
    s["month"] = pd.to_datetime(s.open_ts, unit="s", utc=True).dt.strftime("%Y-%m")
    s["vol"] = s.ticker.map(vol)
    g = s.groupby("month").agg(n=("pnl", "size"), win=("win", "mean"), price=("cost", "mean"), ev_c=("pnl", lambda x: x.mean() * 100),
                               t=("pnl", lambda x: x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 3 and x.std() > 0 else np.nan),
                               avg_volume_k=("vol", lambda x: x.mean() / 1000))
    g.columns = [f"{label} {c}" for c in g.columns]
    return g, s


g1, s22 = raw(StrategyParams(hours=(22, 23), cost_min=0.15, cost_max=0.35), "22-23h")
g2, s20 = raw(StrategyParams(hours=(20, 21), cost_min=0.15, cost_max=0.35), "20-21h")
print("=== 1. Raw edge per signal by month (underdog 15-35c, minutes 12-13, 2c slippage + fees) ===")
print(pd.concat([g1[["22-23h n", "22-23h win", "22-23h ev_c", "22-23h t", "22-23h avg_volume_k"]], g2[["20-21h n", "20-21h win", "20-21h ev_c", "20-21h t"]]], axis=1).round(2).to_string())

early = s22[s22.open_ts < pd.Timestamp("2026-06-01", tz="UTC").timestamp()]
mid = s22[(s22.open_ts >= pd.Timestamp("2026-06-01", tz="UTC").timestamp()) & (s22.open_ts < pd.Timestamp("2026-08-24", tz="UTC").timestamp())]
rest = s22[s22.open_ts >= pd.Timestamp("2026-08-24", tz="UTC").timestamp()]
for lab, x in (("Dec 15 - May 31 (thin liquidity, never seen before)", early), ("Jun 1 - Aug 23", mid), ("Aug 24 - now", rest)):
    print(f"  22-23h {lab}: n={len(x)} win={x.win.mean():.3f} price={x.cost.mean():.3f} EV={x.pnl.mean()*100:+.2f}c  t={x.pnl.mean()/x.pnl.std()*np.sqrt(len(x)):+.2f}")

print("\n=== 2. Full backtests (45d learner, 2c slippage, default sizing, $10k) ===")
P = {"R1  5-30c": StrategyParams(hours=(22, 23)), "R2 15-35c": StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35), "R0  5-35c": StrategyParams(hours=(22, 23), cost_min=.05, cost_max=.35)}
out = []
for start, lab in ((date(2025, 12, 15), "Dec 15 -> now (10 months)"), (date(2025, 12, 15), "")):
    pass
for lab, start, end in (("ALL 10 months", date(2025, 12, 15), hi.date()), ("Dec 15 - May 31 only (OOT)", date(2025, 12, 15), date(2026, 5, 31)), ("Jun 1 - now", date(2026, 6, 1), hi.date())):
    rec = {"period": lab}
    for n, p in P.items():
        m = run_backtest(rows, BacktestConfig(start=start, end=end, params=p, lookback_days=45)).metrics
        rec[n] = f"{m['sharpe']:.2f} / {m['total_return']*100:+.0f}% / {m['max_drawdown']*100:.0f}% ({m['trades']}, {m['win_rate']*100:.0f}%)" if m["trades"] else "no trades"
    out.append(rec)
print(pd.DataFrame(out).to_string(index=False))
print("(cells: Sharpe / return / maxDD (trades, win rate))")

print("\n=== 3. Same with 4c slippage (thin early books -> worse fills) ===")
out = []
for lab, start, end in (("ALL 10 months", date(2025, 12, 15), hi.date()), ("Dec 15 - May 31 only", date(2025, 12, 15), date(2026, 5, 31))):
    rec = {"period": lab}
    for n, p in P.items():
        m = run_backtest(rows, BacktestConfig(start=start, end=end, params=p, lookback_days=45, slippage=0.04)).metrics
        rec[n] = f"{m['sharpe']:.2f} / {m['total_return']*100:+.0f}% / {m['max_drawdown']*100:.0f}% ({m['trades']})" if m["trades"] else "no trades"
    out.append(rec)
print(pd.DataFrame(out).to_string(index=False))

print("\n=== 4. Quote quality in the early months: median spread at the decision minute (22-23h, minutes 12-13) ===")
r = rows[rows.minute.isin([12, 13])].copy()
r["hr"] = pd.to_datetime(r.open_ts, unit="s", utc=True).dt.hour
r = r[r.hr.isin([22, 23]) & (r.bid > 0) & (r.ask < 1)]
r["month"] = pd.to_datetime(r.open_ts, unit="s", utc=True).dt.strftime("%Y-%m")
print(r.groupby("month").apply(lambda x: pd.Series({"quotes": len(x), "median_spread_c": (x.ask - x.bid).median() * 100, "pct_spread_gt_3c": ((x.ask - x.bid) > 0.03).mean() * 100})).round(1).T.to_string())
