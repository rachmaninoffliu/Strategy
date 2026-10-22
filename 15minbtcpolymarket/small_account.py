"""$500 account: does 'bet big' (1.0x Kelly / 5% cap / 200k contracts) beat the default or a middle size?
Part 1  real history Jun 1 -> now, R2+ rules, current engine (throttle scales the cap), initial capital $500.
Part 2  Monte-Carlo of 100 trades (~40 days at the R2+ rate) at a FIXED stake fraction of equity, entry prices resampled from the real trades, true win probability = price + delta,
        for delta = the true gross edge per contract (cents): +15 (the regime's average so far), +9 (recent), +5 (~break-even after costs), 0, -5.  Drawdown throttle, 10% daily-loss pause is ignored, 25% drawdown stop applied.
"""
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.adaptive import AutoConfig
from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.fees import fee_per_contract
from trader.strategy import StrategyParams, drawdown_throttle

rows = load_rows()
lo, hi = data_range(rows)
ALL = StrategyParams(hours=tuple(range(24)), cost_min=.15, cost_max=.35)
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, compounding=True, p_basis="edge",
           p_model="by_hour", hour_k=100.0, core_hours=(22, 23), core_k=10.0, halflife_days=14.0, capital=500.0)
DES = {
    "default 0.25x / 2% / 2k": dict(kelly_mult=.25, max_frac=.02, max_contracts=2000),
    "0.25x / 1% / 2k": dict(kelly_mult=.25, max_frac=.01, max_contracts=2000),
    "middle 0.5x / 3% / 20k": dict(kelly_mult=.5, max_frac=.03, max_contracts=20000),
    "YOURS 1.0x / 5% / 200k": dict(kelly_mult=1.0, max_frac=.05, max_contracts=200000),
}
print("== Part 1: real history Jun 1 -> now, start $500")
tr_ref = None
out = []
for nm, kw in DES.items():
    r = run_backtest(rows, BacktestConfig(start=date(2026, 6, 1), end=hi.date(), params=ALL, **GUI, **kw))
    m, t = r.metrics, r.trades
    if tr_ref is None:
        tr_ref = t
    out.append(dict(sizing=nm, final=f"${m['final_equity']:,.0f}", ret=f"{m['total_return'] * 100:+.0f}%", sharpe=round(m["sharpe"], 2), maxDD=f"{m['max_drawdown'] * 100:.0f}%", trades=m["trades"],
                    avg_stake=f"${t.stake.mean():,.0f}", first_stake=f"${t.stake.iloc[0]:,.0f}", max_stake=f"${t.stake.max():,.0f}", max_contracts=int(t.contracts.max()), halted=int(m.get("windows_halted", 0) > 0)))
print(pd.DataFrame(out).to_string(index=False))
r = run_backtest(rows, BacktestConfig(start=date(2026, 6, 1), end=hi.date(), params=ALL, **GUI, **DES["YOURS 1.0x / 5% / 200k"]))
t = r.trades
print(f"\nYOURS: first 10 trades' stakes ${t.stake.head(10).round(0).tolist()}, contracts {t.contracts.head(10).tolist()}; equity first 10: {t.equity_after.head(10).round(0).tolist()}")

print("\n== Part 2: Monte-Carlo, 100 trades, fixed stake fraction of equity (25% drawdown stop, throttle on), start $500")
prices = tr_ref.quoted.values
rng = np.random.default_rng(7)
N, TRIALS = 100, 4000


def sim(frac: float, delta: float):
    fin, stopped = np.empty(TRIALS), np.zeros(TRIALS, bool)
    for k in range(TRIALS):
        eq = peak = 500.0
        cs, us = rng.choice(prices, N), rng.random(N)
        for i in range(N):
            c = float(cs[i])
            px = min(c + .02, .99)
            cost = px + fee_per_contract("Polymarket US", px)
            f = frac * drawdown_throttle(1 - eq / peak)
            q = int(f * eq / cost)
            if q < 1:
                continue
            win = us[i] < min(c + delta, .99)
            eq += q * (1.0 if win else 0.0) - q * cost
            peak = max(peak, eq)
            if eq <= peak * .75:
                stopped[k] = True
                break
        fin[k] = eq
    return fin, stopped


rows_out = []
for delta in (0.15, 0.09, 0.05, 0.0, -0.05):
    for nm, frac in (("2%", .02), ("3%", .03), ("5%", .05)):
        fin, st = sim(frac, delta)
        rows_out.append(dict(true_edge_c=int(delta * 100), stake=nm, median_final=f"${np.median(fin):,.0f}", p10=f"${np.percentile(fin, 10):,.0f}", p90=f"${np.percentile(fin, 90):,.0f}",
                             P_end_below_500=f"{(fin < 500).mean() * 100:.0f}%", P_hit_25pct_stop=f"{st.mean() * 100:.0f}%"))
print(pd.DataFrame(rows_out).to_string(index=False))
