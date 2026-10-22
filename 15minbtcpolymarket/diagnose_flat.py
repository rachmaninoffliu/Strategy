"""Why did the default preset (R2) place no trades in July / August?  Looks at the raw signals and at the walk-forward probability gate."""
import warnings
from datetime import date, datetime, timezone

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.backtest import PROB_MODES, BacktestConfig, run_backtest
from trader.data import extract_signals, load_rows
from trader.fees import fee_per_contract, fee_total
from trader.strategy import StrategyParams

P = StrategyParams(hours=(22, 23), cost_min=0.15, cost_max=0.35)        # R2
SLIP, FEE, N0, MIN_EDGE = 0.02, "Polymarket US", 20, 0.005
rows = load_rows()
sig = extract_signals(rows, P)
sig["dt"] = pd.to_datetime(sig.open_ts, unit="s", utc=True)
sig["month"] = sig.dt.dt.strftime("%Y-%m")
sig["excess"] = sig.win - sig.cost
print("=== 1. Were there signals in Jul/Aug?  (R2: 22-23 UTC, minute 12/13, underdog 15-35c) ===")
g = sig.groupby("month").agg(signals=("win", "size"), win_rate=("win", "mean"), avg_price=("cost", "mean"), mean_edge_c=("excess", lambda x: x.mean() * 100))
g["breakeven_win"] = g.avg_price
print(g.round(3).to_string())

# walk-forward gate exactly as the engine does it
sig = sig.sort_values("open_ts").reset_index(drop=True)
recs = []
E = 0.0; n = 0
for r in sig.itertuples():
    entry_ts = r.open_ts + 60 * r.minute
    # signals settled before entry
    prior = sig[sig.close_ts <= entry_ts]
    n, E = len(prior), float((prior.win - prior.cost).sum())
    delta = E / (n + N0)
    p = min(max(r.cost + delta, 0), 1)
    px = min(r.cost + SLIP, 0.99)
    cost_all = px + fee_per_contract(FEE, px)
    recs.append((r.dt, r.month, r.cost, delta, p, cost_all, p - cost_all, (p - cost_all) >= MIN_EDGE))
G = pd.DataFrame(recs, columns=["dt", "month", "cost", "learned_edge", "p", "cost_all", "kelly_edge", "bet"])
G["week"] = G.dt.dt.to_period("W").astype(str)
print("\n=== 2. What the probability gate believed (walk-forward, using only signals already settled) ===")
m = G.groupby("month").agg(signals=("bet", "size"), bets=("bet", "sum"), learned_edge_c=("learned_edge", lambda x: x.mean() * 100),
                           needed_edge_c=("cost_all", lambda x: 0), gate_margin_c=("kelly_edge", lambda x: x.mean() * 100))
need = (G.cost_all - G.cost + MIN_EDGE).groupby(G.month).mean() * 100
m["needed_edge_c"] = need
print(m.round(2).to_string())
first_bet = G[G.bet].dt.min()
print(f"\nFirst bet of the whole run: {first_bet:%Y-%m-%d}")
w = G[(G.dt >= '2026-08-17') & (G.dt < '2026-09-20')].groupby("week").agg(signals=("bet", "size"), bets=("bet", "sum"), learned_edge_c=("learned_edge", lambda x: x.mean() * 100))
w["needed_c"] = (G.cost_all - G.cost + MIN_EDGE).groupby(G.week).mean().reindex(w.index) * 100
print("\n=== 3. The burst: how long did the gate take to open? ===")
print(w.round(2).to_string())

# counterfactual: what if it had simply traded Jul-Aug with the burst-era win rate (hindsight p=0.30)?
rowsx = rows
cf = run_backtest(rowsx, BacktestConfig(start=date(2026, 7, 1), end=date(2026, 8, 24), params=P, prob_mode=PROB_MODES[1], fixed_p=0.30, kelly_mult=0.25, max_frac=0.02))
print("\n=== 4. Counterfactual: bet every R2 signal in Jul 1 - Aug 23 with a fixed p = 30% (what the gate prevented) ===")
m4 = cf.metrics
print(f"trades {m4['trades']} | win rate {m4['win_rate']*100:.0f}% | P&L ${m4['total_pnl']:+,.0f} on $10k ({m4['total_return']*100:+.1f}%) | max drawdown {m4['max_drawdown']*100:.1f}%")
print("\nsignals per month by whether the learned edge cleared the gate:")
print(G.groupby(["month", "bet"]).size().unstack(fill_value=0).rename(columns={False: "blocked", True: "bet"}).to_string())
