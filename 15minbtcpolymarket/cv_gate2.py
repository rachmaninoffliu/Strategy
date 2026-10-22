"""Plateau check for the stricter gate: min_edge (net edge the learner must show) x prior strength, 45-day lookback, for R2/R1/R0 and for 2c / 4c slippage."""
import itertools, warnings
warnings.filterwarnings("ignore")
from datetime import date
import numpy as np, pandas as pd
import cv_longspot as lsp, cv_mechanism as mech, cv_final as cf
from trader.backtest import BacktestConfig, run_backtest
from trader.data import load_rows
from trader.strategy import StrategyParams

s = lsp.load_spot_all(); w = lsp.build_windows(s, minutes=(12, 13)); g = mech.fit_g(w)
syn, _ = cf.synthetic_rows(w, g); real = load_rows()
allrows = pd.concat([syn[syn.open_ts < cf.REAL_START], real[["ticker", "minute", "bid", "ask", "open_ts", "yes"]]], ignore_index=True)
V = {"R2": StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35), "R1": StrategyParams(hours=(22, 23)), "R0": StrategyParams(hours=(22, 23), cost_min=.05, cost_max=.35)}
pd.set_option("display.width", 220)
for slip in (0.02, 0.04):
    out = []
    for nm, p in V.items():
        for prior_n, me in itertools.product((20, 40), (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)):
            kw = dict(params=p, prior_n=prior_n, lookback_days=45, min_edge=me, slippage=slip)
            dead = run_backtest(allrows, BacktestConfig(start=date(2024, 1, 20), end=date(2026, 8, 23), **kw))
            full = run_backtest(allrows, BacktestConfig(start=date(2024, 1, 20), end=date(2026, 10, 10), **kw))
            live = full.trades[full.trades.entry_ts >= lsp.CUT]
            out.append(dict(strat=nm, prior_n=prior_n, min_edge_c=me * 100, dead_trades=dead.metrics["trades"], dead_ret=dead.metrics["total_return"] * 100,
                            full_ret=full.metrics["total_return"] * 100, sharpe=full.metrics["sharpe"], maxdd=full.metrics["max_drawdown"] * 100, live_trades=len(live)))
    D = pd.DataFrame(out)
    print(f"\n===== slippage {slip*100:.0f}c: columns by min_edge (c); cells = dead-world return % | full return % | maxDD % (trades dead/live) =====")
    for (nm, pn), x in D.groupby(["strat", "prior_n"]):
        print(f"{nm} prior_n={pn}: " + "  ".join(f"[{r.min_edge_c:.1f}c: {r.dead_ret:+.0f}|{r.full_ret:+.0f}|{r.maxdd:.0f} ({r.dead_trades}/{r.live_trades})]" for r in x.itertuples()))
