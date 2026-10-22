"""Gate sensitivity: does a more skeptical learner cut the dead-regime bleed without giving up the live regime?  (R2, real engine.)
Dead world A = reconstructed quotes 2024-01-20..2025-12-14 + REAL quotes 2025-12-15..2026-08-23 (no regime).  Live world B = REAL quotes, learner warmed up from 2025-12-15, scored Aug 24 -> now.
Because B is one regime, this is a sensitivity table, NOT a tuning exercise."""
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
R2 = StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35)
out = []
for prior_n, lb, me in itertools.product((20, 40, 80), (30, 45, 90), (0.005, 0.02, 0.04)):
    kw = dict(params=R2, prior_n=prior_n, lookback_days=lb, min_edge=me)
    dead = run_backtest(allrows, BacktestConfig(start=date(2024, 1, 20), end=date(2026, 8, 23), **kw))
    full = run_backtest(allrows, BacktestConfig(start=date(2024, 1, 20), end=date(2026, 10, 10), **kw))
    t = full.trades
    live = t[t.entry_ts >= lsp.CUT]
    first = pd.to_datetime(live.entry_ts.min(), unit="s", utc=True).strftime("%m-%d") if len(live) else "-"
    out.append(dict(prior_n=prior_n, lookback=lb, min_edge_c=me * 100, dead_trades=dead.metrics["trades"], dead_ret_pct=dead.metrics["total_return"] * 100,
                    full_ret_pct=full.metrics["total_return"] * 100, full_sharpe=full.metrics["sharpe"], full_maxdd=full.metrics["max_drawdown"] * 100,
                    live_trades=len(live), first_live_trade=first))
D = pd.DataFrame(out)
pd.set_option("display.width", 220)
print(D.round(1).to_string(index=False))
base = D[(D.prior_n == 20) & (D.lookback == 45) & (D.min_edge_c == 0.5)].iloc[0]
print("\nbaseline (app default): dead-world %.1f%%, full %.1f%%, Sharpe %.2f, maxDD %.1f%%" % (base.dead_ret_pct, base.full_ret_pct, base.full_sharpe, base.full_maxdd))
print("configs with smaller dead-world loss AND >= 80%% of the baseline full return:")
print(D[(D.dead_ret_pct > base.dead_ret_pct) & (D.full_ret_pct >= 0.8 * base.full_ret_pct)].round(1).to_string(index=False))
