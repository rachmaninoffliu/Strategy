"""60-day futures for R2 under default vs aggressive sizing, with the GUI's safety rules on, and how fast the engine reacts when the regime ends.
Futures are day-bootstrapped from real history (see project_scenarios.py).  Extra outputs: share of paths that hit the drawdown stop, days between the regime end and the last trade,
and the P&L after the regime ended."""
import sys
import warnings
from datetime import date, timedelta

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
HORIZON, END_DAY = 60, 20
CUT = int(pd.Timestamp("2026-08-24", tz="UTC").timestamp())
rows = load_rows()
_, hi = data_range(rows)
t_end = int(pd.Timestamp(hi.date(), tz="UTC").timestamp())
rows["day"] = rows.open_ts // 86400
days_regime = sorted(rows[(rows.open_ts >= CUT) & (rows.open_ts < t_end)].day.unique())
quiet = rows[(rows.open_ts >= int(pd.Timestamp("2025-12-15", tz="UTC").timestamp())) & (rows.open_ts < CUT)]
days_quiet = sorted(quiet.day.unique())
by_day = {d: g for d, g in rows.groupby("day")}
start_day = t_end // 86400
R2 = StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35)
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, p_model="overall", halflife_days=10.0)
DESIGNS = {
    "R2 default sizing (0.25x / 2% / 2,000)": dict(kelly_mult=0.25, max_frac=0.02, max_contracts=2000),
    "R2 YOUR sizing (1.0x / 5% / 200,000)": dict(kelly_mult=1.0, max_frac=0.05, max_contracts=200000),
    "R2 middle (0.5x / 3% / 20,000)": dict(kelly_mult=0.5, max_frac=0.03, max_contracts=20000),
}
KINDS = ["persists", "fades", "ends (day 20)", "quiet"]


def future(kind, rng):
    picks = []
    for i in range(HORIZON):
        if kind == "persists":
            src = rng.choice(days_regime)
        elif kind == "fades":
            src = rng.choice(days_regime) if rng.random() < 0.5 else rng.choice(days_quiet)
        elif kind == "ends (day 20)":
            src = rng.choice(days_regime) if i < END_DAY else rng.choice(days_quiet)
        else:
            src = rng.choice(days_quiet)
        g = by_day[src].copy()
        g["open_ts"] = g.open_ts + (start_day + i - src) * 86400
        g["ticker"] = g.ticker + f"_s{i}"
        picks.append(g)
    return pd.concat(picks, ignore_index=True)[["ticker", "open_ts", "minute", "bid", "ask", "yes"]]


hist = rows[["ticker", "open_ts", "minute", "bid", "ask", "yes"]]
d0 = pd.Timestamp(t_end, unit="s", tz="UTC").date()
d1 = d0 + timedelta(days=HORIZON - 1)
rng = np.random.default_rng(11)
out = []
for kind in KINDS:
    for k in range(N):
        allrows = pd.concat([hist, future(kind, rng)], ignore_index=True)
        for nm, kw in DESIGNS.items():
            r = run_backtest(allrows, BacktestConfig(start=d0, end=d1, params=R2, **GUI, **kw))
            m, t = r.metrics, r.trades
            rec = dict(future=kind, design=nm, ret=m.get("total_return", 0.0) * 100, dd=m.get("max_drawdown", 0.0) * 100, trades=m["trades"], halted=int(m.get("windows_halted", 0) > 0))
            if len(t):
                day = ((t.entry_ts - t_end) // 86400).astype(int)
                rec["last_day"] = int(day.max())
                if kind == "ends (day 20)":
                    after = t[day >= END_DAY]
                    rec["trades_after_end"] = len(after)
                    rec["last_trade_after_end_day"] = int(day.max()) - END_DAY if len(after) else 0
                    eq_end = float(t[day < END_DAY].equity_after.iloc[-1]) if (day < END_DAY).any() else float(m["initial_capital"])
                    rec["pnl_after_end_pct"] = (float(t.equity_after.iloc[-1]) / eq_end - 1) * 100
            out.append(rec)
    print("done", kind, flush=True)
R = pd.DataFrame(out)
R.to_csv("data/scenario_aggr_results.csv", index=False)
pd.set_option("display.width", 220)
q = lambda x, p: np.percentile(x, p)
tab = R.groupby(["future", "design"]).agg(ret_med=("ret", "median"), ret_p10=("ret", lambda x: q(x, 10)), ret_p90=("ret", lambda x: q(x, 90)), dd_med=("dd", "median"), dd_p10=("dd", lambda x: q(x, 10)),
                                         hit_dd_stop_pct=("halted", lambda x: x.mean() * 100), trades=("trades", "median"))
print("\n60-day outcome (median / p10 / p90 return %, drawdown %, share of paths where the drawdown stop halted trading):")
print(tab.reindex(KINDS, level=0).round(1).to_string())
e = R[R.future == "ends (day 20)"].groupby("design").agg(trades_after_end=("trades_after_end", "median"), days_until_last_trade_after_end=("last_trade_after_end_day", "median"), pnl_after_end_pct_med=("pnl_after_end_pct", "median"),
                                                     pnl_after_end_pct_p10=("pnl_after_end_pct", lambda x: q(x, 10)))
print("\nRegime ends at day 20 - how fast does the engine stop, and what does the lag cost?")
print(e.round(1).to_string())
