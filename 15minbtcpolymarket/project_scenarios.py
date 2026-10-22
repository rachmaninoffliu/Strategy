"""Forward projection: how do different strategy DESIGNS behave over the next ~60 days under different futures?

Futures are built by day-block bootstrap from REAL quote history (all hours), appended after the real history so the learner starts exactly where it is today:
  persists   : 60 days resampled from the regime period (Aug 24 -> now)
  fades      : each day drawn from the regime period with p=0.5, otherwise from the quiet period (about half the excess)
  ends (day 20): 20 regime days, then 40 quiet days
  quiet      : 60 days from the quiet period only (Dec 15 -> Aug 23)
  hour moves : like 'persists', but the hot hours shift by +3 h (22-23 UTC -> 01-02 UTC): what a hard-wired hour cannot follow
Designs are run through the real engine (trader.backtest) with the 45-day learner, 2c slippage, $10k, quarter Kelly, 2% cap.
Reports the distribution (median / 10th / 90th percentile) of 60-day return and max drawdown over N draws per future.

    python project_scenarios.py [N_DRAWS]
"""
import sys
import warnings
from datetime import date, timedelta

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
HORIZON = 60
CUT = int(pd.Timestamp("2026-08-24", tz="UTC").timestamp())
rows = load_rows()
_, hi = data_range(rows)
t_end = int(pd.Timestamp(hi.date(), tz="UTC").timestamp())                      # first day with no data yet (UTC midnight after the last full day)
rows["day"] = rows.open_ts // 86400
days_regime = sorted(rows[(rows.open_ts >= CUT) & (rows.open_ts < t_end)].day.unique())
dead_from = int(pd.Timestamp("2025-12-15", tz="UTC").timestamp())
quiet_rows = rows[(rows.open_ts >= dead_from) & (rows.open_ts < CUT)]
days_quiet = sorted(quiet_rows.day.unique())
# only days with all hours present are usable for all-hours designs
full_quiet = quiet_rows.groupby("day").ticker.nunique()
days_quiet = [d for d in days_quiet if full_quiet.get(d, 0) >= 80]
by_day = {d: g for d, g in rows.groupby("day")}
start_day = t_end // 86400
ALL = tuple(range(24))
R2P = StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35)
ALLP = StrategyParams(hours=ALL, cost_min=.15, cost_max=.35)
DESIGNS = {
    "R2 22-23, 10d decay": (R2P, dict(p_model="overall", min_edge=0.04, halflife_days=10.0)),
    "R2 22-23, 10d decay + DD throttle": (R2P, dict(p_model="overall", min_edge=0.04, halflife_days=10.0, dd_throttle=True)),
    "R2+ core + scout": (ALLP, dict(p_model="by_hour", hour_k=100, min_edge=0.04, halflife_days=14.0, core_hours=(22, 23), core_k=10)),
    "R2+ core + scout + DD throttle": (ALLP, dict(p_model="by_hour", hour_k=100, min_edge=0.04, halflife_days=14.0, core_hours=(22, 23), core_k=10, dd_throttle=True)),
}


def future(kind: str, rng) -> pd.DataFrame:
    picks = []
    for i in range(HORIZON):
        if kind in ("persists", "hour moves"):
            src = rng.choice(days_regime)
        elif kind == "fades":
            src = rng.choice(days_regime) if rng.random() < 0.5 else rng.choice(days_quiet)
        elif kind == "ends (day 20)":
            src = rng.choice(days_regime) if i < 20 else rng.choice(days_quiet)
        else:
            src = rng.choice(days_quiet)
        g = by_day[src].copy()
        g["open_ts"] = g.open_ts + (start_day + i - src) * 86400 + (3 * 3600 if kind == "hour moves" else 0)
        g["ticker"] = g.ticker + f"_s{i}"
        picks.append(g)
    return pd.concat(picks, ignore_index=True)[["ticker", "open_ts", "minute", "bid", "ask", "yes"]]


hist = rows[["ticker", "open_ts", "minute", "bid", "ask", "yes"]]
d0 = date.fromtimestamp(t_end) if False else (pd.Timestamp(t_end, unit="s", tz="UTC")).date()
d1 = d0 + timedelta(days=HORIZON - 1)
rng = np.random.default_rng(7)
res = []
KINDS = ["persists", "fades", "ends (day 20)", "quiet", "hour moves"]
print(f"history to {hi:%Y-%m-%d %H:%M}; regime days {len(days_regime)}, quiet days {len(days_quiet)}; {N} draws x {len(KINDS)} futures x {len(DESIGNS)} designs; horizon {HORIZON} d", flush=True)
for kind in KINDS:
    for k in range(N):
        fut = future(kind, rng)
        allrows = pd.concat([hist, fut], ignore_index=True)
        for nm, (p, kw) in DESIGNS.items():
            r = run_backtest(allrows, BacktestConfig(start=d0, end=d1, params=p, **kw))
            m = r.metrics
            res.append(dict(future=kind, design=nm, ret=m.get("total_return", 0.0) * 100, dd=m.get("max_drawdown", 0.0) * 100, trades=m["trades"]))
    print(f"  done {kind}", flush=True)
R = pd.DataFrame(res)
pd.set_option("display.width", 220)
q = lambda x, p: np.percentile(x, p)
tab = R.groupby(["future", "design"]).agg(ret_med=("ret", "median"), ret_p10=("ret", lambda x: q(x, 10)), ret_p90=("ret", lambda x: q(x, 90)), dd_med=("dd", "median"),
                                         dd_p10=("dd", lambda x: q(x, 10)), trades=("trades", "median"), p_loss=("ret", lambda x: (x < -1).mean() * 100))
print("\n60-day outcome by future and design (return %, drawdown %, share of draws losing >1%):")
print(tab.reindex(KINDS, level=0).round(1).to_string())
R.to_csv("data/scenario_results.csv", index=False)
