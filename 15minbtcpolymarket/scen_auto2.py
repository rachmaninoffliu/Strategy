"""(variant of scen_auto.py: Auto-internals ablation on the same bootstrapped futures)
60-day futures: fixed R2+ sizing (default and the user's aggressive 1.0x/5%/200k) vs the Auto layer (several settings), GUI safety rules on.
Futures are day-bootstrapped from real history (see project_scenarios.py / scen_aggr.py) and appended after the last real window:
  persists  - every day drawn from the post-Aug-24 regime days          (the regime lasts)
  fades     - each day 50/50 regime day / quiet day                     (the regime half-decays)
  ends      - regime days for 20 days, then quiet days only             (the regime ends; how fast does the engine stop, what does the lag cost)
  quiet     - quiet days only                                           (no regime: how much does each design bleed)
  moves     - regime days with the whole day shifted 3h earlier, so the edge sits at 19-20 UTC instead of 22-23  (can the engine find it)
Output: median / p10 / p90 return, median drawdown, share of paths halted by the drawdown stop, trades; plus the post-end cost for 'ends'."""
import sys
import warnings
from dataclasses import replace
from datetime import timedelta

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import trader.strategy as STRAT
from trader.adaptive import AutoConfig
from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/scenario_auto_results.csv"
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
ALL = StrategyParams(hours=tuple(range(24)), cost_min=.15, cost_max=.35)
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, compounding=True, p_basis="edge",
           p_model="by_hour", hour_k=100.0, core_hours=(22, 23), core_k=10.0, halflife_days=14.0)
AGGR_LADDER = ((0.0, 1.0, 0.05), (25000.0, 0.5, 0.03), (100000.0, 0.35, 0.02), (500000.0, 0.2, 0.01))
base = AutoConfig()
FLAT = tuple((r, 1.0) for r in ("HOT", "BUILDING", "FADING")) + (("COLD", 0.0),)
DESIGNS = {
    "AUTO base (z=.25, mem 7/14/30, regime scaling .5/.4)": dict(auto=replace(base, z=0.25), _tc=True),
    "AUTO regime scaling off (only COLD=no trade)": dict(auto=replace(base, z=0.25, regime_mult=FLAT), _tc=True),
    "AUTO memories 10/21/45": dict(auto=replace(base, z=0.25, half_lives=(10.0, 21.0, 45.0)), _tc=True),
    "AUTO single memory 14d": dict(auto=replace(base, z=0.25, half_lives=(14.0,)), _tc=True),
}
KINDS = ["persists", "fades", "ends (day 20)", "quiet", "moves (19-20 UTC)"]
COLS = ["ticker", "open_ts", "minute", "bid", "ask", "yes", "vol"]


def future(kind, rng):
    picks = []
    for i in range(HORIZON):
        if kind == "persists" or kind.startswith("moves"):
            src = rng.choice(days_regime)
        elif kind == "fades":
            src = rng.choice(days_regime) if rng.random() < 0.5 else rng.choice(days_quiet)
        elif kind == "ends (day 20)":
            src = rng.choice(days_regime) if i < END_DAY else rng.choice(days_quiet)
        else:
            src = rng.choice(days_quiet)
        g = by_day[src].copy()
        g["open_ts"] = g.open_ts + (start_day + i - src) * 86400 - (3 * 3600 if kind.startswith("moves") else 0)
        g["ticker"] = g.ticker + f"_s{i}"
        picks.append(g)
    return pd.concat(picks, ignore_index=True)[COLS]


hist = rows[COLS]
d0 = pd.Timestamp(t_end, unit="s", tz="UTC").date()
d1 = d0 + timedelta(days=HORIZON - 1)
rng = np.random.default_rng(11)
out = []
for kind in KINDS:
    for k in range(N):
        allrows = pd.concat([hist, future(kind, rng)], ignore_index=True)
        for nm, kw in DESIGNS.items():
            kw = dict(kw)
            STRAT.THROTTLE_SCALES_CAP = kw.pop("_tc")
            r = run_backtest(allrows, BacktestConfig(start=d0, end=d1, params=ALL, **GUI, **kw))
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
                if kind.startswith("moves"):
                    rec["share_trades_19_20"] = float(pd.to_datetime(t.entry_ts, unit="s", utc=True).dt.hour.isin([19, 20]).mean())
                    rec["first_trade_day"] = int(day.min())
            out.append(rec)
    print("done", kind, flush=True)
    pd.DataFrame(out).to_csv(OUT, index=False)
R = pd.DataFrame(out)
pd.set_option("display.width", 240)
q = lambda x, p: np.percentile(x, p)
tab = R.groupby(["future", "design"]).agg(ret_med=("ret", "median"), ret_p10=("ret", lambda x: q(x, 10)), ret_p90=("ret", lambda x: q(x, 90)), dd_med=("dd", "median"), dd_p10=("dd", lambda x: q(x, 10)),
                                         hit_dd_stop_pct=("halted", lambda x: x.mean() * 100), trades=("trades", "median"))
print("\n60-day outcome (return % median/p10/p90, drawdown %, share of paths halted by the 25% drawdown stop):")
print(tab.reindex(KINDS, level=0).round(1).to_string())
e = R[R.future == "ends (day 20)"].groupby("design").agg(trades_after_end=("trades_after_end", "median"), days_until_last_trade_after_end=("last_trade_after_end_day", "median"),
                                                     pnl_after_end_pct_med=("pnl_after_end_pct", "median"), pnl_after_end_pct_p10=("pnl_after_end_pct", lambda x: q(x, 10)))
print("\nRegime ends at day 20 - how fast does the engine stop, and what does the lag cost?")
print(e.round(1).to_string())
mv = R[R.future.str.startswith("moves")].groupby("design").agg(share_trades_19_20=("share_trades_19_20", "median"), first_trade_day=("first_trade_day", "median"))
print("\nEdge moves to 19-20 UTC - how much of the trading lands there, and how soon is the first trade?")
print(mv.round(2).to_string())
