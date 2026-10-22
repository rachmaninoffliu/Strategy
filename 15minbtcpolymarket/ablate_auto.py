"""Ablation of the Auto layer on the real history (Jun 1 -> now, all hours present, volume available): which component earns its place?
dead = Jun 1 .. Aug 23 (no regime), live = Aug 24 -> now.  Same GUI-like safety rules for every variant."""
import sys
import warnings
from dataclasses import replace
from datetime import date

warnings.filterwarnings("ignore")
import pandas as pd

from trader.adaptive import DEFAULT_LADDER, AutoConfig
from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

rows = load_rows()
_, hi = data_range(rows)
CUT = pd.Timestamp("2026-08-24", tz="UTC").timestamp()
ALL = StrategyParams(hours=tuple(range(24)), cost_min=.15, cost_max=.35)
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, compounding=True, p_basis="edge",
           p_model="by_hour", hour_k=100.0, core_hours=(22, 23), core_k=10.0)
START = date(2026, 6, 1)
FLAT = (("HOT", 1.0), ("BUILDING", 1.0), ("FADING", 1.0), ("COLD", 0.0))


def run(auto=None, **kw):
    cfg = dict(GUI)
    cfg.update(kw)
    return run_backtest(rows, BacktestConfig(start=START, end=hi.date(), params=ALL, auto=auto, **cfg))


def line(nm, r):
    m, t = r.metrics, r.trades
    live = t[t.entry_ts >= CUT]
    first = pd.to_datetime(live.entry_ts.min(), unit="s", utc=True).strftime("%m-%d") if len(live) else "-"
    dead = t[t.entry_ts < CUT]
    return dict(variant=nm, ret=m["total_return"] * 100, sharpe=m["sharpe"], maxdd=m["max_drawdown"] * 100, trades=m["trades"], win=m["win_rate"] * 100, dead_trades=len(dead),
                dead_pnl_k=dead.pnl.sum() / 1000, first_live=first, max_contracts=int(t.contracts.max()), max_stake=float(t.stake.max()))


base = AutoConfig()
V = {}
V["BASELINE R2+ fixed 14d, 0.25x/2%/2000"] = run(None, kelly_mult=.25, max_frac=.02, max_contracts=2000, halflife_days=14.0)
V["BASELINE R2+ fixed 14d, 0.5x/3%/20000"] = run(None, kelly_mult=.5, max_frac=.03, max_contracts=20000, halflife_days=14.0)
# memory: single learners vs the mix (everything else neutral: z=0, flat regime scale, one ladder step, no liquidity cap)
neutral = dict(z=0.0, regime_mult=FLAT, ladder=((0.0, 0.25, 0.02),), participation=0.0, max_contracts=2000)
for h in (7.0, 14.0, 30.0):
    V[f"memory only: single learner {h:g}d"] = run(replace(base, half_lives=(h,), **neutral))
V["memory only: MIX 7/14/30"] = run(replace(base, **neutral))
V["memory only: MIX 5/10/21"] = run(replace(base, half_lives=(5.0, 10.0, 21.0), **neutral))
# evidence haircut
for z in (0.0, 0.25, 0.5, 1.0):
    V[f"+ haircut z={z:g} (mix, flat regime)"] = run(replace(base, **{**neutral, "z": z}))
# regime scaling on top of the mix, z = 0.25
V["+ regime scaling (z=0.25)"] = run(replace(base, **{**neutral, "z": 0.25, "regime_mult": base.regime_mult}))
# equity ladder
for nm, lad in (("ladder default", DEFAULT_LADDER), ("ladder flat 0.5x/3%", ((0.0, 0.5, 0.03),)), ("ladder flat 0.25x/2%", ((0.0, 0.25, 0.02),))):
    V[f"+ {nm} (z=.25, regime)"] = run(replace(base, **{**neutral, "z": 0.25, "regime_mult": base.regime_mult, "ladder": lad, "max_contracts": 20000}))
# liquidity cap
for part in (0.05, 0.10, 0.20, 0.0):
    V[f"+ liquidity {part * 100:.0f}% (ladder default, z=.25, regime)"] = run(replace(base, **{**neutral, "z": 0.25, "regime_mult": base.regime_mult, "ladder": DEFAULT_LADDER, "participation": part, "max_contracts": 200000}))
V["AUTO defaults (z=.5)"] = run(base)
V["AUTO z=.25"] = run(replace(base, z=0.25))
V["AUTO z=0"] = run(replace(base, z=0.0))
D = pd.DataFrame([line(k, v) for k, v in V.items()])
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 60)
print(D.round(1).to_string(index=False))
