"""(1) Every GUI preset on one footing, (2) what aggressive sizing (Kelly 1.0x, 5% stake, 200,000 max contracts) does to R2: return, drawdown, size feasibility, slippage sensitivity.
Real engine and real quotes, GUI-like defaults (4c edge hurdle, drawdown throttle, daily-loss 10% / drawdown-stop 25%, min history 30, 2c slippage, $10k, compounding)."""
import sys
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

rows = load_rows()
_, hi = data_range(rows)
CUT = pd.Timestamp("2026-08-24", tz="UTC").timestamp()
pd.set_option("display.width", 230)
ALL = tuple(range(24))
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, compounding=True, p_basis="edge")


def run(start, p, **kw):
    cfg = dict(GUI)
    cfg.update(kw)
    return run_backtest(rows, BacktestConfig(start=start, end=hi.date(), params=p, **cfg))


def line(r):
    m = r.metrics
    if not m["trades"]:
        return dict(ret="no trades")
    t = r.trades
    return dict(ret=m["total_return"] * 100, sharpe=m["sharpe"], maxdd=m["max_drawdown"] * 100, trades=m["trades"], win=m["win_rate"] * 100,
                max_contracts=int(t.contracts.max()), max_stake=float(t.stake.max()), final_eq=m["final_equity"], paused=m.get("windows_paused", 0), halted=m.get("windows_halted", 0))


# ── 1. presets ──────────────────────────────────────────────────────────────
PRE = {
    "R2+  core 22-23 + all-hours scout (default)": (StrategyParams(hours=ALL, cost_min=.15, cost_max=.35), dict(p_model="by_hour", hour_k=100.0, halflife_days=14.0, core_hours=(22, 23), core_k=10.0)),
    "R2   22-23 only": (StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35), dict(p_model="overall", halflife_days=10.0)),
    "R3   20-23, hour-adaptive": (StrategyParams(hours=(20, 21, 22, 23), cost_min=.15, cost_max=.35), dict(p_model="by_hour", hour_k=40.0, lookback_days=45)),
    "R1   22-23, 5-30c (original)": (StrategyParams(hours=(22, 23), cost_min=.05, cost_max=.30), dict(p_model="overall", lookback_days=45)),
    "All-day underdog, adaptive": (StrategyParams(hours=ALL, cost_min=.15, cost_max=.35), dict(p_model="by_hour", hour_k=40.0, lookback_days=45)),
    "All-day underdog + favourite": (StrategyParams(hours=ALL, cost_min=.15, cost_max=.35, legs=("DOG", "FAV")), dict(p_model="by_hour", hour_k=40.0, lookback_days=45)),
    "Favourite only (min 5-11)": (StrategyParams(hours=ALL, minutes=tuple(range(5, 12)), legs=("FAV",)), dict(p_model="by_hour", hour_k=40.0, lookback_days=45)),
}
which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "presets"):
    out = []
    for nm, (p, kw) in PRE.items():
        for lab, start in (("Jun 1 -> now", date(2026, 6, 1)),) + ((("Dec 15 -> now", date(2025, 12, 15)),) if max(p.hours) >= 20 and len(p.hours) <= 4 else ()):
            d = line(run(start, p, **kw))
            out.append(dict(preset=nm, period=lab, **d))
    print("=== 1. Presets on one footing (GUI defaults: 0.25x Kelly, 2% cap, 4c hurdle, throttle, 10%/25% limits, 2c slippage, $10k) ===")
    print(pd.DataFrame(out).round(1).to_string(index=False))

# ── 2. aggressive sizing ────────────────────────────────────────────────────
if which in ("all", "aggr"):
    R2 = StrategyParams(hours=(22, 23), cost_min=.15, cost_max=.35)
    R2KW = dict(p_model="overall", halflife_days=10.0)
    out = []
    cases = {
        "R2 default sizing (0.25x, 2%, 2,000)": dict(kelly_mult=0.25, max_frac=0.02, max_contracts=2000),
        "YOURS 1.0x / 5% / 200,000 (all safety rules on)": dict(kelly_mult=1.0, max_frac=0.05, max_contracts=200000),
        "YOURS, throttle OFF": dict(kelly_mult=1.0, max_frac=0.05, max_contracts=200000, dd_throttle=False),
        "YOURS, throttle + limits OFF": dict(kelly_mult=1.0, max_frac=0.05, max_contracts=200000, dd_throttle=False, daily_loss_limit=0.0, max_drawdown_stop=0.0),
        "YOURS, limits OFF, throttle on": dict(kelly_mult=1.0, max_frac=0.05, max_contracts=200000, daily_loss_limit=0.0, max_drawdown_stop=0.0),
        "0.5x / 3% / 20,000": dict(kelly_mult=0.5, max_frac=0.03, max_contracts=20000),
    }
    for nm, kw in cases.items():
        for lab, start in (("Jun 1 -> now", date(2026, 6, 1)), ("Dec 15 -> now", date(2025, 12, 15))):
            out.append(dict(case=nm, period=lab, **line(run(start, R2, **R2KW, **kw))))
    print("\n=== 2. R2 (22-23 only) with different sizing ===")
    print(pd.DataFrame(out).round(1).to_string(index=False))

    out = []
    for sl in (0.02, 0.03, 0.04, 0.06, 0.08):
        for nm, kw in (("default", cases["R2 default sizing (0.25x, 2%, 2,000)"]), ("YOURS", cases["YOURS 1.0x / 5% / 200,000 (all safety rules on)"]), ("YOURS no limits", cases["YOURS, throttle + limits OFF"])):
            out.append(dict(slippage_c=sl * 100, sizing=nm, **line(run(date(2025, 12, 15), R2, **R2KW, **{**kw, "slippage": sl}))))
    print("\n=== 3. Slippage sensitivity (Dec 15 -> now). Large orders move the price: the bigger the stake, the more slippage you really pay ===")
    print(pd.DataFrame(out)[["slippage_c", "sizing", "ret", "sharpe", "maxdd", "trades", "max_contracts", "max_stake", "halted"]].round(1).to_string(index=False))
