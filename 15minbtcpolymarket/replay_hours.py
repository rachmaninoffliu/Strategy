"""Cold-start replay on the REAL history (no resampling): when does each scouting setting enter the Aug-24 regime, how much does it lose probing in the quiet months before, and what does it earn?
Jun 1 -> now (all hours complete), GUI safety rules, default sizing 0.25x / 2% / 2,000, learner starts with Jun 1 as its first trading day."""
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import pandas as pd

from trader.adaptive import AutoConfig
from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

rows = load_rows()
_, hi = data_range(rows)
CUT = pd.Timestamp("2026-08-24", tz="UTC").timestamp()
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, compounding=True, p_basis="edge", p_model="by_hour", halflife_days=14.0,
           kelly_mult=.25, max_frac=.02, max_contracts=2000)
DES = {
    "AUTO, no near-core": dict(hour_k=100.0, core_hours=(22, 23), core_k=10.0, auto=AutoConfig()),
    "AUTO, near±2 k=50": dict(hour_k=100.0, core_hours=(22, 23), core_k=10.0, near_k=50.0, near_width=2, auto=AutoConfig()),
}
out = []
for nm, over in DES.items():
    r = run_backtest(rows, BacktestConfig(start=date(2026, 6, 1), end=hi.date(), params=StrategyParams(hours=tuple(range(24)), cost_min=.15, cost_max=.35), **GUI, **over))
    t, m = r.trades, r.metrics
    quiet, live = t[t.entry_ts < CUT], t[t.entry_ts >= CUT]
    hrs = pd.to_datetime(t.entry_ts, unit="s", utc=True).dt.hour
    out.append(dict(design=nm, ret=f"{m['total_return'] * 100:+.0f}%", sharpe=round(m["sharpe"], 2), maxDD=f"{m['max_drawdown'] * 100:.0f}%", trades=m["trades"],
                    quiet_trades=len(quiet), quiet_pnl_pct=f"{quiet.pnl.sum() / 10000 * 100:+.1f}%", first_regime_trade=pd.to_datetime(live.entry_ts.min(), unit="s", utc=True).strftime("%m-%d") if len(live) else "-",
                    outside_22_23=int((~hrs.isin([22, 23])).sum()), outside_pnl_pct=f"{t.pnl[~hrs.isin([22, 23])].sum() / 10000 * 100:+.1f}%"))
pd.set_option("display.width", 250)
print(pd.DataFrame(out).to_string(index=False))
