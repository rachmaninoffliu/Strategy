"""Shared setup for the research scripts: paths, the three strategy classes, a cached context and a compact result row."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))
for k in ("ResMomentum", "ResMeanReversion", "ResPrediction"):
    sys.path.insert(0, os.path.join(ROOT, k))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from resarb import backtest as BT  # noqa: E402
from resarb import strategy as S  # noqa: E402
from strategy_resmom import ResidualMomentum  # noqa: E402,F401
from strategy_resmr import ResidualMeanReversion  # noqa: E402,F401
from strategy_respred import ResidualPrediction  # noqa: E402,F401

OUT = os.path.join(HERE, "reports")
os.makedirs(OUT, exist_ok=True)
START, SPLIT = "2016-09-01", "2023-01-01"


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


def row(bt, **cfg) -> dict:
    st = bt.stats
    a, i, o, g = st.get("all", {}), st.get("is", {}), st.get("oos", {}), st.get("gross_all", {})
    gi = BT._block(bt.daily.gross_ret[bt.daily.index < pd.Timestamp(SPLIT)])
    return {**cfg, "sh_is": i.get("sharpe"), "sh_oos": o.get("sharpe"), "sh_all": a.get("sharpe"), "gross_sh_is": gi.get("sharpe"), "gross_sh_all": g.get("sharpe"),
            "cagr_is": i.get("cagr"), "cagr_oos": o.get("cagr"), "vol": a.get("vol"), "mdd": a.get("max_dd"), "turn": st.get("turnover_ann"), "hold_d": st.get("hold_days"),
            "cost_yr": st.get("cost_ann"), "beta": st.get("beta_spy"), "names": st.get("avg_names")}


def run_bt(strat, ctx, sc=None):
    sc = strat.score(ctx) if sc is None else sc
    return BT.run(ctx.p, ctx.res, sc, ctx.member, strat.decision_bars(ctx.p), strat.port_cfg(), strat.bt_cfg())
