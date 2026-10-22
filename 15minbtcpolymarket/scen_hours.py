"""How fast does R2+ follow a hot hour that MOVES, and what does being faster cost?  60-day bootstrap futures (days resampled from the post-Aug-24 regime / the quiet months), learner warm-started on
today's history, GUI safety rules, default sizing (0.25x / 2% / 2,000).  Futures:
  persists   regime days unchanged (22-23 UTC hot)           - must not get worse
  quiet      quiet days only                                 - the price of scouting: false trades
  ends       regime days for 20 days, then quiet days
  shift-2    regime days moved 2 h earlier   (hot at 20-21 UTC)
  shift+2    moved 2 h later                 (hot at 00-01 UTC)
  shift-6    moved 6 h earlier               (hot at 16-17 UTC)
Per design: median return / p10, median max drawdown, trades; for the shifted futures the share of trades (and of P&L) taken in the NEW hot hours and the day of the first trade there.
Run:  python scen_hours.py [paths_per_future] [out.csv]   (WORKERS env var = processes, default 6)"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.adaptive import AutoConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
WORKERS = int(os.environ.get("WORKERS", "6"))        # each worker holds the whole data set: too many exhaust Windows' commit limit ("paging file is too small")
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/scenario_hours_results.csv"
HORIZON, END_DAY = 60, 20
KINDS = {"persists": 0, "quiet": 0, "ends": 0, "shift-2": -2, "shift+2": 2, "shift-6": -6}
ALL = dict(hours=tuple(range(24)), cost_min=.15, cost_max=.35)
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, compounding=True, p_basis="edge",
           p_model="by_hour", halflife_days=14.0, kelly_mult=.25, max_frac=.02, max_contracts=2000)
DESIGNS = {                                                          # name -> BacktestConfig overrides
    "AUTO, no near-core": dict(hour_k=100.0, core_hours=(22, 23), core_k=10.0, auto=AutoConfig()),
    "AUTO, near±2 k=50": dict(hour_k=100.0, core_hours=(22, 23), core_k=10.0, near_k=50.0, near_width=2, auto=AutoConfig()),
}
EXTRA = {}                                                           # filled by experiments that add designs (see bottom)

_G = {}


def _init():
    from trader.data import data_range, load_rows
    rows = load_rows()
    _, hi = data_range(rows)
    CUT = int(pd.Timestamp("2026-08-24", tz="UTC").timestamp())
    t_end = int(pd.Timestamp(hi.date(), tz="UTC").timestamp())
    rows["day"] = rows.open_ts // 86400
    _G.update(rows=rows, hi=hi, t_end=t_end,
              days_regime=sorted(rows[(rows.open_ts >= CUT) & (rows.open_ts < t_end)].day.unique()),
              days_quiet=sorted(rows[(rows.open_ts >= int(pd.Timestamp("2025-12-15", tz="UTC").timestamp())) & (rows.open_ts < CUT)].day.unique()),
              by_day={d: g for d, g in rows.groupby("day")})


COLS = ["ticker", "open_ts", "minute", "bid", "ask", "yes", "vol"]


def _future(kind, rng):
    G = _G
    shift = KINDS[kind]
    start_day = G["t_end"] // 86400
    picks = []
    for i in range(HORIZON):
        if kind in ("persists", "shift-2", "shift+2", "shift-6"):
            src = rng.choice(G["days_regime"])
        elif kind == "ends":
            src = rng.choice(G["days_regime"]) if i < END_DAY else rng.choice(G["days_quiet"])
        else:
            src = rng.choice(G["days_quiet"])
        g = G["by_day"][src].copy()
        g["open_ts"] = g.open_ts + (start_day + i - src) * 86400 + shift * 3600
        g["ticker"] = g.ticker + f"_s{i}"
        picks.append(g)
    return pd.concat(picks, ignore_index=True)[COLS]


def run_one(args):
    kind, k, name, over = args
    from datetime import date
    from trader.backtest import BacktestConfig, run_backtest
    from trader.strategy import StrategyParams
    G = _G
    if not G:
        _init()
        G = _G
    rng = np.random.default_rng(1000 * list(KINDS).index(kind) + k + 17)
    allrows = pd.concat([G["rows"][COLS], _future(kind, rng)], ignore_index=True)
    d0 = pd.Timestamp(G["t_end"], unit="s", tz="UTC").date()
    d1 = d0 + timedelta(days=HORIZON - 1)
    r = run_backtest(allrows, BacktestConfig(start=d0, end=d1, params=StrategyParams(**ALL), **{**GUI, **over}))
    m, t = r.metrics, r.trades
    rec = dict(future=kind, design=name, path=k, ret=m.get("total_return", 0.0) * 100, dd=m.get("max_drawdown", 0.0) * 100, trades=m["trades"], halted=int(m.get("windows_halted", 0) > 0))
    if len(t):
        hrs = pd.to_datetime(t.entry_ts, unit="s", utc=True).dt.hour
        day = ((t.entry_ts - G["t_end"]) // 86400).astype(int)
        new_hot = {(22 + KINDS[kind]) % 24, (23 + KINDS[kind]) % 24}
        if KINDS[kind] != 0:
            mask = hrs.isin(new_hot)
            rec["share_new"] = float(mask.mean())
            rec["pnl_new_pct_of_start"] = float(t.pnl[mask].sum() / m["initial_capital"] * 100)
            rec["first_new_day"] = int(day[mask].min()) if mask.any() else np.nan
        else:
            rec["share_core"] = float(hrs.isin([22, 23]).mean())
    elif KINDS[kind] != 0:
        rec.update(share_new=0.0, pnl_new_pct_of_start=0.0, first_new_day=np.nan)
    return rec


def summarize(R: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    q = lambda x, p: np.percentile(x, p)
    tab = R.groupby(["future", "design"]).agg(ret_med=("ret", "median"), ret_p10=("ret", lambda x: q(x, 10)), dd_med=("dd", "median"), trades=("trades", "median"))
    order = list(KINDS)
    print("\n60-day outcome (return % median / 10th pct, median max drawdown %, trades):")
    print(tab.reindex(order, level=0).round(1).to_string())
    if "share_new" in R:
        s = R[R.future.str.startswith("shift")].groupby(["future", "design"]).agg(share_new_hot_hours=("share_new", "median"), pnl_in_new_hot_hours_pct=("pnl_new_pct_of_start", "median"),
                                                                                  first_trade_day_in_new_hours=("first_new_day", "median"), never_found=("first_new_day", lambda x: x.isna().mean() * 100))
        print("\nHot hour moved: how much of the trading / P&L lands in the NEW hot hours, and how soon:")
        print(s.round(2).to_string())


if __name__ == "__main__":
    designs = {**DESIGNS, **EXTRA}
    jobs = [(kind, k, name, over) for kind in KINDS for k in range(N) for name, over in designs.items()]
    print(f"{len(jobs)} runs on {WORKERS} workers", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init) as ex:
        res = list(ex.map(run_one, jobs, chunksize=2))
    R = pd.DataFrame(res)
    R.to_csv(OUT, index=False)
    summarize(R)
