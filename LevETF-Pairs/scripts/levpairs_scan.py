"""
Research scan: which time frame / spread definition / OU window / thresholds work, judged on the IN-SAMPLE years only (2017-2021); 2022-2026 is reported next to it but never used to rank.
Fixed gross (no vol target) so the numbers are the strategy's own, equal weights over the three pairs and each pair alone.

  python levpairs_scan.py [tf ...] [--small]    -> reports/scan_<tf>m.csv
"""
from __future__ import annotations

import itertools
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

import levpairs_bridge as BR
import levpairs_data as D
from levpairs_core import SigCfg
from levpairs_sim import PAIR_ORDER, SimCfg, metrics, simulate

IS_END = pd.Timestamp("2021-12-31")


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 60 and x.std() > 0 else float("nan")


def scan_tf(args) -> pd.DataFrame:
    tf, small, feed = args
    panel = BR.load_panel(tf, feed, log=lambda m: None)
    rows = []
    sig_grid = list(itertools.product(("chain", "reset"), ("rolling", "fixed"), (2.0, 5.0, 10.0)))
    rule_grid = list(itertools.product((1.0, 1.5, 2.0, 2.5, 3.0), (0.0, 0.5), (0.0, 1.5), (60.0, 0.0)))
    if small:
        sig_grid = [("reset", "fixed", 5.0), ("reset", "rolling", 5.0), ("chain", "fixed", 5.0)]
        rule_grid = list(itertools.product((1.5, 2.0, 3.0), (0.5,), (0.0, 1.5), (60.0,)))
    base = SimCfg(start="2017-01-01", vol_on=False, g_init=1.0, weight_mode="equal", slip_bp=0.3)
    t0 = time.time()
    for spread, beta_mode, win in sig_grid:
        sc = SigCfg(tf=tf, spread=spread, beta_mode=beta_mode, win_days=win)
        sigs = BR.get_signals(panel, sc, feed, log=lambda m: None)
        for so, scl, er, hl in rule_grid:
            c = replace(base, s_open=so, s_close_long=scl, s_close_short=scl, min_edge_ratio=er, max_halflife_min=hl)
            res = simulate(sigs, c)
            u, tr = res.ret, res.trades
            row = dict(tf=tf, spread=spread, beta=beta_mode, win=win, s_open=so, s_close=scl, edge_ratio=er, hl_max=hl,
                       sharpe_is=sharpe(u[u.index <= IS_END]), sharpe_oos=sharpe(u[u.index > IS_END]), trades_is=int((tr.date <= IS_END).sum()),
                       net_bp_is=float(tr[tr.date <= IS_END].net_bp.mean()) if (tr.date <= IS_END).any() else np.nan, gross_bp_is=float(tr[tr.date <= IS_END].gross_bp.mean()) if (tr.date <= IS_END).any() else np.nan)
            for k in PAIR_ORDER:
                if k in res.unit:
                    pu = res.unit[k]
                    row[f"sh_is_{k}"], row[f"sh_oos_{k}"] = sharpe(pu[pu.index <= IS_END]), sharpe(pu[pu.index > IS_END])
            rows.append(row)
        print(f"  {tf}m {spread}/{beta_mode}/win{win}: {len(rule_grid)} rule sets done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(D.REPORTS, f"scan_{tf}m{'_small' if small else ''}.csv"), index=False)
    return df


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    tfs = [int(a) for a in argv] or [5, 15, 30]
    small = "--small" in sys.argv
    with ProcessPoolExecutor(min(len(tfs), 4)) as ex:
        for df in ex.map(scan_tf, [(tf, small if tf == 1 else False, "sip") for tf in tfs]):
            if len(df):
                top = df.sort_values("sharpe_is", ascending=False).head(6)
                print(f"\n== {df.tf.iloc[0]}m top 6 by in-sample Sharpe")
                print(top[["spread", "beta", "win", "s_open", "s_close", "edge_ratio", "hl_max", "sharpe_is", "sharpe_oos", "trades_is", "net_bp_is", "gross_bp_is"]].round(2).to_string(index=False))
