"""
Numbers for LEVPAIRS_REPORT.md:  python levpairs_report.py > ../reports/report_tables.txt

  1  the default preset on the three pairs (5-minute bars): all / in-sample / out-of-sample, by year, by pair
  2  the same rules on 1 / 5 / 15 / 30-minute bars: realistic fills, optimistic fills (the bounce bias), and zero cost (the gross edge)
  3  Sharpe against the slippage assumption
  4  ablation
  5  what the 1,458-set scan found
"""
from __future__ import annotations

import glob
import os
from dataclasses import replace

import numpy as np
import pandas as pd

import levpairs_bridge as BR
import levpairs_data as D
from levpairs_sim import fmt_metrics
from levpairs_ui.presets import DEFAULT_PRESET, cfgs

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)


def h(t: str) -> None:
    print(f"\n{'=' * 8} {t}")


if __name__ == "__main__":
    sc, c = cfgs(DEFAULT_PRESET)
    h(f"1. {DEFAULT_PRESET}")
    panel = BR.load_panel(5, "sip", log=lambda m: None)
    sigs = BR.get_signals(panel, sc, "sip", log=lambda m: None)
    bt = BR.run_backtest(panel, sigs, c, "2022-01-01")
    print("all :", fmt_metrics(bt.m_all)); print("IS  :", fmt_metrics(bt.m_is)); print("OOS :", fmt_metrics(bt.m_oos))
    print("notes:", bt.notes)
    print(bt.yearly.round(3).to_string())
    print(bt.pairs.round(3).to_string())
    fixed = BR.run_backtest(panel, sigs, replace(c, vol_on=False, g_init=1.0), "2022-01-01")
    print("fixed gross 1.0x:", fmt_metrics(fixed.m_all))

    h("2. time frames: realistic fills / optimistic fills (fill at the signal bar's close) / zero cost (gross edge)")
    rows = []
    for label, cc in (("next_open", c), ("close (bias)", replace(c, fill="close")), ("zero cost", replace(c, cost_mult=0.0, slip_bp=0.0, min_edge_ratio=0.0))):
        t = BR.timeframe_scan(sc, cc, "sip", "2022-01-01", log=lambda m: None)
        t.insert(0, "fill/cost", label)
        rows.append(t)
    print(pd.concat(rows)[["fill/cost", "sharpe", "sharpe_is", "sharpe_oos", "cagr", "maxdd", "trades", "gross_bp", "net_bp", "cost_pct"]].round(3).to_string())

    h("3. Sharpe vs slippage per leg (5-minute bars, default preset)")
    print(BR.cost_breakeven(panel, sigs, c, slips=(0.0, 0.1, 0.2, 0.3, 0.6, 1.0)).round(3).to_string())

    h("4. ablation")
    print(BR.ablation(panel, sc, c, "sip", "2022-01-01", log=lambda m: None).round(3).to_string())

    h("5. the scan")
    files = [f for f in sorted(glob.glob(os.path.join(D.REPORTS, "scan_*m*.csv")))]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(len(df), "rule sets;", df.groupby("tf").size().to_dict())
    print("share with positive in-sample Sharpe:", round((df.sharpe_is > 0).mean(), 3), "| positive in AND out of sample:", round(((df.sharpe_is > 0) & (df.sharpe_oos > 0)).mean(), 3),
          "| best in-sample Sharpe:", round(df.sharpe_is.max(), 2))
    print(df.groupby(["tf", "spread"])[["sharpe_is", "sharpe_oos", "gross_bp_is", "net_bp_is", "trades_is"]].mean().round(2).to_string())
    best = df.sort_values("sharpe_is", ascending=False).head(5)
    print(best[["tf", "spread", "beta", "win", "s_open", "s_close", "edge_ratio", "hl_max", "sharpe_is", "sharpe_oos", "gross_bp_is", "net_bp_is"]].round(2).to_string(index=False))
