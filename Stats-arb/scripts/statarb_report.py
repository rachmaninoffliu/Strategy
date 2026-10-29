"""
Reproducible research run for the stat-arb strategy: python statarb_report.py [--universe core|broad] [--preset NAME] [--trials N]

Writes to Stats-arb/reports/<run name>/ : metrics.json, daily.csv, trades.csv, yearly.csv, ablation.csv, cost.csv, sensitivity.csv, stress.csv, risk.json, equity.png
``--trials`` is how many strategy variants were looked at before this one (it deflates the Sharpe: Bailey & Lopez de Prado). No network access except the optional data refresh.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statarb_bridge as BR
import statarb_data as D
from statarb_sim import dsr, metrics, psr, spy_open_to_open, yearly
from statarb_ui.presets import PRESETS, cfgs

STRESS = [("Dec 2018 sell-off", "2018-10-01", "2018-12-24"), ("COVID crash", "2020-02-19", "2020-04-07"), ("2022 bear market (full year)", "2022-01-03", "2022-10-14"),
          ("SVB / regional banks", "2023-03-08", "2023-03-24"), ("Yen-carry unwind", "2024-07-31", "2024-08-12"), ("Liberation-Day tariffs", "2025-04-02", "2025-04-30"),
          ("Jan 2026 junk squeeze / quant drawdown", "2026-01-02", "2026-01-30"), ("Iran strikes / Hormuz", "2026-02-25", "2026-04-15"), ("Jun-Jul 2026 momentum unwind", "2026-06-22", "2026-07-31"),
          ("Last 60 sessions", None, None)]


def j(x):
    if isinstance(x, dict):
        return {k: j(v) for k, v in x.items()}
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="core")
    from statarb_ui.presets import DEFAULT_PRESET
    ap.add_argument("--preset", default=DEFAULT_PRESET)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--split", default="2022-01-01")
    ap.add_argument("--start", default="2017-06-01")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-extras", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(D.REPORTS, f"report_{a.universe}_{''.join(c for c in a.preset.lower() if c.isalnum())[:24]}")
    os.makedirs(out, exist_ok=True)
    f, p = cfgs(a.preset, a.universe)
    from dataclasses import replace
    p = replace(p, start=a.start, capital=a.capital)
    print(f"universe={a.universe} preset={a.preset!r} -> {out}", flush=True)
    panel = BR.load_panel(a.universe)
    print(panel.describe(), flush=True)
    sig = BR.get_signals(panel, a.universe, f, log=lambda m: print(" ", m, flush=True))
    bt = BR.run_backtest(panel, sig, p, a.split)
    r = bt.result
    d = r.daily
    spy = bt.bench
    d.to_csv(os.path.join(out, "daily.csv"))
    r.trades.to_csv(os.path.join(out, "trades.csv"), index=False)
    bt.yearly.to_csv(os.path.join(out, "yearly.csv"))
    # variance of Sharpe across the variants we looked at is approximated by the standard error of this sample (conservative for the deflation)
    sr_d = bt.m_all["sharpe"] / math.sqrt(252)
    m = {k: bt.m_all for k in ()}
    allm = {"all": bt.m_all, "in_sample": bt.m_is, "out_of_sample": bt.m_oos}
    se_d = math.sqrt((1 + sr_d ** 2 / 2) / bt.m_all["n"])
    allm["all"]["dsr"] = dsr(sr_d, bt.m_all["n"], bt.m_all["skew"], bt.m_all["kurt"], max(a.trials, 2), se_d ** 2)
    allm["all"]["dsr_cross_trial_0.2"] = dsr(sr_d, bt.m_all["n"], bt.m_all["skew"], bt.m_all["kurt"], max(a.trials, 2), (0.2 / math.sqrt(252)) ** 2)    # variants differ by ~0.2 Sharpe units
    allm["all"]["trials"] = a.trials
    json.dump(j(allm), open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print("ALL", {k: round(v, 3) for k, v in bt.m_all.items() if isinstance(v, float)}, flush=True)
    print("IS ", {k: round(v, 3) for k, v in bt.m_is.items() if isinstance(v, float) and k in ("sharpe", "cagr", "vol", "maxdd")}, flush=True)
    print("OOS", {k: round(v, 3) for k, v in bt.m_oos.items() if isinstance(v, float) and k in ("sharpe", "cagr", "vol", "maxdd")}, flush=True)
    # stress windows
    rows = []
    ret = bt.ret
    for name, s0, s1 in STRESS:
        rr = ret.iloc[-60:] if s0 is None else ret[s0:s1]
        sp = spy.reindex(rr.index).fillna(0.0)
        if len(rr) < 3:
            continue
        rows.append(dict(window=name, days=len(rr), strategy=float((1 + rr).prod() - 1), spy=float((1 + sp).prod() - 1), worst_day=float(rr.min()), max_dd=float(((1 + rr).cumprod() / (1 + rr).cumprod().cummax() - 1).min())))
    pd.DataFrame(rows).to_csv(os.path.join(out, "stress.csv"), index=False)
    # risk
    q = ret.quantile([0.01, 0.05])
    risk = dict(var95_1d=float(-q[0.05]), var99_1d=float(-q[0.01]), cvar95_1d=float(-ret[ret <= q[0.05]].mean()), cvar99_1d=float(-ret[ret <= q[0.01]].mean()),
                var95_1d_usd=float(-q[0.05] * a.capital), cvar99_1d_usd=float(-ret[ret <= q[0.01]].mean() * a.capital), worst_day=float(ret.min()), worst_day_date=str(ret.idxmin().date()),
                avg_gross=float(d.gross.mean()), max_gross=float(d.gross.max()), avg_beta=float(d.beta.mean()), avg_turnover=float(d.turnover.mean()), cost_per_year=float(d.cost.mean() * 252),
                borrow_per_year=float(d.borrow.mean() * 252), avg_names=float(d.n_long.mean() + d.n_short.mean()), avg_eap_names=float(d.n_eap.mean()), pct_days_flat=float((d.gross < 0.05).mean()),
                ann_vol_realised=float(ret.std() * math.sqrt(252)), corr_spy=float(np.corrcoef(ret.reindex(spy.index).dropna(), spy.reindex(ret.index).dropna())[0, 1]) if len(ret) > 30 else None)
    json.dump(j(risk), open(os.path.join(out, "risk.json"), "w"), indent=1)
    print("RISK", {k: round(v, 4) if isinstance(v, float) else v for k, v in risk.items()}, flush=True)
    if not a.no_extras:
        print("ablation ...", flush=True)
        BR.ablation(panel, sig, p, a.split, log=lambda m: print(" ", m, flush=True)).to_csv(os.path.join(out, "ablation.csv"))
        print("costs / sensitivity ...", flush=True)
        BR.cost_breakeven(panel, sig, p).to_csv(os.path.join(out, "cost.csv"), index=False)
        BR.sensitivity(panel, sig, p).to_csv(os.path.join(out, "sensitivity.csv"), index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1, 1.4]}, sharex=True)
        ax[0].plot(bt.equity.index, bt.equity, label="strategy (net)", lw=1.6)
        ax[0].plot(bt.equity.index, bt.capital * (1 + spy.reindex(bt.equity.index).fillna(0)).cumprod(), label="SPY", lw=1, color="gray")
        ax[0].axvline(bt.split, color="orange", ls="--", lw=0.8); ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title(f"{a.preset} - {a.universe} universe")
        dd = bt.equity / bt.equity.cummax() - 1
        ax[1].fill_between(dd.index, dd * 100, 0, color="red", alpha=0.4); ax[1].set_ylabel("drawdown %")
        rs = ret.rolling(252).apply(lambda x: x.mean() / x.std() * math.sqrt(252) if x.std() > 0 else np.nan, raw=True)
        ax[2].plot(rs.index, rs, lw=1); ax[2].axhline(0, color="gray", lw=0.8); ax[2].set_ylabel("rolling 1y Sharpe")
        fig.tight_layout(); fig.savefig(os.path.join(out, "equity.png"), dpi=110)
    except Exception as e:                                                             # noqa: BLE001
        print("plot failed:", e)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
