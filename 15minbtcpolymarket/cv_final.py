"""Run the REAL strategy engine (trader.backtest) on a 2.7-year world: BTC-spot-reconstructed Kalshi-style quotes before 2025-12-15, real quotes after.

 1. Stress test: the app's R1/R2/R0 + adaptive probability gate on 23 months of reconstructed quotes (no Kalshi data exists for them).
 2. Fidelity: synthetic vs real over the 10 overlapping months.
 3. All-history equity (synthetic + real).
 4. Is the hour concentration (22-23 UTC) reliable, or the top of a market-wide shift?  Split-half reliability of hourly excess flip rates and an
    empirical-Bayes shrinkage of the 22-23 estimate.
 5. Chart -> cv_report.png
"""
from __future__ import annotations

import warnings
from datetime import date

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv_longspot as lsp
import cv_mechanism as mech
from trader.backtest import BacktestConfig, run_backtest
from trader.data import load_rows
from trader.strategy import StrategyParams

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)
REAL_START = pd.Timestamp("2025-12-15", tz="UTC").timestamp()
HALF = 0.005


def synthetic_rows(w: pd.DataFrame, g) -> pd.DataFrame:
    x = w.copy()
    x["g"] = 0.5
    for mn in g:
        i = x.minute == mn
        x.loc[i, "g"] = g[mn].predict(x.loc[i, "z"].values)
    x["bid"] = np.clip(x.g - HALF, 0.001, 0.998)
    x["ask"] = np.clip(x.g + HALF, 0.002, 0.999)
    x["ticker"] = "S" + x.open_ts.astype(str)
    x["yes"] = x.yes_proxy
    return x[["ticker", "minute", "bid", "ask", "open_ts", "yes"]], x


def summary(res):
    m = res.metrics
    return f"trades {m['trades']:4d} | win {m['win_rate'] * 100:4.1f}% | return {m['total_return'] * 100:+7.1f}% | Sharpe {m['sharpe']:5.2f} | maxDD {m['max_drawdown'] * 100:5.1f}%" if m["trades"] else "no trades"


def main():
    s = lsp.load_spot_all()
    w = lsp.build_windows(s, minutes=(10, 11, 12, 13))
    w12 = w[w.minute.isin([12, 13])]
    g = mech.fit_g(w12)
    # fit g also for minutes 10/11 (needed for the multi-minute presets: R2 uses 12-13 only, so fine)
    syn, wx = synthetic_rows(w12, g)
    real = load_rows()
    P = {"R2 (15-35c)": StrategyParams(hours=(22, 23), cost_min=0.15, cost_max=0.35),
         "R1 (5-30c)": StrategyParams(hours=(22, 23)),
         "R0 (5-35c)": StrategyParams(hours=(22, 23), cost_min=0.05, cost_max=0.35)}

    print("=" * 112)
    print("1. STRESS TEST on reconstructed quotes, 2024-01-20 .. 2025-12-14 (the real engine, 45-day learner, 2c slippage, $10k, quarter-Kelly, 2% cap)")
    print("   Spot data are complete here only where the download finished - months without spot data are skipped automatically.")
    print("=" * 112)
    for nm, p in P.items():
        res = run_backtest(syn, BacktestConfig(start=date(2024, 1, 20), end=date(2025, 12, 14), params=p, lookback_days=45))
        print(f"  {nm:12s} {summary(res)}")
    res_nogate = run_backtest(syn, BacktestConfig(start=date(2024, 1, 20), end=date(2025, 12, 14), params=P["R2 (15-35c)"], prob_mode="Fixed value", fixed_p=0.30))
    print(f"  R2 with NO learning gate (always bets as if p = 30%): {summary(res_nogate)}   <- what an always-on rule loses in the adverse regime")

    print("\n" + "=" * 112)
    print("2. FIDELITY - synthetic vs real over the same 10 months (2025-12-15 .. 2026-10-10), R2")
    print("=" * 112)
    syn_ov = syn[(syn.open_ts >= REAL_START)]
    r_syn = run_backtest(syn_ov, BacktestConfig(start=date(2025, 12, 15), end=date(2026, 10, 10), params=P["R2 (15-35c)"], lookback_days=45))
    r_real = run_backtest(real, BacktestConfig(start=date(2025, 12, 15), end=date(2026, 10, 10), params=P["R2 (15-35c)"], lookback_days=45))
    print(f"  synthetic quotes: {summary(r_syn)}")
    print(f"  real quotes     : {summary(r_real)}")

    print("\n" + "=" * 112)
    print("3. ALL-HISTORY world (synthetic until 2025-12-14, then real quotes)")
    print("=" * 112)
    allrows = pd.concat([syn[syn.open_ts < REAL_START], real[["ticker", "minute", "bid", "ask", "open_ts", "yes"]]], ignore_index=True)
    curves = {}
    for nm, p in P.items():
        res = run_backtest(allrows, BacktestConfig(start=date(2024, 1, 20), end=date(2026, 10, 10), params=p, lookback_days=45))
        curves[nm] = res
        t = res.trades
        t["q"] = pd.to_datetime(t.entry_ts, unit="s", utc=True).dt.to_period("Q").astype(str)
        by_q = t.groupby("q").agg(n=("pnl", "size"), pnl=("pnl", "sum"))
        print(f"  {nm:12s} {summary(res)}")
        print("     P&L by quarter ($):", "  ".join(f"{q}:{v.pnl:+,.0f}({int(v.n)})" for q, v in by_q.iterrows()))

    print("\n" + "=" * 112)
    print("4. Hour concentration since 2026-08-24: split-half reliability + empirical-Bayes shrinkage (dogs priced 12-38c, minute 12, spot-reconstructed)")
    print("=" * 112)
    x = wx[wx.minute == 12].copy()
    dog_yes = x.g < 0.5
    x["g_dog"] = np.where(dog_yes, x.g, 1 - x.g)
    x["win_dog"] = np.where(dog_yes, x.yes_proxy, 1 - x.yes_proxy)
    x["exc"] = x.win_dog - x.g_dog
    x = x[(x.g_dog >= 0.12) & (x.g_dog <= 0.38)]
    periods = {"2024-01..2025-12-14": (0, REAL_START), "2025-12-15..2026-08-23": (REAL_START, lsp.CUT), "2026-08-24..now": (lsp.CUT, 1e12)}
    for lab, (a, b) in periods.items():
        y = x[(x.open_ts >= a) & (x.open_ts < b)]
        mid_ts = y.open_ts.median()
        h1 = y[y.open_ts < mid_ts].groupby("hour").exc.agg(["mean", "size"])
        h2 = y[y.open_ts >= mid_ts].groupby("hour").exc.agg(["mean", "size"])
        j = h1.join(h2, lsuffix="1", rsuffix="2").dropna()
        r = np.corrcoef(j.mean1, j.mean2)[0, 1]
        hm = y.groupby("hour").exc.agg(["mean", "size", "std"])
        se2 = (hm["std"] ** 2 / hm["size"]).mean()
        tau2 = max(hm["mean"].var() - se2, 0.0)
        grand = y.exc.mean()
        z = y[y.hour.isin([22, 23])]
        se_z = z.exc.std() / np.sqrt(len(z))
        shrink = tau2 / (tau2 + se_z ** 2) if tau2 > 0 else 0.0
        est = grand + shrink * (z.exc.mean() - grand)
        print(f"  {lab:24s} all-hours excess {grand * 100:+5.1f}c | 22-23: {z.exc.mean() * 100:+5.1f}c (n={len(z)}, se {se_z * 100:.1f}c) | "
              f"corr of hourly excess between halves = {r:+.2f} | between-hour sd {np.sqrt(tau2) * 100:4.1f}c | EB-shrunk 22-23 = {est * 100:+5.1f}c (shrink {shrink:.2f})")

    # ── chart ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    z = wx[wx.minute == 12].copy()
    dy = z.g < 0.5
    z["g_dog"] = np.where(dy, z.g, 1 - z.g)
    z["exc"] = np.where(dy, z.yes_proxy, 1 - z.yes_proxy) - z.g_dog
    z = z[(z.g_dog >= 0.12) & (z.g_dog <= 0.38)]
    z["day"] = pd.to_datetime(z.open_ts, unit="s", utc=True).dt.floor("D")
    for lab, mask, col in (("22-23 UTC", z.hour.isin([22, 23]), "#d62728"), ("all other hours", ~z.hour.isin([22, 23]), "#1f77b4")):
        d = z[mask].groupby("day").exc.agg(["sum", "size"]).asfreq("D", fill_value=0)
        roll = d["sum"].rolling(42, min_periods=1).sum() / d["size"].rolling(42, min_periods=1).sum().replace(0, np.nan) * 100
        roll[d["size"].rolling(42, min_periods=1).sum() < 40] = np.nan
        ax[0].plot(roll.index, roll.values, label=lab, color=col, lw=1.6)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].axhline(3.7, color="grey", lw=0.8, ls="--")
    ax[0].text(ax[0].get_xlim()[0], 4.5, " break-even after slippage+fees (~3.7c)", fontsize=8, color="grey")
    ax[0].set_ylabel("underdog excess win rate vs pricing (cents)\nrolling 42 days (spot-reconstructed)")
    ax[0].set_title("Underdogs priced 12-38c with 2-3 minutes left: how much more often they win than the market pricing implies")
    ax[0].legend(loc="upper left")
    for nm, res in curves.items():
        e = res.equity
        ax[1].plot(pd.to_datetime(e.ts, unit="s", utc=True), e.equity, label=nm, lw=1.4)
    ax[1].axvline(pd.Timestamp("2025-12-15", tz="UTC"), color="k", lw=0.8, ls=":")
    ax[1].text(pd.Timestamp("2025-12-20", tz="UTC"), ax[1].get_ylim()[1] * 0.9, "real Kalshi quotes ->", fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("equity ($, log) - real engine, 45d learner")
    ax[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("cv_report.png", dpi=110)
    print("\nchart -> cv_report.png")


if __name__ == "__main__":
    main()
