"""
Alpha Syndicate — Can anything beat QQQ on SHARPE? (2016-2026)
Data: Alpaca SIP daily (cached) | Indicators: TA-Lib

OBJECTIVE SET BY SYSTEM OWNER: Sharpe > QQQ buy & hold.
Measured bar: QQQ Sharpe = 0.94 (CAGR 20.20%, vol 22.29%).

METHODOLOGY NOTE (Valid) — READ BEFORE INTERPRETING:
  "Search until something beats the benchmark" is the exact process that
  produced the NDX 13.62% IS -> 4.60% OOS gap. To avoid repeating it:

    1. Hypotheses are PRE-REGISTERED below with a literature citation each.
       No parameter is swept. Each gets ONE canonical parameterisation
       taken from its source paper.
    2. Deflated Sharpe uses n_trials = 20, covering the WHOLE research
       programme to date (10 design variants + daily + 2x 30min + these).
    3. A higher point-estimate Sharpe is NOT evidence. We test whether the
       DIFFERENCE vs QQQ is statistically significant using the
       Jobson-Korkie test with Memmel (2003) correction.

  Point estimates lie. The JK test and DSR are the decision criteria.

PRE-REGISTERED HYPOTHESES:
  C0  QQQ buy & hold                    -- the bar
  C1  60/40 QQQ/cash                    -- naive de-risking control
  H1  Vol-managed QQQ                   -- Moreira & Muir (2017), J.Finance 72(4)
  H2  Trend-following QQQ (200d SMA)    -- Faber (2007), "Quantitative Approach
                                           to Tactical Asset Allocation"
  H3  Vol-managed + trend QQQ           -- H1 x H2 combined
  H4  Time-series momentum, ETF universe-- Moskowitz, Ooi & Pedersen (2012),
                                           J.Fin.Econ 104(2)
  H5  MHDM quarterly rebalance          -- lower-turnover variant; the 30min
                                           run showed gross alpha being eaten
                                           by trading costs

HONEST PRIOR (QR): over 2016-2026 QQQ compounded at 20.2% through a
mega-cap-tech + ZIRP + AI regime. Beating its Sharpe on THIS window mostly
rewards holding more tech beta. Expect most hypotheses to fail; the useful
output is understanding WHY, not finding a winner.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
import talib
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- CONFIG -------------------------------------------------------------------
TRADING_DAYS   = 252
N_TRIALS_DSR   = 20            # whole research programme, honestly counted
COST_BPS       = 3.0           # 1bp commission + 2bp slippage
VOL_TARGET     = 0.15          # Moreira-Muir style scaling target
MAX_LEVERAGE   = 2.0           # canonical cap; MM allow leverage
VOL_LOOKBACK   = 21            # MM use ~1 month realised vol
TREND_SMA      = 200           # Faber
TSMOM_LOOKBACK = 252           # MOP 12-month
BENCH          = "QQQ"
CASH           = "BIL"

DATA   = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data/mhdm_etf_bars.parquet"
OUTDIR = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
os.makedirs(OUTDIR, exist_ok=True)


# -- DATA ---------------------------------------------------------------------
def load():
    p = pd.read_parquet(DATA)
    # Normalise to midnight so this aligns with mhdm_etf_backtest.py's index.
    p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None).dt.normalize()
    px = {}
    for c in ("open", "high", "low", "close"):
        px[c] = p.pivot_table(index="timestamp", columns="symbol",
                              values=c).sort_index().ffill()
    idx = px["close"].dropna().index
    return {k: v.reindex(idx) for k, v in px.items()}


# -- STATISTICS ---------------------------------------------------------------
def sharpe_diff_test(ra, rb):
    """Jobson-Korkie test with Memmel (2003) correction.

    H0: SR_a == SR_b. Returns (z, p_value, two-sided).
    This is the correct way to ask "is this Sharpe actually higher?" --
    comparing point estimates is not.
    """
    df = pd.concat([pd.Series(ra), pd.Series(rb)], axis=1).dropna()
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    T = len(a)
    if T < 30:
        return np.nan, np.nan
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    theta = (1.0 / T) * (2 * (1 - rho)
                         + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2))
    if theta <= 0:
        return np.nan, np.nan
    z = (sa - sb) / math.sqrt(theta)
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def deflated_sharpe(r, n_trials):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurtosis()) + 3.0
    gamma, e = 0.5772156649, math.e
    sr_std = math.sqrt(max((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1), 1e-18))
    sr0 = sr_std * ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
                    + gamma * norm.ppf(1 - 1.0 / (n_trials * e)))
    den = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-18))
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / den))


def metrics(eq, label, bench_rets=None):
    r = eq.pct_change().dropna()
    yrs = len(eq) / TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    m = {
        "label": label,
        "cagr": eq.iloc[-1] ** (1 / yrs) - 1,
        "vol": vol,
        "sharpe": (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "dsr": deflated_sharpe(r, N_TRIALS_DSR),
    }
    if bench_rets is not None:
        z, p = sharpe_diff_test(r, bench_rets)
        m["z"], m["p"] = z, p
    return m


# -- STRATEGY BUILDERS --------------------------------------------------------
def apply_costs(eq_ret, weights):
    """Subtract turnover cost from a daily return series given daily weights."""
    turn = weights.diff().abs().fillna(0.0)
    return eq_ret - turn * COST_BPS / 1e4


def bh(px, sym):
    e = px["close"][sym]
    return (e / e.iloc[0]).rename(sym)


def static_mix(px, sym, w):
    r = px["close"][sym].pct_change().fillna(0.0) * w
    return (1 + r).cumprod()


def vol_managed(px, sym, use_trend=False):
    """Moreira & Muir (2017): scale exposure by inverse realised volatility.

    w_t = min(target / realised_vol_{t-1}, MAX_LEVERAGE)
    Uses t-1 vol only -- no look-ahead.
    """
    c = px["close"][sym]
    r = c.pct_change().fillna(0.0)
    rv = r.rolling(VOL_LOOKBACK).std() * np.sqrt(TRADING_DAYS)
    w = (VOL_TARGET / rv.shift(1)).clip(upper=MAX_LEVERAGE).fillna(0.0)
    if use_trend:
        sma = pd.Series(talib.SMA(c.to_numpy(dtype=float), TREND_SMA), index=c.index)
        w = w.where(c.shift(1) > sma.shift(1), 0.0)
    net = apply_costs(w * r, w)
    return (1 + net).cumprod()


def trend_only(px, sym):
    """Faber (2007): long when price > 200d SMA, else cash."""
    c = px["close"][sym]
    r = c.pct_change().fillna(0.0)
    sma = pd.Series(talib.SMA(c.to_numpy(dtype=float), TREND_SMA), index=c.index)
    w = (c.shift(1) > sma.shift(1)).astype(float)
    net = apply_costs(w * r, w)
    return (1 + net).cumprod()


def tsmom(px, universe):
    """Moskowitz, Ooi & Pedersen (2012): own-trend sign, vol-scaled, equal risk.

    Each asset independently long if its own 12m return > 0. Position scaled
    to equal risk contribution. This is TIME-SERIES momentum, contrasted with
    the cross-sectional ranking used in MHDM.
    """
    c = px["close"][universe]
    r = c.pct_change().fillna(0.0)
    sig = (c.pct_change(TSMOM_LOOKBACK).shift(1) > 0).astype(float)
    rv = r.rolling(60).std().shift(1) * np.sqrt(TRADING_DAYS)
    inv = (VOL_TARGET / rv).clip(upper=MAX_LEVERAGE)
    w = (sig * inv)
    w = w.div(len(universe))                      # equal risk budget per sleeve
    gross = w.sum(axis=1).clip(upper=1.0)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).mul(gross, axis=0).fillna(0.0)
    port = (w * r).sum(axis=1)
    cost = w.diff().abs().sum(axis=1).fillna(0.0) * COST_BPS / 1e4
    return (1 + port - cost).cumprod()


# -- MAIN ---------------------------------------------------------------------
def main():
    print("=" * 96)
    print("SHARPE RESEARCH — target: beat QQQ buy & hold")
    print("=" * 96)

    px = load()
    idx = px["close"].index
    print(f"Window: {idx[0].date()} -> {idx[-1].date()}  ({len(idx)} days)")
    print(f"Pre-registered hypotheses: 5 | DSR n_trials = {N_TRIALS_DSR}\n")

    bench_eq = bh(px, BENCH)
    bench_r = bench_eq.pct_change().dropna()

    equity_uni = [s for s in ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
                              "XLU", "SPY", "QQQ", "IWM", "TLT", "GLD"]
                  if s in px["close"].columns]

    runs = {
        "C0  QQQ buy & hold":            bench_eq,
        "C1  60/40 QQQ/cash":            static_mix(px, BENCH, 0.60),
        "H1  Vol-managed QQQ (MM2017)":  vol_managed(px, BENCH),
        "H2  Trend QQQ 200d (Faber07)":  trend_only(px, BENCH),
        "H3  Vol-managed + trend QQQ":   vol_managed(px, BENCH, use_trend=True),
        "H4  TSMOM universe (MOP2012)":  tsmom(px, equity_uni),
    }

    # H5: reuse the MHDM daily equity curve already produced.
    mhdm_csv = os.path.join(OUTDIR, "mhdm_etf_equity.csv")
    if os.path.exists(mhdm_csv):
        m = pd.read_csv(mhdm_csv, index_col=0, parse_dates=True)
        runs["H5  MHDM daily (built earlier)"] = m["equity"].reindex(idx).ffill().dropna()

    rows = []
    for label, eq in runs.items():
        eq = eq.dropna()
        eq = eq / eq.iloc[0]
        rows.append(metrics(eq, label, bench_r))

    print("=" * 96)
    print(f"{'':<32}{'CAGR':>8}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}"
          f"{'DSR':>7}{'vs QQQ p':>10}")
    print("=" * 96)
    for m in rows:
        p = m.get("p", np.nan)
        pstr = f"{p:8.3f}" if pd.notna(p) else "     n/a"
        print(f"  {m['label']:<30}{m['cagr']*100:7.2f}%{m['vol']*100:7.2f}%"
              f"{m['sharpe']:8.2f}{m['max_dd']*100:8.2f}%"
              f"{m['dsr']:7.3f}{pstr}")

    # -- Verdict ---------------------------------------------------------------
    bench_sharpe = rows[0]["sharpe"]
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  Bar to clear: QQQ Sharpe = {bench_sharpe:.2f}\n")

    winners = []
    for m in rows[1:]:
        higher = m["sharpe"] > bench_sharpe
        sig = pd.notna(m.get("p")) and m["p"] < 0.05
        dsr_ok = pd.notna(m["dsr"]) and m["dsr"] >= 0.95
        if higher and sig and dsr_ok:
            verdict, winners = "ACCEPT", winners + [m]
        elif higher and not sig:
            verdict = "higher point estimate, NOT significant (p>=0.05)"
        elif higher and not dsr_ok:
            verdict = "higher + significant, but DSR<0.95 (selection bias)"
        else:
            verdict = "does not beat QQQ"
        print(f"  {m['label']:<32} {m['sharpe']:.2f}  ->  {verdict}")

    print("\n" + "-" * 96)
    if winners:
        print(f"  {len(winners)} hypothesis/es cleared ALL THREE gates "
              f"(higher Sharpe + p<0.05 + DSR>=0.95):")
        for m in winners:
            print(f"    {m['label']}  Sharpe {m['sharpe']:.2f} "
                  f"(p={m['p']:.3f}, DSR={m['dsr']:.3f})")
    else:
        print("  NO hypothesis cleared all three gates.")
        print("  Any higher point estimate above is within statistical noise of QQQ.")
    print("-" * 96)

    # -- Output ----------------------------------------------------------------
    out = pd.DataFrame({k: v for k, v in runs.items()})
    out.to_csv(os.path.join(OUTDIR, "sharpe_research_equity.csv"))

    fig, ax = plt.subplots(figsize=(13, 7))
    for label, eq in runs.items():
        eq = (eq / eq.dropna().iloc[0]).dropna()
        style = dict(lw=2.0, color="black") if label.startswith("C0") else dict(lw=1.2)
        ax.plot(eq.index, eq, label=f"{label}", **style)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log)")
    ax.set_title(f"Sharpe research vs {BENCH} — {idx[0].date()} to {idx[-1].date()}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUTDIR, "sharpe_research.png")
    plt.savefig(png, dpi=120)
    plt.close()
    print(f"\nSaved: {png}")


if __name__ == "__main__":
    main()
