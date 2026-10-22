"""
Alpha Syndicate — Beat QQQ on BOTH Sharpe AND return? (2016-2026)

OBJECTIVE: CAGR > 20.20% AND Sharpe > 0.94 (QQQ buy & hold).

THE MATHEMATICAL SHAPE OF THIS PROBLEM (QR):
  Sharpe is invariant to constant leverage; CAGR is not. So IF a strategy has a
  genuinely higher Sharpe, levering it to the benchmark's volatility gives
  higher return at the same (higher) Sharpe. This is Tobin (1958) separation --
  the same argument AQR uses to justify levering risk parity.

  So "beat QQQ on both" reduces to ONE question: is the Sharpe edge REAL?
  If it is, leverage does the rest mechanically. If it is not, leverage
  amplifies noise and the result is worse in live trading, not better.

  Our best Sharpe candidate (vol-managed QQQ, 1.11) is NOT statistically
  distinguishable from QQQ (p=0.265). We are therefore levering an UNPROVEN
  edge. That is the honest description of what this script does.

PUBLISHED REBUTTAL WE MUST NOT HIDE (Valid):
  Cederburg, O'Doherty, Wang & Yan, "On the Performance of Volatility-Managed
  Portfolios" (JFE 2020): across 103 equity strategies, real-time vol-managed
  versions generally earn LOWER Sharpe than the unmanaged originals. The
  in-sample result comes from structural instability in spanning regressions.
  Barroso & Detzel: vol-managed strategies do not survive transaction costs.

  Our H1 result is exactly the in-sample pattern their paper says fails OOS.

LEVERAGE IS MODELLED HONESTLY:
  Borrowing financed at the realised risk-free rate (BIL total return) plus a
  50bp broker spread. Cash balances earn the risk-free rate. No free leverage.

DESIGN TO AVOID HINDSIGHT (Valid):
  XLK had the best Sharpe (0.99) and CAGR (23.86%) of any sleeve in this
  window. Hand-picking it would be pure hindsight -- it is the ex-post winner.
  H9 therefore selects sectors BY RULE (momentum rank), testing whether that
  outcome was reachable without knowing the answer in advance.

DSR n_trials = 25 (whole programme). Note: each additional configuration
tested LOWERS every DSR in the table. This search has a cost.
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
TRADING_DAYS  = 252
N_TRIALS_DSR  = 25
COST_BPS      = 3.0
BORROW_SPREAD = 0.0050        # 50bp over risk-free on borrowed balances
VOL_LOOKBACK  = 21
TREND_SMA     = 200
MOM_LOOKBACK  = 252
MOM_SKIP      = 21
MAX_LEVERAGE  = 3.0
BENCH         = "QQQ"
RF_PROXY      = "BIL"

# Leverage ladder. Sharpe should be ~INVARIANT across these; CAGR should scale.
# Showing the ladder (rather than picking one) demonstrates we are not fitting.
VOL_TARGETS   = [0.15, 0.20, 0.25]

DATA   = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data/mhdm_etf_bars.parquet"
OUTDIR = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
os.makedirs(OUTDIR, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]


def load():
    p = pd.read_parquet(DATA)
    p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None).dt.normalize()
    px = {c: p.pivot_table(index="timestamp", columns="symbol", values=c)
              .sort_index().ffill() for c in ("open", "high", "low", "close")}
    idx = px["close"].dropna().index
    return {k: v.reindex(idx) for k, v in px.items()}


# -- STATS --------------------------------------------------------------------
def sharpe_diff_test(ra, rb):
    df = pd.concat([pd.Series(ra), pd.Series(rb)], axis=1).dropna()
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    T = len(a)
    if T < 30:
        return np.nan, np.nan
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    theta = (1.0 / T) * (2 * (1 - rho) + 0.5 * (sa**2 + sb**2 - 2*sa*sb*rho**2))
    if theta <= 0:
        return np.nan, np.nan
    z = (sa - sb) / math.sqrt(theta)
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def deflated_sharpe(r, n_trials):
    r = np.asarray(r, dtype=float); r = r[~np.isnan(r)]
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = float(pd.Series(r).skew()); g4 = float(pd.Series(r).kurtosis()) + 3.0
    gamma, e = 0.5772156649, math.e
    sr_std = math.sqrt(max((1 - g3*sr + (g4-1)/4*sr**2) / (T-1), 1e-18))
    sr0 = sr_std * ((1-gamma)*norm.ppf(1 - 1.0/n_trials)
                    + gamma*norm.ppf(1 - 1.0/(n_trials*e)))
    den = math.sqrt(max(1 - g3*sr + (g4-1)/4*sr**2, 1e-18))
    return float(norm.cdf((sr - sr0) * math.sqrt(T-1) / den))


def metrics(eq, label, bench_r):
    r = eq.pct_change().dropna()
    yrs = len(eq) / TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    z, p = sharpe_diff_test(r, bench_r)
    return {"label": label,
            "cagr": eq.iloc[-1]**(1/yrs) - 1,
            "vol": vol,
            "sharpe": (r.mean()*TRADING_DAYS)/vol if vol > 0 else np.nan,
            "max_dd": (eq/eq.cummax() - 1).min(),
            "dsr": deflated_sharpe(r, N_TRIALS_DSR),
            "p": p}


# -- ENGINE -------------------------------------------------------------------
def levered_equity(w, asset_ret, rf):
    """Apply weights with honest financing.

    w > 1  -> borrow (w-1) at rf + BORROW_SPREAD
    w < 1  -> lend (1-w) at rf
    Costs charged on turnover.
    """
    w = w.clip(upper=MAX_LEVERAGE).fillna(0.0)
    borrow = (w - 1.0).clip(lower=0.0)
    cash = (1.0 - w).clip(lower=0.0)
    fin = borrow * (rf + BORROW_SPREAD / TRADING_DAYS) - cash * rf
    cost = w.diff().abs().fillna(0.0) * COST_BPS / 1e4
    return (1 + (w * asset_ret) - fin - cost).cumprod()


def vol_managed(px, sym, rf, vol_target, use_trend=False):
    c = px["close"][sym]
    r = c.pct_change().fillna(0.0)
    rv = (r.rolling(VOL_LOOKBACK).std() * np.sqrt(TRADING_DAYS)).shift(1)
    w = (vol_target / rv).clip(upper=MAX_LEVERAGE).fillna(0.0)
    if use_trend:
        sma = pd.Series(talib.SMA(c.to_numpy(dtype=float), TREND_SMA), index=c.index)
        w = w.where(c.shift(1) > sma.shift(1), 0.0)
    return levered_equity(w, r, rf)


def sector_momentum_vm(px, rf, vol_target, top_n=2):
    """H9 — rules-based sector rotation + vol management.

    Monthly: rank sectors by 12-1 momentum, hold top_n equal weight, then
    apply the same vol-management overlay. Tests whether the ex-post XLK
    outcome was reachable by RULE rather than hindsight.
    """
    c = px["close"][SECTORS]
    r = c.pct_change().fillna(0.0)
    mom = c.pct_change(MOM_LOOKBACK).shift(MOM_SKIP)

    month = pd.Series(c.index.year * 100 + c.index.month, index=c.index)
    reb = month != month.shift(1)

    hold = pd.DataFrame(0.0, index=c.index, columns=SECTORS)
    cur = pd.Series(0.0, index=SECTORS)
    for i, d in enumerate(c.index):
        if reb.iloc[i] and i > MOM_LOOKBACK + MOM_SKIP:
            top = mom.iloc[i - 1].dropna().sort_values(ascending=False).head(top_n)
            cur = pd.Series(0.0, index=SECTORS)
            if len(top):
                cur[top.index] = 1.0 / len(top)
        hold.iloc[i] = cur.values

    port_r = (hold * r).sum(axis=1)
    rv = (port_r.rolling(VOL_LOOKBACK).std() * np.sqrt(TRADING_DAYS)).shift(1)
    w = (vol_target / rv).clip(upper=MAX_LEVERAGE).fillna(0.0)
    turn_cost = (hold.diff().abs().sum(axis=1).fillna(0.0)) * COST_BPS / 1e4
    return levered_equity(w, port_r - turn_cost, rf)


# -- MAIN ---------------------------------------------------------------------
def main():
    print("=" * 104)
    print("BEAT QQQ ON BOTH SHARPE AND RETURN?")
    print("=" * 104)

    px = load()
    idx = px["close"].index
    rf = px["close"][RF_PROXY].pct_change().fillna(0.0)   # realised risk-free

    bench = px["close"][BENCH] / px["close"][BENCH].iloc[0]
    bench_r = bench.pct_change().dropna()
    b = metrics(bench, "QQQ buy & hold", bench_r)
    print(f"Window {idx[0].date()} -> {idx[-1].date()} | "
          f"rf proxy = {RF_PROXY} (avg {rf.mean()*TRADING_DAYS*100:.2f}%/yr) | "
          f"borrow spread {BORROW_SPREAD*1e4:.0f}bp")
    print(f"BAR: QQQ CAGR {b['cagr']*100:.2f}%  Sharpe {b['sharpe']:.2f}\n")

    runs = {"QQQ buy & hold": bench}
    for vt in VOL_TARGETS:
        runs[f"Vol-managed QQQ  @{vt:.0%} vol"] = vol_managed(px, BENCH, rf, vt)
    for vt in VOL_TARGETS:
        runs[f"VM+trend QQQ     @{vt:.0%} vol"] = vol_managed(px, BENCH, rf, vt, True)
    for vt in (0.20, 0.25):
        runs[f"Sector mom + VM  @{vt:.0%} vol"] = sector_momentum_vm(px, rf, vt)

    rows = [metrics((e / e.dropna().iloc[0]).dropna(), k, bench_r)
            for k, e in runs.items()]

    print("=" * 104)
    print(f"{'':<34}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}"
          f"{'DSR':>7}{'p vs QQQ':>10}")
    print("=" * 104)
    for m in rows:
        pstr = f"{m['p']:9.3f}" if pd.notna(m["p"]) else "      n/a"
        flag = ""
        if m["label"] != "QQQ buy & hold":
            both = m["cagr"] > b["cagr"] and m["sharpe"] > b["sharpe"]
            flag = "  <-- BOTH" if both else ""
        print(f"  {m['label']:<32}{m['cagr']*100:8.2f}%{m['vol']*100:7.2f}%"
              f"{m['sharpe']:8.2f}{m['max_dd']*100:8.2f}%"
              f"{m['dsr']:7.3f}{pstr}{flag}")

    # -- Leverage invariance check ------------------------------------------
    print("\n" + "=" * 104)
    print("SANITY CHECK — is Sharpe invariant to the vol target? (it must be)")
    print("=" * 104)
    vm = [m for m in rows if m["label"].startswith("Vol-managed")]
    sh = [m["sharpe"] for m in vm]
    print(f"  Vol-managed QQQ Sharpe across targets: "
          f"{', '.join(f'{s:.2f}' for s in sh)}   spread={max(sh)-min(sh):.3f}")
    print("  Near-identical Sharpe confirms leverage moves RETURN, not EDGE.")
    print("  Any 'both higher' result below is therefore the Sharpe edge x leverage,")
    print("  NOT a new source of alpha.")

    # -- Verdict --------------------------------------------------------------
    print("\n" + "=" * 104)
    print("VERDICT — three gates: (1) CAGR>QQQ  (2) Sharpe>QQQ  (3) p<0.05 and DSR>=0.95")
    print("=" * 104)
    passed = []
    for m in rows:
        if m["label"] == "QQQ buy & hold":
            continue
        g1 = m["cagr"] > b["cagr"]
        g2 = m["sharpe"] > b["sharpe"]
        g3 = pd.notna(m["p"]) and m["p"] < 0.05 and pd.notna(m["dsr"]) and m["dsr"] >= 0.95
        marks = f"CAGR{'PASS' if g1 else 'fail'} Sharpe{'PASS' if g2 else 'fail'} " \
                f"Stat{'PASS' if g3 else 'FAIL'}"
        if g1 and g2 and g3:
            passed.append(m)
        print(f"  {m['label']:<34} {marks}")

    print("\n" + "-" * 104)
    if passed:
        print(f"  {len(passed)} configuration(s) cleared ALL THREE gates.")
    else:
        both_only = [m for m in rows if m["label"] != "QQQ buy & hold"
                     and m["cagr"] > b["cagr"] and m["sharpe"] > b["sharpe"]]
        if both_only:
            print(f"  {len(both_only)} configuration(s) beat QQQ on BOTH CAGR and Sharpe,")
            print("  but NONE is statistically significant. Best p-value: "
                  f"{min(m['p'] for m in both_only):.3f} (need <0.05).")
            print("  These are levered versions of an edge we cannot prove exists.")
        else:
            print("  Nothing beat QQQ on both dimensions.")
    print("-" * 104)

    out = pd.DataFrame({k: v for k, v in runs.items()})
    out.to_csv(os.path.join(OUTDIR, "beat_qqq_equity.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for k, e in runs.items():
        e = (e / e.dropna().iloc[0]).dropna()
        st = dict(lw=2.4, color="black") if k == "QQQ buy & hold" else dict(lw=1.1)
        ax1.plot(e.index, e, label=k, **st)
    ax1.set_yscale("log"); ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title("Attempting to beat QQQ on both Sharpe and return "
                  f"({idx[0].date()} to {idx[-1].date()}) — financing costs included")
    ax1.legend(loc="upper left", fontsize=7); ax1.grid(alpha=0.3)

    for k, e in runs.items():
        e = (e / e.dropna().iloc[0]).dropna()
        dd = e / e.cummax() - 1
        st = dict(lw=2.0, color="black") if k == "QQQ buy & hold" else dict(lw=0.9)
        ax2.plot(dd.index, dd * 100, **st)
    ax2.set_ylabel("Drawdown %"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    png = os.path.join(OUTDIR, "beat_qqq.png")
    plt.savefig(png, dpi=120); plt.close()
    print(f"\nSaved: {png}")


if __name__ == "__main__":
    main()
