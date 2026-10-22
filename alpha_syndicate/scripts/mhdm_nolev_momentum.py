"""
Alpha Syndicate — MOMENTUM ONLY, NO LEVERAGE (2016-2026)

HARD CONSTRAINTS SET BY SYSTEM OWNER:
  1. NO LEVERAGE. Gross exposure <= 1.0 at every point in time. Asserted in
     code, not merely intended -- run_backtest() raises if ever exceeded.
  2. MOMENTUM SIGNAL ONLY. No volatility-managed overlay, no beta timing.

WHAT CHANGED vs mhdm_etf_backtest.py, AND WHY (stated openly):
  The original run targeted 10% portfolio volatility and consequently sat at
  67.5% average gross exposure. That produced CAGR 6.95% / Sharpe 0.67.
  Diagnosis: the RISK CONSTRAINT, not the momentum signal, drove the shortfall
  -- it was structurally ~1/3 in cash through a decade-long bull market.

  "No leverage" implies "may be fully invested", so the vol-target downscale
  is removed and the book runs at gross = 1.0 when signals are eligible.

  Valid's note: this IS a parameter change made after seeing results, and it
  must be labelled as such. The defence is that it is a MANDATE change
  (risk appetite), not a search over the signal for something that scores
  well. The signal itself -- horizons, skip, gates -- is untouched.

PRE-REGISTERED VARIANTS (canonical parameterisations, no sweeping):
  M1  MHDM top-4, fully invested      -- our signal, vol-target removed
  M2  Dual momentum top-1             -- Antonacci, "Dual Momentum Investing"
  M3  TSMOM equal-weight              -- Moskowitz, Ooi & Pedersen (2012)

DSR n_trials = 28 (whole programme to date). Reminder: this number only ever
goes up, and every row's DSR falls when it does.

BAR: QQQ CAGR 20.20%, Sharpe 0.94, MaxDD -35.00%.
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
TRADING_DAYS = 252
N_TRIALS_DSR = 28
COST_BPS     = 3.0

HORIZONS     = [21, 63, 126, 252]   # UNCHANGED from the original signal
SKIP_DAYS    = 21                   # UNCHANGED
TREND_SMA    = 200                  # UNCHANGED
ABS_MOM      = 252                  # UNCHANGED
ATR_DAYS     = 20                   # UNCHANGED
MAX_WEIGHT   = 0.35                 # UNCHANGED (RM per-position cap)
MAX_GROSS    = 1.00                 # HARD: no leverage

BENCH        = "QQQ"
CASH         = "BIL"
UNIVERSE     = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
                "SPY", "QQQ", "IWM", "TLT", "GLD", CASH]

DATA   = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data/mhdm_etf_bars.parquet"
OUTDIR = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
os.makedirs(OUTDIR, exist_ok=True)


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


def deflated_sharpe(r, n):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = float(pd.Series(r).skew()); g4 = float(pd.Series(r).kurtosis()) + 3.0
    gm, e = 0.5772156649, math.e
    sd = math.sqrt(max((1 - g3*sr + (g4-1)/4*sr**2) / (T-1), 1e-18))
    sr0 = sd * ((1-gm)*norm.ppf(1 - 1.0/n) + gm*norm.ppf(1 - 1.0/(n*e)))
    den = math.sqrt(max(1 - g3*sr + (g4-1)/4*sr**2, 1e-18))
    return float(norm.cdf((sr - sr0) * math.sqrt(T-1) / den))


def metrics(eq, label, bench_r, extra=None):
    r = eq.pct_change().dropna()
    yrs = len(eq) / TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    z, p = sharpe_diff_test(r, bench_r)
    m = {"label": label, "cagr": eq.iloc[-1]**(1/yrs) - 1, "vol": vol,
         "sharpe": (r.mean()*TRADING_DAYS)/vol if vol > 0 else np.nan,
         "max_dd": (eq/eq.cummax() - 1).min(),
         "dsr": deflated_sharpe(r, N_TRIALS_DSR), "p": p}
    if extra:
        m.update(extra)
    return m


# -- SIGNAL (unchanged from the approved design) ------------------------------
def build_signals(px):
    close, high, low = px["close"], px["high"], px["low"]
    syms = close.columns
    roc = {h: pd.DataFrame(index=close.index, columns=syms, dtype=float)
           for h in HORIZONS}
    sma  = pd.DataFrame(index=close.index, columns=syms, dtype=float)
    absm = pd.DataFrame(index=close.index, columns=syms, dtype=float)
    atr  = pd.DataFrame(index=close.index, columns=syms, dtype=float)

    for s in syms:
        c = close[s].to_numpy(float)
        h_ = high[s].to_numpy(float); l_ = low[s].to_numpy(float)
        for h in HORIZONS:
            roc[h][s] = talib.ROC(c, timeperiod=h)
        sma[s]  = talib.SMA(c, timeperiod=TREND_SMA)
        absm[s] = talib.ROC(c, timeperiod=ABS_MOM)
        atr[s]  = talib.ATR(h_, l_, c, timeperiod=ATR_DAYS)

    z_sum = pd.DataFrame(0.0, index=close.index, columns=syms)
    for h in HORIZONS:
        r = roc[h].shift(SKIP_DAYS)
        z = r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1).replace(0, np.nan), axis=0)
        z_sum += z.fillna(0.0)
    score = z_sum / len(HORIZONS)
    score[roc[HORIZONS[-1]].shift(SKIP_DAYS).isna()] = np.nan

    eligible = (absm.shift(SKIP_DAYS) > 0) & (close > sma)
    atr_pct = (atr / close).replace(0, np.nan)
    return score, eligible, atr_pct


def weights_fully_invested(score_row, elig_row, atrp_row, top_n, mode):
    """Fully invested in eligible momentum names; cash otherwise.

    NO vol targeting, NO leverage. Gross is exactly 1.0 by construction.
    """
    w = pd.Series(0.0, index=score_row.index)

    if mode == "tsmom":
        # Own-trend: every eligible name gets an equal share of the book.
        picks = elig_row[elig_row.fillna(False)].index.drop(CASH, errors="ignore")
        if len(picks) == 0:
            w[CASH] = 1.0
            return w
        w[picks] = min(1.0 / len(picks), MAX_WEIGHT)
        w[CASH] = max(0.0, 1.0 - w.sum())
        return w

    cands = score_row.dropna()
    cands = cands[elig_row.reindex(cands.index).fillna(False)]
    cands = cands.drop(labels=[CASH], errors="ignore")
    picks = cands.sort_values(ascending=False).head(top_n)
    if len(picks) == 0:
        w[CASH] = 1.0
        return w

    inv = (1.0 / atrp_row.reindex(picks.index).replace(0, np.nan)).dropna()
    if inv.empty:
        w[CASH] = 1.0
        return w

    raw = (inv / inv.sum()).clip(upper=MAX_WEIGHT)
    raw = raw / raw.sum()                      # renormalise to gross = 1.0
    w[raw.index] = raw.values
    # Unfilled slots (fewer eligible names than top_n) -> cash, not leverage.
    if len(picks) < top_n:
        scale = len(picks) / top_n
        w[raw.index] = raw.values * scale
        w[CASH] = 1.0 - w.sum()
    return w


def run_backtest(px, score, eligible, atrp, top_n, mode):
    close, open_ = px["close"], px["open"]
    dates = close.index
    dret = close.pct_change().fillna(0.0)

    mid = pd.Series(dates.year*100 + dates.month, index=dates)
    first = mid != mid.shift(1)
    warm = max(HORIZONS) + SKIP_DAYS + 5

    eq = 1.0
    w = pd.Series(0.0, index=close.columns); w[CASH] = 1.0
    rec, turns, gross_track = [], [], []

    for i, d in enumerate(dates):
        if i == 0:
            rec.append((d, eq)); continue
        eq *= (1.0 + float((w * dret.loc[d]).sum()))

        if first.iloc[i] and i >= warm:
            prev = dates[i-1]
            tgt = weights_fully_invested(score.loc[prev], eligible.loc[prev],
                                         atrp.loc[prev], top_n, mode)
            g = float(tgt.sum())
            if g > MAX_GROSS + 1e-9:
                raise AssertionError(f"LEVERAGE BREACH on {d.date()}: gross={g:.4f}")
            gross_track.append(g)

            turn = float((tgt - w).abs().sum())
            gap = (open_.loc[d] / close.loc[prev] - 1.0).fillna(0.0)
            eq *= (1.0 + float(((tgt - w) * gap).sum()))
            eq *= (1.0 - turn * COST_BPS / 1e4)
            w = tgt
            turns.append(turn)
        rec.append((d, eq))

    curve = pd.DataFrame(rec, columns=["date", "equity"]).set_index("date")["equity"]
    invested = 1.0 - float(np.mean([g for g in gross_track]) * 0 + 0)  # placeholder
    stats = {"turnover": float(np.mean(turns)) if turns else np.nan,
             "max_gross": float(np.max(gross_track)) if gross_track else np.nan,
             "avg_gross": float(np.mean(gross_track)) if gross_track else np.nan,
             "rebalances": len(turns)}
    return curve, stats


# -- MAIN ---------------------------------------------------------------------
def main():
    print("=" * 102)
    print("MOMENTUM ONLY — NO LEVERAGE")
    print("=" * 102)

    px = load()
    idx = px["close"].index
    bench = px["close"][BENCH] / px["close"][BENCH].iloc[0]
    bench_r = bench.pct_change().dropna()
    b = metrics(bench, "QQQ buy & hold", bench_r)
    print(f"Window {idx[0].date()} -> {idx[-1].date()}")
    print(f"BAR: QQQ CAGR {b['cagr']*100:.2f}%  Sharpe {b['sharpe']:.2f}  "
          f"MaxDD {b['max_dd']*100:.2f}%")
    print(f"Constraint: gross <= {MAX_GROSS:.2f} (asserted in code)\n")

    score, eligible, atrp = build_signals(px)

    variants = [
        ("M1  MHDM top-4, fully invested", 4, "xs"),
        ("M2  Dual momentum top-1 (Antonacci)", 1, "xs"),
        ("M3  TSMOM equal-weight (MOP2012)", 0, "tsmom"),
    ]

    rows, curves = [], {"QQQ buy & hold": bench}
    for label, tn, mode in variants:
        c, st = run_backtest(px, score, eligible, atrp, tn, mode)
        c = c / c.iloc[0]
        curves[label] = c
        rows.append(metrics(c, label, bench_r, st))
        print(f"  {label:<38} max gross={st['max_gross']:.3f}  "
              f"avg gross={st['avg_gross']:.1%}  turnover={st['turnover']:.1%}")

    print("\n" + "=" * 102)
    print(f"{'':<38}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}{'DSR':>7}{'p':>8}")
    print("=" * 102)
    print(f"  {b['label']:<36}{b['cagr']*100:8.2f}%{b['vol']*100:7.2f}%"
          f"{b['sharpe']:8.2f}{b['max_dd']*100:8.2f}%{b['dsr']:7.3f}{'n/a':>8}")
    for m in rows:
        print(f"  {m['label']:<36}{m['cagr']*100:8.2f}%{m['vol']*100:7.2f}%"
              f"{m['sharpe']:8.2f}{m['max_dd']*100:8.2f}%{m['dsr']:7.3f}"
              f"{m['p']:8.3f}")

    print("\n" + "=" * 102)
    print("LEVERAGE AUDIT")
    print("=" * 102)
    mx = max(m["max_gross"] for m in rows)
    print(f"  Highest gross exposure observed across all variants: {mx:.4f}")
    print(f"  Constraint satisfied: {mx <= MAX_GROSS + 1e-9}  "
          f"(backtest raises AssertionError on any breach)")

    print("\n" + "=" * 102)
    print("VERDICT vs QQQ")
    print("=" * 102)
    for m in rows:
        g1 = m["cagr"] > b["cagr"]; g2 = m["sharpe"] > b["sharpe"]
        g3 = pd.notna(m["p"]) and m["p"] < 0.05 and m["dsr"] >= 0.95
        print(f"  {m['label']:<38} CAGR{'PASS' if g1 else 'fail'}  "
              f"Sharpe{'PASS' if g2 else 'fail'}  Stat{'PASS' if g3 else 'FAIL'}")

    best = max(rows, key=lambda m: m["sharpe"])
    print(f"\n  Best momentum variant: {best['label']}")
    print(f"    Sharpe {best['sharpe']:.2f} vs QQQ {b['sharpe']:.2f}  "
          f"| CAGR {best['cagr']*100:.2f}% vs {b['cagr']*100:.2f}%")

    out = pd.DataFrame(curves); out.to_csv(os.path.join(OUTDIR, "nolev_momentum_equity.csv"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
    for k, e in curves.items():
        st = dict(lw=2.4, color="black") if k.startswith("QQQ") else dict(lw=1.3)
        a1.plot(e.index, e, label=k, **st)
    a1.set_yscale("log"); a1.set_ylabel("Growth of $1 (log)")
    a1.set_title(f"Momentum only, no leverage ({idx[0].date()} to {idx[-1].date()})")
    a1.legend(loc="upper left", fontsize=8); a1.grid(alpha=0.3)
    for k, e in curves.items():
        dd = e / e.cummax() - 1
        st = dict(lw=2.0, color="black") if k.startswith("QQQ") else dict(lw=1.0)
        a2.plot(dd.index, dd*100, **st)
    a2.set_ylabel("Drawdown %"); a2.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUTDIR, "nolev_momentum.png")
    plt.savefig(png, dpi=120); plt.close()
    print(f"\nSaved: {png}")


if __name__ == "__main__":
    main()
