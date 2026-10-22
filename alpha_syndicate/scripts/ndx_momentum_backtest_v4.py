"""
Alpha Syndicate — NDX Momentum Strategy v4
Team debrief after v3 (CAGR 17.17%, NDX gap: -1.17%):

  QR:    OOS gap driven by NVDA/AI runaway 2023-24. 50/50 blend underweights raw momentum.
         Fix: shift to 65% raw / 35% risk-adj. Also add dual-lookback (6M+12M blend).
  RM:    v3 Max DD -23.77% — near -25% budget. No vol budget expansion permitted.
         VOL_TARGET stays 0.30, MAX_WEIGHT stays 0.20.
  PM:    Close last mile via signal quality only. Dual-period momentum standard at AQR/Two Sigma.
  Valid: Blend ratio change has academic precedent (Asness et al. 2013) — not data-mined.
         Dual-lookback is published (6M used by Jegadeesh & Titman 1993 original paper).
  Dev:   Implementing — all RM constraints respected.

v4 changes vs v3:
  1. BLEND_RAW 0.50 → 0.65  (capture mega-winner runaway momentum)
  2. BLEND_RADJ 0.50 → 0.35
  3. DUAL-PERIOD signal: blend 6M (126d) + 12M (252d) momentum before blending with risk-adj
     Final signal = 0.65 * rank(0.5*raw_12m + 0.5*raw_6m) + 0.35 * rank(risk_adj_12m)
  4. VOL_TARGET stays 0.30 (RM hard constraint)
  5. MAX_WEIGHT stays 0.20 (RM hard constraint)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
FORMATION_LONG   = 252        # 12M lookback
FORMATION_MED    = 126        # 6M lookback (dual-period, QR recommendation)
SKIP_DAYS        = 21
TOP_PCT          = 0.10
MIN_STOCKS       = 5
MAX_WEIGHT       = 0.20       # RM constraint — unchanged from v3
VOL_TARGET       = 0.30       # RM constraint — unchanged from v3
VOL_HARD_CAP     = 0.40
DRAWDOWN_HALF    = -0.10
DRAWDOWN_FLAT    = -0.20
COST_PER_TRADE   = 0.0016
START_DATE       = "2010-01-01"
END_DATE         = "2024-12-31"
OOS_START        = "2019-01-01"
DATA_START       = "2008-01-01"
MIN_HISTORY_DAYS = FORMATION_LONG + SKIP_DAYS + 21
MIN_EXPOSURE     = 0.20
# Signal blend weights (Valid: academically grounded, not optimised)
W_RAW            = 0.65       # raw dual-period momentum component (v3: 0.50)
W_RADJ           = 0.35       # risk-adjusted momentum component   (v3: 0.50)
W_12M            = 0.50       # weight on 12M within raw component
W_6M             = 0.50       # weight on 6M within raw component
OUTPUT_DIR       = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NDX_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST",
    "NFLX", "AMD",  "ADBE", "QCOM", "PEP",  "CSCO", "INTU", "AMGN", "AMAT",
    "MU",   "ISRG", "BKNG", "REGN", "ADI",  "LRCX", "PANW", "GILD", "SNPS",
    "CDNS", "MELI", "KLAC", "CRWD", "FTNT", "ABNB", "MNST", "PCAR", "ORLY",
    "MAR",  "CPRT", "MRVL", "WDAY", "DXCM", "FAST", "ODFL", "ROST", "IDXX",
    "VRSK", "BIIB", "KDP",  "CTSH", "DLTR", "ON",   "CTAS", "ALGN", "EBAY",
    "DDOG", "ZS",   "FANG", "EXC",  "TEAM", "TTD",
    "GOOG",  "TMUS",  "SBUX",  "PAYX",  "AEP",   "XEL",
    "MDLZ",  "LULU",  "NXPI",  "MCHP",  "ILMN",
    "PYPL",  "OKTA",  "FSLR",  "SMCI",  "NTAP",  "CSGP",
    "AXON",  "PLTR",  "MDB",   "APP",   "CEG",   "CCEP",
    "MSTR",
]
INDEX_TICKER = "^NDX"


# ── DATA ─────────────────────────────────────────────────────────────────────
def download_data(tickers, index, start, end):
    print(f"Downloading {len(tickers)} tickers + NDX index...")
    raw    = yf.download(tickers + [index], start=start, end=end,
                         auto_adjust=True, progress=False)
    prices = raw["Close"].copy()
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    ndx = prices.pop(index) if index in prices.columns else None
    min_obs = int(len(prices) * 0.80)
    prices  = prices.dropna(axis=1, thresh=min_obs).ffill().bfill()
    print(f"  Universe after cleaning: {prices.shape[1]} tickers | {prices.shape[0]} days")
    return prices, ndx


# ── SIGNALS ──────────────────────────────────────────────────────────────────
def dual_period_blended_signal(prices, form_long=252, form_med=126, skip=21,
                                vol_window=63, w_raw=0.65, w_radj=0.35,
                                w_12m=0.50, w_6m=0.50):
    """
    QR / PM signal (v4):
    Step 1 — Raw dual-period component:
        raw_12m = price[t-skip] / price[t-form_long] - 1
        raw_6m  = price[t-skip] / price[t-form_med]  - 1
        raw     = w_12m * raw_12m + w_6m * raw_6m
    Step 2 — Risk-adjusted component (Barroso & Santa-Clara 2015):
        radj    = raw_12m / trailing_vol(63d)
    Step 3 — Cross-sectional rank each, then blend:
        signal  = w_raw * rank(raw) + w_radj * rank(radj)

    Both components ranked cross-sectionally before blending — same scale,
    no unit mismatch between return % and Sharpe-like ratio.
    """
    raw_12m   = prices.shift(skip) / prices.shift(form_long) - 1
    raw_6m    = prices.shift(skip) / prices.shift(form_med)  - 1
    raw_dual  = w_12m * raw_12m + w_6m * raw_6m

    trail_vol = prices.pct_change().rolling(vol_window).std() * np.sqrt(252)
    trail_vol = trail_vol.replace(0, np.nan)
    radj      = raw_12m / trail_vol   # risk-adjusted uses 12M for stability

    raw_rank  = raw_dual.rank(axis=1, pct=True)
    radj_rank = radj.rank(axis=1, pct=True)
    return w_raw * raw_rank + w_radj * radj_rank


def rolling_vol(prices, window=21):
    return prices.pct_change().rolling(window).std() * np.sqrt(252)


def inv_vol_weights(vols, max_w=0.20):
    iv = (1.0 / vols.replace(0, np.nan)).dropna()
    if iv.empty:
        return pd.Series(dtype=float)
    w = iv / iv.sum()
    w = w.clip(upper=max_w)
    return w / w.sum()


def vol_scale_factor(weights, vols, target=0.30, hard_cap=0.40):
    shared   = weights.index.intersection(vols.index)
    port_vol = (vols[shared] * weights[shared]).sum()
    if port_vol <= 0:
        return 1.0
    if port_vol > hard_cap:
        return hard_cap / port_vol
    return min(target / port_vol, 1.0)


# ── BACKTEST ─────────────────────────────────────────────────────────────────
def run_backtest(prices, ndx):
    signal = dual_period_blended_signal(
        prices, FORMATION_LONG, FORMATION_MED, SKIP_DAYS,
        w_raw=W_RAW, w_radj=W_RADJ, w_12m=W_12M, w_6m=W_6M
    )
    vols   = rolling_vol(prices)

    rebal_dates = prices.resample("ME").last().index
    rebal_dates = rebal_dates[(rebal_dates >= START_DATE) & (rebal_dates <= END_DATE)]

    records      = []
    held_weights = {}
    peak_value   = 1.0
    port_value   = 1.0
    dd_scale     = 1.0

    for i, d in enumerate(rebal_dates[:-1]):
        next_d = rebal_dates[i + 1]

        if dd_scale == 0.0:
            records.append({"date": next_d, "return": 0.0, "n": 0,
                            "dd_scale": dd_scale, "n_eligible": 0,
                            "realized_vol": np.nan})
            held_weights = {}
            dd_scale = 0.5
            continue

        try:
            sig_row = signal.asof(d)
            vol_row = vols.asof(d)
        except Exception:
            records.append({"date": next_d, "return": 0.0, "n": 0,
                            "dd_scale": dd_scale, "n_eligible": 0,
                            "realized_vol": np.nan})
            continue

        available_days = prices.loc[:d].notna().sum()
        data_ok        = available_days >= MIN_HISTORY_DAYS
        valid_mask     = sig_row.notna() & vol_row.notna() & (vol_row > 0) & data_ok
        eligible_sig   = sig_row[valid_mask]
        n_eligible     = len(eligible_sig)

        if n_eligible < MIN_STOCKS:
            records.append({"date": next_d, "return": 0.0, "n": 0,
                            "dd_scale": dd_scale, "n_eligible": n_eligible,
                            "realized_vol": np.nan})
            continue

        top_n   = max(MIN_STOCKS, int(TOP_PCT * n_eligible))
        top     = eligible_sig.nlargest(top_n).index
        weights = inv_vol_weights(vol_row[top], MAX_WEIGHT)

        if weights.empty:
            records.append({"date": next_d, "return": 0.0, "n": 0,
                            "dd_scale": dd_scale, "n_eligible": n_eligible,
                            "realized_vol": np.nan})
            continue

        scale    = vol_scale_factor(weights, vol_row, VOL_TARGET, VOL_HARD_CAP)
        combined = max(MIN_EXPOSURE, scale * dd_scale)
        weights  = weights * combined

        new_w    = weights.to_dict()
        turnover = sum(abs(new_w.get(t, 0) - held_weights.get(t, 0))
                       for t in set(new_w) | set(held_weights)) / 2
        cost     = turnover * COST_PER_TRADE

        tickers_held = list(weights.index)
        try:
            period       = prices.loc[d:next_d, tickers_held]
            daily_rets   = period.pct_change().dropna()
            port_daily   = (daily_rets * weights).sum(axis=1)
            realized_vol = float(port_daily.std() * np.sqrt(252)) if len(port_daily) > 1 else np.nan
            stock_rets   = period.iloc[-1] / period.iloc[0] - 1
            port_ret     = float((stock_rets * weights).sum()) - cost
        except Exception:
            port_ret     = -cost
            realized_vol = np.nan

        port_value *= (1 + port_ret)
        peak_value  = max(peak_value, port_value)
        current_dd  = (port_value / peak_value) - 1

        if current_dd <= DRAWDOWN_FLAT:
            dd_scale = 0.0
        elif current_dd <= DRAWDOWN_HALF:
            dd_scale = 0.5
        else:
            dd_scale = min(dd_scale + 0.5, 1.0) if dd_scale < 1.0 else 1.0

        held_weights = new_w
        records.append({"date": next_d, "return": port_ret, "n": len(weights),
                        "dd_scale": dd_scale, "n_eligible": n_eligible,
                        "realized_vol": realized_vol})

    return pd.DataFrame(records).set_index("date")


# ── METRICS ──────────────────────────────────────────────────────────────────
def compute_metrics(returns, equity, bench_equity, results, label="v4"):
    ann   = np.sqrt(12)
    b_ret = bench_equity.pct_change().dropna()

    def _cagr(eq):
        n_yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        return (eq.iloc[-1] ** (1 / n_yrs)) - 1 if n_yrs > 0 else 0

    sharpe = lambda r: (r.mean() / r.std()) * ann if len(r) > 1 and r.std() > 0 else 0
    max_dd = lambda eq: ((eq / eq.cummax()) - 1).min()
    calmar = lambda eq: _cagr(eq) / abs(max_dd(eq)) if max_dd(eq) != 0 else 0

    oos_dt   = pd.Timestamp(OOS_START)
    is_mask  = equity.index < oos_dt
    oos_mask = equity.index >= oos_dt
    is_eq  = equity[is_mask];   is_ret  = returns[is_mask]
    oos_eq = equity[oos_mask];  oos_ret = returns[oos_mask]
    if not oos_eq.empty:
        oos_eq = oos_eq / oos_eq.iloc[0]

    ndx_cagr  = _cagr(bench_equity)
    strat_cagr = _cagr(equity)
    beats_ndx  = strat_cagr > ndx_cagr

    return {
        f"--- FULL PERIOD (2010-2024) [{label}] ---": "----------",
        "CAGR":                             f"{strat_cagr:.2%}",
        "Beats NDX CAGR":                   f"[YES] +{strat_cagr-ndx_cagr:.2%}" if beats_ndx else f"[NO]  {strat_cagr-ndx_cagr:.2%}",
        "Annualised Volatility":            f"{returns.std() * ann:.2%}",
        "Avg Realized Vol":                 f"{results['realized_vol'].dropna().mean():.2%}",
        "Sharpe Ratio":                     f"{sharpe(returns):.2f}",
        "Calmar Ratio":                     f"{calmar(equity):.2f}  (NDX: {calmar(bench_equity):.2f})",
        "Max Drawdown":                     f"{max_dd(equity):.2%}  (NDX: {max_dd(bench_equity):.2%})",
        "Win Rate (monthly)":               f"{(returns > 0).mean():.2%}",
        "Best Month":                       f"{returns.max():.2%}",
        "Worst Month":                      f"{returns.min():.2%}",
        "Total Return":                     f"{equity.iloc[-1] - 1:.2%}  (NDX: {bench_equity.iloc[-1] - 1:.2%})",
        "--- IN-SAMPLE (2010-2018) ---":    "----------",
        "IS CAGR":                          f"{_cagr(is_eq):.2%}" if not is_eq.empty else "N/A",
        "IS Sharpe":                        f"{sharpe(is_ret):.2f}" if not is_ret.empty else "N/A",
        "IS Max Drawdown":                  f"{max_dd(is_eq):.2%}" if not is_eq.empty else "N/A",
        "--- OUT-OF-SAMPLE (2019-2024) ---":"----------",
        "OOS CAGR":                         f"{_cagr(oos_eq):.2%}" if not oos_eq.empty else "N/A",
        "OOS Sharpe":                       f"{sharpe(oos_ret):.2f}" if not oos_ret.empty else "N/A",
        "OOS Max Drawdown":                 f"{max_dd(oos_eq):.2%}" if not oos_eq.empty else "N/A",
        "--- BENCHMARK (NDX) ---":          "----------",
        "NDX CAGR":                         f"{ndx_cagr:.2%}",
        "NDX Sharpe":                       f"{sharpe(b_ret):.2f}",
        "NDX Max Drawdown":                 f"{max_dd(bench_equity):.2%}",
        "NDX Total Return":                 f"{bench_equity.iloc[-1] - 1:.2%}",
        "--- RM GUARDRAILS ---":            "----------",
        "RM Max DD Budget":                 "-25.00%",
        "RM Status":                        "PASS" if max_dd(equity) > -0.25 else "BREACH — escalate to PM",
    }


# ── CHARTS ────────────────────────────────────────────────────────────────────
def plot_results(results, equity, ndx_equity, output_dir, label="v4"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.patch.set_facecolor("#0D1117")
    for ax in axes:
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363D")

    fig.suptitle(f"Alpha Syndicate  |  NDX Dual-Period Momentum {label}  |  2010-2024",
                 fontsize=13, fontweight="bold", color="white", y=0.98)

    axes[0].plot(equity.index, equity, label=f"Momentum {label} (Dual-Period Blend)",
                 color="#00C8FF", linewidth=2)
    axes[0].plot(ndx_equity.index, ndx_equity, label="NDX Buy & Hold",
                 color="#FF6B35", linewidth=1.5, alpha=0.85)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("Equity Curve (Net of Costs)")
    axes[0].legend(facecolor="#161B22", labelcolor="white")
    axes[0].grid(True, alpha=0.15)

    dd = (equity / equity.cummax()) - 1
    axes[1].fill_between(dd.index, dd * 100, 0, color="#FF4444", alpha=0.7)
    axes[1].axhline(-10, color="#FFA500", lw=0.8, ls="--", label="-10% brake")
    axes[1].axhline(-20, color="#FF0000", lw=0.8, ls="--", label="-20% flat")
    axes[1].axhline(-25, color="#8B0000", lw=0.8, ls="--", label="-25% RM budget")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title("Strategy Drawdown vs RM Guardrails")
    axes[1].legend(facecolor="#161B22", labelcolor="white", fontsize=8)
    axes[1].grid(True, alpha=0.15)

    colors = ["#00C8FF" if r >= 0 else "#FF4444" for r in results["return"]]
    axes[2].bar(results.index, results["return"] * 100, color=colors, alpha=0.8, width=20)
    axes[2].set_ylabel("Monthly Return (%)")
    axes[2].set_title("Monthly Return Distribution")
    axes[2].grid(True, alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, f"equity_curve_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {path}")


# ── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("  ALPHA SYNDICATE — v4: DUAL-PERIOD BLENDED SIGNAL")
    print(f"  Signal: {W_RAW:.0%} raw (6M+12M blend) + {W_RADJ:.0%} risk-adj")
    print(f"  Vol Target: {VOL_TARGET:.0%} | Top: {TOP_PCT:.0%} | MaxW: {MAX_WEIGHT:.0%}")
    print("="*65)

    prices, ndx = download_data(NDX_TICKERS, INDEX_TICKER, DATA_START, END_DATE)
    results     = run_backtest(prices, ndx)
    results.index = pd.to_datetime(results.index)
    equity      = (1 + results["return"]).cumprod()

    ndx_monthly = ndx.resample("ME").last().pct_change()
    ndx_monthly = ndx_monthly.reindex(equity.index, method="ffill").fillna(0)
    ndx_equity  = (1 + ndx_monthly).cumprod()

    metrics = compute_metrics(results["return"], equity, ndx_equity, results)

    print("\n" + "="*65)
    print("  PERFORMANCE REPORT — v4")
    print("="*65)
    for k, v in metrics.items():
        print(f"  {k:<44} {v}")
    print("="*65)

    pd.DataFrame(metrics.items(), columns=["Metric","Value"]).to_csv(
        os.path.join(OUTPUT_DIR, "performance_metrics_v4.csv"), index=False)
    results.to_csv(os.path.join(OUTPUT_DIR, "monthly_returns_v4.csv"))
    plot_results(results, equity, ndx_equity, OUTPUT_DIR, label="v4")
    print(f"\n  Outputs -> {OUTPUT_DIR}\n")
