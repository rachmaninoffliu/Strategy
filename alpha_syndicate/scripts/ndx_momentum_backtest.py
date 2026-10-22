"""
Alpha Syndicate — NDX 12-1 Cross-Sectional Momentum Backtest
Dr. Alpha blueprint + Executioner risk overlay
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
FORMATION_DAYS  = 252       # 12 months lookback
SKIP_DAYS       = 21        # 1 month skip (avoid ST reversal)
TOP_PCT         = 0.10      # true top decile (dynamic, not hardcoded N)
MIN_STOCKS      = 5         # minimum stocks to form a portfolio
MAX_WEIGHT      = 0.20      # single stock cap
VOL_TARGET      = 0.15      # annualised portfolio vol target
VOL_HARD_CAP    = 0.20      # hard cap before scaling
REGIME_MA       = 200       # SMA for NDX regime filter
DRAWDOWN_HALF   = -0.10     # reduce to 50% at -10% DD
DRAWDOWN_FLAT   = -0.15     # go to cash at -15% DD
COST_PER_TRADE  = 0.0016    # 16 bps round-trip
START_DATE      = "2010-01-01"
END_DATE        = "2024-12-31"
OOS_START       = "2019-01-01"  # out-of-sample split date
DATA_START      = "2008-01-01"  # extra history for formation period
MIN_HISTORY_DAYS = FORMATION_DAYS + SKIP_DAYS + 21  # FIX 1: strict data-availability gate
VOL_REGIME_WINDOW = 10      # 10-day NDX realized vol for pre-scaler (Executioner/Dr.Alpha)
MIN_EXPOSURE     = 0.20     # floor on combined exposure when regime is ON (Skeptic)
OUTPUT_DIR      = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# NDX representative universe (survivorship note flagged for Skeptic in Phase 5)
NDX_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "COST",
    "NFLX", "AMD",  "ADBE", "QCOM", "PEP",  "CSCO", "INTU", "AMGN", "AMAT",
    "MU",   "ISRG", "BKNG", "REGN", "ADI",  "LRCX", "PANW", "GILD", "SNPS",
    "CDNS", "MELI", "KLAC", "CRWD", "FTNT", "ABNB", "MNST", "PCAR", "ORLY",
    "MAR",  "CPRT", "MRVL", "WDAY", "DXCM", "FAST", "ODFL", "ROST", "IDXX",
    "VRSK", "BIIB", "KDP",  "CTSH", "DLTR", "ON",   "CTAS", "ALGN", "EBAY",
    "DDOG", "ZS",   "FANG", "EXC",  "TEAM", "TTD",
]
INDEX_TICKER = "^NDX"


# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────
def download_data(tickers, index, start, end):
    print(f"Downloading {len(tickers)} tickers + NDX index...")
    raw = yf.download(tickers + [index], start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"].copy()

    # Handle both single and MultiIndex column cases
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    # Separate index
    ndx = prices.pop(index) if index in prices.columns else None
    if ndx is None:
        print("  WARNING: NDX index unavailable — regime filter disabled")

    # Drop columns with >20% missing
    min_obs = int(len(prices) * 0.80)
    prices  = prices.dropna(axis=1, thresh=min_obs)
    prices  = prices.ffill().bfill()

    print(f"  Universe after cleaning: {prices.shape[1]} tickers | {prices.shape[0]} days")
    return prices, ndx


# ── SIGNAL FUNCTIONS ──────────────────────────────────────────────────────────
def momentum_signal(prices, formation=252, skip=21):
    """12-1 month momentum: t-252 to t-21. No look-ahead — all shifts are backward."""
    return prices.shift(skip) / prices.shift(formation) - 1


def rolling_vol(prices, window=21):
    """21-day rolling annualised volatility."""
    return prices.pct_change().rolling(window).std() * np.sqrt(252)


def inv_vol_weights(vols, max_w=0.20):
    """Inverse-vol weights, capped at max_w and renormalised."""
    iv = (1.0 / vols.replace(0, np.nan)).dropna()
    if iv.empty:
        return pd.Series(dtype=float)
    w = iv / iv.sum()
    w = w.clip(upper=max_w)
    return w / w.sum()


def vol_scale_factor(weights, vols, target=0.15, hard_cap=0.20):
    """Scale weights so estimated portfolio vol ≈ target."""
    shared = weights.index.intersection(vols.index)
    port_vol = (vols[shared] * weights[shared]).sum()
    if port_vol <= 0:
        return 1.0
    if port_vol > hard_cap:
        return hard_cap / port_vol
    return min(target / port_vol, 1.0)


# ── MAIN BACKTEST LOOP ────────────────────────────────────────────────────────
def run_backtest(prices, ndx):
    mom         = momentum_signal(prices, FORMATION_DAYS, SKIP_DAYS)
    vols        = rolling_vol(prices)
    ndx_sma     = ndx.rolling(REGIME_MA).mean() if ndx is not None else None
    # Dr. Alpha vol pre-scaler: 10-day realized annualised vol of NDX
    ndx_vol10   = ndx.pct_change().rolling(VOL_REGIME_WINDOW).std() * np.sqrt(252) if ndx is not None else None

    # Month-end rebalance dates within window
    rebal_dates = prices.resample("ME").last().index
    rebal_dates = rebal_dates[(rebal_dates >= START_DATE) & (rebal_dates <= END_DATE)]

    records      = []
    held_weights = {}
    peak_value   = 1.0
    port_value   = 1.0
    dd_scale     = 1.0

    for i, d in enumerate(rebal_dates[:-1]):
        next_d = rebal_dates[i + 1]

        # ── Regime filter ──────────────────────────────────────────────────
        if ndx_sma is not None:
            try:
                regime_ok = bool(ndx.asof(d) > ndx_sma.asof(d))
            except Exception:
                regime_ok = True
        else:
            regime_ok = True

        # ── Cash conditions ────────────────────────────────────────────────
        if not regime_ok or dd_scale == 0.0:
            records.append({"date": next_d, "return": 0.0, "n": 0,
                             "regime": regime_ok, "dd_scale": dd_scale})
            held_weights = {}
            if regime_ok and dd_scale == 0.0:
                dd_scale = 0.5   # begin recovery ramp
            continue

        # ── Eligible stocks ────────────────────────────────────────────────
        try:
            mom_row = mom.asof(d)
            vol_row = vols.asof(d)
        except Exception:
            records.append({"date": next_d, "return": 0.0, "n": 0,
                             "regime": regime_ok, "dd_scale": dd_scale,
                             "n_eligible": 0, "realized_vol": np.nan})
            continue

        # FIX 1: strict data-availability gate — stock must have MIN_HISTORY_DAYS
        # of price history as of rebalance date d (prevents survivorship look-ahead)
        available_days = prices.loc[:d].notna().sum()
        data_ok = available_days >= MIN_HISTORY_DAYS

        valid        = mom_row.notna() & vol_row.notna() & (vol_row > 0) & data_ok
        eligible_mom = mom_row[valid]
        n_eligible   = len(eligible_mom)

        if n_eligible < MIN_STOCKS:
            records.append({"date": next_d, "return": 0.0, "n": 0,
                             "regime": regime_ok, "dd_scale": dd_scale,
                             "n_eligible": n_eligible, "realized_vol": np.nan})
            continue

        # FIX 2: true decile — dynamic top 10% of eligible universe
        top_n   = max(MIN_STOCKS, int(TOP_PCT * n_eligible))
        top     = eligible_mom.nlargest(top_n).index
        weights = inv_vol_weights(vol_row[top], MAX_WEIGHT)

        if weights.empty:
            records.append({"date": next_d, "return": 0.0, "n": 0,
                             "regime": regime_ok, "dd_scale": dd_scale,
                             "n_eligible": n_eligible, "realized_vol": np.nan})
            continue

        # ── Vol scaling + drawdown brake ───────────────────────────────────
        scale = vol_scale_factor(weights, vol_row, VOL_TARGET, VOL_HARD_CAP)
        # Skeptic floor: when regime is ON, never drop below MIN_EXPOSURE
        combined = max(MIN_EXPOSURE, scale * dd_scale)
        weights  = weights * combined

        # ── Transaction costs ──────────────────────────────────────────────
        new_w    = weights.to_dict()
        turnover = sum(abs(new_w.get(t, 0) - held_weights.get(t, 0))
                       for t in set(new_w) | set(held_weights)) / 2
        cost     = turnover * COST_PER_TRADE

        # ── Holding period return + FIX 4: realized vol ───────────────────
        tickers_held = list(weights.index)
        try:
            period      = prices.loc[d:next_d, tickers_held]
            daily_rets  = period.pct_change().dropna()
            port_daily  = (daily_rets * weights).sum(axis=1)
            realized_vol = float(port_daily.std() * np.sqrt(252)) if len(port_daily) > 1 else np.nan
            stock_rets  = period.iloc[-1] / period.iloc[0] - 1
            port_ret    = float((stock_rets * weights).sum()) - cost
        except Exception:
            port_ret     = -cost
            realized_vol = np.nan

        # ── Update portfolio value + drawdown state ────────────────────────
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
                         "regime": regime_ok, "dd_scale": dd_scale,
                         "n_eligible": n_eligible, "realized_vol": realized_vol})

    return pd.DataFrame(records).set_index("date")


# ── PERFORMANCE METRICS ───────────────────────────────────────────────────────
def compute_metrics(returns, equity, bench_equity, results, oos_start=OOS_START):
    ann     = np.sqrt(12)
    b_ret   = bench_equity.pct_change().dropna()

    def _cagr(eq):
        n_yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        return (eq.iloc[-1] ** (1 / n_yrs)) - 1 if n_yrs > 0 else 0

    sharpe  = lambda r:  (r.mean() / r.std()) * ann if len(r) > 1 and r.std() > 0 else 0
    max_dd  = lambda eq: ((eq / eq.cummax()) - 1).min()
    calmar  = lambda eq: _cagr(eq) / abs(max_dd(eq)) if max_dd(eq) != 0 else 0

    # FIX 3: in-sample vs out-of-sample split
    oos_dt   = pd.Timestamp(oos_start)
    is_mask  = equity.index < oos_dt
    oos_mask = equity.index >= oos_dt

    is_eq    = equity[is_mask];   is_ret  = returns[is_mask]
    oos_eq   = equity[oos_mask];  oos_ret = returns[oos_mask]

    # Normalize OOS equity to start at 1
    if not oos_eq.empty:
        oos_eq = oos_eq / oos_eq.iloc[0]

    # FIX 4: avg realized vol vs estimated vol
    avg_realized_vol = results["realized_vol"].dropna().mean()

    return {
        "--- FULL PERIOD (2010-2024) ----": "----------",
        "CAGR":                              f"{_cagr(equity):.2%}",
        "Annualised Volatility":             f"{returns.std() * ann:.2%}",
        "Avg Realized Vol (actual)":         f"{avg_realized_vol:.2%}",
        "Sharpe Ratio":                      f"{sharpe(returns):.2f}",
        "Calmar Ratio":                      f"{calmar(equity):.2f}",
        "Max Drawdown":                      f"{max_dd(equity):.2%}",
        "Win Rate (monthly)":                f"{(returns > 0).mean():.2%}",
        "Best Month":                        f"{returns.max():.2%}",
        "Worst Month":                       f"{returns.min():.2%}",
        "Total Return":                      f"{equity.iloc[-1] - 1:.2%}",
        "--- IN-SAMPLE (2010-2018) ------": "----------",
        "IS CAGR":                           f"{_cagr(is_eq):.2%}" if not is_eq.empty else "N/A",
        "IS Sharpe":                         f"{sharpe(is_ret):.2f}" if not is_ret.empty else "N/A",
        "IS Max Drawdown":                   f"{max_dd(is_eq):.2%}" if not is_eq.empty else "N/A",
        "--- OUT-OF-SAMPLE (2019-2024) --": "----------",
        "OOS CAGR":                          f"{_cagr(oos_eq):.2%}" if not oos_eq.empty else "N/A",
        "OOS Sharpe":                        f"{sharpe(oos_ret):.2f}" if not oos_ret.empty else "N/A",
        "OOS Max Drawdown":                  f"{max_dd(oos_eq):.2%}" if not oos_eq.empty else "N/A",
        "--- BENCHMARK (NDX) ------------": "----------",
        "NDX CAGR":                          f"{_cagr(bench_equity):.2%}",
        "NDX Ann. Vol":                      f"{b_ret.std() * ann:.2%}",
        "NDX Sharpe":                        f"{sharpe(b_ret):.2f}",
        "NDX Max Drawdown":                  f"{max_dd(bench_equity):.2%}",
        "NDX Total Return":                  f"{bench_equity.iloc[-1] - 1:.2%}",
    }


# ── CHARTING ──────────────────────────────────────────────────────────────────
def plot_results(results, equity, ndx_equity, output_dir):
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

    fig.suptitle("Alpha Syndicate  |  NDX 12-1 Momentum  |  2010–2024",
                 fontsize=13, fontweight="bold", color="white", y=0.98)

    # Equity curve
    axes[0].plot(equity.index, equity, label="Momentum Strategy",
                 color="#00C8FF", linewidth=2)
    axes[0].plot(ndx_equity.index, ndx_equity, label="NDX Benchmark",
                 color="#FF6B35", linewidth=1.5, alpha=0.85)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title("Equity Curve (Net of Costs)")
    axes[0].legend(facecolor="#161B22", labelcolor="white")
    axes[0].grid(True, alpha=0.15)

    # Drawdown
    dd = (equity / equity.cummax()) - 1
    axes[1].fill_between(dd.index, dd * 100, 0, color="#FF4444", alpha=0.7)
    axes[1].axhline(-10, color="#FFA500", lw=0.8, ls="--", label="-10% brake")
    axes[1].axhline(-15, color="#FF0000", lw=0.8, ls="--", label="-15% flat")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title("Strategy Drawdown")
    axes[1].legend(facecolor="#161B22", labelcolor="white", fontsize=8)
    axes[1].grid(True, alpha=0.15)

    # Monthly returns
    colors = ["#00C8FF" if r >= 0 else "#FF4444" for r in results["return"]]
    axes[2].bar(results.index, results["return"] * 100, color=colors, alpha=0.8, width=20)
    axes[2].set_ylabel("Monthly Return (%)")
    axes[2].set_title("Monthly Return Distribution")
    axes[2].grid(True, alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {path}")
    return path


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ALPHA SYNDICATE — NDX 12-1 MOMENTUM BACKTEST ENGINE")
    print("="*60)

    prices, ndx = download_data(NDX_TICKERS, INDEX_TICKER, DATA_START, END_DATE)

    results = run_backtest(prices, ndx)
    results.index = pd.to_datetime(results.index)

    equity = (1 + results["return"]).cumprod()

    # Benchmark curve aligned to strategy dates
    ndx_monthly  = ndx.resample("ME").last().pct_change()
    ndx_monthly  = ndx_monthly.reindex(equity.index, method="ffill").fillna(0)
    ndx_equity   = (1 + ndx_monthly).cumprod()

    metrics = compute_metrics(results["return"], equity, ndx_equity, results)

    print("\n" + "="*60)
    print("  PERFORMANCE REPORT")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k:<34} {v}")
    print("="*60)

    # Save outputs
    pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_csv(
        os.path.join(OUTPUT_DIR, "performance_metrics.csv"), index=False)
    results.to_csv(os.path.join(OUTPUT_DIR, "monthly_returns.csv"))

    plot_results(results, equity, ndx_equity, OUTPUT_DIR)

    print("\n  Outputs saved to:", OUTPUT_DIR)
    print("  Dev-01 complete. Handing to The Skeptic for validation.\n")
