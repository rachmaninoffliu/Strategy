"""
Alpha Syndicate — Multi-Horizon Dual Momentum (MHDM), ETF Universe
Data: Alpaca (alpaca-py) | Indicators: TA-Lib 0.7.1

Team proposal approved by System Owner. Design rationale:

  QR:    NDX 12-1 shows IS 13.62% -> OOS 4.60% CAGR. That decay signature is
         single-lookback overfitting + factor crowding. Fix is a BLENDED signal
         across 4 horizons so no single lookback is privileged and there is no
         lookback parameter left to overfit.
  DA:    Alpaca history starts 2016 and delisted-symbol coverage is undocumented.
         A broad single-stock universe would be survivorship-biased (inflates CAGR
         ~1-4%/yr). ETF universe makes that bias structurally ~zero.
  RM:    Vol target 10%, per-position cap 35%, drawdown circuit-breaker at -15%.
         Absolute-momentum gate provides a genuine all-cash state.
  Valid: Conditions of sign-off, all enforced below --
           1. No parameter tuned on test data (horizons fixed by convention).
           2. Signal on t-1 close, fill at t open. No same-bar execution.
           3. Costs modelled: 1bp commission + 2bp slippage.
           4. Deflated Sharpe reported alongside raw Sharpe.
           5. Annual OOS segments reported for stability.
  QT:    Paper only. Monthly ETF rebalance is trivially executable.

NOTE ON "WALK-FORWARD": this strategy has zero fitted parameters, so there is
nothing to train. The annual segmentation below is a STABILITY ANALYSIS (is the
edge consistent across years?), not a parameter optimisation. Labelling it
"walk-forward optimisation" would overstate what was done.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import talib
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Momentum horizons (trading days). Fixed by convention: 1M/3M/6M/12M.
# Valid: these are NOT tuned. Changing them to improve results invalidates the run.
HORIZONS         = [21, 63, 126, 252]
SKIP_DAYS        = 21          # short-term reversal guard, applied to all horizons
TREND_SMA        = 200         # absolute-momentum gate: price > 200d SMA
ABS_MOM_DAYS     = 252         # absolute-momentum gate: 12M total return > 0
ATR_DAYS         = 20          # inverse-vol sizing input
VOL_LOOKBACK     = 60          # realised portfolio vol window for vol targeting
VOL_TARGET       = 0.10        # RM constraint — 10% annualised
MAX_WEIGHT       = 0.35        # RM constraint — per-position cap
TOP_N            = 4           # holdings; unfilled slots -> cash proxy
DD_BREAKER       = -0.15       # RM constraint — de-risk to 50% gross below this
DD_BREAKER_SCALE = 0.50
COMMISSION_BPS   = 1.0         # Valid constraint
SLIPPAGE_BPS     = 2.0         # Valid constraint
N_TRIALS_DSR     = 10          # design variants considered, for Deflated Sharpe

DATA_START       = "2016-01-01"   # Alpaca history floor
END_DATE         = datetime.now().strftime("%Y-%m-%d")
TRADING_DAYS     = 252

CASH_PROXY       = "BIL"
UNIVERSE = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",   # sectors
    "SPY", "QQQ", "IWM",                                       # broad
    "TLT", "GLD",                                              # defensive
    CASH_PROXY,                                                # cash proxy
]
BENCHMARK        = "SPY"

DATA_DIR         = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data"
OUTPUT_DIR       = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
CACHE_FILE       = os.path.join(DATA_DIR, "mhdm_etf_bars.parquet")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── CREDENTIALS ──────────────────────────────────────────────────────────────
def load_credentials():
    """Env vars take precedence; fall back to the alpaca-paper MCP entry.

    Never prints or returns secrets to stdout.
    """
    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if key and secret:
        return key, secret, "environment"

    cfg = os.path.expanduser("~/.claude.json")
    if os.path.exists(cfg):
        with open(cfg, "r", encoding="utf-8") as fh:
            servers = json.load(fh).get("mcpServers", {})
        env = servers.get("alpaca-paper", {}).get("env", {})
        key, secret = env.get("ALPACA_API_KEY"), env.get("ALPACA_SECRET_KEY")
        if key and secret and not key.startswith("YOUR_"):
            return key, secret, "alpaca-paper MCP config"

    raise SystemExit(
        "No Alpaca credentials found. Set ALPACA_API_KEY / ALPACA_SECRET_KEY, "
        "or populate the alpaca-paper entry in ~/.claude.json"
    )


# ── DATA ─────────────────────────────────────────────────────────────────────
def fetch_bars(client, symbols, start, end, feed):
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start,
        end=end,
        adjustment=Adjustment.ALL,      # total return: splits + dividends
        feed=feed,
    )
    return client.get_stock_bars(req).df


def download_data(client, symbols, start, end):
    """Try SIP (consolidated tape); fall back to IEX if not subscribed."""
    for feed, label in ((DataFeed.SIP, "SIP (consolidated)"),
                        (DataFeed.IEX, "IEX (free tier)")):
        try:
            print(f"  Trying {label} feed...")
            df = fetch_bars(client, symbols, start, end, feed)
            print(f"  OK — using {label}")
            return df, label
        except Exception as exc:
            msg = str(exc)[:120]
            print(f"  {label} unavailable: {msg}")
    raise SystemExit("Could not retrieve bars from any feed.")


def load_prices(client):
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached bars: {CACHE_FILE}")
        panel = pd.read_parquet(CACHE_FILE)
        feed_label = "cached"
    else:
        print(f"Downloading {len(UNIVERSE)} ETFs from Alpaca "
              f"({DATA_START} -> {END_DATE})...")
        raw, feed_label = download_data(client, UNIVERSE, DATA_START, END_DATE)
        panel = raw.reset_index()
        panel.to_parquet(CACHE_FILE, index=False)
        print(f"  Cached to {CACHE_FILE}")

    panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_localize(None)
    panel["date"] = panel["timestamp"].dt.normalize()

    fields = {}
    for col in ("open", "high", "low", "close"):
        fields[col] = panel.pivot_table(index="date", columns="symbol",
                                        values=col).sort_index()

    # Require full coverage across the universe; momentum ranks need a stable panel.
    common = None
    for frame in fields.values():
        ok = frame.dropna(axis=1, thresh=int(len(frame) * 0.90)).columns
        common = ok if common is None else common.intersection(ok)

    fields = {k: v[common].ffill() for k, v in fields.items()}
    fields = {k: v.dropna() for k, v in fields.items()}
    idx = fields["close"].index
    for k in fields:
        fields[k] = fields[k].reindex(idx)

    missing = sorted(set(UNIVERSE) - set(common))
    if missing:
        print(f"  WARNING — dropped (insufficient history): {missing}")
    print(f"  Panel: {len(common)} symbols | {len(idx)} days "
          f"| {idx[0].date()} -> {idx[-1].date()}")
    return fields, feed_label


# ── SIGNALS (TA-Lib) ─────────────────────────────────────────────────────────
def build_signals(fields):
    """Multi-horizon momentum blend + absolute-momentum gates + ATR sizing input.

    All series are causal: values at index t use only data up to and including t.
    """
    close, high, low = fields["close"], fields["high"], fields["low"]
    symbols = close.columns

    roc = {h: pd.DataFrame(index=close.index, columns=symbols, dtype=float)
           for h in HORIZONS}
    sma  = pd.DataFrame(index=close.index, columns=symbols, dtype=float)
    absm = pd.DataFrame(index=close.index, columns=symbols, dtype=float)
    atr  = pd.DataFrame(index=close.index, columns=symbols, dtype=float)

    for sym in symbols:
        c = close[sym].to_numpy(dtype=float)
        h_ = high[sym].to_numpy(dtype=float)
        l_ = low[sym].to_numpy(dtype=float)
        for h in HORIZONS:
            roc[h][sym] = talib.ROC(c, timeperiod=h)
        sma[sym]  = talib.SMA(c, timeperiod=TREND_SMA)
        absm[sym] = talib.ROC(c, timeperiod=ABS_MOM_DAYS)
        atr[sym]  = talib.ATR(h_, l_, c, timeperiod=ATR_DAYS)

    # Skip-month: shift each horizon forward so the most recent 21d are excluded.
    # Applied uniformly across horizons — no per-horizon special case to tune.
    z_sum = pd.DataFrame(0.0, index=close.index, columns=symbols)
    for h in HORIZONS:
        r = roc[h].shift(SKIP_DAYS)
        z = r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1).replace(0, np.nan), axis=0)
        z_sum += z.fillna(0.0)
    score = z_sum / len(HORIZONS)
    score[roc[HORIZONS[-1]].shift(SKIP_DAYS).isna()] = np.nan

    eligible = (absm.shift(SKIP_DAYS) > 0) & (close > sma)
    atr_pct  = (atr / close).replace(0, np.nan)
    return score, eligible, atr_pct


def target_weights(score_row, elig_row, atrp_row, returns_hist, dd_scale):
    """Top-N eligible by score, inverse-vol weighted, scaled to VOL_TARGET."""
    cands = score_row.dropna()
    cands = cands[elig_row.reindex(cands.index).fillna(False)]
    cands = cands.drop(labels=[CASH_PROXY], errors="ignore")

    weights = pd.Series(0.0, index=score_row.index)
    picks = cands.sort_values(ascending=False).head(TOP_N)
    if len(picks) == 0:
        weights[CASH_PROXY] = 1.0
        return weights

    inv = 1.0 / atrp_row.reindex(picks.index).replace(0, np.nan)
    inv = inv.dropna()
    if inv.empty:
        weights[CASH_PROXY] = 1.0
        return weights

    raw = inv / inv.sum()
    raw = raw.clip(upper=MAX_WEIGHT)
    # Only TOP_N slots are funded; unfilled slots stay in cash by construction.
    raw = raw * (len(picks) / TOP_N)

    # Vol targeting on the realised vol of this weight vector.
    hist = returns_hist[raw.index].tail(VOL_LOOKBACK)
    if len(hist) >= 20:
        port_vol = float(np.sqrt(TRADING_DAYS) * (hist @ raw).std())
        scale = VOL_TARGET / port_vol if port_vol > 1e-8 else 1.0
    else:
        scale = 1.0
    scale = min(scale, 1.0 / max(raw.sum(), 1e-12)) * dd_scale

    sized = (raw * scale).clip(upper=MAX_WEIGHT)
    gross = sized.sum()
    if gross > 1.0:
        sized = sized / gross
        gross = 1.0

    weights[sized.index] = sized.values
    weights[CASH_PROXY] = weights.get(CASH_PROXY, 0.0) + (1.0 - gross)
    return weights


# ── BACKTEST ─────────────────────────────────────────────────────────────────
def run_backtest(fields, score, eligible, atr_pct):
    """Monthly rebalance. Signal on t-1 close, executed at t open.

    Positions are held in shares between rebalances, so intra-month P&L is
    marked on close-to-close moves of the actual holdings.
    """
    close, open_ = fields["close"], fields["open"]
    dates = close.index
    daily_ret = close.pct_change().fillna(0.0)

    # Rebalance on the first trading day of each month; signal from prior close.
    month_id = pd.Series(dates.year * 100 + dates.month, index=dates)
    is_first = month_id != month_id.shift(1)
    warmup = max(HORIZONS) + SKIP_DAYS + 5
    rebal_dates = [d for i, d in enumerate(dates) if is_first.iloc[i] and i >= warmup]

    equity = 1.0
    peak = 1.0
    weights = pd.Series(0.0, index=close.columns)
    weights[CASH_PROXY] = 1.0

    records, turnovers, weight_log = [], [], {}

    for i, d in enumerate(dates):
        if i == 0:
            records.append((d, equity, 0.0))
            continue

        # Mark existing positions on close-to-close.
        equity *= (1.0 + float((weights * daily_ret.loc[d]).sum()))

        if d in rebal_dates:
            prev = dates[i - 1]                       # signal date (t-1 close)
            dd = equity / peak - 1.0
            dd_scale = DD_BREAKER_SCALE if dd < DD_BREAKER else 1.0

            tgt = target_weights(
                score.loc[prev], eligible.loc[prev], atr_pct.loc[prev],
                daily_ret.loc[:prev], dd_scale,
            )

            turnover = float((tgt - weights).abs().sum())
            # Fill at today's open: adjust for the close->open gap already earned.
            gap = (open_.loc[d] / close.loc[prev] - 1.0).fillna(0.0)
            equity *= (1.0 + float(((tgt - weights) * gap).sum()))

            cost = turnover * (COMMISSION_BPS + SLIPPAGE_BPS) / 1e4
            equity *= (1.0 - cost)

            weights = tgt
            turnovers.append((d, turnover))
            weight_log[d] = tgt[tgt > 1e-6].copy()

        peak = max(peak, equity)
        records.append((d, equity, float(weights.drop(CASH_PROXY, errors="ignore").sum())))

    curve = pd.DataFrame(records, columns=["date", "equity", "gross"]).set_index("date")
    turn = pd.DataFrame(turnovers, columns=["date", "turnover"]).set_index("date")
    return curve, turn, weight_log, rebal_dates


# ── METRICS ──────────────────────────────────────────────────────────────────
def deflated_sharpe(returns, n_trials):
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Computed on per-period (non-annualised) Sharpe, as the formula requires.
    Returns probability that true Sharpe > 0 after adjusting for selection bias.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return np.nan, np.nan

    sr = r.mean() / r.std(ddof=1)
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurtosis()) + 3.0        # non-excess kurtosis

    # Expected max Sharpe under the null across n_trials independent trials.
    e = math.e
    gamma = 0.5772156649
    sr_var = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (T - 1)
    sr_std = math.sqrt(max(sr_var, 1e-18))
    sr0 = sr_std * ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
                    + gamma * norm.ppf(1 - 1.0 / (n_trials * e)))

    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-18))
    dsr = norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom)
    return float(dsr), float(sr0 * math.sqrt(TRADING_DAYS))


def summarise(curve, label, n_trials=N_TRIALS_DSR):
    eq = curve["equity"]
    rets = eq.pct_change().dropna()
    years = len(eq) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (rets.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1.0)
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (rets.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan
    dsr, sr0 = deflated_sharpe(rets, n_trials)
    return {
        "label": label, "years": years, "cagr": cagr, "vol": vol,
        "sharpe": sharpe, "sortino": sortino, "max_dd": dd.min(),
        "dsr": dsr, "sr0_haircut": sr0,
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
    }


def fmt_row(m):
    def pct(x): return f"{x*100:6.2f}%" if pd.notna(x) else "   n/a"
    def num(x): return f"{x:6.2f}" if pd.notna(x) else "   n/a"
    return (f"  {m['label']:<22} {pct(m['cagr'])}  {pct(m['vol'])}  "
            f"{num(m['sharpe'])}  {num(m['sortino'])}  {pct(m['max_dd'])}  "
            f"{num(m['calmar'])}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 78)
    print("Alpha Syndicate — Multi-Horizon Dual Momentum (ETF)")
    print("=" * 78)

    key, secret, source = load_credentials()
    print(f"Credentials loaded from: {source}")
    client = StockHistoricalDataClient(key, secret)

    fields, feed_label = load_prices(client)
    print(f"Data feed: {feed_label}")

    print("\nBuilding signals (TA-Lib ROC / SMA / ATR)...")
    score, eligible, atr_pct = build_signals(fields)

    print("Running backtest (monthly rebalance, t-1 signal -> t open fill)...")
    curve, turn, weight_log, rebal_dates = run_backtest(fields, score, eligible, atr_pct)

    # Benchmark: buy & hold, same window.
    bench_px = fields["close"][BENCHMARK].reindex(curve.index)
    bench = pd.DataFrame({"equity": bench_px / bench_px.iloc[0]})

    strat_m = summarise(curve, "MHDM (strategy)")
    bench_m = summarise(bench, f"{BENCHMARK} buy & hold", n_trials=1)

    print("\n" + "=" * 78)
    print("HEADLINE — full period")
    print("=" * 78)
    print(f"  {'':<22} {'CAGR':>7}  {'Vol':>7}  {'Sharpe':>6}  "
          f"{'Sortino':>6}  {'MaxDD':>7}  {'Calmar':>6}")
    print(fmt_row(strat_m))
    print(fmt_row(bench_m))

    # Raw CAGR is not comparable across different vol levels. Report the
    # vol-matched equivalent so the comparison is like-for-like.
    # Approximation: excess return scales with vol at constant Sharpe.
    # Ignores financing cost of the implied leverage — optimistic for MHDM.
    lev = bench_m["vol"] / strat_m["vol"] if strat_m["vol"] > 0 else np.nan
    matched = strat_m["sharpe"] * bench_m["vol"]
    print(f"\n  Vol-matched check (MHDM levered {lev:.2f}x to {BENCHMARK} vol):")
    print(f"    implied return {matched*100:.2f}%  vs  {BENCHMARK} {bench_m['cagr']*100:.2f}%")
    print(f"    (excludes financing cost of leverage — flatters MHDM)")

    print(f"\n  Deflated Sharpe (n_trials={N_TRIALS_DSR}): "
          f"{strat_m['dsr']:.3f}" if pd.notna(strat_m['dsr']) else "  DSR: n/a")
    print(f"  Selection-bias haircut (annualised SR0): {strat_m['sr0_haircut']:.3f}")
    print(f"  Avg monthly turnover: {turn['turnover'].mean():.1%}")
    print(f"  Avg gross exposure:   {curve['gross'].mean():.1%}")
    print(f"  Rebalances:           {len(rebal_dates)}")

    # ── Stability analysis: annual OOS segments (NOT parameter optimisation) ──
    print("\n" + "=" * 78)
    print("STABILITY — annual segments (no parameters were fitted)")
    print("=" * 78)
    print(f"  {'Year':<22} {'CAGR':>7}  {'Vol':>7}  {'Sharpe':>6}  "
          f"{'Sortino':>6}  {'MaxDD':>7}  {'Calmar':>6}")
    seg_rows = []
    for yr, grp in curve.groupby(curve.index.year):
        if len(grp) < 60:
            continue
        seg = grp.copy()
        seg["equity"] = seg["equity"] / seg["equity"].iloc[0]
        m = summarise(seg, str(yr), n_trials=1)
        seg_rows.append(m)
        print(fmt_row(m))

    if seg_rows:
        yr_cagrs = [m["cagr"] for m in seg_rows if pd.notna(m["cagr"])]
        pos = sum(1 for c in yr_cagrs if c > 0)
        print(f"\n  Positive years: {pos}/{len(yr_cagrs)}  "
              f"| median {np.median(yr_cagrs)*100:.2f}%  "
              f"| worst {min(yr_cagrs)*100:.2f}%  best {max(yr_cagrs)*100:.2f}%")

    # ── Outputs ──────────────────────────────────────────────────────────────
    eq_path = os.path.join(OUTPUT_DIR, "mhdm_etf_equity.csv")
    out = curve.copy()
    out["benchmark"] = bench["equity"]
    out.to_csv(eq_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(curve.index, curve["equity"], label="MHDM (strategy)", lw=1.4)
    ax1.plot(bench.index, bench["equity"], label=f"{BENCHMARK} buy & hold",
             lw=1.1, alpha=0.75)
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title("Multi-Horizon Dual Momentum — ETF universe "
                  f"({curve.index[0].date()} to {curve.index[-1].date()})")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    dd = curve["equity"] / curve["equity"].cummax() - 1.0
    ax2.fill_between(dd.index, dd * 100, 0, alpha=0.5, color="firebrick")
    ax2.axhline(DD_BREAKER * 100, ls="--", lw=1, color="black",
                label=f"RM breaker {DD_BREAKER:.0%}")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "mhdm_etf_equity.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    if weight_log:
        last_date = max(weight_log)
        print("\n" + "=" * 78)
        print(f"CURRENT TARGET ALLOCATION (as of {last_date.date()})")
        print("=" * 78)
        for sym, w in weight_log[last_date].sort_values(ascending=False).items():
            print(f"  {sym:<6} {w:6.1%}")

    print(f"\nSaved: {eq_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
