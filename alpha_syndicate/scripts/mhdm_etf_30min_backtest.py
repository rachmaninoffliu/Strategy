"""
Alpha Syndicate — Multi-Horizon Dual Momentum on 30-MINUTE bars
Data: Alpaca SIP (alpaca-py) | Indicators: TA-Lib 0.7.1

Companion to mhdm_etf_backtest.py (daily). Same signal construction, run on
30-minute bars. Two modes are evaluated because "run it on 30 min" is ambiguous:

  MODE A — CALENDAR-PRESERVED
    Lookbacks stay at their economic length (1M/3M/6M/12M) but are measured in
    30-min bars. Rebalance weekly instead of monthly. This tests whether finer
    execution granularity adds anything to the daily result.

  MODE B — INTRADAY-SCALED
    Lookbacks rescaled to the timeframe: 1d/1w/2w/1m in bars. Rebalance daily.
    This is a genuinely DIFFERENT strategy — short-horizon momentum, which the
    literature says is where short-term REVERSAL dominates (Jegadeesh 1990,
    Lehmann 1990). Included because it is the natural reading of the request,
    with the prior that it will underperform.

DATA QUALITY NOTE (DA):
  Alpaca returns ~31 bars/day at 30min — it includes pre-market, after-hours
  and overnight bars. Those are thin, wide-spread and unreliable for ATR and
  momentum. This script filters to the REGULAR SESSION ONLY (09:30-16:00 ET,
  DST-aware via tz conversion), giving 13 bars/day.

VALID — SELECTION BIAS WARNING:
  Testing 2 additional variants raises the trial count used by the Deflated
  Sharpe Ratio from 10 to 12. Testing more variants makes DSR WORSE, not
  better. Every additional configuration tried is a larger haircut. Do not
  read a higher raw Sharpe here as an improvement without checking DSR.
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

# -- CONFIG --------------------------------------------------------------------
BAR_MINUTES      = 30
SESSION_START    = "09:30"     # ET, regular session
SESSION_END      = "16:00"
BARS_PER_DAY     = 13          # 6.5h / 30min
TRADING_DAYS     = 252
BARS_PER_YEAR    = BARS_PER_DAY * TRADING_DAYS

# Risk / sizing — IDENTICAL to the daily run. Not re-tuned.
VOL_TARGET       = 0.10
MAX_WEIGHT       = 0.35
TOP_N            = 4
DD_BREAKER       = -0.15
DD_BREAKER_SCALE = 0.50
COMMISSION_BPS   = 1.0
SLIPPAGE_BPS     = 2.0
N_TRIALS_DSR     = 12          # raised from 10: two more variants tested

MODES = {
    "A_calendar": {
        "label":      "Mode A — calendar-preserved (1M/3M/6M/12M, weekly reb)",
        "horizons":   [21 * BARS_PER_DAY, 63 * BARS_PER_DAY,
                       126 * BARS_PER_DAY, 252 * BARS_PER_DAY],
        "skip":       21 * BARS_PER_DAY,
        "trend_sma":  200 * BARS_PER_DAY,
        "abs_mom":    252 * BARS_PER_DAY,
        "atr":        20 * BARS_PER_DAY,
        "vol_lb":     60 * BARS_PER_DAY,
        "reb_bars":   5 * BARS_PER_DAY,      # weekly
    },
    "B_intraday": {
        "label":      "Mode B — intraday-scaled (1d/1w/2w/1m, daily reb)",
        "horizons":   [BARS_PER_DAY, 5 * BARS_PER_DAY,
                       10 * BARS_PER_DAY, 21 * BARS_PER_DAY],
        "skip":       BARS_PER_DAY,          # skip 1 day, not 1 month
        "trend_sma":  20 * BARS_PER_DAY,
        "abs_mom":    21 * BARS_PER_DAY,
        "atr":        5 * BARS_PER_DAY,
        "vol_lb":     20 * BARS_PER_DAY,
        "reb_bars":   BARS_PER_DAY,          # daily
    },
}

DATA_START       = "2016-01-01"
END_DATE         = datetime.now().strftime("%Y-%m-%d")

CASH_PROXY       = "BIL"
UNIVERSE = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "SPY", "QQQ", "IWM",
    "TLT", "GLD",
    CASH_PROXY,
]
BENCHMARK        = "SPY"

DATA_DIR         = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data"
OUTPUT_DIR       = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
CACHE_FILE       = os.path.join(DATA_DIR, f"mhdm_etf_{BAR_MINUTES}min_rth.parquet")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- CREDENTIALS --------------------------------------------------------------
def load_credentials():
    key    = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if key and secret:
        return key, secret, "environment"
    cfg = os.path.expanduser("~/.claude.json")
    if os.path.exists(cfg):
        with open(cfg, "r", encoding="utf-8") as fh:
            env = json.load(fh).get("mcpServers", {}) \
                    .get("alpaca-paper", {}).get("env", {})
        key, secret = env.get("ALPACA_API_KEY"), env.get("ALPACA_SECRET_KEY")
        if key and secret and not key.startswith("YOUR_"):
            return key, secret, "alpaca-paper MCP config"
    raise SystemExit("No Alpaca credentials found.")


# -- DATA ---------------------------------------------------------------------
def download_chunked(client, symbols, start, end):
    """Fetch year by year — 10y of 30min bars across 14 symbols is ~1M rows."""
    frames = []
    for yr in range(int(start[:4]), int(end[:4]) + 1):
        y0 = f"{yr}-01-01"
        y1 = f"{min(yr + 1, int(end[:4]) + 1)}-01-01"
        if y0 >= end:
            break
        print(f"    {yr}...", end="", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
            start=y0, end=min(y1, end),
            adjustment=Adjustment.ALL,
            feed=DataFeed.SIP,
        )
        df = client.get_stock_bars(req).df
        frames.append(df.reset_index())
        print(f" {len(df):,} rows")
    return pd.concat(frames, ignore_index=True)


def load_prices(client):
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached RTH bars: {CACHE_FILE}")
        panel = pd.read_parquet(CACHE_FILE)
    else:
        print(f"Downloading {len(UNIVERSE)} ETFs @ {BAR_MINUTES}min "
              f"({DATA_START} -> {END_DATE})...")
        panel = download_chunked(client, UNIVERSE, DATA_START, END_DATE)

        # -- RTH filter (DST-aware) --
        before = len(panel)
        ts = pd.to_datetime(panel["timestamp"], utc=True).dt.tz_convert("America/New_York")
        panel["ts_et"] = ts
        in_session = (ts.dt.time >= pd.Timestamp(SESSION_START).time()) & \
                     (ts.dt.time < pd.Timestamp(SESSION_END).time()) & \
                     (ts.dt.dayofweek < 5)
        panel = panel[in_session].copy()
        print(f"  RTH filter: {before:,} -> {len(panel):,} rows "
              f"({len(panel)/before:.0%} retained)")
        panel.to_parquet(CACHE_FILE, index=False)
        print(f"  Cached to {CACHE_FILE}")

    panel["ts_et"] = pd.to_datetime(panel["ts_et"], utc=True).dt.tz_convert("America/New_York")
    panel["bar"] = panel["ts_et"].dt.tz_localize(None)

    fields = {}
    for col in ("open", "high", "low", "close"):
        fields[col] = panel.pivot_table(index="bar", columns="symbol",
                                        values=col).sort_index()

    common = None
    for frame in fields.values():
        ok = frame.dropna(axis=1, thresh=int(len(frame) * 0.90)).columns
        common = ok if common is None else common.intersection(ok)

    fields = {k: v[common].ffill().dropna() for k, v in fields.items()}
    idx = fields["close"].index
    for k in fields:
        fields[k] = fields[k].reindex(idx)

    missing = sorted(set(UNIVERSE) - set(common))
    if missing:
        print(f"  WARNING — dropped: {missing}")
    obs_per_day = len(idx) / idx.normalize().nunique()
    print(f"  Panel: {len(common)} symbols | {len(idx):,} bars "
          f"| {obs_per_day:.1f} bars/day | {idx[0].date()} -> {idx[-1].date()}")
    return fields


# -- SIGNALS ------------------------------------------------------------------
def build_signals(fields, cfg):
    close, high, low = fields["close"], fields["high"], fields["low"]
    symbols = close.columns

    roc  = {h: pd.DataFrame(index=close.index, columns=symbols, dtype=float)
            for h in cfg["horizons"]}
    sma  = pd.DataFrame(index=close.index, columns=symbols, dtype=float)
    absm = pd.DataFrame(index=close.index, columns=symbols, dtype=float)
    atr  = pd.DataFrame(index=close.index, columns=symbols, dtype=float)

    for sym in symbols:
        c  = close[sym].to_numpy(dtype=float)
        h_ = high[sym].to_numpy(dtype=float)
        l_ = low[sym].to_numpy(dtype=float)
        for h in cfg["horizons"]:
            roc[h][sym] = talib.ROC(c, timeperiod=h)
        sma[sym]  = talib.SMA(c, timeperiod=cfg["trend_sma"])
        absm[sym] = talib.ROC(c, timeperiod=cfg["abs_mom"])
        atr[sym]  = talib.ATR(h_, l_, c, timeperiod=cfg["atr"])

    z_sum = pd.DataFrame(0.0, index=close.index, columns=symbols)
    for h in cfg["horizons"]:
        r = roc[h].shift(cfg["skip"])
        z = r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1).replace(0, np.nan), axis=0)
        z_sum += z.fillna(0.0)
    score = z_sum / len(cfg["horizons"])
    score[roc[cfg["horizons"][-1]].shift(cfg["skip"]).isna()] = np.nan

    eligible = (absm.shift(cfg["skip"]) > 0) & (close > sma)
    atr_pct  = (atr / close).replace(0, np.nan)
    return score, eligible, atr_pct


def target_weights(score_row, elig_row, atrp_row, returns_hist, dd_scale, cfg):
    cands = score_row.dropna()
    cands = cands[elig_row.reindex(cands.index).fillna(False)]
    cands = cands.drop(labels=[CASH_PROXY], errors="ignore")

    weights = pd.Series(0.0, index=score_row.index)
    picks = cands.sort_values(ascending=False).head(TOP_N)
    if len(picks) == 0:
        weights[CASH_PROXY] = 1.0
        return weights

    inv = (1.0 / atrp_row.reindex(picks.index).replace(0, np.nan)).dropna()
    if inv.empty:
        weights[CASH_PROXY] = 1.0
        return weights

    raw = (inv / inv.sum()).clip(upper=MAX_WEIGHT) * (len(picks) / TOP_N)

    hist = returns_hist[raw.index].tail(cfg["vol_lb"])
    if len(hist) >= 20:
        port_vol = float(np.sqrt(BARS_PER_YEAR) * (hist @ raw).std())
        scale = VOL_TARGET / port_vol if port_vol > 1e-8 else 1.0
    else:
        scale = 1.0
    scale = min(scale, 1.0 / max(raw.sum(), 1e-12)) * dd_scale

    sized = (raw * scale).clip(upper=MAX_WEIGHT)
    gross = sized.sum()
    if gross > 1.0:
        sized, gross = sized / gross, 1.0

    weights[sized.index] = sized.values
    weights[CASH_PROXY] = weights.get(CASH_PROXY, 0.0) + (1.0 - gross)
    return weights


# -- BACKTEST -----------------------------------------------------------------
def run_backtest(fields, score, eligible, atr_pct, cfg):
    close, open_ = fields["close"], fields["open"]
    bars = close.index
    bar_ret = close.pct_change().fillna(0.0)

    warmup = max(cfg["horizons"]) + cfg["skip"] + 5
    reb_idx = set(range(warmup, len(bars), cfg["reb_bars"]))

    equity, peak = 1.0, 1.0
    weights = pd.Series(0.0, index=close.columns)
    weights[CASH_PROXY] = 1.0

    records, turnovers, weight_log = [], [], {}

    for i, b in enumerate(bars):
        if i == 0:
            records.append((b, equity, 0.0))
            continue

        equity *= (1.0 + float((weights * bar_ret.loc[b]).sum()))

        if i in reb_idx:
            prev = bars[i - 1]
            dd = equity / peak - 1.0
            dd_scale = DD_BREAKER_SCALE if dd < DD_BREAKER else 1.0

            tgt = target_weights(score.loc[prev], eligible.loc[prev],
                                 atr_pct.loc[prev], bar_ret.loc[:prev],
                                 dd_scale, cfg)

            turnover = float((tgt - weights).abs().sum())
            gap = (open_.loc[b] / close.loc[prev] - 1.0).fillna(0.0)
            equity *= (1.0 + float(((tgt - weights) * gap).sum()))
            equity *= (1.0 - turnover * (COMMISSION_BPS + SLIPPAGE_BPS) / 1e4)

            weights = tgt
            turnovers.append((b, turnover))
            weight_log[b] = tgt[tgt > 1e-6].copy()

        peak = max(peak, equity)
        records.append((b, equity,
                        float(weights.drop(CASH_PROXY, errors="ignore").sum())))

    curve = pd.DataFrame(records, columns=["bar", "equity", "gross"]).set_index("bar")
    turn = pd.DataFrame(turnovers, columns=["bar", "turnover"]).set_index("bar")
    return curve, turn, weight_log


# -- METRICS ------------------------------------------------------------------
def deflated_sharpe(returns, n_trials, ppy):
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return np.nan, np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurtosis()) + 3.0
    gamma, e = 0.5772156649, math.e
    sr_var = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2) / (T - 1)
    sr_std = math.sqrt(max(sr_var, 1e-18))
    sr0 = sr_std * ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
                    + gamma * norm.ppf(1 - 1.0 / (n_trials * e)))
    denom = math.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-18))
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom)), \
           float(sr0 * math.sqrt(ppy))


def summarise(curve, label, ppy, n_trials=N_TRIALS_DSR):
    eq = curve["equity"]
    rets = eq.pct_change().dropna()
    years = len(eq) / ppy
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(ppy)
    sharpe = (rets.mean() * ppy) / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    downside = rets[rets < 0].std() * np.sqrt(ppy)
    sortino = (rets.mean() * ppy) / downside if downside > 0 else np.nan
    dsr, sr0 = deflated_sharpe(rets, n_trials, ppy)
    return {"label": label, "years": years, "cagr": cagr, "vol": vol,
            "sharpe": sharpe, "sortino": sortino, "max_dd": dd.min(),
            "dsr": dsr, "sr0": sr0,
            "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan}


def fmt_row(m):
    def pct(x): return f"{x*100:7.2f}%" if pd.notna(x) else "    n/a"
    def num(x): return f"{x:6.2f}" if pd.notna(x) else "   n/a"
    return (f"  {m['label']:<46} {pct(m['cagr'])} {pct(m['vol'])} "
            f"{num(m['sharpe'])} {pct(m['max_dd'])} {num(m['calmar'])}")


# -- MAIN ---------------------------------------------------------------------
def main():
    print("=" * 100)
    print(f"Alpha Syndicate — MHDM on {BAR_MINUTES}-MINUTE bars (regular session only)")
    print("=" * 100)

    key, secret, source = load_credentials()
    print(f"Credentials: {source}")
    client = StockHistoricalDataClient(key, secret)

    fields = load_prices(client)

    bench_px = fields["close"][BENCHMARK]
    bench = pd.DataFrame({"equity": bench_px / bench_px.iloc[0]})
    bench_m = summarise(bench, f"{BENCHMARK} buy & hold", BARS_PER_YEAR, n_trials=1)

    results, curves = [], {}
    for name, cfg in MODES.items():
        print(f"\n{'-'*100}\n{cfg['label']}\n{'-'*100}")
        print(f"  horizons(bars) = {cfg['horizons']}  skip={cfg['skip']}  "
              f"rebalance every {cfg['reb_bars']} bars")
        score, eligible, atr_pct = build_signals(fields, cfg)
        curve, turn, wlog = run_backtest(fields, score, eligible, atr_pct, cfg)
        m = summarise(curve, cfg["label"], BARS_PER_YEAR)
        m["turnover"] = turn["turnover"].mean()
        m["reb_count"] = len(turn)
        m["gross"] = curve["gross"].mean()
        m["cost_drag"] = (turn["turnover"].mean()
                          * (COMMISSION_BPS + SLIPPAGE_BPS) / 1e4
                          * (BARS_PER_YEAR / cfg["reb_bars"]))
        results.append(m)
        curves[name] = curve
        print(f"  rebalances={m['reb_count']:,}  avg turnover={m['turnover']:.1%}  "
              f"avg gross={m['gross']:.1%}")
        print(f"  annualised cost drag ~ {m['cost_drag']*100:.2f}%")

    print("\n" + "=" * 100)
    print("HEADLINE")
    print("=" * 100)
    print(f"  {'':<46} {'CAGR':>8} {'Vol':>8} {'Sharpe':>6} {'MaxDD':>8} {'Calmar':>6}")
    for m in results:
        print(fmt_row(m))
    print(fmt_row(bench_m))

    print("\n" + "=" * 100)
    print("VALID — DEFLATED SHARPE (selection-bias adjusted)")
    print("=" * 100)
    print(f"  n_trials = {N_TRIALS_DSR} (daily run's 10 + 2 intraday variants)")
    print(f"  Convention: DSR >= 0.95 to accept. Below that, the edge is not "
          f"distinguishable from selection bias.\n")
    for m in results:
        verdict = "PASS" if (pd.notna(m["dsr"]) and m["dsr"] >= 0.95) else "FAIL"
        print(f"  {m['label']:<46}  DSR={m['dsr']:.3f}  "
              f"haircut={m['sr0']:.2f}  ->  {verdict}")

    # -- Output ---------------------------------------------------------------
    out = pd.DataFrame({n: c["equity"] for n, c in curves.items()})
    out[BENCHMARK] = bench["equity"]
    csv_path = os.path.join(OUTPUT_DIR, f"mhdm_etf_{BAR_MINUTES}min_equity.csv")
    out.to_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for name, c in curves.items():
        ax1.plot(c.index, c["equity"], lw=1.3, label=MODES[name]["label"])
    ax1.plot(bench.index, bench["equity"], lw=1.1, alpha=0.75,
             label=f"{BENCHMARK} buy & hold", color="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title(f"MHDM on {BAR_MINUTES}-min bars — regular session only "
                  f"({out.index[0].date()} to {out.index[-1].date()})")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    for name, c in curves.items():
        dd = c["equity"] / c["equity"].cummax() - 1.0
        ax2.plot(dd.index, dd * 100, lw=1.0, label=name)
    bdd = bench["equity"] / bench["equity"].cummax() - 1.0
    ax2.plot(bdd.index, bdd * 100, lw=1.0, color="black", alpha=0.6, label=BENCHMARK)
    ax2.axhline(DD_BREAKER * 100, ls="--", lw=1, color="firebrick")
    ax2.set_ylabel("Drawdown %")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(alpha=0.3)

    png_path = os.path.join(OUTPUT_DIR, f"mhdm_etf_{BAR_MINUTES}min_equity.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
