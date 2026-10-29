"""
Alpha Syndicate — Python port of "Main Supertrend Strategy - SOXL / SOXS"
Honest re-test on Alpaca SIP data with costs and capacity constraints.

WHY THE TRADINGVIEW RESULT CANNOT BE TAKEN AT FACE VALUE:
  The supplied CSV reports $100,000 -> $5,188,851,400 (CAGR 94.34%) over
  2010-2026. Diagnostics on that file:

    1. CAPACITY: average trade notional grows $39k (2010) -> $2.66 BILLION
       (2026). SOXL total AUM is roughly $10-15B. The backtest assumes it can
       trade ~20% of the entire fund, 156x per year, with zero market impact.
    2. COMMISSION: total commission in the file is EXACTLY $0 across 2,551
       trades / 5,102 fills.
    3. LEVERAGE: SOXL is a 3x leveraged ETF. Realised vol is 4.52x QQQ. This
       violates the "no leverage" constraint set for the previous research.
    4. BENCHMARK: comparing a 3x semiconductor fund to QQQ is not a like-for-
       like test. SOXL BUY & HOLD returned 50.56% CAGR over 2016-2026 with
       ZERO strategy logic. QQQ is the wrong bar.
    5. SCRIPT/CSV MISMATCH: the pasted Pine has useDateFilter=true and
       startDate=2023-01-01, but the CSV spans 2010-2026. The exported run
       used different settings than the code supplied.
    6. FILL OPTIMISM: entries are resting limit orders at st*(1 +/- buyback).
       TradingView fills a limit whenever price TOUCHES it. On 1h bars that is
       a generous assumption, and it applies to most entries.

  Prior project memory (2026-06-06) independently found the apparent edge was
  a POINT-WEIGHTING ARTIFACT and that on a percent-of-equity basis the
  Auto-Switch config was a loser (PF 0.942, Net -4.18%).

  NOTE IN FAIRNESS: on a percent basis THIS export shows PF 1.329 over 2,551
  trades with avg trade +0.640%. That is a real signal and survives per-trade
  costs. The problem is the MAGNITUDE, not the existence, of the edge.

WHAT THIS PORT DOES:
  Reimplements the core logic (Supertrend flip, fixed/dynamic targets,
  trailing stop, vol-regime switch) and tests it with:
    - real costs (configurable bps per leg)
    - a capacity cap on position notional
    - SOXL buy & hold as the primary benchmark (QQQ shown for reference)

SIMPLIFICATIONS (stated, not hidden):
  - Entries modelled as market-on-next-bar-open, NOT resting limit orders.
    This removes the fill optimism above. It will UNDERSTATE the TV result.
  - SOXS routing ignored; shorts are modelled as short SOXL (economically
    equivalent before borrow cost, and the Pine keeps the same direction).
  - Option strike inputs are unused in the Pine's logic and are dropped.
"""

import warnings
warnings.filterwarnings("ignore")

import os, json, math
import numpy as np
import pandas as pd
import talib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed

# -- PINE INPUTS (verbatim from the supplied script) ---------------------------
ATR_PERIOD   = 500
FACTOR       = 5.0
RISK_REWARD  = 4.5 / 100
TRAIL_FACTOR = 0.441
PRICE_PERC   = 0.2
RISK_FACTOR  = 1.0
LIMIT_FACTOR = 1.0
VOL_LENGTH   = 20
VOL_SMA_LEN  = 100
VOL_THRESH   = 1.3
ADX_LENGTH   = 14
ADX_BASE     = 25.0
USE_DYNAMIC  = True
EQUITY_FRAC  = 0.50          # strategy.equity * 0.5

# -- HONEST TEST SETTINGS ------------------------------------------------------
COST_BPS_LEG = 5.0           # per leg; SOXL spread is wide, 5bp is generous
CAPACITY_USD = 25_000_000    # max notional per trade (SOXL ADV sanity cap)
BARS_PER_DAY = 7             # 1h RTH bars
BARS_PER_YEAR= BARS_PER_DAY * 252

SYMBOL   = "SOXL"
BENCH    = "QQQ"
START    = "2016-01-01"
END      = "2026-08-22"
OUTDIR   = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
CACHE    = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data/soxl_1h_rth.parquet"
os.makedirs(OUTDIR, exist_ok=True)


def creds():
    import alpaca_keys                                                      # "Alpaca Key.txt" in the project folder, then the environment, then ~/.claude.json
    return alpaca_keys.creds(True)


def load_bars():
    if os.path.exists(CACHE):
        p = pd.read_parquet(CACHE)
    else:
        c = StockHistoricalDataClient(*creds())
        frames = []
        for yr in range(int(START[:4]), int(END[:4]) + 1):
            req = StockBarsRequest(symbol_or_symbols=[SYMBOL, BENCH],
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=f"{yr}-01-01", end=min(f"{yr+1}-01-01", END),
                adjustment=Adjustment.ALL, feed=DataFeed.SIP)
            frames.append(c.get_stock_bars(req).df.reset_index())
        p = pd.concat(frames, ignore_index=True)
        ts = pd.to_datetime(p["timestamp"], utc=True).dt.tz_convert("America/New_York")
        p["et"] = ts
        p = p[(ts.dt.time >= pd.Timestamp("09:30").time())
              & (ts.dt.time < pd.Timestamp("16:00").time())
              & (ts.dt.dayofweek < 5)].copy()
        p.to_parquet(CACHE, index=False)
    p["et"] = pd.to_datetime(p["et"], utc=True).dt.tz_convert("America/New_York")
    p["bar"] = p["et"].dt.tz_localize(None)
    out = {}
    for c_ in ("open", "high", "low", "close"):
        out[c_] = p.pivot_table(index="bar", columns="symbol", values=c_).sort_index().ffill()
    idx = out["close"].dropna().index
    return {k: v.reindex(idx) for k, v in out.items()}


def supertrend(high, low, close, period, factor):
    """TradingView ta.supertrend equivalent. dir=-1 uptrend, dir=+1 downtrend."""
    atr = talib.ATR(high, low, close, timeperiod=period)
    hl2 = (high + low) / 2.0
    up, dn = hl2 + factor * atr, hl2 - factor * atr
    n = len(close)
    st = np.full(n, np.nan)
    d = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if i == 0 or np.isnan(st[i-1]):
            st[i], d[i] = dn[i], 1.0
            continue
        pu, pd_ = up[i-1], dn[i-1]
        dn[i] = max(dn[i], pd_) if close[i-1] > pd_ else dn[i]
        up[i] = min(up[i], pu) if close[i-1] < pu else up[i]
        if st[i-1] == pu:
            d[i] = -1.0 if close[i] > up[i] else 1.0
        else:
            d[i] = 1.0 if close[i] < dn[i] else -1.0
        st[i] = dn[i] if d[i] == -1.0 else up[i]
    return st, d


def run(fields, mode="Always Fixed", cost_bps=COST_BPS_LEG,
        capacity=CAPACITY_USD, equity_frac=EQUITY_FRAC):
    close = fields["close"][SYMBOL].to_numpy(float)
    high  = fields["high"][SYMBOL].to_numpy(float)
    low   = fields["low"][SYMBOL].to_numpy(float)
    open_ = fields["open"][SYMBOL].to_numpy(float)
    idx   = fields["close"].index

    st, d = supertrend(high, low, close, ATR_PERIOD, FACTOR)
    atrv = talib.ATR(high, low, close, timeperiod=VOL_LENGTH) / close
    base = talib.SMA(atrv, timeperiod=VOL_SMA_LEN)
    _, _, adx = talib.ADX(high, low, close, timeperiod=ADX_LENGTH), None, talib.ADX(high, low, close, timeperiod=ADX_LENGTH)

    eq = 100000.0
    pos = 0          # +1 long, -1 short, 0 flat
    qty = 0.0
    entry = np.nan; target = np.nan; fixed = True
    mx = np.nan; mn = np.nan
    curve, trades = [], []

    for i in range(1, len(close)):
        if pos != 0:
            eq += qty * (close[i] - close[i-1]) * pos

        if np.isnan(d[i]) or np.isnan(st[i]):
            curve.append((idx[i], eq)); continue

        hv = (not np.isnan(base[i])) and atrv[i] > base[i] * VOL_THRESH
        is_fixed = (mode == "Always Fixed") or (mode == "Auto-Switch via Vol" and hv)

        want = 1 if d[i] == -1 else -1
        flip = (want != pos)

        # -- manage open position: trailing stop + limit take-profit ----------
        if pos != 0 and not flip:
            if pos > 0:
                mx = high[i] if np.isnan(mx) else max(mx, high[i])
                tp_lim = entry * (1 + PRICE_PERC) if fixed else entry * (1 + (entry - st[i]) / entry * LIMIT_FACTOR)
                if mx >= target:
                    trail = (mx - entry) * (TRAIL_FACTOR if not USE_DYNAMIC else
                             max(0.05, min(0.95, TRAIL_FACTOR + ((ADX_BASE - (ADX_BASE if np.isnan(adx[i]) else adx[i])) / 100)))) + entry
                    if low[i] <= trail:
                        eq -= abs(qty) * close[i] * cost_bps / 1e4
                        trades.append(pos); pos, qty, mx = 0, 0.0, np.nan
                if pos > 0 and high[i] >= tp_lim:
                    eq -= abs(qty) * close[i] * cost_bps / 1e4
                    trades.append(pos); pos, qty, mx = 0, 0.0, np.nan
            else:
                mn = low[i] if np.isnan(mn) else min(mn, low[i])
                tp_lim = entry * (1 - PRICE_PERC) if fixed else entry * (1 - (st[i] - entry) / entry * LIMIT_FACTOR)
                if mn <= target:
                    trail = entry - (entry - mn) * (TRAIL_FACTOR if not USE_DYNAMIC else
                             max(0.05, min(0.95, TRAIL_FACTOR + ((ADX_BASE - (ADX_BASE if np.isnan(adx[i]) else adx[i])) / 100))))
                    if high[i] >= trail:
                        eq -= abs(qty) * close[i] * cost_bps / 1e4
                        trades.append(pos); pos, qty, mn = 0, 0.0, np.nan
                if pos < 0 and low[i] <= tp_lim:
                    eq -= abs(qty) * close[i] * cost_bps / 1e4
                    trades.append(pos); pos, qty, mn = 0, 0.0, np.nan

        # -- flip / entry: market at next bar open ---------------------------
        if flip and i + 1 < len(close):
            px = open_[i + 1]
            if pos != 0:
                eq -= abs(qty) * px * cost_bps / 1e4
                trades.append(pos)
            notional = min(eq * equity_frac, capacity)
            qty = math.floor(notional / px) if px > 0 else 0
            if qty > 0:
                eq -= qty * px * cost_bps / 1e4
                pos = want
                entry = px
                fixed = is_fixed
                risk = abs(entry - st[i]) / entry * RISK_FACTOR
                target = entry * (1 + (RISK_REWARD if fixed else risk)) if pos > 0 \
                         else entry * (1 - (RISK_REWARD if fixed else risk))
                mx = mn = np.nan
            else:
                pos, qty = 0, 0.0
        curve.append((idx[i], eq))

    c = pd.DataFrame(curve, columns=["bar", "equity"]).set_index("bar")["equity"]
    return c, len(trades)


def stats(eq, label, ppy=BARS_PER_YEAR):
    eq = eq / eq.iloc[0]
    r = eq.pct_change().dropna()
    yrs = len(eq) / ppy
    vol = r.std() * np.sqrt(ppy)
    return {"label": label, "cagr": eq.iloc[-1] ** (1 / yrs) - 1, "vol": vol,
            "sharpe": (r.mean() * ppy) / vol if vol > 0 else np.nan,
            "dd": (eq / eq.cummax() - 1).min(), "final": eq.iloc[-1]}


def main():
    print("=" * 96)
    print("SUPERTREND SOXL — honest Python re-test")
    print("=" * 96)
    f = load_bars()
    idx = f["close"].index
    print(f"Bars: {len(idx):,} 1h RTH | {idx[0].date()} -> {idx[-1].date()}")
    print(f"Costs: {COST_BPS_LEG}bp/leg | capacity cap: ${CAPACITY_USD:,}\n")

    rows = []
    bh = f["close"][SYMBOL]; rows.append(stats(bh, "SOXL buy & hold"))
    qq = f["close"][BENCH];  rows.append(stats(qq, "QQQ buy & hold"))

    for mode in ("Always Fixed", "Auto-Switch via Vol", "Always Dynamic"):
        c, n = run(f, mode=mode)
        s = stats(c, f"Supertrend [{mode}]"); s["trades"] = n
        rows.append(s)

    # no-cost / no-capacity variant to isolate how much of TV's result is assumptions
    c0, n0 = run(f, mode="Always Fixed", cost_bps=0.0, capacity=1e18)
    s0 = stats(c0, "Supertrend [Fixed] NO cost/NO cap"); s0["trades"] = n0
    rows.append(s0)

    print(f"{'':<40}{'CAGR':>10}{'Vol':>9}{'Sharpe':>8}{'MaxDD':>9}{'x money':>12}")
    print("=" * 96)
    for m in rows:
        print(f"  {m['label']:<38}{m['cagr']*100:9.2f}%{m['vol']*100:8.2f}%"
              f"{m['sharpe']:8.2f}{m['dd']*100:8.2f}%{m['final']:11,.1f}x")

    print("\n" + "=" * 96)
    print("VERDICT — correct benchmark is SOXL buy & hold, NOT QQQ")
    print("=" * 96)
    b = rows[0]
    for m in rows[2:]:
        better = m["sharpe"] > b["sharpe"]
        print(f"  {m['label']:<40} Sharpe {m['sharpe']:.2f} vs SOXL b&h {b['sharpe']:.2f}"
              f"  -> {'BEATS' if better else 'loses to'} buy & hold")

    out = pd.DataFrame({"SOXL b&h": bh / bh.iloc[0], "QQQ b&h": qq / qq.iloc[0]})
    out.to_csv(os.path.join(OUTDIR, "supertrend_soxl_port.csv"))
    print(f"\nSaved: {OUTDIR}/supertrend_soxl_port.csv")


if __name__ == "__main__":
    main()
