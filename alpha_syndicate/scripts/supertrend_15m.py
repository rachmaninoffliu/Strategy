"""
Alpha Syndicate — Supertrend SOXL strategy, faithful Python port on 15-min bars
Data: Alpaca SIP | Indicators: TA-Lib

FIX vs the first port attempt (which returned -29% CAGR and was WRONG):
  The first version only entered on Supertrend direction FLIPS. The Pine
  actually has TWO entry paths:

    (a) REVERSAL  -> market order, fires when position_size[1] is opposite sign
    (b) RE-ENTRY  -> LIMIT order at st*(1 +/- buyback), fires when flat and the
                     trend is still intact ("buyback" on a pullback)

  Path (b) was missing. After a take-profit the old port sat in cash for the
  rest of the trend instead of buying the dip back into the same trend. That
  single omission accounted for the entire divergence from TradingView.

LIMIT FILL RULE (deliberately conservative):
  TradingView fills a limit whenever price merely TOUCHES it. Here a long limit
  at L fills only if the bar's LOW trades strictly below L, and the fill price
  is min(open, L) -- so a gap through the level gives the better price, but a
  bar that only kisses the level does NOT fill. This is stricter than TV and
  will understate, not flatter, the result.

BENCHMARK:
  SOXL buy & hold is the primary bar. QQQ is shown for reference only -- SOXL
  is a 3x leveraged ETF and returned ~50-62% CAGR over this window with no
  strategy at all, so "beats QQQ" is a property of the instrument.

IMPROVEMENT VARIANTS (tested after the faithful port, all pre-registered):
  I1  vol-targeted sizing      -- cut size when realised vol spikes
  I2  drawdown circuit-breaker -- halve size below -20% from peak
  I3  I1 + I2 combined
  I4  reduced base sizing      -- 25% of equity instead of 50%
  Prior project memory (2026-06-06) independently found vol-targeted sizing --
  not the mode switch -- was the real regime adapter on this strategy.
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

# -- PINE INPUTS (verbatim) ---------------------------------------------------
ATR_PERIOD   = 500
FACTOR       = 5.0
RISK_REWARD  = 4.5 / 100
BUYBACK      = 1.5 / 100
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
EQUITY_FRAC  = 0.50

# -- TEST SETTINGS ------------------------------------------------------------
COST_BPS_LEG = 5.0
CAPACITY_USD = 25_000_000
BARS_PER_DAY = 26                    # 6.5h / 15min
BARS_PER_YEAR= BARS_PER_DAY * 252

SYMBOL, BENCH = "SOXL", "QQQ"
START, END    = "2016-01-01", "2026-08-22"
OUTDIR = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
CACHE  = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data/soxl_15m_rth.parquet"
os.makedirs(OUTDIR, exist_ok=True)


def creds():
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if k and s:
        return k, s
    e = json.load(open(os.path.expanduser("~/.claude.json")))["mcpServers"]["alpaca-paper"]["env"]
    return e["ALPACA_API_KEY"], e["ALPACA_SECRET_KEY"]


def load_bars():
    if os.path.exists(CACHE):
        p = pd.read_parquet(CACHE)
    else:
        c = StockHistoricalDataClient(*creds())
        fr = []
        for yr in range(int(START[:4]), int(END[:4]) + 1):
            print(f"  {yr}...", end="", flush=True)
            req = StockBarsRequest(symbol_or_symbols=[SYMBOL, BENCH],
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=f"{yr}-01-01", end=min(f"{yr+1}-01-01", END),
                adjustment=Adjustment.ALL, feed=DataFeed.SIP)
            df = c.get_stock_bars(req).df.reset_index()
            fr.append(df); print(f" {len(df):,}")
        p = pd.concat(fr, ignore_index=True)
        ts = pd.to_datetime(p["timestamp"], utc=True).dt.tz_convert("America/New_York")
        p["et"] = ts
        n0 = len(p)
        p = p[(ts.dt.time >= pd.Timestamp("09:30").time())
              & (ts.dt.time < pd.Timestamp("16:00").time())
              & (ts.dt.dayofweek < 5)].copy()
        print(f"  RTH filter: {n0:,} -> {len(p):,}")
        p.to_parquet(CACHE, index=False)
    p["et"] = pd.to_datetime(p["et"], utc=True).dt.tz_convert("America/New_York")
    p["bar"] = p["et"].dt.tz_localize(None)
    out = {c_: p.pivot_table(index="bar", columns="symbol", values=c_).sort_index().ffill()
           for c_ in ("open", "high", "low", "close")}
    idx = out["close"].dropna().index
    return {k: v.reindex(idx) for k, v in out.items()}


def supertrend(high, low, close, period, factor):
    """ta.supertrend equivalent. dir = -1 uptrend, +1 downtrend."""
    atr = talib.ATR(high, low, close, timeperiod=period)
    hl2 = (high + low) / 2.0
    ub, lb = hl2 + factor * atr, hl2 - factor * atr
    n = len(close)
    st = np.full(n, np.nan); d = np.full(n, np.nan)
    fu = np.full(n, np.nan); fl = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if np.isnan(fu[i-1]) if i > 0 else True:
            fu[i], fl[i], d[i], st[i] = ub[i], lb[i], 1.0, ub[i]
            continue
        fu[i] = ub[i] if (ub[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = lb[i] if (lb[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]
        if d[i-1] == 1.0:
            d[i] = -1.0 if close[i] > fu[i] else 1.0
        else:
            d[i] = 1.0 if close[i] < fl[i] else -1.0
        st[i] = fl[i] if d[i] == -1.0 else fu[i]
    return st, d


def backtest(f, mode="Always Fixed", equity_frac=EQUITY_FRAC,
             vol_target=None, dd_breaker=None, cost_bps=COST_BPS_LEG,
             capacity=CAPACITY_USD, atr_period=ATR_PERIOD,
             touch_fill=False, sym=None):
    """touch_fill=True reproduces TradingView's assumption that a limit fills
    whenever price merely TOUCHES the level (<= / >=) rather than piercing it."""
    global SYMBOL
    if sym:
        SYMBOL = sym
    o = f["open"][SYMBOL].to_numpy(float); h = f["high"][SYMBOL].to_numpy(float)
    l = f["low"][SYMBOL].to_numpy(float);  c = f["close"][SYMBOL].to_numpy(float)
    idx = f["close"].index

    st, d = supertrend(h, l, c, atr_period, FACTOR)
    atrp  = talib.ATR(h, l, c, timeperiod=VOL_LENGTH) / c
    base  = talib.SMA(atrp, timeperiod=VOL_SMA_LEN)
    adx   = talib.ADX(h, l, c, timeperiod=ADX_LENGTH)
    bar_r = pd.Series(c).pct_change().fillna(0.0).to_numpy()
    rvol  = pd.Series(bar_r).rolling(BARS_PER_DAY * 5).std().to_numpy() * np.sqrt(BARS_PER_YEAR)

    eq = 100_000.0
    peak = eq
    pos, qty, entry, target = 0, 0.0, np.nan, np.nan
    fixed = True
    mx = mn = np.nan
    curve, ntrade, nfill_limit, nfill_mkt = [], 0, 0, 0

    def size_at(px, i):
        frac = equity_frac
        if vol_target is not None and not np.isnan(rvol[i]) and rvol[i] > 0:
            frac = min(frac, frac * (vol_target / rvol[i]))
        if dd_breaker is not None and (eq / peak - 1.0) < dd_breaker:
            frac *= 0.5
        notional = min(eq * frac, capacity)
        return math.floor(notional / px) if px > 0 else 0

    for i in range(1, len(c)):
        if pos != 0:
            eq += qty * (c[i] - c[i-1]) * pos
        peak = max(peak, eq)
        if eq <= 0:
            curve.append((idx[i], 0.0)); continue
        if np.isnan(d[i]) or np.isnan(st[i]):
            curve.append((idx[i], eq)); continue

        hv = (not np.isnan(base[i])) and atrp[i] > base[i] * VOL_THRESH
        is_fixed = (mode == "Always Fixed") or (mode == "Auto-Switch via Vol" and hv)
        want = 1 if d[i] == -1 else -1

        # ---- EXITS (trailing stop + limit take-profit) ----------------------
        if pos != 0 and want == pos:
            adxv = ADX_BASE if np.isnan(adx[i]) else adx[i]
            tf = max(0.05, min(0.95, TRAIL_FACTOR + ((ADX_BASE - adxv) / 100))) \
                 if USE_DYNAMIC else TRAIL_FACTOR
            hit = False
            # NOTE ON SEQUENCING: the trailing level is computed from the
            # running extreme AS OF THE PRIOR BAR, then tested against this
            # bar's range. Updating the extreme with this bar's high and then
            # testing this bar's low against the resulting level exits on any
            # wide-range bar -- which on a 3x ETF at 15min is most of them, and
            # truncates every winner. That was the defect.
            if pos > 0:
                tp = entry * (1 + PRICE_PERC) if fixed else \
                     entry * (1 + (entry - st[i]) / entry * LIMIT_FACTOR)
                if h[i] >= tp:
                    px = max(o[i], tp); hit = True
                elif (not np.isnan(mx)) and mx >= target:
                    tr = entry + (mx - entry) * tf
                    if l[i] <= tr:
                        px = min(o[i], tr); hit = True
                if not hit:
                    mx = h[i] if np.isnan(mx) else max(mx, h[i])
            else:
                tp = entry * (1 - PRICE_PERC) if fixed else \
                     entry * (1 - (st[i] - entry) / entry * LIMIT_FACTOR)
                if l[i] <= tp:
                    px = min(o[i], tp); hit = True
                elif (not np.isnan(mn)) and mn <= target:
                    tr = entry - (entry - mn) * tf
                    if h[i] >= tr:
                        px = max(o[i], tr); hit = True
                if not hit:
                    mn = l[i] if np.isnan(mn) else min(mn, l[i])
            if hit:
                eq += qty * (px - c[i]) * pos
                eq -= abs(qty) * px * cost_bps / 1e4
                pos, qty, mx, mn = 0, 0.0, np.nan, np.nan
                ntrade += 1

        # ---- REVERSAL: market order at next open ----------------------------
        if pos != 0 and want != pos and i + 1 < len(c):
            px = o[i+1]
            eq += qty * (px - c[i]) * pos
            eq -= abs(qty) * px * cost_bps / 1e4
            pos, qty, mx, mn = 0, 0.0, np.nan, np.nan
            ntrade += 1
            q = size_at(px, i)
            if q > 0:
                eq -= q * px * cost_bps / 1e4
                pos, qty, entry, fixed = want, q, px, is_fixed
                risk = abs(entry - st[i]) / entry * RISK_FACTOR
                target = entry * (1 + (RISK_REWARD if fixed else risk)) if pos > 0 \
                         else entry * (1 - (RISK_REWARD if fixed else risk))
                nfill_mkt += 1

        # ---- RE-ENTRY: resting limit on a pullback (the piece I missed) -----
        elif pos == 0:
            lim = st[i] * (1 + BUYBACK) if want > 0 else st[i] * (1 - BUYBACK)
            if touch_fill:
                filled = (l[i] <= lim) if want > 0 else (h[i] >= lim)
            else:
                filled = (l[i] < lim) if want > 0 else (h[i] > lim)
            if filled:
                px = min(o[i], lim) if want > 0 else max(o[i], lim)
                q = size_at(px, i)
                if q > 0:
                    eq -= q * px * cost_bps / 1e4
                    pos, qty, entry, fixed = want, q, px, is_fixed
                    risk = abs(entry - st[i]) / entry * RISK_FACTOR
                    target = entry * (1 + (RISK_REWARD if fixed else risk)) if pos > 0 \
                             else entry * (1 - (RISK_REWARD if fixed else risk))
                    mx = mn = np.nan
                    nfill_limit += 1

        curve.append((idx[i], eq))

    s = pd.DataFrame(curve, columns=["bar", "equity"]).set_index("bar")["equity"]
    return s, {"trades": ntrade, "limit_fills": nfill_limit, "mkt_fills": nfill_mkt}


def stats(eq, label, extra=None):
    eq = eq / eq.iloc[0]
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    yrs = len(eq) / BARS_PER_YEAR
    vol = r.std() * np.sqrt(BARS_PER_YEAR)
    fin = eq.iloc[-1]
    m = {"label": label,
         "cagr": (fin ** (1/yrs) - 1) if fin > 0 else -1.0,
         "vol": vol, "sharpe": (r.mean()*BARS_PER_YEAR)/vol if vol > 0 else np.nan,
         "dd": (eq/eq.cummax()-1).min(), "final": fin}
    if extra:
        m.update(extra)
    return m


def main():
    print("=" * 100)
    print("SUPERTREND SOXL — 15-min bars, faithful port + improvements")
    print("=" * 100)
    f = load_bars()
    idx = f["close"].index
    print(f"Bars: {len(idx):,} @15min RTH | {idx[0].date()} -> {idx[-1].date()}")
    print(f"Costs {COST_BPS_LEG}bp/leg | capacity ${CAPACITY_USD:,} | "
          f"conservative limit fills\n")

    rows = [stats(f["close"][SYMBOL], "SOXL buy & hold  [BENCHMARK]"),
            stats(f["close"][BENCH],  "QQQ buy & hold   [reference]")]

    print("--- faithful port, three Pine modes ---")
    base_curves = {}
    for mode in ("Always Fixed", "Auto-Switch via Vol", "Always Dynamic"):
        c, info = backtest(f, mode=mode)
        base_curves[mode] = c
        rows.append(stats(c, f"Port [{mode}]", info))
        print(f"  {mode:<22} trades={info['trades']:5d}  "
              f"limit fills={info['limit_fills']:5d}  mkt={info['mkt_fills']:4d}")

    print("\n--- improvement variants (on best Pine mode) ---")
    improvements = [
        ("I1 vol-targeted sizing",      dict(vol_target=0.60)),
        ("I2 DD circuit-breaker",       dict(dd_breaker=-0.20)),
        ("I3 vol-target + DD breaker",  dict(vol_target=0.60, dd_breaker=-0.20)),
        ("I4 half sizing (25% equity)", dict(equity_frac=0.25)),
        ("I5 I3 + half sizing",         dict(vol_target=0.60, dd_breaker=-0.20,
                                             equity_frac=0.25)),
    ]
    imp_curves = {}
    for label, kw in improvements:
        c, info = backtest(f, mode="Always Fixed", **kw)
        imp_curves[label] = c
        rows.append(stats(c, label, info))
        print(f"  {label:<30} trades={info['trades']:5d}")

    print("\n" + "=" * 100)
    print(f"{'':<36}{'CAGR':>10}{'Vol':>9}{'Sharpe':>8}{'MaxDD':>9}{'x money':>11}")
    print("=" * 100)
    for m in rows:
        fin = f"{m['final']:10,.1f}x" if m['final'] < 1e6 else f"{m['final']:10.2e}x"
        print(f"  {m['label']:<34}{m['cagr']*100:9.2f}%{m['vol']*100:8.2f}%"
              f"{m['sharpe']:8.2f}{m['dd']*100:8.2f}%{fin}")

    bh = rows[0]
    print("\n" + "=" * 100)
    print("VERDICT — vs SOXL buy & hold (the correct benchmark)")
    print("=" * 100)
    print(f"  Bar: CAGR {bh['cagr']*100:.2f}%  Sharpe {bh['sharpe']:.2f}  "
          f"MaxDD {bh['dd']*100:.2f}%\n")
    for m in rows[2:]:
        b1 = m["cagr"] > bh["cagr"]; b2 = m["sharpe"] > bh["sharpe"]
        b3 = m["dd"] > bh["dd"]
        tag = "BEATS on CAGR+Sharpe" if (b1 and b2) else \
              ("better Sharpe only" if b2 else
               ("better CAGR only" if b1 else "loses"))
        print(f"  {m['label']:<36} {tag:<22} "
              f"(DD {'better' if b3 else 'worse'})")

    out = pd.DataFrame({"SOXL_bh": f["close"][SYMBOL] / f["close"][SYMBOL].iloc[0]})
    for k, v in base_curves.items():
        out[f"port_{k}"] = v / v.iloc[0]
    for k, v in imp_curves.items():
        out[k] = v / v.iloc[0]
    out.to_csv(os.path.join(OUTDIR, "supertrend_15m_equity.csv"))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    bhc = f["close"][SYMBOL] / f["close"][SYMBOL].iloc[0]
    a1.plot(bhc.index, bhc, lw=2.4, color="black", label="SOXL buy & hold")
    for k, v in {**base_curves, **imp_curves}.items():
        v = v / v.iloc[0]
        a1.plot(v.index, v.clip(lower=1e-4), lw=1.1, label=str(k))
    a1.set_yscale("log"); a1.set_ylabel("Growth of $1 (log)")
    a1.set_title(f"Supertrend SOXL @15min ({idx[0].date()} -> {idx[-1].date()})")
    a1.legend(loc="upper left", fontsize=7); a1.grid(alpha=0.3)
    for k, v in {**base_curves, **imp_curves}.items():
        v = (v / v.iloc[0]).clip(lower=1e-4)
        a2.plot(v.index, (v/v.cummax()-1)*100, lw=0.9)
    a2.plot(bhc.index, (bhc/bhc.cummax()-1)*100, lw=2.0, color="black")
    a2.set_ylabel("Drawdown %"); a2.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUTDIR, "supertrend_15m.png")
    plt.savefig(png, dpi=120); plt.close()
    print(f"\nSaved: {png}")


if __name__ == "__main__":
    main()
