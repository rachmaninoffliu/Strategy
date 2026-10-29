"""
Fast triage of mean-reversion families on daily data (vectorised, equal-weight K slots, no leverage: the K slots share 100% of capital, empty slots are cash).
Timing is honest: a market-on-close entry is chosen with the 15:45 price (the MOC cut-off is 15:50) and filled at the official close; market-on-open exits fill at the
official open; continuous-market fills pay the half spread. In-sample 2016-09 .. 2022, out-of-sample 2023 .. 2026-10.

  python mr_triage.py
"""
import numpy as np
import pandas as pd

from mr_data import Daily, snapshot_prices, stats, universe

AUCTION_BP = 1.0                     # per side, market-on-open / market-on-close (no spread crossed; fees + impact)
CONT_BP = 2.5                        # per side, a continuous-market fill (half spread + slippage, top-100)
START = "2016-09-01"

d = Daily()
O, H, L, C = d.open, d.high, d.low, d.close
T, N = C.shape
S45 = snapshot_prices(d, 1545)
S945 = snapshot_prices(d, 945)
spy = d.j("SPY")
dates = d.dates
t0 = int(np.searchsorted(dates, pd.Timestamp(START)))
rows = []


def series(ret_by_day, name, **kw):
    s = pd.Series(ret_by_day, index=dates)[t0:]
    st = stats(s, name)
    st.update(kw)
    rows.append(st)
    return s


def pick(sig, mask, k, lowest=True):
    """Boolean (T x N): the k names with the lowest (highest) signal among the masked, finite cells of each row."""
    x = np.where(mask & np.isfinite(sig), sig, np.nan)
    x = x if lowest else -x
    r = pd.DataFrame(x).rank(axis=1, method="first").to_numpy()
    return (r <= k) & np.isfinite(x)


def book(sel, ret, cost_sides_bp, k):
    """Equal weight 1/k per selected name (cash for empty slots); the return lands on the day it is realised (handled by the caller)."""
    n = sel.sum(1)
    gross = np.where(sel & np.isfinite(ret), ret, 0.0).sum(1) / k
    return gross - (n / k) * cost_sides_bp / 1e4


with np.errstate(invalid="ignore", divide="ignore"):
    prevC = np.vstack([np.full((1, N), np.nan), C[:-1]])
    nextO = np.vstack([O[1:], np.full((1, N), np.nan)])
    nextC = np.vstack([C[1:], np.full((1, N), np.nan)])
    r_on = nextO / C - 1                                   # overnight: close t -> open t+1 (realised on day t+1)
    r_1d = nextC / C - 1                                   # close t -> close t+1
    r_id45 = C / S945 - 1                                  # 9:45 -> close (same day)
    day45 = S45 / prevC - 1                                # today's return up to 15:45 (known before the MOC cut-off)
    intra45 = S45 / O - 1                                  # open -> 15:45
    mkt45 = day45[:, [spy]]
    resid45 = day45 - mkt45                                # market-adjusted (beta 1)
    C5 = np.vstack([np.full((5, N), np.nan), C[:-5]])
    five45 = S45 / C5 - 1
    gap945 = S945 / prevC - 1

shift = lambda x: np.r_[0.0, x[:-1]]                       # a return earned from day t's close is realised on day t+1
spy_bh = np.r_[np.nan, C[1:, spy] / C[:-1, spy] - 1]
series(spy_bh, "SPY buy & hold")
for n in (100, 500):
    U = universe(d, n)
    has45 = np.isfinite(S45)
    M = U & has45 if n == 100 else U
    sig_day = day45 if n == 100 else (C / prevC - 1)       # top-500: no 15-minute data -> the close (optimistic by the last 10 minutes; flagged)
    tag = "" if n == 100 else " [close signal]"
    allon = M & np.isfinite(r_on)
    series(shift(np.where(allon, r_on, 0).sum(1) / np.maximum(allon.sum(1), 1) - 2 * AUCTION_BP / 1e4 * (allon.sum(1) > 0)), f"top{n} overnight, all names", fam="overnight", n=n)
    for k in (5, 10, 20, 50):
        for sname, sig in (("day loss", sig_day), ("market-adj loss", (sig_day - sig_day[:, [spy]]) if n == 500 else resid45), ("intraday loss", intra45 if n == 100 else (C / O - 1)),
                           ("5-day loss", five45 if n == 100 else (C / C5 - 1))):
            sel = pick(sig, M, k)
            series(shift(book(sel, r_on, 2 * AUCTION_BP, k)), f"top{n} overnight, {k} worst {sname}{tag}", fam="overnight", n=n, k=k)
        sel_w = pick(sig_day, M, k, lowest=False)
        series(shift(book(sel_w, r_on, 2 * AUCTION_BP, k)), f"top{n} overnight, {k} best day (control){tag}", fam="overnight", n=n, k=k)
        sel_l = pick(sig_day, M, k)
        ls = 0.5 * book(sel_l, r_on, 2 * AUCTION_BP, k) - 0.5 * book(sel_w, r_on, -2 * AUCTION_BP, k)
        series(shift(ls), f"top{n} overnight L/S {k} losers-winners{tag}", fam="overnight LS", n=n, k=k)
        series(shift(book(sel_l, r_1d, 2 * AUCTION_BP, k)), f"top{n} 1-day hold, {k} worst day{tag}", fam="1-day", n=n, k=k)
        sel5 = pick(five45 if n == 100 else (C / C5 - 1), M, k)
        series(shift(book(sel5, r_1d, 2 * AUCTION_BP, k)), f"top{n} 1-day hold, {k} worst 5-day{tag}", fam="1-day", n=n, k=k)
        if n == 100:
            selg = pick(gap945, M & np.isfinite(S945), k)
            series(book(selg, r_id45, AUCTION_BP + CONT_BP, k), f"top100 gap fade 9:45->close, {k} worst gap", fam="intraday", n=n, k=k)
df = pd.DataFrame(rows)
df.to_csv("reports/mr_triage.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
cols = ["name", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all"]
print(df[cols].sort_values("sh_is", ascending=False).round(3).to_string(index=False))
