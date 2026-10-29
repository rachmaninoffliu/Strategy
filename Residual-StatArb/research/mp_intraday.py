"""
Intraday sleeves (open -> close, long-only, unlevered). They can share the overnight sleeve's capital, which is idle between the opening and the closing auction.

I1  baseline: hold SPY / QQQ / all top-150 names from the opening auction to the closing auction
I2  market intraday momentum (Gao-Han-Li-Zhou 2018 JFE): the return from yesterday's close to 10:00 (or to 15:30) predicts the rest of the day; long SPY / QQQ when positive
I3  stock intraday momentum: top-k of the top-150 by the move from the open (or from yesterday's close) to 10:00, held to the close
I4  stock intraday reversal: the k worst names at 10:00 (long-only), held to the close
I5  noise-area breakout (Zarattini-Aziz-Barbon 2024, "Beat the market"): long SPY / QQQ when the price leaves the average-move band above the open, out below
    max(band, VWAP) or at the close; checked every 15 / 30 minutes
Costs: opening / closing auction 1 bp; a fill during the session pays the half spread: SPY/QQQ 0.5 bp, other ETFs 1 bp, stocks 3 bp in the first hour, 1.5 bp later.
"""
import numpy as np
import pandas as pd

from mp_lib import Book, Ctx, intraday, log, topk
from resarb import data as D

cx = Ctx()
bk = Book("mp_intraday", cx)
C, O, T, N = cx.C, cx.O, cx.T, cx.N
prevC = cx.lag(C)
cl, op, vo, nb = cx.p.cube()
P1000 = cx.slot(1)                                      # close of the 9:45-10:00 bar
P1530 = cx.snap(2)
U = cx.top(150)
live = np.isfinite(O) & np.isfinite(prevC)
bk.add(cx.spy_ret(), "SPY buy & hold", fam="bench")
half = np.where(cx.etf()[0], 1.0, 3.0)
for tk in ("SPY", "QQQ"):
    j = cx.j(tk)
    w = np.zeros((T, N))
    w[:, j] = 1
    bk.add(intraday(cx, w, O, cost_in=1.0), f"I1 {tk} open auction -> close", fam="I1", uni=tk)
m = U & live
w = m / np.maximum(m.sum(1), 1)[:, None]
bk.add(intraday(cx, w, O, cost_in=1.0), "I1 top150 equal weight open -> close", fam="I1")

# I2 market intraday momentum
for tk in ("SPY", "QQQ"):
    j = cx.j(tk)
    with np.errstate(invalid="ignore", divide="ignore"):
        s10 = P1000[:, j] / prevC[:, j] - 1
        s1530 = P1530[:, j] / prevC[:, j] - 1
        sid = P1530[:, j] / O[:, j] - 1
    for name, sig, entry, c_in in (("close->10:00 > 0, enter 10:00", s10, P1000, 0.5), ("close->15:30 > 0, enter 15:30", s1530, P1530, 0.5),
                                   ("open->15:30 > 0, enter 15:30", sid, P1530, 0.5), ("close->10:00 > 0, enter 15:30", s10, P1530, 0.5)):
        for thr in (0.0, 0.0025, 0.005):
            w = np.zeros((T, N))
            w[:, j] = sig > thr
            bk.add(intraday(cx, w, entry, cost_in=c_in), f"I2 {tk} {name} (thr {thr:.2%})", fam="I2", uni=tk, thr=thr)

# I3 / I4 stock intraday momentum / reversal at 10:00
with np.errstate(invalid="ignore", divide="ignore"):
    fo = P1000 / O - 1
    fc = P1000 / prevC - 1
    gap = O / prevC - 1
m10 = U & live & np.isfinite(P1000)
for sn, sv in (("open->10:00", fo), ("close->10:00", fc), ("gap", gap)):
    for k in (5, 10, 20):
        sel = topk(sv, m10, k)
        bk.add(intraday(cx, sel / k, P1000, cost_in=3.0), f"I3 top150 best {k} by {sn}, 10:00 -> close", fam="I3", sig=sn, k=k)
        sel = topk(sv, m10, k, largest=False)
        bk.add(intraday(cx, sel / k, P1000, cost_in=3.0), f"I4 top150 worst {k} by {sn}, 10:00 -> close", fam="I4", sig=sn, k=k)
log(f"I1-I4: {len(bk.rows)} trials")


# I5 noise area on SPY / QQQ
def noise_area(tk, every=2, lookback=14, trail=True, mult=1.0):
    b = D.load_bars(tk)
    b = b[(b.index.normalize() >= cx.dates[0]) & (b.index.normalize() <= cx.dates[-1])]
    j = cx.j(tk)
    day = b.index.normalize()
    tix = {d: i for i, d in enumerate(cx.dates)}
    ti = np.array([tix.get(x, -1) for x in day])
    slot = ((b.index - day) / pd.Timedelta(minutes=15)).astype(int) - 38
    ok = (ti >= 0) & (slot >= 0) & (slot < 26)
    pv = np.full((T, 26), np.nan)
    vw = np.full((T, 26), np.nan)
    vv = np.full((T, 26), np.nan)
    pv[ti[ok], slot[ok]] = b.close.to_numpy()[ok]
    vw[ti[ok], slot[ok]] = b.vwap.to_numpy()[ok]
    vv[ti[ok], slot[ok]] = b.volume.to_numpy()[ok]
    for t in range(T):
        pv[t, nb[t]:] = np.nan
    cvw = np.nancumsum(vw * vv, 1) / np.nancumsum(vv, 1)                               # session VWAP to the end of each bar
    o = O[:, j]
    pc = prevC[:, j]
    move = np.abs(pv / o[:, None] - 1)
    sig = pd.DataFrame(move).rolling(lookback, min_periods=lookback).mean().shift(1).to_numpy() * mult
    ret = np.zeros(T)
    trades = 0
    for t in range(T):
        if not (np.isfinite(o[t]) and np.isfinite(pc[t]) and np.isfinite(C[t, j])):
            continue
        n = int(nb[t])
        upper = max(o[t], pc[t]) * (1 + sig[t])
        pos, px_in = False, 0.0
        r = 0.0
        for s in range(every - 1, n - 1, every):                                         # decision at the end of bar s; never in the closing bar
            p = pv[t, s]
            if not np.isfinite(p) or not np.isfinite(upper[s]):
                continue
            if not pos and p > upper[s]:
                pos, px_in = True, p
                r -= 0.5e-4
                trades += 1
            elif pos and trail and p < max(upper[s], cvw[t, s]):
                r += p / px_in - 1 - 0.5e-4
                pos = False
        if pos:
            r += C[t, j] / px_in - 1 - 1e-4
        ret[t] = r
    s = cx.series(ret)
    return s, trades


for tk in ("SPY", "QQQ"):
    for every in (1, 2):
        for lb in (14, 30):
            for trail in (True, False):
                for mult in (1.0, 1.5):
                    r, ntr = noise_area(tk, every, lb, trail, mult)
                    bk.add(r, f"I5 {tk} noise area, every {15 * every} min, lookback {lb}, {'trailing' if trail else 'hold to close'}, band x{mult}", fam="I5", uni=tk, trades=ntr)
log(f"I5: {len(bk.rows)} trials")
df = bk.save()
with pd.option_context("display.width", 260, "display.max_rows", 400, "display.max_colwidth", 100):
    print(df.sort_values("sh_is", ascending=False)[["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_oos", "vol_all", "mdd_all", "corr_spy"]].round(3).head(45).to_string(index=False))
