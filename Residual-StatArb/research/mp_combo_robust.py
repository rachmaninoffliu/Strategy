"""
QR's neighbourhood check on the time-shared book (not a selection step): vary one setting at a time around the a-priori model
(overnight d45 top-20 of top-150 + ETFs with SPY > 200-day mean; QQQ noise area every 30 min, lookback 14, band x1) and report the spread of outcomes.
"""
import numpy as np
import pandas as pd

from mp_lib import Book, Ctx, overnight, topk
from mp_overnight_ref import signals
from mp_sleeves import noise_area

cx = Ctx()
bk = Book("mp_combo_robust", cx)
live, S = signals(cx, 1)
C = cx.C


def trend(n):
    s = pd.Series(C[:, cx.spy]).rolling(n, min_periods=int(n * 0.75)).mean().to_numpy()
    return (cx.lag(C[:, cx.spy]) > cx.lag(s))[:, None]


def run(k=20, uni="top150+etf", filt=200, every=2, lb=14, mult=1.0, tk="QQQ", tag=""):
    um = {"top150+etf": cx.top(150) | cx.etf(), "top100+etf": cx.top(100) | cx.etf(), "top50+etf": cx.top(50) | cx.etf(), "top150": cx.top(150)}[uni]
    g = trend(filt) if filt else True
    ro = overnight(cx, topk(S["d45"], um & live & g, k), k=k)
    ri = noise_area(cx, tk, every=every, lookback=lb, mult=mult)
    bk.add((1 + ro) * (1 + ri) - 1, tag or f"k{k} {uni} sma{filt} every{every} lb{lb} x{mult} {tk}")


run(tag="BASE")
for k in (10, 15, 25, 30, 40):
    run(k=k, tag=f"k={k}")
for u in ("top100+etf", "top50+etf", "top150"):
    run(uni=u, tag=f"universe={u}")
for f in (0, 100, 150, 250):
    run(filt=f, tag=f"SPY filter SMA{f}" if f else "no SPY filter")
for lb in (7, 10, 20, 30):
    run(lb=lb, tag=f"noise lookback {lb}")
for mult in (0.75, 1.25, 1.5):
    run(mult=mult, tag=f"band x{mult}")
run(every=1, tag="checks every 15 min")
run(tk="SPY", tag="intraday on SPY")
df = bk.save()
with pd.option_context("display.width", 260, "display.max_rows", 100):
    print(df[["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_oos", "vol_all", "mdd_all"]].round(3).to_string(index=False))
    print(df[["sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_oos", "mdd_all"]].describe().round(3).to_string())
