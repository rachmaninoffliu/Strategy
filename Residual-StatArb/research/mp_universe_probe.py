"""
QR's quick probe (2026-10-28): does a wider universe (top-500 by liquidity, point-in-time, as an S&P 500 proxy) look better for the overnight leg?
Only the top-150's names have 15-minute bars, so the signal here is a PROXY: today's move to the CLOSE (look-ahead - the live book decides at 15:30). It is calibrated
on the top-150, where the honest 15:45 signal exists. Rule only, top-20, SPY > 200-day mean, 1 bp per auction side; overnight leg alone (no QQQ).
"""
import numpy as np
import pandas as pd

from mp_lib import Ctx, overnight, topk
from mp_overnight_ref import signals, spy_trend

cx = Ctx()
live, S = signals(cx, 1)
tr = spy_trend(cx)
with np.errstate(invalid="ignore", divide="ignore"):
    proxy = cx.rc                                                                              # close-to-close move: look-ahead proxy
okp = np.isfinite(proxy) & np.isfinite(cx.lag(cx.C)) & (cx.lag(cx.C) >= 5)
t150, t500, etf = cx.top(150), cx.top(500), cx.etf()
spy = cx.spy_ret()


def stats(r):
    r = r.dropna()
    a, b = r[r.index < "2023-01-01"], r[r.index >= "2023-01-01"]
    sh = lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    eq = (1 + r).cumprod()
    return dict(sh=sh(r), is_=sh(a), oos=sh(b), cagr=eq.iloc[-1] ** (252 / len(r)) - 1, mdd=(eq / eq.cummax() - 1).min(), night_bp=r[r != 0].mean() * 1e4)


def run(label, score, mask, k=20, cost=1.0):
    sel = topk(score, mask & tr, k)
    r = overnight(cx, sel, k=k, cost_in=cost, cost_out=cost)
    picks = sel[cx.t0:].sum()
    adv = np.nanmedian(np.where(sel, cx.lag(pd.DataFrame(cx.C * cx.V).rolling(20, min_periods=10).mean().to_numpy()), np.nan)[cx.t0:])
    return dict(label=label, **stats(r), median_adv_musd=adv / 1e6, picks=int(picks))


rows = [
    run("top150+etf, honest 15:45 signal", S["d45"], live & (t150 | etf)),
    run("top150+etf, close proxy", proxy, okp & (t150 | etf)),
    run("top500+etf, close proxy", proxy, okp & (t500 | etf)),
    run("ranks 151-500 only, close proxy", proxy, okp & t500 & ~t150),
    run("top500+etf, close proxy, 3 bp", proxy, okp & (t500 | etf), cost=3.0),
    run("ranks 151-500 only, close proxy, 3 bp", proxy, okp & t500 & ~t150, cost=3.0),
    run("top150+etf, close proxy, k=10", proxy, okp & (t150 | etf), k=10),
    run("top500+etf, close proxy, k=10", proxy, okp & (t500 | etf), k=10),
]
df = pd.DataFrame(rows)
with pd.option_context("display.width", 250):
    print(df.round({"sh": 2, "is_": 2, "oos": 2, "cagr": 3, "mdd": 3, "night_bp": 1, "median_adv_musd": 0}).to_string(index=False))
    print(f"SPY: {stats(spy)}")
