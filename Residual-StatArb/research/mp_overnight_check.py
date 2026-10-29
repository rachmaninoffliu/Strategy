"""
Valid's checks on the overnight-momentum leaders (mp_overnight.py):
  * the exit price: daily-bar open (official open) vs the first trade of the 9:30 bar vs the 9:45 price (a market order after the open, 3 bp half spread);
  * the entry price: daily close (official close) vs the 15:45-16:00 bar's last trade;
  * costs 1 / 2 / 3 bp per auction side;
  * a 15:30 decision (the free delayed SIP feed) instead of 15:45;
  * year by year vs SPY.
"""
import numpy as np
import pandas as pd

from mp_lib import Book, Ctx, overnight, topk

cx = Ctx()
bk = Book("mp_overnight_check", cx)
C = cx.C
prevC = cx.lag(C)
sma200 = pd.DataFrame(C).rolling(200, min_periods=150).mean().to_numpy()
spy_tr = (cx.lag(C[:, cx.spy]) > cx.lag(sma200[:, cx.spy]))[:, None]
cl, op, vo, nb = cx.p.cube()

# how often / how far the daily open and close differ from the bar prints, for the names the strategy trades
U = cx.top(150) | cx.etf()
last_bar = np.full(C.shape, np.nan)
for t in range(cx.T):
    last_bar[t] = cl[t, :, nb[t] - 1]
with np.errstate(invalid="ignore", divide="ignore"):
    do = (cx.first() / cx.O - 1) * 1e4
    dc = (last_bar / C - 1) * 1e4
m = U & np.isfinite(do)
print(f"first trade vs daily open: mean {np.nanmean(do[m]):+.2f} bp, median |.| {np.nanmedian(np.abs(do[m])):.2f}, p90 |.| {np.nanpercentile(np.abs(do[m]), 90):.2f}")
m = U & np.isfinite(dc)
print(f"last bar trade vs daily close: mean {np.nanmean(dc[m]):+.2f} bp, median |.| {np.nanmedian(np.abs(dc[m])):.2f}, p90 |.| {np.nanpercentile(np.abs(dc[m]), 90):.2f}")


def sig(lead):
    S = cx.snap(lead)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = S / prevC - 1
        res = d - np.nan_to_num(cx.beta(60), nan=1.0) * d[:, [cx.spy]]
        cv = cx.cumvol(lead)
        vb = pd.DataFrame(cv).rolling(20, min_periods=10).mean().shift(1).to_numpy()
        vs = np.where(d > 0, cv / vb, np.nan)
    return np.isfinite(S) & np.isfinite(prevC), {"d45": d, "res45": res, "vsurp": vs}


live45, s45 = sig(1)
live30, s30 = sig(2)
UN = {"top50": cx.top(50), "top150": cx.top(150), "top150+etf": cx.top(150) | cx.etf()}
cands = [("top150+etf", "d45", 20), ("top150+etf", "res45", 20), ("top150+etf", "d45", 10), ("top150", "vsurp", 20), ("top150+etf", "vsurp", 20), ("top50", "d45", 5),
         ("top50", "res45", 3), ("top150", "res45", 10)]
bk.add(cx.spy_ret(), "SPY buy & hold")
years = {}
for un, sn, k in cands:
    base = f"{un} {sn} top{k} spy_trend"
    sel = topk(s45[sn], UN[un] & live45 & spy_tr, k)
    r = bk.add(overnight(cx, sel, k=k), f"{base} | open, 1bp", un=un, sig=sn, k=k, chk="base")
    years[base] = bk.rets[f"{base} | open, 1bp"]
    bk.add(overnight(cx, sel, k=k, exit="first"), f"{base} | first trade, 1bp", chk="first")
    bk.add(overnight(cx, sel, k=k, exit="945", cost_out=3.0), f"{base} | 9:45 + 3bp spread", chk="945")
    for c in (2.0, 3.0):
        bk.add(overnight(cx, sel, k=k, cost_in=c, cost_out=c), f"{base} | open, {c:.0f}bp", chk=f"cost{c:.0f}")
    sel30 = topk(s30[sn], UN[un] & live30 & spy_tr, k)
    bk.add(overnight(cx, sel30, k=k), f"{base} | 15:30 decision", chk="1530")
bk.save()
with pd.option_context("display.width", 260, "display.max_rows", 400, "display.max_colwidth", 90):
    df = bk.frame()
    print(df[["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_oos", "vol_all", "mdd_all"]].round(3).to_string(index=False))
    y = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in years.items()})
    y["SPY"] = (1 + bk.bench).groupby(bk.bench.index.year).prod() - 1
    print((y * 100).round(1).T.to_string())
