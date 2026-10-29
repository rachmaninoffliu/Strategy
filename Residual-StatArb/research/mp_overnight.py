"""
M4 - overnight momentum (Lou-Polk-Skouras 2019 "A tug of war"; Berkman-Koch-Tuttle-Zhang 2012 attention; Aboody et al. 2018 overnight persistence).
Long-only, unlevered: at 15:45 rank the names, buy k of them in the closing auction (1/k of equity each), sell them in the next opening auction.

Signals (all known at 15:45):
  d45      today's return to 15:45 (prev close -> 15:45 price)
  res45    d45 minus beta x SPY's d45 (60-session beta)
  hour     the last hour to 15:45 (14:45 -> 15:45)
  vsurp    volume to 15:45 / its 20-session mean, among names up on the day
  on20/60  mean overnight return of the last 20 / 60 sessions (overnight persistence)
  mom      12-1 month momentum (yesterday's close)
  mix      z(d45) + z(on60)
Filters: none | SPY up on the day at 15:45 | SPY above its 200-day mean | the name itself up on the day.
"""
import numpy as np
import pandas as pd

from mp_lib import Book, Ctx, log, overnight, topk

cx = Ctx()
bk = Book("mp_overnight", cx)
C, T, N = cx.C, cx.T, cx.N
S45, S30, S1445 = cx.snap(1), cx.snap(2), cx.snap(5)
prevC = cx.lag(C)
with np.errstate(invalid="ignore", divide="ignore"):
    d45 = S45 / prevC - 1
    beta = cx.beta(60)
    spy45 = d45[:, [cx.spy]]
    res45 = d45 - np.nan_to_num(beta, nan=1.0) * spy45
    hour = S45 / S1445 - 1
    cv = cx.cumvol(1)
    vbase = pd.DataFrame(cv).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    vsurp = np.where(d45 > 0, cv / vbase, np.nan)
    ron_l = np.where(np.isfinite(cx.ron), cx.ron, np.nan)
    on20 = pd.DataFrame(ron_l).rolling(20, min_periods=15).mean().to_numpy()
    on60 = pd.DataFrame(ron_l).rolling(60, min_periods=45).mean().to_numpy()
    mom = cx.lag(C, 21) / cx.lag(C, 252) - 1
    sma200 = pd.DataFrame(C).rolling(200, min_periods=150).mean().to_numpy()


def z(x, mask):
    x = np.where(mask & np.isfinite(x), x, np.nan)
    mu = np.nanmean(x, 1, keepdims=True)
    sd = np.nanstd(x, 1, keepdims=True)
    return (x - mu) / np.where(sd > 0, sd, np.nan)


spy_up = (d45[:, cx.spy] > 0)[:, None]
spy_tr = (cx.lag(C[:, cx.spy]) > cx.lag(sma200[:, cx.spy]))[:, None]
up = d45 > 0
live = np.isfinite(S45) & np.isfinite(prevC)
U = {"top50": cx.top(50), "top100": cx.top(100), "top150": cx.top(150), "etf": cx.etf(), "top150+etf": cx.top(150) | cx.etf()}
bk.add(cx.spy_ret(), "SPY buy & hold", fam="bench")
for un, um in U.items():
    m = um & live
    n_names = np.maximum(m.sum(1), 1)
    bk.add(overnight(cx, m, w=m / n_names[:, None]), f"{un} all names overnight (equal weight)", fam="base", uni=un)
    sigs = {"d45": d45, "res45": res45, "hour": hour, "vsurp": vsurp, "on20": on20, "on60": on60, "mom": mom, "mix": z(d45, m) + z(on60, m)}
    for sn, sv in sigs.items():
        for fn, fm in (("none", True), ("spy_up", spy_up), ("spy_trend", spy_tr), ("name_up", up)):
            mm = m & fm
            for k in (3, 5, 10, 20, 30):
                if un == "etf" and k > 10:
                    continue
                sel = topk(sv, mm, k)
                bk.add(overnight(cx, sel, k=k), f"{un} {sn} top{k} filter={fn}", fam="M4", uni=un, sig=sn, filt=fn, k=k)
    log(f"{un}: {len(bk.rows)} trials")
df = bk.save()
bk.show(n=40)
print()
bk.show(sort="sh_oos", n=15)
