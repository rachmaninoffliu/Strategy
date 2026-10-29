"""
Slow momentum families (long-only, unlevered; decide on the previous session's close, fill in today's closing auction, 1 bp per side):

M1  cross-sectional stock momentum (Jegadeesh-Titman 1993; George-Hwang 2004 52-week high; Blitz-Huij-Martens 2011 residual momentum; Da-Gurun-Warachka 2014 frog in the pan;
    Clenow 2015 regression momentum): top-k of top-100 / 150 / 500, monthly or weekly, optional market / stock trend filter.
M2  ETF rotation / tactical asset allocation (Faber 2007 relative strength; Antonacci 2014 dual momentum; Keller-Keuning 2017 VAA / 2018 DAA canary): top-k of the 59 ETFs
    or a 12-ETF core set, lookbacks 1 / 3 / 6 / 12 months or the 13612W blend, absolute filter, safe asset.
M3  trend timing of SPY / QQQ (moving averages, time-series momentum - Moskowitz-Ooi-Pedersen 2012, volatility targeting capped at 1x - Moreira-Muir 2017).
"""
import numpy as np
import pandas as pd

from mp_lib import Book, Ctx, log, rebalance, topk

cx = Ctx()
bk = Book("mp_slow", cx)
C, T, N = cx.C, cx.T, cx.N
Cl = cx.lag(C)                                                         # yesterday's close: the decision information
dates = cx.dates
mon = dates.to_period("M")
first_of_month = np.r_[True, mon[1:] != mon[:-1]]
wk = dates.to_period("W-FRI")
first_of_week = np.r_[True, wk[1:] != wk[:-1]]
REB = {"monthly": first_of_month, "weekly": first_of_week}
bk.add(cx.spy_ret(), "SPY buy & hold", fam="bench")
qqq = cx.series(np.nan_to_num(cx.rc[:, cx.j("QQQ")]))
bk.add(qqq, "QQQ buy & hold", fam="bench")


def ret_n(n, skip=0):
    return cx.lag(C, 1 + skip) / cx.lag(C, 1 + n) - 1


with np.errstate(invalid="ignore", divide="ignore"):
    rc0 = np.where(np.isfinite(cx.rc), cx.rc, np.nan)
    vol252 = cx.lag(pd.DataFrame(rc0).rolling(252, min_periods=200).std().to_numpy())
    beta = cx.beta(252)
    resid = rc0 - np.nan_to_num(beta, nan=1.0) * rc0[:, [cx.spy]]
    rsum = pd.DataFrame(resid).rolling(231, min_periods=180).sum().to_numpy()             # t-252 .. t-21
    rsd = pd.DataFrame(resid).rolling(231, min_periods=180).std().to_numpy()
    resmom = cx.lag(rsum, 22) / cx.lag(rsd, 22)
    pos = pd.DataFrame(rc0 > 0).rolling(231, min_periods=180).mean().to_numpy()
    neg = pd.DataFrame(rc0 < 0).rolling(231, min_periods=180).mean().to_numpy()
    m121 = ret_n(252, 21)
    ID = np.sign(m121) * (cx.lag(neg, 22) - cx.lag(pos, 22))                                # information discreteness: low = smooth ("frog in the pan")
    hi52 = Cl / cx.lag(pd.DataFrame(C).rolling(252, min_periods=200).max().to_numpy())
SIG ={"mom12_1": m121, "mom6_1": ret_n(126, 21), "mom3_1": ret_n(63, 21), "mom12_0": ret_n(252), "mom1": ret_n(21), "sharpe12_1": m121 / vol252, "hi52": hi52,
       "resmom": resmom, "fip": np.where(m121 > 0, m121 * (ID < np.nanmedian(ID)), np.nan)}


def clenow(n=90):
    """Annualised exponential regression slope x R^2 of log price over n sessions ending yesterday (vectorised via rolling sums)."""
    y = np.log(C)
    ok = np.isfinite(y)
    y0 = np.where(ok, y, 0.0)
    t = np.arange(T, dtype=float)[:, None] * np.ones((1, N))

    def roll(a):
        c = np.cumsum(a, axis=0)
        c[n:] = c[n:] - c[:-n].copy()
        return c
    k = roll(ok.astype(float))
    st, sy, stt, sty, syy = roll(np.where(ok, t, 0)), roll(y0), roll(np.where(ok, t * t, 0)), roll(np.where(ok, t, 0) * y0), roll(y0 * y0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sty - st * sy / k
        vt = stt - st * st / k
        vy = syy - sy * sy / k
        b = cov / vt
        r2 = cov * cov / (vt * vy)
        out = (np.exp(b * 252) - 1) * r2
    out[k < n] = np.nan
    return cx.lag(out)


SIG["clenow90"] = clenow(90)
sma200 = pd.DataFrame(C).rolling(200, min_periods=150).mean().to_numpy()
sma100 = pd.DataFrame(C).rolling(100, min_periods=80).mean().to_numpy()
spy_tr = (Cl[:, cx.spy] > cx.lag(sma200)[:, cx.spy])[:, None]
stock_tr = Cl > cx.lag(sma100)
alive = np.isfinite(Cl)

# ---------------------------------------------------------------- M1
for un in ("top100", "top150", "top500"):
    um = cx.top(int(un[3:])) & alive
    for sn, sv in SIG.items():
        for k in (5, 10, 20, 50):
            for fn, fm in (("none", True), ("spy_trend", spy_tr), ("stock_trend", stock_tr & spy_tr)):
                sel = topk(sv, um & fm, k)
                for rn in ("monthly", "weekly"):
                    w = sel / k
                    if fn == "spy_trend":                                              # out of the market -> the whole book to cash at the next rebalance
                        w = np.where(spy_tr, w, 0.0)
                    bk.add(rebalance(cx, w, REB[rn]), f"M1 {un} {sn} top{k} {fn} {rn}", fam="M1", uni=un, sig=sn, k=k, filt=fn, reb=rn)
    log(f"M1 {un}: {len(bk.rows)} trials")

# ---------------------------------------------------------------- M2
etf = cx.etf()
CORE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "TLT", "IEF", "HYG", "LQD", "DBA"]
core = np.zeros(N, bool)
core[[cx.j(s) for s in CORE]] = True
r1, r3, r6, r12 = ret_n(21), ret_n(63), ret_n(126), ret_n(252)
LB = {"1m": r1, "3m": r3, "6m": r6, "12m": r12, "13612W": 12 * r1 + 4 * r3 + 2 * r6 + r12}
canary = (LB["13612W"][:, cx.j("EEM")] > 0) & (LB["13612W"][:, cx.j("IEF")] > 0)
for setn, sm in (("all59", etf), ("core12", np.broadcast_to(core, (T, N)))):
    m = sm & alive
    for ln, lv in LB.items():
        for k in (1, 2, 3, 5, 10):
            sel = topk(lv, m, k)
            for an in ("none", "abs", "canary"):
                for safe in ("cash", "IEF", "best_bond"):
                    w = sel / k
                    if an == "abs":
                        w = np.where(lv > 0, w, 0.0)
                    elif an == "canary":
                        w = np.where(canary[:, None], w, 0.0)
                    spare = 1 - w.sum(1)
                    if safe == "IEF":
                        w[:, cx.j("IEF")] += spare
                    elif safe == "best_bond":
                        bonds = [cx.j(s) for s in ("IEF", "TLT", "LQD", "GLD")]
                        b = np.array(bonds)[np.nanargmax(np.nan_to_num(lv[:, bonds], nan=-9), axis=1)]
                        w[np.arange(T), b] += spare
                    for rn in ("monthly", "weekly"):
                        bk.add(rebalance(cx, w, REB[rn]), f"M2 {setn} {ln} top{k} {an} safe={safe} {rn}", fam="M2", uni=setn, sig=ln, k=k, filt=an, safe=safe, reb=rn)
    log(f"M2 {setn}: {len(bk.rows)} trials")
# GEM dual momentum
for lb in ("6m", "12m"):
    lv = LB[lb]
    s, e = cx.j("SPY"), cx.j("EFA")
    for safe in ("IEF", "TLT"):
        w = np.zeros((T, N))
        best = np.where(lv[:, s] >= lv[:, e], s, e)
        on = np.nanmax(lv[:, [s, e]], 1) > 0
        w[np.arange(T), np.where(on, best, cx.j(safe))] = 1.0
        bk.add(rebalance(cx, w, REB["monthly"]), f"M2 GEM {lb} safe={safe}", fam="M2", sig=lb, safe=safe)

# ---------------------------------------------------------------- M3
for tk in ("SPY", "QQQ"):
    j = cx.j(tk)
    c = C[:, j]
    cl1 = cx.lag(c)
    daily = np.ones(T, bool)
    for safe in ("cash", "IEF"):
        def put(on, frac=None):
            w = np.zeros((T, N))
            f = on.astype(float) if frac is None else np.where(on, frac, 0.0)
            w[:, j] = f
            if safe == "IEF":
                w[:, cx.j("IEF")] = 1 - f
            return w
        for n in (50, 100, 150, 200, 250):
            sma = cx.lag(pd.Series(c).rolling(n).mean().to_numpy())
            for band in (0.0, 0.01):
                on = np.zeros(T, bool)
                st = False
                for t in range(T):
                    if np.isfinite(sma[t]):
                        st = cl1[t] > sma[t] * (1 + band) if not st else cl1[t] > sma[t] * (1 - band)
                    on[t] = st
                bk.add(rebalance(cx, put(on), daily), f"M3 {tk} > SMA{n} band {band:.0%} safe={safe}", fam="M3", uni=tk, sig=f"sma{n}", safe=safe)
        for lb in ("3m", "6m", "12m"):
            on = LB[lb][:, j] > 0
            bk.add(rebalance(cx, put(on), REB["monthly"]), f"M3 {tk} TSMOM {lb} monthly safe={safe}", fam="M3", uni=tk, sig=f"tsmom{lb}", safe=safe)
        rv = cx.lag(pd.Series(np.nan_to_num(cx.rc[:, j])).rolling(20).std().to_numpy() * np.sqrt(252))
        sma200j = cx.lag(pd.Series(c).rolling(200).mean().to_numpy())
        for tgt in (0.10, 0.15, 0.20):
            frac = np.clip(tgt / rv, 0, 1)
            bk.add(rebalance(cx, put(np.ones(T, bool), frac), daily), f"M3 {tk} vol target {tgt:.0%} (cap 1x) safe={safe}", fam="M3", uni=tk, sig=f"vt{tgt}", safe=safe)
            bk.add(rebalance(cx, put(cl1 > sma200j, frac), daily), f"M3 {tk} vol target {tgt:.0%} + SMA200 safe={safe}", fam="M3", uni=tk, sig=f"vt{tgt}+sma", safe=safe)
log(f"M3: {len(bk.rows)} trials")
df = bk.save()
for fam in ("M1", "M2", "M3"):
    print(f"\n==== {fam}: {int((df.fam == fam).sum())} variants; in-sample Sharpe mean {df[df.fam == fam].sh_is.mean():.2f}, sd {df[df.fam == fam].sh_is.std():.2f}")
    sub = df[df.fam == fam].sort_values("sh_is", ascending=False).head(12)
    with pd.option_context("display.width", 260, "display.max_colwidth", 80):
        print(sub[["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all", "beta"]].round(3).to_string(index=False))
    sub = df[df.fam == fam].sort_values("sh_all", ascending=False).head(5)
    print(sub[["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_oos", "mdd_all"]].round(3).to_string(index=False))
