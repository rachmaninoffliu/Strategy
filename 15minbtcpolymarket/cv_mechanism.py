"""What changed in BTC itself?  Spot-only mechanism test over the whole history (no Kalshi quotes needed).

For every 15-minute window and decision minute m (12 and 13), z = ln(S_m/K)/(sigma*sqrt(15-m)) (distance from the strike in vol units) and
u = ln(F/S_m)/(sigma*sqrt(15-m))  (what happened afterwards, in the same units).  A random walk with the right vol has u ~ (0,1) and no relation to z:
  * reversal coefficient  beta = slope of u on z  (beta < 0: price drifts back toward the strike -> underdogs flip more than priced)
  * variance ratio        Var(u)                  (> 1: moves in the last minutes are bigger than trailing vol says)
  * dog excess flip rate  win_dog - g_dog         (the thing the strategy monetises, from the pricing function fitted on pre-regime Kalshi quotes)
Reported by month for the 22-23 UTC window vs the other hours, plus a rolling 21-day detector and how well it would have predicted the next 21 days.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import cv_longspot as lsp
from trader.data import load_rows

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)


def fit_g(w):
    q = load_rows()
    q = q[q.minute.isin([12, 13])]
    q = q[(q.bid > 0) & (q.ask < 1) & (q.ask >= q.bid) & ((q.ask - q.bid) <= 0.06 + 1e-9)].copy()
    q["mid"] = (q.bid + q.ask) / 2
    m = q.merge(w, on=["open_ts", "minute"])
    g = {}
    for mn in (12, 13):
        tr = m[(m.minute == mn) & (m.open_ts < lsp.CUT)]
        g[mn] = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(tr.z.values, tr.mid.values)
    return g


def slope(z, u):
    z, u = np.asarray(z), np.asarray(u)
    if len(z) < 30:
        return np.nan, np.nan
    b = (z * u).sum() / (z * z).sum()
    r = u - b * z
    se = np.sqrt((r ** 2).sum() / (len(z) - 1) / (z * z).sum())
    return b, se


def main():
    s = lsp.load_spot_all()
    w = lsp.build_windows(s, minutes=(12, 13))
    g = fit_g(w)
    w["g"] = 0.5
    for mn in g:
        i = w.minute == mn
        w.loc[i, "g"] = g[mn].predict(w.loc[i, "z"].values)
    w["u"] = np.log(w.F / w.S) / (w.sig * np.sqrt(15 - w.minute))
    dog_yes = w.g < 0.5
    w["g_dog"] = np.where(dog_yes, w.g, 1 - w.g)
    w["win_dog"] = np.where(dog_yes, w.yes_proxy, 1 - w.yes_proxy)
    w["exc"] = w.win_dog - w.g_dog
    dt = pd.to_datetime(w.open_ts, unit="s", utc=True)
    w["ym"] = dt.dt.strftime("%Y-%m")
    w["grp"] = np.where(w.hour.isin([22, 23]), "22-23", np.where(w.hour.isin([20, 21]), "20-21", "other"))
    w = w[w.minute == 12]                                   # one observation per window
    print(f"windows (minute 12): {len(w):,}  {dt.min():%Y-%m-%d} -> {dt.max():%Y-%m-%d}")

    rows = []
    for (ym, grp), x in w.groupby(["ym", "grp"]):
        b, se = slope(x.z, x.u)
        band = x[(x.g_dog >= 0.12) & (x.g_dog <= 0.38)]
        rows.append(dict(ym=ym, grp=grp, n=len(x), beta=b, beta_t=b / se if se == se else np.nan, var_u=x.u.var(), dog_exc_c=band.exc.mean() * 100 if len(band) else np.nan, n_dog=len(band)))
    R = pd.DataFrame(rows)
    for col, lab in (("beta", "reversal coefficient beta (negative = reversion)"), ("var_u", "Var(u) (1.0 = vol well calibrated)"), ("dog_exc_c", "dog excess flip rate vs pricing (cents; priced 12-38c)")):
        t = R.pivot(index="ym", columns="grp", values=col)
        print(f"\n=== {lab}, by month ===")
        print(t.round(3 if col != "dog_exc_c" else 1).T.to_string())

    print("\n=== Pooled by regime period (minute 12; z -> u regression and dog excess) ===")
    cuts = [("2024-01 .. 2025-12-14", 0, pd.Timestamp("2025-12-15", tz="UTC").timestamp()),
            ("2025-12-15 .. 2026-08-23", pd.Timestamp("2025-12-15", tz="UTC").timestamp(), lsp.CUT),
            ("2026-08-24 .. now", lsp.CUT, 1e12)]
    out = []
    for lab, a, b_ in cuts:
        for grp in ("22-23", "20-21", "other"):
            x = w[(w.open_ts >= a) & (w.open_ts < b_) & (w.grp == grp)]
            b, se = slope(x.z, x.u)
            band = x[(x.g_dog >= 0.12) & (x.g_dog <= 0.38)]
            out.append(dict(period=lab, hours=grp, n=len(x), beta=b, beta_t=b / se, var_u=x.u.var(), dog_n=len(band), dog_win=band.win_dog.mean(), dog_priced=band.g_dog.mean(), excess_c=band.exc.mean() * 100,
                            excess_t=band.exc.mean() / band.exc.std() * np.sqrt(len(band)) if len(band) > 3 else np.nan))
    print(pd.DataFrame(out).round(3).to_string(index=False))

    # rolling detector: trailing 21-day reversal t-stat in 22-23 -> next 21-day dog excess
    print("\n=== Detector: trailing 21-day beta t-stat (22-23 UTC) vs the dog excess in the NEXT 21 days ===")
    x = w[w.grp == "22-23"].copy()
    x["day"] = (x.open_ts // 86400).astype(int)
    d0, d1 = x.day.min(), x.day.max()
    days = np.arange(d0 + 21, d1 - 21, 3)
    rec = []
    for d in days:
        past = x[(x.day >= d - 21) & (x.day < d)]
        fut = x[(x.day >= d) & (x.day < d + 21)]
        fb = fut[(fut.g_dog >= 0.12) & (fut.g_dog <= 0.38)]
        b, se = slope(past.z, past.u)
        pb = past[(past.g_dog >= 0.12) & (past.g_dog <= 0.38)]
        if len(fb) >= 30 and len(pb) >= 30:
            rec.append((d, b / se, pb.exc.mean() * 100, fb.exc.mean() * 100))
    D = pd.DataFrame(rec, columns=["day", "past_beta_t", "past_exc_c", "next_exc_c"])
    D["date"] = pd.to_datetime(D.day * 86400, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    print(f"  evaluation points: {len(D)} (every 3 days)")
    print(f"  corr(past beta t-stat, next-21d excess) = {np.corrcoef(D.past_beta_t, D.next_exc_c)[0, 1]:+.2f};  corr(past excess, next excess) = {np.corrcoef(D.past_exc_c, D.next_exc_c)[0, 1]:+.2f}")
    for lab, mask in (("past beta_t <= -2", D.past_beta_t <= -2), ("past beta_t in (-2,0]", (D.past_beta_t > -2) & (D.past_beta_t <= 0)), ("past beta_t > 0", D.past_beta_t > 0),
                      ("past excess >= +10c", D.past_exc_c >= 10), ("past excess in (0,10)", (D.past_exc_c > 0) & (D.past_exc_c < 10)), ("past excess <= 0", D.past_exc_c <= 0)):
        sub = D[mask]
        if len(sub):
            print(f"  {lab:24s}: {len(sub):3d} points, next-21d dog excess mean {sub.next_exc_c.mean():+6.1f}c (min {sub.next_exc_c.min():+.1f}, max {sub.next_exc_c.max():+.1f})")
    print("\n  latest points:")
    print(D.tail(10)[["date", "past_beta_t", "past_exc_c", "next_exc_c"]].round(2).to_string(index=False))

    # persistence of 'hot' states: after a 21-day stretch with excess >= +10c, how long did it last? (auto-correlation of the excess series)
    ac = []
    for lag in (1, 2, 3, 4):
        ac.append(D.next_exc_c.autocorr(lag=lag * 7))
    print("\n  autocorrelation of the 21-day excess series at lags 21/42/63/84 days (sampled every 3 days):", np.round(ac, 2))


if __name__ == "__main__":
    main()
