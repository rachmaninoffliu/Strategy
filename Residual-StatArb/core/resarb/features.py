"""
Candidate predictors of the residual return, and the tools to test them (no Qt).

Every feature at bar t uses information up to the CLOSE of bar t only. Each comes from a paper:

  rr_k        residual return over the last k bars (k = 1, 2, 4, 13, 26, 130, 520)       short-term reversal / momentum: Lehmann 1990, Jegadeesh 1990, Da-Liu-Schaumburg 2014
  mom_12_1    residual return over days t-252 .. t-21, divided by its volatility         residual momentum: Blitz-Huij-Martens 2011
  ou_s, ou_kappa, ou_r2   OU s-score, speed and fit of the cumulative residual           Avellaneda-Lee 2010, Yeo-Papanicolaou (fast, well-fitted residuals)
  hks         mean residual of the SAME time-of-day bar over the previous 20 days       Heston-Korajczyk-Sadka 2010 (intraday periodicity)
  on_res, id_today        today's overnight residual / today's residual since the open   Lou-Polk-Skouras 2019 (overnight vs intraday), Stubinger-Schneider (overnight gaps)
  on_5d, id_5d            overnight / intraday residual sums over 5 days                 Lou-Polk-Skouras 2019
  vol_surp    volume vs the same bar's 20-day average                                    Avellaneda-Lee 2010 (volume-adjusted signals), Gervais-Kaniel-Mingelgrin 2001
  ivol, r2_fit            idiosyncratic volatility / R^2 of the factor regression        Ang-Hodrick-Xing-Zhang 2006; fit quality
  range, cloc, vwap_dev   bar range, close location in the bar, distance from the day's VWAP   microstructure (Gu-Kelly-Xiu 2020 list liquidity / volatility signals)
  beta, sec_rel_5d        market beta, the sector ETF's 5-day return vs SPY              betting-against-beta, industry momentum (Moskowitz-Grinblatt 1999)
  mkt_today, mkt_vol20, tod   market state and time of day (same for all stocks; useful only inside a non-linear model)   Gao-Han-Li-Zhou 2018 (market intraday momentum)

Target: fwd(h, lag) = residual open-to-open return from the open of bar t+1+lag over h bars (what an order sent after bar t's data arrive would earn).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import data as D
from .factors import Residuals

FEATURES = ["rr_1", "rr_2", "rr_4", "rr_13", "rr_26", "rr_130", "rr_520", "mom_12_1", "ou_s", "ou_kappa", "ou_r2", "hks", "on_res", "id_today", "on_5d", "id_5d", "vol_surp", "ivol",
            "r2_fit", "range", "cloc", "vwap_dev", "beta", "sec_rel_5d", "liq", "mkt_today", "mkt_vol20", "tod"]
MARKET_WIDE = {"mkt_today", "mkt_vol20", "tod"}


def _csum(x: np.ndarray) -> np.ndarray:
    return np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), np.nan_to_num(x)]), axis=0)


def roll_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Sum of the last k rows INCLUDING the current one (NaN = 0)."""
    cs = _csum(x)
    t = np.arange(1, x.shape[0] + 1)
    return cs[t] - cs[np.maximum(t - k, 0)]


def by_day_grid(p: D.Panel, x: np.ndarray, nb: int = 26) -> np.ndarray:
    """T x N -> days x nb x N (NaN where a half day has no bar)."""
    g = np.full((len(p.dates), nb) + x.shape[1:], np.nan, dtype=x.dtype if x.dtype.kind == "f" else float)
    b = np.minimum(p.bar, nb - 1)
    g[p.day, b] = x
    return g


def ou_score(X: np.ndarray, W: int) -> tuple:
    """Avellaneda-Lee: AR(1) fit X_{j+1} = a + b X_j over the last W steps of the cumulative residual X (T x N, float64). Returns s-score, kappa (per bar), R^2 of the fit."""
    x, y = X[:-1], X[1:]
    def rs(z):
        cs = np.cumsum(np.concatenate([np.zeros((1,) + z.shape[1:]), z]), axis=0)
        t = np.arange(1, z.shape[0] + 1)
        return cs[t] - cs[np.maximum(t - W, 0)]
    n = np.minimum(np.arange(1, len(x) + 1), W)[:, None].astype(float)
    Sx, Sy, Sxx, Sxy, Syy = rs(x), rs(y), rs(x * x), rs(x * y), rs(y * y)
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = Sxx - Sx * Sx / n
        b = (Sxy - Sx * Sy / n) / vx
        a = (Sy - b * Sx) / n
        sse = Syy - a * Sy - b * Sxy
        var_e = np.maximum(sse, 0) / np.maximum(n - 2, 1)
        vy = Syy - Sy * Sy / n
        r2 = 1 - np.maximum(sse, 0) / vy
        m = a / (1 - b)
        seq = np.sqrt(var_e / (1 - b * b))
        s = (y - m) / seq                                                                  # y = X at the END of each step: the current value
        kappa = -np.log(b)
    bad = (n < W) | ~(b > 0) | ~(b < 1) | ~np.isfinite(s)
    s[bad], kappa[bad], r2[bad] = np.nan, np.nan, np.nan
    pad = np.full((1,) + X.shape[1:], np.nan)
    return np.vstack([pad, s]), np.vstack([pad, kappa]), np.vstack([pad, r2])


def res_momentum(res: Residuals, form_days: int = 252, skip_days: int = 21, vol_scale: bool = True, min_frac: float = 0.65) -> np.ndarray:
    """Blitz-Huij-Martens residual momentum per DAY (days x N): the sum of the daily residuals over days d-form_days .. d-skip_days-1, divided by their volatility x sqrt(n)
    (a t-statistic; without ``vol_scale`` the plain sum). Row d only uses completed days before d, so it is known before day d opens."""
    ed = res.e_d
    N = ed.shape[1]
    L = max(form_days - skip_days, 2)
    sh = skip_days + 1
    pad = lambda a: np.vstack([np.full((sh, N), np.nan), a[:-sh]]) if sh < len(a) else np.full_like(a, np.nan)
    num = pad(roll_sum(ed, L))
    cnt = pad(roll_sum(np.isfinite(ed).astype(float), L))
    with np.errstate(invalid="ignore", divide="ignore"):
        if vol_scale:
            sq = pad(roll_sum(ed ** 2, L))
            sd = np.sqrt(np.maximum(sq / cnt - (num / cnt) ** 2, 0) * cnt)
            out = num / sd
        else:
            out = num
    out[cnt < min_frac * L] = np.nan
    return out


def forward(res: Residuals, h: int, lag: int = 0) -> np.ndarray:
    """Residual open-to-open return from the open of bar t+1+lag over h bars, aligned to the decision bar t (NaN near the end)."""
    cs = _csum(res.e_oo)
    T = res.e_oo.shape[0]
    t = np.arange(T)
    a, b = t + 1 + lag, t + 1 + lag + h
    out = np.full(res.e_oo.shape, np.nan)
    ok = b <= T
    out[ok] = cs[b[ok]] - cs[a[ok]]
    bad = ~np.isfinite(res.e_oo)
    nan_any = roll_sum(bad.astype(float), h)                                                 # a target window with a missing bar (delisting) is unknown
    idx = np.minimum(t + lag + h, T - 1)
    out[(nan_any[idx] > 0) & ok[:, None]] = np.nan
    return out


def build(p: D.Panel, res: Residuals, lag: int = 0, names: Optional[list] = None, ou_window: int = 130, hks_days: int = 20) -> dict:
    """All features (T x N float32) for the panel. ``lag`` aligns the time-of-day features to the bar the order will be filled in."""
    names = list(names or FEATURES)
    T, N = p.close.shape
    out: dict = {}
    e = res.e_cc
    day = p.day
    f32 = lambda a: np.asarray(a, np.float32)
    for k in (1, 2, 4, 13, 26, 130, 520):
        if f"rr_{k}" in names:
            out[f"rr_{k}"] = f32(np.where(np.isfinite(e), roll_sum(e, k), np.nan))
    if "mom_12_1" in names:
        out["mom_12_1"] = f32(res_momentum(res, 252, 21)[day])                               # known before the open of day d (uses days <= d-22)
    if {"ou_s", "ou_kappa", "ou_r2"} & set(names):
        X = np.cumsum(np.nan_to_num(e), axis=0)
        s, kap, r2 = ou_score(X, ou_window)
        s[~np.isfinite(e)] = np.nan
        out["ou_s"], out["ou_kappa"], out["ou_r2"] = f32(s), f32(kap), f32(r2)
    if "hks" in names:
        g = by_day_grid(p, e)                                                                # days x 26 x N
        cs = np.cumsum(np.concatenate([np.zeros((1,) + g.shape[1:]), np.nan_to_num(g)]), axis=0)
        cn = np.cumsum(np.concatenate([np.zeros((1,) + g.shape[1:]), np.isfinite(g).astype(float)]), axis=0)
        dd = np.arange(g.shape[0])
        lo = np.maximum(dd - hks_days, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_prev = (cs[dd] - cs[lo]) / (cn[dd] - cn[lo])                                # previous hks_days days, same bar (today excluded)
        tgt_bar = p.bar.astype(int) + 1 + lag
        tgt_day = day.copy()
        nb = 26
        roll = tgt_bar >= (p.day_last[day] - p.day_first[day] + 1)                           # the target bar is in the next session
        tgt_day = np.where(roll, np.minimum(day + 1, g.shape[0] - 1), day)
        tgt_bar = np.where(roll, tgt_bar - (p.day_last[day] - p.day_first[day] + 1), tgt_bar)
        tgt_bar = np.minimum(tgt_bar, nb - 1)
        # for the next session's bars, "previous days" means up to and including today: use mean_prev of tgt_day (which excludes tgt_day itself)
        out["hks"] = f32(mean_prev[tgt_day, tgt_bar])
    if {"on_res", "id_today", "on_5d", "id_5d"} & set(names):
        on = res.e_on
        ecs = _csum(e)
        today = ecs[np.arange(1, T + 1)] - ecs[p.day_first[day]]                             # residual since the previous close (first bar includes the gap)
        out["on_res"] = f32(on[day])
        out["id_today"] = f32(today - np.nan_to_num(on[day]))
        idd = res.e_d - np.nan_to_num(on)                                                    # intraday part of each completed day
        on5 = roll_sum(on, 5)
        id5 = roll_sum(idd, 5)
        out["on_5d"] = f32(np.vstack([np.full((1, N), np.nan), on5[:-1]])[day])               # completed days only
        out["id_5d"] = f32(np.vstack([np.full((1, N), np.nan), id5[:-1]])[day])
    if "vol_surp" in names or "liq" in names:
        v = p.volume
        g = by_day_grid(p, v)
        cs = np.cumsum(np.concatenate([np.zeros((1,) + g.shape[1:]), np.nan_to_num(g)]), axis=0)
        dd = np.arange(g.shape[0])
        avg = (cs[dd] - cs[np.maximum(dd - 20, 0)]) / np.maximum(np.minimum(dd, 20), 1)[:, None, None]
        b = np.minimum(p.bar, 25)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["vol_surp"] = f32(np.log1p(v) - np.log1p(avg[day, b]))
        if "liq" in names:
            dv = by_day_grid(p, v * p.close)
            dvd = np.nansum(dv, axis=1)
            cs2 = _csum(dvd)
            adv = (cs2[dd] - cs2[np.maximum(dd - 20, 0)]) / np.maximum(np.minimum(dd, 20), 1)[:, None]
            with np.errstate(divide="ignore"):
                out["liq"] = f32(np.log(np.where(adv > 0, adv, np.nan))[day])
    if "ivol" in names:
        out["ivol"] = f32(np.sqrt(res.idio_var)[day])
    if "r2_fit" in names:
        out["r2_fit"] = f32(res.r2[day])
    if "range" in names or "cloc" in names:
        with np.errstate(invalid="ignore", divide="ignore"):
            rng = p.high - p.low
            out["range"] = f32(rng / p.close)
            out["cloc"] = f32(np.where(rng > 0, (p.close - p.low) / rng - 0.5, 0.0))
    if "vwap_dev" in names:
        pv = _csum(p.vwap * p.volume)
        vv = _csum(p.volume)
        t1 = np.arange(1, T + 1)
        f0 = p.day_first[day]
        with np.errstate(invalid="ignore", divide="ignore"):
            vw = (pv[t1] - pv[f0]) / (vv[t1] - vv[f0])
            out["vwap_dev"] = f32(p.close / vw - 1)
    if "beta" in names:
        out["beta"] = f32(res.b_m[day]) if np.isfinite(res.b_m).any() else f32(np.full((T, N), np.nan))
    if "sec_rel_5d" in names:
        cd = p.daily_close()
        m = p.col[D.MARKET]
        with np.errstate(invalid="ignore", divide="ignore"):
            r5 = np.full_like(cd, np.nan)
            r5[5:] = cd[5:] / cd[:-5] - 1
        rel = r5 - r5[:, [m]]
        sc = res.sector_col
        x = np.where(sc[None, :] >= 0, rel[:, np.maximum(sc, 0)], np.nan)
        out["sec_rel_5d"] = f32(np.vstack([np.full((1, N), np.nan), x[:-1]])[day])
    m = p.col[D.MARKET]
    if "mkt_today" in names:
        cd = p.daily_close()
        prev = np.r_[np.nan, cd[:-1, m]]
        out["mkt_today"] = f32(np.repeat((p.close[:, m] / prev[day] - 1)[:, None], N, 1))
    if "mkt_vol20" in names:
        cd = p.daily_close()[:, m]
        r = np.r_[np.nan, cd[1:] / cd[:-1] - 1]
        v = pd.Series(r).rolling(20, min_periods=15).std().shift(1).values * np.sqrt(252)
        out["mkt_vol20"] = f32(np.repeat(v[day][:, None], N, 1))
    if "tod" in names:
        out["tod"] = f32(np.repeat(p.bar[:, None].astype(float), N, 1))
    return {k: out[k] for k in names if k in out}


# ── cross-sectional tools ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def xs_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross-sectional rank of each row among the masked, finite cells, scaled to [-0.5, 0.5] (NaN elsewhere)."""
    v = np.where(mask & np.isfinite(x), x, np.nan)
    r = pd.DataFrame(v).rank(axis=1, method="average").to_numpy()
    n = np.isfinite(v).sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (r - 0.5) / n - 0.5


def xs_demean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.where(mask & np.isfinite(x), x, np.nan)
    with np.errstate(invalid="ignore"):
        return v - np.nanmean(v, axis=1, keepdims=True)


def row_corr(a: np.ndarray, b: np.ndarray, min_n: int = 20) -> np.ndarray:
    ok = np.isfinite(a) & np.isfinite(b)
    n = ok.sum(1)
    a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = a0.sum(1) / n, b0.sum(1) / n
        da, db = np.where(ok, a0 - ma[:, None], 0.0), np.where(ok, b0 - mb[:, None], 0.0)
        c = (da * db).sum(1) / np.sqrt((da * da).sum(1) * (db * db).sum(1))
    c[n < min_n] = np.nan
    return c


def nw_tstat(x: np.ndarray, lags: int) -> float:
    """Mean / Newey-West standard error of a daily series."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float("nan")
    u = x - x.mean()
    s = (u @ u) / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    return float(x.mean() / np.sqrt(max(s, 1e-18) / n))


def ic_series(feat: np.ndarray, target: np.ndarray, mask: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Rank IC (Spearman) per bar between a feature and a target among the masked cells."""
    if rows is not None:
        feat, target, mask = feat[rows], target[rows], mask[rows]
    return row_corr(xs_rank(feat, mask), xs_rank(target, mask))


def daily_mean(x: np.ndarray, day: np.ndarray, n_days: int) -> np.ndarray:
    s = np.bincount(day, weights=np.nan_to_num(x), minlength=n_days)
    c = np.bincount(day, weights=np.isfinite(x).astype(float), minlength=n_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)
