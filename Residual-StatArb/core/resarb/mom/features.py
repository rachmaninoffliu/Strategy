"""
Everything the decisions read from a MomMarket - one function per quantity, used unchanged by the backtest, the model training and the live engine.

  rankable()       the names that can be ranked on a session: universe member, decision price and previous close known
  spy_trend()      SPY's previous close above its 200-session mean (min 150)
  noise_sigma()    the intraday symbol's noise width per slot: mean |close(slot) / official open - 1| of the previous `lookback` sessions
  feature_frame()  the ML forecast's 36 inputs per (session, name) - research/mp_features.py, raw columns - and the target y_on (close -> next official open)

"Today's open" in every decision is the open of the session's first SIP bar (live cannot read the official daily bar before the session ends); earlier sessions use the
official daily bars. Every rolling window ends at yesterday except the ones that include today's overnight move (on5 / on20 / on60).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..mr import engine as MRE
from .market import closes
from .models import ETF_UNIVERSE

STOCK = ["d45", "res45", "id45", "gap", "hour", "last15", "ibs45", "vsurp", "r1", "r5", "r21", "r63", "mom12_1", "on5", "on20", "on60", "id5", "id20", "id60", "rv20", "rv60",
         "range20", "beta60", "hi52", "c_sma20", "c_sma200", "dvol"]
MARKET = ["m_spy45", "m_spyrv20", "m_spytrend", "m_spyon20", "m_breadth", "m_disp", "m_dow", "m_tom"]
FEATURES = STOCK + MARKET + ["etf"]
ETF_SET = set(ETF_UNIVERSE)


def lag(x: np.ndarray, n: int = 1) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if n < len(x):
        out[n:] = x[:-n]
    return out


def _roll_mean(x: np.ndarray, n: int, mp: int = 0) -> np.ndarray:
    return pd.DataFrame(x).rolling(n, min_periods=mp or max(2, int(n * 0.75))).mean().to_numpy()


def _roll_mean_today(hist: np.ndarray, today: np.ndarray, n: int, mp: int = 0) -> np.ndarray:
    """Mean over the n sessions ending today: ``hist`` for the n-1 earlier sessions, ``today`` for the session itself (NaN skipped, at least mp values)."""
    mp = mp or max(2, int(n * 0.75))
    s = lag(pd.DataFrame(np.nan_to_num(hist)).rolling(n - 1, min_periods=1).sum().to_numpy())
    c = lag(pd.DataFrame(np.isfinite(hist).astype(float)).rolling(n - 1, min_periods=1).sum().to_numpy())
    s, c = np.nan_to_num(s), np.nan_to_num(c)
    tot = s + np.nan_to_num(today)
    cnt = c + np.isfinite(today)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= mp, tot / cnt, np.nan)


def member_mask(mk, m) -> np.ndarray:
    """Universe membership per session (point-in-time topN by month, ETF / list parts always)."""
    return MRE.universe_mask(mk, m)


def rankable(mk, member: np.ndarray) -> np.ndarray:
    return member & np.isfinite(mk.S) & np.isfinite(mk.prev_close())


def spy_trend(mk) -> np.ndarray:
    j = mk.col.get("SPY")
    if j is None:
        return np.zeros(len(mk.dates), bool)
    c = mk.C[:, j]
    sma = pd.Series(c).rolling(200, min_periods=150).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        return lag(c[:, None])[:, 0] > lag(sma[:, None])[:, 0]


def noise_sigma(mk, lookback: int) -> np.ndarray:
    """Sessions x slots: mean |SIP close(slot) / official open - 1| over the previous ``lookback`` sessions (at least half of them; a 13:00 close leaves the late slots empty)."""
    iq = mk.iq
    if not iq:
        return np.full((len(mk.dates), 26), np.nan)
    o = mk.O[:, mk.col[iq["symbol"]]]
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.abs(iq["sip_cl"] / o[:, None] - 1)
    return pd.DataFrame(move).rolling(lookback, min_periods=max(2, lookback // 2)).mean().shift(1).to_numpy()


def turn_of_month(dates: pd.DatetimeIndex) -> np.ndarray:
    """-sessions left in the month (incl. today) on the last two sessions, else the session's number in the month - from the NYSE calendar (live knows the future sessions)."""
    if not len(dates):
        return np.zeros(0)
    cal = closes(dates[0] - pd.Timedelta(days=40), dates[-1] + pd.Timedelta(days=40)).index
    s = pd.Series(1, index=cal)
    mon = cal.to_period("M")
    pos = s.groupby(mon).cumsum()
    left = s[::-1].groupby(mon[::-1]).cumsum()[::-1]
    tom = pd.Series(np.where(left <= 2, -left, pos), index=cal).astype(float)
    return tom.reindex(pd.DatetimeIndex(dates)).to_numpy()


def beta(rc: np.ndarray, spy: int, n: int = 60) -> np.ndarray:
    """Rolling beta to SPY of close-to-close returns over the n sessions ending yesterday (at least n / 2 pairs)."""
    x = rc
    y = np.broadcast_to(rc[:, [spy]], x.shape)
    ok = np.isfinite(x) & np.isfinite(y)
    x0, y0 = np.where(ok, x, 0.0), np.where(ok, y, 0.0)

    def roll(a):
        c = np.cumsum(a, axis=0)
        c[n:] = c[n:] - c[:-n].copy()
        return c
    k = roll(ok.astype(float))
    sx, sy, sxy, syy = roll(x0), roll(y0), roll(x0 * y0), roll(y0 * y0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (sxy - sx * sy / k) / (syy - sy * sy / k)
    b[k < n // 2] = np.nan
    return lag(b)


def arrays(mk, U: np.ndarray) -> tuple:
    """(stock features {name: T x N}, market features {name: T}, etf flag N, decision move d45 T x N)."""
    C, O, H, L, V = mk.C, mk.O, mk.H, mk.L, mk.V
    S, sO = mk.S, mk.sO
    prevC = mk.prev_close()
    spy = mk.col.get("SPY")
    T, N = C.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = C / prevC - 1
        rc = np.where(np.isfinite(rc), rc, np.nan)
        ron = O / prevC - 1
        ron = np.where(np.isfinite(ron), ron, np.nan)
        ron_today = sO / prevC - 1
        rid = C / O - 1
        rid = np.where(np.isfinite(rid), rid, np.nan)
        d = S / prevC - 1
        b = beta(rc, spy) if spy is not None else np.full((T, N), np.nan)
        cv = mk.CV
        spy_d = d[:, [spy]] if spy is not None else np.full((T, 1), np.nan)
        f = {
            "d45": d, "res45": d - np.nan_to_num(b, nan=1.0) * spy_d, "id45": S / sO - 1, "gap": sO / prevC - 1, "hour": S / mk.S60 - 1, "last15": S / mk.S15 - 1,
            "ibs45": (S - mk.LO) / (mk.HI - mk.LO), "vsurp": cv / pd.DataFrame(cv).rolling(20, min_periods=10).mean().shift(1).to_numpy(),
            "r1": lag(rc), "r5": prevC / lag(C, 6) - 1, "r21": prevC / lag(C, 22) - 1, "r63": prevC / lag(C, 64) - 1, "mom12_1": lag(C, 22) / lag(C, 253) - 1,
            "on5": _roll_mean_today(ron, ron_today, 5), "on20": _roll_mean_today(ron, ron_today, 20), "on60": _roll_mean_today(ron, ron_today, 60),
            "id5": lag(_roll_mean(rid, 5)), "id20": lag(_roll_mean(rid, 20)), "id60": lag(_roll_mean(rid, 60)),
            "rv20": lag(pd.DataFrame(rc).rolling(20, min_periods=15).std().to_numpy()), "rv60": lag(pd.DataFrame(rc).rolling(60, min_periods=45).std().to_numpy()),
            "range20": lag(_roll_mean((H - L) / C, 20)), "beta60": b, "hi52": prevC / lag(pd.DataFrame(C).rolling(252, min_periods=200).max().to_numpy()),
            "c_sma20": prevC / lag(_roll_mean(C, 20)) - 1, "c_sma200": prevC / lag(_roll_mean(C, 200, 150)) - 1,
            "dvol": lag(np.log(pd.DataFrame(C * V).rolling(63, min_periods=40).median().to_numpy())),
        }
        nan_t = np.full(T, np.nan)
        mkt = {
            "m_spy45": d[:, spy] if spy is not None else nan_t, "m_spyrv20": f["rv20"][:, spy] if spy is not None else nan_t,
            "m_spytrend": f["c_sma200"][:, spy] if spy is not None else nan_t, "m_spyon20": f["on20"][:, spy] if spy is not None else nan_t,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                                     # a session with no rankable name
            mkt["m_breadth"] = np.nanmean(np.where(U, d > 0, np.nan), 1)
            mkt["m_disp"] = np.nanstd(np.where(U, d, np.nan), 1)
        mkt["m_dow"] = mk.dates.dayofweek.to_numpy().astype(float)
        mkt["m_tom"] = turn_of_month(mk.dates)
    etf = np.array([s in ETF_SET for s in mk.syms], float)
    return f, mkt, etf, d


def feature_frame(mk, U: np.ndarray, rows=None, target: bool = False) -> tuple:
    """(DataFrame with t, j, the FEATURES as float32 [, y_on] for every rankable (session, name) - or only the sessions in ``rows``; the move d45 as T x N)."""
    f, mkt, etf, d = arrays(mk, U)
    sel = U.copy()
    if rows is not None:
        keep = np.zeros(len(U), bool)
        keep[list(rows)] = True
        sel &= keep[:, None]
    ti, ji = np.nonzero(sel)
    df = pd.DataFrame({"t": ti, "j": ji})
    for k in STOCK:
        df[k] = f[k][ti, ji].astype(np.float32)
    for k in MARKET:
        df[k] = mkt[k][ti].astype(np.float32)
    df["etf"] = etf[ji].astype(np.float32)
    if target:
        with np.errstate(invalid="ignore", divide="ignore"):
            y = lag(mk.O[::-1])[::-1] / mk.C - 1                                                # close t -> official open t+1
        df["y_on"] = y[ti, ji]
    df["date"] = mk.dates[ti]
    return df, d
