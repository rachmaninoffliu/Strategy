"""
The intraday noise-area sleeve as a reusable function (Zarattini, Aziz & Barbon 2024, "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)"),
long-only and unlevered, on 15-minute SIP bars:

  band(slot)  = max(today's open, yesterday's close) x (1 + mean over the last `lookback` sessions of |close(slot) / open - 1|)   (the "noise area" for that time of day)
  every `every` bars (from the 10:00 bar on, the last check at the bar that ends 15 or 30 minutes before the close):
      flat and the bar closes above the band          -> buy at that price (half spread)
      long and the bar closes below max(band, VWAP)   -> sell at that price (half spread)        [trailing=True]
  anything still held is sold in the closing auction.
Several symbols split the capital equally (1/n each).
"""
import numpy as np
import pandas as pd

from mp_lib import Ctx
from resarb import data as D

_BARS: dict = {}


def bars_matrix(cx: Ctx, tk: str, feed: str = "sip"):
    """(close, cumulative VWAP) as T x 26 arrays for one symbol, NaN after an early close."""
    if (tk, feed) in _BARS:
        return _BARS[(tk, feed)]
    b = D.load_bars(tk, feed)
    b = b[(b.index.normalize() >= cx.dates[0]) & (b.index.normalize() <= cx.dates[-1])]
    nb = cx.p.cube()[3]
    T = cx.T
    day = b.index.normalize()
    tix = {d: i for i, d in enumerate(cx.dates)}
    ti = np.array([tix.get(x, -1) for x in day])
    slot = ((b.index - day) / pd.Timedelta(minutes=15)).astype(int) - 38
    ok = (ti >= 0) & (slot >= 0) & (slot < 26)
    pv, vw, vv = (np.full((T, 26), np.nan) for _ in range(3))
    pv[ti[ok], slot[ok]] = b.close.to_numpy()[ok]
    vw[ti[ok], slot[ok]] = b.vwap.to_numpy()[ok]
    vv[ti[ok], slot[ok]] = b.volume.to_numpy()[ok]
    for t in range(T):
        pv[t, nb[t]:] = np.nan
        vv[t, nb[t]:] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        cvw = np.nancumsum(np.nan_to_num(vw * vv), 1) / np.nancumsum(np.nan_to_num(vv), 1)
    _BARS[(tk, feed)] = (pv, cvw)
    return _BARS[(tk, feed)]


def noise_area(cx: Ctx, tks, every=2, lookback=14, trail=True, mult=1.0, half_bp=0.5, auction_bp=1.0, gate=None, trades=None, live_feed=None):
    """Daily return series (dated by the session) of the long-only noise-area sleeve on ``tks`` (1/n of the capital each).
    ``gate`` = optional T-bool: trade only on those sessions. ``trades`` = optional list that receives one dict per round trip.
    ``live_feed`` = 'iex': the decision prices and VWAP of the current session come from that feed (the band stays on the SIP history); sessions without
    those bars are skipped."""
    tks = [tks] if isinstance(tks, str) else list(tks)
    nb = cx.p.cube()[3]
    out = np.zeros(cx.T)
    for tk in tks:
        j = cx.j(tk)
        pv, cvw = bars_matrix(cx, tk)
        dp, dv = (pv, cvw) if live_feed is None else bars_matrix(cx, tk, live_feed)
        o, pc, c = cx.O[:, j], cx.lag(cx.C)[:, j], cx.C[:, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            move = np.abs(pv / o[:, None] - 1)
        sig = pd.DataFrame(move).rolling(lookback, min_periods=lookback).mean().shift(1).to_numpy() * mult
        for t in range(cx.T):
            if gate is not None and not gate[t]:
                continue
            if not (np.isfinite(o[t]) and np.isfinite(pc[t]) and np.isfinite(c[t])):
                continue
            if live_feed is not None and not np.isfinite(dp[t]).any():
                continue
            n = int(nb[t])
            upper = max(o[t], pc[t]) * (1 + sig[t])
            pos, px_in, s_in = False, 0.0, -1
            r = 0.0
            for s in range(every - 1, n - 1, every):
                p = dp[t, s]
                if not np.isfinite(p) or not np.isfinite(upper[s]):
                    continue
                if not pos and p > upper[s]:
                    pos, px_in, s_in = True, p, s
                elif pos and trail and p < max(upper[s], dv[t, s]):
                    rr = p / px_in - 1 - 2 * half_bp / 1e4
                    r += rr
                    if trades is not None:
                        trades.append({"date": cx.dates[t], "symbol": tk, "in_slot": s_in, "out_slot": s, "px_in": px_in, "px_out": p, "ret": rr, "exit": "band/VWAP"})
                    pos = False
            if pos:
                rr = c[t] / px_in - 1 - (half_bp + auction_bp) / 1e4
                r += rr
                if trades is not None:
                    trades.append({"date": cx.dates[t], "symbol": tk, "in_slot": s_in, "out_slot": n, "px_in": px_in, "px_out": c[t], "ret": rr, "exit": "close"})
            out[t] += r / len(tks)
    return cx.series(out)
