"""
Market data for the momentum book (no Qt).

  * official daily OHLCV (split + dividend adjusted): the MR app's cache (data/daily_mr.parquet, mr.market.load_daily)
  * from the 15-minute SIP bars of each session, what a decision ``lead`` bars before the close can know (lead 2 = 15:30, 1 = 15:45; on a 13:00 close 12:30 / 12:45):
      S        the last price (close of the last bar that ended by then)
      S15/S60  the last price one / four bars earlier
      HI/LO    the highest / lowest of the first bar's open and the bar closes so far
      CV       the volume so far
      sO       the open of the session's first (09:30) bar - "today's open" for every decision of the day
  * for the intraday symbol (QQQ): every bar of every session - from the SIP bars (the noise-area history) and from the intraday feed (iex or sip: the decision prices and
    VWAP of the day). The feed's sessions without bars (IEX history starts 2024-01-02) fall back to the SIP bars; ``feed_src`` says which one each session used.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import data as D
from ..mr.market import load_daily, session_closes

SLOTS = 26                                                                                      # 15-minute bars of a full session (09:30 ... 15:45)


def lead_bars(minute: int) -> int:
    """Bars between the decision time and the close: 1530 -> 2, 1545 -> 1."""
    hh, mm = divmod(int(minute), 100)
    mins = 16 * 60 - (hh * 60 + mm)
    if mins <= 0 or mins % D.BAR_MIN:
        raise ValueError(f"decision time {minute} must be a {D.BAR_MIN}-minute boundary before 16:00")
    return mins // D.BAR_MIN


_CAL: dict = {}


def closes(start, end) -> pd.Series:
    """Official close time of every NYSE session in [start, end] (13:00 on early-close days), sliced from one cached schedule (building a schedule takes ~0.5 s)."""
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    lo, hi = _CAL.get("range", (None, None))
    if lo is None or s < lo or e > hi:
        lo = min(s, lo or s, pd.Timestamp("2015-01-01"))
        hi = max(e, hi or e, pd.Timestamp(pd.Timestamp.now().year + 2, 12, 31))
        _CAL["c"] = session_closes(lo, hi)
        _CAL["range"] = (lo, hi)
    c = _CAL["c"]
    return c[(c.index >= s) & (c.index <= e)]


def bars_per_session(dates: pd.DatetimeIndex) -> np.ndarray:
    """15-minute bars of each session from the NYSE calendar: 26, or 14 on a 13:00 close."""
    if not len(dates):
        return np.zeros(0, int)
    c = closes(dates[0], dates[-1]).reindex(dates)
    mins = ((c - c.index) / pd.Timedelta(minutes=1)).fillna(960).to_numpy()
    return np.clip(((mins - 570) // D.BAR_MIN).astype(int), 1, SLOTS)


def slot_arrays(bars: pd.DataFrame, dates: pd.DatetimeIndex, nb: np.ndarray) -> dict:
    """One symbol's bars as session x slot arrays: close, volume, VWAP (T x 26) and the first bar's open (T). Slot 0 = the 09:30 bar; bars after an early close are dropped."""
    T = len(dates)
    cl, vo, vw = (np.full((T, SLOTS), np.nan) for _ in range(3))
    op = np.full(T, np.nan)
    if bars is None or not len(bars) or not T:
        return dict(cl=cl, vo=vo, vw=vw, op=op)
    idx = bars.index
    day = idx.normalize()
    ti = np.minimum(np.searchsorted(dates, day), T - 1)
    slot = ((idx - day) / pd.Timedelta(minutes=D.BAR_MIN)).to_numpy().astype(int) - 38
    ok = np.asarray(dates[ti] == day) & (slot >= 0) & (slot < SLOTS) & (slot < nb[ti])
    t, s = ti[ok], slot[ok]
    cl[t, s] = bars.close.to_numpy(float)[ok]
    vo[t, s] = bars.volume.to_numpy(float)[ok]
    if "vwap" in bars:
        vw[t, s] = bars.vwap.to_numpy(float)[ok]
    first = s == 0
    op[t[first]] = bars.open.to_numpy(float)[ok][first]
    return dict(cl=cl, vo=vo, vw=vw, op=op)


def last_upto(cl: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Per session: the last finite close among slots 0 .. k (NaN when k < 0 or none)."""
    T = len(cl)
    pos = np.where(np.isfinite(cl), np.arange(SLOTS), -1)
    pos = np.maximum.accumulate(pos, axis=1)
    kk = np.clip(k, 0, SLOTS - 1)
    p = pos[np.arange(T), kk]
    out = cl[np.arange(T), np.maximum(p, 0)]
    return np.where((k >= 0) & (p >= 0), out, np.nan)


def cum_vwap(vw: np.ndarray, vo: np.ndarray) -> np.ndarray:
    """Session VWAP through each slot (T x 26) from the bars' VWAP and volume; missing bars add nothing."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.cumsum(np.nan_to_num(vw * vo), 1) / np.cumsum(np.nan_to_num(vo), 1)


class MomMarket:
    """Aligned arrays (sessions x symbols). Daily: O H L C V (official). Decision-time snapshot from the SIP bars: S S15 S60 HI LO CV sO. Intraday symbol: ``iq``."""

    def __init__(self, symbols: list, minute: int = 1530, start=None, end=None, daily: Optional[pd.DataFrame] = None, extra_bars: Optional[dict] = None,
                 hist_bars: bool = True, bars_from=None, intraday_symbol: str = "QQQ", intraday_feed: str = "sip", feed_bars: Optional[pd.DataFrame] = None,
                 bars_fn: Optional[Callable[[str, str], pd.DataFrame]] = None):
        """``extra_bars`` = {symbol: today's SIP 15-minute bars} (live); ``hist_bars`` = False skips the cached SIP history; ``bars_from`` = read the cached bars from this day on
        (live needs only the last weeks); ``feed_bars`` = today's bars of the intraday symbol from the intraday feed (live); ``bars_fn(symbol, feed)`` replaces the cache (tests)."""
        self.minute = int(minute)
        self.lead = lead_bars(minute)
        d = load_daily(symbols) if daily is None else daily[daily.symbol.isin(symbols)]
        spy = d[d.symbol == "SPY"] if "SPY" in symbols else d
        cal = pd.DatetimeIndex(sorted(set(spy.date if len(spy) else d.date)))
        for f in list((extra_bars or {}).values()) + [feed_bars]:
            if f is not None and len(f):                                                       # live: today's session exists only in the intraday bars
                cal = cal.union(pd.DatetimeIndex(f.index.normalize().unique()))
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal < pd.Timestamp(end)]
        self.dates = cal
        self.syms = list(symbols)
        self.col = {s: j for j, s in enumerate(self.syms)}
        T, N = len(cal), len(self.syms)
        flds = ["open", "high", "low", "close", "volume"]
        w = d.drop_duplicates(["symbol", "date"], keep="last").set_index(["date", "symbol"])[flds].unstack("symbol") if len(d) else None
        piv = {k: (w[k] if w is not None else pd.DataFrame()).reindex(index=cal, columns=self.syms) for k in flds}
        self.O, self.H, self.L, self.C, self.V = (piv[k].to_numpy(float) for k in flds)
        self.nb = bars_per_session(cal)
        k_last = self.nb - 1 - self.lead
        self.S, self.S15, self.S60, self.HI, self.LO, self.CV, self.sO = (np.full((T, N), np.nan) for _ in range(7))
        self.has_bars = np.zeros(N, bool)
        b0 = pd.Timestamp(bars_from) if bars_from is not None else (cal[0] if T else None)
        load = bars_fn or (lambda s, feed: D.load_bars(s, feed))
        self.iq: dict = {}
        self.isym = intraday_symbol if intraday_symbol in self.col else ""
        for j, s in enumerate(self.syms):
            b = load(s, "sip") if hist_bars else pd.DataFrame(columns=D.BAR_COLS, dtype="float64")
            if b0 is not None and len(b):
                b = b[b.index >= b0]
            x = (extra_bars or {}).get(s)
            if x is not None and len(x):
                b = pd.concat([b[b.index < x.index.min()], x]) if len(b) else x
            if not len(b) or not T:
                continue
            self.has_bars[j] = True
            a = slot_arrays(b, cal, self.nb)
            cl = a["cl"]
            self.S[:, j] = last_upto(cl, k_last)
            self.S15[:, j] = last_upto(cl, k_last - 1)
            self.S60[:, j] = last_upto(cl, k_last - 4)
            upto = np.arange(SLOTS)[None, :] <= k_last[:, None]
            seg = np.concatenate([a["op"][:, None], np.where(upto, cl, np.nan)], axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)                                 # a session with no bar yet: all-NaN rows
                self.HI[:, j] = np.nanmax(seg, 1)
                self.LO[:, j] = np.nanmin(seg, 1)
            cv = np.nansum(np.where(upto, a["vo"], np.nan), 1)
            self.CV[:, j] = np.where(cv > 0, cv, np.nan)
            self.sO[:, j] = a["op"]
            if s == self.isym:
                self._intraday(a, s, intraday_feed, feed_bars, load, b0)

    def _intraday(self, sip: dict, s: str, feed: str, feed_bars, load, b0) -> None:
        """The intraday symbol's arrays: SIP closes (noise history) and the feed's closes / session VWAP / first open (decisions), with the per-session source."""
        T = len(self.dates)
        src = np.full(T, "", dtype=object)
        if feed == "sip":
            f = sip
            src[np.isfinite(sip["cl"]).any(1)] = "sip"
        else:
            fb = load(s, feed)
            if b0 is not None and len(fb):
                fb = fb[fb.index >= b0]
            if feed_bars is not None and len(feed_bars):
                fb = pd.concat([fb[fb.index < feed_bars.index.min()], feed_bars]) if len(fb) else feed_bars
            f = slot_arrays(fb, self.dates, self.nb)
            have = np.isfinite(f["cl"]).any(1)
            for k in ("cl", "vo", "vw"):
                f[k] = np.where(have[:, None], f[k], sip[k])
            f["op"] = np.where(have, f["op"], sip["op"])
            src[have] = feed
            src[~have & np.isfinite(sip["cl"]).any(1)] = "sip"
        self.iq = dict(symbol=s, feed=feed, sip_cl=sip["cl"], cl=f["cl"], vwap=cum_vwap(f["vw"], f["vo"]), op=f["op"], src=src)

    def t(self, day) -> int:
        return int(np.searchsorted(self.dates, pd.Timestamp(day).normalize()))

    def prev_close(self) -> np.ndarray:
        out = np.full_like(self.C, np.nan)
        out[1:] = self.C[:-1]
        return out
