"""
Market data for the mean-reversion models (no Qt).

  * official daily OHLCV (split + dividend adjusted) - Alpaca daily bars; for stocks the survivorship-free Stats-arb cache seeds the history - kept in data/daily_mr.parquet
  * the intraday SNAPSHOT of each session as of a decision time hh:mm (open, high, low, last price of the 15-minute bars that ended by then): what a market-on-close order
    sent before the 15:50 cut-off can know. History from the 15-minute SIP cache; in live from the live feed.
  * Market: aligned arrays (sessions x symbols) of both, for a symbol list
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import data as D

DAILY_FILE = os.path.join(D.DATA_DIR, "daily_mr.parquet")
COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def load_daily(symbols: Optional[list] = None, fix: bool = True) -> pd.DataFrame:
    """Daily bars from the cache, with the same back-adjustment of unadjusted distributions as the 15-minute bars (data.FIX_FILE); the file itself stays raw."""
    if not os.path.exists(DAILY_FILE):
        return pd.DataFrame(columns=COLS)
    d = pd.read_parquet(DAILY_FILE)
    if symbols is not None:
        d = d[d.symbol.isin(symbols)]
    if fix and os.path.exists(D.FIX_FILE):
        fx = pd.read_parquet(D.FIX_FILE)
        fx = fx[fx.symbol.isin(set(d.symbol))]
        if len(fx):
            d = d.copy()
            for r in fx.itertuples():
                m = (d.symbol == r.symbol) & (d.date < pd.Timestamp(r.ex_date))
                d.loc[m, ["open", "high", "low", "close"]] *= float(r.factor)
    return d


def _fetch_daily(client, syms: list, start, end=None) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    end = end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=D.SIP_DELAY_MIN))
    for attempt in range(8):
        try:
            df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame.Day, start=pd.Timestamp(start, tz="UTC").to_pydatetime(),
                                                        end=pd.Timestamp(end).to_pydatetime(), adjustment=Adjustment.ALL, feed=DataFeed.SIP, limit=50_000_000)).df
            if not len(df):
                return pd.DataFrame(columns=COLS)
            df = df.reset_index()
            df["date"] = pd.to_datetime(df.timestamp).dt.tz_convert(D.NY).dt.tz_localize(None).dt.normalize()
            return df[COLS]
        except Exception as e:                                                                # noqa: BLE001
            msg = str(e).lower()
            if any(k in msg for k in ("429", "too many", "timeout", "connection", "502", "503", "504")):
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            raise
    return pd.DataFrame(columns=COLS)


def update_daily(symbols: list, start: str = "2015-06-01", log: Callable[[str], None] = print, client=None) -> pd.DataFrame:
    """Download / top up official daily bars for ``symbols``. A symbol already cached is topped up from a 10-session overlap and downloaded again in full when the overlap shows
    that its adjusted history was rewritten (a split or dividend since the last download). Stocks missing here are seeded from the Stats-arb daily cache first."""
    client = client or D.data_client()
    old = load_daily(fix=False)
    have = set(old.symbol.unique()) if len(old) else set()
    seed_f = os.path.join(D.STATARB_DATA, "statarb_daily_broad.parquet")
    new_syms = [s for s in symbols if s not in have]
    frames = [old] if len(old) else []
    if new_syms and os.path.exists(seed_f):
        seed = pd.read_parquet(seed_f)
        seed = seed[seed.symbol.isin(new_syms)][COLS]
        if len(seed):
            frames.append(seed)
            have |= set(seed.symbol.unique())
            log(f"  seeded {seed.symbol.nunique()} symbols from the Stats-arb daily cache")
    cur = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    full = [s for s in symbols if s not in have]
    last = cur.groupby("symbol").date.max() if len(cur) else pd.Series(dtype="datetime64[ns]")
    top = [s for s in symbols if s in have]
    out = [cur]
    if top:
        t0 = (last.reindex(top).min() - pd.Timedelta(days=16)).strftime("%Y-%m-%d")
        chunks = [top[i:i + 200] for i in range(0, len(top), 200)]
        with ThreadPoolExecutor(3) as ex:
            got = [g for g in ex.map(lambda ch: _fetch_daily(client, ch, t0), chunks) if len(g)]
        if got:
            nw = pd.concat(got, ignore_index=True)
            o = cur.set_index(["symbol", "date"]).close
            n = nw.set_index(["symbol", "date"]).close
            common = o.index.intersection(n.index)
            redo = []
            if len(common):
                rel = (n.reindex(common) / o.reindex(common) - 1).abs().groupby(level=0).max()
                redo = list(rel[rel > 5e-4].index)
            if redo:
                log(f"  {len(redo)} symbols had their adjusted history rewritten - downloading them again")
                full += redo
                out = [cur[~cur.symbol.isin(redo)]]
                nw = nw[~nw.symbol.isin(redo)]
            out.append(nw)
    if full:
        chunks = [full[i:i + 100] for i in range(0, len(full), 100)]
        with ThreadPoolExecutor(3) as ex:
            out += [g for g in ex.map(lambda ch: _fetch_daily(client, ch, start), chunks) if len(g)]
    df = pd.concat(out, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
    os.makedirs(D.DATA_DIR, exist_ok=True)
    df.to_parquet(DAILY_FILE)
    log(f"daily bars: {df.symbol.nunique()} symbols through {df.date.max():%Y-%m-%d}")
    return df


_CLOSES: dict = {}


def session_closes(start, end) -> pd.Series:
    """Official close time (ET, naive) of every NYSE session in [start, end]: 16:00, 13:00 on the early-close days (pandas_market_calendars; 16:00 as a fallback)."""
    key = (str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
    if key not in _CLOSES:
        try:
            import pandas_market_calendars as mcal
            sch = mcal.get_calendar("NYSE").schedule(start_date=key[0], end_date=key[1])
            c = pd.DatetimeIndex(sch.market_close).tz_convert(D.NY).tz_localize(None)
            _CLOSES[key] = pd.Series(c, index=c.normalize())
        except Exception:                                                                    # noqa: BLE001
            d = pd.bdate_range(key[0], key[1])
            _CLOSES[key] = pd.Series(d + pd.Timedelta(hours=16), index=d)
    return _CLOSES[key]


def decision_lead(minute: int) -> pd.Timedelta:
    """How long before the official close the decision is taken: 15:45 -> 15 minutes. On an early-close day the decision moves with the close (12:45 for a 13:00 close)."""
    hh, mm = divmod(int(minute), 100)
    return pd.Timedelta(hours=16) - pd.Timedelta(hours=hh, minutes=mm)


def snapshot_from_bars(bars: pd.DataFrame, minute: int, closes: Optional[pd.Series] = None) -> pd.DataFrame:
    """Per session: open (first bar's open), high, low, last close of the 15-minute bars that ENDED by the decision time (close - lead, i.e. hh:mm on a normal day)."""
    if not len(bars):
        return pd.DataFrame(columns=["open", "high", "low", "last"])
    day = bars.index.normalize()
    close = pd.DatetimeIndex(day + pd.Timedelta(hours=16))
    if closes is not None and len(closes):
        mapped = pd.DatetimeIndex(pd.Series(day).map(closes))
        close = pd.DatetimeIndex(np.where(mapped.isna(), close, mapped))
    cut = close - decision_lead(minute) - pd.Timedelta(minutes=D.BAR_MIN)
    b = bars[bars.index <= cut]
    g = b.groupby(b.index.normalize())
    return pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "last": g.close.last()})


class Market:
    """Aligned arrays (sessions x symbols): official daily O / H / L / C / V and the decision-time snapshot sO / sH / sL / sP. A session with no snapshot has NaN there."""

    def __init__(self, symbols: list, minute: int = 1545, start=None, end=None, daily: Optional[pd.DataFrame] = None, extra_bars: Optional[dict] = None,
                 feed: str = "sip", hist_bars: bool = True, bars_fn: Optional[Callable[[str], pd.DataFrame]] = None):
        """``extra_bars`` = {symbol: today's 15-minute bars} (live); ``hist_bars`` = False skips the cached 15-minute history (live needs only today's snapshot);
        ``bars_fn(symbol)`` replaces the cached SIP history (research: e.g. the live hybrid feed rebuilt from the SIP and IEX caches)."""
        self.minute = int(minute)
        d = load_daily(symbols) if daily is None else daily[daily.symbol.isin(symbols)]
        spy = d[d.symbol == "SPY"] if "SPY" in symbols else d
        cal = pd.DatetimeIndex(sorted(set(spy.date if len(spy) else d.date)))
        if extra_bars:                                                                         # live: today's session exists only in the intraday bars
            for f in extra_bars.values():
                if f is not None and len(f):
                    cal = cal.union(pd.DatetimeIndex(f.index.normalize().unique()))
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal < pd.Timestamp(end)]
        self.dates = cal
        self.syms = list(symbols)
        self.col = {s: j for j, s in enumerate(self.syms)}
        T, N = len(cal), len(self.syms)
        piv = {k: d.pivot_table(index="date", columns="symbol", values=k).reindex(index=cal, columns=self.syms) for k in ("open", "high", "low", "close", "volume")}
        self.O, self.H, self.L, self.C, self.V = (piv[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        self.sO, self.sH, self.sL, self.sP = (np.full((T, N), np.nan) for _ in range(4))
        closes = session_closes(cal[0], cal[-1] + pd.Timedelta(days=1)) if T else None
        for j, s in enumerate(self.syms):
            b = (bars_fn(s) if bars_fn else D.load_bars(s, feed if feed == "iex" else "sip")) if hist_bars else pd.DataFrame(columns=D.BAR_COLS, dtype="float64")
            if extra_bars and s in extra_bars and extra_bars[s] is not None and len(extra_bars[s]):
                x = extra_bars[s]
                b = pd.concat([b[b.index < x.index.min()], x]) if len(b) else x
            sn = snapshot_from_bars(b, self.minute, closes).reindex(cal)
            self.sO[:, j], self.sH[:, j], self.sL[:, j], self.sP[:, j] = (sn[k].to_numpy(float) for k in ("open", "high", "low", "last"))

    def t(self, day) -> int:
        return int(np.searchsorted(self.dates, pd.Timestamp(day).normalize()))
