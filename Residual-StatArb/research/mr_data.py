"""
Daily research data for the mean-reversion study: official daily OHLCV of every U.S. stock / ETF Alpaca has (Stats-arb survivorship-free cache, delisted names included,
split + dividend adjusted), point-in-time liquidity universes (top-N by 63-session median dollar volume, chosen at each month end for the next month), and - for names with
15-minute bars - the 15:30 / 15:45 prices a market-on-close order can still be based on.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
from resarb import data as D  # noqa: E402

DAILY = os.path.join(D.STATARB_DATA, "statarb_daily_broad.parquet")
CACHE = os.path.join(HERE, "reports", "mr_cache")
os.makedirs(CACHE, exist_ok=True)


class Daily:
    """Wide daily arrays (dates x symbols)."""

    def __init__(self, start="2016-01-01", symbols=None):
        f = os.path.join(CACHE, "daily_wide.npz")
        if os.path.exists(f) and symbols is None:
            z = np.load(f, allow_pickle=True)
            self.dates = pd.DatetimeIndex(z["dates"])
            self.syms = list(z["syms"])
            for k in ("open", "high", "low", "close", "volume"):
                setattr(self, k, z[k])
        else:
            d = pd.read_parquet(DAILY)
            d = d[d.date >= pd.Timestamp(start)]
            if symbols is not None:
                d = d[d.symbol.isin(symbols)]
            piv = {k: d.pivot(index="date", columns="symbol", values=k).sort_index() for k in ("open", "high", "low", "close", "volume")}
            self.dates = piv["close"].index
            self.syms = list(piv["close"].columns)
            for k, v in piv.items():
                setattr(self, k, v.reindex(columns=self.syms).to_numpy(float))
            fixes = pd.read_parquet(D.FIX_FILE) if os.path.exists(D.FIX_FILE) else pd.DataFrame(columns=["symbol", "ex_date", "factor"])
            col = {s: j for j, s in enumerate(self.syms)}
            for r in fixes.itertuples():                                                     # the same unadjusted-distribution fixes as the 15-minute bars
                j = col.get(r.symbol)
                if j is not None:
                    m = self.dates < pd.Timestamp(r.ex_date)
                    for k in ("open", "high", "low", "close"):
                        getattr(self, k)[m, j] *= float(r.factor)
            if symbols is None:
                np.savez(f, dates=self.dates.values, syms=np.array(self.syms, dtype=object), open=self.open, high=self.high, low=self.low, close=self.close, volume=self.volume)
        self.col = {s: j for j, s in enumerate(self.syms)}

    def j(self, s):
        return self.col[s]


def universe(daily: Daily, n: int, exclude_etf: bool = True) -> np.ndarray:
    """dates x symbols bool: member of the top-n by 63-session median dollar volume of the previous month end (price >= $10, >= 126 sessions of history, no ETPs)."""
    f = os.path.join(CACHE, f"univ_{n}_{int(exclude_etf)}.npy")
    if os.path.exists(f):
        return np.load(f)
    px = pd.DataFrame(daily.close, index=daily.dates, columns=daily.syms)
    dv = pd.DataFrame(daily.close * daily.volume, index=daily.dates, columns=daily.syms)
    med = dv.rolling(63, min_periods=40).median()
    hist = px.notna().cumsum()
    names = pd.read_parquet(os.path.join(D.STATARB_DATA, "statarb_assets.parquet")).drop_duplicates("symbol").set_index("symbol").name
    bad = D.etp_symbols(names) | set(D.ETFS) | {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO", "UUP", "VIXY", "ITA"} if exclude_etf else set()
    dup = set(D.DUP_CLASS)
    mon = daily.dates.to_period("M")
    out = np.zeros(daily.close.shape, bool)
    me = np.r_[np.where(mon[1:] != mon[:-1])[0], len(mon) - 1]
    for k, i in enumerate(me):
        row = med.iloc[i]
        ok = row.notna() & (px.iloc[i] >= 10) & (hist.iloc[i] >= 126) & ~row.index.isin(list(bad | dup))
        top = row[ok].sort_values(ascending=False).head(n).index
        nxt = me[k] + 1
        end = me[k + 1] + 1 if k + 1 < len(me) else len(mon)
        cols = [daily.col[s] for s in top]
        out[nxt:end][:, cols] = True
    np.save(f, out)
    return out


def snapshot_prices(daily: Daily, minute: int = 945) -> np.ndarray:
    """dates x symbols: the price at hh:mm ET (the close of the 15-minute bar ending then) from the 15-minute SIP cache - what an order sent before the 15:50 market-on-close
    cut-off can use. NaN where no 15-minute bars exist."""
    f = os.path.join(CACHE, f"snap_{minute}.npy")
    if os.path.exists(f):
        return np.load(f)
    hh, mm = divmod(minute, 100)
    bar_start = pd.Timedelta(hours=hh, minutes=mm) - pd.Timedelta(minutes=15)
    out = np.full(daily.close.shape, np.nan)
    for fn in os.listdir(D.bars_dir("sip")):
        s = fn[:-8].replace("_", ".")
        if s not in daily.col:
            continue
        b = D.load_bars(s)
        if not len(b):
            continue
        sel = b[(b.index - b.index.normalize()) == bar_start].close
        sel.index = sel.index.normalize()
        out[:, daily.col[s]] = sel.reindex(daily.dates).to_numpy(float)
    np.save(f, out)
    return out


def stats(r: pd.Series, name: str = "", split: str = "2023-01-01") -> dict:
    r = r.dropna()
    out = {"name": name}
    for lab, x in (("all", r), ("is", r[r.index < split]), ("oos", r[r.index >= split])):
        if len(x) < 30:
            continue
        eq = (1 + x).cumprod()
        yrs = len(x) / 252
        out[f"cagr_{lab}"] = eq.iloc[-1] ** (1 / yrs) - 1
        out[f"sh_{lab}"] = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
        out[f"mdd_{lab}"] = (eq / eq.cummax() - 1).min()
        out[f"vol_{lab}"] = x.std() * np.sqrt(252)
    return out
