"""Synthetic market for the tests: SPY, QQQ, IWM, the sector SPDRs and stocks with known betas and mean-reverting residuals, written as 15-minute bar files into a temp data dir.
Import this module BEFORE resarb (it points RESARB_DATA_DIR at the temp dir)."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

TMP = tempfile.mkdtemp(prefix="resarb_test_")
os.environ["RESARB_DATA_DIR"] = TMP
os.environ["RESARB_SETTINGS_FILE"] = os.path.join(TMP, "settings.ini")
for k in [k for k in os.environ if k.endswith("_ALPACA_API_KEY") or k.endswith("_ALPACA_SECRET_KEY") or k.startswith("ALPACA_")]:
    del os.environ[k]                                                                         # the tests never see real keys ...
os.environ["ALPACA_KEY_FILE"] = "none"                                                        # ... nor the project's key file

SECTORS = {"Tech": "XLK", "Fin": "XLF", "Health": "XLV", "Energy": "XLE"}


def sessions(n_days: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    end = (end or pd.Timestamp.now().normalize() - pd.Timedelta(days=1))
    d = pd.bdate_range(end=end, periods=n_days)
    return d


def bar_index(days: pd.DatetimeIndex, half_days: tuple = ()) -> pd.DatetimeIndex:
    out = []
    for d in days:
        nb = 14 if d in half_days else 26
        out.extend(d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * k) for k in range(nb))
    return pd.DatetimeIndex(out)


def make_market(n_stocks: int = 24, n_days: int = 320, seed: int = 0, end=None) -> dict:
    """Returns dict(frames, betas, sectors, days, ts) and writes the bar files, universe and sectors into TMP."""
    from resarb import data as D
    rng = np.random.default_rng(seed)
    days = sessions(n_days, end)
    ts = bar_index(days)
    T = len(ts)
    first = np.r_[True, ts.normalize()[1:] != ts.normalize()[:-1]]
    vol = np.where(first, 0.006, 0.0015)                                                      # the first bar carries the overnight move
    rm = rng.normal(0.00002, 1, T) * vol
    sec_ret = {e: rng.normal(0, 1, T) * vol * 0.6 for e in D.SECTOR_ETF.values()}
    frames, betas, sec_of = {}, {}, {}

    def to_frame(r, p0=100.0):
        c = p0 * np.cumprod(1 + r)
        o = np.r_[p0, c[:-1]] * (1 + rng.normal(0, 0.0002, T))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0005, T)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0005, T)))
        v = rng.integers(50_000, 150_000, T).astype(float)
        return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v, "vwap": (o + c) / 2, "trades": v / 100}, index=ts)
    frames["SPY"] = to_frame(rm, 400.0)
    frames["QQQ"] = to_frame(rm * 1.2 + rng.normal(0, 1, T) * vol * 0.2, 350.0)
    frames["IWM"] = to_frame(rm * 1.1 + rng.normal(0, 1, T) * vol * 0.3, 200.0)
    for e, s in sec_ret.items():
        frames[e] = to_frame(rm + s, 80.0)
    names = list(SECTORS)
    for i in range(n_stocks):
        sym = "S" + chr(65 + i // 26) + chr(65 + i % 26)
        sec = names[i % len(names)]
        e = SECTORS[sec]
        bm, bs = 0.6 + 0.8 * rng.random(), 0.3 + 0.9 * rng.random()
        x = np.zeros(T)
        eps = rng.normal(0, 1, T) * vol * 0.8
        for t in range(1, T):
            x[t] = 0.97 * x[t - 1] + eps[t]                                                   # mean-reverting cumulative residual
        idio = np.diff(np.r_[0.0, x])
        r = bm * rm + bs * (frames[e].close.pct_change().fillna(0).values - rm) + idio
        frames[sym] = to_frame(r, 50.0 + 100 * rng.random())
        betas[sym] = (bm, bs)
        sec_of[sym] = sec
    os.makedirs(D.bars_dir("sip"), exist_ok=True)
    for s, f in frames.items():
        f.to_parquet(D.bars_path(s, "sip"))
    months = pd.date_range(days[0].to_period("M").to_timestamp(), "2031-12-01", freq="MS")
    stocks = [s for s in frames if s not in D.ETFS]
    u = pd.DataFrame([dict(month=m, symbol=s, rank=k + 1, adv=1e9 - k, sector=sec_of[s]) for m in months for k, s in enumerate(stocks)])
    u.to_parquet(D.UNIVERSE_FILE)
    pd.DataFrame({"symbol": list(sec_of), "sector": list(sec_of.values())}).to_parquet(D.SECTOR_FILE)
    pd.DataFrame([dict(symbol=s, name=s, exchange="NYSE", tradable=True, shortable=True, etb=(s != stocks[0]), fractionable=True) for s in frames]).to_parquet(D.ASSETS_FILE)
    return dict(frames=frames, betas=betas, sectors=sec_of, days=days, ts=ts, stocks=stocks)
