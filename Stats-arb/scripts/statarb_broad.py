"""
Survivorship-free, point-in-time stock universe for the stat-arb model (no Qt).

Alpaca keeps daily bars for delisted tickers (SIVB, FRC, BBBY, TWTR ... all come back), so instead of "today's large caps" the universe can be every U.S. common stock that was ever liquid,
and the model's own liquidity filter (price >= $5, 20-session median dollar volume >= min_adv, evaluated each day with data up to that day) decides what is tradable when. That removes the
survivorship bias that flatters any reversal strategy's long leg, and it is the universe the Avellaneda-Lee paper used (stocks above ~USD 1bn market cap, ~1,000+ names).

  python statarb_broad.py            download / top up the broad cache (resumable; ~15-25 min the first time)
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import pandas as pd

import statarb_data as D

ASSETS_FILE = os.path.join(D.DATA_DIR, "statarb_assets.parquet")
BROAD_FILE = os.path.join(D.DATA_DIR, "statarb_daily_broad.parquet")
MONTHLY_FILE = os.path.join(D.DATA_DIR, "statarb_monthly_screen.parquet")
SECTOR_FILE = os.path.join(D.DATA_DIR, "statarb_sectors.parquet")
_JUNK = (r"\b(?:ETF|ETN|Fund|Funds|Warrants?|Rights?|Units?|Notes?|Preferred|Pfd|Debentures?|Subordinated|Acquisition Corp|Acquisition Corporation|Series [A-Z]|"
         r"Depositary Shares|Index|Bitcoin|Ethereum|Leveraged|Inverse|2X|3X)\b|%")


def list_assets(refresh: bool = False) -> pd.DataFrame:
    """Every U.S. equity Alpaca has ever listed on NYSE / Nasdaq / Amex, active and inactive, minus ETFs, warrants, units, preferreds and share-class tickers with punctuation."""
    if os.path.exists(ASSETS_FILE) and not refresh:
        df = pd.read_parquet(ASSETS_FILE)
    else:
        from alpaca.trading.client import TradingClient
        k, s = D.creds(True)
        tc = TradingClient(k, s, paper=True)
        rows = []
        for st in ("active", "inactive"):
            for x in tc.get("/assets", {"status": st, "asset_class": "us_equity"}):
                rows.append(dict(symbol=x["symbol"], name=x.get("name"), exchange=x.get("exchange"), status=x.get("status"), tradable=x.get("tradable"), shortable=x.get("shortable"),
                                 etb=x.get("easy_to_borrow"), marginable=x.get("marginable"), fractionable=x.get("fractionable")))
        df = pd.DataFrame(rows)
        os.makedirs(D.DATA_DIR, exist_ok=True)
        df.to_parquet(ASSETS_FILE)
    keep = df.exchange.isin(["NYSE", "NASDAQ", "AMEX"]) & ~df.name.fillna("").str.contains(_JUNK, regex=True) & df.symbol.str.fullmatch(r"[A-Z]{1,5}")
    return df[keep | df.symbol.isin(["BRK.B", "BF.B"])].reset_index(drop=True)


def _get_bars(client, syms: list, tf, start: str, adjustment: str) -> pd.DataFrame:
    import re
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=D.SIP_DELAY_MIN)
    syms = list(syms)
    for feed in (DataFeed.SIP, DataFeed.IEX):
        for attempt in range(8):
            if not syms:
                return pd.DataFrame()
            try:
                df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms, timeframe=tf, start=pd.Timestamp(start, tz="UTC").to_pydatetime(), end=end.to_pydatetime(),
                                                            adjustment=Adjustment(adjustment), feed=feed)).df
                return df.reset_index() if len(df) else pd.DataFrame()
            except Exception as e:                                                     # noqa: BLE001
                msg = str(e).lower()
                bad = re.search(r"invalid symbol: ([^\"\s]+)", str(e))
                if bad and bad.group(1) in syms:                                       # one unknown ticker fails the whole request - drop it and retry
                    syms.remove(bad.group(1))
                    continue
                if "429" in msg or "rate" in msg or "timeout" in msg or "connection" in msg:
                    time.sleep(3 * (attempt + 1))
                    continue
                if feed == DataFeed.SIP and ("subscription" in msg or "403" in msg):
                    break
                time.sleep(1)
    return pd.DataFrame()


def screen_liquid_symbols(min_adv: float = 8e6, min_months: int = 4, log: Callable[[str], None] = print, workers: int = 4) -> list:
    """Monthly bars for every listed / delisted candidate -> the names that were ever liquid (average dollar volume per session >= min_adv for >= min_months months)."""
    from alpaca.data.timeframe import TimeFrame
    syms = sorted(list_assets().symbol.unique())
    if os.path.exists(MONTHLY_FILE):
        m = pd.read_parquet(MONTHLY_FILE)
    else:
        client = D.data_client(True)
        chunks = [syms[i:i + 100] for i in range(0, len(syms), 100)]
        log(f"screening {len(syms)} symbols with monthly bars ({len(chunks)} requests) ...")
        frames = []
        with ThreadPoolExecutor(workers) as ex:
            for i, df in enumerate(ex.map(lambda ch: _get_bars(client, ch, TimeFrame.Month, D.CACHE_START, "raw"), chunks)):
                if len(df):
                    frames.append(df[["symbol", "timestamp", "close", "volume"]])
                if (i + 1) % 10 == 0:
                    log(f"  monthly {i + 1}/{len(chunks)}")
        m = pd.concat(frames, ignore_index=True)
        m.to_parquet(MONTHLY_FILE)
    m = m.assign(dv=m.close * m.volume / 21.0)
    ok = m[m.dv >= min_adv].groupby("symbol").size()
    keep = sorted(ok[ok >= min_months].index)
    log(f"screen: {len(keep)} of {m.symbol.nunique()} symbols were liquid (>= ${min_adv / 1e6:.0f}M/day) for >= {min_months} months")
    return keep


def update_broad_cache(symbols: Optional[list] = None, log: Callable[[str], None] = print, workers: int = 4) -> pd.DataFrame:
    """Daily split+dividend-adjusted bars for the screened symbols. Resumable AND incremental: symbols already cached are topped up from a 10-day overlap (and re-downloaded in full if the
    overlap shows that a split / dividend rewrote their adjusted history); new symbols are downloaded in full."""
    from alpaca.data.timeframe import TimeFrame
    symbols = list(symbols or screen_liquid_symbols(log=log))
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    old = pd.read_parquet(BROAD_FILE) if os.path.exists(BROAD_FILE) else pd.DataFrame(columns=cols)
    have = set(old.symbol.unique()) if len(old) else set()
    client = D.data_client(True)

    def fetch(ch: list, start: str) -> pd.DataFrame:
        df = _get_bars(client, ch, TimeFrame.Day, start, "all")
        if not len(df):
            return df
        df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        return df[cols]

    def run(chs: list, start: str, label: str) -> list:
        out = []
        with ThreadPoolExecutor(workers) as ex:
            for i, df in enumerate(ex.map(lambda ch: fetch(ch, start), chs)):
                if len(df):
                    out.append(df)
                if (i + 1) % 5 == 0:
                    log(f"  {label} {i + 1}/{len(chs)}")
        return out

    frames = [old] if len(old) else []
    if len(old):                                                                       # 1. top up the symbols we already have (only those that still trade can have new bars)
        last = old.date.max()
        alive = old.groupby("symbol").date.max()
        alive = [s for s in symbols if s in have and alive.get(s, pd.Timestamp(0)) >= last - pd.Timedelta(days=45)]
        start = (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        log(f"topping up {len(alive)} symbols from {start} ...")
        new = run([alive[i:i + 80] for i in range(0, len(alive), 80)], start, "top-up")
        if new:
            nw = pd.concat(new, ignore_index=True)
            o = old.set_index(["symbol", "date"]).close
            n = nw.set_index(["symbol", "date"]).close
            common = o.index.intersection(n.index)
            redo = []
            if len(common):
                rel = (n.reindex(common) / o.reindex(common) - 1).abs().groupby(level=0).max()
                redo = list(rel[rel > 5e-4].index)
            if redo:
                log(f"  {len(redo)} symbols had their adjusted history rewritten - re-downloading them in full")
                full = run([redo[i:i + 40] for i in range(0, len(redo), 40)], D.CACHE_START, "redo")
                frames = [old[~old.symbol.isin(redo)]]
                nw = nw[~nw.symbol.isin(redo)]
                frames += full
            frames.append(nw)
    todo = [s for s in symbols if s not in have]
    log(f"broad daily bars: {len(have)} cached, {len(todo)} new to download")
    if todo:
        frames += run([todo[i:i + 40] for i in range(0, len(todo), 40)], D.CACHE_START, "daily")
        if len(frames) > 1:
            pd.concat(frames, ignore_index=True).to_parquet(BROAD_FILE)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    etf = D.load_cache()
    etf = etf[etf.symbol.isin(D.FACTOR_ETFS)][cols]
    out = pd.concat([out[~out.symbol.isin(D.FACTOR_ETFS)], etf], ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    out.to_parquet(BROAD_FILE)
    log(f"broad cache: {out.symbol.nunique()} symbols through {out.date.max():%Y-%m-%d}, {len(out):,} rows")
    return out


def derive_sectors(long: pd.DataFrame) -> pd.Series:
    """Sector label for every stock = the sector SPDR its SPY-excess returns correlate with most (the curated labels are kept where we have them). Only used to group names."""
    px = long.pivot(index="date", columns="symbol", values="close").sort_index()
    r = px.pct_change(fill_method=None).clip(-0.3, 0.3)
    etf = r[list(D.SECTOR_ETF.values())].sub(r["SPY"], axis=0)
    stocks = [c for c in r.columns if c not in D.FACTOR_ETFS]
    ex = r[stocks].sub(r["SPY"], axis=0)
    import numpy as np
    a = ex.to_numpy(float)
    b = etf.to_numpy(float)
    ma, mb = np.isfinite(a).astype(float), np.isfinite(b).astype(float)
    a0, b0 = np.nan_to_num(a), np.nan_to_num(b)
    n = ma.T @ mb
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a, mean_b = (a0.T @ mb) / n, (ma.T @ b0) / n
        cov = (a0.T @ b0) / n - mean_a * mean_b
        va = ((a0 ** 2).T @ mb) / n - mean_a ** 2
        vb = (ma.T @ (b0 ** 2)) / n - mean_b ** 2
        corr = pd.DataFrame(cov / np.sqrt(va * vb), index=ex.columns, columns=etf.columns).where(n >= 120)
    inv = {v: k for k, v in D.SECTOR_ETF.items()}
    sec = corr.idxmax(axis=1).map(inv)
    for s in stocks:
        if s in D.SECTOR_OF:
            sec[s] = D.SECTOR_OF[s]
    sec = sec.fillna("Tech")
    pd.DataFrame({"symbol": sec.index, "sector": sec.values}).to_parquet(SECTOR_FILE)
    return sec


def load_broad_panel(log: Callable[[str], None] = print, min_adv: float = 1e7) -> D.Panel:
    if not os.path.exists(BROAD_FILE):
        raise FileNotFoundError("broad cache missing - run:  python statarb_broad.py")
    long = pd.read_parquet(BROAD_FILE)
    sec = pd.read_parquet(SECTOR_FILE).set_index("symbol").sector if os.path.exists(SECTOR_FILE) else derive_sectors(long)
    # drop names that were never liquid enough to matter (keeps the arrays small); the model's own point-in-time filter still decides who is tradable on each date
    dv = long.assign(dv=long.close * long.volume).pivot(index="date", columns="symbol", values="dv").rolling(20, min_periods=15).median().max()
    keep = [s for s in dv.index if s in sec.index and s not in D.FACTOR_ETFS and dv[s] >= 0.7 * min_adv]
    return D.Panel(long, sector_map=sec, stocks=keep)


if __name__ == "__main__":
    update_broad_cache()
    long = pd.read_parquet(BROAD_FILE)
    derive_sectors(long)
    p = load_broad_panel()
    print(p.describe())
