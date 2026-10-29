"""
Research tool (REPORT.md section 6): historical SIP quotes of the book's picks at market-order times - the ask before the close for the buys, the bid before the close for
the sales there, the bid after the open for the overnight sales - to measure what market orders in place of the auctions would have cost. The app's backtest prices the
auctions (the model); the live engine's fills are compared with those prices. live.py uses _raw_daily() for the raw official prices.

Quotes are raw prices and the backtest's are split / dividend adjusted, so each quote is kept as a RATIO to the raw official close / open of its session and the backtest
fills at  adjusted official price x ratio.  The SIP quote taken is the last one at or before the time (a 1-second window; names without a quote in it: the last one since
09:30:00 for the open - never a pre-open quote - and in the 30 seconds before for the close).

  <data>/mom_quotes.parquet   date, symbol, at ('close-10' = 10 s before the close, 'open+10' = 10 s after the open), bid, ask, quote_ts, ref (raw official price)
  python -m resarb.mom.quotes <app folder> [--close-s 10] [--open-s 10]
        fetches the missing quotes of every day's picks of the presets' rankings at k = 5 .. 20 (and QQQ's closing sales) and of the live book's journaled fills -
        read-only data calls, resumable; run it after 'Update data' to extend the backtest and the live-vs-backtest comparison to the newest sessions
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .. import data as D

QUOTES_FILE = os.path.join(D.DATA_DIR, "mom_quotes.parquet")
COLS = ["date", "symbol", "at", "bid", "ask", "quote_ts", "ref"]
_CACHE: dict = {}


def close_tag(s: int) -> str:
    return f"close-{int(s)}"


def open_tag(s: int) -> str:
    return f"open{int(s):+d}"


def load() -> Optional[pd.DataFrame]:
    """The cached quotes (None when there are none). Cached per file modification time."""
    if not os.path.exists(QUOTES_FILE):
        return None
    mt = os.path.getmtime(QUOTES_FILE)
    hit = _CACHE.get(QUOTES_FILE)
    if hit and hit[0] == mt:
        return hit[1]
    df = pd.read_parquet(QUOTES_FILE)
    df["date"] = pd.to_datetime(df.date).dt.normalize()
    _CACHE[QUOTES_FILE] = (mt, df)
    return df


def quote_open_s(open_s: int) -> int:
    """The quote that prices a sale sent at the open + ``open_s``: an order sent before 09:30:10 (before the open it is queued for it) is priced at the 09:30:10 bid -
    paper fills it at the first quotes after the open, and those of the first seconds are erratic."""
    return max(int(open_s), 10)


def ratio_table(close_s: int, open_s: int) -> Optional[dict]:
    """{(date, symbol): ratio} for 'close_ask', 'close_bid' (quote / raw official close) and 'open_bid' (quote / raw official open) at the given times; None without data."""
    q = load()
    if q is None or not len(q):
        return None
    q = q[(q.ref > 0) & (q.bid > 0) & (q.ask >= q.bid)]
    c, o = q[q["at"] == close_tag(close_s)], q[q["at"] == open_tag(quote_open_s(open_s))]
    if not len(c) and not len(o):
        return None
    key = lambda f: list(zip(f.date, f.symbol))
    return {"close_ask": dict(zip(key(c), c.ask / c.ref)), "close_bid": dict(zip(key(c), c.bid / c.ref)), "open_bid": dict(zip(key(o), o.bid / o.ref))}


def ratios(mk, close_s: int, open_s: int) -> Optional[dict]:
    """The ratio table as T x N arrays aligned to the market (NaN where no quote); None without data."""
    tab = ratio_table(close_s, open_s)
    if tab is None:
        return None
    T, N = len(mk.dates), len(mk.syms)
    ti = {d: i for i, d in enumerate(pd.DatetimeIndex(mk.dates).normalize())}
    out = {}
    for name, m in tab.items():
        a = np.full((T, N), np.nan)
        for (d, s), r in m.items():
            i, j = ti.get(d), mk.col.get(s)
            if i is not None and j is not None and np.isfinite(r):
                a[i, j] = r
        out[name] = a
    return out


# ── download ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _quotes(dc, syms: list, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest
    utc = lambda x: x.tz_localize(D.NY).tz_convert("UTC").to_pydatetime()
    for attempt in range(6):
        try:
            df = dc.get_stock_quotes(StockQuotesRequest(symbol_or_symbols=syms, start=utc(start), end=utc(end), feed=DataFeed.SIP)).df
            return df.reset_index() if len(df) else pd.DataFrame(columns=["symbol", "timestamp", "bid_price", "ask_price"])
        except Exception as e:                                                                # noqa: BLE001
            if any(k in str(e).lower() for k in ("429", "too many", "timeout", "connection", "502", "503", "504")):
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            raise
    return pd.DataFrame(columns=["symbol", "timestamp", "bid_price", "ask_price"])


def last_quotes(dc, syms: list, at: pd.Timestamp, lo: pd.Timestamp) -> list:
    """[(symbol, bid, ask, quote time)] of the last valid SIP quote at or before ``at``: a 1-second window, then [lo, at] for the names without one."""
    out, need = {}, list(syms)
    for start in (at - pd.Timedelta(seconds=1), lo):
        if not need:
            break
        df = _quotes(dc, need, start, at + pd.Timedelta(microseconds=1))
        if len(df):
            df = df[(df.bid_price > 0) & (df.ask_price >= df.bid_price)]
            for s, g in df.groupby("symbol"):
                x = g.iloc[-1]
                out[s] = (float(x.bid_price), float(x.ask_price), pd.Timestamp(x.timestamp).tz_convert(D.NY).tz_localize(None))
        need = [s for s in need if s not in out]
        if start == lo:
            break
    return [(s, *v) for s, v in out.items()]


def _raw_daily(dc, syms: list, start, end) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    frames = []
    end = min(pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC"),
              pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=D.SIP_DELAY_MIN + 4))                    # the free plan: no SIP data of the last 15 minutes
    start = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    for i in range(0, len(syms), 100):
        req = StockBarsRequest(symbol_or_symbols=syms[i:i + 100], timeframe=TimeFrame.Day, start=start.to_pydatetime(), end=end.to_pydatetime(),
                               adjustment=Adjustment.RAW, feed=DataFeed.SIP)
        for attempt in range(5):
            try:
                df = dc.get_stock_bars(req).df
                break
            except Exception as e:                                                            # noqa: BLE001
                if attempt == 4 or not any(k in str(e).lower() for k in ("429", "too many", "timeout", "connection", "502", "503", "504")):
                    raise
                time.sleep(min(5 * (attempt + 1), 30))
        if len(df):
            df = df.reset_index()
            df["date"] = pd.to_datetime(df.timestamp).dt.tz_convert(D.NY).dt.tz_localize(None).dt.normalize()
            frames.append(df[["symbol", "date", "open", "close"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["symbol", "date", "open", "close"])


def fetch(jobs: list, log: Callable[[str], None] = print, workers: int = 3, per_min: int = 180, prefix: str = "MB") -> pd.DataFrame:
    """``jobs`` = [(session date, 'close' | 'open', at, lo, symbols)]: downloads the missing quotes, adds the raw official price, saves every 100 jobs. Returns the cache."""
    from ..live import Throttle
    dc = D.data_client(prefix)
    have = load()
    ok = have[have.ref.notna()] if have is not None else None                                  # a row saved without its official price is fetched again
    done = set(zip(ok.date, ok.symbol, ok["at"])) if ok is not None else set()
    todo = []
    for d, side, at, lo, syms, tag in jobs:
        need = [s for s in syms if (pd.Timestamp(d), s, tag) not in done]
        if need:
            todo.append((d, side, at, lo, need, tag))
    log(f"{len(todo):,} of {len(jobs):,} quote snapshots to download ({sum(len(j[4]) for j in todo):,} name-times)")
    thr, lock, rows, t0 = Throttle(per_min), threading.Lock(), [], time.time()

    def one(job):
        d, side, at, lo, syms, tag = job
        res = []
        for i in range(0, len(syms), 50):
            thr.wait()
            res += [dict(date=pd.Timestamp(d), symbol=s, at=tag, bid=b, ask=a, quote_ts=ts) for s, b, a, ts in last_quotes(dc, syms[i:i + 50], at, lo) if lo <= ts <= at]
        return res

    def save(new):
        if not new:
            return
        nd = pd.DataFrame(new)
        refs = _raw_daily(dc, sorted(set(nd.symbol)), nd.date.min() - pd.Timedelta(days=3), nd.date.max() + pd.Timedelta(days=2))
        ref = {(s, pd.Timestamp(t)): (o, c) for s, t, o, c in zip(refs.symbol, refs.date, refs.open, refs.close)}
        nd["ref"] = [ref.get((s, d), (np.nan, np.nan))[0 if a.startswith("open") else 1] for s, d, a in zip(nd.symbol, nd.date, nd["at"])]
        cur = load()
        out = pd.concat([cur, nd[COLS]], ignore_index=True) if cur is not None else nd[COLS]
        out = out.drop_duplicates(["date", "symbol", "at"], keep="last").sort_values(["date", "at", "symbol"]).reset_index(drop=True)
        tmp = QUOTES_FILE + ".tmp"
        out.to_parquet(tmp)
        os.replace(tmp, QUOTES_FILE)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k in range(0, len(todo), 100):
            batch = todo[k:k + 100]
            got = [r for res in ex.map(one, batch) for r in res]
            with lock:
                save(got)
            log(f"  {min(k + 100, len(todo)):,} / {len(todo):,} snapshots, {time.time() - t0:.0f} s")
    return load()


def jobs_for(trades: pd.DataFrame, closes: pd.Series, close_s: int, open_s: int, intraday_symbol: str = "QQQ") -> list:
    """The snapshots a backtest's trades need: each entry day's close (its buys and the intraday symbol's closing sale) and each exit day's open (its overnight sales)."""
    on = trades[trades.leg == "OVERNIGHT"]
    it = trades[(trades.leg == "INTRADAY") & trades.exit_how.astype(str).str.contains("clos")]               # "closing auction" / "market at the close"
    out = []
    by_close: dict = {}
    for d, s in zip(pd.to_datetime(on.entry_date), on.symbol):
        by_close.setdefault(d.normalize(), set()).add(s)
    for d in pd.to_datetime(it.exit_date):
        by_close.setdefault(d.normalize(), set()).add(intraday_symbol)
    for d, syms in by_close.items():
        c = closes.get(d)
        if c is None:
            continue
        at = c - pd.Timedelta(seconds=close_s)
        out.append((d, "close", at, at - pd.Timedelta(seconds=30), sorted(syms), close_tag(close_s)))
    by_open: dict = {}
    for d, s in zip(pd.to_datetime(on.exit_date.dropna()), on[on.exit_date.notna()].symbol):
        by_open.setdefault(d.normalize(), set()).add(s)
    for d, syms in by_open.items():
        op = d + pd.Timedelta(hours=9, minutes=30)
        at = op + pd.Timedelta(seconds=open_s)
        out.append((d, "open", at, op, sorted(syms), open_tag(open_s)))
    return out


def jobs_from_journal(folder: str, closes: pd.Series, close_s: int, open_s: int) -> list:
    """The snapshots the live book's own fills need for their comparison with the backtest (journaled 'fill' rows of the paper and live journals)."""
    import json
    by: dict = {}
    for mode in ("paper", "live"):
        p = os.path.join(folder, "reports", f"mb_journal_{mode}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("kind") != "fill" or not float(r.get("filled_qty") or 0):
                    continue
                suf = str(r.get("cid", "")).split("-")[-1]
                side = "open" if suf[:1] == "o" else ("close" if suf[:1] in ("e", "x") or suf[:2] == "iC" else None)
                if side:
                    by.setdefault((pd.Timestamp(r["day"]).normalize(), side), set()).add(r["symbol"])
    out = []
    for (d, side), syms in by.items():
        if side == "close" and closes.get(d) is not None:
            at = closes.get(d) - pd.Timedelta(seconds=close_s)
            out.append((d, "close", at, at - pd.Timedelta(seconds=30), sorted(syms), close_tag(close_s)))
        elif side == "open":
            op = d + pd.Timedelta(hours=9, minutes=30)
            out.append((d, "open", op + pd.Timedelta(seconds=open_s), op, sorted(syms), open_tag(open_s)))
    return out


def main(argv=None) -> int:
    """python -m resarb.mom.quotes <app folder>: the missing quotes of every day's picks (rule + ML and rule, k = 5 .. 20) at the given times."""
    import argparse
    from . import engine as E
    from . import presets as PR
    from .market import closes as session_closes
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--close-s", type=int, default=10)
    ap.add_argument("--open-s", type=int, default=10)
    ap.add_argument("--start", default="2018-01-02")
    a = ap.parse_args(argv)
    if a.open_s < 0:
        raise SystemExit("an order sent before the open fills at the first quotes after it: use --open-s 0 or more")
    tr = PR.get(PR.TRADED)
    mk = E.load_market(tr, log=lambda s: print(s, flush=True))
    t0 = int(np.searchsorted(mk.dates, pd.Timestamp(a.start)))
    rows = range(max(t0, 1), len(mk.dates))
    preps = {"rule+ml": E.prepare(tr, mk, a.folder, rows=rows), "rule": E.prepare(PR.get(PR.SHADOW), mk, rows=rows)}
    trades = []
    for ranking, P in preps.items():
        for k in (5, 10, 15, 20):
            m = PR.get(PR.TRADED if ranking == "rule+ml" else PR.SHADOW)
            m = type(m).from_dict({**m.to_dict(), "k": k})
            trades.append(E.run(m, mk, a.folder, start=a.start, prep=P).trades)
            print(f"picks: {ranking} k={k}", flush=True)
    tr_all = pd.concat(trades, ignore_index=True)
    cl = session_closes(mk.dates[0], mk.dates[-1] + pd.Timedelta(days=1))
    jobs = jobs_for(tr_all, cl, a.close_s, a.open_s, tr.intraday_symbol) + jobs_from_journal(a.folder, cl, a.close_s, a.open_s)
    fetch(jobs, log=lambda s: print(s, flush=True))
    q = load()
    print(f"cache: {len(q):,} quotes, {q.date.min():%Y-%m-%d} .. {q.date.max():%Y-%m-%d}; missing raw official price: {q.ref.isna().mean():.2%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
