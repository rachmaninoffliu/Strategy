"""
Market data for the residual stat-arb strategies (no Qt).

  * 15-minute regular-hours bars (09:30 ... 15:45 bar starts, 26 a day, 14 on half days) from Alpaca, split + dividend adjusted, one parquet file per symbol under data/bars15 (SIP) and
    data/bars15_iex (IEX, the feed the free plan gives in real time). Coarser time frames (30 min, 1 h, 1 day) are built from these.
  * a point-in-time universe: at every month end the N most liquid U.S. common stocks (63-session median dollar volume, price >= $10, >= 126 sessions of history) become the members for the
    next month. History comes from the survivorship-free daily cache of the Stats-arb project (delisted names included); later months are screened live from Alpaca daily bars.
  * credentials: the project's key file "Alpaca Key.txt" (alpaca_keys.py); a strategy's own <PREFIX>_ALPACA_API_KEY / _SECRET_KEY variables override it (a separate account).

  python -m resarb.data universe        build the monthly universe 2016 -> now
  python -m resarb.data download        download / top up the SIP 15-minute bars of every member + the factor ETFs (resumable)
  python -m resarb.data iex             IEX 15-minute bars since 2024 (to measure the live-feed gap)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))           # the Residual-StatArb folder (core/resarb/ sits two levels below it)
DATA_DIR = os.environ.get("RESARB_DATA_DIR", os.path.join(ROOT, "data"))
STATARB_DATA = os.environ.get("RESARB_STATARB_DATA", os.path.join(os.path.dirname(ROOT), "Stats-arb", "data"))
UNIVERSE_FILE = os.path.join(DATA_DIR, "resarb_universe.parquet")
SECTOR_FILE = os.path.join(DATA_DIR, "resarb_sectors.parquet")
ASSETS_FILE = os.path.join(DATA_DIR, "resarb_assets.parquet")
START = "2016-01-01"
SIP_DELAY_MIN = 16                                                                           # the free plan refuses SIP data from the last 15 minutes
NY = "America/New_York"
BAR_MIN = 15

SECTOR_ETF = {"Tech": "XLK", "Comm": "XLC", "Disc": "XLY", "Stap": "XLP", "Energy": "XLE", "Fin": "XLF", "Health": "XLV", "Ind": "XLI", "Mat": "XLB", "Util": "XLU", "RE": "XLRE"}
MARKET = "SPY"
ETFS = ["SPY", "QQQ", "IWM"] + list(SECTOR_ETF.values())
DUP_CLASS = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA", "UA": "UAA", "LBTYK": "LBTYA", "LBTYB": "LBTYA", "DISCK": "DISCA", "ZG": "Z", "BF.A": "BF.B", "LEN.B": "LEN",
             "HEI.A": "HEI", "BRK.A": "BRK.B", "CMCSK": "CMCSA", "VIAC": "PARA", "QRTEB": "QRTEA"}        # keep one share class (two classes have the same residual)


def bars_dir(feed: str = "sip") -> str:
    d = os.path.join(DATA_DIR, "bars15" if feed == "sip" else "bars15_iex")
    os.makedirs(d, exist_ok=True)
    return d


# ── credentials ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def creds(prefix: Optional[str] = None, paper: bool = True) -> tuple[str, str]:
    """Keys for one strategy's account: <PREFIX>_ALPACA_API_KEY + <PREFIX>_ALPACA_SECRET_KEY if set (a separate account for that strategy; live: <PREFIX>_ALPACA_LIVE_...),
    otherwise the project's key file "Alpaca Key.txt" in the project folder, then ALPACA_API_KEY / ALPACA_SECRET_KEY, then ~/.claude.json (see alpaca_keys.py)."""
    if prefix:
        tag = "" if paper else "LIVE_"
        k, s = os.environ.get(f"{prefix}_ALPACA_{tag}API_KEY"), os.environ.get(f"{prefix}_ALPACA_{tag}SECRET_KEY")
        if k and s:
            return k, s
    from . import alpaca_keys
    return alpaca_keys.creds(paper)


def own_keys(prefix: str, paper: bool = True) -> bool:
    """True when the keys are the project's own: the strategy's <PREFIX>_ variables, the project's key file or the ALPACA_* variables - not the ~/.claude.json fallback of the
    Claude tools. (An account can still be claimed by one strategy only: see live.claim_account.)"""
    tag = "" if paper else "LIVE_"
    if os.environ.get(f"{prefix}_ALPACA_{tag}API_KEY") and os.environ.get(f"{prefix}_ALPACA_{tag}SECRET_KEY"):
        return True
    from . import alpaca_keys
    return alpaca_keys.source(paper) in ("file", "env")


def key_source(prefix: str, paper: bool = True) -> str:
    """'strategy env', 'key file', 'environment variables', '~/.claude.json' or 'no keys' - for status lines (never the keys)."""
    tag = "" if paper else "LIVE_"
    if os.environ.get(f"{prefix}_ALPACA_{tag}API_KEY") and os.environ.get(f"{prefix}_ALPACA_{tag}SECRET_KEY"):
        return f"{prefix}_ALPACA_* variables"
    from . import alpaca_keys
    return alpaca_keys.describe(paper)


def harden(client, connect: float = 5.0, read: float = 30.0):
    """alpaca-py makes its HTTP calls without a timeout: one stalled connection would freeze the engine for ever. Give the client's session a default (connect, read) timeout."""
    import requests
    from requests.adapters import HTTPAdapter

    class TimeoutAdapter(HTTPAdapter):
        def send(self, request, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (connect, read)
            return super().send(request, **kwargs)
    sess = getattr(client, "_session", None)
    if isinstance(sess, requests.Session):
        sess.mount("https://", TimeoutAdapter())
        sess.mount("http://", TimeoutAdapter())
    return client


def data_client(prefix: Optional[str] = None):
    from alpaca.data.historical import StockHistoricalDataClient
    k, s = creds(prefix, True)
    return harden(StockHistoricalDataClient(k, s))


def trading_client(prefix: Optional[str], paper: bool = True):
    from alpaca.trading.client import TradingClient
    k, s = creds(prefix, paper)
    return harden(TradingClient(k, s, paper=paper))


# ── 15-minute bars ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
BAR_COLS = ["open", "high", "low", "close", "volume", "vwap", "trades"]


def rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the regular-hours bars: weekday, bar start 09:30 ... 15:45 ET."""
    t = df.index
    m = (t.weekday < 5) & (((t.hour == 9) & (t.minute >= 30)) | ((t.hour >= 10) & (t.hour < 16)))
    return df[m]


def _to_frame(raw: pd.DataFrame) -> dict:
    """alpaca-py bars frame (MultiIndex symbol, timestamp UTC) -> {symbol: frame indexed by the ET bar start (naive), regular hours only}."""
    if raw is None or not len(raw):
        return {}
    raw = raw.reset_index()
    raw["ts"] = pd.to_datetime(raw["timestamp"]).dt.tz_convert(NY).dt.tz_localize(None)
    raw = raw.rename(columns={"trade_count": "trades"})
    out = {}
    for sym, g in raw.groupby("symbol"):
        f = g.set_index("ts")[[c for c in BAR_COLS if c in g.columns]].sort_index()
        out[sym] = rth(f[~f.index.duplicated(keep="last")]).astype("float64")
    return out


def fetch_bars(client, symbols: list, start, end=None, feed: str = "sip", minutes: int = BAR_MIN) -> dict:
    """15-minute (or ``minutes``) bars for a few symbols between ``start`` and ``end`` (default: 16 minutes ago, the SIP delay rule; IEX may run up to now). Retries rate limits and drops a
    symbol Alpaca calls invalid (one bad ticker fails the whole multi-symbol request)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    utc = lambda x: (pd.Timestamp(x).tz_localize(NY) if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x)).tz_convert("UTC")
    if end is None:
        end = pd.Timestamp.now(tz="UTC") - (pd.Timedelta(minutes=SIP_DELAY_MIN) if feed == "sip" else pd.Timedelta(0))
    syms = list(symbols)
    last = None
    for attempt in range(10):
        if not syms:
            return {}
        try:
            req = StockBarsRequest(symbol_or_symbols=syms, timeframe=TimeFrame(minutes, TimeFrameUnit.Minute), start=utc(start).to_pydatetime(), end=utc(end).to_pydatetime(),
                                   adjustment=Adjustment.ALL, feed=DataFeed.IEX if feed == "iex" else DataFeed.SIP, limit=50_000_000)    # a total limit makes alpaca-py ask for 10,000-bar pages (else 1,000)
            return _to_frame(client.get_stock_bars(req).df)
        except Exception as e:                                                               # noqa: BLE001
            last = e
            msg = str(e).lower()
            bad = re.search(r"invalid symbol: ([^\"\s,]+)", str(e))
            if bad and bad.group(1) in syms:
                syms.remove(bad.group(1))
                continue
            if any(k in msg for k in ("429", "too many", "rate", "timeout", "timed out", "connection", "502", "503", "504")):
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            raise
    raise RuntimeError(f"bars request failed after retries: {type(last).__name__}: {str(last)[:160]}")


def bars_path(sym: str, feed: str = "sip") -> str:
    return os.path.join(bars_dir(feed), f"{sym.replace('.', '_')}.parquet")


def load_bars(sym: str, feed: str = "sip", fix: bool = True) -> pd.DataFrame:
    p = bars_path(sym, feed)
    if not os.path.exists(p):
        return pd.DataFrame(columns=BAR_COLS, dtype="float64")
    f = pd.read_parquet(p)
    return apply_fixes(sym, f) if fix else f


# ── unadjusted distributions ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
FIX_FILE = os.path.join(DATA_DIR, "resarb_price_fixes.parquet")
_FIXES: Optional[pd.DataFrame] = None
# distributions Alpaca's corporate-action feed does not list (checked by hand against the bars: a gap down on the ex-date with no market move to explain it)
MANUAL_FIXES = [("JCI", "2016-10-31", "Adient spin-off (1 ADNT per 10 JCI)"),
                ("RTX", "2020-04-03", "UTX distributed Carrier + Otis, merged with Raytheon, renamed RTX"),
                ("FOXA", "2019-03-19", "21st Century Fox sold assets to Disney; the ticker continues as the new Fox Corp")]
# erroneous opening prints: the NYSE opening-auction failure of 2023-01-24 (hundreds of stocks opened at wrong prices, many trades later busted) - the first bar's open
# becomes its VWAP (the fill price the backtest uses must be one that could really be traded)
BAD_OPENS = ["2023-01-24 09:30"]


def apply_fixes(sym: str, f: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust the prices before an ex-date that Alpaca left unadjusted (a spin-off / an ETF's in-kind distribution): prices before the date x factor."""
    global _FIXES
    if _FIXES is None:
        _FIXES = pd.read_parquet(FIX_FILE) if os.path.exists(FIX_FILE) else pd.DataFrame(columns=["symbol", "ex_date", "factor"])
    if not len(f):
        return f
    bad = f.index.isin(pd.DatetimeIndex(BAD_OPENS))
    fx = _FIXES[_FIXES.symbol == sym]
    if not len(fx) and not bad.any():
        return f
    f = f.copy()
    if bad.any() and "vwap" in f:
        f.loc[bad, "open"] = f.loc[bad, "vwap"]
        f.loc[bad, "high"] = f.loc[bad, ["vwap", "close"]].max(axis=1)
        f.loc[bad, "low"] = f.loc[bad, ["vwap", "close"]].min(axis=1)
    for r in fx.itertuples():
        m = f.index < pd.Timestamp(r.ex_date)
        for c in ("open", "high", "low", "close", "vwap"):
            if c in f:
                f.loc[m, c] = f.loc[m, c] * float(r.factor)
    return f


def build_price_fixes(symbols: list, log: Callable[[str], None] = print, etf_gap: float = 0.12, min_drop: float = 0.03) -> pd.DataFrame:
    """Find the distributions the adjusted bars still show as an overnight drop: (1) every spin-off Alpaca lists for the symbols (ex-date), kept only if the stock really gapped down more
    than ``min_drop`` beyond the market that morning (most modern spin-offs ARE adjusted - GE, DHR - older ones are not: JCI 2016, UTX/RTX 2020); (2) a hedge ETF whose overnight return
    differs from SPY's by more than ``etf_gap`` (XLF's 2016 distribution of XLRE). The factor removes the jump relative to SPY: (1 + r_stock) / (1 + r_SPY)."""
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.requests import CorporateActionsRequest
    k, s = creds()
    cc = CorporateActionsClient(k, s)
    spins = []
    for i in range(0, len(symbols), 50):
        ch = symbols[i:i + 50]
        try:
            r = cc.get_corporate_actions(CorporateActionsRequest(symbols=ch, start=pd.Timestamp(START).date(), end=pd.Timestamp.now().date(), limit=10_000))
        except Exception as e:                                                               # noqa: BLE001
            log(f"corporate actions {ch[0]}..: {type(e).__name__}: {str(e)[:100]}")
            continue
        d = r.dict() if hasattr(r, "dict") else r
        d = d.get("data", d)
        for x in d.get("spin_offs", []):
            spins.append((str(x.get("source_symbol")), pd.Timestamp(x.get("ex_date"))))
    spy = load_bars(MARKET, fix=False)
    sd = spy.groupby(spy.index.normalize()).agg(o=("open", "first"), c=("close", "last"))
    spy_on = sd.o / sd.c.shift() - 1
    rows = []

    def gap(sym: str, day: pd.Timestamp):
        x = load_bars(sym, fix=False)
        if not len(x):
            return None
        g = x.groupby(x.index.normalize()).agg(o=("open", "first"), c=("close", "last"))
        if day not in g.index:
            return None
        i = g.index.get_loc(day)
        if i == 0:
            return None
        return float(g.o.iloc[i] / g.c.iloc[i - 1] - 1), float(spy_on.get(day, 0.0))
    events = [(s_, d_, "spin-off (Alpaca corporate actions)") for s_, d_ in sorted(set(spins))] + [(s_, pd.Timestamp(d_), w_) for s_, d_, w_ in MANUAL_FIXES]
    seen = set()
    for sym, day, why in events:
        if sym not in symbols or (sym, day) in seen:
            continue
        seen.add((sym, day))
        gp = gap(sym, day)
        if gp is None:
            continue
        r, m = gp
        if (1 + r) / (1 + m) - 1 < -min_drop:
            rows.append(dict(symbol=sym, ex_date=day, factor=(1 + r) / (1 + m), why=f"{why}: unadjusted gap {r:+.1%} (SPY {m:+.1%})"))
    for e in ETFS:
        x = load_bars(e, fix=False)
        if not len(x) or e == MARKET:
            continue
        g = x.groupby(x.index.normalize()).agg(o=("open", "first"), c=("close", "last"))
        on = g.o / g.c.shift() - 1
        dev = (1 + on) / (1 + spy_on.reindex(on.index).fillna(0)) - 1
        for day, v in dev[dev < -etf_gap].items():
            rows.append(dict(symbol=e, ex_date=day, factor=1 + v, why=f"ETF distribution: overnight {on[day]:+.1%} vs SPY {spy_on.get(day, 0):+.1%}"))
    df = pd.DataFrame(rows, columns=["symbol", "ex_date", "factor", "why"])
    df.to_parquet(FIX_FILE)
    global _FIXES
    _FIXES = None
    for r in df.itertuples():
        log(f"  price fix {r.symbol} {r.ex_date:%Y-%m-%d}: x{r.factor:.4f} before ({r.why})")
    return df


def update_bars(symbols: list, feed: str = "sip", start: str = START, workers: int = 4, log: Callable[[str], None] = print, client=None, stop: Callable[[], bool] = lambda: False) -> dict:
    """Download or top up the 15-minute bars of ``symbols`` (resumable: one file per symbol). A cached symbol is topped up from a 7-day overlap; when the overlap shows that a split or a
    dividend rewrote its adjusted history, it is downloaded again in full. Returns {symbol: 'new' | 'topped up' | 'redone' | 'empty' | 'error ...'}."""
    client = client or data_client()
    out: dict = {}

    def one(sym: str) -> tuple:
        if stop():
            return sym, "stopped"
        try:
            old = load_bars(sym, feed, fix=False)                                              # the file stays raw; the fixes are applied when it is read
            if len(old):
                s0 = old.index.max() - pd.Timedelta(days=7)
                new = fetch_bars(client, [sym], s0, feed=feed).get(sym)
                if new is None or not len(new):
                    return sym, "no new bars"
                common = old.index.intersection(new.index)
                if len(common) and float((new.close.reindex(common) / old.close.reindex(common) - 1).abs().max()) > 5e-4:
                    full = fetch_bars(client, [sym], start, feed=feed).get(sym)
                    if full is not None and len(full):
                        full.to_parquet(bars_path(sym, feed))
                        return sym, "redone"
                df = pd.concat([old[old.index < new.index.min()], new])
                df.to_parquet(bars_path(sym, feed))
                return sym, "topped up"
            new = fetch_bars(client, [sym], start, feed=feed).get(sym)
            if new is None or not len(new):
                pd.DataFrame(columns=BAR_COLS, dtype="float64").to_parquet(bars_path(sym, feed))
                return sym, "empty"
            new.to_parquet(bars_path(sym, feed))
            return sym, "new"
        except Exception as e:                                                               # noqa: BLE001
            return sym, f"error {type(e).__name__}: {str(e)[:100]}"

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, (sym, st) in enumerate(ex.map(one, list(symbols))):
            out[sym] = st
            if st.startswith("error"):
                log(f"  {sym}: {st}")
            if (i + 1) % 10 == 0 or i + 1 == len(symbols):
                log(f"  {feed} bars {i + 1}/{len(symbols)} ({time.time() - t0:.0f} s)")
    return out


def _symbols_for_month(u: pd.DataFrame, month: pd.Timestamp, n: int) -> list:
    """Symbols whose bars are needed in ``month``: members of the previous month (to price leavers) up to 13 months ahead (the history a new member's betas and momentum need)."""
    lo, hi = month - pd.DateOffset(months=1), month + pd.DateOffset(months=13)
    m = u[(u.month >= lo) & (u.month <= hi) & (u["rank"] <= n)]
    if not len(m):
        m = u[(u.month == u.month.max()) & (u["rank"] <= n)]
    return sorted(set(m.symbol) | set(ETFS))


def download_by_day(feed: str = "sip", start: str = START, end: Optional[str] = None, n: int = 150, workers: int = 6, log: Callable[[str], None] = print,
                    stop: Callable[[], bool] = lambda: False) -> None:
    """Bulk history: one request per session for all the symbols needed that month, regular hours only (no extended-hours pages), many sessions in parallel. Results go to monthly
    chunk files first (resumable), then ``assemble`` writes one file per symbol."""
    u = load_universe()
    if not len(u):
        raise FileNotFoundError("build the universe first:  python -m resarb.data universe")
    cdir = os.path.join(DATA_DIR, f"chunks_{feed}")
    os.makedirs(cdir, exist_ok=True)
    client = data_client()
    last = pd.Timestamp(end) if end else pd.Timestamp.now(tz=NY).tz_localize(None)
    months = pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(), last, freq="MS")
    t0 = time.time()
    import threading
    lock, stamps = threading.Lock(), []

    def throttle(per_min: int = 160) -> None:                                                # the free plan allows 200 data requests a minute
        while True:
            with lock:
                now = time.monotonic()
                while stamps and now - stamps[0] > 60:
                    stamps.pop(0)
                if len(stamps) < per_min:
                    stamps.append(now)
                    return
                wait = 60 - (now - stamps[0]) + 0.05
            time.sleep(wait)
    for mo in months:
        if stop():
            return
        f = os.path.join(cdir, f"{mo:%Y-%m}.parquet")
        is_last = mo.to_period("M") == last.to_period("M")
        if os.path.exists(f) and not is_last:
            continue
        syms = _symbols_for_month(u, mo, n)
        days = [d for d in pd.bdate_range(mo, min(mo + pd.offsets.MonthEnd(0), last.normalize())) if d >= pd.Timestamp(start)]

        def one(d):
            out = {}
            for i in range(0, len(syms), 300):
                e = d + pd.Timedelta(hours=16)
                if feed == "sip":
                    e = min(e, pd.Timestamp.now(tz=NY).tz_localize(None) - pd.Timedelta(minutes=SIP_DELAY_MIN))
                if e <= d + pd.Timedelta(hours=9, minutes=30):
                    continue
                throttle()
                out.update(fetch_bars(client, syms[i:i + 300], d + pd.Timedelta(hours=9, minutes=30), e, feed=feed))
            return out
        frames = []
        with ThreadPoolExecutor(workers) as ex:
            for res in ex.map(one, days):
                for s, fr in res.items():
                    if len(fr):
                        frames.append(fr.assign(symbol=s))
        df = pd.concat(frames) if frames else pd.DataFrame(columns=BAR_COLS + ["symbol"])
        df.index.name = "ts"
        df.reset_index().to_parquet(f)
        log(f"  {feed} {mo:%Y-%m}: {len(syms)} symbols, {len(days)} sessions, {len(df):,} bars ({time.time() - t0:.0f} s)")


def assemble(feed: str = "sip", log: Callable[[str], None] = print) -> int:
    """Monthly chunk files -> one parquet per symbol (replaces the symbol files)."""
    cdir = os.path.join(DATA_DIR, f"chunks_{feed}")
    files = sorted(os.path.join(cdir, f) for f in os.listdir(cdir) if f.endswith(".parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["symbol", "ts"], keep="last").sort_values(["symbol", "ts"])
    n = 0
    for s, g in df.groupby("symbol"):
        g.set_index("ts")[BAR_COLS].to_parquet(bars_path(s, feed))
        n += 1
    log(f"assembled {n} {feed} symbol files from {len(files)} months ({len(df):,} bars)")
    return n


# ── trading calendar ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
_SESSIONS: dict = {}


def sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions between two dates (inclusive), from pandas_market_calendars (weekdays as a fallback). Knows FUTURE sessions too - live code needs that to tell whether today
    is the last session of the month."""
    key = (str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
    if key not in _SESSIONS:
        try:
            import pandas_market_calendars as mcal
            d = mcal.get_calendar("NYSE").valid_days(start_date=key[0], end_date=key[1]).tz_localize(None).normalize()
        except Exception:                                                                    # noqa: BLE001
            d = pd.bdate_range(key[0], key[1])
        _SESSIONS[key] = pd.DatetimeIndex(d)
    return _SESSIONS[key]


def month_end_sessions(start, end) -> pd.DatetimeIndex:
    """The last NYSE session of every month that overlaps [start, end]."""
    s = sessions(pd.Timestamp(start) - pd.offsets.MonthBegin(1), pd.Timestamp(end) + pd.offsets.MonthEnd(1))
    return pd.DatetimeIndex(pd.Series(s, index=s).groupby(s.to_period("M")).max().values)


# ── universe ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ETP = (r"ProShares|Direxion|iShares|SPDR|Invesco QQQ|Vanguard|Global X|VanEck|Grayscale|MicroSectors|GraniteShares|Tuttle|Defiance|T-Rex|YieldMax|Volatility Shares|"
       r"\bETF\b|\bETN\b|UltraPro|Ultra(?:Short)? |\b[23]X\b|Daily .* (?:Bull|Bear)")


def etp_symbols(names: pd.Series) -> set:
    """Exchange-traded products among symbol -> asset-name pairs (leveraged / inverse / index funds are not stocks: they have no residual of their own)."""
    return set(names[names.fillna("").str.contains(ETP, regex=True)].index)


def _rank_month(px: pd.DataFrame, dv: pd.DataFrame, sec: pd.Series, n_max: int, min_px: float = 10.0, min_hist: int = 126, exclude: set = frozenset()) -> pd.DataFrame:
    """Members for the month after each month end in ``px`` / ``dv`` (daily close and dollar volume, wide): rank by the 63-session median dollar volume."""
    med = dv.rolling(63, min_periods=40).median()
    hist = px.notna().cumsum()
    me_idx = px.groupby(px.index.to_period("M")).apply(lambda g: g.index.max())
    rows = []
    for per, d in me_idx.items():
        row = med.loc[d]
        ok = row.notna() & (px.loc[d] >= min_px) & (hist.loc[d] >= min_hist) & row.index.isin(sec.index) & ~row.index.isin(ETFS) & ~row.index.isin(list(exclude))
        ok &= ~row.index.isin([k for k, v in DUP_CLASS.items() if v in row.index[ok.values]])
        r = row[ok].sort_values(ascending=False).head(n_max)
        month = (per + 1).to_timestamp()
        for k, (sym, v) in enumerate(r.items()):
            rows.append(dict(month=month, symbol=sym, rank=k + 1, adv=float(v), sector=str(sec.get(sym))))
    return pd.DataFrame(rows)


def build_universe_history(n_max: int = 150, log: Callable[[str], None] = print) -> pd.DataFrame:
    """Monthly members 2016 -> the last month the Stats-arb daily cache covers (survivorship-free: delisted names included)."""
    f = os.path.join(STATARB_DATA, "statarb_daily_broad.parquet")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} is missing - it holds the survivorship-free daily history (build it with Stats-arb/scripts/statarb_broad.py)")
    d = pd.read_parquet(f, columns=["symbol", "date", "close", "volume"])
    sec = pd.read_parquet(os.path.join(STATARB_DATA, "statarb_sectors.parquet")).set_index("symbol").sector
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame({"symbol": sec.index, "sector": sec.values}).to_parquet(SECTOR_FILE)
    px = d.pivot(index="date", columns="symbol", values="close").sort_index()
    dv = d.assign(dv=d.close * d.volume).pivot(index="date", columns="symbol", values="dv").sort_index()
    names = pd.read_parquet(os.path.join(STATARB_DATA, "statarb_assets.parquet")).drop_duplicates("symbol").set_index("symbol").name
    etp = etp_symbols(names)
    u = _rank_month(px, dv, sec, n_max, exclude=etp)
    u = u[u.month >= pd.Timestamp("2016-04-01")]                                             # the first months lack the 63-session window
    u.to_parquet(UNIVERSE_FILE)
    log(f"universe: {u.month.min():%Y-%m} .. {u.month.max():%Y-%m}, {u.symbol.nunique()} symbols ever, {n_max} a month")
    return u


def list_assets(prefix: Optional[str] = None, refresh: bool = False) -> pd.DataFrame:
    """Active U.S. equities with the flags that matter (tradable, shortable, easy to borrow). Raw REST (alpaca-py's asset model chokes on some exchange codes)."""
    if os.path.exists(ASSETS_FILE) and not refresh and time.time() - os.path.getmtime(ASSETS_FILE) < 86400:
        return pd.read_parquet(ASSETS_FILE)
    tc = trading_client(prefix, True)
    rows = [dict(symbol=x["symbol"], name=x.get("name"), exchange=x.get("exchange"), tradable=bool(x.get("tradable")), shortable=bool(x.get("shortable")),
                 etb=bool(x.get("easy_to_borrow")), fractionable=bool(x.get("fractionable"))) for x in tc.get("/assets", {"status": "active", "asset_class": "us_equity"})]
    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(ASSETS_FILE)
    return df


def update_universe(n_max: int = 150, log: Callable[[str], None] = print, prefix: Optional[str] = None, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Add the months the cache is missing (up to the current month) by screening every active NYSE / Nasdaq / Amex common stock with ~10 months of Alpaca daily bars. New names get a
    sector from the sector ETF their market-excess returns correlate with most."""
    from alpaca.data.timeframe import TimeFrame
    u = load_universe()
    now = now or pd.Timestamp.now(tz=NY).tz_localize(None)
    want = pd.Timestamp(now.year, now.month, 1)
    if len(u) and u.month.max() >= want:
        return u
    a = list_assets(prefix)
    junk = r"\b(?:ETF|ETN|Fund|Trust|Warrants?|Rights?|Units?|Notes?|Preferred|Pfd|Depositary|Acquisition|Index|Leveraged|Inverse|2X|3X)\b|%"
    cand = a[a.exchange.isin(["NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"]) & a.tradable & a.symbol.str.fullmatch(r"[A-Z]{1,5}") & ~a.name.fillna("").str.contains(junk, regex=True)]
    syms = sorted(set(cand.symbol) | set(ETFS))
    client = data_client(prefix)
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    start = (want - pd.DateOffset(months=11)).tz_localize(NY).tz_convert("UTC")
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=SIP_DELAY_MIN)
    frames = []

    def chunk(ch):
        for attempt in range(6):
            try:
                df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=ch, timeframe=TimeFrame.Day, start=start.to_pydatetime(), end=end.to_pydatetime(),
                                                            adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df
                return df.reset_index() if len(df) else None
            except Exception as e:                                                           # noqa: BLE001
                bad = re.search(r"invalid symbol: ([^\"\s,]+)", str(e))
                if bad and bad.group(1) in ch:
                    ch = [s for s in ch if s != bad.group(1)]
                    continue
                time.sleep(2 * (attempt + 1))
        return None
    chunks = [syms[i:i + 200] for i in range(0, len(syms), 200)]
    log(f"screening {len(syms)} active symbols with daily bars ({len(chunks)} requests) ...")
    with ThreadPoolExecutor(4) as ex:
        for df in ex.map(chunk, chunks):
            if df is not None:
                frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d.timestamp).dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    px = d.pivot_table(index="date", columns="symbol", values="close").sort_index()
    dv = (d.assign(dv=d.close * d.volume)).pivot_table(index="date", columns="symbol", values="dv").sort_index()
    sec = sectors()
    new = [s for s in px.columns if s not in sec.index and s not in ETFS]
    if new:
        sec = pd.concat([sec, derive_sectors(px, new)])
        pd.DataFrame({"symbol": sec.index, "sector": sec.values}).to_parquet(SECTOR_FILE)
    m = _rank_month(px, dv, sec, n_max, min_hist=min(126, len(px) - 5), exclude=etp_symbols(a.set_index("symbol").name))
    m = m[(m.month > (u.month.max() if len(u) else pd.Timestamp(0))) & (m.month <= want)]
    u = pd.concat([u, m], ignore_index=True)
    u.to_parquet(UNIVERSE_FILE)
    log(f"universe updated through {u.month.max():%Y-%m} ({len(m)} new rows)")
    return u


def derive_sectors(px: pd.DataFrame, names: list) -> pd.Series:
    r = px.pct_change(fill_method=None).clip(-0.3, 0.3)
    etf = r[[e for e in SECTOR_ETF.values() if e in r]].sub(r[MARKET], axis=0)
    inv = {v: k for k, v in SECTOR_ETF.items()}
    out = {}
    for s in names:
        x = r[s].sub(r[MARKET])
        c = etf.corrwith(x, min_periods=60) if x.notna().sum() >= 60 else pd.Series(dtype=float)
        out[s] = inv[c.idxmax()] if len(c.dropna()) else "Tech"
    return pd.Series(out, name="sector")


def load_universe() -> pd.DataFrame:
    if not os.path.exists(UNIVERSE_FILE):
        return pd.DataFrame(columns=["month", "symbol", "rank", "adv", "sector"])
    return pd.read_parquet(UNIVERSE_FILE)


def sectors() -> pd.Series:
    if not os.path.exists(SECTOR_FILE):
        return pd.Series(dtype=object, name="sector")
    return pd.read_parquet(SECTOR_FILE).set_index("symbol").sector


def members(month, n: int = 100, u: Optional[pd.DataFrame] = None) -> list:
    u = load_universe() if u is None else u
    m = pd.Timestamp(month).to_period("M").to_timestamp()
    return list(u[(u.month == m) & (u["rank"] <= n)].sort_values("rank").symbol)


# ── panel ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Panel:
    """Wide 15-minute arrays (rows = bar starts of the master calendar = SPY's regular-hours bars, columns = symbols) plus day bookkeeping.

    A stock bar that is missing inside a day the stock trades (no print in 15 minutes) gets the previous close as its prices and zero volume; a day without any bar stays NaN (not listed yet,
    halted, delisted)."""

    def __init__(self, frames: dict, symbols: list, start=None, end=None):
        cal = frames[MARKET].index
        if start is not None:
            cal = cal[cal >= pd.Timestamp(start)]
        if end is not None:
            cal = cal[cal < pd.Timestamp(end)]
        self.ts = pd.DatetimeIndex(cal)
        self.syms = list(symbols)
        self.col = {s: j for j, s in enumerate(self.syms)}
        T, N = len(self.ts), len(self.syms)
        dates = self.ts.normalize()
        self.dates, self.day = np.unique(dates.values, return_inverse=True)
        self.dates = pd.DatetimeIndex(self.dates)
        self.day = self.day.astype(np.int32)
        first = np.r_[True, self.day[1:] != self.day[:-1]]
        self.day_first = np.where(first)[0]
        self.day_last = np.r_[self.day_first[1:] - 1, T - 1]
        self.bar = (np.arange(T) - self.day_first[self.day]).astype(np.int16)
        self.minute = (self.ts.hour * 60 + self.ts.minute).values.astype(np.int16)        # minutes since midnight of the bar start
        arr = {k: np.full((T, N), np.nan) for k in ("open", "high", "low", "close", "vwap")}
        vol = np.zeros((T, N))
        for j, s in enumerate(self.syms):
            f = frames.get(s)
            if f is None or not len(f):
                continue
            f = f[~f.index.duplicated(keep="last")].reindex(self.ts)
            traded_day = pd.Series(f.close.notna().values).groupby(self.day).transform("any").values
            c = f.close.groupby(self.day).ffill().values
            c = np.where(traded_day, c, np.nan)
            first_c = pd.Series(f.close.values).groupby(self.day).transform("first").values
            lead = np.isnan(c) & traded_day                                               # bars before the day's first print: use the first print (no trade yet = no change)
            c[lead] = first_c[lead]
            o = np.where(np.isnan(f.open.values), c, f.open.values)
            arr["close"][:, j] = c
            arr["open"][:, j] = o
            arr["high"][:, j] = np.where(np.isnan(f.high.values), c, f.high.values)
            arr["low"][:, j] = np.where(np.isnan(f.low.values), c, f.low.values)
            arr["vwap"][:, j] = np.where(np.isnan(f.vwap.values), c, f.vwap.values) if "vwap" in f else c
            vol[:, j] = np.nan_to_num(f.volume.values)
        self.open, self.high, self.low, self.close, self.vwap = arr["open"], arr["high"], arr["low"], arr["close"], arr["vwap"]
        self.volume = vol

    def describe(self) -> str:
        return f"{len(self.syms)} symbols · {len(self.ts):,} bars · {self.dates[0]:%Y-%m-%d} .. {self.dates[-1]:%Y-%m-%d}"

    def daily_close(self) -> np.ndarray:
        return self.close[self.day_last]

    def daily_open(self) -> np.ndarray:
        return self.open[self.day_first]

    def tail(self, n_bars: int) -> "Panel":
        """The last ``n_bars`` rows (whole days are not enforced)."""
        return self.slice(max(len(self.ts) - n_bars, 0), len(self.ts))

    def slice(self, i0: int, i1: int) -> "Panel":
        q = object.__new__(Panel)
        q.syms, q.col = self.syms, self.col
        q.ts = self.ts[i0:i1]
        for k in ("open", "high", "low", "close", "vwap", "volume", "bar", "minute"):
            setattr(q, k, getattr(self, k)[i0:i1])
        d = self.day[i0:i1]
        d0 = d[0] if len(d) else 0
        q.day = (d - d0).astype(np.int32)
        q.dates = self.dates[d0:(d[-1] + 1 if len(d) else d0)]
        first = np.r_[True, q.day[1:] != q.day[:-1]] if len(d) else np.zeros(0, bool)
        q.day_first = np.where(first)[0]
        q.day_last = np.r_[q.day_first[1:] - 1, len(d) - 1] if len(d) else np.zeros(0, int)
        return q


def load_panel(symbols: list, start=None, end=None, feed: str = "sip", extra: Optional[dict] = None) -> Panel:
    """Panel for ``symbols`` + the factor ETFs from the cache (``extra`` = {symbol: frame} bars appended in memory, e.g. today's live bars)."""
    syms = list(dict.fromkeys(list(ETFS) + [s for s in symbols if s not in ETFS]))
    frames = {}
    for s in syms:
        f = load_bars(s, feed)
        if extra and s in extra and len(extra[s]):
            x = extra[s]
            f = pd.concat([f[f.index < x.index.min()], x]) if len(f) else x
        frames[s] = f
    if MARKET not in frames or not len(frames[MARKET]):
        raise FileNotFoundError(f"no {feed} bars for {MARKET} - run:  python -m resarb.data download")
    return Panel(frames, syms, start, end)


def universe_symbols(n: int = 150) -> list:
    u = load_universe()
    return sorted(set(u[u["rank"] <= n].symbol))


def membership_mask(panel: Panel, n: int, u: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Boolean (days x symbols): is the symbol a member (rank <= n) in that day's month? ETFs are never members."""
    u = load_universe() if u is None else u
    u = u[u["rank"] <= n]
    months = panel.dates.to_period("M").to_timestamp()
    m = np.zeros((len(panel.dates), len(panel.syms)), bool)
    by = u.groupby("month").symbol.apply(set).to_dict()
    for k, mo in enumerate(months):
        s = by.get(mo)
        if s is None and len(by):                                                               # a month beyond the table: keep the last known membership (live, before the monthly refresh)
            prior = [x for x in by if x <= mo]
            s = by[max(prior)] if prior else set()
        for sym in s or ():
            j = panel.col.get(sym)
            if j is not None:
                m[k, j] = True
    return m


def status() -> str:
    u = load_universe()
    n = len([f for f in os.listdir(bars_dir("sip")) if f.endswith(".parquet")])
    ni = len([f for f in os.listdir(bars_dir("iex")) if f.endswith(".parquet")])
    spy = load_bars(MARKET)
    return (f"universe: {u.month.min():%Y-%m} .. {u.month.max():%Y-%m}, {u.symbol.nunique()} symbols" if len(u) else "universe: not built") + \
        f" · SIP 15-min files: {n}" + (f" (SPY through {spy.index.max():%Y-%m-%d %H:%M})" if len(spy) else "") + f" · IEX files: {ni}"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "universe":
        build_universe_history()
        update_universe()
    elif cmd == "download":
        download_by_day("sip", n=int(os.environ.get("RESARB_N", "100")), workers=int(os.environ.get("RESARB_WORKERS", "6")))
        assemble("sip")
        build_price_fixes(universe_symbols(150) + list(ETFS))
    elif cmd == "fixes":
        build_price_fixes(universe_symbols(150) + list(ETFS))
    elif cmd == "iex":
        download_by_day("iex", start="2024-01-01", n=100, workers=int(os.environ.get("RESARB_WORKERS", "6")))
        assemble("iex")
    elif cmd == "topup":
        res = update_bars(list(ETFS) + universe_symbols(150), "sip", workers=int(os.environ.get("RESARB_WORKERS", "4")))
        print(pd.Series(res).str.split().str[0].value_counts().to_string())
    print(status())
