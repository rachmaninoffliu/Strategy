"""
Statistical-arbitrage data layer (no Qt): the trading universe, the Alpaca daily-bar cache and the credential helper.

* Universe  ~330 liquid U.S. large caps tagged with an 11-sector GICS-style label, plus the ETFs the risk model uses as macro / geopolitical factors
            (SPY market, USO oil, TLT rates, UUP dollar, GLD gold, VIXY volatility, the 11 sector SPDRs).
* Bars      daily OHLCV, split- AND dividend-adjusted (adjustment='all'): a residual-reversion model must not see an ex-dividend drop as a price move.
* Cache     one parquet file, topped up incrementally. Adjusted history is rewritten by Alpaca whenever a split / dividend happens, so every top-up re-fetches a small overlap and
            re-downloads a symbol's whole history if the overlap no longer matches the cache.

SURVIVORSHIP BIAS: the universe is today's large caps. Names that were large in 2016 and have since been delisted / acquired are missing, which flatters the LONG leg of any
reversal strategy (losers that kept falling to zero are not in the sample). DA/Valid flag this; the backtest report quantifies it with the 'leave-sector-out' / sub-universe checks.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                      # the Stats-arb folder (scripts/ sits one level below it)
DATA_DIR = os.environ.get("STATARB_DATA_DIR", os.path.join(ROOT, "data"))
REPORTS = os.environ.get("STATARB_REPORTS_DIR", os.path.join(ROOT, "reports"))
CACHE_FILE = os.path.join(DATA_DIR, "statarb_daily_2015.parquet")
CACHE_START = "2015-01-02"
SIP_DELAY_MIN = 16

SECTOR_ETF = {"Tech": "XLK", "Comm": "XLC", "Disc": "XLY", "Stap": "XLP", "Energy": "XLE", "Fin": "XLF", "Health": "XLV", "Ind": "XLI", "Mat": "XLB", "Util": "XLU", "RE": "XLRE"}
MACRO_ETFS = {"SPY": "S&P 500 (market)", "QQQ": "Nasdaq-100", "IWM": "Russell 2000", "USO": "Crude oil (geopolitics / Hormuz)", "TLT": "20y Treasuries (rates / Fed)",
              "UUP": "U.S. dollar", "GLD": "Gold (safe haven)", "VIXY": "Short-term VIX futures (fear)", "ITA": "Aerospace & defense"}
FACTOR_ETFS = sorted(set(MACRO_ETFS) | set(SECTOR_ETF.values()))

_U = {
    "Tech": "AAPL MSFT NVDA AVGO ORCL CRM ADBE AMD CSCO ACN IBM INTU TXN QCOM AMAT MU LRCX ADI KLAC SNPS CDNS NOW PANW ANET MSI APH ROP FTNT HPQ HPE NTAP WDC STX GLW KEYS TEL TER "
            "MCHP ON NXPI SWKS FICO CDW PTC TYL FFIV AKAM EPAM IT TRMB VRSN INTC DELL SMCI GDDY ZBRA JBL GEN TDY FSLR ENPH",
    "Comm": "GOOGL GOOG META NFLX DIS CMCSA T VZ TMUS CHTR EA TTWO WBD OMC IPG LYV FOXA NWSA MTCH PARA LUMN",
    "Disc": "AMZN TSLA HD MCD NKE LOW SBUX TJX BKNG ORLY AZO ROST CMG MAR HLT GM F LEN DHI PHM NVR EBAY EXPE YUM DRI DPZ ULTA BBY GRMN POOL TSCO KMX LVS WYNN MGM CZR RCL CCL NCLH "
            "APTV BWA LKQ HAS TPR RL DECK LULU WHR MHK ETSY GPC",
    "Stap": "PG KO PEP COST WMT PM MO MDLZ CL KMB GIS HSY SYY KR ADM STZ KHC MKC CAG CPB SJM HRL TSN CLX CHD EL TAP KDP MNST DG DLTR TGT BG LW KVUE",
    "Energy": "XOM CVX COP EOG SLB PSX MPC VLO OXY WMB KMI OKE HAL BKR DVN FANG CTRA APA EQT TRGP",
    "Fin": "JPM BAC WFC C GS MS BLK SCHW AXP USB PNC TFC COF BK STT MET PRU AIG ALL TRV CB PGR AFL MMC AON ICE CME SPGI MCO V MA PYPL FITB RF KEY HBAN CFG MTB NTRS AMP SYF BRK.B "
           "CINF L WRB HIG PFG TROW IVZ BEN NDAQ CBOE MSCI RJF GL AJG FDS",
    "Health": "UNH JNJ LLY ABBV MRK PFE TMO ABT DHR BMY AMGN GILD CVS CI ELV HUM MDT ISRG SYK BSX ZTS VRTX REGN BIIB IQV A MTD WAT EW BDX BAX ZBH HCA MCK COR CAH DGX LH IDXX RMD "
              "ALGN DXCM STE COO TFX CNC MOH HOLX INCY",
    "Ind": "GE CAT HON UNP UPS BA RTX LMT NOC GD DE MMM ETN EMR ITW PH CMI FDX CSX NSC WM RSG ADP PCAR CTAS FAST ODFL JCI TT CARR OTIS AME ROK DOV XYL SWK WAB EXPD CHRW JBHT DAL "
           "UAL LUV AAL ALK URI PWR HWM TDG LHX TXT HII IR VRSK J MAS ALLE NDSN GNRC",
    "Mat": "LIN SHW APD ECL NEM FCX DOW DD NUE STLD VMC MLM PPG IFF CTVA ALB MOS CF FMC IP PKG AMCR BALL AVY CE EMN",
    "Util": "NEE DUK SO D AEP EXC SRE XEL ED PEG WEC ES EIX PCG AEE DTE ETR FE PPL CMS CNP NI LNT EVRG ATO AES",
    "RE": "PLD AMT EQIX CCI PSA O WELL SPG DLR AVB EQR VTR ARE EXR MAA UDR ESS INVH HST KIM REG FRT BXP VICI IRM SBAC CBRE",
}
SECTOR_OF = {s: sec for sec, names in _U.items() for s in names.split()}
UNIVERSE = sorted(SECTOR_OF)
ALL_SYMBOLS = sorted(set(UNIVERSE) | set(FACTOR_ETFS))


# ── credentials: the project's key file "Alpaca Key.txt" in the project folder, then ALPACA_API_KEY / ALPACA_SECRET_KEY, then ~/.claude.json (alpaca_keys.py) ──────
def creds(paper: bool = True) -> tuple[str, str]:
    import alpaca_keys
    return alpaca_keys.creds(paper)


def data_client(paper: bool = True):
    from alpaca.data.historical import StockHistoricalDataClient
    k, s = creds(paper)
    return StockHistoricalDataClient(k, s)


# ── download ─────────────────────────────────────────────────────────────────────
def _fetch(client, symbols: list[str], start: str, end: Optional[pd.Timestamp], log: Callable[[str], None]) -> pd.DataFrame:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    end_ts = end if end is not None else pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=SIP_DELAY_MIN)
    rows = []
    for i in range(0, len(symbols), 40):
        chunk = symbols[i:i + 40]
        for feed in (DataFeed.SIP, DataFeed.IEX):
            try:
                req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=pd.Timestamp(start, tz="UTC").to_pydatetime(), end=end_ts.to_pydatetime(),
                                       adjustment=Adjustment.ALL, feed=feed)
                df = client.get_stock_bars(req).df
                break
            except Exception as e:                                                     # noqa: BLE001
                if feed == DataFeed.IEX:
                    log(f"  chunk {i // 40 + 1}: fetch failed ({type(e).__name__}: {str(e)[:100]})")
                    df = None
                else:
                    log(f"  chunk {i // 40 + 1}: SIP not available ({str(e)[:80]}) - falling back to IEX")
        if df is None or df.empty:
            continue
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        rows.append(df[["symbol", "date", "open", "high", "low", "close", "volume"]])
        log(f"  fetched {min(i + 40, len(symbols))}/{len(symbols)} symbols")
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")


def load_cache() -> pd.DataFrame:
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    return pd.read_parquet(CACHE_FILE)


def update_cache(symbols: Optional[list[str]] = None, log: Callable[[str], None] = print, full: bool = False) -> pd.DataFrame:
    """Top the cache up from Alpaca (read-only data requests). Returns the long-format frame. Offline / no keys -> the cache as it is."""
    symbols = list(symbols or ALL_SYMBOLS)
    cache = load_cache()
    try:
        client = data_client(True)
    except Exception as e:                                                             # noqa: BLE001
        log(f"  no Alpaca data credentials ({e}) - using the cache")
        return cache
    os.makedirs(DATA_DIR, exist_ok=True)
    have = set(cache.symbol.unique()) if len(cache) else set()
    if full or not have:
        log(f"downloading {len(symbols)} symbols from {CACHE_START} ...")
        new = _fetch(client, symbols, CACHE_START, None, log)
        out = new
    else:
        last = cache.date.max()
        start = (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        missing = [s for s in symbols if s not in have]
        log(f"topping up from {start} ({len(symbols) - len(missing)} symbols) + {len(missing)} new symbols ...")
        new = _fetch(client, [s for s in symbols if s in have], start, None, log)
        extra = _fetch(client, missing, CACHE_START, None, log) if missing else pd.DataFrame()
        old_close = cache.set_index(["symbol", "date"]).close
        cmp = new.set_index(["symbol", "date"]).close.reindex(old_close.index.intersection(new.set_index(["symbol", "date"]).index))
        if len(cmp):
            rel = (cmp / old_close.reindex(cmp.index) - 1).abs().groupby(level=0).max()
            redo = list(rel[rel > 5e-4].index)                                         # adjusted history was rewritten (split / dividend) -> take the whole symbol again
            if redo:
                log(f"  {len(redo)} symbols had their adjusted history rewritten - re-downloading them in full")
                full_redo = _fetch(client, redo, CACHE_START, None, log)
                cache = cache[~cache.symbol.isin(redo)]
                new = pd.concat([new[~new.symbol.isin(redo)], full_redo], ignore_index=True)
        out = pd.concat([cache, new, extra], ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    if len(out):
        out.to_parquet(CACHE_FILE)
        log(f"cache: {out.symbol.nunique()} symbols, {out.date.min():%Y-%m-%d} -> {out.date.max():%Y-%m-%d}, {len(out):,} rows")
    return out


# ── panel ────────────────────────────────────────────────────────────────────────
class Panel:
    """Wide date x symbol frames of open / close / volume for stocks and for the factor ETFs."""

    def __init__(self, long: pd.DataFrame, sector_map: Optional[pd.Series] = None, stocks: Optional[list] = None):
        long = long.copy()
        long["date"] = pd.to_datetime(long["date"])
        piv = {f: long.pivot(index="date", columns="symbol", values=f).sort_index() for f in ("open", "close", "volume")}
        self.open_all, self.close_all, self.volume_all = piv["open"], piv["close"], piv["volume"]
        etfs = [s for s in FACTOR_ETFS if s in self.close_all.columns]
        smap = dict(SECTOR_OF) if sector_map is None else {**SECTOR_OF, **sector_map.to_dict()}
        stocks_default = [s for s in UNIVERSE if s in self.close_all.columns] if sector_map is None else [s for s in self.close_all.columns if s in smap and s not in FACTOR_ETFS]
        stocks = [c for c in stocks if c in self.close_all.columns] if stocks is not None else stocks_default
        self.stocks, self.etfs = stocks, etfs
        self.open, self.close, self.volume = self.open_all[stocks], self.close_all[stocks], self.volume_all[stocks]
        self.etf_close, self.etf_open = self.close_all[etfs], self.open_all[etfs]
        self.sector = pd.Series({s: smap[s] for s in stocks})
        share = self.close.notna().mean(axis=1)
        ref = float(share.iloc[-60:].median()) if len(share) > 60 else float(share.median())
        keep = int(len(share))
        while keep > 1 and share.iloc[keep - 1] < 0.5 * ref:                          # e.g. ETFs refreshed but the stock cache was not
            keep -= 1
        if keep < len(share):
            for k in ("open_all", "close_all", "volume_all", "open", "close", "volume", "etf_close", "etf_open"):
                setattr(self, k, getattr(self, k).iloc[:keep])
            self.trimmed = len(share) - keep
        self.dates = self.close.index

    def describe(self) -> str:
        return f"{len(self.stocks)} stocks + {len(self.etfs)} ETFs · {self.dates[0]:%Y-%m-%d} → {self.dates[-1]:%Y-%m-%d} · {len(self.dates):,} sessions"


def load_panel(refresh: bool = False, log: Callable[[str], None] = print) -> Panel:
    long = update_cache(log=log) if refresh else load_cache()
    if long.empty:
        raise FileNotFoundError("no cached stat-arb bars yet - press 'Update data' (needs the Alpaca paper keys in ~/.claude.json)")
    return Panel(long)


def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York")


if __name__ == "__main__":
    import sys
    p = load_panel(refresh=True) if "--refresh" in sys.argv else load_panel()
    print(p.describe())
    print(f"universe {len(UNIVERSE)} symbols defined; {len(p.stocks)} with data; missing: {sorted(set(UNIVERSE) - set(p.stocks))}")


# ── earnings calendar (idiosyncratic-jump control) ───────────────────────────────────
EARN_FILE = os.path.join(DATA_DIR, "statarb_earnings.parquet")


def load_earnings() -> pd.DataFrame:
    if not os.path.exists(EARN_FILE):
        return pd.DataFrame(columns=["symbol", "ts"])
    return pd.read_parquet(EARN_FILE)


def update_earnings(symbols: Optional[list[str]] = None, log: Callable[[str], None] = print, max_age_days: float = 3.0, workers: int = 3) -> pd.DataFrame:
    """Historical + scheduled earnings timestamps from Yahoo (read-only). A symbol is re-fetched when its rows are older than ``max_age_days`` (the next date can be revised).
    Yahoo throttles silently (empty answers instead of errors), so: small chunks, a checkpoint after each, and a cool-down + retry for ACTIVE symbols that came back empty
    (delisted symbols legitimately have no calendar and are not retried)."""
    from concurrent.futures import ThreadPoolExecutor
    import yfinance as yf
    symbols = list(symbols or UNIVERSE)
    old = load_earnings()
    fresh = set()
    if len(old) and "fetched" in old.columns:
        age = pd.Timestamp.now() - old.groupby("symbol").fetched.max()
        fresh = set(age[age < pd.Timedelta(days=max_age_days)].index)
    active = None
    af = os.path.join(DATA_DIR, "statarb_assets.parquet")
    if os.path.exists(af):
        a = pd.read_parquet(af)
        active = set(a[a.status == "active"].symbol)
    nodata_f = os.path.join(DATA_DIR, "statarb_earnings_nodata.json")
    nodata: dict = {}
    try:
        nodata = json.load(open(nodata_f)) if os.path.exists(nodata_f) else {}
    except (OSError, ValueError):
        nodata = {}
    recent_nodata = {s_ for s_, t_ in nodata.items() if (pd.Timestamp.now() - pd.Timestamp(t_)) < pd.Timedelta(days=7)}          # Yahoo has nothing for them: do not ask again every refresh
    todo = [s for s in symbols if s not in fresh and s not in recent_nodata and (active is None or s in active)]      # delisted names have no calendar - not worth thousands of empty Yahoo calls each refresh
    log(f"earnings calendar: {len(todo)} symbols to fetch ({len(symbols) - len(todo)} fresh or delisted)")

    def one(sym: str):
        try:
            d = yf.Ticker(sym).get_earnings_dates(limit=100)
            if d is None or d.empty:
                return sym, None
            ts = pd.DatetimeIndex(d.index).tz_convert("America/New_York").tz_localize(None)
            return sym, pd.DataFrame({"symbol": sym, "ts": ts, "fetched": pd.Timestamp.now()})
        except Exception:                                                              # noqa: BLE001
            return sym, None

    rows: list = []
    got: set = set()

    def flush() -> pd.DataFrame:
        cur = pd.concat([old[~old.symbol.isin(got)]] + rows, ignore_index=True) if rows else old
        if len(cur):
            os.makedirs(DATA_DIR, exist_ok=True)
            cur.to_parquet(EARN_FILE)
        return cur

    pending = list(todo)
    for rnd in range(4):
        failed = []
        for i in range(0, len(pending), 100):
            chunk = pending[i:i + 100]
            with ThreadPoolExecutor(workers) as ex:
                res = list(ex.map(one, chunk))
            ok = [(s_, df) for s_, df in res if df is not None]
            for s_, df in ok:
                rows.append(df); got.add(s_)
            miss = [s_ for s_, df in res if df is None and (active is None or s_ in active)]
            failed += miss
            if len(pending) > 150 and len(ok) < 0.3 * len([s_ for s_ in chunk if active is None or s_ in active]):          # a big batch answering mostly empty = throttled: back off
                time.sleep(45)
            time.sleep(0.5)
            flush()
            log(f"  earnings round {rnd + 1}: {min(i + 100, len(pending))}/{len(pending)} · {len(got)} stored")
        if not failed or (rnd >= 1 and set(failed) == set(pending)):                    # nothing left, or a retry recovered nothing: Yahoo simply has no data for them
            break
        pending = failed
        if len(failed) > 150:
            log(f"  {len(failed)} active symbols came back empty (throttled?) - cooling down 60 s, then retrying")
            time.sleep(60)
        elif rnd >= 1:
            break                                                                      # a few stragglers after a retry: Yahoo simply has nothing for them
        else:
            log(f"  {len(failed)} symbols came back empty - one quick retry")
    out = flush()
    missing = [s_ for s_ in todo if s_ not in got]
    for s_ in missing:
        nodata[s_] = str(pd.Timestamp.now())
    try:
        json.dump(nodata, open(nodata_f, "w"))
    except OSError:
        pass
    log(f"earnings calendar: {out.symbol.nunique() if len(out) else 0} symbols stored; no data for {len(missing)} (mostly delisted)")
    return out


def future_sessions(last, n: int) -> pd.DatetimeIndex:
    """The next ``n`` NYSE sessions after ``last`` (holidays included in the calendar), as naive midnight timestamps."""
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("NYSE")
        days = cal.valid_days(start_date=pd.Timestamp(last) + pd.Timedelta(days=1), end_date=pd.Timestamp(last) + pd.Timedelta(days=int(n * 2.2) + 10))
        return pd.DatetimeIndex(days.tz_localize(None).normalize()[:n])
    except Exception:                                                                  # noqa: BLE001
        return pd.bdate_range(pd.Timestamp(last) + pd.Timedelta(days=1), periods=n)


def earnings_reaction_matrix(panel: Panel, earn: pd.DataFrame, extra: int = 0) -> np.ndarray:
    """Boolean (dates [+ extra future sessions] x stocks): True on the session where the market first trades the earnings news (same day for pre-market reports, next session for
    after-close ones). ``extra`` > 0 appends that many future NYSE sessions so the SCHEDULE of announcements just beyond the last bar is visible (needed live)."""
    dv = panel.dates.values
    if extra:
        dv = np.concatenate([dv, future_sessions(panel.dates[-1], extra).values])
    T = len(dv)
    col = {c: i for i, c in enumerate(panel.stocks)}
    out = np.zeros((T, len(panel.stocks)), dtype=bool)
    if earn is None or earn.empty:
        return out
    for sym, g in earn.groupby("symbol"):
        j = col.get(sym)
        if j is None:
            continue
        ts = pd.DatetimeIndex(g.ts)
        day = ts.normalize().values
        amc = (ts.hour >= 15) & (ts.hour <= 23)
        pos = np.where(amc, np.searchsorted(dv, day, side="right"), np.searchsorted(dv, day, side="left"))
        pos = pos[(pos >= 0) & (pos < T)]
        out[pos, j] = True
    return out


def next_earnings(symbols: list[str], earn: Optional[pd.DataFrame] = None, today: Optional[pd.Timestamp] = None) -> dict:
    """{symbol: next scheduled reaction session (Timestamp, naive) or None}"""
    earn = load_earnings() if earn is None else earn
    today = (today or pd.Timestamp.now(tz="America/New_York").tz_localize(None)).normalize()
    out = {}
    for sym, g in earn[earn.symbol.isin(symbols)].groupby("symbol"):
        ts = pd.DatetimeIndex(g.ts)
        day = ts.normalize() + pd.to_timedelta(((ts.hour >= 15)).astype(int), unit="D")
        fut = day[day >= today]
        out[sym] = fut.min() if len(fut) else None
    return out
