"""Bars, the cache, freshness and the market clock (no Qt). The cache is the strategy's own parquet store, topped up from Alpaca on demand."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import supertrend_gui as GU
import supertrend_strategy as S

ET = ZoneInfo("America/New_York")
DATA_DIR = os.environ.get("SUPERTREND_DATA_DIR", "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data")
CACHE_START = "2016-01-01"            # every window is served from ONE cache file per symbol/timeframe (a different start year used to create a second, partial file)
SIP_DELAY_MIN = 16                    # Alpaca's free SIP feed trails real time by ~15 min
OPEN_T, CLOSE_T = (9, 30), (16, 0)


def now_et() -> datetime:
    """Wall clock in New York as a naive datetime (the bars are indexed the same way)."""
    return datetime.now(ET).replace(tzinfo=None)


# ── cache ─────────────────────────────────────────────────────────────────────────
def cache_path(symbol: str, tf: int) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_{tf}m_{CACHE_START[:4]}_split.parquet")


def _ohlc(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("bar").sort_index()[["open", "high", "low", "close"]].ffill().dropna()


def read_cache(symbol: str, tf: int) -> pd.DataFrame:
    path = cache_path(symbol, tf)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no cached {symbol} {tf}-min bars yet ({path}) - connect to the internet and press 'Update data'")
    p = pd.read_parquet(path)
    if "symbol" in p.columns:
        p = p[p.symbol == symbol]
    p = p.copy()
    p["bar"] = pd.to_datetime(p["et"], utc=True).dt.tz_convert(S.ET).dt.tz_localize(None)
    return _ohlc(p)


def load_bars(symbol: str, tf: int, refresh: bool = True, log: Callable = print) -> pd.DataFrame:
    """All cached bars for ``symbol``, topped up from Alpaca first when ``refresh``. If the top-up fails (offline, no keys) the cache is used and the log says so."""
    if refresh:
        try:
            end = now_et().strftime("%Y-%m-%d")
            panel = S.load_alpaca_bars([symbol], tf, CACHE_START, end, rth_only=True, cache_dir=DATA_DIR, dividend_adjusted=False, log=log)
            return _ohlc(panel[panel.symbol == symbol])
        except Exception as exc:                          # noqa: BLE001
            log(f"  could not refresh from Alpaca ({type(exc).__name__}: {str(exc)[:120]}) - using the cached bars")
    return read_cache(symbol, tf)


def describe_bars(bars: pd.DataFrame) -> str:
    if bars is None or not len(bars):
        return "no bars"
    return f"{bars.index[0]:%Y-%m-%d} → {bars.index[-1]:%Y-%m-%d %H:%M} ET · {len(bars):,} bars"


def expected_last_bar(now: datetime, tf: int = 15) -> datetime:
    """Start time of the newest bar a complete cache should hold at ``now`` (ignores exchange holidays, so it can only err on the side of asking for a refresh)."""
    t = now - timedelta(minutes=SIP_DELAY_MIN)
    day = t.replace(hour=0, minute=0, second=0, microsecond=0)
    open_ts, close_ts = day.replace(hour=OPEN_T[0], minute=OPEN_T[1]), day.replace(hour=CLOSE_T[0], minute=CLOSE_T[1])
    if t.weekday() < 5 and t >= close_ts:
        return close_ts - timedelta(minutes=tf)
    if t.weekday() < 5 and t >= open_ts:
        return (t - timedelta(minutes=tf)).replace(second=0, microsecond=0) - timedelta(minutes=(t.minute % tf))
    d = day - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.replace(hour=CLOSE_T[0], minute=CLOSE_T[1]) - timedelta(minutes=tf)


def needs_refresh(last_bar: Optional[datetime], now: datetime, last_attempt: float = 0.0, tf: int = 15, min_gap_s: float = 1200.0) -> bool:
    """True when the cache is behind what Alpaca has by now. At most one attempt per 20 minutes so a holiday does not trigger a download on every click."""
    if last_bar is None:
        return True
    if time.time() - last_attempt < min_gap_s:
        return False
    return expected_last_bar(now, tf) - last_bar >= timedelta(minutes=tf)


def freshness_text(last_bar: Optional[datetime], now: datetime) -> tuple[str, bool]:
    """(text, is_stale) describing how current the newest bar is."""
    if last_bar is None:
        return "No local data yet - press 'Update data'.", True
    lag = expected_last_bar(now) - last_bar
    stale = lag >= timedelta(minutes=15)
    age = now - last_bar
    a = f"{int(age.total_seconds() // 60)} min" if age < timedelta(hours=2) else f"{age.total_seconds() / 3600:.1f} h" if age < timedelta(days=2) else f"{age.days} days"
    msg = f"Data through {last_bar:%Y-%m-%d %H:%M} ET ({a} ago)."
    if stale:
        msg += "  Newer bars may exist - a run refreshes them first (if 'Refresh data first' is ticked), or press 'Update data'."
    return msg, stale


# ── window + warm-up ──────────────────────────────────────────────────────────────
def warmup_bars(values: dict) -> int:
    """Bars needed before the first tradable bar so every indicator (ATR, vol baseline, ADX) is primed."""
    return int(max(values.get("atrPeriod", 500), values.get("volLength", 20) + values.get("volSmaLength", 100), 3 * values.get("adxLength", 14))) + 5


def window(bars: pd.DataFrame, start: str, end: str, warm: int = 0) -> tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    """(bars from ``warm`` bars before ``start`` to the end of ``end``, timestamp of the first bar on/after ``start``)."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    i0, i1 = int(bars.index.searchsorted(t0)), int(bars.index.searchsorted(t1))
    if i0 >= i1:
        return bars.iloc[0:0], None
    return bars.iloc[max(0, i0 - warm):i1], bars.index[i0]


# ── the signal, computed exactly as the live engine does (no config globals) ──────
def compute_signal(bars: pd.DataFrame, cfg: S.Config, now: Optional[datetime] = None) -> Optional[dict]:
    """Mirror of supertrend_gui.Engine.signal(): the SIGNAL bar is the last CLOSED bar, the MONITOR bar may still be forming. Returns None while warming up.
    Kept as a pure function so the Signal monitor can show what the engine would decide without touching the engine's module-level config."""
    h, l, c, o = (bars[k].to_numpy(float) for k in ("high", "low", "close", "open"))
    st, d = S.supertrend(h, l, c, cfg.atrPeriod, cfg.factor)
    last = len(c) - 1
    now = now or now_et()
    forming = (bars.index[last] + timedelta(minutes=15)) > now
    i = last - 1 if (forming and last >= 1) else last
    if i < 0 or np.isnan(st[i]) or np.isnan(d[i]):
        return None
    want = 1 if d[i] == -1 else -1
    adx, high_vol = GU.regime_inputs(cfg, h, l, c, i)
    return {"bar": bars.index[i], "close": float(c[i]), "st": float(st[i]), "open": float(o[i]), "high": float(h[i]), "low": float(l[i]), "want": want, "adx": adx, "high_vol": high_vol, "forming": bool(forming),
            "mon_bar": bars.index[last], "mon_high": float(h[last]), "mon_low": float(l[last]),
            "limit": float(st[i] * (1 + cfg.buyback)) if want > 0 else float(st[i] * (1 - cfg.buyback)), "st_series": st, "dir_series": d}


def position_levels(state: dict, cfg: S.Config, adx: Optional[float] = None) -> Optional[dict]:
    """Exit levels of the open position the engine saved (a mirror of Engine.tp_level / trail_level; every level is a SOXL price). None when flat.
    ``adx`` is the ADX of the last closed bar (compute_signal()["adx"]); it only matters with the ADX dynamic trail on."""
    if not state or not state.get("pos"):
        return None
    ref, se, long_ = float(state["entry_ref"]), float(state["supert_entry"]), state.get("side") == "long"
    fixed = bool(state.get("active_fixed", True))
    if long_:
        tp = ref * (1 + cfg.price_perc) if fixed else ref * (1 + (ref - se) / ref * cfg.limitfactor)
    else:
        tp = ref * (1 - cfg.price_perc) if fixed else ref * (1 - (se - ref) / ref * cfg.limitfactor)
    tf = GU.dynamic_trail_factor(cfg, adx)
    ext = state.get("max_high") if long_ else state.get("min_low")
    target = float(state["target"])
    armed = ext is not None and ((ext >= target) if long_ else (ext <= target))
    trail = (ref + (ext - ref) * tf) if (armed and long_) else (ref - (ref - ext) * tf) if armed else None
    return {"side": "long" if long_ else "short", "symbol": state.get("symbol", ""), "entry_ref": ref, "target": target, "tp": tp, "trail": trail, "armed": bool(armed), "extreme": ext}


# ── market clock (weekday + regular hours; holidays are only known to the broker) ──
def is_rth(now: datetime) -> bool:
    t = (now.hour, now.minute)
    return now.weekday() < 5 and OPEN_T <= t < CLOSE_T


def next_open(now: datetime) -> datetime:
    d = now.replace(hour=OPEN_T[0], minute=OPEN_T[1], second=0, microsecond=0)
    if now >= d or now.weekday() >= 5:
        d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def next_bar_close(now: datetime, tf: int = 15) -> Optional[datetime]:
    """When the bar that is forming right now closes (None outside the regular session)."""
    if not is_rth(now):
        return None
    base = now.replace(hour=OPEN_T[0], minute=OPEN_T[1], second=0, microsecond=0)
    k = int((now - base).total_seconds() // (tf * 60)) + 1
    return min(base + timedelta(minutes=tf * k), now.replace(hour=CLOSE_T[0], minute=CLOSE_T[1], second=0, microsecond=0))


def fmt_td(sec: float) -> str:
    sec = max(0, int(sec))
    if sec >= 86400:
        return f"{sec // 86400}d {sec % 86400 // 3600}h"
    if sec >= 3600:
        return f"{sec // 3600}h {sec % 3600 // 60:02d}m"
    return f"{sec // 60}m {sec % 60:02d}s"
