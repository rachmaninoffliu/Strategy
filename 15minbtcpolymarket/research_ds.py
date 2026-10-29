"""Research dataset for new BTC 15-minute strategy ideas: every underdog signal (15-35c, minutes 12-13, all hours, Jun 1 -> now) joined with BTC spot features known at the decision instant.

    from research_ds import build; ds = build()

Columns: the signal (ticker, open_ts, minute, hour, dow, side, cost, win, net = flat 1-contract net P&L after 2c slippage and the Kalshi fee), the Kalshi quote path (mid / spread / volume,
mid change over the last 3 and 6 minutes), and Coinbase BTC-USD 1-minute spot features: realised vol over the last 30 min / 2 h / 24 h and its 7-day median, the variance ratio of 5-minute to 1-minute returns
(< 1 = mean-reverting), the spot distance to the strike in vol units, the previous window's move.  Everything is computed from data strictly BEFORE the decision instant.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from trader.data import extract_signals, load_rows
from trader.fees import fee_per_contract
from trader.strategy import StrategyParams

warnings.filterwarnings("ignore")
D = Path(__file__).parent / "data"
SLIP = 0.02


def load_spot() -> pd.DataFrame:
    parts = [pd.read_csv(D / f, usecols=["ts", "close"]) for f in ("cb_1m_long.csv", "cb_1m.csv", "cb_1m_recent.csv") if (D / f).exists()]
    s = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
    s["ts"] = (s.ts // 60 * 60).astype("int64")
    s = s.drop_duplicates("ts").set_index("ts")
    grid = pd.RangeIndex(int(s.index.min()), int(s.index.max()) + 60, 60)
    return s.reindex(grid).ffill()                                      # minute grid, gaps carried forward (rare)


def spot_features(s: pd.DataFrame) -> pd.DataFrame:
    c = s.close.values.astype(float)
    lc = np.log(c)
    r1 = pd.Series(np.r_[0.0, np.diff(lc)], index=s.index)
    r5 = pd.Series(lc, index=s.index).diff(5)
    f = pd.DataFrame(index=s.index)
    f["close"] = c
    for n in (30, 120, 1440):
        f[f"sig{n}"] = r1.rolling(n, min_periods=n // 2).std()
    f["sig30_med7d"] = f.sig30.rolling(7 * 1440, min_periods=1440).median()
    f["sig120_med7d"] = f.sig120.rolling(7 * 1440, min_periods=1440).median()
    for w in (360, 1440, 4320):                                          # variance ratio: 5-min variance / (5 x 1-min variance) over the trailing window
        f[f"vr{w}"] = (r5.pow(2).rolling(w, min_periods=w // 2).mean() / 5.0) / r1.pow(2).rolling(w, min_periods=w // 2).mean()
    f["ret60"] = pd.Series(lc, index=s.index).diff(60)                   # last-hour spot move
    f["ret15"] = pd.Series(lc, index=s.index).diff(15)
    f["absret15"] = f.ret15.abs()
    f["abs15_med7d"] = f.absret15.rolling(7 * 1440, min_periods=1440).median()
    return f


def build(start: str = "2026-06-01") -> pd.DataFrame:
    rows = load_rows()
    p = StrategyParams(hours=tuple(range(24)), minutes=(12, 13), cost_min=0.15, cost_max=0.35, legs=("DOG",))
    sig = extract_signals(rows, p).reset_index(drop=True)
    sig = sig[sig.open_ts >= int(pd.Timestamp(start, tz="UTC").timestamp())].copy()
    px = np.minimum(sig.cost.values + SLIP, 0.99)
    sig["net"] = sig.win.values - px - np.array([fee_per_contract("Kalshi", float(x)) for x in px])
    sig["entry_ts"] = sig.open_ts + 60 * sig.minute
    sig["dow"] = pd.to_datetime(sig.open_ts, unit="s", utc=True).dt.dayofweek
    sig["mid"] = (sig.bid + sig.ask) / 2
    sig["spread"] = sig.ask - sig.bid
    # quote path of the same market: mid 3 and 6 minutes before the decision, and the minute volume
    r = rows.copy()
    r["mid"] = (r.bid + r.ask) / 2
    key = r.set_index(["ticker", "minute"]).mid
    for k in (3, 6):
        idx = pd.MultiIndex.from_arrays([sig.ticker.values, (sig.minute - k).values])
        sig[f"dmid{k}"] = (sig.mid.values - key.reindex(idx).values)
    # the YES-side move seen from the dog's point of view (dog = the side priced < 50c): positive = the dog got MORE expensive recently
    yes_dog = (sig.side == "YES").values
    for k in (3, 6):
        sig[f"dog_move{k}"] = np.where(yes_dog, sig[f"dmid{k}"], -sig[f"dmid{k}"])
    vmin = r.set_index(["ticker", "minute"]).vol
    sig["vol_dec"] = vmin.reindex(pd.MultiIndex.from_arrays([sig.ticker.values, sig.minute.values])).values
    # spot features at the decision instant: the last COMPLETE minute bucket before it
    f = spot_features(load_spot())
    t = (sig.entry_ts.values // 60 * 60 - 60)
    for col in f.columns:
        sig[col] = f[col].reindex(t).values
    s_open = f.close.reindex((sig.open_ts.values - 60)).values
    sig["d_open"] = np.log(sig.close / s_open)                           # spot move since the window opened
    mk = pd.read_csv(D / "markets.csv", usecols=["ticker", "strike"])
    sig = sig.merge(mk, on="ticker", how="left")
    sig["d_strike"] = np.log(sig.close / sig.strike)
    tau = 15 - sig.minute
    sig["z_strike"] = sig.d_strike / (sig.sig30 * np.sqrt(tau))          # distance to the strike in realised-vol units (positive = spot above the strike)
    # dog-side distance: how far the spot is from the strike on the side that makes the dog win (always >= 0 when the dog is 'out of the money')
    sig["z_dog"] = np.where(yes_dog, -sig.z_strike, sig.z_strike)        # YES dog wins if the spot finishes above the strike: it needs +|z| ; z_dog = distance it still has to travel
    sig["vol_state30"] = sig.sig30 / sig.sig30_med7d
    sig["vol_state120"] = sig.sig120 / sig.sig120_med7d
    sig["move_state"] = sig.absret15 / sig.abs15_med7d
    # the previous window's outcome magnitude (spot move over the 15 minutes before this window opened)
    s_prev = f.close.reindex((sig.open_ts.values - 15 * 60 - 60)).values
    sig["prev_ret"] = np.log(s_open / s_prev)
    sig["prev_abs"] = sig.prev_ret.abs()
    sig["period"] = np.where(sig.open_ts < pd.Timestamp("2026-08-24", tz="UTC").timestamp(), "1 dead Jun1-Aug23",
                             np.where(sig.open_ts < pd.Timestamp("2026-09-28", tz="UTC").timestamp(), "2 hot Aug24-Sep27", "3 recent Sep28-now"))
    return sig.dropna(subset=["sig30", "sig30_med7d", "vr1440", "close"]).reset_index(drop=True)


if __name__ == "__main__":
    ds = build()
    print(ds.shape, ds.period.value_counts().to_dict())
    print(ds[["net", "vol_state30", "vr1440", "z_dog", "dog_move3"]].describe().round(3).to_string())
