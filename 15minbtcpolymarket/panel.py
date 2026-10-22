"""Build the decision panel: one row per (market, decision-minute j) with Kalshi quotes, spot proxy and vol.

Timing convention (no look-ahead):
  market opens at T0 (strike K = BRTI 60s avg ending T0) and settles at T0+900s.
  Decision j (1..14) is taken at T0+60j using the quote at the CLOSE of Kalshi candle j and the Coinbase
  1-minute close that ends at the same instant (candle starting T0+60j-60).  Sigma uses only closes <= that instant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


def load_markets() -> pd.DataFrame:
    m = pd.read_csv(DATA / "markets.csv").sort_values("open_ts").reset_index(drop=True)
    m["yes"] = (m.result == "yes").astype(int)
    nxt = m.strike.shift(-1)
    contiguous = (m.open_ts.shift(-1) - m.open_ts) == 900
    m["final"] = np.where(contiguous & (nxt > 1000), nxt, m.expiration_value)
    m = m[(m.strike > 1000)].copy()
    return m


def load_spot() -> pd.DataFrame:
    s = pd.read_csv(DATA / "cb_1m.csv").sort_values("ts").drop_duplicates("ts")
    s = s.set_index("ts")
    grid = np.arange(s.index.min(), s.index.max() + 60, 60)
    s = s.reindex(grid)
    s["close"] = s.close.ffill(limit=5)
    s["open"] = s.open.fillna(s.close)
    s["high"] = s.high.fillna(s.close)
    s["low"] = s.low.fillna(s.close)
    lr = np.log(s.close).diff()
    # EWMA 1-minute vol (half-life 90 min); shift is handled by only reading values at/<= the decision instant
    var = (lr ** 2).ewm(halflife=90, min_periods=60).mean()
    s["sig"] = np.sqrt(var)
    s["sig_fast"] = np.sqrt((lr ** 2).ewm(halflife=20, min_periods=30).mean())
    s["sig_long"] = np.sqrt((lr ** 2).ewm(halflife=3 * 1440, min_periods=1440).mean())
    s["lr"] = lr
    return s


def build_panel(min_month: str | None = None) -> pd.DataFrame:
    m = load_markets()
    c = pd.read_csv(DATA / "candles.csv", on_bad_lines="skip")
    c = c[(c.minute >= 1) & (c.minute <= 15)]
    spot = load_spot()
    cl = spot.close
    sig = spot.sig

    rows = []
    cm = c.merge(m[["ticker", "open_ts", "close_ts", "strike", "final", "yes", "volume"]], on="ticker", how="inner")
    cm = cm[cm.minute <= 14].copy()
    t_dec = cm.open_ts + 60 * cm.minute          # decision instant (end of candle j)
    idx_end = t_dec - 60                         # Coinbase candle that ENDS at t_dec starts at t_dec-60
    cm["S"] = cl.reindex(idx_end.values).values
    cm["sig"] = sig.reindex(idx_end.values).values
    cm["sig_fast"] = spot.sig_fast.reindex(idx_end.values).values
    cm["sig_long"] = spot.sig_long.reindex(idx_end.values).values
    # strike proxy on Coinbase: avg of the 1-min typical price over the minute before open -> basis offset
    o_idx = (cm.open_ts - 60).values
    cm["cb_k"] = ((spot.open + spot.high + spot.low + spot.close) / 4).reindex(o_idx).values
    cm["cb_k_close"] = cl.reindex(o_idx).values
    cm["S0"] = cl.reindex(cm.open_ts.values - 60).values   # Coinbase close at T0
    cm["vr"] = cm.sig / cm.sig_long                    # short vs long realised-vol ratio, known at decision time
    cm["tau_min"] = 15 - cm.minute
    cm["dt"] = pd.to_datetime(cm.open_ts, unit="s", utc=True)
    cm["month"] = cm.dt.dt.strftime("%Y-%m")
    return cm


if __name__ == "__main__":
    p = build_panel()
    print(p.shape)
    print(p.head())
