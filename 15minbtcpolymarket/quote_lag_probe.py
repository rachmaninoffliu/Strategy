"""Read-only probe: how stale is the live REST quote (what the live engine reads) compared with the 1-minute candle close the backtest uses?
Samples GET /markets/{ticker} at +0.5 s, +2 s and +6 s after every minute boundary of the next full 15-minute window, then (after the window has closed) reads the candles of that window
and prints, per minute, REST quote vs the candle close at the same minute.  Public Kalshi API only, no account, no orders."""
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, ".")
from trader.kalshi import KalshiQuotes

q = KalshiQuotes()
now = time.time()
w = (int(now) // 900 + 1) * 900                       # next window start
print(f"waiting for window {datetime.fromtimestamp(w, timezone.utc):%H:%M}Z", flush=True)
time.sleep(max(0, w - time.time() - 2))
tick = None
while tick is None:
    tick = q.window_ticker(w)
    if tick is None:
        time.sleep(0.5)
print("ticker", tick, flush=True)
rec = []
for k in range(1, 15):
    t_end = w + 60 * k
    for off in (0.5, 2.0, 6.0):
        time.sleep(max(0, t_end + off - time.time()))
        qq = q.quote(tick)
        rec.append(dict(minute=k, offset_s=off, rest_bid=qq[0] if qq else None, rest_ask=qq[1] if qq else None, taken=datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]))
time.sleep(max(0, w + 900 + 90 - time.time()))
from collect import fetch_candles
rows = fetch_candles(dict(ticker=tick, open_ts=w, close_ts=w + 900))
cd = pd.DataFrame(rows, columns=["ticker", "minute", "end_ts", "bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c", "px_o", "px_h", "px_l", "px_c", "volume"])
df = pd.DataFrame(rec).merge(cd[["minute", "bid_c", "ask_c"]], on="minute", how="left")
df["d_bid_c"] = (df.rest_bid - df.bid_c).round(3)
df["d_ask_c"] = (df.rest_ask - df.ask_c).round(3)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
for off, g in df.groupby("offset_s"):
    print(f"offset {off}s: mean |rest_bid - candle_close_bid| = {g.d_bid_c.abs().mean() * 100:.2f}c, median {g.d_bid_c.abs().median() * 100:.2f}c, max {g.d_bid_c.abs().max() * 100:.1f}c over {g.d_bid_c.notna().sum()} minutes")
df.to_csv("data/quote_lag_probe.csv", index=False)
