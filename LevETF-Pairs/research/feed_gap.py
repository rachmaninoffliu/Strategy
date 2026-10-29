"""Why do SIP and IEX data change the result so much?
 A. ticks of one session: how old is the last print at a 5-minute boundary, how long until the first print of the next bar, how many trades there are (IEX vs SIP)
 B. the backtest in the IEX-only world and the SIP world over the same window: bars with no trade, the close->open jump, and the three fill models
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\LevETF-Pairs\scripts")
os.chdir(r"C:\Users\xinyu\PycharmProjects\pythonProject\LevETF-Pairs\scripts")
import levpairs_bridge as BR
import levpairs_data as D
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockTradesRequest
from levpairs_ui.presets import DEFAULT_PRESET, cfgs

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
NYTZ = "America/New_York"

# ── A. ticks ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
cl = D.data_client()
day = "2026-10-23"
t0, t1 = pd.Timestamp(f"{day} 10:00", tz=NYTZ), pd.Timestamp(f"{day} 14:00", tz=NYTZ)
bounds = pd.date_range(t0 + pd.Timedelta(minutes=5), t1, freq="5min")
rows = []
for sym in D.SYMBOLS:
    for feed in ("sip", "iex"):
        tr = cl.get_stock_trades(StockTradesRequest(symbol_or_symbols=sym, start=t0.tz_convert("UTC"), end=(t1 + pd.Timedelta(minutes=1)).tz_convert("UTC"), feed=DataFeed.SIP if feed == "sip" else DataFeed.IEX)).data.get(sym, [])
        ts = pd.DatetimeIndex([x.timestamp for x in tr]).tz_convert(NYTZ)
        age, wait = [], []
        for b in bounds:
            i = ts.searchsorted(b, side="left")
            if 0 < i < len(ts):
                age.append((b - ts[i - 1]).total_seconds()); wait.append((ts[i] - b).total_seconds())
        rows.append(dict(sym=sym, feed=feed, trades_per_min=len(ts) / 240.0, age_med=np.median(age), age_p90=np.quantile(age, 0.9), wait_med=np.median(wait), wait_p90=np.quantile(wait, 0.9)))
tk = pd.DataFrame(rows)
print("A. one session, 10:00-14:00 ET, 48 five-minute boundaries")
print(tk.pivot(index="sym", columns="feed").round(2).to_string())
share = tk.pivot(index="sym", columns="feed", values="trades_per_min")
print("IEX trades as a share of SIP trades:", (share["iex"] / share["sip"]).round(3).to_dict())

# ── B. backtest worlds ────────────────────────────────────────────────────────────────────────────────────────────────────
sc, c = cfgs(DEFAULT_PRESET)
c = replace(c, start="2020-09-01")
out = {}
for feed in ("sip", "iex"):
    panel = {D.pair_name(p): D.load_pairdata(p, 5, feed, start="2020-08-03", max_missing=1.0) for p in D.PAIRS}
    sigs = BR.get_signals(panel, sc, feed, log=lambda m: None)
    miss = {k: (d.L_miss.mean(), d.B_miss.mean()) for k, d in panel.items()}
    gaps = {}
    for k, sg in sigs.items():
        g = np.abs(sg.eps_gap[:, 1:]) * 1e4
        gaps[k] = float(np.nanmean(g))
    tab, g2 = BR.fill_comparison(panel, sigs, c, split="2023-01-01")
    out[feed] = (tab, g2, miss, gaps)
    print(f"\nB. {feed.upper()} world, 5-minute bars, 2020-09 .. 2026-10")
    print("   empty 5-minute bars (3x leg, 1x leg):", {k: (round(a, 3), round(b, 3)) for k, (a, b) in miss.items()})
    print("   mean |close -> next open| of the spread, all bars (bp):", {k: round(v, 2) for k, v in gaps.items()})
    print(tab[["sharpe", "trades", "gross_bp", "net_bp"]].round(2).to_string())
    print("   mean move in the trade's favour close -> open: entry %+.2f bp, exit %+.2f bp, trades %d" % (g2["gap_in"], g2["gap_out"], g2["trades"]))
