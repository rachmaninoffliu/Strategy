"""Fetch daily bars for the optimizer and cross-check TQQQ against the TradingView dump.

TradingView charts show split-adjusted, NOT dividend-adjusted prices, so we use Yahoo 'Close' (auto_adjust=False),
which is split-adjusted only. TQQQ is compared to the exact bars dumped from TradingView.
"""
import json, os
import numpy as np
import pandas as pd
import yfinance as yf

OUT = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt"
os.makedirs(OUT, exist_ok=True)

SYMS = ["QQQ", "SPY", "TQQQ", "SPXL"]
for s in SYMS:
    df = yf.download(s, start="1998-01-01", end="2026-10-09", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df.to_csv(os.path.join(OUT, f"yf_{s}_1D.csv"))
    print(s, len(df), df.index[0].date(), df.index[-1].date(), "last close", round(float(df['Close'].iloc[-1]), 4))

# ---- cross-check TQQQ: TradingView dump vs Yahoo ----
tv = json.load(open(os.path.join(OUT, "tv_TQQQ_1D.json")))
b = pd.DataFrame(tv["bars"], columns=["t", "Open", "High", "Low", "Close", "Volume"])
# TradingView daily bar time = session open (09:30 ET) in UTC seconds -> use ET date
b["date"] = pd.to_datetime(b["t"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
b = b.set_index("date")
b.to_csv(os.path.join(OUT, "tv_TQQQ_1D.csv"))
y = pd.read_csv(os.path.join(OUT, "yf_TQQQ_1D.csv"), index_col="date", parse_dates=True)
j = b.join(y, lsuffix="_tv", rsuffix="_yf", how="inner")
print("\nTQQQ overlap bars:", len(j), "of TV", len(b), "/ YF", len(y))
for col in ["Open", "High", "Low", "Close"]:
    r = (j[col + "_tv"] / j[col + "_yf"] - 1).abs()
    print(f"{col}: max abs rel diff {r.max():.5f}  mean {r.mean():.6f}  share >0.1%: {(r > 0.001).mean():.4f}  share >0.5%: {(r > 0.005).mean():.4f}")
missing_tv = sorted(set(y.index) - set(b.index))[:5]
missing_yf = sorted(set(b.index) - set(y.index))[:5]
print("dates only in YF (first 5):", [str(d.date()) for d in missing_tv], "| only in TV:", [str(d.date()) for d in missing_yf])
