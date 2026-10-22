"""Backfill candles (single-market route, historical partition) only for chosen UTC hours in a date range."""
import csv, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from collect import CANDLE_FIELDS, DATA, fetch_candles
start=int(pd.Timestamp(sys.argv[1],tz="UTC").timestamp()); end=int(pd.Timestamp(sys.argv[2],tz="UTC").timestamp()); hours=[int(h) for h in sys.argv[3].split(",")]
m=pd.read_csv(DATA/"markets.csv")
m=m[(m.open_ts>=start)&(m.open_ts<end)]; m=m[pd.to_datetime(m.open_ts,unit="s",utc=True).dt.hour.isin(hours)]
done=set(pd.read_csv(DATA/"candles.csv",usecols=["ticker"],on_bad_lines="skip").ticker.unique())
todo=[r for r in m.to_dict("records") if r["ticker"] not in done]
print(f"{len(todo)} markets to backfill (hours {hours})",flush=True)
n=0
with open(DATA/"candles.csv","a",newline="") as fh, ThreadPoolExecutor(4) as ex:
    w=csv.writer(fh)
    for res in ex.map(fetch_candles,todo):
        n+=1
        if res: w.writerows(res)
        if n%100==0: fh.flush(); print(f"  {n}/{len(todo)}",flush=True)
print("HOURS_DONE")
