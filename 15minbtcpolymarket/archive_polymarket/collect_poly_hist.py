"""Polymarket US btc-updown-15m 1-minute display prices for settled windows (GET /v1/price-history)."""
import csv, time, requests, sys
from datetime import datetime, timezone, timedelta
G="https://gateway.polymarket.us"
start=datetime(2026,9,21,0,0,tzinfo=timezone.utc); end=datetime.now(timezone.utc).replace(second=0,microsecond=0)-timedelta(minutes=20)
S=requests.Session(); out=[]; n=0; miss=0; ws=start; pause=0.0
t0=time.time()
with open("data/poly_hist.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["ws","ts","long","short"])
    while ws<end:
        slug=f"cpc-btc-updown-15m-{ws:%Y-%m-%d-%H%M}z"; s=int(ws.timestamp())
        for k in range(5):
            time.sleep(max(0,pause-time.time()))
            try:
                r=S.get(G+"/v1/price-history",params={"symbol":slug,"fidelity":1,"timestamp.startTimestamp":s,"timestamp.endTimestamp":s+900},timeout=30)
            except Exception:
                time.sleep(2); continue
            if r.status_code==429: pause=time.time()+3+2*k; continue
            if r.status_code==200:
                h=r.json().get("history",[])
                for p in h: w.writerow([s,p["timestamp"],p["longPrice"],p["shortPrice"]])
                if not h: miss+=1
            break
        n+=1; ws+=timedelta(minutes=15); time.sleep(0.25)
        if n%100==0: fh.flush(); print(n,"windows",miss,"empty",round(time.time()-t0),"s",flush=True)
print("done",n,"windows; empty",miss)
