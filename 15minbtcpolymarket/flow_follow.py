import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, pandas as pd
from strategy import fee_pc
pd.set_option("display.width",250)
series=sys.argv[1]; big=float(sys.argv[2])
DATA="data/"
mp="data/markets.csv" if series=="KXBTC15M" else f"data/multi/{series}_markets.csv"
m=pd.read_csv(mp).set_index("ticker"); m["yes"]=(m.result=="yes").astype(int)
t=pd.read_csv(f"data/tape/{series}.csv",on_bad_lines="skip").sort_values(["ticker","ts_ms"])
t=t.join(m[["open_ts","close_ts","yes"]],on="ticker")
t["sec"]=(t.ts_ms/1000-t.open_ts)                         # seconds since market open
t=t[(t.sec>=120)&(t.sec<=840)&(t.yes_price>0.03)&(t.yes_price<0.97)]
rows=[]
for tk,g in t.groupby("ticker"):
    g=g.reset_index(drop=True); ts=g.sec.values; side=g.taker_side.values; px=g.yes_price.values; ct=g["count"].values; y=g.yes.iloc[0]
    sig_idx=np.where(ct>=big)[0]
    last_used=-1e9
    for i in sig_idx:
        if ts[i]-last_used<30: continue                      # one signal per 30 s per market
        for lag in (2,5,15):
            j=np.searchsorted(ts,ts[i]+lag)
            # follower pays the ask: next taker trade on the SAME side at least `lag` later (its price is the ask then)
            k=j
            while k<len(ts) and (side[k]!=side[i]) and ts[k]-ts[i]<45: k+=1
            if k>=len(ts) or side[k]!=side[i] or ts[k]-ts[i]>=45: continue
            if side[i]=="yes": cost=px[k]; win=y
            else: cost=1-px[k]; win=1-y
            rows.append((tk,ts[i],lag,side[i],ct[i],px[i],cost,win,win-cost-float(fee_pc(cost)),y))
        last_used=ts[i]
d=pd.DataFrame(rows,columns=["ticker","sec","lag","side","size","px_sig","cost","win","pnl","yes"])
print(f"{series}: tape markets={t.ticker.nunique()}  big-trade threshold={big}  signals(lag2)={len(d[d.lag==2])}")
for lag,g in d.groupby("lag"):
    gm=g.groupby("ticker").pnl.mean()
    print(f"lag {lag:2d}s: n={len(g):5d} win={g.win.mean():.3f} cost={g.cost.mean():.3f} EV after fee={g.pnl.mean()*100:+.2f}c  t(mkt-clustered)={gm.mean()/gm.std()*np.sqrt(len(gm)):+.2f}")
g=d[d.lag==5]
print("by side:",g.groupby("side").pnl.agg(n="size",ev=lambda x:x.mean()*100).round(2).to_dict())
g=g.assign(zone=pd.cut(g.cost,[0,.2,.4,.6,.8,1]))
print(g.groupby("zone",observed=True).pnl.agg(n="size",ev=lambda x:x.mean()*100).round(2).T.to_string())
