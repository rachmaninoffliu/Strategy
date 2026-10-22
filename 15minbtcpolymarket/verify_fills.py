"""Replay each candidate signal against the REAL trade tape: what did takers on that side actually pay right after?"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from datetime import datetime
from collect import get
from strategy import fee_pc
import forward_test as ft
ft.START=int(pd.Timestamp("2026-08-06",tz="UTC").timestamp())
from panel import build_panel
P=build_panel()
P=P[(P.open_ts>=ft.START)&(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
P["hr"]=pd.to_datetime(P.open_ts,unit="s",utc=True).dt.hour; P["mid"]=(P.bid_c+P.ask_c)/2
P=P[P.minute.isin([12,13])]
yd=P.mid<0.5; P["yes_dog"]=yd; P["cost"]=np.where(yd,P.ask_c,1-P.bid_c); P["win"]=np.where(yd,P.yes,1-P.yes)
P=P[(P.cost>=.05)&(P.cost<=.30)]
T=P.sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first()
grp=sys.argv[1]
sel=T[T.hr.isin([22,23])] if grp=="target" else T[~T.hr.isin([20,21,22,23])].sample(240,random_state=1)
print(grp,"signals",len(sel),flush=True)
rows=[]
for i,r in enumerate(sel.itertuples()):
    t0=int(r.open_ts+60*r.minute)
    out=[]; cur=None
    for _ in range(4):
        d=get("/markets/trades",{"ticker":r.ticker,"limit":1000,"cursor":cur,"min_ts":t0-1,"max_ts":t0+75})
        if not d or not d.get("trades"): break
        out+=d["trades"]; cur=d.get("cursor")
        if not cur: break
    if not out: continue
    tr=pd.DataFrame(out); tr["ms"]=pd.to_datetime(tr.created_time,format="ISO8601",utc=True).astype("int64")//10**6
    tr["p"]=tr.yes_price_dollars.astype(float); tr["n"]=tr.count_fp.astype(float)
    side="yes" if r.yes_dog else "no"                       # we BUY the dog: YES if yes is dog else NO (taker_side=='no')
    s=tr[tr.taker_side==side].sort_values("ms")
    s["pay"]=s.p if side=="yes" else 1-s.p
    rec=dict(ticker=r.ticker,hr=r.hr,cand_cost=r.cost,win=r.win,n_trades=len(s))
    for dl in (0,2,5,10):
        ss=s[s.ms>=(t0+dl)*1000]
        if ss.empty: rec[f"pay{dl}"]=np.nan; continue
        cs=ss.n.cumsum(); k=np.searchsorted(cs.values,100)            # VWAP of the first ~100 contracts a taker on that side bought
        k=min(k,len(ss)-1); w=ss.iloc[:k+1]
        rec[f"pay{dl}"]=(w.pay*w.n).sum()/w.n.sum()
    rows.append(rec)
    if i%40==0: print(" ",i,len(rows),flush=True)
D=pd.DataFrame(rows)
print(f"\n[{grp}] markets with tape in window: {len(D)}")
print("candle-implied cost %.3f"%D.cand_cost.mean())
for dl in (0,2,5,10):
    x=D.dropna(subset=[f"pay{dl}"]); pay=x[f"pay{dl}"]
    pnl=x.win-pay-fee_pc(pay.values)
    gm=pnl
    print(f"delay {dl:2d}s: n={len(x):4d}  tape price paid by same-side takers {pay.mean():.3f} (vs candle {x.cand_cost.mean():.3f})  win {x.win.mean():.3f}  EV after fee {pnl.mean()*100:+.2f}c  t={pnl.mean()/pnl.std()*np.sqrt(len(x)):+.2f}")
D.to_csv(f"data/verify_{grp}.csv",index=False)
