import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from panel import build_panel
from strategy import fee_pc
pd.set_option("display.width",250)
P=build_panel()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
P["mid"]=(P.bid_c+P.ask_c)/2
Y=P.assign(side="YES",cost=P.ask_c,win=P.yes,fav=P.mid>=0.5); N=P.assign(side="NO",cost=1-P.bid_c,win=1-P.yes,fav=P.mid<0.5)
R=pd.concat([Y,N]); R["pnl"]=R.win-R.cost-fee_pc(R.cost)
R["wk"]=pd.to_datetime(R.open_ts,unit="s").dt.to_period("W").astype(str)
def rule(a,b,fav,lo,hi):
    s=R[(R.minute>=a)&(R.minute<=b)&(R.fav==fav)&(R.cost>=lo)&(R.cost<hi)].sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first()
    return s
rules={"LATE FAV 12-13 ask .70-.95 (buy fav)":(12,13,True,.70,.95),
       "LATE FADE 12-13 dog .05-.30 (buy dog)":(12,13,False,.05,.30),
       "MID FAV 5-11 ask .90-.95 (buy fav)":(5,11,True,.90,.95),
       "MID FAV 3-11 ask .80-.90 (buy fav)":(3,11,True,.80,.90)}
out={}
for name,args in rules.items():
    s=rule(*args); g=s.groupby("wk").agg(n=("pnl","size"),ev=("pnl",lambda x:x.mean()*100),win=("win","mean"))
    out[name]=g
    print("\n",name,f" ALL: n={len(s)} EV={s.pnl.mean()*100:+.2f}c  t={s.pnl.mean()/s.pnl.std()*np.sqrt(len(s)):+.2f}  win={s.win.mean():.3f}")
    print(g.round(3).T.to_string())
