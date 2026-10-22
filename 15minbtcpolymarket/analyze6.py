import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from panel import build_panel
from strategy import fee_pc
pd.set_option("display.width",250)
P=build_panel()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)&P.sig.notna()&P.sig_long.notna()].copy()
P["mid"]=(P.bid_c+P.ask_c)/2
P["vr"]=P.sig/P.sig_long                 # short-horizon vol vs 3-day-halflife vol (known at decision time)
P["vrf"]=P.sig_fast/P.sig_long
Y=P.assign(side="YES",cost=P.ask_c,win=P.yes,fav=P.mid>=0.5); N=P.assign(side="NO",cost=1-P.bid_c,win=1-P.yes,fav=P.mid<0.5)
R=pd.concat([Y,N]); R["pnl"]=R.win-R.cost-fee_pc(R.cost)
def rule(a,b,fav,lo,hi):
    return R[(R.minute>=a)&(R.minute<=b)&(R.fav==fav)&(R.cost>=lo)&(R.cost<hi)].sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first()
rules={"LATE FAV 12-13 .70-.95":(12,13,True,.70,.95),"LATE DOG 12-13 .05-.30":(12,13,False,.05,.30),
       "LATE DOG 12-13 .10-.20":(12,13,False,.10,.20),"MID FAV 5-11 .90-.95":(5,11,True,.90,.95),"MID FAV 3-11 .80-.90":(3,11,True,.80,.90)}
for name,args in rules.items():
    s=rule(*args)
    s["vq"]=pd.qcut(s.vr,4,labels=["Q1 calm","Q2","Q3","Q4 hot"])
    g=s.groupby("vq",observed=True).agg(n=("pnl","size"),vr=("vr","mean"),win=("win","mean"),cost=("cost","mean"),ev_c=("pnl",lambda x:x.mean()*100),se=("pnl",lambda x:x.std()/np.sqrt(len(x))*100))
    g["t"]=g.ev_c/g.se
    print("\n",name,"(buckets by short-vs-long vol ratio at decision time)"); print(g.round(3).to_string())
