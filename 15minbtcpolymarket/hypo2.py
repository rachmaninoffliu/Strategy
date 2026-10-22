import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from panel import build_panel
from strategy import fee_pc
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
P=build_panel()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
P["mid"]=(P.bid_c+P.ask_c)/2
HO=int(pd.Timestamp("2026-09-10",tz="UTC").timestamp()); P["split"]=np.where(P.open_ts<HO,"disc","hold")
dt=pd.to_datetime(P.open_ts,unit="s",utc=True); P["hr"]=dt.dt.hour; P["dow"]=dt.dt.dayofweek
P["blk"]=pd.cut(P.hr,[-1,3,7,11,15,19,23],labels=["00-03","04-07","08-11","12-15","16-19","20-23"])   # UTC
P["wkend"]=np.where(P.dow>=5,"wkend","wkday")
Y=P.assign(side="YES",cost=P.ask_c,win=P.yes,fav=P.mid>=0.5); N=P.assign(side="NO",cost=1-P.bid_c,win=1-P.yes,fav=P.mid<0.5)
R=pd.concat([Y,N]); R["pnl"]=R.win-R.cost-fee_pc(R.cost)
def rule(df,a,b,fav,lo,hi):
    return df[(df.minute>=a)&(df.minute<=b)&(df.fav==fav)&(df.cost>=lo)&(df.cost<hi)].sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first()
RULES={"FAV 3-11 ask>=.80":(3,11,True,.80,1.0),"FAV 5-11 .90-.95":(5,11,True,.90,.95),"LATE FAV 12-13 .70-.95":(12,13,True,.70,.95),"LATE DOG 12-13 .05-.30":(12,13,False,.05,.30)}
def tab(s,key):
    g=s.groupby(key,observed=True).agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),se=("pnl",lambda x:x.std()/np.sqrt(len(x))*100),win=("win","mean"))
    g["t"]=g.ev_c/g.se; return g
for name,args in RULES.items():
    print("\n=====",name)
    for key in ("blk","wkend"):
        d=tab(rule(R[R.split=="disc"],*args),key); h=tab(rule(R[R.split=="hold"],*args),key)
        out=d[["n","ev_c","t"]].join(h[["n","ev_c","t"]],lsuffix="_disc",rsuffix="_hold")
        print(out.round(2).to_string())
