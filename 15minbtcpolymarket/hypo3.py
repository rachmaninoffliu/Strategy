import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from panel import build_panel, load_spot
from strategy import fee_pc
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",200)
P=build_panel()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
P["mid"]=(P.bid_c+P.ask_c)/2
dt=pd.to_datetime(P.open_ts,unit="s",utc=True); P["hr"]=dt.dt.hour; P["slot"]=dt.dt.strftime("%H:%M"); P["date"]=dt.dt.date; P["wk"]=dt.dt.to_period("W").astype(str)
Y=P.assign(side="YES",cost=P.ask_c,win=P.yes,fav=P.mid>=0.5); N=P.assign(side="NO",cost=1-P.bid_c,win=1-P.yes,fav=P.mid<0.5)
R=pd.concat([Y,N]); R["pnl"]=R.win-R.cost-fee_pc(R.cost)
def rule(df,a,b,fav,lo,hi): return df[(df.minute>=a)&(df.minute<=b)&(df.fav==fav)&(df.cost>=lo)&(df.cost<hi)].sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first()
LF=rule(R,12,13,True,.70,.95); LD=rule(R,12,13,False,.05,.30)
for nm,s in (("LATE FAV buy .70-.95",LF),("LATE DOG buy .05-.30",LD)):
    s["blk"]=np.where(s.hr.between(20,23),"20-23","other")
    print("\n=====",nm)
    print(s.groupby("blk").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),t=("pnl",lambda x:x.mean()/x.std()*np.sqrt(len(x))),win=("win","mean")).round(2).to_string())
    b=s[s.blk=="20-23"]
    print("by UTC hour:");print(b.groupby("hr").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),win=("win","mean")).round(2).T.to_string())
    print("by 15-min slot:");print(b.groupby("slot").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100)).round(1).T.to_string())
    print("by week (20-23 only):");print(b.groupby("wk").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100)).round(2).T.to_string())
    print("share of weeks with the expected sign:", (b.groupby("wk").pnl.mean()<0).mean() if "FAV" in nm else (b.groupby("wk").pnl.mean()>0).mean())
# what is different about 20-23 UTC?  vol, volume, spread, price-path
sp=load_spot()
P["blk"]=np.where(P.hr.between(20,23),"20-23","other")
m1=P[P.minute==12].copy()
print("\ncontext at minute 12:"); print(m1.groupby("blk").agg(sig_1m_bps=("sig",lambda s:s.mean()*1e4),vr=("vr","mean"),spread_c=("ask_c",lambda s:(s-m1.loc[s.index,'bid_c']).mean()*100),mkt_volume=("volume","mean")).round(3).to_string())
