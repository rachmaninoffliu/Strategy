import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from panel import build_panel
from strategy import fee_pc
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
P=build_panel()
P=P[(P.open_ts>=int(pd.Timestamp("2026-06-01",tz="UTC").timestamp()))&(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
dt=pd.to_datetime(P.open_ts,unit="s",utc=True); P["hr"]=dt.dt.hour; P["mo"]=dt.dt.strftime("%Y-%m"); P["wk"]=dt.dt.to_period("W").astype(str)
P["mid"]=(P.bid_c+P.ask_c)/2
yd=P.mid<0.5; P["cost"]=np.where(yd,P.ask_c,1-P.bid_c); P["win"]=np.where(yd,P.yes,1-P.yes)
L=P[P.minute.isin([12,13])&(P.cost>=.05)&(P.cost<=.30)&P.hr.isin([20,21,22,23])]
T=L.sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first(); T["pnl"]=T.win-T.cost-fee_pc(T.cost)
T["old"]=T.open_ts<1785974400
print("markets with 20-23h candles by period:",T.groupby("old").size().to_dict())
for name,sub in (("22-23 UTC",T[T.hr.isin([22,23])]),("20-21 UTC",T[T.hr.isin([20,21])])):
    print(f"\n== {name}: Jun1-Aug5 (OUT-OF-TIME, lower-liquidity regime) vs Aug6+")
    g=sub.groupby("old").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),se=("pnl",lambda x:x.std()/np.sqrt(len(x))*100),win=("win","mean"),cost=("cost","mean"))
    g["t"]=g.ev_c/g.se; print(g.round(3).to_string())
    print(sub[sub.old].groupby("mo").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),win=("win","mean")).round(2).T.to_string())
