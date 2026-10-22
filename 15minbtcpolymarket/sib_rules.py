import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multi_calib import load_all
from strategy import fee_pc
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",300)
P=load_all()
HO=int(pd.Timestamp("2026-09-10",tz="UTC").timestamp()); P["split"]=np.where(P.open_ts<HO,"disc","hold")
Y=P.assign(side="YES",cost=P.ask_c,win=P.yes,fav=P.mid>=0.5); N=P.assign(side="NO",cost=1-P.bid_c,win=1-P.yes,fav=P.mid<0.5)
R=pd.concat([Y,N]); R["pnl"]=R.win-R.cost-fee_pc(R.cost)
def rule(df,a,b,fav,lo,hi):
    return df[(df.minute>=a)&(df.minute<=b)&(df.fav==fav)&(df.cost>=lo)&(df.cost<hi)].sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first()
RULES={"FAV 3-8 ask .80-.90":(3,8,True,.80,.90),"FAV 3-11 ask .90-.97":(3,11,True,.90,.97),"FAV 9-11 .80-.95":(9,11,True,.80,.95),"LATE FAV 12-13 .70-.95":(12,13,True,.70,.95),"LATE DOG 12-13 .05-.30":(12,13,False,.05,.30),"FAV 1-4 .60-.80":(1,4,True,.60,.80)}
sib=R[R.series!="KXBTC15M"]
for name,args in RULES.items():
    s=rule(sib,*args)
    def stat(x): return pd.Series({"n":len(x),"win":x.win.mean(),"cost":x.cost.mean(),"ev_c":x.pnl.mean()*100,"t_win":x.pnl.mean()/x.pnl.std()*np.sqrt(len(x))})
    # cluster-robust t: aggregate pnl by window (same-window assets are correlated)
    byw=s.groupby("open_ts").pnl.mean(); tw=byw.mean()/byw.std()*np.sqrt(len(byw))
    print(f"\n== {name}   ALL siblings: n={len(s)} win={s.win.mean():.3f} cost={s.cost.mean():.3f} EV={s.pnl.mean()*100:+.2f}c  t(window-clustered)={tw:+.2f}")
    print(s.groupby("split").apply(stat).round(3).T.to_string())
    print(s.groupby("series").apply(stat).round(3).T.to_string())
