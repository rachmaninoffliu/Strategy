import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
from panel import build_panel
from strategy import fee_pc
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
P=build_panel()
P=P[(P.open_ts>=1785974400)&(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
dt=pd.to_datetime(P.open_ts,unit="s",utc=True); P["hr"]=dt.dt.hour; P["dow"]=dt.dt.dayofweek; P["wk"]=dt.dt.to_period("W").astype(str)
P["mid"]=(P.bid_c+P.ask_c)/2
yd=P.mid<0.5; P["cost"]=np.where(yd,P.ask_c,1-P.bid_c); P["win"]=np.where(yd,P.yes,1-P.yes)
L=P[P.minute.isin([12,13])&(P.cost>=.05)&(P.cost<=.30)]
T=L.sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first(); T["pnl"]=T.win-T.cost-fee_pc(T.cost)
R=T[T.hr.isin([22,23])]
print("R1 trades",len(R),"EV %.2fc win %.3f cost %.3f"%(R.pnl.mean()*100,R.win.mean(),R.cost.mean()))
print("\nby week:"); print(R.groupby("wk").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),win=("win","mean")).round(2).T.to_string())
print("\nby weekday (0=Mon):"); print(R.groupby("dow").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),win=("win","mean")).round(2).T.to_string())
print("\nby minute:"); print(R.groupby("minute").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100)).round(2).T.to_string())
print("\nby cost bucket:"); R["cb"]=pd.cut(R.cost,[.05,.10,.15,.20,.25,.30]); print(R.groupby("cb",observed=True).agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),win=("win","mean"),cost=("cost","mean")).round(3).T.to_string())
# bootstrap by day (clusters), and leave-one-week-out
R["day"]=pd.to_datetime(R.open_ts,unit="s",utc=True).dt.date
dm=R.groupby("day").pnl.agg(["sum","size"]); rng=np.random.default_rng(0)
bs=[]
for _ in range(4000):
    ix=rng.integers(0,len(dm),len(dm)); bs.append(dm["sum"].values[ix].sum()/dm["size"].values[ix].sum())
print("\nday-block bootstrap EV/contract: mean %.2fc  95%% CI [%.2f, %.2f]c  P(EV<=0)=%.4f"%(np.mean(bs)*100,np.percentile(bs,2.5)*100,np.percentile(bs,97.5)*100,np.mean(np.array(bs)<=0)))
# model check: what does a spot-based Gaussian say about these dog trades?
R2=R.merge(P[["ticker","minute","S","strike","sig","tau_min"]].drop_duplicates(["ticker","minute"]),on=["ticker","minute"],how="left",suffixes=("","_p"))
tau=(R2.tau_min-40/60).clip(lower=20/60); z=np.log(R2.S/R2.strike)/(R2.sig*np.sqrt(tau)); pm=norm.cdf(z)
yd2=R2.cost.values  # dog cost
p_yes_model=pm; dog_is_yes=(R2.win==R2.yes)&False
# dog side: if dog is YES then cost=ask(yes). identify via mid: recompute
dog_yes=(R2.cost.values==R2.ask_c.values)
p_dog_model=np.where(dog_yes,p_yes_model,1-p_yes_model)
print("\nspot-model P(dog wins): mean %.3f | market-implied (cost) %.3f | realised win %.3f"%(np.nanmean(p_dog_model),R2.cost.mean(),R2.win.mean()))
