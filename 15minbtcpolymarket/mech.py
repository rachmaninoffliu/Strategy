import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from panel import load_spot, load_markets
pd.set_option("display.width",250)
s=load_spot(); cl=s.close
m=load_markets(); m=m[(m.open_ts>=1785974400)].copy()
t0=m.open_ts.values
def px(minute): return cl.reindex(t0+60*minute-60).values       # Coinbase close at end of minute `minute` (ts = start of that candle)
m["S0"]=px(0); m["S9"]=px(9); m["S12"]=px(12); m["S15"]=px(15)
m["d12"]=np.log(m.S12/m.strike)*1e4                 # displacement from the strike (bps) at minute 12
m["r_prev3"]=np.log(m.S12/m.S9)*1e4                 # last 3-min return before decision
m["r_next3"]=np.log(m.S15/m.S12)*1e4                # next 3-min return (what the market is pricing)
m["hr"]=pd.to_datetime(m.open_ts,unit="s",utc=True).dt.hour
m["blk"]=pd.cut(m.hr,[-1,3,7,11,15,19,23],labels=["00-03","04-07","08-11","12-15","16-19","20-23"])
m=m.dropna(subset=["d12","r_prev3","r_next3"])
print("n",len(m))
rows=[]
for b,g in m.groupby("blk",observed=True):
    X=sm.add_constant(g[["d12"]]); r=sm.OLS(g.r_next3,X).fit(cov_type="HC1")
    X2=sm.add_constant(g[["r_prev3"]]); r2=sm.OLS(g.r_next3,X2).fit(cov_type="HC1")
    vr=g.r_next3.var()/ (g.r_prev3.var())
    rows.append((b,len(g),r.params["d12"],r.tvalues["d12"],r2.params["r_prev3"],r2.tvalues["r_prev3"],g.r_next3.std(),g.d12.std()))
print(pd.DataFrame(rows,columns=["blk","n","beta_disp","t","beta_prev3","t","sd_next3_bps","sd_disp_bps"]).round(3).to_string(index=False))
# hour-level within 20-23 and halves
cut=np.quantile(m.open_ts,.5)
for name,g in (("20-23 UTC half1",m[(m.blk=="20-23")&(m.open_ts<cut)]),("20-23 UTC half2",m[(m.blk=="20-23")&(m.open_ts>=cut)]),("other half1",m[(m.blk!="20-23")&(m.open_ts<cut)]),("other half2",m[(m.blk!="20-23")&(m.open_ts>=cut)])):
    r=sm.OLS(g.r_next3,sm.add_constant(g[["d12"]])).fit(cov_type="HC1"); r2=sm.OLS(g.r_next3,sm.add_constant(g[["r_prev3"]])).fit(cov_type="HC1")
    print(f"{name:16s} n={len(g):5d}  next3 ~ displacement: beta={r.params['d12']:+.3f} (t={r.tvalues['d12']:+.2f}) | next3 ~ prev3: beta={r2.params['r_prev3']:+.3f} (t={r2.tvalues['r_prev3']:+.2f})")
