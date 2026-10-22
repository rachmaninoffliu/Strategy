import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm
from panel import build_panel, load_markets
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",200)
P=build_panel()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)&(P.open_ts>=1785974400)].copy()   # liquid regime, Aug 6+
P["mid"]=(P.bid_c+P.ask_c)/2
dt=pd.to_datetime(P.open_ts,unit="s",utc=True); P["hr"]=dt.dt.hour
# realised 15-min vol by hour from the strike chain (bps of log return of consecutive strikes)
m=load_markets().sort_values("open_ts"); m=m[m.open_ts>=1785974400].copy(); m["ret"]=np.log(m.strike.shift(-1)/m.strike)*1e4
m["hr"]=pd.to_datetime(m.open_ts,unit="s",utc=True).dt.hour
real=m.groupby("hr").ret.agg(real_sd_bps="std", real_mad_bps=lambda s:s.abs().median())
# market-implied vol at minute 12 (3 min left): invert p = Phi(ln(S/K)/(sigma*sqrt(tau_eff)))
x=P[(P.minute==12)&(P.mid.between(.15,.85))].copy()
x["tau_eff"]=(3-40/60)
x["d"]=np.log(x.S/x.strike)*1e4                         # bps distance
x["sgn_ok"]=np.sign(x.d)==np.sign(x.mid-0.5)
x["z_imp"]=norm.ppf(x.mid)
x=x[x.sgn_ok&(x.z_imp.abs()>0.25)]
x["sig_imp_1m"]=(x.d.abs()/x.z_imp.abs()/np.sqrt(x.tau_eff))       # bps per sqrt(min)
x["sig_real_1m"]=x.sig*1e4
g=x.groupby("hr").agg(n=("mid","size"),sig_imp=("sig_imp_1m","median"),sig_real_ewma=("sig_real_1m","median"))
g["imp/real"]=g.sig_imp/g.sig_real_ewma
ctx=P[P.minute==12].groupby("hr").agg(spread_c=("ask_c",lambda s:(s-P.loc[s.index,"bid_c"]).mean()*100),mkt_vol_M=("volume_y",lambda s:s.mean()/1e6))
out=real.join(g).join(ctx); print(out.round(3).to_string())
# calibration slope on late-window quotes by hour
import statsmodels.api as sm
lm=lambda s:np.log(s.clip(.01,.99)/(1-s.clip(.01,.99)))
rows=[]
for h,s in P[(P.minute>=12)&(P.minute<=13)].groupby("hr"):
    r=sm.Logit(s.yes,sm.add_constant(lm(s.mid).to_frame("lm"))).fit(disp=0,cov_type="cluster",cov_kwds={"groups":s.ticker.astype("category").cat.codes})
    rows.append((h,len(s),r.params["lm"],r.bse["lm"]))
print("\nlate-window (min 12-13) calibration slope by UTC hour (1=calibrated):"); print(pd.DataFrame(rows,columns=["hr","n","slope","se"]).round(3).T.to_string())
