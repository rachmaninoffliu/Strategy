import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from panel import build_panel
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
P=build_panel()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)&(P.open_ts>=1785974400)].copy()
P["mid"]=(P.bid_c+P.ask_c)/2
P["lm"]=np.log(P.mid.clip(.01,.99)/(1-P.mid.clip(.01,.99)))
P["hr"]=pd.to_datetime(P.open_ts,unit="s",utc=True).dt.hour
cut=np.quantile(P.open_ts,0.5)
L=P[(P.minute>=12)&(P.minute<=13)]
def slopes(df):
    out={}
    for h,s in df.groupby("hr"):
        r=sm.Logit(s.yes,sm.add_constant(s[["lm"]])).fit(disp=0,cov_type="cluster",cov_kwds={"groups":s.ticker.astype("category").cat.codes})
        out[h]=(r.params["lm"],r.bse["lm"])
    return pd.DataFrame(out,index=["slope","se"]).T
A=slopes(L[L.open_ts<cut]); B=slopes(L[L.open_ts>=cut])
J=A.join(B,lsuffix="_1",rsuffix="_2")
print(J.round(3).T.to_string())
print("\ncorr of per-hour late slopes, half1 vs half2: %.3f  (Spearman %.3f)"%(J.slope_1.corr(J.slope_2),J.slope_1.corr(J.slope_2,method="spearman")))
# smooth: 4-hour blocks
P["b4"]=(P.hr//4)
L=P[(P.minute>=12)&(P.minute<=13)]
A=[];B=[]
for b,s in L.groupby("b4"):
    for name,sub,lst in (("1",s[s.open_ts<cut],A),("2",s[s.open_ts>=cut],B)):
        r=sm.Logit(sub.yes,sm.add_constant(sub[["lm"]])).fit(disp=0,cov_type="cluster",cov_kwds={"groups":sub.ticker.astype("category").cat.codes}); lst.append((b,r.params["lm"],r.bse["lm"]))
print("\n4-hour blocks (UTC)  half1 slope | half2 slope:")
for a,b in zip(A,B): print(f"  {a[0]*4:02d}-{a[0]*4+3:02d}   {a[1]:.3f} ({a[2]:.3f})  |  {b[1]:.3f} ({b[2]:.3f})")
# OOS log-loss: does an hour-block-specific late slope help out of sample?  fit on half1, score on half2
def ll(y,p): p=np.clip(p,1e-4,1-1e-4); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
tr=L[L.open_ts<cut]; te=L[L.open_ts>=cut].copy()
base=ll(te.yes,te.mid)
# global late slope
g=sm.Logit(tr.yes,sm.add_constant(tr[["lm"]])).fit(disp=0); p_g=g.predict(sm.add_constant(te[["lm"]]))
# block-specific
p_b=pd.Series(index=te.index,dtype=float)
for b,s in tr.groupby("b4"):
    r=sm.Logit(s.yes,sm.add_constant(s[["lm"]])).fit(disp=0); idx=te.b4==b
    p_b[idx]=r.predict(sm.add_constant(te.loc[idx,["lm"]]))
print("\nOOS (late window, half2) log-loss: market %.5f | global slope %.5f | block-specific slope %.5f"%(base,ll(te.yes,p_g),ll(te.yes,p_b)))
print("gain vs market (bp-nats): global %.1f, block %.1f"%((base-ll(te.yes,p_g))*1e4,(base-ll(te.yes,p_b))*1e4))
