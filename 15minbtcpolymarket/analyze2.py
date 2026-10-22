import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
from panel import build_panel
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)

P=build_panel(); P=P[P.open_ts>=1785974400].copy()
ok=(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)&P.S.notna()&P.sig.notna()
P=P[ok].copy()
P["mid"]=(P.bid_c+P.ask_c)/2
tau_eff=(P.tau_min-40/60).clip(lower=20/60)               # minutes; BRTI 60s-average settlement
P["z"]=np.log(P.S/P.strike)/(P.sig*np.sqrt(tau_eff))
P["p_model"]=norm.cdf(P.z)
P["lm"]=np.log(P.mid.clip(.005,.995)/(1-P.mid.clip(.005,.995)))
P["tr"]=P.open_ts<np.quantile(P.open_ts,0.5)
print("rows",len(P),"train markets",P[P.tr].ticker.nunique(),"test markets",P[~P.tr].ticker.nunique())

def ll(y,p): p=np.clip(p,1e-4,1-1e-4); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
for name,sub in (("train(first half)",P[P.tr]),("TEST(second half)",P[~P.tr])):
    print("\n==",name,"log-loss: market mid %.4f | raw gaussian model %.4f"%(ll(sub.yes,sub.mid),ll(sub.yes,sub.p_model)))
# 1) Does z add information beyond the market price?  logit(yes) ~ a + b*lm + c*z   (cluster by market)
for name,sub in (("ALL",P),("minute>=9",P[P.minute>=9]),("minute 5-8",P[(P.minute>=5)&(P.minute<=8)]),("minute<=4",P[P.minute<=4])):
    X=sm.add_constant(sub[["lm","z"]].clip(-8,8)); r=sm.Logit(sub.yes,X).fit(disp=0,cov_type="cluster",cov_kwds={"groups":sub.ticker.astype("category").cat.codes})
    print(f"{name:11s} n={len(sub):6d}  b_market={r.params['lm']:.3f} (se {r.bse['lm']:.3f})   c_z={r.params['z']:.3f} (se {r.bse['z']:.3f}, t={r.tvalues['z']:.2f})")
# 2) model-vs-market disagreement, by minute: z-model prob vs mid
P["gap"]=P.p_model-P.mid
print("\nmean(p_model - mid) by minute (c):"); print((P.groupby("minute").gap.mean()*100).round(2).to_string())
