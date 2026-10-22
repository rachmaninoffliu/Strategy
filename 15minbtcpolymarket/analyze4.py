import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
from panel import build_panel, load_spot
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)

P=build_panel(); P=P[P.open_ts>=1785974400].copy()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
P["mid"]=(P.bid_c+P.ask_c)/2
P=P.sort_values(["ticker","minute"])
cl=lambda x:x.clip(.01,.99)
P["lm"]=np.log(cl(P.mid)/(1-cl(P.mid)))
g=P.groupby("ticker")
P["d1"]=P.lm-g.lm.shift(1)                # change in market logit over last minute
P["d2"]=P.lm-g.lm.shift(2)
P["rng"]=(P.px_h-P.px_l)                  # intra-minute trade-price range (jumpiness)
# spot z with basis correction (trailing mean of K - coinbase typical price; past data only)
sp=load_spot(); typ=(sp.open+sp.high+sp.low+sp.close)/4
kb=P.drop_duplicates("ticker")[["ticker","open_ts","strike"]].sort_values("open_ts")
kb["basis"]=kb.strike.values-typ.reindex(kb.open_ts.values-60).values
kb["basis_bar"]=kb.basis.rolling(96,min_periods=20).mean().shift(1)     # trailing 24h, excludes current
P=P.merge(kb[["ticker","basis_bar"]],on="ticker",how="left")
tau_eff=(P.tau_min-40/60).clip(lower=20/60)
P["z"]=np.log((P.S+P.basis_bar.fillna(0))/P.strike)/(P.sig*np.sqrt(tau_eff))
P["p_model"]=norm.cdf(P.z); print("mean(p_model-mid) after basis fix: %.2fc"%((P.p_model-P.mid).mean()*100))
P["late"]=((P.minute>=12)&(P.minute<=13)).astype(float)
P["lm_late"]=P.lm*P.late
P["half"]=np.where(P.open_ts<np.quantile(P.open_ts,0.5),0,1)
D=P.dropna(subset=["d1","d2","z","lm"]).copy()
def ll(y,p): p=np.clip(p,1e-4,1-1e-4); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
feats_sets={"market only":["lm"],
            "+ z":["lm","z"],
            "+ momentum d1,d2":["lm","d1","d2"],
            "+ late-slope":["lm","lm_late"],
            "+ z + d1 + late-slope":["lm","z","d1","lm_late"]}
tr=D[D.half==0]; te=D[D.half==1]
print("\nfit on H1 (%d markets), evaluate on H2 (%d markets)  |  base log-loss H2 (market mid) = %.5f"%(tr.ticker.nunique(),te.ticker.nunique(),ll(te.yes,te.mid)))
for name,fs in feats_sets.items():
    r=sm.Logit(tr.yes,sm.add_constant(tr[fs].clip(-8,8))).fit(disp=0,cov_type="cluster",cov_kwds={"groups":tr.ticker.astype("category").cat.codes})
    p=r.predict(sm.add_constant(te[fs].clip(-8,8)))
    coefs="  ".join(f"{k}={v:.3f}(t={r.tvalues[k]:.1f})" for k,v in r.params.items() if k!="const")
    print(f"{name:24s} H2 log-loss {ll(te.yes,p):.5f}  gain vs market {1e4*(ll(te.yes,te.mid)-ll(te.yes,p)):+.1f} bp-nats | {coefs}")
# late-window slope by minute
print("\nper-minute calibration slope of logit(yes) on market logit (1.0 = calibrated; <1 = market overconfident):")
rows=[]
for j,s in D.groupby("minute"):
    r=sm.Logit(s.yes,sm.add_constant(s[["lm"]].clip(-8,8))).fit(disp=0,cov_type="cluster",cov_kwds={"groups":s.ticker.astype("category").cat.codes})
    rows.append((j,len(s),r.params["lm"],r.bse["lm"],r.params["const"],r.bse["const"]))
print(pd.DataFrame(rows,columns=["minute","n","slope","se","intercept","se_i"]).round(3).to_string(index=False))
