import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from maker_bt import simulate
from panel import load_spot
pd.set_option("display.width",250)
series="KXETH15M"
spot=load_spot(); cl=spot.close
res={}
for name,(q,th) in {"touch,q=0":(0,False),"touch,q=100":(100,False),"through":(0,True)}.items():
    d=simulate(series,queue=q,through=th)
    t_dec=d.open_ts+60*d.minute
    d["r1"]=np.log(cl.reindex(t_dec.values-60).values/cl.reindex(t_dec.values-120).values)*1e4      # last full minute (bps)
    d["r3"]=np.log(cl.reindex(t_dec.values-60).values/cl.reindex(t_dec.values-240).values)*1e4
    d["sig"]=spot.sig.reindex(t_dec.values-60).values*1e4
    d["a1"]=d.r1.abs()/d.sig; d["a3"]=d.r3.abs()/(d.sig*np.sqrt(3))
    d["calm"]=pd.qcut(d.a1,4,labels=["Q1 calmest","Q2","Q3","Q4 wildest"])
    d["vol_lvl"]=pd.qcut(d.sig,4,labels=["low vol","Q2","Q3","high vol"])
    res[name]=d
    print(f"\n=== {series} {name}: EV per contract (c) and adverse-selection (vs mid) by calmness of the last minute")
    g=d.groupby("calm",observed=True).agg(fills=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),vs_mid_c=("pnl_vs_mid",lambda x:x.mean()*100),
        t=("pnl",lambda x:x.mean()/x.std()*np.sqrt(len(x))))
    print(g.round(2).to_string())
    g=d.groupby("vol_lvl",observed=True).agg(fills=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100),vs_mid_c=("pnl_vs_mid",lambda x:x.mean()*100))
    print(g.round(2).to_string())
# the cleanest 'smart' passive rule: calm last minute AND far from 50/50 extremes excluded? just calm & mid-zone
d=res["touch,q=100"]; s=d[(d.calm=="Q1 calmest")]
print("\ncalmest-quartile fills, touch q=100: EV %.2fc, vs-mid %.2fc, n=%d, per-market mean t=%.2f"%(s.pnl.mean()*100,s.pnl_vs_mid.mean()*100,len(s),(lambda mm: mm.mean()/mm.std()*np.sqrt(len(mm)))(s.groupby('ticker').pnl.mean())))
