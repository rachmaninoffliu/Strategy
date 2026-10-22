import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from panel import build_panel
from strategy import BucketCalibrator, Calibrator, Config, simulate, summarize
pd.set_option("display.width",270); pd.set_option("display.max_columns",40)
P=build_panel()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
print("panel rows",len(P),"markets",P.ticker.nunique(),pd.to_datetime(P.open_ts.min(),unit="s").date(),"->",pd.to_datetime(P.open_ts.max(),unit="s").date())
fit=lambda tr: BucketCalibrator.fit(tr)
rows=[]; logs={}
def go(label,**kw):
    cfg=Config(**kw); tr=simulate(P,fit,cfg); logs[label]=tr; rows.append(summarize(tr,label=label))
base=dict(shrink=1.0,min_edge=0.005,kelly_mult=0.5)
for slip in (0.0,0.005,0.01):
    s=f"slip{slip*100:.1f}c"
    go(f"FAV              {s}",legs=("FAV",),slip=slip,**base)
    go(f"FADE             {s}",legs=("FADE",),slip=slip,**base)
    go(f"FADE vr<1.1     {s}",legs=("FADE",),slip=slip,max_vr=1.1,**base)
    go(f"FAV+FADE vr<1.1 {s}",legs=("FAV","FADE"),slip=slip,max_vr=1.1,**base)
R=pd.DataFrame(rows).set_index("label")
cols=["trades","per_day","win","avg_cost","ev_c","ret_per_trade","ci_lo","ci_hi","total_ret","maxdd","sharpe","avg_frac"]
print(R[[c for c in cols if c in R.columns]].round(3).to_string())
