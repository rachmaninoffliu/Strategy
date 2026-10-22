import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from panel import build_panel
from strategy import Calibrator, Config, simulate, summarize
pd.set_option("display.width",260); pd.set_option("display.max_columns",40)

P=build_panel()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
print("panel rows",len(P),"markets",P.ticker.nunique(),pd.to_datetime(P.open_ts.min(),unit="s").date(),"->",pd.to_datetime(P.open_ts.max(),unit="s").date())
fit=lambda tr: Calibrator.fit(tr)

rows=[]
def go(label,**kw):
    cfg=Config(**kw); tr=simulate(P,fit,cfg); s=summarize(tr,label=label); rows.append(s); return tr
go("FAV  only          slip0",legs=("FAV",))
go("FADE only          slip0",legs=("FADE",))
go("FAV+FADE           slip0",legs=("FAV","FADE"))
go("FAV  only          slip0.5c",legs=("FAV",),slip=0.005)
go("FADE only          slip0.5c",legs=("FADE",),slip=0.005)
go("FAV+FADE           slip0.5c",legs=("FAV","FADE"),slip=0.005)
go("FAV  only          slip1c",legs=("FAV",),slip=0.01)
go("FADE only          slip1c",legs=("FADE",),slip=0.01)
go("FAV+FADE           slip1c",legs=("FAV","FADE"),slip=0.01)
R=pd.DataFrame(rows).set_index("label")
print(R[["trades","per_day","win","avg_cost","ev_c","ret_per_trade","ci_lo","ci_hi","total_ret","maxdd","sharpe","avg_frac"]].round(3).to_string())
