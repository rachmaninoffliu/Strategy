import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm
from panel import build_panel
from strategy import fee_pc
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
P=build_panel()
HO=int(pd.Timestamp("2026-09-10",tz="UTC").timestamp())
P["split"]=np.where(P.open_ts<HO,"disc","hold")
print("markets disc/hold:",P[P.split=="disc"].ticker.nunique(),P[P.split=="hold"].ticker.nunique(), "| rows",len(P))
# ---- H1: opening minute.  candle 1 = first 60s of trading.  What do first quotes/prints look like vs the truth?
c1=P[P.minute==1].copy()
for col in ["bid_o","ask_o","px_o","px_h","px_l","bid_c","ask_c","px_c"]:
    s=c1[col].dropna(); 
    print(f"minute-1 {col:6s}: n={len(s):5d} mean={s.mean():.4f} sd={s.std():.4f}")
c1["mid_o"]=(c1.bid_o+c1.ask_o)/2
ok=c1[(c1.bid_o>0)&(c1.ask_o<1)&(c1.ask_o>=c1.bid_o)]
print("\nfirst-quote mid at open: n=%d mean=%.4f  vs realised P(yes)=%.4f"%(len(ok),ok.mid_o.mean(),ok.yes.mean()))
ok["b"]=pd.cut(ok.mid_o,[0,.3,.4,.45,.48,.52,.55,.6,.7,1])
t=ok.groupby("b",observed=True).agg(n=("yes","size"),mid=("mid_o","mean"),freq=("yes","mean"),spread=("ask_o",lambda s:(s-ok.loc[s.index,"bid_o"]).mean()))
t["gap_c"]=(t.freq-t.mid)*100; print(t.round(4).to_string())
# opening trade price (first print) vs outcome
f=c1[c1.px_o.notna()].copy(); f["b"]=pd.cut(f.px_o,[0,.3,.4,.45,.48,.52,.55,.6,.7,1])
t=f.groupby("b",observed=True).agg(n=("yes","size"),px=("px_o","mean"),freq=("yes","mean")); t["gap_c"]=(t.freq-t.px)*100
print("\nfirst PRINT price px_o vs outcome:"); print(t.round(4).to_string())
