import numpy as np, pandas as pd, itertools
from scipy.stats import norm
from panel import build_panel
pd.set_option("display.width",250); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",400)
C=100
def fee_pc(p): return np.ceil(0.07*C*p*(1-p)*100-1e-9)/100/C

P=build_panel(); P=P[P.open_ts>=1785974400].copy()
P=P[(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)].copy()
P["mid"]=(P.bid_c+P.ask_c)/2
# two rows per decision: buy YES at ask / buy NO at (1-bid)
Y=P.assign(side="YES",cost=P.ask_c,win=P.yes,fav=P.mid>=0.5)
N=P.assign(side="NO",cost=1-P.bid_c,win=1-P.yes,fav=P.mid<0.5)
R=pd.concat([Y,N]); R["fee"]=fee_pc(R.cost)
R["half"]=np.where(R.open_ts<np.quantile(P.open_ts,0.5),"H1","H2")
R["month"]=R.month

def run(minute_lo,minute_hi,fav,lo,hi,slip=0.0):
    s=R[(R.minute>=minute_lo)&(R.minute<=minute_hi)&(R.fav==fav)&(R.cost>=lo)&(R.cost<hi)].sort_values(["ticker","minute"])
    s=s.groupby("ticker",as_index=False).first()          # sequential rule: first qualifying minute, ONE trade per market
    s["pnl"]=s.win-(s.cost+slip)-fee_pc(s.cost+slip)
    return s

rules=[]
for (a,b) in ((1,4),(5,8),(9,11),(12,13)):
    for fav in (True,False):
        edges=[(.5,.7),(.7,.8),(.8,.9),(.9,.95),(.95,.98),(.98,1.0)] if fav else [(.0,.05),(.05,.1),(.1,.2),(.2,.3),(.3,.5)]
        for lo,hi in edges:
            s=run(a,b,fav,lo,hi)
            if len(s)<150: continue
            h1=s[s.half=="H1"].pnl; h2=s[s.half=="H2"].pnl
            s1=run(a,b,fav,lo,hi,0.01)
            rules.append(dict(mins=f"{a}-{b}",role="FAV" if fav else "DOG",ask=f"{lo:.2f}-{hi:.2f}",n=len(s),win=s.win.mean(),cost=s.cost.mean(),
                              ev_c=s.pnl.mean()*100,se_c=s.pnl.std()/np.sqrt(len(s))*100,ev_H1=h1.mean()*100,ev_H2=h2.mean()*100,n1=len(h1),n2=len(h2),ev_slip1c=s1.pnl.mean()*100))
T=pd.DataFrame(rules); T["t"]=T.ev_c/T.se_c
print(T.round(3).to_string(index=False))
print("\nrules:",len(T)," with EV>0:",(T.ev_c>0).sum()," with t>2:",(T.t>2).sum()," EV>0 in BOTH halves:",((T.ev_H1>0)&(T.ev_H2>0)).sum(), " EV>0 after +1c slippage:",(T.ev_slip1c>0).sum())
