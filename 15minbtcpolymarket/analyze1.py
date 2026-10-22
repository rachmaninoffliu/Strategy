import numpy as np, pandas as pd
from panel import build_panel
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)

C=100
def fee_pc(p, C=C):          # Kalshi taker fee per contract for an order of C contracts (round-up to cent on the order)
    return np.ceil(0.07*C*p*(1-p)*100-1e-9)/100/C

P=build_panel()
P=P[P.open_ts>=1785974400].copy()          # post-cutoff regime (Aug 6 -> now): liquid
print("rows",len(P),"markets",P.ticker.nunique())
ok=(P.bid_c>0)&(P.ask_c<1)&(P.ask_c>=P.bid_c)&((P.ask_c-P.bid_c)<=0.06)
P=P[ok].copy(); print("valid-quote rows",len(P))
P["mid"]=(P.bid_c+P.ask_c)/2
P["spread"]=P.ask_c-P.bid_c
# buy YES at ask, buy NO at 1-bid
P["pnl_yes"]=P.yes-P.ask_c-fee_pc(P.ask_c)
P["no_ask"]=1-P.bid_c
P["pnl_no"]=(1-P.yes)-P.no_ask-fee_pc(P.no_ask)

print("\n== Calibration of mid price vs outcome (all minutes) ==")
P["mbin"]=pd.cut(P.mid,[0,.02,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,1.0])
cal=P.groupby("mbin",observed=True).agg(n=("yes","size"),mid=("mid","mean"),freq=("yes","mean"),spread=("spread","mean"))
cal["gap_c"]=(cal.freq-cal.mid)*100
print(cal.round(4).to_string())

print("\n== Unconditional taker EV per contract (after fee) by minute ==")
g=P.groupby("minute").agg(n=("yes","size"),ev_yes_c=("pnl_yes",lambda x:x.mean()*100),ev_no_c=("pnl_no",lambda x:x.mean()*100),spread_c=("spread",lambda x:x.mean()*100))
print(g.round(3).to_string())

print("\n== Buy-the-favourite: EV after fee by minute x ask bucket (favourite side only, cents/contract; win rate in brackets) ==")
fav_yes=P.mid>=0.5
F=pd.concat([
  P[fav_yes].assign(side="Y",cost=P.ask_c,win=P.yes,pnl=P.pnl_yes),
  P[~fav_yes].assign(side="N",cost=P.no_ask,win=1-P.yes,pnl=P.pnl_no)])
F["cbin"]=pd.cut(F.cost,[.5,.6,.7,.8,.9,.95,.98,1.0])
F["late"]=pd.cut(F.minute,[0,4,8,11,12,13,14],labels=["1-4","5-8","9-11","12","13","14"])
t=F.groupby(["late","cbin"],observed=True).agg(n=("win","size"),win=("win","mean"),cost=("cost","mean"),ev_c=("pnl",lambda x:x.mean()*100),se_c=("pnl",lambda x:x.std()/np.sqrt(len(x))*100))
t["z"]=t.ev_c/t.se_c
print(t.round(3).to_string())
