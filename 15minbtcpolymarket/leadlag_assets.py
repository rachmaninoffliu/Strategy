import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from multi_calib import load_all
from strategy import fee_pc
pd.set_option("display.width",250)
P=load_all()
P["key"]=P.win.astype(str)+"_"+P.minute.astype(str)
btc=P[P.series=="KXBTC15M"][["win","minute","lm","yes"]].rename(columns={"lm":"lm_btc","yes":"yes_btc"})
# previous-minute BTC logit (for change)
btc=btc.sort_values(["win","minute"]); btc["lm_btc_prev"]=btc.groupby("win").lm_btc.shift(1); btc["d_btc"]=btc.lm_btc-btc.lm_btc_prev
S=P[P.series!="KXBTC15M"].merge(btc,on=["win","minute"],how="inner")
S=S.sort_values(["ticker","minute"]); S["lm_prev"]=S.groupby("ticker").lm.shift(1); S["d_own"]=S.lm-S.lm_prev
HO=int(pd.Timestamp("2026-09-10",tz="UTC").timestamp()); S["half"]=np.where(S.open_ts<HO,"disc","hold")
print("sibling rows",len(S))
def fit(df,cols):
    return sm.Logit(df.yes,sm.add_constant(df[cols].clip(-8,8))).fit(disp=0,cov_type="cluster",cov_kwds={"groups":df.win})
for ser in ["ALL"]+sorted(S.series.unique()):
    d=S if ser=="ALL" else S[S.series==ser]
    d=d.dropna(subset=["lm_btc","d_btc","d_own"])
    r=fit(d,["lm","lm_btc"]); r2=fit(d,["lm","d_btc","d_own"])
    print(f"{ser:10s} n={len(d):6d} | yes ~ own_lm + btc_lm: own={r.params['lm']:.3f} btc={r.params['lm_btc']:+.3f} (t={r.tvalues['lm_btc']:+.2f}) | with changes: btc_d={r2.params['d_btc']:+.3f} (t={r2.tvalues['d_btc']:+.2f}) own_d={r2.params['d_own']:+.3f} (t={r2.tvalues['d_own']:+.2f})")
# economic size: rows where BTC says strongly one way and sibling quote disagrees
d=S.dropna(subset=["lm_btc"]).copy()
d["dis"]=(1/(1+np.exp(-d.lm_btc)))-d.mid
big=d[(d.dis.abs()>0.25)&d.minute.between(3,12)].copy()
big["buy_yes"]=big.dis>0
big["cost"]=np.where(big.buy_yes,big.ask_c,1-big.bid_c); big["win"]=np.where(big.buy_yes,big.yes,1-big.yes)
big["pnl"]=big.win-big.cost-fee_pc(big.cost)
g=big.sort_values(["ticker","minute"]).groupby("ticker",as_index=False).first()
print("\nrule: sibling quote disagrees with BTC quote by >25c (BTC-implied P minus sibling mid) -> buy the BTC-implied side, first signal per market")
print("n=%d  win=%.3f  cost=%.3f  EV=%+.2fc  t(win-clustered)=%+.2f"%(len(g),g.win.mean(),g.cost.mean(),g.pnl.mean()*100,g.groupby('win').pnl.mean().mean()/g.groupby('win').pnl.mean().std()*np.sqrt(g.win.nunique()) if False else g.groupby('open_ts').pnl.mean().mean()/g.groupby('open_ts').pnl.mean().std()*np.sqrt(g.open_ts.nunique())))
print(g.groupby("half").agg(n=("pnl","size"),ev_c=("pnl",lambda x:x.mean()*100)).round(2).to_string())
