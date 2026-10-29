import sys, pickle, os, time
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import statarb_data as D, statarb_core as C, statarb_sim as S

SP = os.path.dirname(os.path.abspath(__file__))
PAPER = dict(neutral=False, event_filter=False, earn_filter=False, weight_pow=0.0, shock=False, band=0.0, scale_band=0.0, vol_target=1.0, g_init=1.0, g_min=1.0, g_max=1.0, tc_bps=0.0,
             borrow_bps=0.0, name_cap=1.0)

def job(args):
    name, kw = args
    panel = D.load_panel()
    sig = pickle.load(open(os.path.join(SP, f"sig_{name}.pkl"), "rb"))
    p = replace(C.PortCfg(), **{**PAPER, **kw})
    r = S.simulate(sig, panel, p)
    d = r.daily
    ret = r.ret
    IS, OOS = ret[:"2021-12-31"], ret["2022-01-01":]
    mi, mo, ma = S.metrics(IS), S.metrics(OOS), S.metrics(ret)
    return (name, kw, ma["sharpe"], mi["sharpe"], mo["sharpe"], d.gross_ret.mean() * 252, d.turnover.mean(), d.n_long.mean(), d.n_short.mean(), ma["vol"])

if __name__ == "__main__":
    names = sys.argv[1:] or ["pca15"]
    jobs = []
    for n in names:
        jobs += [(n, {}), (n, {"s_open": 1.75}), (n, {"s_open": 2.25, "s_close_long": 0.75, "s_close_short": 0.75})]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(job, jobs))
    print(f"{'signal':12s} {'cfg':30s} gross%/y  shr(0bp) IS   OOS   turn  nL+nS   vol")
    for name, kw, sa, si, so, gr, tn, nl, ns, vol in res:
        print(f"{name:12s} {str(kw):30s} {gr:+6.1%}   {sa:5.2f}   {si:5.2f} {so:5.2f}  {tn:4.2f}  {nl:3.0f}+{ns:3.0f} {vol:5.1%}")
