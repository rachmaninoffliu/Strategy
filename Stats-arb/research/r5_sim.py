import sys, pickle, os, time
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
from dataclasses import replace
import statarb_data as D, statarb_core as C, statarb_sim as S

SP = os.path.dirname(os.path.abspath(__file__))
panel = D.load_panel()
sig = pickle.load(open(os.path.join(SP, f"sig_{sys.argv[1] if len(sys.argv) > 1 else 'pca15'}.pkl"), "rb"))
spy = S.spy_open_to_open(panel)

def run(name, **kw):
    p = replace(C.PortCfg(), **kw)
    t = time.time()
    r = S.simulate(sig, panel, p)
    ret = r.ret
    IS, OOS = ret[:"2021-12-31"], ret["2022-01-01":]
    mi, mo, ma = S.metrics(IS, spy), S.metrics(OOS, spy), S.metrics(ret, spy)
    d = r.daily
    print(f"{name:34s} ALL sh {ma['sharpe']:5.2f} cagr {ma['cagr']:+6.1%} dd {ma['maxdd']:6.1%} | IS sh {mi['sharpe']:5.2f} | OOS sh {mo['sharpe']:5.2f} cagr {mo['cagr']:+6.1%} | "
          f"turn {d.turnover.mean():.2f}/d gross {d.gross.mean():.2f} pos {d.n_long.mean():.0f}+{d.n_short.mean():.0f} cost {d.cost.sum()/len(d)*252:.1%}/y gross-ret {d.gross_ret.mean()*252:+.1%}/y  ({time.time()-t:.0f}s)", flush=True)
    return r

if __name__ == "__main__":
    paper = dict(neutral=False, event_filter=False, earn_filter=False, weight_pow=0.0, shock=False, band=0.0, scale_band=0.0, vol_target=1.0, g_init=1.0, g_min=1.0, g_max=1.0, tc_bps=5.0, borrow_bps=0.0, name_cap=1.0)
    run("paper-like (5bp, eq-wt, bang-bang)", **paper)
    run("  paper @ 0bp cost", **{**paper, "tc_bps": 0.0})
    run("  paper @ 3bp cost", **{**paper, "tc_bps": 3.0})
    run("+ inverse-vol weights", **{**paper, "weight_pow": 1.0, "tc_bps": 3.0})
    run("+ earnings+news filter", **{**paper, "weight_pow": 1.0, "tc_bps": 3.0, "event_filter": True, "earn_filter": True})
    run("+ neutralisation", **{**paper, "weight_pow": 1.0, "tc_bps": 3.0, "event_filter": True, "earn_filter": True, "neutral": True, "name_cap": 0.04, "band": 0.2, "scale_band": 0.15})
    run("+ vol target 10% & shock", **{**paper, "weight_pow": 1.0, "tc_bps": 3.0, "event_filter": True, "earn_filter": True, "neutral": True, "name_cap": 0.04, "band": 0.2, "scale_band": 0.15,
                                      "vol_target": 0.10, "g_min": 0.25, "g_max": 2.0, "shock": True, "borrow_bps": 50.0})
