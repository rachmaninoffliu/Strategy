import sys, pickle, os, time, itertools
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import statarb_data as D, statarb_core as C, statarb_sim as S

SP = os.path.dirname(os.path.abspath(__file__))
BASE = dict(neutral=False, event_filter=False, earn_filter=False, weight_pow=1.0, shock=False, band=0.0, scale_band=0.0, vol_target=1.0, g_init=1.0, g_min=1.0, g_max=2.0, tc_bps=1.5,
            borrow_bps=50.0, name_cap=0.05)

def job(args):
    name, label, kw = args
    panel = D.load_panel()
    sig = pickle.load(open(os.path.join(SP, f"sig_{name}.pkl"), "rb"))
    p = replace(C.PortCfg(), **{**BASE, **kw})
    r = S.simulate(sig, panel, p)
    d, ret = r.daily, r.ret
    IS, OOS = ret[:"2021-12-31"], ret["2022-01-01":]
    ma, mi, mo = S.metrics(ret), S.metrics(IS), S.metrics(OOS)
    return (name, label, ma["sharpe"], mi["sharpe"], mo["sharpe"], ma["cagr"], ma["vol"], ma["maxdd"], d.gross_ret.mean() * 252, d.cost.mean() * 252, d.n_long.mean(), d.n_short.mean(), d.turnover.mean())

if __name__ == "__main__":
    jobs = []
    for name in sys.argv[1:] or ["pca15"]:
        for lab, kw in {
            "LS 1.25": {},
            "LS 1.25 boost1": {"vol_boost": 1.0},
            "LS 1.75 boost1": {"s_open": 1.75, "vol_boost": 1.0},
            "LS 1.25 events-filtered": {"event_filter": True, "earn_filter": True},
            "LS 1.25 boost1 events-filtered": {"vol_boost": 1.0, "event_filter": True, "earn_filter": True},
            "LongOnly+ETFhedge 1.25": {"allow_short": False, "hedge": "etf"},
            "LongOnly+ETFhedge 1.25 boost1": {"allow_short": False, "hedge": "etf", "vol_boost": 1.0},
            "LongOnly+ETFhedge 1.75 boost1": {"allow_short": False, "hedge": "etf", "vol_boost": 1.0, "s_open": 1.75},
            "LongOnly+ETFhedge 1.25 boost1 filt": {"allow_short": False, "hedge": "etf", "vol_boost": 1.0, "event_filter": True, "earn_filter": True},
            "LongOnly+ETFhedge 1.25 boost2": {"allow_short": False, "hedge": "etf", "vol_boost": 2.0},
        }.items():
            jobs.append((name, lab, kw))
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(job, jobs))
    print(f"{'signal':9s} {'config':36s} Sh(all) IS   OOS   CAGR   vol   maxDD  gross/y cost/y  nL  nS  turn")
    for r in res:
        print(f"{r[0]:9s} {r[1]:36s} {r[2]:5.2f} {r[3]:5.2f} {r[4]:5.2f} {r[5]:+6.1%} {r[6]:5.1%} {r[7]:6.1%} {r[8]:+6.1%} {r[9]:5.1%} {r[10]:4.0f} {r[11]:3.0f} {r[12]:.2f}")
