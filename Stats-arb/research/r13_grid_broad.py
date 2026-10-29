import sys
from rlib import *
from concurrent.futures import ProcessPoolExecutor

BASE = dict(neutral=False, event_filter=False, earn_filter=False, weight_pow=1.0, shock=False, band=0.2, scale_band=0.15, vol_target=1.0, g_init=1.0, g_min=1.0, g_max=2.0, tc_bps=1.5,
            borrow_bps=50.0, name_cap=0.03, start="2017-06-01")

def job(args):
    universe, name, label, kw = args
    panel = panel_for(universe)
    sig = pickle.load(open(os.path.join(SP, f"sig_{universe}_{name}.pkl"), "rb"))
    p = replace(C.PortCfg(), **{**BASE, **kw})
    r = S.simulate(sig, panel, p)
    d, ret = r.daily, r.ret
    IS, OOS = ret[:"2021-12-31"], ret["2022-01-01":]
    ma, mi, mo = S.metrics(ret), S.metrics(IS), S.metrics(OOS)
    return (universe, label, ma["sharpe"], mi["sharpe"], mo["sharpe"], ma["cagr"], ma["vol"], ma["maxdd"], d.gross_ret.mean() * 252, d.cost.mean() * 252, d.n_long.mean(), d.n_short.mean(), d.turnover.mean())

if __name__ == "__main__":
    universe, name = sys.argv[1], sys.argv[2]
    cfgs = {
        "paper rules  tc=0": dict(tc_bps=0.0, weight_pow=0.0, band=0.0, scale_band=0.0, borrow_bps=0.0),
        "paper rules  tc=1.5 tiered": dict(weight_pow=0.0, band=0.0, scale_band=0.0),
        "invvol, band  tc=1.5": {},
        "invvol, band  tc=3": dict(tc_bps=3.0),
        "s_open 1.75": dict(s_open=1.75),
        "s_open 2.25 close .75": dict(s_open=2.25, s_close_long=0.75, s_close_short=0.75),
        "vol_boost 1": dict(vol_boost=1.0),
        "long-only+ETF hedge": dict(allow_short=False, hedge="etf"),
        "neutral (proj) tc=1.5": dict(neutral=True),
        "max_hold 6": dict(max_hold=6),
        "close early (0.25/0.25)": dict(s_close_long=0.25, s_close_short=0.25),
        "close at 0 (hold to revert)": dict(s_close_long=0.0, s_close_short=0.0),
    }
    jobs = [(universe, name, l, kw) for l, kw in cfgs.items()]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(job, jobs))
    print(f"{'universe':8s} {'config':30s} Sh(all)  IS   OOS   CAGR   vol   maxDD  gross/y cost/y  nL  nS  turn")
    for r in res:
        print(f"{r[0]:8s} {r[1]:30s} {r[2]:5.2f} {r[3]:5.2f} {r[4]:5.2f} {r[5]:+6.1%} {r[6]:5.1%} {r[7]:6.1%} {r[8]:+6.1%} {r[9]:5.1%} {r[10]:4.0f} {r[11]:3.0f} {r[12]:.2f}")
