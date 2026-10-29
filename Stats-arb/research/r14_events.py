import sys
from rlib import *
from concurrent.futures import ProcessPoolExecutor
from r13_grid_broad import job, BASE
import r13_grid_broad as G

if __name__ == "__main__":
    universe, name = sys.argv[1], sys.argv[2]
    cfgs = {}
    for vr in (0.0, 1.5, 2.0, 3.0):
        for hold in (4, 8):
            for so in (1.25, 1.75):
                cfgs[f"vr>={vr} hold{hold} s>{so}"] = dict(min_vr=vr, max_hold=hold, s_open=so, weight_pow=1.0, band=0.2, tc_bps=1.5, s_close_long=0.25, s_close_short=0.25)
    jobs = [(universe, name, l, kw) for l, kw in cfgs.items()]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(job, jobs))
    print(f"{'universe':8s} {'config':30s} Sh(all)  IS   OOS   CAGR   vol   maxDD  gross/y cost/y  nL  nS  turn")
    for r in res:
        print(f"{r[0]:8s} {r[1]:30s} {r[2]:5.2f} {r[3]:5.2f} {r[4]:5.2f} {r[5]:+6.1%} {r[6]:5.1%} {r[7]:6.1%} {r[8]:+6.1%} {r[9]:5.1%} {r[10]:4.0f} {r[11]:3.0f} {r[12]:.2f}")
