import sys, os
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
from dataclasses import replace
import numpy as np, pandas as pd
import statarb_bridge as BR, statarb_data as D, statarb_sim as S
from statarb_ui.presets import cfgs

if __name__ == "__main__":
    universe = sys.argv[1] if len(sys.argv) > 1 else "broad"
    name = "Earnings-premium sleeve (screened by the OU model)"
    f, p = cfgs(name, universe)
    p = replace(p, start="2017-06-01")
    panel = BR.load_panel(universe)
    sig = BR.get_signals(panel, universe, f, log=lambda m: None)
    spy = S.spy_open_to_open(panel)
    base_univ = sig.univ.copy()
    print(f"{universe}: EAP sleeve by point-in-time liquidity band (ADV = 20-session median $ volume)")
    for lo, hi in ((5e7, 1e13), (1e8, 1e13), (2e8, 1e13), (5e8, 1e13), (1e9, 1e13), (5e7, 2e8), (2e8, 1e9)):
        sig.univ = base_univ & (sig.adv >= lo) & (sig.adv < hi)
        for pool in ("all", "notrich"):
            r = S.simulate(sig, panel, replace(p, eap_pool=pool)).ret
            m, mi, mo = S.metrics(r, spy), S.metrics(r[:"2021-12-31"]), S.metrics(r["2022-01-01":])
            nn = sig.univ[2000:].sum(1).mean()
            print(f"  ADV ${lo/1e6:>5.0f}M-{'inf' if hi>1e12 else f'{hi/1e6:.0f}M':>5}  pool {pool:8s} names/day {nn:5.0f}  Sharpe {m['sharpe']:5.2f} (IS {mi['sharpe']:5.2f} OOS {mo['sharpe']:5.2f})  CAGR {m['cagr']:+6.1%} vol {m['vol']:5.1%} maxDD {m['maxdd']:6.1%}", flush=True)
