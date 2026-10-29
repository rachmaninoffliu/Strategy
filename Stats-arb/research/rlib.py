import sys, os, time, pickle
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
from dataclasses import replace
import statarb_data as D, statarb_core as C, statarb_sim as S, statarb_bridge as BR

SP = os.path.dirname(os.path.abspath(__file__))

def panel_for(universe):
    return BR.load_panel(universe, refresh=False)

def sig_for(universe, name, cfg, panel=None, earn=True, n_jobs=8):
    f = os.path.join(SP, f"sig_{universe}_{name}.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    panel = panel or panel_for(universe)
    t = time.time()
    sig = C.build_signals(panel, cfg, earnings=D.load_earnings() if earn else None, n_jobs=n_jobs)
    print(f"[{universe}/{name}] stage A {time.time()-t:.0f}s ({len(panel.stocks)} names)", flush=True)
    pickle.dump(sig, open(f, "wb"), protocol=4)
    return sig

VARIANTS = {
    "pca15": C.FactorCfg(),
    "pca55": C.FactorCfg(var_explained=0.55),
    "pca8": C.FactorCfg(n_factors=8),
    "pca15_vt": C.FactorCfg(volume_time=True),
    "etf": C.FactorCfg(factor_mode="etf"),
}
