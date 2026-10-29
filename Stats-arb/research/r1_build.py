import sys, time, pickle, os
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
import numpy as np, pandas as pd
import statarb_data as D, statarb_core as C

SP = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {
    "pca15": C.FactorCfg(),
    "pca15_vt": C.FactorCfg(volume_time=True),
    "pca55": C.FactorCfg(var_explained=0.55),
    "etf": C.FactorCfg(factor_mode="etf"),
    "pca15_sd": C.FactorCfg(sector_demean=True),
    "pca15_w30": C.FactorCfg(est_window=30),
    "pca15_w20": C.FactorCfg(est_window=20),
    "pca8": C.FactorCfg(n_factors=8),
}

def build(name, cfg, panel, earn, n_jobs=8):
    f = os.path.join(SP, f"sig_{name}.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    t = time.time()
    sig = C.build_signals(panel, cfg, earnings=earn, n_jobs=n_jobs)
    print(f"[{name}] stage A {time.time()-t:.0f}s", flush=True)
    pickle.dump(sig, open(f, "wb"), protocol=4)
    return sig

if __name__ == "__main__":
    panel = D.load_panel(); earn = D.load_earnings()
    for k in (sys.argv[1:] or ["pca15"]):
        build(k, VARIANTS[k], panel, earn)
