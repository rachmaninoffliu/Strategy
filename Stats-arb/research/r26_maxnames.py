import sys
sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\Stats-arb\scripts")
from dataclasses import replace
import numpy as np, pandas as pd
import statarb_bridge as BR, statarb_sim as S
from statarb_ui.presets import cfgs, DEFAULT_PRESET
if __name__ == "__main__":
    f, p = cfgs(DEFAULT_PRESET)
    p = replace(p, start="2017-06-01")
    panel = BR.load_panel("broad")
    sig = BR.get_signals(panel, "broad", f, log=lambda m: None)
    spy = S.spy_open_to_open(panel)
    for mn in (0, 150, 100, 60, 40):
        r = S.simulate(sig, panel, replace(p, eap_max_names=mn))
        m, mi, mo = S.metrics(r.ret, spy), S.metrics(r.ret[:"2021-12-31"]), S.metrics(r.ret["2022-01-01":])
        d = r.daily
        print(f"max_names {mn:4d}: Sharpe {m['sharpe']:5.2f} (IS {mi['sharpe']:5.2f} OOS {mo['sharpe']:5.2f}) CAGR {m['cagr']:+.1%} vol {m['vol']:.1%} DD {m['maxdd']:.1%} gross {d.gross.mean():.2f} names {d.n_eap.mean():.0f} turn {d.turnover.mean():.2f}", flush=True)
