"""Where does the close-fill profit come from?  Mean spread move after an extreme s-score, split into the jump from the signal bar's close to the next open (the first seconds) and the rest of the next bar."""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, r"C:\Users\xinyu\PycharmProjects\pythonProject\LevETF-Pairs\scripts")
os.chdir(r"C:\Users\xinyu\PycharmProjects\pythonProject\LevETF-Pairs\scripts")
import levpairs_bridge as BR
from levpairs_ui.presets import DEFAULT_PRESET, cfgs

sc, c = cfgs(DEFAULT_PRESET)
panel = BR.load_panel(5, "sip", log=lambda m: None)
sigs = BR.get_signals(panel, sc, "sip", log=lambda m: None)

print("mean move of the spread (bp of the 3x leg) in the direction that profits a trade opened at s, by threshold of |s|")
print(f"{'pair':10s} {'|s|>':>5s} {'n':>8s} {'close->next open':>17s} {'next open->close':>17s} {'next close->close+1':>20s}")
for name, sg in sigs.items():
    s = sg.s[:, :-2]
    gap = sg.eps_gap[:, 1:-1]            # close of bar t -> open of bar t+1
    body = sg.eps_body[:, 1:-1]          # open of bar t+1 -> close of bar t+1
    body2 = sg.eps_body[:, 2:]           # the bar after that
    for thr in (1.0, 2.0, 3.0):
        m = np.isfinite(s) & (np.abs(s) > thr)
        d = -np.sign(s[m])               # long the spread when s is very negative
        print(f"{name:10s} {thr:5.1f} {m.sum():8d} {np.mean(d * gap[m]) * 1e4:17.2f} {np.mean(d * body[m]) * 1e4:17.2f} {np.mean(d * body2[m]) * 1e4:20.2f}")

res = BR.fill_comparison(panel, sigs, c)
tab, gaps = res
print()
print(tab[["sharpe", "trades", "gross_bp", "net_bp"]].round(2).to_string())
print("mean move in the trade's favour between the signal close and the next open: at entry %+.2f bp, at exit %+.2f bp" % (gaps["gap_in"], gaps["gap_out"]))
