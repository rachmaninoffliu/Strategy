"""
Pick each strategy's settings by IN-SAMPLE Sharpe only (Valid's rule) across round 1 (r2 / r3 / r4) and round 2 (r8, partial trading). Writes research/reports/chosen.json and
trials.json (all variants tried per strategy and the variance of their in-sample Sharpe ratios - the inputs of the deflated Sharpe ratio).

  python r7_choose.py [feed]
"""
import json
import os
import sys

import numpy as np

from common import OUT, pd

feed = sys.argv[1] if len(sys.argv) > 1 else "iex"
sel = open(os.path.join(OUT, "selected_features.txt")).read().strip()
r8 = pd.read_csv(os.path.join(OUT, f"r8_costaware_{feed}.csv")) if os.path.exists(os.path.join(OUT, f"r8_costaware_{feed}.csv")) else pd.DataFrame(columns=["strategy"])
base = {"universe_n": 100, "feed": feed, "cost_model": "spread", "etf_cost_bp": 0.5, "hedge": "etf"}
num = lambda x, d=None: d if x is None or (isinstance(x, float) and np.isnan(x)) else x
chosen, trials = {}, {}

g = pd.concat([pd.read_csv(os.path.join(OUT, f"r2_momentum_{feed}.csv")), r8[r8.strategy == "ResMomentum"]], ignore_index=True).sort_values("sh_is", ascending=False)
b = g.iloc[0]
chosen["ResMomentum"] = {**base, "beta_days": 252, "trade_time": "15:45", "form_days": int(b.form), "skip_days": int(b.skip), "rebalance": b.reb, "q_in": float(b.q_in),
                         "q_out": float(b.q_out), "trade_rate": float(num(b.get("trade_rate"), 1.0))}
trials["ResMomentum"] = [len(g), float(g.sh_is.var())]

g = pd.concat([pd.read_csv(os.path.join(OUT, f"r3_meanrev_{feed}.csv")), r8[r8.strategy == "ResMeanReversion"]], ignore_index=True).sort_values("sh_is", ascending=False)
b = g.iloc[0]
v = {**base, "beta_days": 126, "rebalance": b.reb, "flat_eod": bool(num(b.get("flat_eod"), False)), "mode": b["mode"], "trade_rate": float(num(b.get("trade_rate"), 1.0))}
if b["mode"] == "ou":
    v.update(ou_window=int(b.W), s_open=float(b.s_open), s_close=float(b.s_close), method="threshold", names=30)
else:
    v.update(xs_bars=int(b.xs_bars), q_in=float(b.q_in), q_out=float(b.q_out), method="quantile", trade_time="15:45")
chosen["ResMeanReversion"] = v
trials["ResMeanReversion"] = [len(g), float(g.sh_is.var())]

g = pd.concat([pd.read_csv(os.path.join(OUT, f"r4_predict_{feed}.csv")), r8[r8.strategy == "ResPrediction"]], ignore_index=True).sort_values("sh_is", ascending=False)
b = g.iloc[0]
chosen["ResPrediction"] = {**base, "beta_days": 126, "train_years": 3.0, "min_train_years": 2.0, "retrain": "quarterly", "sample_every": 2, "model": b.model, "horizon": int(b.horizon),
                           "features": sel if b.features == "selected" else "all", "rebalance": b.reb, "trade_time": "15:45", "q_in": float(b.q_in), "q_out": float(b.q_out),
                           "trade_rate": float(num(b.get("trade_rate"), 1.0))}
trials["ResPrediction"] = [len(g), float(g.sh_is.var())]
json.dump(chosen, open(os.path.join(OUT, "chosen.json"), "w"), indent=1)
json.dump(trials, open(os.path.join(OUT, "trials.json"), "w"), indent=1)
print(json.dumps(chosen, indent=1))
print(json.dumps(trials, indent=1))
