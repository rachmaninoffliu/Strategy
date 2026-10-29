"""
Rule-based tests of the convergence idea (no fitted parameters, so the whole 2016-26 sample is out-of-sample): ONE trade a day at a fixed time, held to the official close (MOC, free exit).
  vwap_rev  : buy below the session VWAP, sell above (the paper's VWAP benchmark, held to the settlement)
  open_mom  : trade in the direction of the move since the open
  r30_mom   : trade in the direction of the first-30-minute return (the documented 'intraday momentum': the early move predicts the late one)
  prev_mom  : trade in the direction of the move since yesterday's close
Net bp per trade = gross - (half quoted spread + slippage). t-stat over days. Reports are per product x rule x time.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import ia_data as D, ia_study as S

TIMES = {"10:00": 5, "11:15": 20, "12:00": 29, "13:30": 47, "14:00": 53, "14:30": 59, "15:30": 71}      # decision j is at 9:35 + 5j minutes
IDX = {f: i for i, f in enumerate(S.FEATS)}

def run(args):
    sym, half = args
    dd = S.build_days(sym)
    cost = half + S.SLIP_BP
    X, y = dd["X"], dd["y"]
    yrs = dd["days"].year.values
    out = []
    for tname, j in TIMES.items():
        f = {"vwap_rev": -np.sign(X[:, j, IDX["vwap_dist"]]), "open_mom": np.sign(X[:, j, IDX["from_open"]]), "r30_mom": np.sign(X[:, j, IDX["r30"]]), "prev_mom": np.sign(X[:, j, IDX["from_prev"]])}
        for rule, s in f.items():
            if rule == "r30_mom" and j < 5:
                continue
            gross = s * y[:, j]
            net = gross - cost * (s != 0)
            for label, m in (("2016-26", yrs >= 2016), ("2016-21", yrs <= 2021), ("2022-26", yrs >= 2022)):
                n_ = net[m]
                out.append(dict(symbol=sym, rule=rule, time=tname, period=label, n=int(m.sum()), gross_bp=gross[m].mean(), net_bp=n_.mean(), t_net=n_.mean() / n_.std(ddof=1) * np.sqrt(len(n_)), cost_bp=cost,
                                sharpe=n_.mean() / n_.std(ddof=1) * np.sqrt(252), hit=(n_ > 0).mean()))
    return out

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    sp = pd.read_csv(os.path.join(D.REPORTS, "spreads.csv")).set_index("symbol")
    syms = [s for s in D.ALL if os.path.exists(D.bars_path(s))]
    jobs = [(s, float(sp.spread_bp.get(s, 2.0)) / 2.0) for s in syms]
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(run, jobs):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(D.REPORTS, "rule_tests.csv"), index=False)
    print("saved", len(df))
