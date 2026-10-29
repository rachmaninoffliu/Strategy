"""
Valid's checks on the prediction-enhanced time-shared book (overnight ranking = rank average of the rule "today's move" and the walk-forward gradient-boosted
forecast of the overnight return; intraday = QQQ noise area). Window 2018-01 -> 2026-10 (the first walk-forward year is 2018).
  * the same model rebuilt at 15:30 (features at 15:30, models retrained) - the free delayed-SIP path;
  * an a-priori ensemble (rule + ridge + both tree models, equal rank weights) instead of the single in-sample pick;
  * costs 2 / 3 bp per auction side; the first-trade exit; k = 10 / 30;
  * year by year, deflated Sharpe with every trial counted.
"""
import glob
import os

import numpy as np
import pandas as pd

from mp_features import build, columns, to_matrix, walk_forward
from mp_lib import OUT, SPLIT, Book, Ctx, dsr, log, overnight, psr, topk
from mp_overnight_ref import signals, spy_trend
from mp_sleeves import noise_area

cx = Ctx()
bk = Book("mp_predict_check", cx)
W0 = "2018-01-01"
tr = spy_trend(cx)
ri = noise_area(cx, "QQQ", every=2, lookback=14)
ri_c2 = noise_area(cx, "QQQ", every=2, lookback=14, half_bp=1.0, auction_bp=2.0)
ri_c3 = noise_area(cx, "QQQ", every=2, lookback=14, half_bp=1.5, auction_bp=3.0)
bench = cx.spy_ret()
bk.add(bench[bench.index >= W0], "SPY buy & hold")
main = {}


def rk(x, U):
    return pd.DataFrame(np.where(U & np.isfinite(x), x, np.nan)).rank(axis=1, pct=True).to_numpy()


for lead in (1, 2):
    tag = "15:45" if lead == 1 else "15:30"
    live, S = signals(cx, lead)
    U = (cx.top(150) | cx.etf()) & live
    df = build(cx, lead)
    pre = "" if lead == 1 else "1530_"
    P = {}
    P["hgbraw"] = to_matrix(cx, df, walk_forward(df, "y_on", "hgb", columns(df, "raw"), True, key=f"{pre}y_on_hgbraw_rank"))
    P["hgb"] = to_matrix(cx, df, walk_forward(df, "y_on", "hgb", columns(df, "rank"), True, key=f"{pre}y_on_hgb_rank"))
    P["ridge"] = to_matrix(cx, df, walk_forward(df, "y_on", "ridge", columns(df, "rank"), True, key=f"{pre}y_on_ridge_rank"))
    log(f"{tag}: models ready")
    sig_blend = rk(P["hgbraw"], U) + rk(S["d45"], U)
    sig_ens = rk(P["hgbraw"], U) + rk(P["hgb"], U) + rk(P["ridge"], U) + rk(S["d45"], U)
    for name, sig in (("rule", S["d45"]), ("rule + trees (IS pick)", sig_blend), ("rule + 3 models (ensemble)", sig_ens)):
        for k in (10, 20, 30):
            sel = topk(sig, U & tr, k)
            ro = overnight(cx, sel, k=k)
            r = ((1 + ro) * (1 + ri) - 1)
            lab = f"{tag} {name} top{k} + QQQ"
            bk.add(r[r.index >= W0], lab, lead=lead, model=name, k=k, chk="base")
            if k == 20:
                main[lab] = r[r.index >= W0]
                bk.add(ro[ro.index >= W0], f"{lab} | overnight leg alone", chk="leg")
                for c, rr in ((2.0, ri_c2), (3.0, ri_c3)):
                    x = (1 + overnight(cx, sel, k=k, cost_in=c, cost_out=c)) * (1 + rr) - 1
                    bk.add(x[x.index >= W0], f"{lab} | auctions {c:.0f} bp", chk=f"cost{c:.0f}")
                x = (1 + overnight(cx, sel, k=k, exit="first")) * (1 + ri) - 1
                bk.add(x[x.index >= W0], f"{lab} | first-trade exit", chk="first")
df_out = bk.save()
trial_sh = []
for f in glob.glob(os.path.join(OUT, "mp_*.csv")):
    if any(x in f for x in ("_check", "_ic", "_robust")):
        continue
    d = pd.read_csv(f)
    if "sh_is" in d and "label" in d:
        trial_sh += list(d.loc[~d.label.str.contains("buy & hold", na=False), "sh_is"].dropna())
n_all = len(trial_sh) + 1871
v_all = float(np.var(trial_sh))
rows = []
for lab, r in main.items():
    a, b = r[r.index < SPLIT], r[r.index >= SPLIT]
    d1, e1 = dsr(a, n_all, v_all)
    rows.append({"model": lab, "sh_is": a.mean() / a.std() * np.sqrt(252), "sh_oos": b.mean() / b.std() * np.sqrt(252), "dsr_all": d1, "exp_max_all": e1,
                 "psr_oos>0": psr(b), "psr_oos>1.38": psr(b, 1.377), "psr_oos>2": psr(b, 2.0)})
with pd.option_context("display.width", 300, "display.max_rows", 200, "display.max_colwidth", 100):
    print(df_out[["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all", "beta"]].round(3).to_string(index=False))
    print(f"\ntrials counted: {n_all} (variance of IS Sharpe {v_all:.3f})")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    y = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in main.items()})
    b = bench[bench.index >= W0]
    y["SPY"] = (1 + b).groupby(b.index.year).prod() - 1
    print((y * 100).round(1).T.to_string())
