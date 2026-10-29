"""
Valid: robustness of each strategy's chosen settings (research/reports/chosen.json, picked on in-sample Sharpe in r2-r4).
  * cost per side 0 .. 8 bp; fill at the bar VWAP instead of the open print; one bar of extra delay (the free delayed SIP feed)
  * PCA factor model instead of the ETF model; 50 instead of 100 names; years one by one
  * deflated Sharpe ratio for the number of variants tried in the grid (trials) and the variance of their Sharpe ratios

  python r6_robustness.py
"""
import json
import os

import numpy as np

from common import BT, OUT, SPLIT, ResidualMeanReversion, ResidualMomentum, ResidualPrediction, S, log, pd, row, run_bt

chosen = json.load(open(os.path.join(OUT, "chosen.json")))
trials = json.load(open(os.path.join(OUT, "trials.json")))                                    # {key: [n variants, var of their annual Sharpe]}
classes = {"ResMomentum": ResidualMomentum, "ResMeanReversion": ResidualMeanReversion, "ResPrediction": ResidualPrediction}
out = []
for key, vals in chosen.items():
    cls = classes[key]
    base = cls(vals)
    ctx = S.context(base, log=log)
    sc = base.score(ctx, log)
    bt = run_bt(base, ctx, sc)
    st = bt.stats
    r = bt.daily.ret
    n_tr, var_tr = trials.get(key, [1, 0.0])
    ri = r[r.index < SPLIT]
    sh_is = st["is"]["sharpe"]
    yrs = len(ri) / 252
    # formal DSR (variance of the trials' in-sample Sharpe ratios) and a noise-only DSR (the sampling variance of a Sharpe estimate over the in-sample years, ~(1 + SR^2/2) / years)
    dsr_formal = BT.dsr(sh_is, len(ri), float(ri.skew()), float(ri.kurt() + 3), int(n_tr), float(var_tr))
    dsr_noise = BT.dsr(sh_is, len(ri), float(ri.skew()), float(ri.kurt() + 3), int(n_tr), (1 + sh_is ** 2 / 2) / yrs)
    dsr_oos = BT.psr(st["oos"]["sharpe"], int(st["oos"]["days"]), float(r[r.index >= SPLIT].skew()), float(r[r.index >= SPLIT].kurt() + 3)) if "sharpe" in st["oos"] else np.nan
    out.append(row(bt, strategy=key, case="chosen", dsr=dsr_formal, dsr_noise=dsr_noise, psr_oos=dsr_oos, trials=n_tr))
    log(f"{key} chosen: IS {sh_is:.2f} OOS {st['oos'].get('sharpe', np.nan):.2f}; {n_tr} trials; DSR formal {dsr_formal:.2f}, noise-only {dsr_noise:.2f}; PSR(OOS > 0) {dsr_oos:.2f}")
    for c in BT.cost_curve(bt, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]).itertuples():
        out.append(dict(strategy=key, case=f"cost x{c.cost_x:.1f} ({c.avg_bp_side:.2f} bp/side)", sh_all=c.sharpe, sh_is=c.sharpe_is, cagr_is=c.cagr, mdd=c.max_dd))
    variants = {"fill at bar VWAP": {"fill": "vwap"}, "+1 bar delay (free delayed SIP)": {"feed": "sip_delayed"}, "50 names": {"universe_n": 50}}
    for name, ch in variants.items():
        st2 = cls({**vals, **ch})
        if "universe_n" in ch:
            c2 = S.context(st2, log=log)
            b2 = run_bt(st2, c2)
        else:
            c2 = S.Context(p=ctx.p, res=ctx.res, member=ctx.member, member_d=ctx.member_d, lag=st2.lag)       # the delay also moves the prediction target
            sc2 = sc if st2.lag == base.lag else st2.score(c2, log)
            b2 = BT.run(ctx.p, ctx.res, sc2, ctx.member, st2.decision_bars(ctx.p), st2.port_cfg(), st2.bt_cfg())
        out.append(row(b2, strategy=key, case=name))
        log(f"{key} | {name}: IS {out[-1]['sh_is']:.2f} OOS {out[-1]['sh_oos']:.2f}")
    st3 = cls({**vals, "factor_model": "pca", "hedge": "project"})
    c3 = S.context(st3, log=log)
    b3 = run_bt(st3, c3)
    out.append(row(b3, strategy=key, case="PCA factors (15), projection hedge"))
    log(f"{key} | PCA: IS {out[-1]['sh_is']:.2f} OOS {out[-1]['sh_oos']:.2f}")
    for y, v in st["by_year"].items():
        out.append(dict(strategy=key, case=f"year {y}", cagr_is=v))
df = pd.DataFrame(out)
df.to_csv(f"{OUT}/r6_robustness.csv", index=False)
pd.set_option("display.width", 250)
print(df[["strategy", "case", "sh_is", "sh_oos", "sh_all", "gross_sh_all", "cagr_is", "cagr_oos", "mdd", "turn", "dsr", "dsr_noise", "psr_oos"]].round(3).to_string(index=False))
