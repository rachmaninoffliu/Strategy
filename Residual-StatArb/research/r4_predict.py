"""
QR / Valid: residual prediction. Walk-forward ridge (and one gradient-boosting run) on the feature set chosen from r1's IN-SAMPLE t-statistics, for several forecast horizons. Every
forecast comes from a model fitted on data whose targets were known before the refit date; settings are picked on in-sample Sharpe only.

  python r4_predict.py [feed] [features]        features = comma list (default: read research/reports/selected_features.txt)
"""
import os
import sys

from common import OUT, ResidualPrediction, S, log, pd, row, run_bt

feed = sys.argv[1] if len(sys.argv) > 1 else "iex"
sel = sys.argv[2] if len(sys.argv) > 2 else open(os.path.join(OUT, "selected_features.txt")).read().strip()
base = {"universe_n": 100, "feed": feed, "beta_days": 126, "cost_model": "spread", "etf_cost_bp": 0.5, "hedge": "etf", "train_years": 3.0, "min_train_years": 2.0, "retrain": "quarterly",
        "sample_every": 2, "start": "2016-09-01"}
ctx = S.context(ResidualPrediction(base), log=log)
log(ctx.p.describe())
rows = []
grid = [(4, "1h", 0.2, 0.3), (13, "1h", 0.2, 0.3), (26, "1h", 0.2, 0.35), (26, "daily", 0.2, 0.3), (130, "daily", 0.2, 0.35), (130, "weekly", 0.2, 0.3)]
for fs_name, fs in (("selected", sel), ("all", "all")):
    for k, (h, reb, qi, qo) in enumerate(grid):
        st = ResidualPrediction({**base, "model": "ridge", "horizon": h, "features": fs, "rebalance": reb, "trade_time": "15:45", "q_in": qi, "q_out": qo})
        sc = st.score(ctx, log)
        bt = run_bt(st, ctx, sc)
        rows.append(row(bt, model="ridge", features=fs_name, horizon=h, reb=reb, q_in=qi, q_out=qo))
        r = rows[-1]
        log(f"[{fs_name} {k + 1}/{len(grid)}] ridge h={h} {reb}: IS {r['sh_is']:.2f} OOS {r['sh_oos']:.2f} gross IS {r['gross_sh_is']:.2f} turn {r['turn']:.0f}x")
df = pd.DataFrame(rows).sort_values("sh_is", ascending=False)
best = df.iloc[0]
log(f"IS-best ridge: {best.features} h={best.horizon} {best.reb} - now one gradient-boosting run with the same settings")
st = ResidualPrediction({**base, "model": "gbm", "horizon": int(best.horizon), "features": sel if best.features == "selected" else "all", "rebalance": best.reb, "trade_time": "15:45",
                         "q_in": best.q_in, "q_out": best.q_out})
sc = st.score(ctx, log)
bt = run_bt(st, ctx, sc)
rows.append(row(bt, model="gbm", features=best.features, horizon=int(best.horizon), reb=best.reb, q_in=best.q_in, q_out=best.q_out))
df = pd.DataFrame(rows).sort_values("sh_is", ascending=False)
df.to_csv(f"{OUT}/r4_predict_{feed}.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))

