"""
QR round 2 (pre-registered after round 1 showed gross edges eaten by turnover): Garleanu & Pedersen (2013) partial trading - each rebalance moves only `trade_rate` of the way to
the target. Same in-sample-only selection; these variants are ADDED to each strategy's trial count.

  python r8_costaware.py [feed]
"""
import itertools
import sys

from common import OUT, ResidualMeanReversion, ResidualMomentum, ResidualPrediction, S, log, pd, row, run_bt

feed = sys.argv[1] if len(sys.argv) > 1 else "iex"
sel = open(f"{OUT}/selected_features.txt").read().strip()
rows = []
base = {"universe_n": 100, "feed": feed, "cost_model": "spread", "etf_cost_bp": 0.5, "hedge": "etf"}
ctx = S.context(ResidualMeanReversion({**base, "beta_days": 126}), log=log)
for k, (kb, reb, rate) in enumerate(itertools.product([1, 2, 4, 13], ["15min", "1h"], [0.1, 0.25, 0.5])):
    st = ResidualMeanReversion({**base, "beta_days": 126, "mode": "xs", "xs_bars": kb, "q_in": 0.1, "q_out": 0.25, "rebalance": reb, "method": "quantile", "trade_rate": rate})
    bt = run_bt(st, ctx)
    rows.append(row(bt, strategy="ResMeanReversion", mode="xs", xs_bars=kb, W=None, reb=reb, trade_rate=rate, q_in=0.1, q_out=0.25))
    log(f"[MR xs {k + 1}/24] k={kb} {reb} rate={rate}: IS {rows[-1]['sh_is']:.2f} OOS {rows[-1]['sh_oos']:.2f} gross IS {rows[-1]['gross_sh_is']:.2f} turn {rows[-1]['turn']:.0f}x")
for k, (W, rate) in enumerate(itertools.product([26, 130], [0.1, 0.25])):
    st = ResidualMeanReversion({**base, "beta_days": 126, "mode": "ou", "ou_window": W, "s_open": 1.5, "s_close": 0.5, "rebalance": "15min", "method": "threshold", "names": 30,
                                "trade_rate": rate})
    bt = run_bt(st, ctx)
    rows.append(row(bt, strategy="ResMeanReversion", mode="ou", W=W, xs_bars=None, reb="15min", trade_rate=rate, s_open=1.5, s_close=0.5))
    log(f"[MR ou {k + 1}/4] W={W} rate={rate}: IS {rows[-1]['sh_is']:.2f} OOS {rows[-1]['sh_oos']:.2f} gross IS {rows[-1]['gross_sh_is']:.2f} turn {rows[-1]['turn']:.0f}x")
pb = {**base, "beta_days": 126, "train_years": 3.0, "min_train_years": 2.0, "retrain": "quarterly", "sample_every": 2, "model": "ridge", "features": sel, "q_in": 0.2, "q_out": 0.3}
for k, (h, reb, rate) in enumerate(itertools.product([4, 13, 26], ["15min", "1h"], [0.1, 0.25])):
    st = ResidualPrediction({**pb, "horizon": h, "rebalance": reb, "trade_rate": rate})
    sc = st.score(ctx, log)
    bt = run_bt(st, ctx, sc)
    rows.append(row(bt, strategy="ResPrediction", model="ridge", features="selected", horizon=h, reb=reb, trade_rate=rate, q_in=0.2, q_out=0.3))
    log(f"[Pred {k + 1}/12] h={h} {reb} rate={rate}: IS {rows[-1]['sh_is']:.2f} OOS {rows[-1]['sh_oos']:.2f} gross IS {rows[-1]['gross_sh_is']:.2f} turn {rows[-1]['turn']:.0f}x")
ctx_m = S.context(ResidualMomentum({**base, "beta_days": 252}), log=log)
for k, (F, rate) in enumerate(itertools.product([63, 126, 252], [0.1, 0.25])):
    st = ResidualMomentum({**base, "beta_days": 252, "form_days": F, "skip_days": 21, "rebalance": "daily", "trade_time": "15:45", "q_in": 0.1, "q_out": 0.25, "trade_rate": rate})
    bt = run_bt(st, ctx_m)
    rows.append(row(bt, strategy="ResMomentum", form=F, skip=21, reb="daily", trade_rate=rate, q_in=0.1, q_out=0.25))
    log(f"[Mom {k + 1}/6] F={F} daily rate={rate}: IS {rows[-1]['sh_is']:.2f} OOS {rows[-1]['sh_oos']:.2f} gross IS {rows[-1]['gross_sh_is']:.2f} turn {rows[-1]['turn']:.0f}x")
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/r8_costaware_{feed}.csv", index=False)
pd.set_option("display.width", 250)
for s, g in df.groupby("strategy"):
    print(f"\n== {s} ==")
    print(g.sort_values("sh_is", ascending=False).drop(columns=[c for c in g.columns if g[c].isna().all()]).round(3).to_string(index=False))
