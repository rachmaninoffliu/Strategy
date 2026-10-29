"""
QR / Valid: residual momentum grid. Every variant is chosen on IN-SAMPLE (2016-09 .. 2022) Sharpe only; the out-of-sample column is reported, never used to pick.

  python r2_momentum.py [feed]        feed = iex (lag 0, default) | sip_delayed (lag 1)
"""
import itertools
import sys

from common import OUT, ResidualMomentum, S, log, pd, row, run_bt

feed = sys.argv[1] if len(sys.argv) > 1 else "iex"
base = {"universe_n": 100, "feed": feed, "beta_days": 252, "cost_model": "spread", "etf_cost_bp": 0.5, "trade_time": "15:45", "hedge": "etf"}
ctx = S.context(ResidualMomentum(base), log=log)
log(ctx.p.describe())
rows = []
grid = list(itertools.product([252, 126, 63, 21], [21, 5, 0], ["monthly", "weekly", "daily"], [(0.2, 0.2), (0.2, 0.35), (0.1, 0.25)]))
for k, (F, Sk, reb, (qi, qo)) in enumerate(grid):
    if Sk >= F // 2:
        continue
    st = ResidualMomentum({**base, "form_days": F, "skip_days": Sk, "rebalance": reb, "q_in": qi, "q_out": qo})
    bt = run_bt(st, ctx)
    rows.append(row(bt, form=F, skip=Sk, reb=reb, q_in=qi, q_out=qo))
    r = rows[-1]
    log(f"[{k + 1}/{len(grid)}] F={F} S={Sk} {reb} q={qi}/{qo}: IS {r['sh_is']:.2f} OOS {r['sh_oos']:.2f} gross IS {r['gross_sh_is']:.2f} turn {r['turn']:.0f}x")
df = pd.DataFrame(rows).sort_values("sh_is", ascending=False)
df.to_csv(f"{OUT}/r2_momentum_{feed}.csv", index=False)
pd.set_option("display.width", 250)
print(df.head(20).round(3).to_string(index=False))
print(f"\n{len(df)} variants; IS Sharpe mean {df.sh_is.mean():.2f} sd {df.sh_is.std():.2f}; OOS of the IS-best: {df.iloc[0].sh_oos:.2f}")

