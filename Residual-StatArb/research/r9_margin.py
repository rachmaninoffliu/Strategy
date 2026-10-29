"""
RM: how much capital does each chosen book really use overnight (stocks + ETF hedges, x capital)? A Reg-T margin account allows 2x overnight (4x intraday), so the stock gross is
sized so that the 95th percentile of the overnight total stays under 1.9x. Sharpe does not change with the scale; return and drawdown scale with it.
"""
import json
import os

from common import OUT, ResidualMeanReversion, ResidualMomentum, ResidualPrediction, S, log, pd, run_bt

chosen = json.load(open(os.path.join(OUT, "chosen.json")))
classes = {"ResMomentum": ResidualMomentum, "ResMeanReversion": ResidualMeanReversion, "ResPrediction": ResidualPrediction}
rows = []
for key, vals in chosen.items():
    st = classes[key](vals)
    ctx = S.context(st, log=log)
    bt = run_bt(st, ctx)
    s = bt.stats
    k = 1.9 / s["overnight_gross_p95"]
    rows.append(dict(strategy=key, stock_gross=1.0, overnight_total_mean=s["overnight_gross_all"], overnight_total_p95=s["overnight_gross_p95"], sharpe=s["all"]["sharpe"],
                     cagr=s["all"]["cagr"], vol=s["all"]["vol"], max_dd=s["all"]["max_dd"], stock_gross_for_1p9x=round(k, 2), cagr_at_that_gross=s["all"]["cagr"] * k,
                     vol_at_that_gross=s["all"]["vol"] * k))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/r9_margin.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
