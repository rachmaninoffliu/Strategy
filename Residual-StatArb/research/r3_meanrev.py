"""
QR / Valid: residual mean-reversion grid - the Avellaneda-Lee OU s-score on 15-minute residuals and the cross-sectional residual reversal. Chosen on IN-SAMPLE Sharpe only.

  python r3_meanrev.py [feed]
"""
import itertools
import sys

from common import OUT, ResidualMeanReversion, S, log, pd, row, run_bt

feed = sys.argv[1] if len(sys.argv) > 1 else "iex"
base = {"universe_n": 100, "feed": feed, "beta_days": 126, "cost_model": "spread", "etf_cost_bp": 0.5, "hedge": "etf"}
ctx = S.context(ResidualMeanReversion(base), log=log)
log(ctx.p.describe())
rows = []
ou = list(itertools.product([26, 52, 130, 260], [(1.25, 0.5), (1.5, 0.5), (2.0, 0.75), (2.0, 0.25)], ["15min", "1h"], [False, True]))
for k, (W, (so, sc), reb, eod) in enumerate(ou):
    st = ResidualMeanReversion({**base, "mode": "ou", "ou_window": W, "s_open": so, "s_close": sc, "rebalance": reb, "flat_eod": eod, "method": "threshold", "names": 30})
    bt = run_bt(st, ctx)
    rows.append(row(bt, mode="ou", W=W, s_open=so, s_close=sc, reb=reb, flat_eod=eod))
    r = rows[-1]
    log(f"[ou {k + 1}/{len(ou)}] W={W} {so}/{sc} {reb} eod={eod}: IS {r['sh_is']:.2f} OOS {r['sh_oos']:.2f} gross IS {r['gross_sh_is']:.2f} turn {r['turn']:.0f}x")
xs = list(itertools.product([1, 2, 4, 13, 26, 130], [(0.1, 0.25), (0.2, 0.35)], ["15min", "1h", "daily"], [False, True]))
for k, (kb, (qi, qo), reb, eod) in enumerate(xs):
    if reb == "daily" and eod:
        continue
    st = ResidualMeanReversion({**base, "mode": "xs", "xs_bars": kb, "q_in": qi, "q_out": qo, "rebalance": reb, "trade_time": "15:45" if reb == "daily" else "15:45", "flat_eod": eod,
                                "method": "quantile"})
    bt = run_bt(st, ctx)
    rows.append(row(bt, mode="xs", xs_bars=kb, q_in=qi, q_out=qo, reb=reb, flat_eod=eod))
    r = rows[-1]
    log(f"[xs {k + 1}/{len(xs)}] k={kb} q={qi}/{qo} {reb} eod={eod}: IS {r['sh_is']:.2f} OOS {r['sh_oos']:.2f} gross IS {r['gross_sh_is']:.2f} turn {r['turn']:.0f}x")
df = pd.DataFrame(rows).sort_values("sh_is", ascending=False)
df.to_csv(f"{OUT}/r3_meanrev_{feed}.csv", index=False)
pd.set_option("display.width", 250)
print(df.head(25).round(3).to_string(index=False))
print(f"\n{len(df)} variants; IS Sharpe mean {df.sh_is.mean():.2f} sd {df.sh_is.std():.2f}; OOS of the IS-best: {df.iloc[0].sh_oos:.2f}")
print("best gross (zero-cost) IS Sharpe:", df.sort_values("gross_sh_is", ascending=False).head(5)[["mode", "W", "xs_bars", "reb", "gross_sh_is", "sh_is", "turn"]].to_string(index=False))

