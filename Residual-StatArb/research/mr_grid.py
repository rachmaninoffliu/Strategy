"""
QR / Valid: grid over long-only mean-reversion models in the trade-level, no-leverage engine. Chosen on IN-SAMPLE (2016-09 .. 2022) Sharpe; every variant counts as a trial.

  python mr_grid.py stocks        top-100 and top-50 large caps
  python mr_grid.py etf           the 59-ETF universe (countries, sectors, industries, bonds, commodities)
"""
import itertools
import sys
import time

import pandas as pd

from mr_run import E, MRModel, market_for

fam = sys.argv[1] if len(sys.argv) > 1 else "stocks"
universes = ["top100", "top50"] if fam == "stocks" else ["etf"]
entries = [("rsi2", "<", 5), ("rsi2", "<", 10), ("rsi2", "<", 20), ("ibs", "<", 0.1), ("ibs", "<", 0.2), ("ret1", "<", -0.02), ("ret1", "<", -0.03),
           ("z5", "<", -1.5), ("z5", "<", -2.0), ("ret3", "<", -0.05), ("ret5", "<", -0.07)]
exits = [dict(exit_signal="ibs", exit_op=">", exit_value=0.7, exit_sma_n=5, max_hold=10), dict(exit_signal="rsi2", exit_op=">", exit_value=60, exit_sma_n=0, max_hold=10),
         dict(exit_signal="", exit_sma_n=5, max_hold=10), dict(exit_signal="", exit_sma_n=0, max_hold=1)]
grid = list(itertools.product(universes, entries, [0, 200], range(len(exits)), [5, 10, 20], ["signal", "vol_low"]))
rows = []
t0 = time.time()
for k, (uni, (sig, op, val), tr, xi, slots, rank) in enumerate(grid):
    x = exits[xi]
    m = MRModel(name=f"{uni} {sig}{op}{val} trend{tr} exit{xi} k{slots} {rank}", universe=uni, entry_signal=sig, entry_op=op, entry_value=val, trend_n=tr, slots=slots,
                rank_by=rank, **x)
    r = E.run(m, market_for(m))
    s = r.stats
    rows.append(dict(universe=uni, signal=sig, op=op, value=val, trend=tr, exit=xi, slots=slots, rank=rank, sh_is=s["is"].get("sharpe"), sh_oos=s["oos"].get("sharpe"),
                     sh_all=s["all"].get("sharpe"), cagr_is=s["is"].get("cagr"), cagr_oos=s["oos"].get("cagr"), cagr_all=s["all"].get("cagr"), mdd=s["all"].get("max_dd"),
                     trades=s["trades"], win=s.get("win_rate"), expo=s["avg_exposure"], hold=s.get("avg_hold")))
    if (k + 1) % 25 == 0:
        print(f"[{k + 1}/{len(grid)}] {time.time() - t0:.0f} s; best IS so far {max(r_['sh_is'] or -9 for r_ in rows):.2f}", flush=True)
df = pd.DataFrame(rows).sort_values("sh_is", ascending=False)
df.to_csv(f"reports/mr_grid_{fam}.csv", index=False)
pd.set_option("display.width", 250)
print(df.head(25).round(3).to_string(index=False))
print(f"\n{len(df)} variants; IS Sharpe mean {df.sh_is.mean():.2f} sd {df.sh_is.std():.2f}; share OOS > 0: {(df.sh_oos > 0).mean():.0%}")
