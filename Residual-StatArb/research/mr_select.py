"""
QR / Valid, stage 2: around the rule both grids agree on (IBS < 0.2 above the 200-day mean, out on IBS > 0.7 or a close above the 5-day mean):
  A  universes incl. stocks + ETFs together (more independent bets), slots, ranking
  B  a second entry condition, the IBS threshold
  C  robustness of the pick: decision time, costs, the close-decided look-ahead version (what most published IBS / RSI(2) backtests do)
Chosen on IN-SAMPLE Sharpe; every variant here is counted in the trial total of the deflated Sharpe.

  python mr_select.py A|B|C
"""
import sys
import time

import pandas as pd

from mr_run import E, MRModel, market_for

BASE = dict(entry_signal="ibs", entry_op="<", entry_value=0.2, trend_n=200, exit_signal="ibs", exit_op=">", exit_value=0.7, exit_sma_n=5, max_hold=10, minute=1530)
part = sys.argv[1] if len(sys.argv) > 1 else "A"
runs = []
if part in ("A", "A45"):
    b = {**BASE, "minute": 1545 if part == "A45" else 1530}
    for uni in ["top50+etf", "top100+etf", "etf", "top50", "top100"]:
        for k in (5, 10, 15, 20, 30):
            for rk in ("signal", "vol_low"):
                runs.append(MRModel(name=f"{uni} k{k} {rk} @{b['minute']}", universe=uni, slots=k, rank_by=rk, **b))
elif part == "D":                                                                             # the short side alone: fade a close near the high in a down-trend
    for uni in ["etf", "top50", "top100"]:
        for thr in (0.8, 0.9):
            for k in (5, 10, 20):
                runs.append(MRModel(name=f"SHORT {uni} ibs>{thr} k{k} @1545", universe=uni, direction="short", slots=k, rank_by="signal",
                                    **{**BASE, "minute": 1545, "entry_op": ">", "entry_value": thr}))
elif part == "E":                                                                             # sizing: inverse volatility (bonds / gold get more $, semis less), more slots
    uni = sys.argv[2] if len(sys.argv) > 2 else "top50+etf"
    for k in (10, 15, 20, 30):
        for vt in (0.0, 0.01, 0.015, 0.02):
            for mx in ((2.0, 3.0) if vt else (1.0,)):
                b = {**BASE, "minute": 1545}
                runs.append(MRModel(name=f"{uni} k{k} " + (f"inv_vol {vt:.3f} max{mx:g}" if vt else "equal") + " @1545", universe=uni, slots=k, rank_by="signal",
                                    sizing="inv_vol" if vt else "equal", vol_target=vt or 0.015, max_slot_mult=mx, **b))
elif part == "F":                                                                             # holding: next open (overnight) / exit variants; a wider stock universe
    b = {**BASE, "minute": 1545}
    for uni in ("top50+etf", "top100+etf", "top150+etf"):
        for k in (20, 30):
            runs.append(MRModel(name=f"{uni} k{k} rule @1545", universe=uni, slots=k, rank_by="signal", **b))
            runs.append(MRModel(name=f"{uni} k{k} OVERNIGHT @1545", universe=uni, slots=k, rank_by="signal", **{**b, "hold": "overnight"}))
            runs.append(MRModel(name=f"{uni} k{k} exit ibs>0.5 @1545", universe=uni, slots=k, rank_by="signal", **{**b, "exit_value": 0.5}))
            runs.append(MRModel(name=f"{uni} k{k} no-sma-exit @1545", universe=uni, slots=k, rank_by="signal", **{**b, "exit_sma_n": 0}))
elif part == "B":
    uni = sys.argv[2] if len(sys.argv) > 2 else "top50+etf"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    for thr in (0.1, 0.15, 0.2, 0.25, 0.3):
        for e2 in [("", "<", 0.0), ("ret1", "<", 0.0), ("rsi2", "<", 30.0), ("z5", "<", -1.0), ("ret3", "<", 0.0), ("ret1", "<", -0.01)]:
            b = {**BASE, "entry_value": thr, "minute": 1545}
            runs.append(MRModel(name=f"{uni} k{k} ibs<{thr} & {e2[0] or '-'}{e2[1] if e2[0] else ''}{e2[2] if e2[0] else ''}", universe=uni, slots=k, rank_by="signal",
                                entry2_signal=e2[0], entry2_op=e2[1], entry2_value=e2[2], **b))
elif part == "C":
    uni = sys.argv[2] if len(sys.argv) > 2 else "top50+etf"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    for minute in (1530, 1545, 1600):
        for cost in (1.0, 2.0, 3.0, 5.0):
            if minute == 1600 and cost != 1.0:
                continue
            b = {**BASE, "minute": minute}
            tag = " (LOOK-AHEAD: decided on the close it trades at)" if minute == 1600 else ""
            runs.append(MRModel(name=f"{uni} k{k} @{minute} cost {cost:g}bp{tag}", universe=uni, slots=k, rank_by="signal", cost_auction_bp=cost, **b))
rows = []
t0 = time.time()
for m in runs:
    r = E.run(m, market_for(m))
    s = r.stats
    rows.append(dict(model=m.name, sh_is=s["is"].get("sharpe"), sh_oos=s["oos"].get("sharpe"), sh_all=s["all"].get("sharpe"), cagr_is=s["is"].get("cagr"),
                     cagr_oos=s["oos"].get("cagr"), cagr=s["all"].get("cagr"), mdd=s["all"].get("max_dd"), vol=s["all"].get("vol"), trades=s["trades"], win=s.get("win_rate"),
                     expo=s["avg_exposure"], max_expo=s["max_exposure"], beta=s.get("beta_spy")))
    print(f"{m.name}: {E.summary(s)} ({time.time() - t0:.0f} s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"reports/mr_select_{part}{'_' + '_'.join(sys.argv[2:]) if len(sys.argv) > 2 else ''}.csv", index=False)
pd.set_option("display.width", 250)
print(df.sort_values("sh_is", ascending=False).round(3).to_string(index=False))
