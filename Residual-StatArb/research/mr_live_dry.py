"""
QT: a DRY RUN of the mean-reversion live engine against the real Alpaca paper account - read-only (account, positions, clock, bars). The engine is NOT armed: nothing can be sent.
Runs the pre-flight, then plans the latest session's decision exactly as the loop would (outside the session: the bars available at that day's decision time), and compares
it with the same decision rebuilt from the backtest's data (the SIP cache) - the live-vs-backtest check for one day.

  python mr_live_dry.py [preset index]
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))

import pandas as pd  # noqa: E402

from resarb.mr import engine as E  # noqa: E402
from resarb.mr import live as ML  # noqa: E402
from resarb.mr import presets as PR  # noqa: E402
from resarb.mr.market import Market  # noqa: E402
from resarb.mr.models import Pos, Signals, decide  # noqa: E402

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
m = PR.get(PR.preset_names()[idx])
feed = ML.default_feed(m.minute)
folder = tempfile.mkdtemp(prefix="mr_dry_")                                                   # journal / state of this check stay out of the app folder
eng = ML.MRLiveEngine(m, ML.MRLiveSettings(armed=False, feed=feed), folder, log=lambda msg, k="info": print(f"  [{k}] {msg}"))
assert not eng.live.armed
print(f"model: {m.name} · feed {feed}")
for c in eng.preflight():
    print(f"  {c.level.upper():4s} {c.name}: {c.msg}")
plan = eng.plan(force=True)
print(f"\nplan for {plan.day:%Y-%m-%d} (snapshot {plan.snapshot_to:%H:%M}) · capital ${plan.capital:,.0f} · {len(plan.positions)} position(s) · notes:")
for n in plan.notes:
    print("   -", n)
pd.set_option("display.width", 220)
print(plan.candidates.head(25).to_string(index=False))
print(plan.orders[["symbol", "side", "qty", "tif", "ref_px", "notional", "why", "ok"]].to_string(index=False))
# the same decision from the backtest's data (SIP cache), same positions and capital
day = plan.day
syms = list(dict.fromkeys(eng.universe_symbols(day) + list(plan.positions.symbol)))
mk = Market(syms, minute=m.minute, start=day - pd.Timedelta(days=460), end=day + pd.Timedelta(days=1))
if len(mk.dates) and mk.dates[-1] == day:
    t = len(mk.dates) - 1
    pos = {r.symbol: Pos(r.symbol, int(r.shares), max(t - int(r.age), 0), float(r.entry_px)) for r in plan.positions.itertuples()}
    dec = decide(m, t, Signals(mk), E.universe_mask(mk, m)[t], pos, plan.capital)
    bt = {s: abs(sh) for s, sh, _, _ in dec.entries}
    lv = {r.symbol: int(r.qty) for r in plan.orders.itertuples() if r.kind == "entry"}
    same = sum(1 for s in bt if lv.get(s) == bt[s])
    print(f"\nentries live {len(lv)} vs backtest data {len(bt)}: {same} identical (ticker + shares); live only {sorted(set(lv) - set(bt))}, backtest only {sorted(set(bt) - set(lv))}")
else:
    print(f"\n{day:%Y-%m-%d} is not in the SIP cache yet - update the data, then compare")
print("\nDRY RUN - the engine was not armed; no order was sent.")
