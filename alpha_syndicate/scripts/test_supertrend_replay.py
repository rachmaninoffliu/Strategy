"""Does the LIVE Engine trade exactly like the (live-accurate) backtest?   Usage:  python test_supertrend_replay.py

The real supertrend_gui.Engine.poll_once() is driven bar by bar (twice per bar: at the open and just before the close) against a simulated broker (supertrend_ui/replay.py) and every
round trip - plus the position still open at the end - is compared with SupertrendStrategy(strict_entries=True) on the same bars. Nothing here touches Alpaca or the network."""
import os
import sys
import time

os.environ.setdefault("PYTHONUTF8", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd                                           # noqa: E402

import supertrend_gui as GU                                   # noqa: E402
from supertrend_ui import data as D                           # noqa: E402
from supertrend_ui import params as P                         # noqa: E402
from supertrend_ui import replay as R                         # noqa: E402
from supertrend_ui.verify import run_replay_process           # noqa: E402

BARS = D.read_cache("SOXL", 15)
END = f"{BARS.index[-1]:%Y-%m-%d}"
VALS = dict(P.DEFAULTS)
t0 = time.time()


def window(months):
    return f"{(pd.Timestamp(END) - pd.DateOffset(months=months)):%Y-%m-%d}"


# 1. the reference settings, SOXL shortable: identical trades, identical entry prices, same open position
res = R.replay_live(BARS, VALS, window(2), END, shortable=True)
s = res.stats
assert res.identical and s["bt_trades"] == s["live_trades"] >= 10 and s["backtest_only"] == s["live_only"] == 0 and s["full_matches"] == s["bt_trades"], R.describe(res)
assert s["entry_px_diff_max"] < 1e-3, s["entry_px_diff_max"]                                       # every entry fills at the backtest's price
oc = s["open_check"]
assert oc["status"] in ("match", "timing") and oc["status"] != "error", oc                         # the Engine's frozen levels for the position it still holds == the backtest's
if s["bt_open"] is not None:
    assert s["live_open"] is not None and s["live_open"]["side"] == s["bt_open"]["side"] and oc["live"] is not None and oc["expected"] is not None, (s["bt_open"], s["live_open"], oc)
else:
    assert s["live_open"] is None and oc["live"] is None and oc["expected"] is None, (s["live_open"], oc)
print(f"1. IDENTICAL: {s['bt_trades']} trades over {s['bars']:,} bars, open position check {oc['status'].upper()}  ({res.seconds:.0f} s)")

# 2. the harness can fail: an Engine that never enters must NOT compare equal
real_decide = GU.Engine.decide
GU.Engine.decide = lambda self, *a, **k: None
try:
    dead = R.replay_live(BARS, VALS, window(1), END, shortable=True)
finally:
    GU.Engine.decide = real_decide
assert not dead.identical and dead.stats["live_trades"] < dead.stats["bt_trades"] and dead.stats["backtest_only"] > 0, R.describe(dead)
print(f"2. a broken engine is detected ({dead.stats['bt_trades']} backtest trades, {dead.stats['live_trades']} live)")

# 3. non-default sizing / exits (half size, tighter run-up, 1% re-entry, 8% take-profit, trail at 30%)
vals3 = {**VALS, "equity_frac": 50.0, "riskReward": 3.0, "buyback": 1.0, "price_perc": 8.0, "trailFactor": 0.3}
r3 = R.replay_live(BARS, vals3, window(2), END, shortable=True)
assert r3.identical and r3.stats["bt_trades"] >= 10, R.describe(r3)
print(f"3. IDENTICAL with non-default settings: {r3.stats['bt_trades']} trades")

# 4. SOXL not shortable: shorts go to SOXS, whose exits are POLLED (a few entries/exits differ by design) - it must still track the backtest closely
r4 = R.replay_live(BARS, VALS, window(2), END, shortable=False)
s4 = r4.stats
assert s4["bt_trades"] >= 10 and s4["live_trades"] >= 10 and abs(s4["bt_trades"] - s4["live_trades"]) <= max(3, s4["bt_trades"] // 8) and s4["full_matches"] >= int(0.8 * s4["bt_trades"]), R.describe(r4)
print(f"4. SOXS route: {s4['full_matches']} of {s4['bt_trades']} trades match exactly ({s4['live_trades']} live) - polled exits differ by design")

# 5. the same replay in a CHILD process, as the Backtest tab's Verify button runs it
prog = []
out = run_replay_process(VALS, window(1), END, True, progress=lambda i, n: prog.append((i, n)))
assert out["identical"] and out["stats"]["bt_trades"] == out["stats"]["live_trades"] and out["strict"] is True and out["text"].startswith("IDENTICAL") and out["rows"], out["text"]
assert out["stats"]["open_check"]["status"] in ("match", "timing") and prog and prog[-1][0] <= prog[-1][1]
print(f"5. child process OK ({len(out['rows'])} rows)")

# 6. the classic (look-ahead) backtest is NOT what a live engine does: compared with it over a long window, trades differ
r6 = R.replay_live(BARS, VALS, window(6), END, shortable=True, strict=False)
print(f"6. classic backtest vs live engine over 6 months: {R.describe(r6, limit=3).splitlines()[0]}")

print(f"replay test OK ({time.time() - t0:.0f} s)")
