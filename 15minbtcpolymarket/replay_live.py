"""Prove that live trading and the backtest take the same decisions.

Replays the LIVE engine (the code that sends real orders, with a paper broker) over a past period on a simulated clock - same Kalshi candle quotes as the backtest, a probability model that learns window by
window - and compares every trade with the backtest's: market, side, contracts, price, probability, P&L.  Exit code 0 only if all runs are identical.

    python replay_live.py                                  # every GUI preset, last 6 weeks
    python replay_live.py --preset R2+ --start 2026-06-01   # one preset, from a date
    python replay_live.py --preset AUTO --capital 25000 --slippage 0.03
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, datetime, timedelta, timezone

warnings.filterwarnings("ignore")

from trader.data import data_range, load_rows
from trader.presets import PRESETS, find_preset, preset_config
from trader.replay import verify


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="", help="preset name or prefix (default: every preset)")
    ap.add_argument("--start", default="", help="first day, YYYY-MM-DD (default: 42 days before the end of the data)")
    ap.add_argument("--end", default="", help="last day inclusive (default: the last day of the data)")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--slippage", type=float, default=0.02, help="assumed slippage added to the quoted ask (both sides)")
    ap.add_argument("--show", type=int, default=8, help="how many differences to print per run")
    a = ap.parse_args()

    rows = load_rows()
    lo, hi = data_range(rows)
    end = date.fromisoformat(a.end) if a.end else (hi - timedelta(seconds=1)).date()
    start = date.fromisoformat(a.start) if a.start else end - timedelta(days=41)
    names = [find_preset(a.preset)] if a.preset else [n for n, v in PRESETS.items() if v is not None]
    print(f"data {lo:%Y-%m-%d} -> {hi:%Y-%m-%d %H:%M} UTC | replay {start} -> {end} | capital ${a.capital:,.0f}, assumed slippage {a.slippage * 100:g}c\n")
    bad = 0
    for n in names:
        cfg = preset_config(n, start, end, capital=a.capital, slippage=a.slippage)
        cmp, bt, rp = verify(rows, cfg)
        print(f"{n.split(' · ')[0].split(' — ')[0][:44]:44s} backtest {len(bt.trades):4d} trades | live engine {len(rp.trades):4d} trades | {rp.windows:6d} windows in {rp.seconds:4.1f}s | "
              f"{'IDENTICAL' if cmp.identical else 'DIFFERENT'}")
        if not cmp.identical:
            bad += 1
            print(cmp.text(a.show))
    print("\n" + ("All runs identical: the live engine takes exactly the backtest's trades." if not bad else f"{bad} run(s) differ - see above."))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
