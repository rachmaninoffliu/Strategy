"""Headless PAPER session (no orders, no credentials): runs the live engine against real-time Kalshi quotes.
Usage: python paper_run.py [--capital 1000] [--minutes 30] [--lookback 0]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trader.data import DATA, load_rows
from trader.kalshi import KalshiQuotes
from trader.live import LiveConfig, LiveEngine, PaperBroker, build_stats
from trader.strategy import StrategyParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--lookback", type=int, default=0)
    ap.add_argument("--hours", default="22,23", help="UTC window-open hours (pipeline testing: relax these)")
    ap.add_argument("--decisions", default="12,13", help="decision minutes")
    ap.add_argument("--cmin", type=float, default=0.05)
    ap.add_argument("--cmax", type=float, default=0.30)
    ap.add_argument("--manual-p", type=float, default=0.0, help="force a manual win probability (pipeline testing only)")
    ap.add_argument("--legs", default="DOG", help="DOG, FAV or DOG,FAV")
    ap.add_argument("--fmin", type=float, default=0.85)
    ap.add_argument("--fmax", type=float, default=0.97)
    ap.add_argument("--p-model", default="overall", choices=["overall", "by_hour"])
    ap.add_argument("--p-basis", default="edge", choices=["edge", "win_rate"])
    a = ap.parse_args()
    from trader.strategy import parse_int_list
    params = StrategyParams(hours=parse_int_list(a.hours), minutes=parse_int_list(a.decisions, 1, 14), cost_min=a.cmin, cost_max=a.cmax,
                            legs=tuple(x.strip().upper() for x in a.legs.split(',')), fav_min=a.fmin, fav_max=a.fmax)
    cfg = LiveConfig(capital=a.capital, params=params, mode="paper", lookback_days=a.lookback,
                     prob_source="manual" if a.manual_p > 0 else "history", manual_p=a.manual_p,
                     p_model=a.p_model, p_basis=a.p_basis)
    stats = build_stats(load_rows(), cfg)
    print(f"{params.describe()} | {len(stats._t)} stat groups, {stats.n_total(time.time())} historical signals; paper capital ${a.capital:,.0f}; "
          f"running {a.minutes:.0f} min", flush=True)

    def emit(kind: str, d: dict) -> None:
        if kind == "log":
            print(d["msg"], flush=True)

    from trader.ledger import Ledger
    eng = LiveEngine(cfg, KalshiQuotes(), PaperBroker(cfg.fee_model, cfg.paper_slippage), stats, emit, log_dir=DATA / "live", ledger=Ledger())          # saved to the same trade ledger the GUI shows
    end = time.time() + a.minutes * 60
    nxt = eng.next_decision_in(time.time())
    print(f"next decision in {nxt:.0f}s" if nxt is not None else "no decision windows configured", flush=True)
    while time.time() < end:
        eng.tick()
        time.sleep(0.25)
    eng.push_status()
    print(f"done. trades={len(eng.trades)} equity=${eng.equity:,.2f} open={len(eng._positions)} pending={len(eng._shadow)}", flush=True)


if __name__ == "__main__":
    main()
