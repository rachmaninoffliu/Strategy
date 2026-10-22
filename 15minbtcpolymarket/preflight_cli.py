"""Pre-flight go / no-go from the terminal (same checks as the GUI's Start button; nothing is ever sent to an exchange).

    python preflight_cli.py                         # paper-style check with the default R2+ preset, $1,000
    python preflight_cli.py --capital 500 --live --key kalshi.pem --key-id <uuid> [--env production]    # also checks the Kalshi account (default env: demo)
    python preflight_cli.py --no-refresh            # do not download fresh Kalshi data first

Exit code: 0 = GO, 1 = CAUTION, 2 = NO-GO  (handy for a Task Scheduler job at 21:40 UTC).
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

from trader import preflight
from trader.live import LiveConfig
from trader.strategy import StrategyParams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--live", action="store_true", help="also check the Kalshi account")
    ap.add_argument("--key", default="", help="path to your Kalshi private key file (.pem)")
    ap.add_argument("--key-id", default="", help="your Kalshi API Key ID (UUID)")
    ap.add_argument("--env", default="demo", choices=["demo", "production"], help="Kalshi environment for the account check (paper checks always read production market data)")
    ap.add_argument("--hours", default="22,23", help="hours to trade with --r2-only (R2+ scouts all 24 hours with 22,23 as its core)")
    ap.add_argument("--r2-only", action="store_true", help="check the hard-wired R2 preset instead of the default R2+ (core + scout)")
    ap.add_argument("--cmin", type=float, default=0.15)
    ap.add_argument("--cmax", type=float, default=0.35)
    ap.add_argument("--cap", type=float, default=2.0, help="max stake per trade, %% of bankroll")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    from trader.strategy import parse_int_list
    common = dict(capital=a.capital, max_frac=a.cap / 100, mode="live" if a.live else "paper")
    if a.r2_only:
        cfg = LiveConfig(params=StrategyParams(hours=parse_int_list(a.hours), cost_min=a.cmin, cost_max=a.cmax), lookback_days=45, halflife_days=10.0, **common)
    else:                                                    # default preset R2+: core 22-23 + skeptical all-hours scout, 14-day decaying memory
        cfg = LiveConfig(params=StrategyParams(hours=tuple(range(24)), cost_min=a.cmin, cost_max=a.cmax), p_model="by_hour", hour_k=100.0, halflife_days=14.0,
                         core_hours=(22, 23), core_k=10.0, lookback_days=45, **common)
    rep = preflight.run_preflight(cfg, live=a.live, key_path=a.key, key_id=a.key_id, env=a.env if a.live else "production", refresh=not a.no_refresh, log=lambda s: print("  ..", s[:140]))
    print()
    print(rep.text())
    p = preflight.save_report(rep)
    if p:
        print(f"\nreport saved: {p}")
    return {"GO": 0, "CAUTION": 1, "NO-GO": 2}[rep.verdict]


if __name__ == "__main__":
    sys.exit(main())
