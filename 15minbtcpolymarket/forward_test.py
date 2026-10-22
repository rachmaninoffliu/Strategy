"""Pre-registered FORWARD TEST (read-only, paper only).

Rule R1 -- frozen 2026-10-07, before any data after that date was looked at:
  series KXBTC15M; market window opens at 22:00-23:59 UTC; decision minute 12 or 13 (first that qualifies);
  buy the UNDERDOG side at the ask: YES if mid < 0.5 else NO (cost = 1 - yes_bid); require cost in [0.05, 0.30];
  hold to settlement; 1 contract; Kalshi taker fee.
Evidence at registration (discovery Aug-6..Sep-9 / hold-out Sep-10..Oct-7): +4.9c / +7.3c per contract, win ~22%,
BTC-only (siblings do not show it), no mechanism found => treat as HYPOTHESIS.

Decision rule (SPRT, sd per contract 0.40, H0 EV=0 vs H1 EV=+3c, alpha .05, beta .20):
  LLR += (d/s^2) * (x - d/2),  d=.03, s=.40;   LLR >= +2.77 -> ACCEPT edge;   LLR <= -1.56 -> REJECT (stop).
  Also stop (REJECT) if cumulative paper P&L per contract < -3c after 150 trades.
Run weekly:  python forward_test.py        (refreshes data via collect.py, evaluates only markets opened on/after START)
"""
import subprocess
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from strategy import fee_pc

START = int(datetime(2026, 10, 8, tzinfo=timezone.utc).timestamp())
for _a in sys.argv[1:]:                      # --start=YYYY-MM-DD only for replaying history as a sanity check
    if _a.startswith("--start="):
        START = int(datetime.fromisoformat(_a.split("=")[1]).replace(tzinfo=timezone.utc).timestamp())
D, S = 0.03, 0.40


def refresh() -> None:
    since = str(START - 86400)
    for args in (["markets"], ["batch", since], ["gapfill", "3"], ["spot"]):
        subprocess.run([sys.executable, "collect.py", *args], check=False)


def evaluate() -> pd.DataFrame:
    from panel import build_panel
    P = build_panel()
    P = P[(P.open_ts >= START) & (P.bid_c > 0) & (P.ask_c < 1) & (P.ask_c >= P.bid_c) & ((P.ask_c - P.bid_c) <= 0.06)].copy()
    P["hr"] = pd.to_datetime(P.open_ts, unit="s", utc=True).dt.hour
    P = P[P.hr.isin([22, 23]) & P.minute.isin([12, 13])]
    P["mid"] = (P.bid_c + P.ask_c) / 2
    yes_is_dog = P.mid < 0.5
    P["cost"] = np.where(yes_is_dog, P.ask_c, 1 - P.bid_c).round(6)       # round: 1 - 0.70 = 0.30000000000000004 wrongly fell outside the registered band [0.05, 0.30] (fixed 2026-10-10; the rule is unchanged)
    P["win"] = np.where(yes_is_dog, P.yes, 1 - P.yes)
    P = P[(P.cost >= 0.05) & (P.cost <= 0.30)]
    T = P.sort_values(["ticker", "minute"]).groupby("ticker", as_index=False).first().sort_values("open_ts")
    T["pnl"] = T.win - T.cost - fee_pc(T.cost)
    return T


if __name__ == "__main__":
    if "--no-refresh" not in sys.argv:
        refresh()
    T = evaluate()
    if T.empty:
        print("no qualifying forward trades yet")
        sys.exit(0)
    llr = np.cumsum((D / S ** 2) * (T.pnl.values - D / 2))
    n = len(T)
    ev = T.pnl.mean() * 100
    se = T.pnl.std() / np.sqrt(n) * 100 if n > 1 else float("nan")
    print(f"forward trades n={n}  win={T.win.mean():.3f}  avg cost={T.cost.mean():.3f}  EV={ev:+.2f}c  (se {se:.2f}c)  cumulative paper P&L/contract={T.pnl.sum()*100:+.1f}c")
    print(f"SPRT LLR={llr[-1]:+.2f}   accept>= +2.77   reject<= -1.56")
    if llr[-1] >= 2.77:
        print("VERDICT: ACCEPT -- edge confirmed at pre-registered thresholds")
    elif llr[-1] <= -1.56 or (n >= 150 and T.pnl.mean() < -0.03):
        print("VERDICT: REJECT -- stop, do not trade")
    else:
        print("VERDICT: CONTINUE -- keep paper-trading; no capital")
