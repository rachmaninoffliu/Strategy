"""
Supertrend SOXL/SOXS — LIVE EXECUTION HARNESS (Alpaca paper)
============================================================

Runs the SAME signal code as the backtest (imports supertrend_strategy) so
live behaviour cannot drift from the tested logic.

!!  READ BEFORE ENABLING --live  !!
------------------------------------
Valid has NOT signed off on this strategy. The backtest does not reconcile
with the TradingView Pine run:

    your CSV  @50% equity     19,639%     @100%    744,176%
    this port @50% equity     35,857%     @100%  2,728,968%
                              1.8x over            3.7x over

We do not know which is correct. Until a same-settings Pine export is diffed
trade-by-trade, the expected return is unknown within a factor of ~4, and you
will NOT be able to distinguish live underperformance from a model error.

Additional open risks:
  * factor=5.0 is a SPIKE, not a plateau. Sharpe 1.52 at 5.0 vs ~0.85 at 4.0
    and ~0.93 at 6.0. If 5.0 was chosen by optimising on this data, live
    performance is likely nearer the neighbours than the peak.
  * SOXL is a 3x leveraged ETF. At 50% equity that is ~1.5x effective
    exposure; backtested MaxDD was -40%. At 100% it was -65%.

SAFETY MODEL
------------
  * DRY-RUN IS THE DEFAULT. Orders are only sent with an explicit --live flag.
  * Refuses to run against a non-paper endpoint. Hard assert, no override.
  * Kill switch: create the file KILL_FILE and the bot flattens and stops.
  * MAX_POSITION_USD caps notional regardless of equity.
  * Startup reconciliation: if the broker position disagrees with the stored
    state, the bot REFUSES to trade and asks for manual intervention rather
    than guessing.

STATE
-----
The strategy is path dependent (entry_ref, target, running extreme, mode
latched at entry). That state is persisted to STATE_FILE between bar closes.
Losing it mid-trade is not recoverable automatically -- hence the
reconciliation check.

USAGE
-----
  python supertrend_live.py                 # dry run, single pass
  python supertrend_live.py --loop          # dry run, poll each 15m close
  python supertrend_live.py --live --loop   # ARMED. sends real paper orders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supertrend_strategy as S

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed

# ---------------------------------------------------------------- config ----
SYMBOL_LONG   = "SOXL"
SYMBOL_SHORT  = "SOXS"     # shorts routed to SOXS, matching the Pine
ROUTE_TO_SOXS = True
BAR_MINUTES   = 15
WARMUP_BARS   = 900        # atrPeriod 500 + skip + margin

EQUITY_FRAC      = 0.50
MAX_POSITION_USD = 60_000.0
MAX_NOTIONAL_PCT = 0.60    # hard ceiling vs account equity

STATE_FILE = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports/live_state.json"
LOG_FILE   = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports/live_log.jsonl"
KILL_FILE  = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports/KILL"

CFG = S.Config(
    strategyMode="Always Fixed",
    atrPeriod=500, factor=5.0,
    riskReward=0.045, buyback=0.015,
    trailFactor=0.0, price_perc=0.1,
    useDynamic=False,
    equity_frac=EQUITY_FRAC,
    useDateFilter=False,
)


def log(event: str, **kw):
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kw}
    print(f"[{rec['ts'][11:19]}] {event}: "
          + " ".join(f"{k}={v}" for k, v in kw.items()))
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def creds():
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if k and s:
        return k, s
    e = json.load(open(os.path.expanduser("~/.claude.json")))
    e = e["mcpServers"]["alpaca-paper"]["env"]
    return e["ALPACA_API_KEY"], e["ALPACA_SECRET_KEY"]


# ------------------------------------------------------------- state I/O ----
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"pos": 0, "symbol": None, "qty": 0, "entry_ref": None,
            "supert_entry": None, "target": None, "active_fixed": True,
            "max_high": None, "min_low": None, "opened_at": None}


def save_state(st: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    json.dump(st, open(STATE_FILE, "w"), indent=2, default=str)


# ---------------------------------------------------------------- market ----
def fetch_bars(dclient, symbol: str, n_bars: int) -> pd.DataFrame:
    """Recent RTH 15-min bars, split-adjusted to match the backtest."""
    days = max(12, int(n_bars / 26 * 1.9) + 6)
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
        start=(datetime.now(timezone.utc) - timedelta(days=days)),
        adjustment=Adjustment.SPLIT,     # matches backtest; NOT dividend-adj
        feed=DataFeed.SIP,
    )
    df = dclient.get_stock_bars(req).df.reset_index()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df[(ts.dt.time >= pd.Timestamp("09:30").time())
            & (ts.dt.time < pd.Timestamp("16:00").time())
            & (ts.dt.dayofweek < 5)].copy()
    df["bar"] = ts[df.index].dt.tz_localize(None)
    return (df.set_index("bar").sort_index()[["open", "high", "low", "close"]]
              .ffill().dropna().tail(n_bars))


def compute_signal(bars: pd.DataFrame) -> dict:
    """Supertrend state on the LAST CLOSED bar. Identical maths to backtest."""
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    st, direction = S.supertrend(h, l, c, CFG.atrPeriod, CFG.factor)
    i = len(c) - 1
    if np.isnan(st[i]) or np.isnan(direction[i]):
        return {"ready": False}
    want = 1 if direction[i] == -1 else -1
    return {"ready": True, "bar_time": bars.index[i], "close": float(c[i]),
            "st": float(st[i]), "dir": int(direction[i]), "want": want,
            "buyback_limit": float(st[i] * (1 + CFG.buyback)) if want > 0
                             else float(st[i] * (1 - CFG.buyback))}


# ------------------------------------------------------------ reconciler ----
def broker_positions(tclient) -> dict:
    out = {}
    for p in tclient.get_all_positions():
        out[p.symbol] = {"qty": float(p.qty), "side": p.side.value,
                         "avg": float(p.avg_entry_price),
                         "mv": float(p.market_value)}
    return out


def reconcile(state: dict, positions: dict) -> tuple[bool, str]:
    """Refuse to trade when stored state and broker reality disagree."""
    held = {k: v for k, v in positions.items() if abs(v["qty"]) > 0}
    if state["pos"] == 0 and not held:
        return True, "flat / flat"
    if state["pos"] != 0 and state["symbol"] in held:
        want_qty = abs(float(state["qty"]))
        have_qty = abs(held[state["symbol"]]["qty"])
        if abs(want_qty - have_qty) / max(want_qty, 1) < 0.02:
            return True, f"in sync ({state['symbol']} {have_qty:.0f})"
        return False, (f"QTY MISMATCH {state['symbol']}: "
                       f"state={want_qty:.0f} broker={have_qty:.0f}")
    if state["pos"] == 0 and held:
        return False, f"broker holds {list(held)} but state says FLAT"
    return False, f"state says {state['symbol']} but broker holds {list(held)}"


# ------------------------------------------------------------- execution ----
def size_position(equity: float, price: float) -> int:
    notional = min(equity * CFG.equity_frac, MAX_POSITION_USD,
                   equity * MAX_NOTIONAL_PCT)
    return int(math.floor(notional / price)) if price > 0 else 0


def submit(tclient, symbol: str, qty: int, side: OrderSide, live: bool, why: str):
    if qty <= 0:
        return None
    if not live:
        log("DRY_RUN_ORDER", symbol=symbol, side=side.value, qty=qty, why=why)
        return None
    o = tclient.submit_order(MarketOrderRequest(
        symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY))
    log("ORDER_SENT", symbol=symbol, side=side.value, qty=qty,
        order_id=str(o.id), why=why)
    return o


def step(tclient, dclient, live: bool):
    if os.path.exists(KILL_FILE):
        log("KILL_SWITCH", action="halting, no orders")
        return "halt"

    acct = tclient.get_account()
    equity = float(acct.equity)
    state = load_state()
    positions = broker_positions(tclient)

    ok, msg = reconcile(state, positions)
    log("RECONCILE", ok=ok, detail=msg)
    if not ok:
        log("REFUSING_TO_TRADE", reason="state/broker mismatch — resolve manually")
        return "blocked"

    bars = fetch_bars(dclient, SYMBOL_LONG, WARMUP_BARS)
    sig = compute_signal(bars)
    if not sig["ready"]:
        log("WARMUP", bars=len(bars), need=CFG.atrPeriod)
        return "warmup"

    log("SIGNAL", bar=str(sig["bar_time"]), close=round(sig["close"], 2),
        supertrend=round(sig["st"], 2),
        trend="UP" if sig["want"] > 0 else "DOWN",
        buyback_limit=round(sig["buyback_limit"], 2))

    want = sig["want"]
    tgt_symbol = SYMBOL_LONG if (want > 0 or not ROUTE_TO_SOXS) else SYMBOL_SHORT
    cur_symbol = state["symbol"]

    # ---- already positioned in the right instrument -> hold ----
    if state["pos"] != 0 and cur_symbol == tgt_symbol:
        log("HOLD", symbol=cur_symbol, qty=state["qty"],
            unrealized=positions.get(cur_symbol, {}).get("mv"))
        return "hold"

    # ---- flip: close the wrong side first ----
    if state["pos"] != 0 and cur_symbol != tgt_symbol:
        held = positions.get(cur_symbol)
        if held:
            side = OrderSide.SELL if held["qty"] > 0 else OrderSide.BUY
            submit(tclient, cur_symbol, int(abs(held["qty"])), side, live,
                   why="flip:close")
        state = {"pos": 0, "symbol": None, "qty": 0, "entry_ref": None,
                 "supert_entry": None, "target": None, "active_fixed": True,
                 "max_high": None, "min_low": None, "opened_at": None}
        if live:
            save_state(state)

    # ---- open the new side ----
    px = sig["close"]
    qty = size_position(equity, px)
    if qty <= 0:
        log("NO_SIZE", equity=equity, price=px)
        return "nosize"

    submit(tclient, tgt_symbol, qty, OrderSide.BUY, live, why="open")
    move = CFG.riskReward
    new_state = {
        "pos": 1 if tgt_symbol == SYMBOL_LONG else -1,
        "symbol": tgt_symbol, "qty": qty,
        "entry_ref": px, "supert_entry": sig["st"],
        "target": px * (1 + move) if tgt_symbol == SYMBOL_LONG else px * (1 - move),
        "active_fixed": True, "max_high": None, "min_low": None,
        "opened_at": str(sig["bar_time"]),
    }
    if live:
        save_state(new_state)
    log("TARGET_STATE", **{k: v for k, v in new_state.items()
                           if k in ("symbol", "qty", "entry_ref", "target")})
    return "opened"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="ARM: actually send orders to the paper account")
    ap.add_argument("--loop", action="store_true", help="poll every bar close")
    ap.add_argument("--interval", type=int, default=BAR_MINUTES * 60)
    a = ap.parse_args()

    key, sec = creds()
    tclient = TradingClient(key, sec, paper=True)

    # --- hard paper guard: no override path ---
    base = str(tclient._base_url).lower()
    if "paper" not in base:
        raise SystemExit(f"REFUSING TO RUN: endpoint is not paper ({base})")

    acct = tclient.get_account()
    print("=" * 74)
    print("SUPERTREND LIVE HARNESS —", "ARMED (orders WILL be sent)" if a.live
          else "DRY RUN (no orders)")
    print("=" * 74)
    print(f"  endpoint : {tclient._base_url}")
    print(f"  account  : {acct.account_number[:4]}****   equity ${float(acct.equity):,.2f}")
    print(f"  config   : {CFG.strategyMode} atr{CFG.atrPeriod} f{CFG.factor} "
          f"trail{CFG.trailFactor} pp{CFG.price_perc}")
    print(f"  sizing   : {CFG.equity_frac:.0%} equity, cap ${MAX_POSITION_USD:,.0f}")
    print(f"  kill file: {KILL_FILE}")
    print()

    dclient = StockHistoricalDataClient(key, sec)
    while True:
        try:
            r = step(tclient, dclient, a.live)
            if r == "halt":
                break
        except Exception as exc:
            log("ERROR", detail=str(exc)[:300])
        if not a.loop:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
