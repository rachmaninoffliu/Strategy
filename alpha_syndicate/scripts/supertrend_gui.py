"""
Supertrend SOXL/SOXS — live trading ENGINE (library)
====================================================

This module holds the live Engine (a background thread that reports through a queue), the guard rails and the credential helper. The window is
supertrend_app.py (PyQt6, package supertrend_ui/), which imports and drives this Engine; the previous Tkinter window is supertrend_app_tk.py.
Running this file directly just opens that window.

ENGINE FEATURES
  * Session baseline: starting capital + start date (P&L / compounding base)
  * Reports open positions, signal, next-trade plan, exit levels and every order (the exact payload) as events
  * Rolling event log, also written to live_log.jsonl
  * PAPER / LIVE account, ARMED (send orders) or DRY RUN
  * Guard rails: market hours, daily loss limit, trade cap, KILL file
  * SOXL exits rest at the broker; SOXS exits are polled and sent at market

RISK NOTICE (displayed in-app, repeated here)
  Valid has not signed off. The backtest does not reconcile with the
  TradingView Pine run -- this port is 1.8x over at 50% sizing and 3.7x over
  at 100%. Expected return is unknown within ~4x. factor=5.0 is a parameter
  SPIKE (Sharpe 1.52 vs ~0.85/~0.93 at 4.0/6.0), so live results may land
  nearer the neighbours than the peak. SOXL/SOXS are 3x leveraged.

  Run  python supertrend_app.py      (python supertrend_gui.py hands off to it)
"""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import traceback
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import talib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supertrend_strategy as S

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (MarketOrderRequest, LimitOrderRequest,
                                     StopOrderRequest, GetOrdersRequest,
                                     GetPortfolioHistoryRequest,
                                     TakeProfitRequest, StopLossRequest)
from alpaca.trading.enums import (OrderSide, TimeInForce, QueryOrderStatus,
                                  OrderClass)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed

# --------------------------------------------------------------- constants --
SYMBOL_LONG, SYMBOL_SHORT = "SOXL", "SOXS"
PREFER_SHORT_SOXL = True   # short SOXL directly when borrow allows;
                           # fall back to buying SOXS only if hard-to-borrow

# Pine sizing asymmetry:
#   longs  -> strategy.default_entry_qty()  == the chosen % of equity
#   shorts -> f_soxlShortQty / f_soxsQtyFromEquity == equity * 0.5
# so a short is always HALF the notional of a long at the same setting.
SHORT_EQUITY_MULT = 0.5

# --------------------------------------------------------- display colours --
# Direction colours, NOT P&L colours: long = green, short = red.
#
# CRITICAL: a SOXL short is expressed at the broker as a LONG SOXS position,
# so `position.side` reads "long" for what is strategically a SHORT. Colour and
# labelling must therefore be derived from the SYMBOL, never from the raw
# broker side, or every short shows up green.
COL_LONG  = "#16a34a"   # green
COL_SHORT = "#dc2626"   # red
COL_FLAT  = "#6b7280"   # grey


def strategy_side(symbol=None, side=None) -> str:
    """Strategy direction ('long'/'short') behind a symbol and/or raw side."""
    if symbol == SYMBOL_SHORT:
        return "short"                      # long SOXS == short SOXL
    if side is not None and str(side).lower() in ("sell", "short", "-1", "s"):
        return "short"                      # direct SOXL short
    return "long"


def side_colour(side: str) -> str:
    return COL_SHORT if str(side).lower().startswith("s") else COL_LONG


BAR_MINUTES   = 15
# Bars of history loaded on every poll. The Supertrend (Wilder ATR, a 500-bar seed)
# only converges to the backtest's full-history value after ~7 ATR lengths: a
# 900-bar window flipped the trend direction on ~1.5% of bars and, replayed against
# the backtest, missed or invented about a third of the trades. 4000 bars = ~150
# sessions (one request of about 300 days).
WARMUP_BARS   = 4000
POLL_SECONDS  = 60

# --- runtime guard rails (the loop runs until STOPped, so these matter) ---
REQUIRE_MARKET_OPEN = True     # never send orders outside RTH
MAX_TRADES_PER_DAY  = 12       # circuit breaker on churn
DAILY_LOSS_LIMIT_PCT = -8.0    # halt for the day beyond this drawdown

REPORTS   = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
STATE_FILE= os.path.join(REPORTS, "gui_state.json")
LOG_FILE  = os.path.join(REPORTS, "live_log.jsonl")
KILL_FILE = os.path.join(REPORTS, "KILL")
os.makedirs(REPORTS, exist_ok=True)

CFG = S.Config(strategyMode="Always Fixed", atrPeriod=500, factor=5.0,
               riskReward=0.045, buyback=0.015, trailFactor=0.0,
               price_perc=0.1, useDynamic=False, useDateFilter=False,
               commission_bps=5.0)      # only read by simulate_session (the 5 bp reference)

LIVE_CONFIRM = "TRADE REAL MONEY"


# ------------------------------------------------ backtest-parity helpers --
# Mirrors of SupertrendStrategy.run / ._trail_factor so the live exits and the
# entry mode follow the backtest. They take the Config explicitly (not the CFG
# global) so the Signal monitor can reuse them (supertrend_ui.data).
def regime_inputs(cfg, h, l, c, i):
    """(adx, high_vol) at bar ``i``, computed exactly as SupertrendStrategy.run does.

    adx is None unless the ADX dynamic trail is on; high_vol is None unless the
    mode is 'Auto-Switch via Vol'. While an indicator is still warming up adx is
    NaN and high_vol is False, as in the backtest."""
    adx = high_vol = None
    if cfg.useDynamic:
        adx = float(talib.ADX(h, l, c, timeperiod=cfg.adxLength)[i])
    if cfg.strategyMode == "Auto-Switch via Vol":
        atr_pct = talib.ATR(h, l, c, timeperiod=cfg.volLength) / c
        base = talib.SMA(atr_pct, timeperiod=cfg.volSmaLength)
        high_vol = bool(base[i] == base[i] and atr_pct[i] > base[i] * cfg.volThreshold)
    return adx, high_vol


def dynamic_trail_factor(cfg, adx_val=None):
    """The trailing factor the backtest uses (SupertrendStrategy._trail_factor).

    With the ADX trail on, trailFactor moves by how far ADX sits from its
    baseline and is clamped to 0.05..0.95; an unknown/NaN ADX counts as the
    baseline. With it off the trailing factor is used literally."""
    if not cfg.useDynamic:
        return cfg.trailFactor
    a = cfg.adxBase if adx_val is None or adx_val != adx_val else adx_val
    return max(0.05, min(0.95, cfg.trailFactor + ((cfg.adxBase - a) / 100.0)))


# ------------------------------------- the open position: backtest vs account --
# ONE shape for both sides so the Backtest and Live tabs show (and compare) the
# same fields. Prices marked "SOXL" are strategy prices (levels are always SOXL
# levels, even when a short is held as long SOXS); price / now / pnl are in the
# instrument actually held.
#   side, symbol, qty, opened (str), price (avg fill), now, pnl_pct, pnl_usd,
#   entry_ref, tp, target, trail (None = not armed yet), extreme, fixed, tracked
def backtest_open_view(ot):
    """SupertrendStrategy.open_trade (the position the backtest still holds after its
    last bar) -> the common shape. None when the backtest is flat."""
    if not ot:
        return None
    return {"side": ot["side"], "symbol": SYMBOL_LONG, "qty": int(ot["qty"]),
            "opened": str(ot.get("fill_time") or ot.get("entry_time")),
            "price": float(ot["price"]), "now": float(ot["last_close"]),
            "pnl_pct": float(ot["unrealized_pct"]), "pnl_usd": float(ot["unrealized_usd"]),
            "entry_ref": float(ot["entry_ref"]), "tp": float(ot["tp"]), "target": float(ot["target"]),
            "trail": ot.get("trail"), "extreme": ot.get("extreme"),
            "fixed": bool(ot["active_fixed"]), "tracked": True, "asof": str(ot.get("last_time"))}


def compare_open(live, expected, forming=False, tol=0.005):
    """Is the account holding what the backtest would hold?  -> {"status", "text", "diffs"}.

    status: "match" (same side and the same frozen levels within ``tol``) · "differs" ·
    "timing" (only a difference the forming bar can explain: the backtest decides at a
    bar's CLOSE, live acts at its open or inside it) · "untracked" (the account holds a
    position this app did not open, so no levels exist to compare).
    Quantity is NOT compared: it follows the equity each side compounds from."""
    def out(status, text, diffs=()):
        return {"status": status, "text": text, "diffs": list(diffs)}
    if live is None and expected is None:
        return out("match", "Both flat: the account holds nothing and the backtest holds nothing.")
    if live is None or expected is None or live["side"] != expected["side"]:
        if live is None:
            text = f"Backtest holds a {expected['side']} (entry {expected['entry_ref']:.2f}) but the account is flat."
        elif expected is None:
            text = f"The account holds a {live['side']} but the backtest is flat."
        else:
            text = f"The account holds a {live['side']} but the backtest holds a {expected['side']}."
        if forming:
            return out("timing", text + " The current 15-minute bar is still forming: a reversal, a fill or an exit inside it "
                                        "reaches the backtest when the bar closes.", [text])
        return out("differs", text, [text])
    if not live.get("tracked", True):
        return out("untracked", f"The account holds a {live['side']} that this app did not open, so there are no frozen levels to compare "
                                f"(the backtest's entry reference is {expected['entry_ref']:.2f}).")
    diffs = []
    for key, label in (("entry_ref", "entry reference"), ("target", "run-up target"), ("tp", "take-profit")):
        a, b = live.get(key), expected.get(key)
        if a is not None and b is not None and b and abs(a / b - 1.0) > tol:
            diffs.append(f"{label}: account {a:.2f} vs backtest {b:.2f}")
    la, ea = live.get("trail") is not None, expected.get("trail") is not None
    if la != ea:
        diffs.append("trailing stop: " + ("armed on the account, not in the backtest" if la else "armed in the backtest, not on the account"))
    elif la and abs(live["trail"] / expected["trail"] - 1.0) > tol:
        diffs.append(f"trailing stop: account {live['trail']:.2f} vs backtest {expected['trail']:.2f}")
    if diffs:
        return out("timing" if forming else "differs", "Same side, different levels - " + "; ".join(diffs), diffs)
    return out("match", f"Same {live['side']}, same entry reference {live['entry_ref']:.2f}, same take-profit {live['tp']:.2f}"
                        + (f", same trailing stop {live['trail']:.2f}." if la else ", trailing stop not armed on either."))


def creds(paper: bool):
    """Alpaca keys: the project's key file "Alpaca Key.txt" in the project folder, then ALPACA_API_KEY / ALPACA_SECRET_KEY, then ~/.claude.json (alpaca_keys.py)."""
    import alpaca_keys
    return alpaca_keys.creds(paper)


# =============================================================== engine =====
class Engine(threading.Thread):
    """Background strategy loop. All UI updates go through a Queue."""

    def __init__(self, ui_q: queue.Queue, paper: bool,
                 capital: float, equity_frac: float, start_date: str,
                 armed: bool):
        super().__init__(daemon=True)
        self.q = ui_q
        self.paper = paper
        self.capital = capital
        self.equity_frac = equity_frac
        self.start_date = start_date
        self.armed = armed
        # NB: do NOT name this _stop -- threading.Thread._stop() is an
        # internal method Python calls during join()/cleanup. Shadowing it
        # with an Event raises "'Event' object is not callable" and breaks
        # the thread lifecycle, which is what stopped the GUI working.
        self._halt = threading.Event()
        self._guard_day = None
        self._trades_today = 0
        self._day_start_equity = None
        self.report_open = True     # emit the account-vs-backtest open position every poll (the replay harness turns it off)
        self.tclient = None
        self.dclient = None

    # ---- helpers ----
    def emit(self, kind, **kw):
        self.q.put({"kind": kind, **kw})

    def log(self, msg, level="info"):
        rec = {"ts": datetime.now().strftime("%H:%M:%S"), "msg": msg, "level": level}
        self.emit("log", **rec)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                     "msg": msg, "level": level}) + "\n")
        except Exception:
            pass

    def market_open(self):
        try:
            c = self.tclient.get_clock()
            return bool(c.is_open), c
        except Exception:
            return False, None

    def guard(self, equity):
        """Returns (ok_to_trade, reason). Enforced every poll."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._guard_day != today:
            self._guard_day, self._trades_today = today, 0
            self._day_start_equity = equity
        if REQUIRE_MARKET_OPEN:
            is_open, clk = self.market_open()
            if not is_open:
                nxt = getattr(clk, "next_open", None)
                return False, f"market CLOSED (next open {nxt})"
        if self._trades_today >= MAX_TRADES_PER_DAY:
            return False, f"daily trade cap hit ({self._trades_today})"
        if self._day_start_equity:
            dd = (equity / self._day_start_equity - 1) * 100
            if dd <= DAILY_LOSS_LIMIT_PCT:
                return False, f"daily loss limit hit ({dd:.2f}%)"
        return True, "ok"

    def stop(self):
        self._halt.set()

    # ---- market ----
    def fetch_bars(self):
        days = max(14, int(WARMUP_BARS / 26 * 1.9) + 6)
        req = StockBarsRequest(
            symbol_or_symbols=[SYMBOL_LONG],
            timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
            start=datetime.now(timezone.utc) - timedelta(days=days),
            adjustment=Adjustment.SPLIT, feed=DataFeed.SIP)
        df = self.dclient.get_stock_bars(req).df.reset_index()
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        df = df[(ts.dt.time >= pd.Timestamp("09:30").time())
                & (ts.dt.time < pd.Timestamp("16:00").time())
                & (ts.dt.dayofweek < 5)].copy()
        df["bar"] = ts[df.index].dt.tz_localize(None)
        return (df.set_index("bar").sort_index()
                  [["open", "high", "low", "close"]].ffill().dropna().tail(WARMUP_BARS))

    def signal(self, bars):
        """Separate the SIGNAL bar from the MONITOR bar.

        During RTH Alpaca returns the currently-FORMING bar as the last row.
        Feeding that partial bar to Supertrend lets the direction REPAINT --
        it can flip mid-bar and flip back before the close, causing entries the
        backtest would never take. Entries must therefore use the last CLOSED
        bar, exactly as the backtest does.

        Exits are the opposite case: a bar's high/low are cumulative running
        extremes, so the forming bar is precisely what detects an intrabar
        touch of the take-profit or trailing level. That is what we want for
        monitoring, so mon_high / mon_low come from the latest row even when it
        is still forming.

          want / st / open  -> last CLOSED bar   (entries, level anchoring)
          adx / high_vol    -> last CLOSED bar   (ADX trail factor, Auto-Switch mode)
          mon_high/mon_low  -> latest bar        (exit trigger detection)

        adx is None unless the ADX dynamic trail is on, high_vol None unless the
        mode is Auto-Switch (see regime_inputs).
        """
        h = bars["high"].to_numpy(float)
        l = bars["low"].to_numpy(float)
        c = bars["close"].to_numpy(float)
        o = bars["open"].to_numpy(float)
        st, d = S.supertrend(h, l, c, CFG.atrPeriod, CFG.factor)

        last = len(c) - 1
        now_et = datetime.now(timezone.utc).astimezone(
            ZoneInfo("America/New_York")).replace(tzinfo=None)
        forming = (bars.index[last] + timedelta(minutes=BAR_MINUTES)) > now_et
        i = last - 1 if (forming and last >= 1) else last     # signal bar

        if i < 0 or np.isnan(st[i]) or np.isnan(d[i]):
            return None
        want = 1 if d[i] == -1 else -1
        adx, high_vol = regime_inputs(CFG, h, l, c, i)   # same bar as the entry signal
        return {"bar": bars.index[i], "close": float(c[i]), "st": float(st[i]),
                "open": float(o[i]), "high": float(h[i]), "low": float(l[i]),
                "want": want, "adx": adx, "high_vol": high_vol,
                "forming": bool(forming),
                "mon_bar": bars.index[last],
                "mon_high": float(h[last]), "mon_low": float(l[last]),
                "mon_open": float(o[last]), "mon_close": float(c[last]),
                "limit": float(st[i] * (1 + CFG.buyback)) if want > 0
                         else float(st[i] * (1 - CFG.buyback))}

    def resolve_short_route(self, force_soxs=False):
        """Prefer a direct SOXL short; fall back to buying SOXS.

        Refined gate: `shortable` is the availability flag and is what decides
        the route. `easy_to_borrow` is a COST signal (HTB names carry a borrow
        fee) and is reported for information only -- a hard-to-borrow name is
        still shortable. Requiring easy_to_borrow was too strict.

        Broker flags can also be stale or wrong, so route selection is only a
        best guess: send() catches a rejected short and re-routes to SOXS at
        execution time (see short_rejected handling).
        """
        if force_soxs or not PREFER_SHORT_SOXL:
            return SYMBOL_SHORT, OrderSide.BUY, "SOXS (forced)"
        try:
            acct = self.tclient.get_account()
            a = self.tclient.get_asset(SYMBOL_LONG)
            if not acct.shorting_enabled:
                return SYMBOL_SHORT, OrderSide.BUY, "account shorting disabled -> SOXS"
            if a.shortable:
                fee = "" if a.easy_to_borrow else " (HTB - borrow fee applies)"
                return SYMBOL_LONG, OrderSide.SELL, f"short SOXL{fee}"
            why = f"broker reports SOXL not shortable -> SOXS"
        except Exception as exc:
            why = f"borrow check failed ({str(exc)[:50]}) -> SOXS"
        return SYMBOL_SHORT, OrderSide.BUY, why

    def session_fills(self):
        """Filled orders since start_date, oldest first."""
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            orders = self.tclient.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.CLOSED, after=start, limit=500))
        except Exception:
            return []
        fills = [o for o in orders
                 if o.filled_qty and float(o.filled_qty) > 0 and o.filled_at]
        return sorted(fills, key=lambda o: o.filled_at)

    def replay_session(self):
        """Replay fills from start_date to get closed trades + running equity.

        Average-cost accounting per symbol. Realised P&L compounds onto the
        user's stated initial capital, and that running figure -- not the raw
        account balance -- is what sizes the next order.
        """
        book, realised, trades = {}, 0.0, []
        for o in self.session_fills():
            sym = o.symbol
            qty = float(o.filled_qty)
            px = float(o.filled_avg_price or 0)
            signed = qty if o.side == OrderSide.BUY else -qty
            cq, ca = book.get(sym, (0.0, 0.0))
            if cq == 0 or (cq > 0) == (signed > 0):
                nq = cq + signed
                book[sym] = (nq, (ca * abs(cq) + px * abs(signed)) / abs(nq)) \
                    if nq else (0.0, 0.0)
            else:
                close_qty = min(abs(signed), abs(cq))
                pnl = (px - ca) * close_qty * (1 if cq > 0 else -1)
                realised += pnl
                trades.append({"time": o.filled_at, "symbol": sym,
                               "side": "LONG" if cq > 0 else "SHORT",
                               "qty": close_qty, "entry": ca, "exit": px,
                               "pnl": pnl,
                               "pct": (px / ca - 1) * 100 * (1 if cq > 0 else -1)
                                      if ca else 0.0})
                rem = cq + signed
                book[sym] = (rem, px if rem and (rem > 0) != (cq > 0) else ca)
        return trades, realised, book

    def fetch_bars_since(self, start: datetime):
        """Bars covering start..now PLUS enough history to warm up Supertrend."""
        span = max((datetime.now() - start).days, 1)
        days = span + 90                      # ~500 bars warmup ≈ 20 trading days
        req = StockBarsRequest(
            symbol_or_symbols=[SYMBOL_LONG],
            timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
            start=datetime.now(timezone.utc) - timedelta(days=days),
            adjustment=Adjustment.SPLIT, feed=DataFeed.SIP)
        df = self.dclient.get_stock_bars(req).df.reset_index()
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
        df = df[(ts.dt.time >= pd.Timestamp("09:30").time())
                & (ts.dt.time < pd.Timestamp("16:00").time())
                & (ts.dt.dayofweek < 5)].copy()
        df["bar"] = ts[df.index].dt.tz_localize(None)
        return (df.set_index("bar").sort_index()
                  [["open", "high", "low", "close"]].ffill().dropna())

    def simulate_session(self):
        """Replay the STRATEGY on real bars from start_date to now.

        The paper account has no fill history for the chosen window, so the
        running equity cannot be read from the broker. Instead we run the same
        backtest engine over real market data from start_date forward, seeded
        with the user's initial capital. The resulting equity is what the next
        live order is sized from -- so live sizing picks up exactly where the
        simulated session left off.

        Warmup bars BEFORE start_date are included so Supertrend is valid, but
        only trades on/after start_date count toward the session.
        """
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        bars = self.fetch_bars_since(start)
        # Every strategy input (risk/limit factor, ADX, volatility regime,
        # pessimistic ordering, commission) is the user's CFG; only the
        # session's own sizing is set here. Touch fills stay OFF and the size
        # cap off: this stands in for real broker fills.
        cfg = replace(CFG, useDateFilter=False,
                      initial_capital=self.capital,
                      equity_frac=self.equity_frac,
                      short_mult=SHORT_EQUITY_MULT,
                      touch_fill=False, capacity_usd=1e18,
                      strict_entries=True)          # stands in for what live orders could really have done
        eq, tr = S.SupertrendStrategy(cfg).run(bars)
        eq_s = eq[eq.index >= start]
        if len(eq_s) < 2:
            return [], self.capital, 0.0, bars
        ret = float(eq_s.iloc[-1] / eq_s.iloc[0])
        equity = self.capital * ret
        rows = []
        if len(tr):
            sub = tr[tr.exit_time >= start]
            run = self.capital
            for _, r in sub.iterrows():
                mult = SHORT_EQUITY_MULT if r["side"] == "short" else 1.0
                pnl = run * self.equity_frac * mult * (r["return_pct"] / 100.0)
                run += pnl
                rows.append({"time": r["exit_time"], "symbol": SYMBOL_LONG
                             if r["side"] == "long" else SYMBOL_SHORT,
                             "side": r["side"].upper(), "qty": r["qty"],
                             "entry": r["price"], "exit": r["exit_price"],
                             "pnl": pnl, "pct": r["return_pct"],
                             "equity": run})
            if rows:
                equity = rows[-1]["equity"]
        return rows, equity, (equity / self.capital - 1) * 100, bars

    def session_equity(self, account_equity: float, use_cache: bool = False) -> dict:
        """Running equity for sizing.

        Uses REAL broker fills when the account has them; otherwise falls back
        to simulating the strategy over the same window so the session still
        has a defensible compounding base.
        """
        if use_cache and getattr(self, "_sess_cache", None) is not None:
            return self._sess_cache
        real, realised, _ = self.replay_session()
        unreal = 0.0
        try:
            for _, pos in self.positions().items():
                unreal += pos["upl"]
        except Exception:
            pass

        if real:
            eq = max(self.capital + realised + unreal, 0.0)
            self._sess_cache = {"baseline": self.capital, "realised": realised,
                    "unrealised": unreal, "equity": eq, "source": "broker fills",
                    "n_trades": len(real), "trades": real}
            return self._sess_cache

        sim, sim_eq, sim_ret, _ = self.simulate_session()
        self._sess_cache = {"baseline": self.capital, "realised": sim_eq - self.capital,
                "unrealised": 0.0, "equity": max(sim_eq, 0.0),
                "source": f"SIMULATED ({sim_ret:+.2f}% since {self.start_date})",
                "n_trades": len(sim), "trades": sim}
        return self._sess_cache

    def next_trade_plan(self, sig, sess, pos) -> dict:
        """What the next order WOULD be, at current prices. Display only."""
        want = sig["want"]
        if want > 0:
            tgt, side, route = SYMBOL_LONG, OrderSide.BUY, "long SOXL"
        else:
            tgt, side, route = self.resolve_short_route()
        held = {k: v for k, v in pos.items() if abs(v["qty"]) > 1e-9}
        want_short_leg = (side == OrderSide.SELL)
        cur = held.get(tgt)
        in_place = cur is not None and (
            (cur["qty"] < 0) if want_short_leg else (cur["qty"] > 0))

        trigger = sig["limit"]
        native = (tgt == SYMBOL_LONG)
        if native:
            px = trigger
            when = "limit @ %.2f" % trigger
        else:
            px = self.last_price(tgt) or 0.0
            now = sig.get("px") or sig.get("mon_close") or sig["close"]
            reached = (now <= trigger) if want > 0 else (now >= trigger)
            when = ("SOXL %.2f is at/better than %.2f - market now" % (now, trigger) if reached
                    else "waiting: SOXL %.2f, needs %s %.2f" % (now, "a dip to" if want > 0 else "a rally to", trigger))
        mult = 1.0 if want > 0 else SHORT_EQUITY_MULT
        budget = sess["equity"] * self.equity_frac * mult
        qty = int(math.floor(budget / px)) if px > 0 else 0
        return {"symbol": tgt, "side": side.value, "route": route,
                "mult": mult,
                "price": px, "qty": qty, "notional": qty * px,
                "budget": budget, "when": when, "in_place": in_place,
                "frac": self.equity_frac}

    def last_price(self, symbol):
        """Ask/bid midpoint for the instrument we are actually going to trade."""
        q = self.dclient.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[symbol]))[symbol]
        a, b = float(q.ask_price or 0), float(q.bid_price or 0)
        if a > 0 and b > 0:
            return (a + b) / 2
        return a or b

    def open_orders(self, symbol=None):
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return [o for o in self.tclient.get_orders(req)
                    if symbol is None or o.symbol == symbol]
        except Exception:
            return []

    def positions(self):
        out = {}
        for p in self.tclient.get_all_positions():
            out[p.symbol] = {"qty": float(p.qty), "side": p.side.value,
                             "avg": float(p.avg_entry_price),
                             "mv": float(p.market_value),
                             "upl": float(p.unrealized_pl),
                             "uplpc": float(p.unrealized_plpc) * 100,
                             "cur": float(p.current_price)}
        return out

    # ---- order ----
    def send(self, symbol, qty, side, why, limit_price=None, reroute=False):
        """Send (or, in a dry run, only show) one order. Returns the broker's order id, or None.
        ``reroute``: this SELL OPENS a short, so if the broker rejects it fall back to buying SOXS. Never set for a closing sell: a rejected exit must not buy SOXS."""
        payload = {"ticker": symbol, "action": side.value, "quantity": int(qty)}
        if limit_price is not None:
            payload["type"] = "limit"
            payload["limit_price"] = round(float(limit_price), 2)
        else:
            payload["type"] = "market"
        if not self.armed:
            self.log(f"DRY RUN — would send {json.dumps(payload)}", "warn")
            self.emit("trade_popup", payload=payload, why=why, dry=True,
                      mode="PAPER" if self.paper else "LIVE")
            return
        if limit_price is None:
            req = MarketOrderRequest(symbol=symbol, qty=int(qty), side=side,
                                     time_in_force=TimeInForce.DAY)
        else:
            req = LimitOrderRequest(symbol=symbol, qty=int(qty), side=side,
                                    limit_price=round(float(limit_price), 2),
                                    time_in_force=TimeInForce.DAY)
        try:
            self._trades_today += 1
            order = self.tclient.submit_order(req)
        except Exception as exc:
            msg = str(exc)
            if reroute and side == OrderSide.SELL and symbol == SYMBOL_LONG:
                self.log(f"SOXL short REJECTED ({msg[:70]}) — re-routing to SOXS",
                         "warn")
                self._rerouted = True
                px2 = self.last_price(SYMBOL_SHORT)
                if px2 and px2 > 0:
                    q2 = int(math.floor(qty * (limit_price or px2) / px2)) \
                         if limit_price else int(math.floor(
                             (qty * self.last_price(SYMBOL_LONG)) / px2))
                    return self.send(SYMBOL_SHORT, q2, OrderSide.BUY,
                                     why=why + " [fallback after SOXL reject]")
            self.log(f"order failed: {msg[:120]}", "err")
            return
        payload["order_id"] = str(order.id)
        payload["status"] = str(order.status)
        self.log(f"ORDER SENT {json.dumps(payload)}", "trade")
        self.emit("trade_popup", payload=payload, why=why, dry=False,
                  mode="PAPER" if self.paper else "LIVE")
        return str(order.id)

    # ---- main loop ----
    def run(self):
        try:
            k, s = creds(self.paper)
            self.tclient = TradingClient(k, s, paper=self.paper)
            self.dclient = StockHistoricalDataClient(k, s)
            base = str(self.tclient._base_url)
            if self.paper and "PAPER" not in base.upper():
                self.log("REFUSING: paper selected but endpoint is not paper", "err")
                self.emit("stopped"); return
            acct = self.tclient.get_account()
            self.log(f"connected {base} acct {acct.account_number[:4]}**** "
                     f"equity ${float(acct.equity):,.2f}", "ok")
            self.emit("account", equity=float(acct.equity),
                      buying_power=float(acct.buying_power),
                      number=acct.account_number[:4] + "****")
        except Exception as exc:
            self.log(f"connect failed: {exc}", "err")
            self.emit("stopped"); return

        while not self._halt.is_set():
            try:
                if os.path.exists(KILL_FILE):
                    self.log("KILL FILE present — halting", "err")
                    break
                self.poll_once()
            except Exception as exc:
                self.log(f"loop error: {exc}", "err")
                traceback.print_exc()
            self._halt.wait(POLL_SECONDS)
        self.emit("stopped")

    def poll_once(self):
        """One pass of the strategy loop: what run() repeats every POLL_SECONDS.
        Split out of run() unchanged so the replay harness (supertrend_ui/replay.py)
        drives exactly the code the live loop runs."""
        acct = self.tclient.get_account()
        equity = float(acct.equity)
        pos = self.positions()
        self.emit("account", equity=equity,
                  buying_power=float(acct.buying_power),
                  number=acct.account_number[:4] + "****")
        self._sess_cache = None
        sess = self.session_equity(equity)
        self.emit("session",
                  **{k2: v for k2, v in sess.items() if k2 != "trades"},
                  frac=self.equity_frac, start=self.start_date)
        self.emit("trades", trades=sess["trades"])
        self.emit("positions", positions=pos,
                  baseline=self.capital, start=self.start_date)

        bars = self.fetch_bars()
        sig = self.signal(bars)
        if not sig:
            self.log(f"warming up ({len(bars)} bars, need {CFG.atrPeriod})")
        else:
            try:                                  # the CURRENT SOXL price: entries compare it with the re-entry level
                sig["px"] = self.last_price(SYMBOL_LONG) or sig["mon_close"]
            except Exception:
                sig["px"] = sig["mon_close"]
            self.emit("signal", **{k2: (str(v) if k2 == "bar" else v)
                                   for k2, v in sig.items()})
            try:
                self.emit("plan", **self.next_trade_plan(sig, sess, pos))
            except Exception as exc:
                self.log("plan calc failed: %s" % exc, "warn")
            ok, why = self.guard(equity)
            self.emit("guard", ok=ok, reason=why,
                      trades_today=self._trades_today,
                      cap=MAX_TRADES_PER_DAY)
            if not ok:
                self.log(f"entries paused: {why}", "warn")
            self.adopt_pending(sig, pos, bars)    # a resting limit that filled: freeze its exit levels
            if self.manage_position(sig, pos, bars):
                self.log("position closed by exit rule — "
                         "no new entry this bar")
            elif ok:
                self.decide(sig, pos, equity)
            if self.report_open:
                self.report_open_position(sig, pos, bars)

    # ------------------------------------------------------------------
    # THE OPEN POSITION: what the account holds vs what the backtest holds
    # ------------------------------------------------------------------
    def live_open(self, sig, pos):
        """The position the account holds, in the common shape (see backtest_open_view),
        with the exit levels this app froze for it. None when the account is flat."""
        held = {k: v for k, v in pos.items() if abs(v["qty"]) > 1e-9}
        if not held:
            return None
        sym = max(held, key=lambda k: abs(held[k].get("mv", 0.0)))
        p = held[sym]
        side = strategy_side(sym, p["side"])
        st = self.trade_state_load()
        tracked = bool(st.get("pos")) and st.get("symbol") == sym
        v = {"side": side, "symbol": sym, "qty": int(abs(p["qty"])), "opened": str(st.get("opened")) if tracked else None,
             "price": p["avg"], "now": p["cur"], "pnl_pct": p["uplpc"], "pnl_usd": p["upl"],
             "entry_ref": None, "tp": None, "target": None, "trail": None, "extreme": None,
             "fixed": None, "tracked": tracked, "asof": str(sig.get("mon_bar")), "others": len(held) - 1}
        if tracked:
            adx = sig.get("adx")
            v.update(entry_ref=st["entry_ref"], tp=self.tp_level(st), target=st["target"], trail=self.trail_level(st, adx),
                     extreme=st.get("max_high") if side == "long" else st.get("min_low"), fixed=bool(st["active_fixed"]))
        return v

    def expected_open(self, bars, sig):
        """The position the STRICT backtest (strict_entries: only what a live order can
        know) holds after the last CLOSED bar, simulated from the session start date
        over the same bars this poll loaded. None when it is flat."""
        closed = bars.iloc[:-1] if sig.get("forming") else bars
        cfg = replace(CFG, useDateFilter=True, startDate=self.start_date, endDate="2099-12-31",
                      initial_capital=self.capital, equity_frac=self.equity_frac, short_mult=SHORT_EQUITY_MULT,
                      touch_fill=False, capacity_usd=1e18, strict_entries=True)
        strat = S.SupertrendStrategy(cfg)
        strat.run(closed)
        return backtest_open_view(strat.open_trade)

    def report_open_position(self, sig, pos, bars):
        """Emit ``openpos``: the account's position, the backtest's, and whether they match.
        Display only - it never touches an order and a failure here never stops the loop."""
        try:
            live = self.live_open(sig, pos)
            exp = self.expected_open(bars, sig)
            cmp_ = compare_open(live, exp, forming=bool(sig.get("forming")))
            pend = self.trade_state_load().get("pending")
            self.emit("openpos", live=live, expected=exp, cmp=cmp_, pending=pend, want=sig["want"], limit=sig["limit"],
                      asof=str(sig["bar"]), forming=bool(sig.get("forming")))
        except Exception as exc:
            self.log(f"open-position check failed: {exc}", "warn")

    # ------------------------------------------------------------------
    # POSITION MANAGEMENT — trailing stop + limit take-profit
    # ------------------------------------------------------------------
    # These are the exits that produce most of the backtested edge: median
    # winner +9.99% vs holding to a Supertrend flip. Without them the live
    # engine runs a materially different (and weaker) strategy.
    #
    # IMPORTANT: every level is a SOXL price. When shorts are routed to SOXS
    # the position is HELD in SOXS but the exit decision is made by watching
    # SOXL, then the SOXS position is closed at market. A SOXL-priced stop
    # cannot be attached to a SOXS position.
    #
    # Polling caveat: levels are evaluated on completed 15-min bars plus the
    # live quote. Price can pierce a level and retrace between polls, so fills
    # will differ from the backtest's intrabar assumption. Accepted trade-off
    # of cross-instrument routing.
    def trade_state_load(self):
        try:
            if os.path.exists(STATE_FILE):
                return json.load(open(STATE_FILE))
        except Exception:
            pass
        return {}

    def trade_state_save(self, st):
        try:
            json.dump(st, open(STATE_FILE, "w"), indent=2, default=str)
        except Exception:
            pass

    # ---- bar bookkeeping shared by entries and exits ----
    def cur_bar(self, sig):
        """Start of the 15-minute bar 'now' falls in: the forming bar, or the next
        bar when its first print has not arrived yet."""
        return sig["mon_bar"] if sig.get("forming") else sig["bar"] + timedelta(minutes=BAR_MINUTES)

    def fill_ref(self, sig):
        """SOXL open of the bar a MARKET fill happens on: the backtest's entry
        reference (Pine: entryPrice := open of the bar the fill occurred on).
        Right after a bar closes that is the forming bar's open; if no print of
        the new bar has arrived yet, the live quote stands in for it."""
        if sig.get("forming") and sig.get("mon_open"):
            return sig["mon_open"]
        try:
            return self.last_price(SYMBOL_LONG) or sig["close"]
        except Exception:
            return sig["close"]

    def _fold_from(self, sig, reentry_bar=None):
        """First bar whose range counts for a position's exits and trailing extreme.
        A market entry at a bar's open (reversal) counts from that very bar; an
        intrabar fill (resting limit / SOXS trigger) from the NEXT bar, exactly as
        the backtest manages exits from the bar after the fill."""
        try:
            if reentry_bar is not None:
                return str(pd.Timestamp(reentry_bar) + timedelta(minutes=BAR_MINUTES))
            return str(self.cur_bar(sig))
        except Exception:
            return None

    def record_entry(self, sig, symbol, side, qty, ref=None, reentry_bar=None):
        """Freeze the Pine reference levels at entry.

        ref: SOXL open of the bar the fill happened on (default: the signal bar's
        open, the old behaviour); reentry_bar: start of the bar an INTRABAR
        re-entry filled in (None for a market entry at a bar's open)."""
        c = CFG
        ref = sig["open"] if ref is None else ref   # Pine: entryPrice := open
        st_e = sig["st"]                       # Pine: supert := st
        risk = abs(ref - st_e) / ref * c.riskfactor if ref else 0.0
        # backtest: Always Fixed, or Auto-Switch while the volatility regime is high
        fixed = (c.strategyMode == "Always Fixed"
                 or (c.strategyMode == "Auto-Switch via Vol" and bool(sig.get("high_vol"))))
        move = c.riskReward if fixed else risk
        target = ref * (1 + move) if side == "long" else ref * (1 - move)
        instr_entry = ref if symbol == SYMBOL_LONG else (self.last_price(symbol) or ref)
        state = {"pos": 1 if side == "long" else -1, "side": side,
                 "symbol": symbol, "qty": int(qty),
                 "instr_entry": instr_entry, "exit_order_ids": [],
                 "resting_tp": None, "resting_stop": None,
                 "entry_ref": ref, "supert_entry": st_e, "target": target,
                 "active_fixed": fixed, "max_high": None, "min_low": None,
                 "opened": str(sig["bar"]),
                 "fold_from": self._fold_from(sig, reentry_bar), "ext_bar": None}
        self.log(f"levels set: ref {ref:.2f} target {target:.2f} "
                 f"TP {self.tp_level(state):.2f} "
                 f"[{'fixed' if fixed else 'dynamic'} mode]", "ok")
        self.place_exit_orders(state, sig.get("adx"))   # rest the TP immediately
        self.trade_state_save(state)
        return state

    def trail_factor(self, adx_val=None):
        """ADX of the last closed bar -> trailing factor (see dynamic_trail_factor)."""
        return dynamic_trail_factor(CFG, adx_val)

    def tp_level(self, state):
        c = CFG
        ref, se = state["entry_ref"], state["supert_entry"]
        if state["side"] == "long":
            return (ref * (1 + c.price_perc) if state["active_fixed"]
                    else ref * (1 + (ref - se) / ref * c.limitfactor))
        return (ref * (1 - c.price_perc) if state["active_fixed"]
                else ref * (1 - (se - ref) / ref * c.limitfactor))

    def trail_level(self, state, adx=None):
        """None until the run-up target has been reached (Pine arms it then).

        adx: ADX of the last closed bar (sig["adx"]); only read with the ADX
        dynamic trail on. Without it the baseline is used, as before."""
        tf = self.trail_factor(adx)
        if state["side"] == "long":
            mx = state.get("max_high")
            if mx is None or mx < state["target"]:
                return None
            return state["entry_ref"] + (mx - state["entry_ref"]) * tf
        mn = state.get("min_low")
        if mn is None or mn > state["target"]:
            return None
        return state["entry_ref"] - (state["entry_ref"] - mn) * tf

    # ------------------------------------------------------------------
    # BROKER-SIDE EXITS  (resting limit TP + stop, OCO)
    # ------------------------------------------------------------------
    # Replaces the polled/synthetic exits. These orders live AT THE BROKER, so
    # they still fire if this app crashes, the machine sleeps, or the network
    # drops -- which the polling approach could not survive.
    #
    # EXIT POLICY BY INSTRUMENT:
    #
    #   LONG  (SOXL)  -> RESTING orders at the broker. Levels are native SOXL
    #                    prices, so a limit TP and a stop can be priced exactly.
    #                    Survives app/machine failure.
    #
    #   SHORT (SOXS)  -> POLLED detection on SOXL, then a MARKET order to close
    #                    SOXS. Deliberate: SOXL and SOXS are separately
    #                    rebalanced 3x funds, not exact inverses, so a mirrored
    #                    SOXS level drifts from its SOXL trigger. An imprecise
    #                    resting order is worse than an exact polled one.
    #                    TRADE-OFF: no broker-side protection on SOXS -- if this
    #                    app stops, a SOXS position is unmanaged until restart.
    def _mirror(self, soxl_level, state):
        """Translate a SOXL price level onto the instrument actually held."""
        if state["symbol"] == SYMBOL_LONG:
            return soxl_level
        move = soxl_level / state["entry_ref"] - 1.0
        return max(state["instr_entry"] * (1 - move), 0.01)

    def cancel_exit_orders(self, state):
        for oid in state.get("exit_order_ids", []):
            try:
                self.tclient.cancel_order_by_id(oid)
            except Exception:
                pass
        state["exit_order_ids"] = []

    def rests_at_broker(self, state):
        """Only SOXL positions get resting orders (see EXIT POLICY above)."""
        return state.get("symbol") == SYMBOL_LONG

    def place_exit_orders(self, state, adx=None):
        """Rest a take-profit limit, and a stop once the trail has armed.

        Uses an OCO pair when both legs exist so a fill on one cancels the
        other. Before the trail arms (price has not reached the run-up target)
        only the TP limit rests -- matching Pine, where strategy.exit is called
        with stop = na until maxHigh >= targetPrice.
        """
        if not self.rests_at_broker(state):
            tp = self.tp_level(state)
            tr = self.trail_level(state, adx)
            self.log(f"{state['symbol']}: exits are POLLED -> MARKET "
                     f"(SOXL TP {tp:.2f}"
                     + (f", stop {tr:.2f})" if tr else ", stop not armed)"),
                     "warn")
            state["exit_order_ids"] = []
            state["resting_tp"] = state["resting_stop"] = None
            return
        if not self.armed:
            tp_s = self.tp_level(state)
            tr = self.trail_level(state, adx)
            self.log(f"DRY RUN — would rest TP {tp_s:.2f}" +
                     (f" + STOP {tr:.2f}" if tr else " (stop not armed yet)"),
                     "warn")
            return
        self.cancel_exit_orders(state)
        qty = int(abs(state["qty"]))
        if qty <= 0:
            return
        sym = state["symbol"]
        # SOXL long and SOXS long (SOXS stands in for a SOXL short) are closed by
        # SELLING. A DIRECT SOXL short is closed by BUYING: a SELL take-profit
        # below the market would be marketable and would ADD to the short.
        exit_side = (OrderSide.BUY if (sym == SYMBOL_LONG and state["side"] == "short")
                     else OrderSide.SELL)
        tp_price = round(self._mirror(self.tp_level(state), state), 2)
        trail = self.trail_level(state, adx)
        ids = []
        try:
            if trail is None:
                o = self.tclient.submit_order(LimitOrderRequest(
                    symbol=sym, qty=qty, side=exit_side,
                    limit_price=tp_price, time_in_force=TimeInForce.GTC))
                ids.append(str(o.id))
                self.log(f"RESTING take-profit: {exit_side.value.upper()} {qty} {sym} @ {tp_price}",
                         "trade")
            else:
                stop_price = round(self._mirror(trail, state), 2)
                o = self.tclient.submit_order(LimitOrderRequest(
                    symbol=sym, qty=qty, side=exit_side,
                    limit_price=tp_price, time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.OCO,
                    take_profit=TakeProfitRequest(limit_price=tp_price),
                    stop_loss=StopLossRequest(stop_price=stop_price)))
                ids.append(str(o.id))
                for leg in (getattr(o, "legs", None) or []):
                    ids.append(str(leg.id))
                self.log(f"RESTING OCO: {exit_side.value.upper()} {qty} {sym} "
                         f"TP {tp_price} / STOP {stop_price}", "trade")
        except Exception as exc:
            self.log(f"could not rest exit orders: {str(exc)[:110]} "
                     f"— falling back to polled exits", "err")
        state["exit_order_ids"] = ids
        state["resting_tp"] = tp_price
        state["resting_stop"] = (round(self._mirror(trail, state), 2)
                                 if trail is not None else None)

    def sync_exit_orders(self, state, adx=None):
        """Re-rest the bracket when the trail moves the stop (new extreme, or --
        with the ADX trail on -- a new ADX).

        No-op for SOXS: those exits are polled and closed at market, so there
        is nothing resting to re-price.
        """
        if not self.rests_at_broker(state):
            return
        trail = self.trail_level(state, adx)
        want_stop = round(self._mirror(trail, state), 2) if trail is not None else None
        have_stop = state.get("resting_stop")
        if want_stop is None:
            return
        if have_stop is None or abs(want_stop - have_stop) >= 0.01:
            self.log(f"trail ratcheted -> re-resting stop "
                     f"{have_stop} -> {want_stop}")
            self.place_exit_orders(state, adx)

    # ---- limit entries: exit levels are frozen when the fill is seen ----
    def adopt_pending(self, sig, positions, bars=None):
        """A resting buyback LIMIT has no exit levels until it fills (its fill time
        and price are not known when it is placed). decide() saves it as 'pending';
        when the position shows up here, freeze the Pine levels exactly as the
        backtest does: reference = the OPEN of the bar the fill happened on, exits
        managed from the bar AFTER it. Returns True when a fill was adopted."""
        st = self.trade_state_load()
        pend = st.get("pending")
        if not pend or st.get("pos"):
            return False
        held = positions.get(pend["symbol"])
        want_long = pend["side"] == "long"
        if held is not None and abs(held["qty"]) > 1e-9 and ((held["qty"] > 0) == want_long):
            ref, fill_bar = self._fill_bar_open(pend, sig, bars)
            self.log(f"limit filled: {pend['symbol']} {pend['side']} {abs(held['qty']):,.0f} "
                     f"(fill bar {fill_bar}, its open {ref:.2f}) - freezing exit levels", "ok")
            self.record_entry(sig, pend["symbol"], pend["side"], abs(held["qty"]),
                              ref=ref, reentry_bar=fill_bar)
            return True
        if not any(str(o.id) == pend["order_id"] for o in self.open_orders(pend["symbol"])):
            self.trade_state_save({})          # cancelled / expired without a fill
        return False

    def _fill_bar_open(self, pend, sig, bars):
        """(open of the bar the pending limit filled in, that bar's start)."""
        try:
            fa = self.tclient.get_order_by_id(pend["order_id"]).filled_at
            if fa is not None and bars is not None:
                t = pd.Timestamp(fa)
                t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("America/New_York").tz_localize(None)
                k = int(bars.index.searchsorted(t, side="right")) - 1
                if k >= 0:
                    return float(bars["open"].iloc[k]), bars.index[k]
        except Exception:
            pass
        if sig.get("forming"):
            return sig["mon_open"], sig["mon_bar"]
        return sig["open"], sig["bar"]

    def _note_exit(self, sig, state=None, bars=None):
        """Remember the bar an exit happened on. The backtest never re-enters on
        the bar it exited on (the same price move would be counted twice), so
        decide() waits for the next bar."""
        bar = None
        try:
            for oid in (state or {}).get("exit_order_ids", []):
                fa = getattr(self.tclient.get_order_by_id(oid), "filled_at", None)
                if fa is not None:
                    t = pd.Timestamp(fa)
                    t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert("America/New_York").tz_localize(None)
                    bar = t.floor(f"{BAR_MINUTES}min")
                    break
        except Exception:
            bar = None
        self._exit_bar = bar if bar is not None else self.cur_bar(sig)

    def _scan_exits(self, state, sig, bars, tp):
        """Walk the bars since entry the way the backtest does. Each bar is tested
        against the take-profit and the trail AS THEY STOOD BEFORE IT (the trailing
        extreme only ever holds bars that already CLOSED) and, if it did not end the
        trade, folded into the extreme. The forming bar is tested too, so an
        intrabar touch counts at once, but it is never folded. A bar that closed
        between two polls is still tested, so no exit is missed because of polling.
        Returns (hit text or None, low, high of the last bar tested)."""
        long_ = state["side"] == "long"
        adx = sig.get("adx")
        fold_from = pd.Timestamp(state["fold_from"])
        done = pd.Timestamp(state["ext_bar"]) if state.get("ext_bar") else None

        def test(hi, lo):
            trail = self.trail_level(state, adx)
            if long_:
                tp_ok, stop_ok = hi >= tp, (trail is not None and lo <= trail)
            else:
                tp_ok, stop_ok = lo <= tp, (trail is not None and hi >= trail)
            if CFG.pessimistic and stop_ok:
                return f"trailing stop @ {trail:.2f}"
            if tp_ok:
                return f"limit take-profit @ {tp:.2f}"
            if stop_ok:
                return f"trailing stop @ {trail:.2f}"
            return None

        lo = hi = None
        seg = bars[(bars.index >= fold_from) & (bars.index <= sig["bar"])]
        for ts, row in seg.iterrows():
            if done is not None and ts <= done:
                continue
            lo, hi = float(row["low"]), float(row["high"])
            hit = test(hi, lo)
            if hit:
                return hit, lo, hi
            if long_:
                state["max_high"] = hi if state.get("max_high") is None else max(state["max_high"], hi)
            else:
                state["min_low"] = lo if state.get("min_low") is None else min(state["min_low"], lo)
            state["ext_bar"] = str(ts)
        if sig.get("forming") and sig["mon_bar"] >= fold_from:
            lo, hi = sig["mon_low"], sig["mon_high"]
            hit = test(hi, lo)
            if hit:
                return hit, lo, hi
        if lo is None:
            lo, hi = sig["mon_low"], sig["mon_high"]
        return None, lo, hi

    def manage_position(self, sig, positions, bars=None):
        """Test TP / trail and keep the trailing extreme. Returns True if an exit fired.

        With ``bars`` (the live loop passes them) the bars since entry are walked
        exactly like the backtest (see _scan_exits). Without them the monitor bar
        alone is tested, as before."""
        state = self.trade_state_load()
        if not state or not state.get("pos"):
            return False
        held = {k: v for k, v in positions.items() if abs(v["qty"]) > 1e-9}
        if not held:
            self._note_exit(sig, state, bars)    # a resting TP / stop filled at the broker
            self.trade_state_save({})
            return False

        # SEQUENCING (same defect fixed in the backtest): the trailing level
        # must be derived from the running extreme AS OF THE PRIOR BAR, then
        # tested against THIS bar's range. Updating the extreme with this bar's
        # high and then testing this bar's low against the resulting level
        # fires on any wide-range bar and truncates every winner.
        adx = sig.get("adx")       # last CLOSED bar, as the backtest's adx[i]
        tp = self.tp_level(state)
        if bars is not None and state.get("fold_from"):
            hit, ml, mh = self._scan_exits(state, sig, bars, tp)
            trail = self.trail_level(state, adx)
            if not hit:
                self.sync_exit_orders(state, adx)   # ratchet the resting stop
        else:
            trail = self.trail_level(state, adx)     # from PRIOR extreme
            # trigger off the MONITOR bar (may be the forming bar) so an intrabar
            # touch is caught within one poll rather than waiting for bar close
            mh, ml = sig.get("mon_high", sig["high"]), sig.get("mon_low", sig["low"])
            hit = None
            if state["side"] == "long":
                if mh >= tp:
                    hit = f"limit take-profit @ {tp:.2f}"
                elif trail is not None and ml <= trail:
                    hit = f"trailing stop @ {trail:.2f}"
            else:
                if ml <= tp:
                    hit = f"limit take-profit @ {tp:.2f}"
                elif trail is not None and mh >= trail:
                    hit = f"trailing stop @ {trail:.2f}"

            # only extend the extreme when this bar did NOT close the trade
            if not hit:
                self.sync_exit_orders(state, adx)   # ratchet the resting stop
                if state["side"] == "long":
                    state["max_high"] = mh if state.get("max_high") is None \
                        else max(state["max_high"], mh)
                else:
                    state["min_low"] = ml if state.get("min_low") is None \
                        else min(state["min_low"], ml)
            trail = self.trail_level(state, adx) if not hit else trail  # refresh for display only

        self.emit("levels", side=state["side"], entry_ref=state["entry_ref"],
                  target=state["target"], tp=tp, trail=trail,
                  extreme=state.get("max_high") if state["side"] == "long"
                  else state.get("min_low"), armed=trail is not None)

        if not hit:
            self.trade_state_save(state)
            return False

        mode = ("polled backstop" if self.rests_at_broker(state)
                else "polled -> MARKET (SOXS policy)")
        self.log(f"EXIT {hit} (monitor bar {sig.get('mon_bar')} "
                 f"{ml:.2f}-{mh:.2f}) — {mode}", "trade")
        self.cancel_exit_orders(state)
        # a resting exit may have filled while we decided: only close what is STILL held
        try:
            held = {k: v for k, v in self.positions().items() if abs(v["qty"]) > 1e-9}
        except Exception:
            pass
        for sym, pdata in held.items():
            side = OrderSide.BUY if pdata["qty"] < 0 else OrderSide.SELL
            self.send(sym, abs(pdata["qty"]), side, why=f"exit: {hit}")
        self._note_exit(sig)
        self.trade_state_save({})
        return True

    def decide(self, sig, pos, equity):
        """Mirror the Pine's two entry paths, with instrument-correct pricing.

          REVERSAL (holding the opposite side) -> MARKET
          RE-ENTRY (flat, trend intact)        -> trigger at st*(1 +/- buyback)

        CRITICAL: the buyback level is a SOXL price. When shorts are routed to
        SOXS (because SOXL is hard to borrow), that level CANNOT be used as a
        SOXS limit price -- SOXL trades ~120 and SOXS ~47, so such an order
        fills instantly at market. Instead the SOXL level is a TRIGGER: we wait
        for SOXL's bar to reach it, then send a MARKET order in SOXS.
        Sizing likewise uses the traded instrument's own price.
        """
        want = sig["want"]
        held = {k: v for k, v in pos.items() if abs(v["qty"]) > 1e-9}

        if want > 0:
            tgt, side, route = SYMBOL_LONG, OrderSide.BUY, "long SOXL"
        else:
            tgt, side, route = self.resolve_short_route()

        want_short_leg = (side == OrderSide.SELL)
        cur = held.get(tgt)
        if cur is not None:
            already = (cur["qty"] < 0) if want_short_leg else (cur["qty"] > 0)
            if already:
                return

        # an exit on THIS bar: the backtest never re-enters on the bar it exited on
        eb = getattr(self, "_exit_bar", None)
        if not held and eb is not None and self.cur_bar(sig) <= eb:
            self.log("exited on this bar - waiting for the next bar before any re-entry "
                     "(the backtest never re-enters on the bar it exited on)")
            return

        # ---- close anything that is not the target exposure (MARKET) ----
        st_prev = self.trade_state_load()
        if st_prev.get("exit_order_ids"):
            self.cancel_exit_orders(st_prev)
            self.trade_state_save(st_prev)
        flipped = False
        for sym, p in held.items():
            if sym == tgt and ((p["qty"] < 0) == want_short_leg):
                continue
            close_side = OrderSide.BUY if p["qty"] < 0 else OrderSide.SELL
            self.send(sym, abs(p["qty"]), close_side, why=f"flip: close {sym}")
            flipped = True

        trigger = sig["limit"]
        native_limit = (tgt == SYMBOL_LONG)   # trigger is on SOXL's own scale
        # the rule: if the CURRENT market price is already at or better than the
        # re-entry level, enter now; if not, wait for it
        px_now = sig.get("px") or sig.get("mon_close") or sig["close"]
        better = (px_now <= trigger) if want > 0 else (px_now >= trigger)
        need = "a dip to" if want > 0 else "a rally to"

        if flipped:
            kind, limit_price, fire = "MARKET (reversal)", None, True
        elif native_limit:
            kind, limit_price, fire = (f"LIMIT @ {trigger:.2f} (buyback)" +
                                       (f" - SOXL {px_now:.2f} is already at/better: fills now"
                                        if better else f" - waits (SOXL {px_now:.2f})"),
                                       trigger, True)
        else:
            # routed to SOXS: SOXL level is only a trigger, tested against the CURRENT price
            fire, limit_price = better, None
            kind = (f"MARKET on {tgt} (SOXL {px_now:.2f} is at/better than the trigger {trigger:.2f})"
                    if better else f"WAITING — SOXL {px_now:.2f}, needs {need} {trigger:.2f}")

        if not fire:
            self.log(f"{route} | {kind}")
            return

        # ---- when flat, any resting order that is not THE entry we want is stale ----
        if not held:
            for o in self.open_orders():
                if o.symbol not in (SYMBOL_LONG, SYMBOL_SHORT):
                    continue
                lim = float(o.limit_price) if getattr(o, "limit_price", None) else None
                same = (native_limit and not flipped and o.symbol == tgt and lim is not None
                        and abs(lim - round(trigger, 2)) < 0.01
                        and (("BUY" in str(o.side).upper()) == (side == OrderSide.BUY)))
                if same:
                    return                                  # already resting
                if not self.armed:
                    self.log(f"dry run — would cancel stale order {o.symbol} "
                             f"@ {o.limit_price}", "warn")
                    continue
                try:
                    self.tclient.cancel_order_by_id(o.id)
                    self.log(f"cancelled stale order {o.symbol} {o.side} @ {o.limit_price}")
                except Exception:
                    pass

        # ---- size on the TRADED instrument's own price ----
        px = limit_price if limit_price is not None else self.last_price(tgt)
        if not px or px <= 0:
            self.log(f"no price for {tgt}", "warn")
            return
        sess = self.session_equity(equity, use_cache=True)
        mult = 1.0 if want > 0 else SHORT_EQUITY_MULT
        budget = sess["equity"] * self.equity_frac * mult
        qty = int(math.floor(budget / px))
        if qty <= 0:
            self.log("no size available", "warn")
            return

        self.log(f"route: {route} | {kind} | sizing off {tgt} @ {px:.2f}")
        self._rerouted = False
        oid = self.send(tgt, qty, side, limit_price=limit_price,
                        why=f"trend {'UP' if want > 0 else 'DOWN'} — {route} — {kind}",
                        reroute=(side == OrderSide.SELL and tgt == SYMBOL_LONG))
        held_sym = SYMBOL_SHORT if getattr(self, "_rerouted", False) else tgt
        side_s = "long" if want > 0 else "short"
        if limit_price is None or held_sym != tgt:
            # a MARKET order: freeze exit levels now (SOXL reference, regardless of
            # which instrument carries the exposure). The reference is the open of
            # the bar the fill happens on; an intrabar re-entry (SOXS trigger)
            # manages exits from the NEXT bar, a reversal at the open from this one
            self.record_entry(sig, held_sym, side_s, qty, ref=self.fill_ref(sig),
                              reentry_bar=None if flipped else self.cur_bar(sig))
        elif oid:
            # a resting LIMIT: its exit levels need the fill bar's open, known only
            # when it fills - adopt_pending() freezes them then
            self.trade_state_save({"pending": {
                "order_id": oid, "symbol": tgt, "side": side_s, "qty": int(qty),
                "limit": round(trigger, 2), "placed": str(self.cur_bar(sig))}})


# ================================================================== GUI =====
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Supertrend SOXL/SOXS — Trading Panel")
        self.geometry("880x760")
        self.q: queue.Queue = queue.Queue()
        self.engine: Engine | None = None
        self._build()
        self.after(150, self._drain)

    # ---------- layout ----------
    def _build(self):
        pad = dict(padx=8, pady=4)

        warn = tk.Label(self, bg="#7f1d1d", fg="white", justify="left", anchor="w",
                        font=("Segoe UI", 8),
                        text=("  RISK: backtest does NOT reconcile with the Pine run "
                              "(1.8x-3.7x). Expected return unknown within ~4x.  "
                              "factor=5.0 is a parameter spike.  SOXL/SOXS are 3x leveraged."))
        warn.pack(fill="x")

        # --- session settings ---
        f = ttk.LabelFrame(self, text="Session")
        f.pack(fill="x", **pad)
        ttk.Label(f, text="Initial capital $").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.cap = tk.StringVar(value="100000")
        ttk.Entry(f, textvariable=self.cap, width=14).grid(row=0, column=1, sticky="w")

        ttk.Label(f, text="Start date").grid(row=0, column=2, sticky="w", padx=(18, 6))
        self.sdate = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(f, textvariable=self.sdate, width=14).grid(row=0, column=3, sticky="w")

        ttk.Label(f, text="Equity per trade").grid(row=1, column=0, sticky="w", padx=6)
        self.frac = tk.StringVar(value="100")
        ttk.Combobox(f, textvariable=self.frac, width=11,
                     values=["10", "25", "50", "75", "100"]).grid(row=1, column=1, sticky="w")

        ttk.Label(f, text="Account").grid(row=1, column=2, sticky="w", padx=(18, 6))
        self.mode = tk.StringVar(value="PAPER")
        mb = ttk.Combobox(f, textvariable=self.mode, width=11,
                          values=["PAPER", "LIVE"], state="readonly")
        mb.grid(row=1, column=3, sticky="w")
        mb.bind("<<ComboboxSelected>>", self._mode_changed)

        self.armed = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="ARM (actually send orders)",
                        variable=self.armed).grid(row=2, column=0, columnspan=2,
                                                  sticky="w", padx=6, pady=(2, 6))

        b = ttk.Frame(f); b.grid(row=2, column=2, columnspan=2, sticky="e", padx=6)
        self.btn_start = ttk.Button(b, text="START", command=self.start)
        self.btn_stop = ttk.Button(b, text="STOP", command=self.stop, state="disabled")
        self.btn_flat = ttk.Button(b, text="FLATTEN", command=self.flatten)
        for w in (self.btn_start, self.btn_stop, self.btn_flat):
            w.pack(side="left", padx=3)

        # --- status strip ---
        self.status = tk.Label(self, text="IDLE", bg="#374151", fg="white",
                               font=("Segoe UI", 10, "bold"), anchor="w")
        self.status.pack(fill="x", padx=8)

        # --- account ---
        fa = ttk.LabelFrame(self, text="Account"); fa.pack(fill="x", **pad)
        self.lbl_acct = ttk.Label(fa, text="—", font=("Consolas", 10))
        self.lbl_acct.pack(anchor="w", padx=8, pady=6)

        # --- session equity ---
        fe = ttk.LabelFrame(self, text="Session Equity (compounding base for sizing)")
        fe.pack(fill="x", **pad)
        self.lbl_sess = tk.Label(fe, text="-", font=("Consolas", 10),
                                 anchor="w", justify="left")
        self.lbl_sess.pack(anchor="w", padx=8, pady=6)

        # --- next trade ---
        fn = ttk.LabelFrame(self, text="Next Trade")
        fn.pack(fill="x", **pad)
        self.lbl_next = tk.Label(fn, text="-", font=("Consolas", 11, "bold"),
                                 anchor="w", justify="left", fg="#1d4ed8")
        self.lbl_next.pack(anchor="w", padx=8, pady=8)

        # --- session trades ---
        ft = ttk.LabelFrame(self, text="Trades since start date")
        ft.pack(fill="both", expand=False, **pad)
        cols = ("time", "symbol", "side", "qty", "entry", "exit", "pnl", "pct")
        self.tv = ttk.Treeview(ft, columns=cols, show="headings", height=6)
        widths = (140, 60, 58, 70, 78, 78, 92, 70)
        for c, w in zip(cols, widths):
            self.tv.heading(c, text=c.upper())
            self.tv.column(c, width=w, anchor="e" if c not in
                           ("time", "symbol", "side") else "w")
        sb = ttk.Scrollbar(ft, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb.pack(side="right", fill="y", pady=6)
        # direction colours: long = green, short = red (NOT win/loss)
        self.tv.tag_configure("long", foreground=COL_LONG)
        self.tv.tag_configure("short", foreground=COL_SHORT)

        # --- position ---
        fp = ttk.LabelFrame(self, text="Open Position"); fp.pack(fill="x", **pad)
        self.lbl_pos = tk.Label(fp, text="NO OPEN POSITION", font=("Consolas", 12, "bold"),
                                fg="#6b7280", anchor="w", justify="left")
        self.lbl_pos.pack(anchor="w", padx=8, pady=8)

        # --- signal ---
        fs = ttk.LabelFrame(self, text="Signal"); fs.pack(fill="x", **pad)
        self.lbl_sig = ttk.Label(fs, text="—", font=("Consolas", 10))
        self.lbl_sig.pack(anchor="w", padx=8, pady=6)

        # --- log ---
        fl = ttk.LabelFrame(self, text="Log"); fl.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(fl, height=14, bg="#0b1020", fg="#d1d5db",
                           font=("Consolas", 9), wrap="none")
        self.log.pack(fill="both", expand=True, padx=6, pady=6)
        for tag, col in (("err", "#f87171"), ("warn", "#fbbf24"),
                         ("ok", "#34d399"), ("trade", "#60a5fa")):
            self.log.tag_config(tag, foreground=col)

    # ---------- actions ----------
    def _mode_changed(self, _=None):
        if self.mode.get() == "LIVE":
            top = tk.Toplevel(self); top.title("Confirm LIVE trading"); top.grab_set()
            tk.Label(top, bg="#7f1d1d", fg="white", padx=14, pady=10, justify="left",
                     text=("LIVE MODE USES REAL MONEY.\n\n"
                           "Valid has not signed off on this strategy. The backtest\n"
                           "does not reconcile with your Pine run (1.8x-3.7x apart).\n"
                           "SOXL/SOXS are 3x leveraged ETFs.\n\n"
                           f"Type   {LIVE_CONFIRM}   to continue.")).pack()
            e = tk.Entry(top, width=34); e.pack(pady=8); e.focus_set()

            def ok():
                if e.get().strip() == LIVE_CONFIRM:
                    top.destroy()
                    self._log_line("LIVE mode selected", "err")
                else:
                    self.mode.set("PAPER"); top.destroy()
                    messagebox.showinfo("Reverted", "Confirmation failed — back to PAPER.")
            ttk.Button(top, text="Confirm", command=ok).pack(pady=(0, 10))
            top.protocol("WM_DELETE_WINDOW", lambda: (self.mode.set("PAPER"), top.destroy()))

    def start(self):
        try:
            cap = float(self.cap.get()); frac = float(self.frac.get()) / 100.0
            datetime.strptime(self.sdate.get(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Capital must be a number and date YYYY-MM-DD.")
            return
        if cap <= 0 or not (0 < frac <= 1):
            messagebox.showerror("Invalid input", "Check capital / equity per trade.")
            return
        paper = self.mode.get() == "PAPER"
        if self.armed.get():
            if not messagebox.askyesno(
                "Arm trading",
                f"Send REAL orders to the {'PAPER' if paper else 'LIVE'} account?\n\n"
                f"Capital ${cap:,.0f} · {frac:.0%} per trade\n\n"
                + ("" if paper else "THIS USES REAL MONEY.")):
                return
        json.dump({"capital": cap, "start": self.sdate.get(), "frac": frac,
                   "mode": self.mode.get()}, open(STATE_FILE, "w"), indent=2)
        self.engine = Engine(self.q, paper, cap, frac, self.sdate.get(),
                             self.armed.get())
        self.engine.start()
        self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        armed = "ARMED" if self.armed.get() else "DRY RUN"
        col = "#166534" if paper else "#7f1d1d"
        self.status.config(text=f"RUNNING · {self.mode.get()} · {armed}", bg=col)

    def market_open(self):
        try:
            c = self.tclient.get_clock()
            return bool(c.is_open), c
        except Exception:
            return False, None

    def guard(self, equity):
        """Returns (ok_to_trade, reason). Enforced every poll."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._guard_day != today:
            self._guard_day, self._trades_today = today, 0
            self._day_start_equity = equity
        if REQUIRE_MARKET_OPEN:
            is_open, clk = self.market_open()
            if not is_open:
                nxt = getattr(clk, "next_open", None)
                return False, f"market CLOSED (next open {nxt})"
        if self._trades_today >= MAX_TRADES_PER_DAY:
            return False, f"daily trade cap hit ({self._trades_today})"
        if self._day_start_equity:
            dd = (equity / self._day_start_equity - 1) * 100
            if dd <= DAILY_LOSS_LIMIT_PCT:
                return False, f"daily loss limit hit ({dd:.2f}%)"
        return True, "ok"

    def stop(self):
        if self.engine:
            self.engine.stop()
        self.btn_stop.config(state="disabled")
        self.status.config(text="STOPPING…", bg="#374151")

    def flatten(self):
        if not messagebox.askyesno("Flatten", "Close ALL open positions now?"):
            return
        try:
            paper = self.mode.get() == "PAPER"
            k, s = creds(paper)
            tc = TradingClient(k, s, paper=paper)
            tc.close_all_positions(cancel_orders=True)
            self._log_line("FLATTEN sent — all positions closed", "trade")
        except Exception as exc:
            self._log_line(f"flatten failed: {exc}", "err")

    # ---------- queue drain ----------
    def _log_line(self, msg, level="info"):
        self.log.insert("end", f"[{datetime.now():%H:%M:%S}] {msg}\n", level)
        self.log.see("end")

    def _drain(self):
        try:
            while True:
                m = self.q.get_nowait()
                k = m["kind"]
                if k == "log":
                    self.log.insert("end", f"[{m['ts']}] {m['msg']}\n", m.get("level", "info"))
                    self.log.see("end")
                elif k == "account":
                    self.lbl_acct.config(
                        text=f"{m['number']}   equity ${m['equity']:,.2f}   "
                             f"buying power ${m['buying_power']:,.2f}")
                elif k == "session":
                    self.lbl_sess.config(
                        text=(f"baseline ${m['baseline']:,.2f} from {m['start']}   |   "
                              f"{m['n_trades']} trades [{m.get('source','')}]   "
                              f"realised ${m['realised']:+,.2f}   "
                              f"open ${m['unrealised']:+,.2f}\n"
                              f"RUNNING EQUITY ${m['equity']:,.2f}   "
                              f"x {m['frac']:.0%} chosen  ->  "
                              f"long budget ${m['equity']*m['frac']:,.2f}   "
                              f"short budget ${m['equity']*m['frac']*0.5:,.2f}"))
                elif k == "trades":
                    self._render_trades(m["trades"])
                elif k == "plan":
                    if m["in_place"]:
                        self.lbl_next.config(
                            text=f"position already correct in {m['symbol']} - holding",
                            fg=COL_FLAT)
                    else:
                        _ss = strategy_side(m["symbol"], m["side"])
                        self.lbl_next.config(
                            text=(f"{_ss.upper()} {m['qty']:,} {m['symbol']} "
                                  f"@ ~{m['price']:.2f}  =  ${m['notional']:,.2f}\n"
                                  f"  budget ${m['budget']:,.2f} "
                                  f"({m['frac']:.0%} x {m.get('mult',1.0):.0%} "
                                  f"{'LONG' if m.get('mult',1.0)==1.0 else 'SHORT'})   "
                                  f"route: {m['route']}\n"
                                  f"  {m['when']}"),
                            fg=side_colour(_ss))
                elif k == "positions":
                    self._render_pos(m["positions"], m["baseline"], m["start"])
                elif k == "signal":
                    self.lbl_sig.config(
                        text=f"bar {m['bar']}   close {m['close']:.2f}   "
                             f"supertrend {m['st']:.2f}   "
                             f"trend {'UP  -> SOXL' if m['want'] > 0 else 'DOWN -> SOXS'}   "
                             f"buyback limit {m['limit']:.2f}")
                elif k == "trade_popup":
                    self._popup(m)
                elif k == "stopped":
                    self.btn_start.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.status.config(text="STOPPED", bg="#374151")
        except queue.Empty:
            pass
        self.after(200, self._drain)

    def _render_trades(self, trades):
        for i in self.tv.get_children():
            self.tv.delete(i)
        for t in trades[-200:]:
            tag = strategy_side(t.get("symbol"), t.get("side"))
            ts = t["time"]
            ts = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)[:16]
            self.tv.insert("", "end", tags=(tag,), values=(
                ts, t["symbol"], t["side"], f"{t['qty']:,.0f}",
                f"{t['entry']:.2f}", f"{t['exit']:.2f}",
                f"{t['pnl']:+,.2f}", f"{t['pct']:+.2f}%"))
        if trades:
            self.tv.see(self.tv.get_children()[-1])

    def _render_pos(self, positions, baseline, start):
        held = {k: v for k, v in positions.items() if abs(v["qty"]) > 1e-9}
        if not held:
            self.lbl_pos.config(text="NO OPEN POSITION", fg="#6b7280")
            return
        lines, col = [], COL_FLAT
        for sym, p in held.items():
            sside = strategy_side(sym, p["side"])
            col = side_colour(sside)
            note = (f"   (long {SYMBOL_SHORT} = short {SYMBOL_LONG})"
                    if sym == SYMBOL_SHORT else "")
            lines.append(f"{sym}   {sside.upper()}{note}   qty {abs(p['qty']):,.0f}\n"
                         f"  entry {p['avg']:.2f}   last {p['cur']:.2f}   "
                         f"value ${p['mv']:,.2f}\n"
                         f"  unrealised ${p['upl']:,.2f}  ({p['uplpc']:+.2f}%)")
        self.lbl_pos.config(text="\n".join(lines) + f"\n  [baseline ${baseline:,.0f} from {start}]",
                            fg=col)

    def _popup(self, m):
        p = m["payload"]
        title = ("DRY RUN — order NOT sent" if m["dry"]
                 else f"ORDER SENT — {m['mode']}")
        top = tk.Toplevel(self); top.title(title); top.attributes("-topmost", True)
        bg = "#92400e" if m["dry"] else ("#166534" if m["mode"] == "PAPER" else "#7f1d1d")
        tk.Label(top, text=title, bg=bg, fg="white", font=("Segoe UI", 11, "bold"),
                 padx=16, pady=8).pack(fill="x")
        tk.Label(top, text=m["why"], font=("Segoe UI", 9), padx=16, pady=(8, 2)).pack(anchor="w")
        body = tk.Text(top, height=len(p) + 2, width=52, font=("Consolas", 10),
                       bg="#0b1020", fg="#d1d5db")
        body.insert("end", json.dumps(p, indent=2))
        body.config(state="disabled"); body.pack(padx=14, pady=10)
        ttk.Button(top, text="OK", command=top.destroy).pack(pady=(0, 12))
        self._log_line(f"{'DRY RUN' if m['dry'] else 'SENT'} {json.dumps(p)}", "trade")


def _launch():
    """Redirect to the consolidated app.

    This module is now the LIVE ENGINE LIBRARY (Engine, AlertRouter, creds) that
    supertrend_app.py imports. Running it directly used to open a live-only
    window with no Parameters or Backtest tab, which is confusing -- so it now
    hands off to the full three-tab application instead.
    """
    print("=" * 66)
    print("  supertrend_gui.py is the live-engine library.")
    print("  Launching the full application (Parameters + Backtest + Live)...")
    print("=" * 66, flush=True)
    import supertrend_app
    supertrend_app._launch()


if __name__ == "__main__":
    import traceback
    try:
        _launch()
    except Exception:
        print("\nFAILED TO START:\n", flush=True)
        traceback.print_exc()
        try:
            input("\nPress Enter to close...")
        except Exception:
            pass
