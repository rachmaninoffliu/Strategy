"""
Main Supertrend Strategy - SOXL / SOXS  —  faithful Python translation
=====================================================================

1:1 port of the Pine v5 script "Main Supertrend Strategy - SOXL Chart /
SOXS Short Alerts", running on Alpaca market data.

Every Pine input is exposed on Config with the same name and default.
Strategy logic is unchanged from the Pine. This module is the STRATEGY ONLY —
no benchmarking, no improvement variants. See supertrend_15m.py for research.

STRUCTURE MIRRORS THE PINE
--------------------------
  supertrend()          ta.supertrend(factor, atrPeriod)
  Config                the input.* block
  AlertRouter           the f_json*/f_*Msg helper block (SOXS routing)
  SupertrendStrategy    the entry/exit/management blocks

TWO ENTRY PATHS (both present in the Pine, both implemented)
------------------------------------------------------------
  REVERSAL  position_size[1] is opposite sign  -> MARKET order
  RE-ENTRY  position_size[1] is same sign/flat -> LIMIT at st*(1 +/- buyback)

  The re-entry ("buyback") path is what lets the strategy get back into an
  intact trend after a take-profit. Omitting it collapses returns.

EXIT SEQUENCING (important)
---------------------------
  The trailing level is derived from the running extreme AS OF THE PRIOR BAR,
  then tested against the current bar's range. Updating the extreme with the
  current bar's high and then testing the same bar's low against the resulting
  level fires on any wide-range bar and truncates every winner.

USAGE
-----
  python supertrend_strategy.py                        # defaults: SOXL 15min
  python supertrend_strategy.py --symbol SOXL --tf 60  # 1h
  python supertrend_strategy.py --start 2020-01-01 --equity-frac 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import talib

warnings.filterwarnings("ignore")

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed


# ============================================================================
# CONFIG  — mirrors the Pine input.* block, same names, same defaults
# ============================================================================
@dataclass
class Config:
    # --- Master Strategy Mode ---
    strategyMode: str = "Always Fixed"   # Always Fixed | Always Dynamic

    # --- Backtest Range ---
    useDateFilter: bool = True
    startDate: str = "2023-01-01"
    endDate: str = "2099-12-31"

    # --- Alert Routing ---
    alertRouting: str = "Map SOXL shorts to SOXS"   # or "Direct SOXL long/short"
    soxlAlertSymbol: str = "SOXL"
    soxsAlertSymbol: str = "SOXS"
    soxsPriceSymbol: str = "SOXS"
    manualSoxsCloseQty: float = 0.0

    # --- Core Inputs ---
    atrPeriod: int = 500
    factor: float = 5.0
    riskReward: float = 4.5 / 100
    buyback: float = 1.5 / 100
    trailFactor: float = 0
    price_perc: float = 0.1
    riskfactor: float = 1.0
    limitfactor: float = 1.0

    # --- Volatility Regime Filter ---
    volLength: int = 20
    volSmaLength: int = 100
    volThreshold: float = 1.3

    # --- Dynamic Trailing ---
    useDynamic: bool = True
    adxLength: int = 14
    adxBase: float = 25.0

    # --- No-Trade Months (Jan..Dec) ---
    noTradeMonths: tuple = field(default_factory=lambda: tuple([False] * 12))
    closeOnBlockedMonth: bool = True

    # --- Execution (not in Pine; TradingView takes these from Properties) ---
    initial_capital: float = 100_000.0
    equity_frac: float = 1      # percent-of-equity sizing
    commission_bps: float = 0      # per leg
    capacity_usd: float = 100_000_000_000 # max notional per trade
    touch_fill: bool = False         # True = TradingView's touch-fill rule
    short_mult: float = 1.0          # Pine sizes shorts at equity*0.5 while
                                     # longs use the full chosen %. Set 0.5 to
                                     # reproduce that asymmetry.
    pessimistic: bool = False        # if a bar can hit BOTH the TP and the
                                     # trailing stop, assume the STOP filled
                                     # first (conservative intrabar ordering)
    strict_entries: bool = False     # OFF = the original behaviour. ON = exits and
                                     # flat re-entries use only what a real order
                                     # KNOWS at the open of the bar (the previous
                                     # bar's trend / regime / ADX). The original
                                     # uses THIS bar's close for them, so a flat
                                     # re-entry can fire on the very bar whose
                                     # close flips the trend - a trade a live
                                     # engine (and Pine) can never take.

    @property
    def routeShortsToSOXS(self) -> bool:
        return self.alertRouting == "Map SOXL shorts to SOXS"


# ============================================================================
# INDICATOR  — ta.supertrend equivalent
# ============================================================================
def supertrend(high, low, close, period: int, factor: float):
    """Returns (st, dir). dir == -1 uptrend (long), dir == +1 downtrend (short).

    Matches Pine's ta.supertrend: ATR uses Wilder smoothing (talib.ATR), and the
    bands ratchet — the lower band only rises, the upper band only falls, until
    price closes through them.
    """
    atr = talib.ATR(high, low, close, timeperiod=period)
    hl2 = (high + low) / 2.0
    upper, lower = hl2 + factor * atr, hl2 - factor * atr

    n = len(close)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    fu = np.full(n, np.nan)   # final upper band
    fl = np.full(n, np.nan)   # final lower band

    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if i == 0 or np.isnan(fu[i - 1]):
            fu[i], fl[i] = upper[i], lower[i]
            direction[i] = 1.0
            st[i] = upper[i]
            continue

        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]

        if direction[i - 1] == 1.0:
            direction[i] = -1.0 if close[i] > fu[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] < fl[i] else -1.0

        st[i] = fl[i] if direction[i] == -1.0 else fu[i]

    return st, direction


# ============================================================================
# ALERT ROUTER  — the f_json* / f_*Msg helper block
# ============================================================================
class AlertRouter:
    """Builds the webhook JSON payloads exactly as the Pine helpers do."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.soxsOpenAlertQty: float | None = None
        self.soxsLastOpenAlertQty: float | None = None

    @staticmethod
    def _abs_qty(qty: float) -> int:
        return int(math.floor(abs(qty if qty == qty else 0)))

    def _json(self, ticker: str, action: str, qty: float) -> str:
        return json.dumps({"ticker": ticker, "action": action,
                           "quantity": self._abs_qty(qty)}, separators=(",", ":"))

    def _json_notional(self, ticker: str, action: str, dollars: float) -> str:
        return json.dumps({"ticker": ticker, "action": action,
                           "quantity": round(dollars, 2),
                           "quantityType": "dollar_amount"}, separators=(",", ":"))

    @staticmethod
    def is_before_935(ts: pd.Timestamp) -> bool:
        """f_isBefore935() — pre-9:35 ET entries size by notional, not shares."""
        return (ts.hour * 100 + ts.minute) < 935

    def soxs_qty_from_equity(self, equity: float, soxs_price: float) -> int:
        """f_soxsQtyFromEquity() — 50% of equity / SOXS price, floored."""
        if soxs_price and soxs_price > 0:
            return int(math.floor((equity * self.cfg.equity_frac) / soxs_price))
        return 0

    def soxl_short_qty(self, equity: float, price: float) -> int:
        """f_soxlShortQty() — 50% of equity / SOXL price, floored."""
        if price and price > 0:
            return int(math.floor((equity * self.cfg.equity_frac) / price))
        return 0

    def soxs_exit_qty(self, fallback: float) -> float:
        """f_soxsExitQty() — never recompute a close from current price."""
        if not self.cfg.routeShortsToSOXS:
            return fallback
        for q in (self.soxsOpenAlertQty, self.soxsLastOpenAlertQty,
                  self.cfg.manualSoxsCloseQty):
            if q is not None and q == q:
                return q
        return fallback

    # -- message builders ---------------------------------------------------
    def long_entry(self, qty):  return self._json(self.cfg.soxlAlertSymbol, "buy", qty)
    def long_exit(self, qty):   return self._json(self.cfg.soxlAlertSymbol, "sell", qty)

    def short_entry(self, qty, equity, ts):
        c = self.cfg
        if not c.routeShortsToSOXS:
            return self._json(c.soxlAlertSymbol, "sell", qty)
        if self.is_before_935(ts):
            return self._json_notional(c.soxsAlertSymbol, "buy",
                                       int(equity * c.equity_frac))
        return self._json(c.soxsAlertSymbol, "buy", qty)

    def short_exit(self, qty):
        c = self.cfg
        return (self._json(c.soxsAlertSymbol, "sell", qty) if c.routeShortsToSOXS
                else self._json(c.soxlAlertSymbol, "buy", qty))

    def reversal(self, side: str, close_qty, open_qty, equity, ts) -> list[str]:
        """Reversal alerts.

        SOXS routing  -> TWO messages (different tickers, cannot be merged).
        Direct SOXL   -> ONE message with the combined quantity.
        """
        c = self.cfg
        if side == "long_to_short":
            if c.routeShortsToSOXS:
                return [self._json(c.soxlAlertSymbol, "sell", close_qty),
                        self.short_entry(open_qty, equity, ts)]
            return [self._json(c.soxlAlertSymbol, "sell", close_qty + open_qty)]
        else:  # short_to_long
            if c.routeShortsToSOXS:
                return [self._json(c.soxsAlertSymbol, "sell", close_qty),
                        self._json(c.soxlAlertSymbol, "buy", open_qty)]
            return [self._json(c.soxlAlertSymbol, "buy", close_qty + open_qty)]


# ============================================================================
# STRATEGY ENGINE
# ============================================================================
class SupertrendStrategy:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.router = AlertRouter(cfg)

    # -- helpers ------------------------------------------------------------
    def _blocked_month(self, ts) -> bool:
        return bool(self.cfg.noTradeMonths[ts.month - 1])

    def _in_date_range(self, ts) -> bool:
        c = self.cfg
        if not c.useDateFilter:
            return True
        return pd.Timestamp(c.startDate) <= ts <= pd.Timestamp(c.endDate)

    def _size(self, equity: float, price: float, side: str = "long") -> int:
        c = self.cfg
        mult = c.short_mult if side == "short" else 1.0
        notional = min(equity * c.equity_frac * mult, c.capacity_usd)
        return int(math.floor(notional / price)) if price > 0 else 0

    def _trail_factor(self, adx_val: float) -> float:
        c = self.cfg
        if not c.useDynamic:
            return c.trailFactor
        a = c.adxBase if (adx_val != adx_val) else adx_val
        return max(0.05, min(0.95, c.trailFactor + ((c.adxBase - a) / 100.0)))

    # -- main loop ----------------------------------------------------------
    def run(self, bars: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        """bars: DataFrame indexed by timestamp with open/high/low/close."""
        c = self.cfg
        o = bars["open"].to_numpy(float)
        h = bars["high"].to_numpy(float)
        l = bars["low"].to_numpy(float)
        cl = bars["close"].to_numpy(float)
        idx = bars.index

        st, direction = supertrend(h, l, cl, c.atrPeriod, c.factor)
        atr_pct = talib.ATR(h, l, cl, timeperiod=c.volLength) / cl
        vol_base = talib.SMA(atr_pct, timeperiod=c.volSmaLength)
        adx = talib.ADX(h, l, cl, timeperiod=c.adxLength)

        equity = c.initial_capital
        pos, qty = 0, 0
        entry_price = entry_ref = supert_entry = target = np.nan
        last_mark = np.nan   # price the open position was last marked at
        active_fixed = True
        max_high = min_low = np.nan

        curve, trades = [], []
        open_rec = None
        open_fill_i = None        # bar the open position FILLED on (a reversal fills at the next bar's open)
        self.open_trade = None    # the position still open after the last bar (set at the end of run)

        def close_position(i, px, reason):
            nonlocal equity, pos, qty, max_high, min_low, open_rec, last_mark
            equity += qty * (px - last_mark) * pos
            equity -= abs(qty) * px * c.commission_bps / 1e4
            r = (px / open_rec["price"] - 1.0) * (1 if open_rec["side"] == "long" else -1)
            alert = (self.router.long_exit(qty) if open_rec["side"] == "long"
                     else self.router.short_exit(self.router.soxs_exit_qty(qty)))
            trades.append({**open_rec, "exit_time": idx[i], "exit_price": px,
                           "exit_reason": reason, "return_pct": r * 100,
                           "pnl_usd": qty * (px - open_rec["price"]) *
                                      (1 if open_rec["side"] == "long" else -1),
                           "equity_after": equity, "exit_alert": alert})
            pos, qty, max_high, min_low, open_rec = 0, 0, np.nan, np.nan, None

        def open_position(i, px, side, alert, ref=None, st_ref=None, qty_fixed=None, fill_i=None):
            """px is the FILL price (drives P&L and sizing).

            Pine keeps a SEPARATE reference price for all target/trail maths:

                if ta.crossover(strategy.position_size, 0)
                    supert     := st
                    entryPrice := open      <-- the bar's OPEN, not the fill

            For a limit re-entry filled at st*(1+buyback) on a bar that opened
            higher, entryPrice > fill. Every target, TP limit and trailing level
            is therefore computed off a HIGHER reference than the fill price.
            Using the fill for both makes targets easier to reach, cutting
            winners short. entry_ref reproduces Pine's behaviour.
            Pine also freezes `supert` at entry; the dynamic TP uses that frozen
            value, not the live Supertrend.
            """
            nonlocal equity, pos, qty, entry_price, entry_ref, supert_entry
            nonlocal target, active_fixed, max_high, min_low, open_rec, last_mark, open_fill_i
            # qty_fixed: on a reversal the quantity was already computed (and
            # embedded in the alert) BEFORE the closing leg changed equity.
            # Recomputing here produced an alert/fill mismatch -- the webhook
            # said buy 9802 while the position opened at 9799. Pine computes
            # the size ONCE and uses it for both.
            q = qty_fixed if qty_fixed is not None else self._size(equity, px, side)
            if q <= 0:
                return
            equity -= q * px * c.commission_bps / 1e4
            pos = 1 if side == "long" else -1
            qty = q
            entry_price = px            # fill price -> P&L
            last_mark = px              # mark from the FILL, not prior close
            # Pine's entryPrice := open OF THE BAR THE FILL OCCURRED ON.
            # Reversals fill at o[i+1] so ref == fill; only limit re-entries
            # (filled inside bar i) have open != fill.
            entry_ref = o[i] if ref is None else ref
            supert_entry = st[i] if st_ref is None else st_ref
            risk = abs(entry_ref - supert_entry) / entry_ref * c.riskfactor
            move = c.riskReward if active_fixed else risk
            target = entry_ref * (1 + move) if pos > 0 else entry_ref * (1 - move)
            max_high = min_low = np.nan
            open_rec = {"side": side, "entry_time": idx[i], "price": px,
                        "qty": q, "entry_alert": alert}
            open_fill_i = i if fill_i is None else fill_i

        def reverse(i, want_i, fixed_i):
            """REVERSAL: the trend flipped at bar i's close -> market order at the NEXT bar's open."""
            nonlocal active_fixed
            px = o[i + 1]
            close_qty = abs(qty)
            open_qty = self._size(equity, px,
                                 "long" if want_i > 0 else "short")
            side = "long_to_short" if pos > 0 else "short_to_long"
            msgs = self.router.reversal(side, close_qty, open_qty, equity, idx[i])
            if c.routeShortsToSOXS and side == "long_to_short":
                self.router.soxsOpenAlertQty = open_qty
                self.router.soxsLastOpenAlertQty = open_qty
            close_position(i, px, "reversal")
            active_fixed = fixed_i
            nxt = i + 1
            # strict_entries: the order is sent at bar i's close, so the Supertrend it can
            # freeze is st[i]; st[nxt] only exists once the fill bar has closed
            st_ref = st[i] if c.strict_entries else (st[nxt] if not np.isnan(st[nxt]) else st[i])
            open_position(i, px, "long" if want_i > 0 else "short",
                          " | ".join(msgs), ref=px, st_ref=st_ref,
                          qty_fixed=open_qty, fill_i=nxt)

        for i in range(1, len(cl)):
            ts = idx[i]
            if pos != 0:
                # Mark from the price the position was last marked at. Using
                # cl[i-1] double-counts the cl[i-1]->fill gap on reversals,
                # where the new position is opened at o[i+1] rather than cl[i].
                equity += qty * (cl[i] - last_mark) * pos
                last_mark = cl[i]
            if equity <= 0:
                curve.append((ts, 0.0))
                continue
            if np.isnan(direction[i]) or np.isnan(st[i]):
                curve.append((ts, equity))
                continue

            # --- regime + mode for this bar ---
            high_vol = (vol_base[i] == vol_base[i]) and atr_pct[i] > vol_base[i] * c.volThreshold
            is_fixed = (c.strategyMode == "Always Fixed" or
                        (c.strategyMode == "Auto-Switch via Vol" and high_vol))

            long_sig = direction[i] == -1
            want = 1 if long_sig else -1
            tradeable = self._in_date_range(ts) and not self._blocked_month(ts)

            # --- auto-close in a blocked month ---
            if c.closeOnBlockedMonth and self._blocked_month(ts) and pos != 0:
                close_position(i, cl[i], "blocked_month")
                curve.append((ts, equity))
                continue

            # what this bar's orders may know: the trend / regime / ADX as of THIS
            # bar's close (original behaviour) or, with strict_entries, as of the
            # PREVIOUS bar's close (what exists when the bar opens)
            known, known_fixed, adx_k = want, is_fixed, adx[i]
            if c.strict_entries:
                if np.isnan(direction[i - 1]):
                    curve.append((ts, equity))
                    continue
                known = 1 if direction[i - 1] == -1 else -1
                hv_prev = (vol_base[i - 1] == vol_base[i - 1]) and atr_pct[i - 1] > vol_base[i - 1] * c.volThreshold
                known_fixed = (c.strategyMode == "Always Fixed" or
                               (c.strategyMode == "Auto-Switch via Vol" and hv_prev))
                adx_k = adx[i - 1]

            # --- manage open position (exits) ---
            if pos != 0 and known == pos:
                tf = self._trail_factor(adx_k)
                hit, px = False, np.nan
                if pos > 0:
                    tp = (entry_ref * (1 + c.price_perc) if active_fixed else
                          entry_ref * (1 + (entry_ref - supert_entry) / entry_ref * c.limitfactor))
                    trail = (entry_ref + (max_high - entry_ref) * tf
                             if max_high == max_high and max_high >= target else np.nan)
                    stop_ok = (trail == trail) and l[i] <= trail
                    tp_ok = h[i] >= tp
                    if c.pessimistic and stop_ok:
                        px, hit = min(o[i], trail), True
                    elif tp_ok:
                        px, hit = max(o[i], tp), True
                    elif stop_ok:
                        px, hit = min(o[i], trail), True
                    if not hit:
                        max_high = h[i] if max_high != max_high else max(max_high, h[i])
                else:
                    tp = (entry_ref * (1 - c.price_perc) if active_fixed else
                          entry_ref * (1 - (supert_entry - entry_ref) / entry_ref * c.limitfactor))
                    trail = (entry_ref - (entry_ref - min_low) * tf
                             if min_low == min_low and min_low <= target else np.nan)
                    stop_ok = (trail == trail) and h[i] >= trail
                    tp_ok = l[i] <= tp
                    if c.pessimistic and stop_ok:
                        px, hit = max(o[i], trail), True
                    elif tp_ok:
                        px, hit = min(o[i], tp), True
                    elif stop_ok:
                        px, hit = max(o[i], trail), True
                    if not hit:
                        min_low = l[i] if min_low != min_low else min(min_low, l[i])
                if hit:
                    close_position(i, px, "target" if px >= tp and pos > 0 else "trail")
                    # A position closed on this bar cannot be re-opened on the
                    # same bar. In Pine the re-entry limit order is placed at
                    # bar open, i.e. BEFORE this exit filled, so it cannot also
                    # fill after it. Allowing both produced ~39% too many trades.
                    curve.append((ts, equity))
                    continue

            # --- REVERSAL: market order at next open ---
            if pos != 0 and want != pos and tradeable and i + 1 < len(cl):
                reverse(i, want, is_fixed)

            # --- RE-ENTRY: resting limit at st*(1 +/- buyback) ---
            elif pos == 0 and tradeable and not np.isnan(st[i - 1]):
                # LOOK-AHEAD FIX: the resting limit is priced off the PREVIOUS
                # bar's Supertrend. st[i] is not known until bar i closes, so
                # pricing the order off st[i] and then testing bar i's low
                # against it uses information that did not exist when the order
                # would have been placed. Pine places the order at bar close;
                # it can only fill on a subsequent bar.
                ref = st[i - 1]
                limit = ref * (1 + c.buyback) if known > 0 else ref * (1 - c.buyback)
                if c.touch_fill:
                    filled = (l[i] <= limit) if known > 0 else (h[i] >= limit)
                else:
                    filled = (l[i] < limit) if known > 0 else (h[i] > limit)
                if filled:
                    px = min(o[i], limit) if known > 0 else max(o[i], limit)
                    active_fixed = known_fixed
                    if known > 0:
                        alert = self.router.long_entry(self._size(equity, px))
                    else:
                        q = self._size(equity, px)
                        alert = self.router.short_entry(q, equity, ts)
                        if c.routeShortsToSOXS:
                            self.router.soxsOpenAlertQty = q
                            self.router.soxsLastOpenAlertQty = q
                    # strict_entries: a fill inside bar i can only freeze the last CLOSED bar's Supertrend
                    open_position(i, px, "long" if known > 0 else "short", alert,
                                  st_ref=st[i - 1] if c.strict_entries else None)
                    # strict_entries: this bar's own close may flip the trend right after
                    # the intrabar fill. A real order is reversed AT that close (fill at the
                    # next open), exactly like a position that was already open. The original
                    # loop has passed its reversal check for this bar, so it would hold the
                    # position one bar too long.
                    if (c.strict_entries and pos != 0 and want != pos and tradeable
                            and i + 1 < len(cl)):
                        reverse(i, want, is_fixed)

            curve.append((ts, equity))

        eq = pd.DataFrame(curve, columns=["time", "equity"]).set_index("time")["equity"]
        # the position still open after the last bar, with its frozen levels
        if pos != 0 and open_rec is not None and len(cl):
            k = len(cl) - 1
            tf = self._trail_factor(adx[k - 1] if (c.strict_entries and k >= 1) else adx[k])
            if pos > 0:
                tp = (entry_ref * (1 + c.price_perc) if active_fixed else
                      entry_ref * (1 + (entry_ref - supert_entry) / entry_ref * c.limitfactor))
                trail = (entry_ref + (max_high - entry_ref) * tf
                         if max_high == max_high and max_high >= target else np.nan)
            else:
                tp = (entry_ref * (1 - c.price_perc) if active_fixed else
                      entry_ref * (1 - (supert_entry - entry_ref) / entry_ref * c.limitfactor))
                trail = (entry_ref - (entry_ref - min_low) * tf
                         if min_low == min_low and min_low <= target else np.nan)
            self.open_trade = {
                "side": open_rec["side"], "entry_time": open_rec["entry_time"],
                "fill_time": idx[min(open_fill_i if open_fill_i is not None else k, k)],
                "price": open_rec["price"], "qty": qty, "entry_ref": entry_ref,
                "supert_entry": supert_entry, "target": target, "tp": tp,
                "trail": None if trail != trail else float(trail),
                "active_fixed": bool(active_fixed),
                "extreme": None if (max_high if pos > 0 else min_low) != (max_high if pos > 0 else min_low)
                else float(max_high if pos > 0 else min_low),
                "last_time": idx[k], "last_close": float(cl[k]),
                "unrealized_pct": (cl[k] / open_rec["price"] - 1.0) * 100 * pos,
                "unrealized_usd": float(qty * (cl[k] - open_rec["price"]) * pos)}
        return eq, pd.DataFrame(trades)


# ============================================================================
# DATA
# ============================================================================
ET = "America/New_York"
SIP_REALTIME_GUARD = pd.Timedelta(minutes=16)   # stay clear of the SIP delay window


def data_client():
    """Alpaca historical client with the project's keys ("Alpaca Key.txt", then the environment, then ~/.claude.json - alpaca_keys.py)."""
    import alpaca_keys
    return StockHistoricalDataClient(*alpaca_keys.creds(True))


def request_window(start: str, end: str):
    """User-entered dates -> UTC bounds, INCLUSIVE of both calendar days in ET.

    The old code passed the bare date string as `end`, which Alpaca reads as
    00:00Z on that day -- silently dropping the whole final session. Anchoring
    to 23:59:59 ET is what makes "pull everything between the two dates the
    user typed" actually true.
    """
    s = pd.Timestamp(f"{str(start)[:10]} 00:00:00", tz=ET)
    e = pd.Timestamp(f"{str(end)[:10]} 23:59:59", tz=ET)
    now = pd.Timestamp.now(tz="UTC")
    if e > now:
        e = now - SIP_REALTIME_GUARD
    return s.tz_convert("UTC"), e.tz_convert("UTC")


def fetch_bars_window(client, symbols, timeframe_min, start_utc, end_utc,
                      rth_only=True, dividend_adjusted=False, log=print):
    """Fetch [start_utc, end_utc] in <=1-year chunks. Returns a raw panel."""
    if end_utc <= start_utc:
        return pd.DataFrame()
    unit = TimeFrameUnit.Minute if timeframe_min < 60 else TimeFrameUnit.Hour
    amt = timeframe_min if timeframe_min < 60 else timeframe_min // 60
    frames, cur = [], start_utc
    while cur < end_utc:
        nxt = min(cur + pd.DateOffset(years=1), end_utc)
        log(f"  fetching {cur.tz_convert(ET):%Y-%m-%d} -> "
            f"{nxt.tz_convert(ET):%Y-%m-%d} ...")
        df = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame(amt, unit),
            start=cur.to_pydatetime(), end=nxt.to_pydatetime(),
            adjustment=(Adjustment.ALL if dividend_adjusted else Adjustment.SPLIT),
            feed=DataFeed.SIP)).df.reset_index()
        if len(df):
            frames.append(df)
        cur = nxt
    if not frames:
        return pd.DataFrame()
    p = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(p["timestamp"], utc=True).dt.tz_convert(ET)
    p["et"] = ts
    if rth_only:
        p = p[(ts.dt.time >= pd.Timestamp("09:30").time()) &
              (ts.dt.time < pd.Timestamp("16:00").time()) &
              (ts.dt.dayofweek < 5)].copy()
    return p


def _meta_path(cache):
    return cache + ".meta.json"


def _read_meta(cache, have):
    """Window already fetched into this cache. Bootstrapped from the data for
    pre-existing caches written before coverage tracking existed."""
    mp = _meta_path(cache)
    if os.path.exists(mp):
        try:
            m = json.load(open(mp))
            return (pd.Timestamp(m["covered_start"]).tz_convert("UTC"),
                    pd.Timestamp(m["covered_end"]).tz_convert("UTC"))
        except Exception:
            pass
    if have is not None and len(have):
        return have["et"].min().tz_convert("UTC"), have["et"].max().tz_convert("UTC")
    return None, None


def _write_meta(cache, cov_s, cov_e):
    json.dump({"covered_start": cov_s.isoformat(),
               "covered_end": cov_e.isoformat(),
               "fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
              open(_meta_path(cache), "w"), indent=2)


def load_alpaca_bars(symbols, timeframe_min, start, end, rth_only=True,
                     cache_dir="C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data",
                     dividend_adjusted=False, log=print):
    """Bars for [start, end], guaranteeing the cache covers the whole window.

    dividend_adjusted=False (default) uses Adjustment.SPLIT.
    TradingView charts are split-adjusted but NOT dividend-adjusted. Using
    Adjustment.ALL shifts historical prices down (-8.97% in 2016 decaying to
    0.00% today for SOXL) and desynchronises signals from the Pine backtest.

    CACHING: the cache is a superset store, not a snapshot. Coverage is
    tracked in a sidecar .meta.json and the missing head/tail are fetched on
    every call, so a stale file can never be silently replayed as if current.
    The old `if os.path.exists(cache)` short-circuit froze every backtest on
    the day the file happened to be created.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = "-".join(sorted(symbols))
    adj_tag = "div" if dividend_adjusted else "split"
    cache = os.path.join(cache_dir,
                         f"{tag}_{timeframe_min}m_{str(start)[:4]}_{adj_tag}.parquet")

    req_s, req_e = request_window(start, end)
    have = pd.read_parquet(cache) if os.path.exists(cache) else None
    if have is not None and len(have):
        have["et"] = pd.to_datetime(have["et"], utc=True).dt.tz_convert(ET)
    else:
        have = None
    cov_s, cov_e = _read_meta(cache, have)

    pieces, client = ([have] if have is not None else []), None
    todo = []
    if have is None or cov_s is None:
        todo.append((req_s, req_e, "full"))
    else:
        if req_s < cov_s:
            todo.append((req_s, cov_s, "back-fill"))
        if req_e > cov_e:
            # restart at the last covered DAY so a partial final session completes
            todo.append((cov_e.tz_convert(ET).normalize().tz_convert("UTC"),
                         req_e, "top-up"))

    added = 0
    for s_utc, e_utc, why in todo:
        if client is None:
            client = data_client()
        log(f"  {why}: {s_utc.tz_convert(ET):%Y-%m-%d} -> {e_utc.tz_convert(ET):%Y-%m-%d}")
        got = fetch_bars_window(client, symbols, timeframe_min, s_utc, e_utc,
                                rth_only=rth_only,
                                dividend_adjusted=dividend_adjusted, log=log)
        if len(got):
            pieces.append(got)
            added += len(got)
            log(f"    +{len(got):,} bars")

    if not pieces:
        raise RuntimeError(f"no bars for {symbols} {start}..{end}")

    p = pd.concat(pieces, ignore_index=True)
    p["et"] = pd.to_datetime(p["et"], utc=True).dt.tz_convert(ET)
    # later pieces win, so a re-pulled partial session is replaced by the full one
    p = (p.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
           .sort_values(["symbol", "timestamp"], ignore_index=True))

    if todo:
        new_s = min(req_s, cov_s) if cov_s is not None else req_s
        new_e = max(req_e, cov_e) if cov_e is not None else req_e
        if added:                       # avoid rewriting MBs when nothing is new
            p.to_parquet(cache, index=False)
        _write_meta(cache, new_s, new_e)

    p["bar"] = p["et"].dt.tz_localize(None)
    warn_if_stale(p, req_e, log=log)
    return p


def warn_if_stale(panel, req_e, log=print):
    """Loud, non-fatal notice when the newest bar trails the requested end.

    The failure this exists for was invisible precisely because a truncated
    result still renders a clean equity curve.
    """
    if panel is None or not len(panel):
        return
    last = panel["et"].max()
    gap = (req_e.tz_convert(ET) - last).total_seconds() / 86400.0
    log(f"  coverage: {panel['et'].min():%Y-%m-%d} -> {last:%Y-%m-%d %H:%M} "
        f"({len(panel):,} bars)")
    if gap > 4:
        log(f"  !! STALE DATA: newest bar is {gap:.1f} days behind the requested "
            f"end ({req_e.tz_convert(ET):%Y-%m-%d}). Results are NOT current.")


def to_ohlc(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = panel[panel["symbol"] == symbol].set_index("bar").sort_index()
    return df[["open", "high", "low", "close"]].ffill().dropna()


# ============================================================================
# REPORTING
# ============================================================================
def summarize(eq: pd.Series, trades: pd.DataFrame, bars_per_year: int) -> dict:
    e = eq / eq.iloc[0]
    r = e.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    yrs = len(e) / bars_per_year
    vol = r.std() * np.sqrt(bars_per_year)
    wins = trades[trades.return_pct > 0] if len(trades) else trades
    losses = trades[trades.return_pct < 0] if len(trades) else trades
    gp = wins.return_pct.sum() if len(wins) else 0.0
    gl = -losses.return_pct.sum() if len(losses) else 0.0
    return {
        "final_equity": eq.iloc[-1],
        "total_return_pct": (e.iloc[-1] - 1) * 100,
        "cagr_pct": ((e.iloc[-1] ** (1 / yrs) - 1) * 100) if e.iloc[-1] > 0 else -100.0,
        "vol_pct": vol * 100,
        "sharpe": (r.mean() * bars_per_year) / vol if vol > 0 else np.nan,
        "max_dd_pct": (e / e.cummax() - 1).min() * 100,
        "trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100) if len(trades) else np.nan,
        "profit_factor": (gp / gl) if gl > 0 else np.nan,
        "avg_trade_pct": trades.return_pct.mean() if len(trades) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description="Supertrend SOXL/SOXS strategy")
    ap.add_argument("--symbol", default="SOXL")
    ap.add_argument("--tf", type=int, default=15, help="timeframe in minutes")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    # IMPORTANT: every Pine-input flag defaults to None. Only flags the user
    # actually typed override the Config dataclass. Giving these argparse
    # defaults would silently shadow edits made to Config above — editing
    # trailFactor in the dataclass would appear to do nothing.
    ap.add_argument("--mode", default=None,
                    choices=["Always Fixed", "Always Dynamic", "Auto-Switch via Vol"])
    ap.add_argument("--equity-frac", type=float, default=None)
    ap.add_argument("--commission-bps", type=float, default=None)
    ap.add_argument("--touch-fill", action="store_true", default=None,
                    help="use TradingView's touch-fill rule for limits")
    ap.add_argument("--no-date-filter", action="store_true", default=None)
    ap.add_argument("--atr-period", type=int, default=None)
    ap.add_argument("--factor", type=float, default=None)
    ap.add_argument("--risk-reward", type=float, default=None, help="percent, e.g. 4.5")
    ap.add_argument("--buyback", type=float, default=None, help="percent, e.g. 1.5")
    ap.add_argument("--trail-factor", type=float, default=None)
    ap.add_argument("--price-perc", type=float, default=None)
    ap.add_argument("--no-dynamic", action="store_true", default=None,
                    help="disable ADX dynamic trailing (uses trailFactor as-is)")
    ap.add_argument("--dividend-adjusted", action="store_true",
                    help="use total-return prices; default is SPLIT-only to match "
                         "TradingView charts")
    ap.add_argument("--out", default="C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports")
    a = ap.parse_args()

    over = {}
    if a.mode is not None:            over["strategyMode"] = a.mode
    if a.equity_frac is not None:     over["equity_frac"] = a.equity_frac
    if a.commission_bps is not None:  over["commission_bps"] = a.commission_bps
    if a.touch_fill is not None:      over["touch_fill"] = True
    if a.no_date_filter is not None:  over["useDateFilter"] = False
    if a.atr_period is not None:      over["atrPeriod"] = a.atr_period
    if a.factor is not None:          over["factor"] = a.factor
    if a.risk_reward is not None:     over["riskReward"] = a.risk_reward / 100.0
    if a.buyback is not None:         over["buyback"] = a.buyback / 100.0
    if a.trail_factor is not None:    over["trailFactor"] = a.trail_factor
    if a.price_perc is not None:      over["price_perc"] = a.price_perc
    if a.no_dynamic is not None:      over["useDynamic"] = False
    if a.start:                       over["startDate"] = a.start
    if a.end:                         over["endDate"] = a.end

    cfg = Config(**over)

    print("=" * 78)
    print(f"Supertrend Strategy — {a.symbol} @ {a.tf}min")
    print("=" * 78)
    print(f"mode={cfg.strategyMode} | atrPeriod={cfg.atrPeriod} factor={cfg.factor} "
          f"| RR={cfg.riskReward:.4f} buyback={cfg.buyback:.4f} trail={cfg.trailFactor} price_perc={cfg.price_perc}")
    print(f"useDynamic={cfg.useDynamic} | dateFilter={cfg.useDateFilter} | prices={'dividend-adj' if a.dividend_adjusted else 'split-only'}")
    print(f"sizing={cfg.equity_frac:.0%} equity | {cfg.commission_bps}bp/leg | "
          f"{'touch' if cfg.touch_fill else 'pierce'} fills")
    print(f"alert routing: {cfg.alertRouting}\n")

    panel = load_alpaca_bars([a.symbol], a.tf, a.start, a.end,
                             dividend_adjusted=a.dividend_adjusted)
    bars = to_ohlc(panel, a.symbol)
    bpd = round(390 / a.tf)
    bpy = bpd * 252
    print(f"bars: {len(bars):,} | {bars.index[0].date()} -> {bars.index[-1].date()} "
          f"| {bpd}/day\n")

    eq, trades = SupertrendStrategy(cfg).run(bars)
    s = summarize(eq, trades, bpy)

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  Final equity     ${s['final_equity']:>18,.2f}")
    print(f"  Total return     {s['total_return_pct']:>18,.2f}%")
    print(f"  CAGR             {s['cagr_pct']:>18,.2f}%")
    print(f"  Volatility       {s['vol_pct']:>18,.2f}%")
    print(f"  Sharpe           {s['sharpe']:>18,.2f}")
    print(f"  Max drawdown     {s['max_dd_pct']:>18,.2f}%")
    print(f"  Trades           {s['trades']:>18,}")
    print(f"  Win rate         {s['win_rate_pct']:>18,.2f}%")
    print(f"  Profit factor    {s['profit_factor']:>18,.3f}")
    print(f"  Avg trade        {s['avg_trade_pct']:>18,.3f}%")

    os.makedirs(a.out, exist_ok=True)
    tp = os.path.join(a.out, f"supertrend_{a.symbol}_{a.tf}m_trades.csv")
    ep = os.path.join(a.out, f"supertrend_{a.symbol}_{a.tf}m_equity.csv")
    trades.to_csv(tp, index=False)
    eq.to_csv(ep)
    print(f"\n  trade log: {tp}")
    print(f"  equity:    {ep}")

    if len(trades):
        print("\n  last 3 trades (with webhook payloads):")
        for _, t in trades.tail(3).iterrows():
            print(f"    {t.entry_time} {t.side:<5} @ {t.price:8.2f} -> "
                  f"{t.exit_price:8.2f}  {t.return_pct:+6.2f}%  [{t.exit_reason}]")
            print(f"      entry alert: {t.entry_alert}")
            print(f"      exit  alert: {t.exit_alert}")


if __name__ == "__main__":
    main()
