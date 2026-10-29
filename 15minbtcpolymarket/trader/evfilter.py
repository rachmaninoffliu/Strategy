"""EV-regime filter (Qt-free): trade an hour of the day only while that hour's TRAILING net EV clears a threshold.

A "regime" is a UTC hour of the day (the same hours the Regime radar lists).  An hour's trailing net EV is what the radar's 'Net EV 14 d' / 'Net EV 28 d' columns show: the mean net P&L per contract
(win - price - slippage - fee) of that hour's underdog signals that SETTLED before now and opened within the last N days.  A rule is "N d net EV > T"; several rules must ALL hold.  An hour with
fewer than ``min_n`` signals in a rule's window has no usable EV there and does not trade.

The class is a drop-in for ``SignalStats`` (``p`` / ``n_total`` / ``add_live``), so the backtest and the live engine run the very same code:
  * hour passes  -> p = price + the hour's trailing GROSS edge (the smaller of the rules' windows), so Kelly sizes the bet off what the rule just verified;
  * hour fails   -> p = price: zero edge, Kelly sizes to nothing (no trade), the signal is still recorded and learned from.
Only signals settled before the entry count, so the backtest has no look-ahead.  Preset names: 'FILTER-14', 'FILTER-28', 'FILTER-BOTH' (trader.presets).
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from .fees import fee_per_contract

WINDOW = 900
DEFAULT_MIN_N = 8


def describe_rules(rules: tuple) -> str:
    return " AND ".join(f"{d} d net EV > {t * 100:g}c" for d, t in rules) if rules else "off"


class TrailingEVStats:
    def __init__(self, rules: tuple, min_n: int = DEFAULT_MIN_N, slippage: float = 0.02, fee_model: str = "Kalshi"):
        if not rules:
            raise ValueError("an EV filter needs at least one rule")
        self.rules = tuple((int(d), float(t)) for d, t in rules)
        self.min_n, self.slip, self.fee_model = int(min_n), float(slippage), fee_model
        self._rec: dict = defaultdict(list)                 # hour -> sorted [(close_ts, net, gross)]
        self._ct: dict = defaultdict(list)                  # hour -> close_ts (sorted)
        self._cn: dict = defaultdict(lambda: [0.0])         # hour -> prefix sums of net
        self._cg: dict = defaultdict(lambda: [0.0])         # hour -> prefix sums of gross
        self._leg: dict = defaultdict(list)                 # leg -> close_ts (sorted), for n_total
        self.last_hist_open = 0

    # ── building ──
    def _net(self, win: float, cost: float) -> float:
        px = min(float(cost) + self.slip, 0.99)
        return float(win) - px - fee_per_contract(self.fee_model, px)

    def add(self, leg: str, hour: int, close_ts: int, win: int, cost: float) -> None:
        h, row = int(hour), (int(close_ts), self._net(win, cost), float(win) - float(cost))
        ct = self._ct[h]
        insort(self._leg[leg], row[0])
        if not ct or row[0] >= ct[-1]:                        # in order (always, apart from rare late results): O(1)
            self._rec[h].append(row); ct.append(row[0])
            self._cn[h].append(self._cn[h][-1] + row[1]); self._cg[h].append(self._cg[h][-1] + row[2])
            return
        insort(self._rec[h], row)                             # out of order: rebuild this hour
        self._ct[h] = [r[0] for r in self._rec[h]]
        cn, cg = [0.0], [0.0]
        for _t, n, g in self._rec[h]:
            cn.append(cn[-1] + n); cg.append(cg[-1] + g)
        self._cn[h], self._cg[h] = cn, cg

    def add_live(self, leg: str, hour: int, open_ts: int, win: bool, cost: float, **_kw) -> None:
        """A signal that settled while trading (ignored if the local history file already contains it)."""
        if open_ts > self.last_hist_open:
            self.add(leg, hour, int(open_ts) + WINDOW, int(win), cost)

    @classmethod
    def from_signals(cls, sig: pd.DataFrame, rules: tuple, min_n: int = DEFAULT_MIN_N, slippage: float = 0.02, fee_model: str = "Kalshi") -> "TrailingEVStats":
        s = cls(rules, min_n, slippage, fee_model)
        if len(sig):
            d = sig.sort_values("close_ts")
            for leg, hour, ct, w, c in zip(d.leg.values, d.hour.values, d.close_ts.values, d.win.values, d.cost.values):
                s.add(str(leg), int(hour), int(ct), int(w), float(c))
            s.last_hist_open = int(sig.open_ts.max())
        return s

    # ── querying ──
    def window(self, hour: int, ts: float, days: int) -> tuple[int, float, float]:
        """(n, mean net EV, mean gross edge) of the hour's signals that settled by ``ts`` and opened within the last ``days`` days (the radar's trailing window)."""
        ct = self._ct.get(int(hour))
        if not ct:
            return 0, float("nan"), float("nan")
        hi = bisect_right(ct, ts)
        lo = min(bisect_left(ct, ts - days * 86400 + WINDOW), hi)          # open_ts = close_ts - 900 >= ts - days
        n = hi - lo
        if n <= 0:
            return 0, float("nan"), float("nan")
        return n, (self._cn[int(hour)][hi] - self._cn[int(hour)][lo]) / n, (self._cg[int(hour)][hi] - self._cg[int(hour)][lo]) / n

    def check(self, hour: int, ts: float) -> tuple[bool, float, list]:
        """(passes, gross edge the bet is sized with, [(days, n, mean net EV, threshold, ok), ...])."""
        parts, ok, gross = [], True, float("inf")
        for days, thr in self.rules:
            n, net, g = self.window(hour, ts, days)
            good = n >= self.min_n and net > thr
            parts.append((days, n, net, thr, good))
            ok &= good
            if n:
                gross = min(gross, g)
        return ok, (gross if ok else float("nan")), parts

    def passes(self, hour: int, ts: float) -> bool:
        return self.check(hour, ts)[0]

    def explain(self, hour: int, ts: float) -> str:
        ok, _g, parts = self.check(hour, ts)
        bits = [f"{d} d EV {('%+.1f' % (net * 100)) + 'c' if n else 'n/a'} (n={n}) {'>' if good else 'not >'} {thr * 100:g}c" + (f", needs n>={self.min_n}" if n < self.min_n else "") for d, n, net, thr, good in parts]
        return f"EV filter, hour {int(hour):02d} UTC: " + "; ".join(bits) + (" -> TRADE" if ok else " -> no trade")

    def p(self, leg: str, hour: int, cost: float, ts: float) -> tuple[float, int, int]:
        """(win probability to size with, signals of this leg settled by ts, signals in the hour's longest window)."""
        n_leg = bisect_right(self._leg.get(leg, []), ts)
        ok, gross, parts = self.check(hour, ts)
        nh = max((n for _d, n, _e, _t, _g in parts), default=0)
        return (min(max(float(cost) + gross, 0.0), 1.0) if ok else float(cost)), n_leg, nh

    def n_total(self, entry_ts: float, leg: Optional[str] = None) -> int:
        legs = [leg] if leg else list(self._leg)
        return sum(bisect_right(self._leg.get(lg, []), entry_ts) for lg in legs)

    def passing_hours(self, ts: float) -> list:
        return [h for h in range(24) if self.passes(h, ts)]
