"""Historical win-probability model for Kelly sizing.

Every qualifying signal counts as an observation (traded or not), per (leg, UTC hour). Two bases:

  'win_rate'  raw historical win rate        p0 = (wins + c*n0) / (n + n0)       c = this signal's market-implied price
  'edge'      price-adjusted historical edge d0 = sum(win - cost_i) / (n + n0)    p0 = c + d0
              (a trade priced at 6c is NOT credited with the win rate of 25c contracts; the edge shrinks toward 0 = skeptical)

Hour level (mode 'by_hour') shrinks the (leg, hour) estimate toward the leg estimate with strength k:
  win_rate: p1 = (wins_h + k*p0)/(n_h + k)       edge: d1 = (E_h + k*d0)/(n_h + k),  p1 = c + d1
Only signals that settled BEFORE the entry are used, so the backtest has no look-ahead.

Core hours: in 'by_hour' mode the hours in ``core_hours`` can use a smaller shrinkage ``core_k`` (they carry an informative prior: the hours you chose to trade),
while every other hour keeps the skeptical ``hour_k``. The engine then trades the core quickly and still discovers - more slowly - an edge that moves to another hour.

Memory: ``lookback_days`` = hard window (all signals in the last N days count equally); ``halflife_days`` > 0 = exponential decay instead
(a signal's weight halves every ``halflife_days``; the prior ``prior_n`` and the hour-shrinkage ``hour_k`` are then in effective-sample units).
Decay reacts faster than a hard window at both ends of a regime: it enters sooner and, more importantly, stops betting sooner when the edge disappears.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Optional

import pandas as pd

MODES = ("overall", "by_hour")
EPOCH0 = 1_764_547_200          # 2025-12-01: reference time for the decay weights (keeps exp() arguments small)
BASES = ("win_rate", "edge")


class SignalStats:
    def __init__(self, mode: str = "overall", hour_k: float = 40.0, prior_n: float = 20.0, lookback_days: int = 0,
                 basis: str = "edge", halflife_days: float = 0.0, core_hours: tuple = (), core_k: float = 0.0, near_k: float = 0.0, near_width: int = 2):
        if mode not in MODES or basis not in BASES:
            raise ValueError((mode, basis))
        if halflife_days and halflife_days < 3:
            raise ValueError("halflife_days must be 0 (off) or >= 3")
        self.mode, self.k, self.n0, self.lookback, self.basis = mode, float(hour_k), float(prior_n), int(lookback_days), basis
        self.hl = float(halflife_days)
        self.core = frozenset(int(h) for h in core_hours) if core_k and core_k > 0 else frozenset()
        self.core_k = float(core_k or 0.0)
        self.near_k = float(near_k or 0.0)                       # hours within ``near_width`` of a core hour (circular) use this lighter shrinkage than the far scout hours: a moving regime most plausibly moves next door
        self.near = (frozenset((c + d) % 24 for c in self.core for d in range(-int(near_width), int(near_width) + 1)) - self.core) if (self.near_k > 0 and self.core) else frozenset()
        self.lam = math.log(2.0) / (self.hl * 86400.0) if self.hl > 0 else 0.0
        self._dn: dict = defaultdict(lambda: [0.0])   # decayed prefix sums (weights exp(lam*(t-EPOCH0))): counts
        self._dw: dict = defaultdict(lambda: [0.0])   #                                                       wins
        self._de: dict = defaultdict(lambda: [0.0])   #                                                       (win - cost)
        self._t: dict = defaultdict(list)             # key -> close_ts (sorted)
        self._w: dict = defaultdict(lambda: [0.0])    # key -> prefix sums of wins
        self._e: dict = defaultdict(lambda: [0.0])    # key -> prefix sums of (win - cost)
        self.last_hist_open = 0

    # ── building ──
    def _append(self, key, close_ts: int, win: float, excess: float) -> None:
        t = self._t[key]
        if t and close_ts < t[-1]:                    # out-of-order (rare): rebuild this key in order
            w, e = self._w[key], self._e[key]
            rows = [(t[i], w[i + 1] - w[i], e[i + 1] - e[i]) for i in range(len(t))] + [(close_ts, win, excess)]
            rows.sort()
            self._t[key] = [r[0] for r in rows]
            cw, ce, dn, dw, de = [0.0], [0.0], [0.0], [0.0], [0.0]
            for ts, a, b in rows:
                cw.append(cw[-1] + a)
                ce.append(ce[-1] + b)
                if self.lam:
                    s = math.exp(self.lam * (ts - EPOCH0))
                    dn.append(dn[-1] + s); dw.append(dw[-1] + s * a); de.append(de[-1] + s * b)
            self._w[key], self._e[key] = cw, ce
            if self.lam:
                self._dn[key], self._dw[key], self._de[key] = dn, dw, de
            return
        t.append(close_ts)
        self._w[key].append(self._w[key][-1] + win)
        self._e[key].append(self._e[key][-1] + excess)
        if self.lam:
            s = math.exp(self.lam * (close_ts - EPOCH0))
            self._dn[key].append(self._dn[key][-1] + s)
            self._dw[key].append(self._dw[key][-1] + s * win)
            self._de[key].append(self._de[key][-1] + s * excess)

    def add(self, leg: str, hour: int, close_ts: int, win: int, cost: float) -> None:
        ex = float(win) - float(cost)
        self._append((leg, None), close_ts, float(win), ex)
        self._append((leg, int(hour)), close_ts, float(win), ex)

    def add_live(self, leg: str, hour: int, open_ts: int, win: bool, cost: float) -> None:
        """A signal settled while trading; ignored if the local history file already contains it."""
        if open_ts > self.last_hist_open:
            self.add(leg, hour, int(open_ts) + 900, int(win), cost)

    @classmethod
    def from_signals(cls, sig: pd.DataFrame, **kw) -> "SignalStats":
        s = cls(**kw)
        if len(sig):
            d = sig.sort_values("close_ts")
            for leg, hour, ct, w, c in zip(d.leg.values, d.hour.values, d.close_ts.values, d.win.values, d.cost.values):
                s.add(str(leg), int(hour), int(ct), int(w), float(c))
            s.last_hist_open = int(sig.open_ts.max())
        return s

    # ── querying ──
    def count(self, key, entry_ts: float) -> tuple[int, float, float]:
        """(n, wins, sum of (win - cost)) among signals settled before entry_ts (and inside the lookback window)."""
        t = self._t.get(key)
        if not t:
            return 0, 0.0, 0.0
        hi = bisect_right(t, entry_ts)
        lo = bisect_left(t, entry_ts - self.lookback * 86400) if self.lookback > 0 else 0
        lo = min(lo, hi)
        w, e = self._w[key], self._e[key]
        return hi - lo, w[hi] - w[lo], e[hi] - e[lo]

    def cell_mass(self, leg: str, hour: int, entry_ts: float) -> float:
        """Effective number of observations behind the (leg, hour) estimate: its own signals plus the shrinkage prior it leans on (used for its standard error)."""
        if self.lam:
            if self.mode == "by_hour":
                return self.decayed((leg, int(hour)), entry_ts)[0] + self.k_for(hour)
            return self.decayed((leg, None), entry_ts)[0] + self.n0
        if self.mode == "by_hour":
            return self.count((leg, int(hour)), entry_ts)[0] + self.k_for(hour)
        return self.count((leg, None), entry_ts)[0] + self.n0

    def k_for(self, hour: int) -> float:
        h = int(hour)
        return self.core_k if h in self.core else self.near_k if h in self.near else self.k

    def decayed(self, key, entry_ts: float) -> tuple[float, float, float]:
        """(effective n, effective wins, effective sum of (win - cost)) of signals settled before entry_ts, each weighted 0.5 ** (age / half-life)."""
        t = self._t.get(key)
        if not t:
            return 0.0, 0.0, 0.0
        hi = bisect_right(t, entry_ts)
        sc = math.exp(-self.lam * (entry_ts - EPOCH0))
        return self._dn[key][hi] * sc, self._dw[key][hi] * sc, self._de[key][hi] * sc

    def p(self, leg: str, hour: int, cost: float, entry_ts: float) -> tuple[float, int, int]:
        """(probability, n at leg level, n at hour level); n are raw counts even when decay is on (the estimate itself uses effective counts)."""
        n, w, e = self.count((leg, None), entry_ts)
        nh = n
        if self.lam:
            ne, we, ee = self.decayed((leg, None), entry_ts)
            if self.basis == "edge":
                d = ee / (ne + self.n0)
                if self.mode == "by_hour":
                    nh, _wh, _eh = self.count((leg, int(hour)), entry_ts)
                    nhe, _whe, ehe = self.decayed((leg, int(hour)), entry_ts)
                    kk = self.k_for(hour)
                    d = (ehe + kk * d) / (nhe + kk)
                return min(max(cost + d, 0.0), 1.0), n, nh
            p0 = (we + cost * self.n0) / (ne + self.n0)
            if self.mode == "by_hour":
                nh, _wh, _eh = self.count((leg, int(hour)), entry_ts)
                nhe, whe, _ehe = self.decayed((leg, int(hour)), entry_ts)
                kk = self.k_for(hour)
                p0 = (whe + kk * p0) / (nhe + kk)
            return p0, n, nh
        if self.basis == "edge":
            d = e / (n + self.n0)
            if self.mode == "by_hour":
                nh, _wh, eh = self.count((leg, int(hour)), entry_ts)
                kk = self.k_for(hour)
                d = (eh + kk * d) / (nh + kk)
            return min(max(cost + d, 0.0), 1.0), n, nh
        p0 = (w + cost * self.n0) / (n + self.n0)
        if self.mode == "by_hour":
            nh, wh, _eh = self.count((leg, int(hour)), entry_ts)
            kk = self.k_for(hour)
            p0 = (wh + kk * p0) / (nh + kk)
        return p0, n, nh

    def n_total(self, entry_ts: float, leg: Optional[str] = None) -> int:
        legs = [leg] if leg else [k[0] for k in self._t if k[1] is None]
        return sum(self.count((lg, None), entry_ts)[0] for lg in legs)
