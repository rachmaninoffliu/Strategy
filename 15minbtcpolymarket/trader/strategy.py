"""Strategy rules (underdog and/or favourite legs) + Kelly position sizing. Pure functions, shared by backtest and live."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .fees import fee_per_contract, fee_total

ALL_HOURS = tuple(range(24))
MIN_HISTORY = 30              # refuse to trade a leg on fewer historical signals than this (one number for the backtest and the live engine)


def parse_int_list(text: str, lo: int = 0, hi: int = 23) -> tuple[int, ...]:
    """'22,23' or '20-23' or '20-21,23' -> (20,21,23). Raises ValueError on garbage."""
    out: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    if not out or min(out) < lo or max(out) > hi:
        raise ValueError(f"values must be in {lo}..{hi}: {text!r}")
    return tuple(sorted(out))


def fmt_int_list(v: tuple) -> str:
    """(0,1,2,3,22,23) -> '0-3,22,23'"""
    v = sorted(v)
    out, i = [], 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[j] + 1:
            j += 1
        out.append(str(v[i]) if i == j else f"{v[i]}-{v[j]}")
        i = j + 1
    return ",".join(out)


@dataclass(frozen=True)
class StrategyParams:
    """A leg is a rule 'buy this side at the decision minute if its ask is inside [lo, hi]; hold to settlement'.
      DOG  the underdog (YES if mid < 0.5 else NO)   priced cost_min..cost_max      -> low win rate, large payoff
      FAV  the favourite                              priced fav_min..fav_max         -> high win rate, small payoff
    One signal per leg per market (its first qualifying decision minute); at most ONE trade per 15-minute window."""
    hours: tuple = (22, 23)              # UTC hour of the window OPEN
    minutes: tuple = (12, 13)            # decision minute(s): quote at open + 60*m seconds
    cost_min: float = 0.05
    cost_max: float = 0.30
    max_spread: float = 0.06             # ignore books wider than this (stale / empty)
    legs: tuple = ("DOG",)
    fav_min: float = 0.85
    fav_max: float = 0.97

    def describe(self) -> str:
        parts = []
        if "DOG" in self.legs:
            parts.append(f"underdog {self.cost_min:.2f}-{self.cost_max:.2f}")
        if "FAV" in self.legs:
            parts.append(f"favourite {self.fav_min:.2f}-{self.fav_max:.2f}")
        return f"{' + '.join(parts)} | hours UTC {fmt_int_list(self.hours)} | minutes {fmt_int_list(self.minutes)}"


@dataclass(frozen=True)
class Signal:
    side: str            # 'YES' or 'NO' (the side we BUY)
    cost: float          # quoted ask of that side
    bid: float           # YES bid
    ask: float           # YES ask
    minute: int
    leg: str = "DOG"


def evaluate_quote_all(bid: float, ask: float, minute: int, hour: int, p: StrategyParams) -> list[Signal]:
    """YES bid/ask at a decision instant -> the signal of every enabled leg that qualifies (same rule the backtest applies)."""
    if hour not in p.hours or minute not in p.minutes:
        return []
    if not (bid > 0 and ask < 1 and ask >= bid and (ask - bid) <= p.max_spread + 1e-9):
        return []
    mid = (bid + ask) / 2
    yes_is_dog = mid < 0.5
    out: list[Signal] = []
    if "DOG" in p.legs:
        side, cost = ("YES", ask) if yes_is_dog else ("NO", 1.0 - bid)
        if p.cost_min - 1e-9 <= cost <= p.cost_max + 1e-9:
            out.append(Signal(side, round(cost, 6), bid, ask, minute, "DOG"))
    if "FAV" in p.legs:
        side, cost = ("NO", 1.0 - bid) if yes_is_dog else ("YES", ask)
        if p.fav_min - 1e-9 <= cost <= p.fav_max + 1e-9:
            out.append(Signal(side, round(cost, 6), bid, ask, minute, "FAV"))
    return out


def evaluate_quote(bid: float, ask: float, minute: int, hour: int, p: StrategyParams) -> Optional[Signal]:
    s = evaluate_quote_all(bid, ask, minute, hour, p)
    return s[0] if s else None


# ── Kelly ─────────────────────────────────────────────────────────────────────
def kelly_full(p: float, cost_all: float) -> float:
    """Full-Kelly fraction for a contract that costs ``cost_all`` (price + fee) and pays $1 w.p. ``p``:
    f* = (p - c) / (1 - c).  Zero when p <= c."""
    if not (0.0 < cost_all < 1.0):
        return 0.0
    return max(0.0, (p - cost_all) / (1.0 - cost_all))


def drawdown_throttle(dd: float) -> float:
    """Kelly-multiplier scale from the current drawdown below the equity peak (dd >= 0): full size until -8%, half to -15%, quarter beyond.
    A regime that ends costs the learner several days of stale evidence; this caps what those days can cost."""
    return 0.25 if dd >= 0.15 else 0.5 if dd >= 0.08 else 1.0


THROTTLE_SCALES_CAP = True      # the throttle shrinks the stake cap as well as the Kelly multiplier; with only the multiplier scaled, a binding cap (e.g. 1.0x Kelly / 5%) made "half size" a no-op


def throttled_sizing(mult: float, cap: float, equity: float, peak: float, on: bool) -> tuple[float, float]:
    """(Kelly multiplier, stake cap) after the drawdown throttle - the ONE place both the backtest and the live engine size through."""
    thr = drawdown_throttle(1.0 - equity / peak) if on and peak > 0 else 1.0
    return mult * thr, cap * (thr if THROTTLE_SCALES_CAP else 1.0)


@dataclass
class Sizing:
    contracts: int
    f_full: float        # full-Kelly fraction (of bankroll)
    f_used: float        # after multiplier and cap
    stake: float         # dollars at risk incl. fee
    cost_all: float      # per-contract all-in cost used for Kelly
    edge: float          # p - cost_all
    reason: str = ""


def size_bet(bankroll: float, p: float, price: float, fee_model: str, mult: float, cap: float,
             max_contracts: int, min_edge: float) -> Sizing:
    """Kelly stake -> whole contracts. ``price`` is the expected fill price (incl. any slippage)."""
    ref = fee_per_contract(fee_model, price)
    cost_all = price + ref
    edge = p - cost_all
    if bankroll <= 0:
        return Sizing(0, 0.0, 0.0, 0.0, cost_all, edge, "no bankroll")
    if edge < min_edge:
        return Sizing(0, kelly_full(p, cost_all), 0.0, 0.0, cost_all, edge, f"edge {edge * 100:+.2f}c < min {min_edge * 100:.2f}c")
    f_full = kelly_full(p, cost_all)
    f_used = min(mult * f_full, cap)
    budget = f_used * bankroll
    n = int(min(math.floor(budget / cost_all), max_contracts))
    while n > 0 and n * price + fee_total(fee_model, n, price) > budget + 1e-9:
        n -= 1
    if n < 1:
        return Sizing(0, f_full, f_used, 0.0, cost_all, edge, f"stake ${budget:.2f} < 1 contract")
    stake = n * price + fee_total(fee_model, n, price)
    return Sizing(n, f_full, f_used, stake, cost_all, edge)
