"""Adaptive ("Auto") layer - the strategy detects the regime and sets its own memory and size at the moment of each trade.  Qt-free; used identically by the backtest and the live engine.

What adapts, and why each piece is conservative:

  1. MEMORY (regime detection)  Several learners of the SAME edge with different half-lives (default 7 / 14 / 30 days) run side by side.  Their predictions are averaged with weights
     that follow how well each one predicted the signals that have settled recently (Bayesian-model-averaging style; floor on every weight).  When the regime is stable the slow learners
     win (less noise); at a regime change the fast one wins (reacts in days).  No parameter is "selected by best past P&L" - an approach that failed out of sample in earlier tests.
  2. REGIME LABEL               fast edge vs slow edge, both after an evidence haircut: HOT (both confirm), BUILDING (only the fast one confirms - new regime, thin evidence),
     FADING (the slow one still confirms but the fast one has dropped), COLD (neither: no trade).  The label scales the size.
  3. EVIDENCE-AWARE KELLY       Kelly uses the edge minus z standard errors (z = 0.25): thin evidence = smaller bet.
  4. EQUITY LADDER              Kelly multiplier and stake cap come from a table keyed by the account's current equity (bigger account -> smaller fraction), then are scaled by the regime
     label and by the drawdown throttle.
  5. LIQUIDITY AT TRADE TIME    Max contracts = participation x the previous minute's traded volume (capped by an absolute ceiling) - the order size adapts to how liquid the market is right now.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .fees import fee_per_contract
from .probmodel import EPOCH0, SignalStats
from .strategy import Sizing, drawdown_throttle, size_bet

REGIMES = ("HOT", "BUILDING", "FADING", "COLD")
DEFAULT_LADDER = ((0.0, 0.50, 0.03), (25_000.0, 0.35, 0.025), (100_000.0, 0.25, 0.02), (500_000.0, 0.15, 0.01))     # (equity >=, Kelly multiplier, stake cap as a fraction of equity)
SIGMA2 = 0.20                                  # variance scale of a win/lose outcome used for the learner weights


@dataclass(frozen=True)
class AutoConfig:
    enabled: bool = True
    half_lives: tuple = (7.0, 14.0, 30.0)      # the learners' memories, fastest first
    mix_tau: float = 3.0                       # tempering of the learner weights (higher = closer to equal weights)
    mix_halflife_days: float = 21.0            # memory of the predictive-error tracker
    mix_floor: float = 0.05                    # every learner keeps at least this weight
    z: float = 0.25                            # evidence haircut: edge - z * standard error
    fade_ratio: float = 0.6                    # fast edge below this share of the slow edge = FADING
    regime_mult: tuple = (("HOT", 1.0), ("BUILDING", 0.5), ("FADING", 0.4), ("COLD", 0.0))
    ladder: tuple = DEFAULT_LADDER
    participation: float = 0.10                # max order = this share of the previous minute's traded contracts (0 = ignore liquidity)
    max_contracts: int = 200_000               # absolute ceiling
    throttle: bool = True                      # drawdown throttle (scales BOTH the multiplier and the cap here)

    def regime_scale(self, regime: str) -> float:
        return dict(self.regime_mult).get(regime, 0.0)

    def tier(self, equity: float) -> tuple[int, float, float]:
        """(index, Kelly multiplier, stake cap fraction) for the account's current equity."""
        k = 0
        for i, (thr, _m, _c) in enumerate(self.ladder):
            if equity >= thr:
                k = i
        return k, self.ladder[k][1], self.ladder[k][2]


def parse_ladder(text: str) -> tuple:
    """'0:0.5/3, 25000:0.35/2.5, 100000:0.25/2' -> ((0, .5, .03), (25000, .35, .025), ...);  value = equity threshold : Kelly multiplier / stake cap in %."""
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            thr, rest = part.split(":")
            mult, cap = rest.split("/")
            out.append((float(thr.replace("_", "")), float(mult), float(cap) / 100.0))
        except ValueError:
            raise ValueError(f"bad ladder step '{part}' (use  equity:multiplier/stake%  e.g. 25000:0.35/2.5)") from None
    if not out:
        raise ValueError("the equity ladder needs at least one step")
    out.sort()
    if out[0][0] > 0:
        out.insert(0, (0.0, out[0][1], out[0][2]))
    for thr, m, c in out:
        if not (0 < m <= 1.0) or not (0 < c <= 0.5):
            raise ValueError(f"ladder step {thr:g}: Kelly multiplier must be in (0, 1] and the stake cap in (0, 50]%")
    return tuple(out)


def format_ladder(ladder: tuple) -> str:
    return ", ".join(f"{thr:g}:{m:g}/{c * 100:g}" for thr, m, c in ladder)


def describe_regime_scale(auto: AutoConfig) -> str:
    return " · ".join(f"{r} {auto.regime_scale(r):g}" if auto.regime_scale(r) > 0 else f"{r} no trade" for r in REGIMES)


def describe_liquidity(auto: AutoConfig) -> str:
    return (f"{auto.participation * 100:g}% of the last minute's traded contracts, ceiling {auto.max_contracts:,}" if auto.participation > 0 else f"ignored, ceiling {auto.max_contracts:,}")


@dataclass
class Prediction:
    p_used: float          # probability used for sizing (price + haircut edge)
    d_mix: float           # weighted edge estimate (gross of costs)
    se: float              # its standard error
    d_fast: float
    d_slow: float
    se_fast: float
    se_slow: float
    weights: tuple         # learner weights, fastest first
    n: int                 # signals behind the slow learner at leg level
    nh: int                # ... at hour level
    cost: float
    vec: tuple             # each learner's predicted win probability (for the error tracker)


class AdaptiveStats:
    """Drop-in for SignalStats (p / n_total / add_live / count) averaging several memories; ``predict`` exposes the pieces the Auto sizing needs."""

    def __init__(self, auto: AutoConfig, **learner_kw):
        self.auto = auto
        learner_kw = dict(learner_kw)
        learner_kw.pop("halflife_days", None)
        learner_kw["lookback_days"] = 0
        self.learners = [SignalStats(halflife_days=float(h), **learner_kw) for h in auto.half_lives]
        self.lam = math.log(2.0) / (auto.mix_halflife_days * 86400.0)
        self._ct: list[float] = []                            # close_ts of tracked signals (sorted)
        self._cum: list[list[float]] = [[0.0] for _ in self.learners]
        self.last_hist_open = 0

    # ── building ──
    @classmethod
    def from_signals(cls, sig: pd.DataFrame, auto: AutoConfig, **kw) -> "AdaptiveStats":
        s = cls(auto, **kw)
        s.learners = [SignalStats.from_signals(sig, halflife_days=float(h), **{**{k: v for k, v in kw.items() if k != "halflife_days"}, "lookback_days": 0}) for h in auto.half_lives]
        if len(sig):
            d = sig.sort_values("close_ts")
            for leg, hour, ot, mn, ct, w, c in zip(d.leg.values, d.hour.values, d.open_ts.values, d.minute.values, d.close_ts.values, d.win.values, d.cost.values):
                vec = s._vec(str(leg), int(hour), float(c), float(ot) + 60.0 * int(mn))
                s._track(float(ct), int(w), vec)
            s.last_hist_open = int(sig.open_ts.max())
        return s

    def _vec(self, leg: str, hour: int, cost: float, entry_ts: float) -> tuple:
        return tuple(L.p(leg, hour, cost, entry_ts)[0] for L in self.learners)

    def _track(self, close_ts: float, win: int, vec: tuple) -> None:
        sc = math.exp(self.lam * (close_ts - EPOCH0))
        if self._ct and close_ts < self._ct[-1]:               # out of order (rare): rebuild in order
            old = [(self._ct[i], [self._cum[k][i + 1] - self._cum[k][i] for k in range(len(self.learners))]) for i in range(len(self._ct))]
            old.append((close_ts, [sc * (win - p) ** 2 for p in vec]))
            old.sort(key=lambda r: r[0])
            self._ct = [r[0] for r in old]
            self._cum = [[0.0] for _ in self.learners]
            for _t, errs in old:
                for k, e in enumerate(errs):
                    self._cum[k].append(self._cum[k][-1] + e)
            return
        self._ct.append(close_ts)
        for k, p in enumerate(vec):
            self._cum[k].append(self._cum[k][-1] + sc * (win - p) ** 2)

    def add_live(self, leg: str, hour: int, open_ts: int, win: bool, cost: float, minute: int = 12) -> None:
        """A signal settled while trading (ignored if the local history already contains it)."""
        if open_ts <= self.last_hist_open:
            return
        vec = self._vec(leg, int(hour), float(cost), float(open_ts) + 60.0 * minute)       # point-in-time: signals that settled before this entry
        for L in self.learners:
            L.add_live(leg, hour, open_ts, win, cost)
        self._track(float(open_ts) + 900.0, int(win), vec)

    # ── querying ──
    def weights(self, t: float) -> tuple:
        K = len(self.learners)
        hi = bisect_right(self._ct, t)
        if hi == 0 or K == 1:
            return tuple([1.0 / K] * K)
        sc = math.exp(-self.lam * (t - EPOCH0))
        S = [self._cum[k][hi] * sc for k in range(K)]
        smin = min(S)
        w = np.array([math.exp(-(s - smin) / (2.0 * SIGMA2 * self.auto.mix_tau)) for s in S])
        w = w / w.sum()
        w = np.maximum(w, self.auto.mix_floor)
        return tuple((w / w.sum()).tolist())

    def predict(self, leg: str, hour: int, cost: float, entry_ts: float) -> Prediction:
        ps, ds, ses = [], [], []
        n = nh = 0
        for i, L in enumerate(self.learners):
            p, n_, nh_ = L.p(leg, hour, cost, entry_ts)
            if i == len(self.learners) - 1:
                n, nh = n_, nh_
            ps.append(p)
            ds.append(p - cost)
            ses.append(math.sqrt(max(p * (1.0 - p), 0.05) / max(L.cell_mass(leg, hour, entry_ts), 1.0)))
        w = self.weights(entry_ts)
        d_mix = float(sum(wi * di for wi, di in zip(w, ds)))
        se = float(sum(wi * si for wi, si in zip(w, ses)))
        p_used = min(max(cost + d_mix - self.auto.z * se, 0.0), 1.0)
        return Prediction(p_used=p_used, d_mix=d_mix, se=se, d_fast=ds[0], d_slow=ds[-1], se_fast=ses[0], se_slow=ses[-1], weights=w, n=n, nh=nh, cost=cost, vec=tuple(ps))

    def p(self, leg: str, hour: int, cost: float, entry_ts: float) -> tuple[float, int, int]:
        pr = self.predict(leg, hour, cost, entry_ts)
        return pr.p_used, pr.n, pr.nh

    def count(self, key, entry_ts: float):
        return self.learners[-1].count(key, entry_ts)

    def n_total(self, entry_ts: float, leg: Optional[str] = None) -> int:
        return self.learners[-1].n_total(entry_ts, leg)


def classify_regime(pred: Prediction, costs: float, hurdle: float, auto: AutoConfig) -> str:
    """HOT / BUILDING / FADING / COLD from the fast and the slow learner (both after the evidence haircut and costs)."""
    nf = pred.d_fast - auto.z * pred.se_fast - costs
    ns = pred.d_slow - auto.z * pred.se_slow - costs
    if nf >= hurdle and ns >= hurdle:
        return "FADING" if pred.d_fast < auto.fade_ratio * pred.d_slow else "HOT"
    if nf >= hurdle:
        return "BUILDING"
    if ns >= hurdle:
        return "FADING"
    return "COLD"


def snapshot(auto: AutoConfig, stats: AdaptiveStats, leg: str, hour: int, cost: float, entry_ts: float, equity: float, peak: float, fee_model: str, slippage: float, min_edge: float) -> dict:
    """What the Auto layer would do right now for a typical signal in ``hour`` (for the status strip), plus which hours currently look tradable."""
    d = decide(auto, stats, leg, hour, cost, entry_ts, equity, peak, equity, None, fee_model, slippage, min_edge)
    pr = d.pred
    live_hours = []
    for h in range(24):
        dh = decide(auto, stats, leg, h, cost, entry_ts, equity, peak, equity, None, fee_model, slippage, min_edge)
        if dh.regime != "COLD":
            live_hours.append((h, dh.regime))
    return dict(regime=d.regime, mult=d.mult, cap=d.cap, tier=d.tier, max_contracts=d.max_contracts, weights=pr.weights, half_lives=auto.half_lives, d_fast=pr.d_fast, d_slow=pr.d_slow, d_mix=pr.d_mix,
                se=pr.se, p=pr.p_used, n=pr.n, live_hours=live_hours, sized=d.sizing.contracts >= 1)


VOLUME_FALLBACK = 2000.0       # contracts: assumed last-minute volume when no reading has ever been available (so the order stays small rather than unbounded)
VOLUME_MEMORY = 20             # how many recent readings the fallback median looks at


def resolve_volume(reading: Optional[float], seen: list) -> tuple[float, bool]:
    """The liquidity-cap input of ONE decision - the single rule the live engine and the backtest share.  A finite reading counts (and is remembered in ``seen``); a missing one falls back to the
    median of the recent readings, or to ``VOLUME_FALLBACK`` before there was any - never to 'unlimited'.  Returns (contracts, was_measured)."""
    if reading is not None and np.isfinite(reading):
        seen.append(float(reading))
        del seen[:-VOLUME_MEMORY]
        return float(reading), True
    return (float(np.median(seen)) if seen else VOLUME_FALLBACK), False


@dataclass
class AutoDecision:
    sizing: Sizing
    pred: Prediction
    regime: str
    mult: float            # effective Kelly multiplier (ladder x regime x throttle)
    cap: float             # effective stake cap (fraction of equity)
    tier: int
    liq_cap: Optional[int]
    max_contracts: int
    p_used: float


def decide(auto: AutoConfig, stats: AdaptiveStats, leg: str, hour: int, cost: float, entry_ts: float, equity: float, peak: float, bankroll: float,
           volume: Optional[float], fee_model: str, slippage: float, min_edge: float, pred: Optional[Prediction] = None) -> AutoDecision:
    """Everything the Auto mode decides for ONE signal at the moment of the trade."""
    pred = pred or stats.predict(leg, hour, cost, entry_ts)
    px = min(cost + slippage, 0.99)
    costs = slippage + fee_per_contract(fee_model, px)
    regime = classify_regime(pred, costs, min_edge, auto)
    rs = auto.regime_scale(regime)
    tier, tmult, tcap = auto.tier(equity)
    thr = drawdown_throttle(1.0 - equity / peak) if (auto.throttle and peak > 0) else 1.0
    mult, cap = tmult * rs * thr, tcap * rs * thr
    liq = None
    max_c = auto.max_contracts
    if auto.participation > 0 and volume is not None and np.isfinite(volume) and volume >= 0:        # a reading of 0 means nothing traded: no liquidity (NaN = unknown -> ceiling)
        liq = int(auto.participation * volume)
        max_c = min(max_c, liq)
    if rs <= 0 or max_c < 1:
        sz = Sizing(0, 0.0, 0.0, 0.0, px, 0.0, f"regime {regime}" if rs <= 0 else "no liquidity")
    else:
        sz = size_bet(bankroll, pred.p_used, px, fee_model, mult, cap, int(max_c), min_edge)
    return AutoDecision(sizing=sz, pred=pred, regime=regime, mult=mult, cap=cap, tier=tier, liq_cap=liq, max_contracts=int(max_c), p_used=pred.p_used)
