"""
Mean-reversion model definitions, their signals, and decide() - the single decision function the backtest and the live engine both call.

Every signal of session t is computed with what is known at the decision time hh:mm of that session: the official daily bars of the PREVIOUS sessions and the session's
own snapshot (open / high / low / last price of the 15-minute bars that ended by hh:mm). Orders placed then fill in the closing auction (official close); positions held
"overnight" leave in the next opening auction (official open).

Signals (value per symbol):
  ibs        internal bar strength (last - low) / (high - low) of the session so far           Pagonidis (2013)
  rsi2       2-period Wilder RSI of the closes, with the last price as today's close            Connors & Alvarez (2008)
  ret1       return since the previous close; retN (N = 2..20) since N closes ago              Lehmann (1990), Jegadeesh (1990)
  zN         (last - mean of the last N closes) / their standard deviation                       Bollinger-type stretch
  gap        today's open vs the previous close
  intraday   last vs today's open
  sma_dist   last / 200-session mean - 1 (trend filter input)
  vol20      volatility of the last 20 daily returns (ranking / risk)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "MDY", "IJR", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "XBI", "IBB", "KRE", "XHB", "XRT",
                "ITB", "XOP", "GDX", "VNQ", "XME", "KBE", "OIH", "EFA", "EEM", "EWJ", "EWZ", "FXI", "EWG", "EWU", "EWT", "EWY", "INDA", "EWC", "EWA", "EWH", "MCHI", "TLT", "IEF",
                "HYG", "LQD", "TIP", "EMB", "GLD", "SLV", "USO", "UNG", "DBA", "VTV", "VUG", "IWD", "IWF"]


@dataclass
class MRModel:
    name: str = "custom"
    universe: str = "etf"                # 'etf' | 'top100' | 'top50' | 'list:SPY,QQQ,...' | parts joined with '+', e.g. 'top50+etf'
    minute: int = 1545                   # decision time: 1545 needs a real-time feed (IEX); 1530 works with the free 15-minute-delayed SIP
    direction: str = "long"              # long | short
    entry_signal: str = "ibs"
    entry_op: str = "<"
    entry_value: float = 0.2
    entry2_signal: str = ""              # optional second entry condition (both must hold); '' = none
    entry2_op: str = "<"
    entry2_value: float = 0.0
    trend_n: int = 0                     # > 0: a long needs last > the N-session mean (a short: below)
    hold: str = "rule"                   # 'rule' = exit on the exit rules below; 'overnight' = leave in the next opening auction
    exit_signal: str = "ibs"             # '' = none
    exit_op: str = ">"
    exit_value: float = 0.8
    exit_sma_n: int = 5                  # > 0: a long also leaves when last > the N-session mean
    max_hold: int = 10                   # sessions
    slots: int = 5                       # at most this many positions; each gets 1 / slots of the equity
    rank_by: str = "signal"              # which candidates get the free slots: 'signal' (most extreme) | 'vol_low' | 'vol_high'
    sizing: str = "equal"                # 'equal' = every slot the same $; 'inv_vol' = a slot x min(max_slot_mult, vol_target / the name's 20-day volatility) (the total stays <= 1x)
    vol_target: float = 0.015            # daily volatility a full slot is sized for (inv_vol)
    max_slot_mult: float = 2.0           # inv_vol: a calm name gets at most this many slots' worth
    max_gross: float = 1.0               # long + short value / equity - 1.0 = no leverage
    cash_reserve: float = 0.02           # equity kept in cash: a closing-auction price above the decision price cannot push the book over max_gross
    cost_auction_bp: float = 1.0         # per side, opening / closing auction
    borrow_bp_yr: float = 30.0
    min_price: float = 5.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MRModel":
        keys = MRModel.__dataclass_fields__
        return MRModel(**{k: v for k, v in d.items() if k in keys})


# ── signals ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    return pd.DataFrame(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _prev(a: np.ndarray, k: int = 1) -> np.ndarray:
    return np.vstack([np.full((k,) + a.shape[1:], np.nan), a[:-k]]) if k < len(a) else np.full_like(a, np.nan)


class Signals:
    """All signal arrays (sessions x symbols) of a Market, computed once; row(t) = the values at session t's decision time."""

    def __init__(self, mk):
        self.mk = mk
        C, P = mk.C, mk.sP
        self.cache: dict = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            self.prevC = _prev(C)
            self.last = P
            self.base = {"ibs": (P - mk.sL) / (mk.sH - mk.sL), "ret1": P / self.prevC - 1, "gap": mk.sO / self.prevC - 1, "intraday": P / mk.sO - 1}
            r = C / self.prevC - 1
            self.base["vol20"] = _prev(pd.DataFrame(r).rolling(20, min_periods=15).std().to_numpy())

    def sma(self, n: int) -> np.ndarray:
        """Mean of the previous n-1 closes and the last price."""
        key = ("sma", n)
        if key not in self.cache:
            s = pd.DataFrame(self.mk.C).rolling(n - 1, min_periods=n - 1).sum().to_numpy() if n > 1 else 0.0
            self.cache[key] = (_prev(s) + self.last) / n if n > 1 else self.last
        return self.cache[key]

    def get(self, name: str) -> np.ndarray:
        if name in self.base:
            return self.base[name]
        if name in self.cache:
            return self.cache[name]
        C = self.mk.C
        with np.errstate(invalid="ignore", divide="ignore"):
            if name.startswith("rsi"):
                n = int(name[3:] or 2)
                dlt = C - self.prevC
                au = _wilder(np.where(dlt > 0, dlt, 0.0), n)
                ad = _wilder(np.where(dlt < 0, -dlt, 0.0), n)
                a = 1.0 / n
                d_now = self.last - self.prevC                                              # today's change, measured to the decision-time price
                u = (1 - a) * _prev(au) + a * np.where(d_now > 0, d_now, 0.0)
                dd = (1 - a) * _prev(ad) + a * np.where(d_now < 0, -d_now, 0.0)
                v = 100 - 100 / (1 + u / dd)
                v = np.where(dd == 0, 100.0, v)
            elif name.startswith("ret"):
                n = int(name[3:])
                v = self.last / _prev(C, n) - 1
            elif name.startswith("z"):
                n = int(name[1:])
                m = self.sma(n)
                sd = _prev(pd.DataFrame(C).rolling(n, min_periods=n).std().to_numpy())
                v = (self.last - m) / sd
            elif name == "sma_dist":
                v = self.last / self.sma(200) - 1
            else:
                raise KeyError(f"unknown signal {name!r}")
        self.cache[name] = v
        return v

    def row(self, t: int, names: list) -> dict:
        return {n: self.get(n)[t] for n in names}


def needed_signals(m: MRModel) -> list:
    out = {m.entry_signal, "vol20"}
    for s in (m.exit_signal, m.entry2_signal):
        if s:
            out.add(s)
    return sorted(out)


def universe_parts(u: str) -> list:
    """'top50+etf' -> ['top50', 'etf'] ('list:A,B' stays one part)."""
    return [p.strip() for p in u.split("+") if p.strip()]


def _cmp(x: np.ndarray, op: str, v: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return {"<": x < v, ">": x > v, "<=": x <= v, ">=": x >= v}[op]


# ── the decision ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Pos:
    symbol: str
    shares: int                          # signed: > 0 long, < 0 short
    entry_t: int                         # session index of the entry
    entry_px: float
    exit_open_t: Optional[int] = None    # 'overnight' positions: the session whose opening auction they leave in
    trade_id: int = 0


@dataclass
class Decision:
    exits: list = field(default_factory=list)        # (symbol, reason)
    entries: list = field(default_factory=list)      # (symbol, signed shares, signal value, rank)
    candidates: list = field(default_factory=list)   # (symbol, signal value, rank, taken) - every name that passed the entry rule, in the ranking order
    notes: list = field(default_factory=list)


def decide(m: MRModel, t: int, sig: Signals, member: np.ndarray, positions: dict, equity: float, sma_cache: Optional[dict] = None) -> Decision:
    """The closing-auction decision of session t. ``member`` = which symbols may be entered today (bool per symbol), ``positions`` = {symbol: Pos} held now (the broker's
    positions in live), ``equity`` = account value at the decision-time prices. Returns the exits and the entries (whole shares, sized at the decision-time price)."""
    mk = sig.mk
    out = Decision()
    last = sig.last[t]
    col = mk.col
    # 'overnight' positions leave in the opening auction; one still held at the decision (no open print / the live engine missed the auction) leaves in this closing auction
    for s, p in positions.items():
        if p.exit_open_t is not None and p.exit_open_t <= t:
            out.exits.append((s, "overnight exit missed at the open: leaves in the closing auction"))
    # rule exits
    if m.hold == "rule":
        ex_v = sig.get(m.exit_signal)[t] if m.exit_signal else None
        sma_x = sig.sma(m.exit_sma_n)[t] if m.exit_sma_n else None
        for s, p in positions.items():
            if p.exit_open_t is not None:
                continue
            j = col.get(s)
            if j is None or not np.isfinite(last[j]):
                continue                                                                        # no price now: keep (a delisting is handled by the engine)
            age = t - p.entry_t
            longp = p.shares > 0
            reason = ""
            if ex_v is not None and np.isfinite(ex_v[j]) and bool(_cmp(np.array([ex_v[j]]), m.exit_op if longp else _flip(m.exit_op), m.exit_value if longp else _mirror(m, m.exit_value))[0]):
                reason = f"{m.exit_signal} {ex_v[j]:.2f} {m.exit_op if longp else _flip(m.exit_op)} {m.exit_value if longp else _mirror(m, m.exit_value):g}"
            elif sma_x is not None and np.isfinite(sma_x[j]) and ((last[j] > sma_x[j]) if longp else (last[j] < sma_x[j])):
                reason = f"price {'above' if longp else 'below'} its {m.exit_sma_n}-session mean"
            elif age >= m.max_hold:
                reason = f"time stop ({age} sessions)"
            if reason:
                out.exits.append((s, reason))
    leaving = {s for s, _ in out.exits}
    staying = {s: p for s, p in positions.items() if s not in leaving}
    free = m.slots - len(staying)
    ev = sig.get(m.entry_signal)[t]
    ok = member & np.isfinite(ev) & np.isfinite(last) & (last >= m.min_price)
    long_side = m.direction == "long"
    ok &= _cmp(ev, m.entry_op, m.entry_value)
    if m.entry2_signal:
        ok &= _cmp(sig.get(m.entry2_signal)[t], m.entry2_op, m.entry2_value)
    if m.trend_n:
        tr = sig.sma(m.trend_n)[t]
        with np.errstate(invalid="ignore"):
            ok &= (last > tr) if long_side else (last < tr)
    held_now = {col[s] for s in positions if s in col}
    cand = [j for j in np.where(ok)[0] if j not in held_now]
    if not cand:
        return out
    if m.rank_by == "vol_low":
        key = sig.get("vol20")[t][cand]
    elif m.rank_by == "vol_high":
        key = -sig.get("vol20")[t][cand]
    else:
        key = ev[cand] if m.entry_op in ("<", "<=") else -ev[cand]                              # the most extreme signal first
    order = [cand[i] for i in np.argsort(np.nan_to_num(key, nan=np.inf), kind="stable")]
    out.candidates = [(mk.syms[j], float(ev[j]), r, False) for r, j in enumerate(order, start=1)]
    if free <= 0 or equity <= 0:
        out.notes.append(f"{len(order)} candidate(s), no free slot ({len(staying)} of {m.slots} held)")
        return out
    gross_now = sum(abs(p.shares) * (last[col[s]] if s in col and np.isfinite(last[col[s]]) else p.entry_px) for s, p in staying.items())
    if not np.isfinite(equity) or not np.isfinite(gross_now):
        out.notes.append("equity or a position value unknown at the decision time - no entries")
        return out
    budget = (m.max_gross - m.cash_reserve) * equity
    room = max(budget - gross_now, 0.0)
    per = budget / m.slots
    vol = sig.get("vol20")[t] if m.sizing == "inv_vol" else None
    for rank, j in enumerate(order[:free], start=1):
        mult = 1.0
        if vol is not None:
            mult = min(m.max_slot_mult, m.vol_target / vol[j]) if np.isfinite(vol[j]) and vol[j] > 0 else 1.0
        notional = min(per * mult, room)
        sh = int(notional // last[j])
        if sh <= 0:
            continue
        room -= sh * last[j]
        out.entries.append((mk.syms[j], sh if long_side else -sh, float(ev[j]), rank))
    taken = {e[0] for e in out.entries}
    out.candidates = [(s, v, r, s in taken) for s, v, r, _ in out.candidates]
    if len(order) > free:
        out.notes.append(f"{len(order)} candidates for {free} free slot(s): the {free} with the {'most extreme ' + m.entry_signal if m.rank_by == 'signal' else m.rank_by} were taken")
    return out


def _flip(op: str) -> str:
    return {"<": ">", ">": "<", "<=": ">=", ">=": "<="}[op]


def _mirror(m: MRModel, v: float) -> float:
    """Exit threshold for a short position (mirror image of the long rule for bounded signals)."""
    if m.exit_signal == "ibs":
        return 1.0 - v
    if m.exit_signal.startswith("rsi"):
        return 100.0 - v
    return -v
