"""
The momentum-book model and its two decision functions - the single code path the backtest and the live engine both call.

Overnight leg (Lou, Polk & Skouras 2019; Berkman et al. 2012): at the decision time (15:30 = the 15-minute SIP bars that ended by then) rank the universe by
  rule      today's move: last price / yesterday's official close - 1
  rule+ml   the rank of that move + the rank of a walk-forward gradient-boosting forecast of the overnight return (ml.py)
and buy the k best in the closing auction (1 / k of the capital each, whole shares at the decision price), only when SPY's last close is above its 200-session mean.
They are sold in the next opening auction.

Intraday leg (Zarattini, Aziz & Barbon 2024, "Beat the Market"): on the intraday symbol (QQQ), long-only, every 30 minutes from 10:00 to 30 minutes before the close:
  band(slot) = max(today's open, yesterday's close) x (1 + mean of |close(slot) / open - 1| over the previous 14 sessions)
  flat and the bar closes above the band -> buy;  long and the bar closes below max(band, session VWAP) -> sell;  still long after the last check -> closing auction.
The two legs share the same capital: the overnight book is sold at the open before the intraday leg can buy, and the intraday position is sold in the closing auction
in which the overnight book is bought, so gross stays <= 1x at every moment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..mr.models import ETF_UNIVERSE  # noqa: F401  (re-exported: the 59 ETFs of the universe)

RANKINGS = ("rule", "rule+ml")
INTRADAY_FEEDS = ("iex", "sip")
SECTOR_OF_ETF = {"XLK": "Tech", "SMH": "Tech", "XLF": "Fin", "KRE": "Fin", "KBE": "Fin", "XLE": "Energy", "XOP": "Energy", "OIH": "Energy", "XLV": "Health",
                 "XBI": "Health", "IBB": "Health", "XLI": "Ind", "XLY": "Disc", "XHB": "Disc", "XRT": "Disc", "ITB": "Disc", "XLP": "Stap", "XLU": "Util", "XLB": "Mat",
                 "XME": "Mat", "GDX": "Mat", "XLRE": "RE", "VNQ": "RE", "XLC": "Comm"}   # the sector ETFs count in their sector; broad, country, bond, commodity ETFs in none


@dataclass
class MomModel:
    name: str = "custom"
    universe: str = "top150+etf"         # 'topN' (point-in-time monthly liquidity ranks) | 'etf' | 'list:A,B,...' | parts joined with '+'
    minute: int = 1530                   # overnight decision time: 1530 works with the free 16-minute-delayed SIP; 1545 needs the real-time SIP (paid)
    overnight: bool = True
    k: int = 20                          # names bought overnight, 1 / k of the capital each
    ranking: str = "rule"                # 'rule' | 'rule+ml'
    trend_filter: bool = True            # overnight only when SPY's last close is above its 200-session mean
    min_price: float = 0.0
    intraday_symbol: str = "QQQ"         # '' = no intraday leg
    intraday_every: int = 2              # check every N 15-minute bars (2 = every 30 minutes)
    intraday_lookback: int = 14          # sessions in the noise band
    intraday_trail: bool = True          # sell when the price falls below max(band, VWAP)
    intraday_mult: float = 1.0           # band width multiplier
    intraday_feed: str = "iex"           # decision prices / VWAP of the intraday leg: 'iex' (free real-time feed) | 'sip' (real-time SIP: paid plan)
    cash_reserve: float = 0.02           # kept in cash: a closing price above the decision price cannot push the book over 1x
    sector_cap: int = 0                  # > 0: at most this many overnight names per sector (the 11 sectors of the universe file; sector ETFs count in theirs, other
                                         # ETFs in none) - the next-best name of another sector takes the place; 0 = off
    cost_auction_bp: float = 1.0         # per side, opening / closing auction
    half_spread_bp: float = 0.5          # per side, intraday market orders
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "MomModel":
        keys = MomModel.__dataclass_fields__
        return MomModel(**{k: v for k, v in d.items() if k in keys})

    @property
    def uses_ml(self) -> bool:
        return self.overnight and self.ranking == "rule+ml"

    @property
    def uses_auctions(self) -> bool:
        """The model trades in the auctions: the overnight leg buys in the closing auction and sells in the opening one, and the intraday position still held after the
        last check is sold in the closing auction. (Live can send market DAY orders in their place - MomLiveSettings.auction_orders.)"""
        return self.overnight or bool(self.intraday_symbol)

    def auction_costs(self) -> tuple:
        """(cost at the open, cost at the close, the intraday symbol's cost at the close) per side, as fractions."""
        c = self.cost_auction_bp / 1e4
        return c, c, c


# ── overnight leg ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def pct_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank (average ties, rank / count) of the finite values of x inside mask; NaN elsewhere."""
    ok = mask & np.isfinite(x)
    out = np.full(len(x), np.nan)
    if ok.any():
        out[ok] = rankdata(x[ok], method="average") / ok.sum()
    return out


def overnight_score(m: MomModel, d45: np.ndarray, pred: Optional[np.ndarray], U: np.ndarray) -> np.ndarray:
    """The ranking score of one session (higher = bought first); NaN = not rankable."""
    if m.ranking == "rule":
        return np.where(U & np.isfinite(d45), d45, np.nan)
    if pred is None:
        return np.full(len(d45), np.nan)
    a, b = U & np.isfinite(pred), U & np.isfinite(d45)
    if a.sum() and (a == b).all():                                                              # one division: equal rank sums are exactly equal (ties go to the ticker order)
        out = np.full(len(d45), np.nan)
        out[a] = (rankdata(pred[a], method="average") + rankdata(d45[a], method="average")) / a.sum()
        return out
    return pct_rank(pred, U) + pct_rank(d45, U)


@dataclass
class OvernightDecision:
    entries: list = field(default_factory=list)      # (symbol, shares, score, rank, move, forecast)
    exits: list = field(default_factory=list)        # (symbol, reason) - overnight positions still held at the decision that are not picked again
    keeps: list = field(default_factory=list)        # still held and picked again: kept, no order
    candidates: list = field(default_factory=list)   # (symbol, score, rank, move, forecast, taken) - the best-ranked names
    notes: list = field(default_factory=list)


def decide_overnight(m: MomModel, syms: list, U: np.ndarray, trend: bool, d45: np.ndarray, pred: Optional[np.ndarray], last: np.ndarray, held: dict, cap: float,
                     n_show: int = 60, sectors: Optional[list] = None) -> OvernightDecision:
    """The closing-auction decision of one session. ``U`` = rankable names (universe member, decision price and previous close known), ``held`` = {symbol: shares} of overnight
    positions still held at the decision (their opening-auction sale failed), ``cap`` = capital at the decision-time prices. Entries are whole shares at the decision price.
    ``sectors`` (aligned with ``syms``; None / '' = no sector) applies the model's sector cap: in score order, a name whose sector already has sector_cap picks is skipped."""
    out = OvernightDecision()
    score = overnight_score(m, d45, pred, U)
    if m.ranking == "rule+ml" and not np.isfinite(score).any():
        out.notes.append("no ML forecast for this session (train the model of this year) - no overnight entries")
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(score) & np.isfinite(last) & (last > 0) & (last >= m.min_price)
    if m.intraday_symbol and m.intraday_symbol in syms:
        ok[syms.index(m.intraday_symbol)] = False                                              # one order per symbol per auction: the intraday symbol is not bought overnight
    order = sorted(np.where(ok)[0], key=lambda j: (-score[j], syms[j]))
    picks: list = []
    if trend and m.overnight:
        if m.sector_cap > 0 and sectors is not None:
            n_sec: dict = {}
            skipped = []
            for j in order:
                if len(picks) >= m.k:
                    break
                sec = sectors[j]
                if sec and n_sec.get(sec, 0) >= m.sector_cap:
                    skipped.append(f"{syms[j]} ({sec})")
                    continue
                if sec:
                    n_sec[sec] = n_sec.get(sec, 0) + 1
                picks.append(j)
            if skipped:
                out.notes.append(f"sector cap {m.sector_cap}: skipped {', '.join(skipped)}")
        else:
            picks = order[:m.k]
    if m.overnight and not trend:
        out.notes.append("SPY's last close is below its 200-session mean: no overnight positions today")
    pick_syms = {syms[j] for j in picks}
    for s in held:
        if s in pick_syms:
            out.keeps.append(s)
        else:
            out.exits.append((s, "no opening-auction sale: leaves in the closing auction"))
    per = (1.0 - m.cash_reserve) * cap / max(m.k, 1)
    rank_of = {j: r for r, j in enumerate(order, start=1)}
    fc = (lambda j: float(pred[j])) if pred is not None else (lambda j: float("nan"))
    if cap > 0 and np.isfinite(cap):
        for j in picks:
            s = syms[j]
            if s in held:
                continue
            sh = int(per // last[j])
            if sh > 0:
                out.entries.append((s, sh, float(score[j]), rank_of[j], float(d45[j]), fc(j)))
    elif picks:
        out.notes.append("capital unknown or zero - no entries")
    taken = {e[0] for e in out.entries} | set(out.keeps)
    out.candidates = [(syms[j], float(score[j]), rank_of[j], float(d45[j]), fc(j), syms[j] in taken) for j in order[:max(n_show, m.k)]]
    return out


# ── intraday leg ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def check_slots(nb: int, every: int) -> list:
    """Slots checked in a session of ``nb`` bars: every ``every`` bars from the bar that ends at 10:00, the last one ending 30 minutes before the close (as the paper)."""
    return list(range(every - 1, nb - 1, every))


def noise_upper(open_today: float, prev_close: float, sigma_row: np.ndarray, mult: float = 1.0) -> np.ndarray:
    """The upper noise band of each slot of the session."""
    if not (np.isfinite(open_today) and np.isfinite(prev_close)):
        return np.full(len(sigma_row), np.nan)
    return max(open_today, prev_close) * (1 + mult * sigma_row)


def noise_action(held: bool, px: float, upper: float, vwap: float, trail: bool = True) -> Optional[str]:
    """'buy', 'sell' or None at a check: buy above the band when flat; sell below max(band, VWAP) when long."""
    if not (np.isfinite(px) and np.isfinite(upper)):
        return None
    if not held:
        return "buy" if px > upper else None
    stop = upper if not np.isfinite(vwap) else max(upper, vwap)
    return "sell" if trail and px < stop else None


def intraday_shares(m: MomModel, equity: float, cash: float, px: float) -> int:
    """Whole shares a buy at ``px`` gets: (1 - cash reserve) of the capital, never more than the cash (no leverage)."""
    fill = px * (1 + m.half_spread_bp / 1e4)
    budget = min((1.0 - m.cash_reserve) * equity, cash)
    if not (np.isfinite(budget) and np.isfinite(fill)) or budget <= 0 or fill <= 0:
        return 0
    return int(budget // fill)


def slot_end(day, slot: int) -> pd.Timestamp:
    """Clock time at which bar ``slot`` of ``day`` ends (slot 0 = 09:30-09:45)."""
    return pd.Timestamp(day).normalize() + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=15 * (slot + 1))
