"""Semantic matching: title normalisation, strike / inequality parsing and containment structures.

Produces three kinds of structures for the detector / backtester:
* ``MarketPair``  - the same (or exactly opposite) question on Kalshi and Polymarket   -> Type I
* ``Basket``      - mutually-exclusive outcome sets, single- or cross-venue            -> Type II
* ``SubsetEdge``  - A implies B strike ladders (same underlying / expiry / style)      -> Type III

Matching is heuristic: every structure carries a ``confidence`` and human-readable ``reasons`` so the UI can
show *why* two contracts were paired. Resolution-rule equivalence still needs a human check (order ticket).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

from core.models import Market, Relation, Venue

# ── vocabulary ────────────────────────────────────────────────────────────────
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
           "oct": 10, "nov": 11, "dec": 12}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|" \
       r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_RE = re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s*(\d{{4}}))?", re.I)
ISO_DATE_RE = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")
TICKER_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(?:\d{4}|\d{2})?(?![0-9])")
TIME_RE = re.compile(r"\b(\d{1,2})(?::\d{2})?\s*(am|pm)\b", re.I)
YEAR_RE = re.compile(r"\b20[2-3]\d\b")
NUM_RE = re.compile(
    r"(?<![\w.])\$?\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(k|mm|m|bn|b|thousand|million|billion|%)?(?![\w])",
    re.I)

UNDERLYINGS: list[tuple[str, re.Pattern]] = [(n, re.compile(p, re.I)) for n, p in [
    ("btc", r"\b(bitcoin|btc)\b"), ("eth", r"\b(ethereum|ether|eth)\b"), ("sol", r"\b(solana|sol)\b"),
    ("xrp", r"\b(xrp|ripple)\b"), ("doge", r"\b(dogecoin|doge)\b"),
    ("cpi", r"\b(cpi|consumer price|inflation)\b"),
    ("fed", r"\b(fed funds|fomc|federal reserve|fed rate|interest rate|rate cut|rate hike|the fed)\b"),
    ("gdp", r"\b(gdp|gross domestic)\b"), ("unemp", r"\b(unemployment|jobless)\b"),
    ("nfp", r"\b(nonfarm|non-farm|payrolls|jobs report)\b"),
    ("spx", r"\b(s&p|spx|s and p 500)\b"), ("ndx", r"\b(nasdaq|ndx)\b"), ("dow", r"\b(dow jones|djia)\b"),
    ("gold", r"\bgold\b"), ("oil", r"\b(wti|crude|brent|oil price)\b"),
    ("eurusd", r"\b(eur/usd|eurusd)\b"), ("usdjpy", r"\b(usd/jpy|usdjpy)\b"),
]]

TOUCH_RE = re.compile(r"\b(reach|reaches|hit|hits|touch|touches|dip to|fall to|drop to|at any time|ever|"
                      r"all[- ]time high|new high|maximum|max)\b", re.I)
GT_RE = re.compile(r"\b(above|over|greater than|higher than|more than|at least|or more|or above|or higher|"
                   r"and above|exceed[s]?|surpass(?:es)?|top[s]?|reach(?:es)?(?=\s+\$?\d)|\(high\))\b|>=|>|≥|↑|⬆|\(high\)",
                   re.I)
LT_RE = re.compile(r"\b(below|under|less than|lower than|fewer than|at most|or less|or below|or lower|"
                   r"dip[s]?|drop[s]? below|fall[s]? below|\(low\))\b|<=|<|≤|↓|⬇|\(low\)", re.I)

QUAL_RE = re.compile(
    r"\b([1-4]q|[12]h|(?:1st|2nd|3rd|4th|first|second|third|fourth)\s+\w+|q[1-4]|h[12]|halftime|half|quarter|quarterfinal|semifinal|final|innings?|period|set|map|spread|handicap|"
    r"total|exact score|both teams|mvp|top|bottom|relegat\w*|playoff\w*|group|annualized|yoy|mom|qoq|core|"
    r"settle\w*|intraday)\b", re.I)

STOP = set("""will the a an be is are was were of on in at to by for and or than more less above below over under
before after price market yes no this that it its from with as if any than end close closing new who what which
does do did have has had not win wins winner price prices level levels reach reaches hit hits touch first
""".split())
CATCH_ALL = ("other", "none", "tie", "draw", "no winner", "field", "any other", "neither", "no one")
GENERIC_LABEL = set("""the of and a to in on for will win wins winner yes no at by mr ms dr""".split())


# ── text helpers ──────────────────────────────────────────────────────────────
def market_text(m: Market) -> str:
    parts: list[str] = []
    for p in (m.event_title, m.title, m.outcome_label):
        p = (p or "").strip()
        if p and p.lower() not in [x.lower() for x in parts] and not any(p.lower() in x.lower() for x in parts):
            parts.append(p)
    return " | ".join(parts)


def _multiplier(sfx: str) -> float:
    s = (sfx or "").lower()
    if s in ("k", "thousand"):
        return 1e3
    if s in ("m", "mm", "million"):
        return 1e6
    if s in ("b", "bn", "billion"):
        return 1e9
    return 1.0


def _strip_dates(text: str) -> str:
    t = DATE_RE.sub(" ", text)
    t = ISO_DATE_RE.sub(" ", t)
    t = TIME_RE.sub(" ", t)
    return YEAR_RE.sub(" ", t)


def extract_dates(text: str, ticker: str = "") -> list[str]:
    out: set[str] = set()
    for m in DATE_RE.finditer(text):
        out.add(f"{_MONTHS[m.group(1)[:3].lower()]:02d}-{int(m.group(2)):02d}")
    for m in ISO_DATE_RE.finditer(text):
        out.add(f"{int(m.group(2)):02d}-{int(m.group(3)):02d}")
    if ticker:
        for m in TICKER_DATE_RE.finditer(ticker):
            mon = _MONTHS.get(m.group(2).lower())
            if mon:
                out.add(f"{mon:02d}-{int(m.group(3)):02d}")
    return sorted(out)


def extract_hour(text: str) -> Optional[int]:
    hrs = set()
    for m in TIME_RE.finditer(text):
        h = int(m.group(1)) % 12 + (12 if m.group(2).lower() == "pm" else 0)
        hrs.add(h)
    return hrs.pop() if len(hrs) == 1 else None


def detect_underlying(text: str) -> str:
    for name, rx in UNDERLYINGS:
        if rx.search(text):
            return name
    return ""


def parse_strike(text: str) -> tuple[str, Optional[float], Optional[float], str]:
    """-> (direction, strike, strike_hi, style). direction in {gt, lt, between, ''}; style in {terminal, touch, ''}."""
    clean = _strip_dates(text)
    nums: list[tuple[int, float, str]] = []
    for m in NUM_RE.finditer(clean):
        raw = float(m.group(1).replace(",", ""))
        sfx = (m.group(2) or "").lower()
        nums.append((m.start(), raw * _multiplier(sfx), sfx))
    touch = bool(TOUCH_RE.search(clean))
    gt, lt = GT_RE.search(clean), LT_RE.search(clean)
    style = "touch" if touch else ("terminal" if (gt or lt) else "")
    ex = re.search(r"\b(exactly|precisely|equal to|equals)\b", clean, re.I)
    if ex and nums:                                # 'exactly 9 launches' is ONE bin of a mutually-exclusive set
        after = [n for n in nums if n[0] >= ex.end() - 1]
        return "eq", (after[0] if after else nums[0])[1], None, style or "terminal"
    range_pat = re.search(r"\d\s?(?:-|–|to)\s?\$?\d", clean)
    two_sided = bool(gt and lt)                    # 'at least 2.0 but below 2.25' is a bounded bin, not 'above 2.0'
    if len(nums) >= 2 and (re.search(r"\bbetween\b", clean, re.I) or two_sided
                           or (range_pat and not gt and not lt)):
        lo, hi = sorted((nums[0][1], nums[1][1]))
        return "between", lo, hi, style or "terminal"
    if not nums:
        return "", None, None, style
    direction = ""
    if gt and not lt:
        direction, pos = "gt", gt.end()
    elif lt and not gt:
        direction, pos = "lt", lt.end()
    elif gt and lt:
        first = gt if gt.start() < lt.start() else lt
        direction, pos = ("gt" if first is gt else "lt"), first.end()
    elif re.search(r"\b(dip to|fall to|drop to)\b", clean, re.I):
        direction, pos = "lt", 0
    else:
        # "hit $X" / "touch $X" without an explicit up/down marker is ambiguous (depends on spot): no structure
        return "", None, None, style
    after = [n for n in nums if n[0] >= pos - 1]
    strike = (after[0] if after else nums[-1])[1]
    return direction, strike, None, style


_SERIES_POS_RE = re.compile(r"\b(game|week|round|match|leg|stage|race|day)\s+\d+\b")


def tokenize(text: str) -> list[str]:
    t = _SERIES_POS_RE.sub(" ", _strip_dates(text.lower()))
    t = re.sub(r"[^\w\s%.$]", " ", t)
    out = []
    for w in t.split():
        w = w.strip(".$")
        if not w or w in STOP:
            continue
        if re.fullmatch(r"\d[\d,]*(\.\d+)?", w):
            w = w.replace(",", "")
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def enrich(m: Market) -> Market:
    """Fill semantic features (idempotent). Structured strikes supplied by a connector are kept."""
    text = market_text(m)
    if not m.tokens:
        m.tokens = sorted(set(tokenize(text)) | {'@' + q.lower() for q in QUAL_RE.findall(text)})
    if not m.date_tokens:
        m.date_tokens = extract_dates(text, m.market_id if m.venue is Venue.KALSHI else "")
    if m.hour_token is None:
        m.hour_token = extract_hour(text)
    if not m.underlying:
        m.underlying = detect_underlying(text)
    d, s, hi, style = parse_strike(text)
    if not m.direction and d:
        m.direction, m.strike, m.strike_hi = d, s, hi
    if not m.style:
        m.style = style if style else ("terminal" if m.direction in ("gt", "lt", "between") else "")
    return m


def similarity(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return 0.5 * inter / len(sa | sb) + 0.5 * inter / min(len(sa), len(sb))


def _close(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=1e-4, abs_tol=0.011)


def _dates_ok(a: Market, b: Market) -> bool:
    if a.date_tokens and b.date_tokens and not (set(a.date_tokens) & set(b.date_tokens)):
        return False
    if a.hour_token is not None and b.hour_token is not None and a.hour_token != b.hour_token:
        return False
    return True


def _hours_apart(a: Market, b: Market) -> Optional[float]:
    if not a.settle_time or not b.settle_time:
        return None
    return abs((a.settle_time - b.settle_time).total_seconds()) / 3600.0


# ── structures ────────────────────────────────────────────────────────────────
@dataclass
class MarketPair:
    a: Market                       # Kalshi
    b: Market                       # Polymarket
    relation: Relation
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class SubsetEdge:
    inner: Market                   # A  (A implies B)
    outer: Market                   # B
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class Basket:
    name: str
    scope: str                                     # "kalshi" | "polymarket" | "cross"
    outcomes: list[list[Market]]                   # per outcome: candidate markets (cheapest venue wins)
    exhaustive: bool
    exhaustive_reason: str
    settle_time: Optional[datetime]
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)


# ── Type I: binary pairs ──────────────────────────────────────────────────────
def match_pairs(kalshi: list[Market], poly: list[Market], date_tol_h: float = 24.0,
                min_conf: float = 0.0) -> list[MarketPair]:
    for m in kalshi:
        enrich(m)
    for m in poly:
        enrich(m)
    by_und: dict[str, list[Market]] = defaultdict(list)
    inv: dict[str, list[Market]] = defaultdict(list)
    for p in poly:
        if p.underlying:
            by_und[p.underlying].append(p)
        for t in p.tokens:
            inv[t].append(p)
    common = {t for t, v in inv.items() if len(v) > 60}
    pairs: dict[tuple[str, str], MarketPair] = {}
    # Games: team / city names are COMMON tokens on a sports-heavy venue, so rare-token blocking misses them.
    # Block on (date, entity) among markets of the same exchange-native kind instead.
    idx_de: dict[tuple[str, str, str], list[Market]] = defaultdict(list)
    for p in poly:
        if p.kind in ("moneyline", "spread", "total") and p.date_tokens:
            keys = set(_entities(p)) | {"#" + s for s in p.subject_ids}          # '#min' = team-abbreviation key
            for d in p.date_tokens:
                for e in keys:
                    idx_de[(p.kind, d, e)].append(p)

    for k in kalshi:
        cands: set[str] = set()
        pool: dict[str, Market] = {}
        if k.underlying and k.direction in ("gt", "lt", "between") and k.strike is not None:
            for p in by_und.get(k.underlying, []):
                pool[p.market_id] = p
        hits: dict[str, int] = defaultdict(int)
        for t in k.tokens:
            if t in common or len(t) < 3:
                continue
            for p in inv.get(t, []):
                hits[p.market_id] += 1
                pool.setdefault(p.market_id, p)
        if k.kind in ("moneyline", "spread", "total") and k.date_tokens:
            for d in k.date_tokens:
                for e in set(_entities(k)) | {"#" + s for s in k.subject_ids}:
                    for p in idx_de.get((k.kind, d, e), ()):
                        pool.setdefault(p.market_id, p)
                        hits[p.market_id] = max(hits.get(p.market_id, 0), 2)     # blocked by date+entity+kind
        for pid, p in pool.items():
            if hits.get(pid, 0) < 2 and not (k.underlying and p.underlying == k.underlying):
                continue
            pr = _score_pair(k, p, date_tol_h)
            if pr and pr.confidence >= min_conf:
                key = (k.market_id, p.market_id)
                if key not in pairs or pairs[key].confidence < pr.confidence:
                    pairs[key] = pr
    # keep the best partner per market on each side (avoid one-to-many spam)
    best_k: dict[str, MarketPair] = {}
    for pr in pairs.values():
        cur = best_k.get(pr.a.market_id)
        if cur is None or pr.confidence > cur.confidence:
            best_k[pr.a.market_id] = pr
    best_p: dict[str, MarketPair] = {}
    for pr in best_k.values():
        cur = best_p.get(pr.b.market_id)
        if cur is None or pr.confidence > cur.confidence:
            best_p[pr.b.market_id] = pr
    return sorted(best_p.values(), key=lambda x: -x.confidence)


_ENT_RE = re.compile(r"\b[A-Z][\w'’.\-]*")
_ENT_STOP = set("""will who what when how which the a an more markets market winner winners champion champions game games
match matches vs v both teams team to score btts total totals over under yes no exact correct first last next
price rate
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday""".split())


def _entities(m: Market) -> set[str]:
    """Distinctive capitalised names in the EVENT title (teams, people, places): the context a market lives in."""
    text = m.event_title or m.title
    ents = {w.lower().strip(".'’-") for w in _ENT_RE.findall(text)}
    return {e for e in ents if len(e) > 1 and e not in _ENT_STOP and not e.isdigit()}


def _event_ok(k: Market, p: Market) -> bool:
    """Both markets must sit in the same event: 'Croatia vs England: BTTS' is not 'Spain vs Czechia: BTTS'."""
    ek, ep = _entities(k), _entities(p)
    if not ek or not ep:
        return True
    return len(ek & ep) / min(len(ek), len(ep)) >= 0.75


_MACRO = {"cpi", "fed", "gdp", "unemp", "nfp"}
_REGION_ALIAS = {"us": "us", "usa": "us", "u.s": "us", "u.s.": "us", "america": "us", "american": "us",
                 "eurozone": "eu", "euro": "eu", "europe": "eu", "european": "eu", "eu": "eu",
                 "uk": "uk", "britain": "uk", "british": "uk", "mexico": "mexico", "mexican": "mexico",
                 "canada": "canada", "canadian": "canada", "germany": "germany", "german": "germany",
                 "france": "france", "french": "france", "italy": "italy", "italian": "italy", "spain": "spain",
                 "japan": "japan", "japanese": "japan", "china": "china", "chinese": "china", "india": "india",
                 "indian": "india", "brazil": "brazil", "brazilian": "brazil", "russia": "russia",
                 "australia": "australia", "korea": "korea", "turkey": "turkey", "argentina": "argentina",
                 "switzerland": "switzerland", "sweden": "sweden", "norway": "norway", "poland": "poland"}


def _regions(m: Market) -> frozenset:
    """Economies a macro market is about ('US real GDP' vs 'Mexico GDP' vs 'Eurozone GDP')."""
    words = re.findall(r"[a-z][a-z.]*", market_text(m).lower())
    return frozenset(_REGION_ALIAS[w] for w in words if w in _REGION_ALIAS)


_AMBIGUOUS = re.compile(r"\b(spread|handicap|margin|cover[s]?|wins? by|loses? by)\b", re.I)


def _subject_ok(k: Market, p: Market) -> bool:
    """The team / candidate / outcome each market is ABOUT must agree.

    Whole-question text similarity is not enough: 'San Lorenzo wins' and 'Defensa y Justicia win' share an event title
    but are opposite sides of the match (K YES + P NO would pay $0 if the opponent wins).
    """
    la, lb = _norm_label(k.outcome_label), _norm_label(p.outcome_label)
    if la and lb:
        if _is_catch_all(k.outcome_label) != _is_catch_all(p.outcome_label):
            return False                              # 'Draw' / 'Other' / 'Tie' only ever matches the same catch-all
        return len(la & lb) / max(len(la), len(lb)) >= 0.5
    if la:                                     # Kalshi subject must be named by Polymarket's own question
        if _is_catch_all(p.title) and not _is_catch_all(k.outcome_label):
            return False
        return len(la & _norm_label(p.title)) / len(la) >= 0.5
    if lb:
        return len(lb & _norm_label(f"{k.event_title} {k.title}")) / len(lb) >= 0.5
    return True


_LEAGUE_ALIAS = {"cfb": "ncaaf", "cbb": "ncaamb", "ncaab": "ncaamb", "ncaaw": "ncaawb"}


def _league_ok(k: Market, p: Market) -> bool:
    a, b = _LEAGUE_ALIAS.get(k.league, k.league), _LEAGUE_ALIAS.get(p.league, p.league)
    return bool(a) and a == b


def _score_pair(k: Market, p: Market, tol_h: float) -> Optional[MarketPair]:
    dh = _hours_apart(k, p)
    if dh is None or dh > tol_h:
        return None
    if not _dates_ok(k, p):
        return None
    if k.kind and p.kind and k.kind != p.kind and {k.kind, p.kind} & {"moneyline", "spread", "total"}:
        return None                                   # a moneyline is never a prop / spread / total / season future
    # Game moneylines: the exchange-native team abbreviation (Kalshi ticker suffix == Polymarket US displayAbbreviation)
    # plus league plus date is a stronger key than names ('Minnesota' vs 'Timberwolves', 'Atlanta' vs 'Falcons').
    if (k.kind == p.kind == "moneyline" and set(k.subject_ids) & set(p.subject_ids) and _league_ok(k, p)
            and set(k.date_tokens) & set(p.date_tokens) and dh <= min(tol_h, 6.0)):
        return MarketPair(k, p, Relation.EQUIVALENT, 0.95,
                          [f"team '{sorted(set(k.subject_ids) & set(p.subject_ids))[0].upper()}' + league "
                           f"{k.league.upper()} + date", f"settle Δ{dh:.1f}h"])
    if _AMBIGUOUS.search(market_text(k)) or _AMBIGUOUS.search(market_text(p)):
        return None                                   # 'LT wins by over 2.5' is not Polymarket's 'LT (+2.5)' outcome
    if k.style and p.style and k.style != p.style:
        return None                                   # "reach $X" (touch) is not "above $X at T" (terminal)
    if {t for t in k.tokens if t.startswith('@')} != {t for t in p.tokens if t.startswith('@')}:
        return None                                   # e.g. Q4 GDP vs full-year GDP, 1st half vs full game
    sim = similarity(k.tokens, p.tokens)
    time_f = max(0.0, 1.0 - dh / tol_h)
    reasons: list[str] = []
    structured = (k.underlying and k.underlying == p.underlying and k.direction and p.direction
                  and k.strike is not None and p.strike is not None)
    if structured:
        if not _event_ok(k, p) or (k.underlying in _MACRO and _regions(k) != _regions(p)):
            return None
        same_strike = _close(k.strike, p.strike) and (k.direction != "between" or _close(k.strike_hi, p.strike_hi))
        if not same_strike:
            return None
        if k.direction == p.direction:
            rel = Relation.EQUIVALENT
        elif {k.direction, p.direction} == {"gt", "lt"}:
            rel = Relation.INVERSE
        else:
            return None
        conf = 0.55 + 0.25 * min(1.0, sim / 0.5) + 0.20 * time_f
        reasons += [f"{k.underlying.upper()} {k.direction} {k.strike:g}", f"settle Δ{dh:.1f}h", f"text sim {sim:.2f}"]
        if rel is Relation.INVERSE:
            conf -= 0.10
            reasons.append("inverse pair: tie-at-strike risk")
    else:
        # An event title naming BOTH teams dilutes whole-text similarity; the subject gate below makes it safe to
        # also score the market's own title + outcome label.
        sim = max(sim, similarity(tokenize(f"{k.title} {k.outcome_label}"), tokenize(f"{p.title} {p.outcome_label}")))
        strong_ctx = bool(set(k.date_tokens) & set(p.date_tokens)) and bool(_entities(k)) and bool(_entities(p)) \
            and _event_ok(k, p)
        if sim < (0.45 if strong_ctx else 0.60):
            return None
        if not _subject_ok(k, p) or not _event_ok(k, p):
            return None
        if ({t for t in k.tokens if _NUMERIC_TOKEN.match(t)} != {t for t in p.tokens if _NUMERIC_TOKEN.match(t)}):
            return None                               # 'top 4' vs 'top 6', '3+ goals' vs '2+ goals'
        if k.direction and p.direction and (k.direction != p.direction or not _close(k.strike, p.strike)):
            return None
        rel = Relation.EQUIVALENT
        conf = 0.65 * sim + 0.35 * time_f
        if strong_ctx and _norm_label(k.outcome_label) and _norm_label(p.outcome_label):
            conf = min(0.99, conf + 0.10)             # same date + same event + same named subject: corroborated
            reasons.append('date+event+subject corroborated')
        reasons += [f"text sim {sim:.2f}", f"settle Δ{dh:.1f}h"]
    if k.style and not p.style or p.style and not k.style:
        conf -= 0.05
    return MarketPair(k, p, rel, round(max(0.0, min(conf, 0.99)), 3), reasons)


# ── Type III: strike ladders / containment ────────────────────────────────────
_NUMERIC_TOKEN = re.compile(r"^[\d.,%$]+$")


def stem_tokens(m: Market) -> set[str]:
    """Question stem: tokens with all numbers removed (so 'above 100k' and 'above 95k' share a stem)."""
    return {t for t in m.tokens if not _NUMERIC_TOKEN.match(t)}


def stem_key(m: Market) -> str:
    """Exact question template with every number replaced by '#': two markets are rungs of ONE strike ladder only
    if their templates are identical ('will cpi rise more than #% in december' ...)."""
    t = _strip_dates(market_text(m).lower())
    letters = lambda d: "".join(chr(97 + int(c)) for c in d)          # noqa: E731  '1' -> 'b'
    t = re.sub(r"\b(\d+)(st|nd|rd|th|q|h)\b", lambda x: letters(x.group(1)) + x.group(2), t)    # 1q, 2nd, 1h
    t = re.sub(r"\b(game|map|set|period|quarter|half|round|innings?|week|leg|stage|match|race|day)\s+(\d+)\b",
               lambda x: f"{x.group(1)} {letters(x.group(2))}", t)
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", "#", t)
    t = re.sub(r"[^\w#%$]+", " ", t)
    return " ".join(t.split())


def _ladder_edges(ms: list[Market], direction: str, conf: float, tag: str,
                  cross_only: Optional[tuple[list[Market], list[Market]]] = None,
                  tol_h: float = 2.0) -> list[SubsetEdge]:
    """All A -> B pairs inside one ladder (or across two ladders): higher 'above' / lower 'below' strike implies B."""
    edges: list[SubsetEdge] = []
    pairs = ([(a, b) for a in ms for b in ms] if cross_only is None else
             [(a, b) for a in cross_only[0] for b in cross_only[1]] +
             [(a, b) for a in cross_only[1] for b in cross_only[0]])
    for a, b in pairs:
        if a.venue == b.venue and a.market_id == b.market_id:
            continue
        if _close(a.strike, b.strike):
            continue
        dh = _hours_apart(a, b)
        if dh is None or dh > tol_h:                # monotonicity only holds for the SAME resolution moment
            continue
        implies = a.strike > b.strike if direction == "gt" else a.strike < b.strike
        if not implies:
            continue
        edges.append(SubsetEdge(a, b, conf, [f"{(a.underlying or '?').upper()} {direction} {a.strike:g} ⊆ "
                                             f"{direction} {b.strike:g}", tag]))
    return edges


def build_subset_edges(markets: list[Market], tol_h: float = 2.0, max_family: int = 300) -> list[SubsetEdge]:
    """Edges A -> B where A implies B.

    * same-event ladders (one venue, one event, several strikes) - the classic monotonicity check;
    * cross-venue ladders - only when the two events share underlying, direction, resolution style, settle time
      (+/- ``tol_h``), date/hour tokens and a >=55% question-stem match.
    Grouping by *underlying alone* is deliberately NOT used: 'CPI MoM', 'core CPI YoY' and other countries'
    budget balances are different quantities and would create phantom edges.
    """
    ladders: dict[tuple, list[Market]] = defaultdict(list)
    for m in markets:
        enrich(m)
        if m.mutually_exclusive or m.neg_risk:
            continue                                # outcomes of a mutually-exclusive event are BINS, never nested
        if m.direction in ("gt", "lt") and m.strike is not None and m.settle_time:
            # exchange-native ladder id when the venue provides one (game total vs team total vs quarter total!)
            ladders[(m.venue, m.event_id, m.style or "?", m.direction, m.ladder_key or stem_key(m))].append(m)
    edges: list[SubsetEdge] = []
    for (venue, eid, style, direction, _stem), ms in ladders.items():
        if len(ms) >= 2:
            ms = sorted(ms, key=lambda x: -x.volume)[:max_family]
            edges += _ladder_edges(ms, direction, 0.90 if style != "?" else 0.75, "same-event ladder", tol_h=tol_h)
    by_und: dict[tuple, list[tuple[tuple, list[Market]]]] = defaultdict(list)
    for key, ms in ladders.items():
        if key[0] is Venue.POLYMARKET and ms[0].underlying:
            by_und[(ms[0].underlying, key[2], key[3])].append((key, ms))
    for key, kms in ladders.items():
        if key[0] is not Venue.KALSHI or not kms[0].underlying:
            continue
        k0 = kms[0]
        for pkey, pms in by_und.get((k0.underlying, key[2], key[3]), []):
            p0 = pms[0]
            dh = _hours_apart(k0, p0)
            if dh is None or dh > tol_h or not _dates_ok(k0, p0):
                continue
            sim = similarity(stem_tokens(k0), stem_tokens(p0))
            if sim < 0.55:
                continue
            conf = 0.70 + 0.25 * min(1.0, sim)
            kk = sorted(kms, key=lambda x: -x.volume)[:max_family]
            pp = sorted(pms, key=lambda x: -x.volume)[:max_family]
            edges += _ladder_edges([], key[3], conf, f"cross-venue ladder (stem sim {sim:.2f}, settle Δ{dh:.1f}h)",
                                   cross_only=(kk, pp), tol_h=tol_h)
    return edges


# ── Type II: MECE baskets ─────────────────────────────────────────────────────
def _norm_label(s: str) -> set[str]:
    return {w for w in re.sub(r"[^\w\s]", " ", (s or "").lower()).split() if w not in GENERIC_LABEL}


def _is_catch_all(label: str) -> bool:
    l = (label or "").lower()
    return any(c in l for c in CATCH_ALL)


def _ladder_covers_line(ms: list[Market]) -> bool:
    """True when strike-typed outcomes tile (-inf, +inf): an open-low, an open-high and contiguous bins between."""
    iv = []
    for m in ms:
        if m.direction == "gt" and m.strike is not None:
            iv.append((m.strike, math.inf))
        elif m.direction == "lt" and m.strike is not None:
            iv.append((-math.inf, m.strike))
        elif m.direction == "between" and m.strike is not None and m.strike_hi is not None:
            iv.append((m.strike, m.strike_hi))
        else:
            return False
    if len(iv) != len(ms) or len(iv) < 2:
        return False
    iv.sort()
    if iv[0][0] != -math.inf or iv[-1][1] != math.inf:
        return False
    widths = [h - l for l, h in iv if math.isfinite(l) and math.isfinite(h) and h > l]
    tol = 0.35 * (sorted(widths)[len(widths) // 2] if widths else 0.0) + 1e-9
    reach = iv[0][1]
    for lo, hi in iv[1:]:
        if lo - reach > tol:
            return False
        reach = max(reach, hi)
    return True


def _basket_settle(ms: list[Market]) -> Optional[datetime]:
    ts = [m.settle_time for m in ms if m.settle_time]
    return max(ts) if ts else None


def _complete_outcomes(ms: list[Market], historical: bool = False) -> Optional[list[Market]]:
    """Outcomes still tradable. Events with any outcome in an unknown state (closed, undetermined, YES) are skipped:
    buying "all" of an incomplete set is NOT a riskless position."""
    live: list[Market] = []
    for m in ms:
        if m.status == "open" or (historical and m.status == "settled"):
            live.append(m)
        elif m.status == "settled" and m.result == "no":
            continue                                       # already resolved NO: drop from the set
        else:
            return None
    return live if len(live) >= 2 else None


def kalshi_baskets(markets: list[Market], max_legs: int = 60, historical: bool = False) -> list[Basket]:
    """``markets`` must contain EVERY market of each event (all statuses, no volume filtering)."""
    ev: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        if m.venue is Venue.KALSHI and m.mutually_exclusive:
            ev[m.event_id].append(m)
    out = []
    for eid, all_ms in ev.items():
        ms = _complete_outcomes(all_ms, historical)
        if ms is None or len(ms) > max_legs:
            continue
        for m in ms:
            enrich(m)
        exh, why = False, "mutually exclusive, exhaustiveness unproven (only buy-all-NO is riskless)"
        if any(_is_catch_all(m.outcome_label or m.title) for m in ms):
            exh, why = True, "catch-all outcome present"
        elif _ladder_covers_line(ms):
            exh, why = True, "strike ladder tiles the whole number line"
        out.append(Basket(name=ms[0].event_title or eid, scope="kalshi", outcomes=[[m] for m in ms],
                          exhaustive=exh, exhaustive_reason=why, settle_time=_basket_settle(ms)))
    return out


def poly_baskets(markets: list[Market], max_legs: int = 60, historical: bool = False) -> list[Basket]:
    """``markets`` must contain EVERY market of each event (all statuses, no volume filtering)."""
    ev: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        if m.venue is Venue.POLYMARKET and m.neg_risk and m.yes_token:
            ev[m.event_id].append(m)
    out = []
    for eid, all_ms in ev.items():
        ms = _complete_outcomes(all_ms, historical)
        if ms is None or len(ms) > max_legs:
            continue
        for m in ms:
            enrich(m)
        has_other = any(_is_catch_all(m.outcome_label or m.title) for m in ms)
        aug = any(m.neg_risk_augmented for m in ms)
        exh = has_other or not aug
        why = ("neg-risk event (exactly one outcome resolves YES)" if not aug else
               ("augmented neg-risk with 'Other' outcome" if has_other else
                "augmented neg-risk without 'Other' - named outcomes may not be exhaustive"))
        out.append(Basket(name=ms[0].event_title or eid, scope="polymarket", outcomes=[[m] for m in ms],
                          exhaustive=exh, exhaustive_reason=why, settle_time=_basket_settle(ms)))
    return out


def match_baskets(kb: list[Basket], pb: list[Basket], tol_h: float = 24.0) -> list[Basket]:
    """Cross-venue baskets: same event, outcomes aligned 1:1 by label; each outcome may be bought on either venue."""
    out: list[Basket] = []
    for a in kb:
        ta = set(tokenize(a.name))
        for b in pb:
            if len(a.outcomes) != len(b.outcomes):
                continue
            if a.settle_time and b.settle_time and abs((a.settle_time - b.settle_time).total_seconds()) / 3600 > tol_h:
                continue
            if similarity(ta, tokenize(b.name)) < 0.45:
                continue
            used: set[int] = set()
            aligned: list[list[Market]] = []
            ok = True
            for ka in a.outcomes:
                la = _norm_label(ka[0].outcome_label or ka[0].title)
                best, bj = 0.0, -1
                for j, pb_out in enumerate(b.outcomes):
                    if j in used:
                        continue
                    lb = _norm_label(pb_out[0].outcome_label or pb_out[0].title)
                    if not la or not lb:
                        continue
                    sc = len(la & lb) / min(len(la), len(lb))
                    if sc > best:
                        best, bj = sc, j
                if best < 0.5 or bj < 0:
                    ok = False
                    break
                used.add(bj)
                aligned.append([ka[0], b.outcomes[bj][0]])
            if not ok:
                continue
            exh = a.exhaustive or b.exhaustive
            out.append(Basket(name=a.name, scope="cross", outcomes=aligned, exhaustive=exh,
                              exhaustive_reason=f"{a.exhaustive_reason} / {b.exhaustive_reason}",
                              settle_time=max(filter(None, [a.settle_time, b.settle_time]), default=None),
                              confidence=0.8,
                              notes=["outcome labels aligned by name - verify both rulebooks"]))
    return out
