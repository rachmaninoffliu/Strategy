"""Pre-flight go / no-go check, run before every trading session (the GUI runs it automatically when you press Start, and at 21:45 UTC).

It answers one question - "can I run this strategy live right now?" - from evidence, not hope:

  data      local history fresh (auto-updates from Kalshi when stale) · Kalshi quote feed and exchange up · PC clock
  evidence  the pre-registered forward test (SPRT, frozen 2026-10-07) re-evaluated on the freshly updated data ·
            trailing 14/28-day edge of the preset you are about to trade · the learner gate (would it bet today?)
  risk      sizing vs the validated limits · preset is one that was actually tested · previous session ended cleanly
  account   (LIVE only) Kalshi key, buying power vs capital, clock skew, resting orders / positions, price-tick structure of the live market (no order is sent)

Verdict (always about LIVE trading; paper is always allowed):
  GO       every check passed
  CAUTION  advice, not a gate: live may run with the user's own settings; ``Report.stake_cap`` is only a SUGGESTED per-trade cap while the evidence is unconfirmed
  NO-GO    a technical failure only (no data, feed / exchange / account / clock problem, extreme sizing).  EVIDENCE NEVER BLOCKS: a rejected forward test or a negative trailing edge is a warning
           with a suggested stake cap, because the engine's own learner gate already sits out when it sees no edge - and an engine that is not running cannot catch the next regime
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests

from . import kalshi
from .data import DATA, coverage, data_range, extract_signals, load_rows, update_data
from .fees import fee_per_contract
from .kalshi import KalshiQuotes, base_url
from . import radar
from .adaptive import decide, format_ladder, snapshot
from .evfilter import describe_rules
from .live import MIN_HISTORY, LiveConfig, auto_active, build_stats
from .strategy import StrategyParams, fmt_int_list, size_bet

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"
WINDOW = 900

# ── the pre-registered forward test (forward_test.py, frozen 2026-10-07) ───────
FORWARD_START_TS = int(datetime(2026, 10, 8, tzinfo=timezone.utc).timestamp())
FORWARD_RULE = StrategyParams(hours=(22, 23), minutes=(12, 13), cost_min=0.05, cost_max=0.30, max_spread=0.06)   # "R1", 1 contract, Kalshi fee
SPRT_D, SPRT_S, SPRT_ACCEPT, SPRT_REJECT = 0.03, 0.40, 2.77, -1.56
FORWARD_MAX_TRADES = 150

CAUTION_CAP = 0.005                  # SUGGESTED per-trade stake cap while the evidence is not confirmed (0.5% of bankroll); never enforced - the GUI only offers it as an option
STALE_UPDATE_S = 45 * 60             # update from Kalshi when the newest settled window is older than this ...
MIN_UPDATE_GAP_S = 20 * 60           # ... and the last pre-flight update is older than this (Kalshi's candle history trails by ~2 h, so "older than 45 min" is the norm)
RERUN_FORWARD_AFTER_S = 12 * 3600
VALIDATED_HOURS = {20, 21, 22, 23}
VALIDATED_MULT, VALIDATED_CAP = 0.5, 0.02
AUTO_VALIDATED_CAP = 0.03            # the first step of the default Auto equity ladder (0.5x Kelly / 3%) is the tested starting size


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str
    cap: Optional[float] = None      # a WARN/FAIL with a cap means "live is acceptable only with per-trade stake <= cap"


@dataclass
class ForwardResult:
    n: int = 0
    win: float = float("nan")
    avg_cost: float = float("nan")
    ev_c: float = float("nan")
    se_c: float = float("nan")
    cum_c: float = 0.0
    llr: float = 0.0
    verdict: str = "CONTINUE"        # ACCEPT | REJECT | CONTINUE
    decided_at: Optional[int] = None  # open_ts of the trade at which a boundary was first crossed

    def line(self) -> str:
        if self.n == 0:
            return "no qualifying forward trades yet"
        return (f"n={self.n} forward trades since 2026-10-08: win {self.win * 100:.0f}%, avg price {self.avg_cost * 100:.0f}c, EV {self.ev_c:+.1f}c "
                f"(se {self.se_c:.1f}c), SPRT LLR {self.llr:+.2f} (accept >= {SPRT_ACCEPT:+.2f}, reject <= {SPRT_REJECT:+.2f}) -> {self.verdict}")


@dataclass
class Report:
    checks: list = field(default_factory=list)
    created: float = 0.0
    live: bool = False
    capital: float = 0.0
    forward: Optional[ForwardResult] = None
    rows: Optional[pd.DataFrame] = None          # freshly loaded history (the GUI adopts it)
    client: Optional[object] = None              # Kalshi client built during the check (the GUI adopts it)
    updated_data: bool = False
    notes: list = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if any(c.status == FAIL for c in self.checks):
            return "NO-GO"
        if any(c.status == WARN for c in self.checks):
            return "CAUTION"
        return "GO"

    @property
    def stake_cap(self) -> Optional[float]:
        caps = [c.cap for c in self.checks if c.cap is not None and c.status in (WARN, FAIL)]
        return min(caps) if caps else None

    def of(self, status: str) -> list:
        return [c for c in self.checks if c.status == status]

    def headline(self) -> str:
        v = self.verdict
        if v == "GO":
            return "GO — every check passed. You can run live with your configured settings."
        if v == "CAUTION":
            cap = self.stake_cap
            capt = f" Suggested (optional) per-trade stake cap while the evidence is unconfirmed: {cap * 100:.1f}% of bankroll - you can also continue with your own settings." if cap is not None else ""
            return "CAUTION — no blocking problem, but see the warnings." + capt
        return "NO-GO — do not run live. Paper trading is still fine. " + "; ".join(c.name for c in self.of(FAIL)) + "."

    def text(self) -> str:
        t = datetime.fromtimestamp(self.created, timezone.utc)
        out = [f"PRE-FLIGHT  {t:%Y-%m-%d %H:%M:%S} UTC   (session mode: {'LIVE' if self.live else 'paper'}; capital ${self.capital:,.2f})",
               f"VERDICT FOR LIVE TRADING: {self.headline()}", ""]
        for c in self.checks:
            out.append(f"[{c.status:4s}] {c.group:8s} {c.name}: {c.detail}")
        for n in self.notes:
            out.append(f"       note: {n}")
        return "\n".join(out)


# ── persistent state (forward-test bookkeeping, previous session) ─────────────
def state_dir() -> Path:
    return Path(os.environ.get("TRADER_STATE_DIR") or DATA)


def load_state(d: Optional[Path] = None) -> dict:
    p = (d or state_dir()) / "preflight_state.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict, d: Optional[Path] = None) -> None:
    d = d or state_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "preflight_state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    except OSError:
        pass


def record_session_end(why: str, mode: str, pnl: float, trades: int, d: Optional[Path] = None, now: Optional[float] = None) -> None:
    st = load_state(d)
    st["last_session"] = dict(ts=now if now is not None else time.time(), why=why, mode=mode, pnl=float(pnl), trades=int(trades))
    save_state(st, d)


def _ago(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{sec / 60:.0f} min" if sec < 5400 else f"{sec / 3600:.1f} h" if sec < 172800 else f"{sec / 86400:.1f} days"


def _hhmm(ts: float) -> str:
    return f"{datetime.fromtimestamp(ts, timezone.utc):%H:%M:%S}"


# ── forward test ──────────────────────────────────────────────────────────────
def evaluate_forward(rows: pd.DataFrame, start_ts: int = FORWARD_START_TS) -> ForwardResult:
    """Exactly forward_test.py: R1 underdog, 22-23 UTC, minutes 12/13, 5-30c, 1 contract, Kalshi fee, SPRT D=3c / sd 40c.
    A boundary crossing is terminal (first crossing wins), as in a sequential test; forward_test.py only looks at the final LLR."""
    s = extract_signals(rows, FORWARD_RULE)
    s = s[s.open_ts >= start_ts].sort_values("open_ts")
    if s.empty:
        return ForwardResult()
    fee = {c: fee_per_contract("Kalshi", float(c)) for c in s.cost.unique()}
    pnl = (s.win - s.cost - s.cost.map(fee)).values.astype(float)
    llr_path = np.cumsum((SPRT_D / SPRT_S ** 2) * (pnl - SPRT_D / 2))
    n = len(pnl)
    verdict, decided = "CONTINUE", None
    hit_acc, hit_rej = np.where(llr_path >= SPRT_ACCEPT)[0], np.where(llr_path <= SPRT_REJECT)[0]
    first_acc = hit_acc[0] if len(hit_acc) else None
    first_rej = hit_rej[0] if len(hit_rej) else None
    if first_acc is not None and (first_rej is None or first_acc < first_rej):
        verdict, decided = "ACCEPT", int(s.open_ts.iloc[first_acc])
    elif first_rej is not None:
        verdict, decided = "REJECT", int(s.open_ts.iloc[first_rej])
    elif n >= FORWARD_MAX_TRADES and pnl.mean() < -0.03:
        verdict = "REJECT"
    return ForwardResult(n=n, win=float(s.win.mean()), avg_cost=float(s.cost.mean()), ev_c=float(pnl.mean() * 100),
                         se_c=float(pnl.std(ddof=1) / np.sqrt(n) * 100) if n > 1 else float("nan"), cum_c=float(pnl.sum() * 100),
                         llr=float(llr_path[-1]), verdict=verdict, decided_at=decided)


def forward_needs_run(state: dict, rows: pd.DataFrame, now: float) -> tuple[bool, str]:
    fw = state.get("forward") or {}
    hi = int(rows.open_ts.max()) + WINDOW if len(rows) else 0
    if not fw:
        return True, "first check on this PC"
    ago = _ago(now - fw.get("ts", 0))
    if hi > fw.get("data_hi", 0):
        k = (hi - fw.get("data_hi", 0)) // WINDOW
        return True, f"{k} new windows settled since the last check ({ago} ago)"
    if now - fw.get("ts", 0) > RERUN_FORWARD_AFTER_S:
        return True, f"last check was {ago} ago"
    return False, f"up to date (last check {ago} ago, no new settled windows)"


# ── individual checks ─────────────────────────────────────────────────────────
def next_decision(now: float, p: StrategyParams, grace: float = 25.0) -> Optional[float]:
    w0 = int(now // WINDOW) * WINDOW - WINDOW
    for k in range(0, 4 * 49):
        w = w0 + WINDOW * k
        if datetime.fromtimestamp(w, timezone.utc).hour in p.hours:
            for m in sorted(p.minutes):
                d = w + 60 * m
                if d >= now - grace:
                    return float(d)
    return None


def timing_check(cfg: LiveConfig, now: float) -> Check:
    nd = next_decision(now, cfg.params)
    hrs = fmt_int_list(cfg.params.hours)
    if nd is None:
        return Check("timing", "Schedule", WARN, f"no decision point found for hours {hrs} UTC")
    mins = (nd - now) / 60
    t = datetime.fromtimestamp(now, timezone.utc)
    msg = f"now {t:%H:%M:%S} UTC; next decision {_hhmm(nd)} UTC (in {mins:.0f} min). Strategy hours (UTC window open): {hrs}."
    if mins > 6 * 60:
        return Check("timing", "Schedule", INFO, msg + " Nothing will trade for hours - start closer to the window (about 21:45 UTC).")
    return Check("timing", "Schedule", PASS, msg)


def data_check(rows: pd.DataFrame, now: float, updated: bool, log) -> Check:
    if rows.empty:
        return Check("data", "Local history", FAIL, "no local data - click 'Update data' (needs the public Kalshi API)")
    lo, hi = data_range(rows)
    lag = now - hi.timestamp()
    s = f"{len(rows):,} decision rows, {lo:%Y-%m-%d} → {hi:%Y-%m-%d %H:%M} UTC ({_ago(lag)} behind){' — updated just now' if updated else ''}"
    if lag <= 4 * 3600:
        return Check("data", "Local history", PASS, s + (" — Kalshi publishes candle history with a ~2 h delay; this is normal" if lag > 3600 else ""))
    if lag <= 36 * 3600:
        return Check("data", "Local history", WARN, s + ". The probability estimate will not include the most recent windows.")
    return Check("data", "Local history", FAIL, s + ". Too old to estimate the current regime - run 'Update data'.")


def coverage_check(rows: pd.DataFrame, cfg: LiveConfig, now: float, coverage_fn=coverage) -> Check:
    try:
        have, tot = coverage_fn(rows, cfg.params, int(now - 28 * 86400), int(now))
    except Exception as e:                                           # noqa: BLE001
        return Check("data", "Candle coverage (28 d)", INFO, f"could not verify ({type(e).__name__})")
    if tot == 0:
        return Check("data", "Candle coverage (28 d)", WARN, "no markets in the strategy hours for the last 28 days")
    frac = have / tot
    s = f"{have}/{tot} markets in the strategy hours have candle data ({frac * 100:.0f}%)"
    return Check("data", "Candle coverage (28 d)", PASS if frac >= 0.97 else WARN, s + ("" if frac >= 0.97 else " — missing windows make the edge estimate unreliable; run 'Update data'"))


def forward_check(rows: pd.DataFrame, now: float, state: dict, coverage_fn=coverage) -> tuple[list, ForwardResult]:
    need, why = forward_needs_run(state, rows, now)
    fr = evaluate_forward(rows)
    state["forward"] = dict(ts=now, data_hi=int(rows.open_ts.max()) + WINDOW if len(rows) else 0, n=fr.n, llr=fr.llr, verdict=fr.verdict, ev_c=fr.ev_c)
    detail = fr.line() + f".  Re-evaluated on {'fresh' if need else 'unchanged'} data: {why}."
    out = []
    if fr.verdict == "REJECT":
        out.append(Check("evidence", "Forward test (pre-registered R1)", WARN, detail + f" The frozen rule failed its own test: expect little or no edge. The engine's learner gate still decides whether it bets at all; "
                                                                                    f"a {CAUTION_CAP * 100:.1f}% per-trade cap is suggested (optional).", cap=CAUTION_CAP))
    elif fr.verdict == "ACCEPT":
        out.append(Check("evidence", "Forward test (pre-registered R1)", PASS, detail))
    else:
        early = f" Only {fr.n} trades so far - at ~3 trades/day the test needs weeks." if fr.n < 60 else ""
        out.append(Check("evidence", "Forward test (pre-registered R1)", WARN, detail + early + f" Not yet confirmed: a {CAUTION_CAP * 100:.1f}% per-trade cap is suggested (optional).", cap=CAUTION_CAP))
    try:
        have, tot = coverage_fn(rows, FORWARD_RULE, FORWARD_START_TS, int(now))
        if tot and have < tot:
            out.append(Check("evidence", "Forward-period data", WARN, f"{tot - have} of {tot} forward-period markets have no candle data, so the forward test undercounts - run 'Update data'."))
    except Exception:                                                # noqa: BLE001
        pass
    return out, fr


def window_stats(sig: pd.DataFrame, cfg: LiveConfig, now: float, days: int) -> dict:
    s = sig[(sig.close_ts <= now) & (sig.open_ts >= now - days * 86400)]
    if s.empty:
        return dict(n=0, win=float("nan"), price=float("nan"), ev=float("nan"), se=float("nan"))
    px = np.minimum(s.cost.values + cfg.paper_slippage, 0.99)
    fee = np.array([fee_per_contract(cfg.fee_model, float(c)) for c in px])
    pnl = s.win.values - px - fee
    n = len(pnl)
    return dict(n=n, win=float(s.win.mean()), price=float(s.cost.mean()), ev=float(pnl.mean()), se=float(pnl.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"))


def regime_check(rows: pd.DataFrame, cfg: LiveConfig, now: float) -> tuple[Check, Check]:
    """Is the edge there *now*?  Net EV per contract (after slippage and fee) of the preset's own signals over the last 28 and 14 days."""
    pe = cfg.params
    if cfg.p_model == "by_hour" and cfg.core_hours:                     # all-hours scout preset: judge the core hours (pooled all-hours is dominated by cold hours the engine does not trade)
        pe = StrategyParams(hours=tuple(cfg.core_hours), minutes=pe.minutes, cost_min=pe.cost_min, cost_max=pe.cost_max, max_spread=pe.max_spread, legs=pe.legs,
                            fav_min=pe.fav_min, fav_max=pe.fav_max)
    sig = extract_signals(rows, pe)
    if cfg.ev_rules:                                                    # EV-filter preset: judge only the signals the filter would have taken (rule evaluated at each signal's entry)
        est = build_stats(rows, cfg)
        recent = sig[sig.open_ts >= min(now - 28 * 86400, FORWARD_START_TS)]
        keep = [est.passes(int(h), float(o + 60 * m)) for h, o, m in zip(recent.hour.values, recent.open_ts.values, recent.minute.values)]
        sig = recent[np.array(keep, dtype=bool)] if len(recent) else recent
    w28, w14 = window_stats(sig, cfg, now, 28), window_stats(sig, cfg, now, 14)
    fmt = lambda w: (f"n={w['n']}, win {w['win'] * 100:.0f}% at avg {w['price'] * 100:.0f}c, net EV {w['ev'] * 100:+.1f}c (se {w['se'] * 100:.1f}c)" if w["n"] else "no signals")
    detail = f"last 28 d: {fmt(w28)} | last 14 d: {fmt(w14)}  [net of {cfg.paper_slippage * 100:.0f}c slippage + {cfg.fee_model} fee]"
    hurdle = cfg.min_edge
    if w28["n"] < 20:
        c28 = Check("evidence", "Trailing edge (your preset)", WARN, detail + " Too few signals to judge the current regime.", cap=CAUTION_CAP)
    elif w28["ev"] < 0:
        c28 = Check("evidence", "Trailing edge (your preset)", WARN, detail + " Negative over the last 28 days: no regime to harvest right now. The engine's learner gate sits out until it sees one - leaving it running costs nothing and catches the next one.", cap=CAUTION_CAP)
    elif w28["ev"] < hurdle:
        c28 = Check("evidence", "Trailing edge (your preset)", WARN, detail + f" Positive but below the {hurdle * 100:.0f}c conviction hurdle: the engine will mostly sit out.", cap=CAUTION_CAP)
    elif w14["n"] >= 10 and w14["ev"] < 0:
        c28 = Check("evidence", "Trailing edge (your preset)", WARN, detail + " The last two weeks are negative: the edge may be fading.", cap=CAUTION_CAP)
    else:
        c28 = Check("evidence", "Trailing edge (your preset)", PASS, detail)
    since = sig[(sig.open_ts >= FORWARD_START_TS) & (sig.close_ts <= now)]
    fw = window_stats(since, cfg, now, 10_000)
    c_fw = Check("evidence", "Your preset since the freeze (Oct 8)", INFO, f"{fmt(fw)}  [not part of the pre-registered test]")
    return c28, c_fw


def gate_check(rows: pd.DataFrame, cfg: LiveConfig, now: float) -> Check:
    if cfg.prob_source == "manual":
        return Check("evidence", "Learner gate", WARN, f"manual win probability {cfg.manual_p * 100:.0f}% bypasses the evidence gate: only use a number you can justify.", cap=CAUTION_CAP)
    stats = build_stats(rows, cfg)
    legs = cfg.params.legs
    thin = [(lg, stats.n_total(now, lg)) for lg in legs if stats.n_total(now, lg) < MIN_HISTORY]
    if thin:
        return Check("evidence", "Learner gate", WARN, "not enough history: " + ", ".join(f"{lg} n={n} (need {MIN_HISTORY})" for lg, n in thin) + " - the engine will only record signals until it has enough.")
    if cfg.ev_rules:
        hrs = stats.passing_hours(now)
        txt = f"EV filter ({describe_rules(cfg.ev_rules)}, >= {cfg.ev_min_n} signals per window)"
        if not hrs:
            return Check("evidence", "EV filter", WARN, txt + ": no hour of the day passes right now - the engine will only record signals; live money would sit idle.")
        return Check("evidence", "EV filter", PASS, txt + f": hours passing right now (UTC): {', '.join(f'{h:02d}' for h in hrs)}. Each is re-checked at every decision with the signals settled by then.")
    leg = legs[0]
    typ = (cfg.params.cost_min + cfg.params.cost_max) / 2 if leg == "DOG" else (cfg.params.fav_min + cfg.params.fav_max) / 2
    hour = next((h for h in cfg.params.hours if h >= 22), cfg.params.hours[0])
    if auto_active(cfg):
        snap = snapshot(cfg.auto, stats, leg, hour, typ, now, cfg.capital, cfg.capital, cfg.fee_model, cfg.paper_slippage, cfg.min_edge)
        live_h = [(h, r) for h, r in snap["live_hours"] if h in cfg.params.hours]
        base = (f"AUTO: learners {'/'.join(f'{h:g}' for h in cfg.auto.half_lives)} d (weights {'/'.join(f'{w * 100:.0f}%' for w in snap['weights'])}); at {typ * 100:.0f}c the {snap['d_fast'] * 100:+.1f}c fast / "
                f"{snap['d_slow'] * 100:+.1f}c slow edge estimates (hour {hour:02d}) are {snap['regime']}")
        if not live_h:
            return Check("evidence", "Learner gate (Auto)", WARN, base + " → every tradable hour is COLD: it will only record signals tonight; live money would sit idle.")
        h0 = live_h[0][0]
        d = decide(cfg.auto, stats, leg, h0, typ, now, cfg.capital, cfg.capital, cfg.capital, None, cfg.fee_model, cfg.paper_slippage, cfg.min_edge)
        hrs = ", ".join(f"{h:02d} {r}" for h, r in live_h)
        if d.sizing.contracts:
            return Check("evidence", "Learner gate (Auto)", PASS, base + f" → OPEN in hours {hrs}: Kelly x{d.mult:.2f} (ladder step {d.tier + 1}), stake cap {d.cap * 100:.1f}% → about ${d.sizing.stake:,.2f} ({d.sizing.contracts:,} contracts) of ${cfg.capital:,.0f}.")
        return Check("evidence", "Learner gate (Auto)", WARN, base + f" → hours {hrs} are not COLD but Kelly sizes to zero ({d.sizing.reason}).")
    p, n, _ = stats.p(leg, hour, typ, now)
    px = min(typ + cfg.paper_slippage, 0.99)
    z = size_bet(cfg.capital, p, px, cfg.fee_model, cfg.kelly_mult, cfg.max_frac, cfg.max_contracts, cfg.min_edge)
    base = f"{leg} at {typ * 100:.0f}c: learned win probability {p * 100:.1f}% vs all-in cost {z.cost_all * 100:.1f}c (n={n} signals, {('half-life ' + format(cfg.halflife_days, '.0f') + ' d') if cfg.halflife_days else 'last ' + str(cfg.lookback_days or 'all') + ' days'})"
    if z.contracts:
        return Check("evidence", "Learner gate", PASS, base + f" → OPEN: the engine would stake ${z.stake:,.2f} ({z.contracts:,} contracts, {z.f_used * 100:.2f}% of ${cfg.capital:,.0f}).")
    return Check("evidence", "Learner gate", WARN, base + f" → CLOSED ({z.reason}). It will only record signals tonight; live money would sit idle.")


def radar_check(rows: pd.DataFrame, cfg: LiveConfig, now: float) -> Check:
    """Scout all 24 hours (nothing is traded): is the edge where the preset trades, or has it moved?"""
    rc = radar.RadarConfig(cost_min=cfg.params.cost_min, cost_max=cfg.params.cost_max, minutes=cfg.params.minutes, fee_model=cfg.fee_model, slippage=cfg.paper_slippage, min_edge=cfg.min_edge,
                           hour_k=cfg.hour_k, halflife_days=cfg.halflife_days or radar.SCOUT_HALFLIFE, core_hours=tuple(cfg.core_hours), core_k=cfg.core_k, near_k=cfg.near_k, near_width=cfg.near_width,
                           ev_rules=tuple(cfg.ev_rules), ev_min_n=cfg.ev_min_n)
    active = tuple(cfg.core_hours) if (cfg.p_model == "by_hour" and cfg.core_hours) else tuple(cfg.params.hours)
    df = radar.hour_radar(rows, now, rc, active)
    state, txt = radar.summarize(df, active, rc)
    if state == "moved":
        return Check("evidence", "Regime radar (all 24 hours)", WARN, txt, cap=CAUTION_CAP)
    return Check("evidence", "Regime radar (all 24 hours)", PASS if state == "ok" else INFO, txt)


def config_checks(cfg: LiveConfig, live: bool) -> list:
    out = []
    p = cfg.params
    core_scout = cfg.p_model == "by_hour" and cfg.core_k > 0 and bool(cfg.core_hours) and set(cfg.core_hours) <= VALIDATED_HOURS
    tested = p.legs == ("DOG",) and (set(p.hours) <= VALIDATED_HOURS or core_scout) and p.cost_max <= 0.45
    if tested and core_scout and not set(p.hours) <= VALIDATED_HOURS:
        out.append(Check("risk", "Preset", PASS, f"core hours {','.join(map(str, cfg.core_hours))} + scout of all other hours (shrinkage {cfg.hour_k:.0f}{f', {cfg.near_k:.0f} within ±{cfg.near_width} h of the core' if cfg.near_k else ''}) — the tested adaptive design."))
    elif tested:
        out.append(Check("risk", "Preset", PASS, f"{p.describe()} — within the tested family (evening underdog)."))
    else:
        out.append(Check("risk", "Preset", WARN, f"{p.describe()} — outside the tested family; all-day and favourite variants lost money after costs in testing.", cap=CAUTION_CAP))
    if auto_active(cfg):
        a = cfg.auto
        _t0, m0, c0 = a.ladder[0]
        big, over, vcap = c0 > 0.05 or m0 > 1.0, c0 > AUTO_VALIDATED_CAP + 1e-9 or m0 > VALIDATED_MULT + 1e-9, AUTO_VALIDATED_CAP
        s = (f"AUTO sizing, ladder {format_ladder(a.ladder)} (first step {m0:g}x Kelly / {c0 * 100:g}% cap, shrinking as equity grows), evidence haircut z={a.z:g}, liquidity {a.participation * 100:g}% of last-minute volume "
             f"(ceiling {a.max_contracts:,}), min edge {cfg.min_edge * 100:.1f}c, daily loss limit {cfg.daily_loss_limit * 100:.0f}%, drawdown stop {cfg.max_drawdown_stop * 100:.0f}%, drawdown throttle {'on' if a.throttle else 'OFF'}")
    else:
        vcap = VALIDATED_CAP
        big = cfg.max_frac > 0.05 or cfg.kelly_mult > 1.0
        over = cfg.max_frac > VALIDATED_CAP + 1e-9 or cfg.kelly_mult > VALIDATED_MULT + 1e-9
        s = (f"Kelly {cfg.kelly_mult:.2f}x, cap {cfg.max_frac * 100:.1f}% per trade, min edge {cfg.min_edge * 100:.1f}c, daily loss limit {cfg.daily_loss_limit * 100:.0f}%, "
             f"drawdown stop {cfg.max_drawdown_stop * 100:.0f}%, drawdown throttle {'on' if cfg.dd_throttle else 'OFF'}")
    if big:
        out.append(Check("risk", "Sizing", FAIL, s + f" — exceeds the validated limits ({VALIDATED_MULT}x Kelly, {vcap * 100:.0f}% cap) by far."))
    elif over or cfg.max_drawdown_stop > 0.25 + 1e-9 or cfg.daily_loss_limit > 0.10 + 1e-9:
        out.append(Check("risk", "Sizing", WARN, s + f" — above the validated limits ({VALIDATED_MULT}x, {vcap * 100:.0f}% cap, 10% daily, 25% drawdown)."))
    else:
        out.append(Check("risk", "Sizing", PASS, s))
    if live and cfg.max_slippage > 0.05:
        out.append(Check("risk", "Order limit", WARN, f"live IOC limit = quoted ask + {cfg.max_slippage * 100:.0f}c is loose; testing assumed ~2c slippage"))
    return out


def previous_session_check(state: dict, now: float) -> Check:
    ls = state.get("last_session")
    if not ls:
        return Check("risk", "Previous session", INFO, "no earlier session recorded on this PC")
    ago = now - ls.get("ts", 0)
    why = str(ls.get("why", ""))
    bad = any(k in why.lower() for k in ("kill", "halt", "drawdown", "error", "broker"))
    s = f"{ls.get('mode', '?')} session ended {_ago(ago)} ago: {why} (P&L {ls.get('pnl', 0):+,.2f}, {ls.get('trades', 0)} trades)"
    if bad and ago < 7 * 86400:
        return Check("risk", "Previous session", WARN, s + " — it was stopped by a safety mechanism; understand why before restarting.")
    return Check("risk", "Previous session", PASS, s)


def kalshi_checks(now: float, quotes=None, http_get=requests.get, clock: Callable[[], float] = time.time, env: str = "production") -> list:
    out = []
    try:
        t_req = clock()
        r = http_get(base_url(env) + "/exchange/status", timeout=6)
        t_req = (t_req + clock()) / 2                     # local time at the middle of the request, to compare with the server's Date header
        if r.status_code != 200:
            out.append(Check("feed", "Kalshi exchange", WARN, f"status endpoint returned HTTP {r.status_code}"))
        else:
            j = r.json()
            ok = bool(j.get("exchange_active", True)) and bool(j.get("trading_active", True))
            out.append(Check("feed", "Kalshi exchange", PASS if ok else FAIL, "exchange and trading active" if ok else f"exchange reports inactive: {j}"))
            d = r.headers.get("Date")
            if d:
                skew = parsedate_to_datetime(d).timestamp() - t_req          # the Date header has 1 s resolution
                st = FAIL if abs(skew) > 20 else WARN if abs(skew) > 5 else PASS
                out.append(Check("feed", "PC clock", st, f"{skew:+.1f}s vs Kalshi's server clock" + ("" if st == PASS else " — decision timing and order signatures need a clock within a few seconds; sync Windows time")))
    except Exception as e:                                           # noqa: BLE001
        out.append(Check("feed", "Kalshi exchange", FAIL, f"unreachable ({type(e).__name__}: {e}) — the strategy's signal source is down"))
        return out
    try:
        q = quotes or KalshiQuotes(base=base_url(env))
        w = int(now // WINDOW) * WINDOW
        tk = q.window_ticker(w)
        qq = q.quote(tk) if tk else None
        if not tk or not qq:
            out.append(Check("feed", "Live quote", FAIL, f"no quote for the current window ({datetime.fromtimestamp(w, timezone.utc):%H:%M} UTC)"))
        else:
            bid, ask = qq
            sp = ask - bid
            out.append(Check("feed", "Live quote", PASS if 0 <= sp <= 0.06 else WARN, f"{tk}: YES bid {bid:.2f} / ask {ask:.2f}" + ("" if 0 <= sp <= 0.06 else f" (spread {sp * 100:.0f}c is wider than the 6c the strategy accepts)")))
    except Exception as e:                                           # noqa: BLE001
        out.append(Check("feed", "Live quote", FAIL, f"quote request failed ({type(e).__name__}: {e})"))
    return out


def account_checks(cfg: LiveConfig, client, now: float, quotes=None) -> list:
    out = []
    q = quotes or KalshiQuotes(base=client.base)
    try:
        shard = q.exchange_index() if hasattr(q, "exchange_index") else None          # the exchange shard BTC 15-minute orders are funded from
        bal = client.balance(shard)
        tot = client.balance() if shard is not None else bal
        bp = bal["buying_power"]
        name = q.shard_names().get(shard) if (shard is not None and hasattr(q, "shard_names")) else None
        where = (f" on exchange shard {shard}" + (f" ({name})" if name else "")) if shard is not None else ""
        split = ", ".join(f"shard {i}: ${v:,.2f}" for i, v in sorted(tot.get("breakdown", {}).items()))
        elsewhere = tot["buying_power"] - bp if shard is not None else 0.0
        want = min(cfg.capital, cfg.fund_max) if cfg.fund_max > 0 else cfg.capital
        if shard is not None and cfg.auto_fund and bp < want - 1.0 and elsewhere >= 1.0:
            move = min(want - bp, elsewhere)
            st, msg = (PASS if bp + elsewhere >= 0.999 * want else WARN), (f"AUTO-FUND is on: ${bp:,.2f}{where} now; the session will move ${move:,.2f} from your other shards "
                                                                              f"(total cash ${tot['buying_power']:,.2f}: {split}) onto it before trading - cash only moves between shards of your own account")
        elif shard is not None and bp < 1.0:
            st, msg = FAIL, (f"NO CASH{where}, where BTC 15-minute orders are funded from - Kalshi would reject every order. Your total cash is ${tot['buying_power']:,.2f}" + (f" ({split})" if split else "") +
                             f". The Kalshi app moves cash between shards by itself, the API does not: transfer cash to shard {shard} (Exchange Indexes page in your Kalshi account settings), then run this check again")
        elif bp >= cfg.capital:
            st, msg = PASS, f"buying power ${bp:,.2f}{where} covers the ${cfg.capital:,.2f} session capital"
        elif bp >= 0.5 * cfg.capital:
            st, msg = WARN, f"buying power ${bp:,.2f}{where} is below the ${cfg.capital:,.2f} session capital - stakes will be capped to what is there" + (f" (total cash ${tot['buying_power']:,.2f}: {split})" if shard is not None and split else "")
        else:
            st, msg = FAIL, f"buying power ${bp:,.2f}{where} is far below the ${cfg.capital:,.2f} session capital - fund it (cash on another exchange shard does not count) or lower the capital" + (f" (total cash ${tot['buying_power']:,.2f}: {split})" if shard is not None and split else "")
        out.append(Check("account", "Buying power", st, msg + f" (cash ${bal['balance']:,.2f}, portfolio value ${bal['portfolio_value']:,.2f}; {client.env})"))
    except kalshi.ApiError as e:
        out.append(Check("account", "Account access", FAIL, f"balance request failed: {e}"))
        return out
    skew = client.skew_s
    if skew is not None:
        st = FAIL if abs(skew) > 20 else WARN if abs(skew) > 5 else PASS
        out.append(Check("account", "Clock vs Kalshi", st, f"{skew:+.1f}s" + ("" if st == PASS else " — signed requests are rejected when the clock is off")))
    try:
        orders = client.open_orders()
        out.append(Check("account", "Resting orders", PASS if not orders else WARN, "none" if not orders else f"{len(orders)} resting orders on the account - cancel them or they will interact with this session"))
    except kalshi.ApiError as e:
        out.append(Check("account", "Resting orders", WARN, f"could not read: {e}"))
    try:
        pos = client.positions()
        btc = [str(p.get("ticker", "")) for p in pos if str(p.get("ticker", "")).startswith(kalshi.SERIES + "-")]
        out.append(Check("account", "Open positions", PASS if not btc else WARN, f"{len(pos)} positions, none in BTC 15-minute markets" if not btc else f"{len(btc)} unsettled BTC 15-minute positions from an earlier session: {', '.join(btc[:3])}"))
    except kalshi.ApiError as e:
        out.append(Check("account", "Open positions", WARN, f"could not read: {e}"))
    try:                                                      # the order path rounds limit prices to whole cents: the live market must accept that
        tk = q.window_ticker(int(now // WINDOW) * WINDOW)
        m = q.market_info(tk) if (tk and hasattr(q, "market_info")) else None
        if not m:
            out.append(Check("account", "Order price ticks", WARN, "no open market to inspect right now - retry nearer a window"))
        else:
            ranges = [r for r in (m.get("price_ranges") or []) if float(r.get("end", 0)) > 0.10 and float(r.get("start", 1)) < 0.60]
            fine = all(float(r["step"]) <= 0.01 + 1e-9 and abs(0.01 / float(r["step"]) - round(0.01 / float(r["step"]))) < 1e-6 for r in ranges) if ranges else None
            if m.get("status") != "active":
                out.append(Check("account", "Order price ticks", WARN, f"{tk} is '{m.get('status')}', not active"))
            elif fine is None:
                out.append(Check("account", "Order price ticks", INFO, f"{tk}: no price_ranges in the market data; assuming 1c ticks"))
            else:
                out.append(Check("account", "Order price ticks", PASS if fine else WARN, f"{tk}: price steps {', '.join(format(float(r['step']), 'g') for r in ranges)} in the 10c-60c range" + ("" if fine else " - coarser than the 1c the order path assumes")))
    except Exception as e:                                    # noqa: BLE001
        out.append(Check("account", "Order price ticks", WARN, f"could not check ({type(e).__name__})"))
    return out


# ── orchestration ─────────────────────────────────────────────────────────────
def _safe(group: str, name: str, fn: Callable[[], object], bad: str = FAIL):
    try:
        return fn()
    except Exception as e:                                           # noqa: BLE001
        return [Check(group, name, bad, f"check crashed: {type(e).__name__}: {e}")]


def run_preflight(cfg: LiveConfig, *, live: bool, key_path: str = "", key_id: str = "", env: str = "production", client=None, log: Callable[[str], None] = print, now: Optional[float] = None,
                  refresh: bool = True, rows: Optional[pd.DataFrame] = None, loader: Callable[[], pd.DataFrame] = load_rows,
                  updater: Callable[[Callable[[str], None]], bool] = update_data, quotes=None, http_get=requests.get,
                  coverage_fn=coverage, state_directory: Optional[Path] = None, cooldown_s: float = 1.5) -> Report:
    fixed_now = now                                  # tests pin the clock; in real use every step re-reads it (the data update can take minutes)
    clock: Callable[[], float] = (lambda: fixed_now) if fixed_now is not None else time.time
    now = clock()
    rep = Report(created=now, live=live, capital=cfg.capital)
    state = load_state(state_directory)

    log("[1/6] schedule and local data")
    if rows is None:
        rows = _safe("data", "Local history", loader, FAIL)
        if isinstance(rows, list):
            rep.checks += rows
            rows = pd.DataFrame(columns=["ticker", "open_ts", "minute", "bid", "ask", "yes"])
    hi = int(rows.open_ts.max()) + WINDOW if len(rows) else 0
    if refresh and now - hi > STALE_UPDATE_S and now - state.get("data_update_ts", 0) > MIN_UPDATE_GAP_S:
        log(f"      newest settled window is {_ago(now - hi)} old -> updating from Kalshi (public API)…")
        try:
            ok = updater(log)
            time.sleep(cooldown_s)                                     # the download just used Kalshi's public rate limit: let it recover before the account checks
            rows = loader()
            rep.updated_data = True
            state["data_update_ts"] = clock()
            save_state(state, state_directory)
            if not ok:
                rep.notes.append("the data updater reported a problem; continuing with what is on disk")
        except Exception as e:                                       # noqa: BLE001
            rep.notes.append(f"data update failed ({type(e).__name__}: {e}); continuing with what is on disk")
    rep.rows = rows
    now = clock()
    rep.checks.append(timing_check(cfg, now))
    rep.checks.append(data_check(rows, now, rep.updated_data, log))
    if len(rows):
        rep.checks += _safe("data", "Candle coverage (28 d)", lambda: [coverage_check(rows, cfg, now, coverage_fn)], WARN)

    log("[2/6] pre-registered forward test")
    if len(rows):
        res = _safe("evidence", "Forward test (pre-registered R1)", lambda: forward_check(rows, now, state, coverage_fn))
        if isinstance(res, tuple):
            checks, rep.forward = res
            rep.checks += checks
            save_state(state, state_directory)
        else:
            rep.checks += res

    log("[3/6] trailing edge and learner gate")
    if len(rows):
        res = _safe("evidence", "Trailing edge (your preset)", lambda: list(regime_check(rows, cfg, now)))
        rep.checks += res
        rep.checks += _safe("evidence", "Learner gate", lambda: [gate_check(rows, cfg, now)])
        rep.checks += _safe("evidence", "Regime radar (all 24 hours)", lambda: [radar_check(rows, cfg, now)], WARN)

    log("[4/6] risk configuration")
    rep.checks += _safe("risk", "Sizing", lambda: config_checks(cfg, live))
    rep.checks.append(previous_session_check(state, now))

    log("[5/6] Kalshi quote feed and clock")
    rep.checks += _safe("feed", "Kalshi exchange", lambda: kalshi_checks(clock(), quotes, http_get, clock, env))

    log("[6/6] Kalshi account" if live else "[6/6] Kalshi account (skipped in paper mode)")
    if not live:
        rep.checks.append(Check("account", "Kalshi account", SKIP, "paper mode sends no orders - the account is checked when you start in LIVE mode"))
    else:
        if client is None:
            try:
                if not key_path:
                    raise kalshi.CredentialError("no key file path given")
                client = kalshi.KalshiClient(kalshi.load_credentials(key_path, key_id), env)
            except Exception as e:                                   # noqa: BLE001
                rep.checks.append(Check("account", "Account access", FAIL, f"cannot open the Kalshi client: {e}"))
        if client is not None:
            rep.client = client
            rep.checks += _safe("account", "Account access", lambda: account_checks(cfg, client, clock(), quotes))
    return rep


def save_report(rep: Report, d: Optional[Path] = None) -> Optional[Path]:
    d = d or (state_dir() / "live")                  # TRADER_STATE_DIR (tests) keeps reports out of the real data folder
    try:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"preflight_{datetime.fromtimestamp(rep.created, timezone.utc):%Y%m%d_%H%M%S}.txt"
        p.write_text(rep.text(), encoding="utf-8")
        return p
    except OSError:
        return None
