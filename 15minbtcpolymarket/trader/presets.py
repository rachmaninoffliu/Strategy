"""The strategy presets of the app (Qt-free): one definition for the GUI's preset box, the command-line tools and the tests.

``preset_config`` turns a preset into the BacktestConfig the GUI's Backtest tab would build with that preset selected and every other box at its default, so scripts reproduce what the GUI shows."""
from __future__ import annotations

from datetime import date
from typing import Optional

from .adaptive import AutoConfig
from .backtest import BacktestConfig
from .evfilter import DEFAULT_MIN_N
from .strategy import StrategyParams, parse_int_list

PRESETS = {
    "R2+ · Core 22-23 UTC + all-hours scout (default)": dict(hours="0-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                           pmodel=1, pbasis=0, look=45, half=14, hourk=100, core="22,23", corek=10, near=50, nearw=2),
    "AUTO · Adaptive regime + equity-scaled sizing": dict(hours="0-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                         pmodel=1, pbasis=0, look=45, half=14, hourk=100, core="22,23", corek=10, near=50, nearw=2, auto=True),
    "R2 · 22-23 UTC only (hours hard-wired)": dict(hours="22,23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                  pmodel=0, pbasis=0, look=45, half=10),
    "R3 · More trades, 20-23 UTC, hour-adaptive": dict(hours="20-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                      pmodel=1, pbasis=0, look=45),
    "R1 · Original (22-23 UTC, 5-30¢)": dict(hours="22,23", minutes="12,13", dog=True, cmin=0.05, cmax=0.30, fav=False, fmin=0.85, fmax=0.97,
                                            pmodel=0, pbasis=0, look=45),
    "All-day underdog, adaptive (experimental)": dict(hours="0-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                     pmodel=1, pbasis=0, look=45),
    "All-day underdog + favourite (experimental)": dict(hours="0-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=True, fmin=0.85, fmax=0.97,
                                                       pmodel=1, pbasis=0, look=45),
    "Favourite only — high win rate (experimental)": dict(hours="0-23", minutes="5-11", dog=False, cmin=0.15, cmax=0.35, fav=True, fmin=0.85, fmax=0.97,
                                                         pmodel=1, pbasis=0, look=45),
    "FILTER-14 · Trade only hours whose 14 d net EV > 10c": dict(hours="0-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                                    pmodel=0, pbasis=0, look=0, half=0, evr=((14, 0.10),), evn=8),
    "FILTER-28 · Trade only hours whose 28 d net EV > 5c": dict(hours="0-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                                   pmodel=0, pbasis=0, look=0, half=0, evr=((28, 0.05),), evn=8),
    "FILTER-BOTH · Trade only hours with 14 d net EV > 10c AND 28 d > 5c": dict(hours="0-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                                                   pmodel=0, pbasis=0, look=0, half=0, evr=((14, 0.10), (28, 0.05)), evn=8),
    "PROJ-EV · Trade any hour while its projected EV > 0, stop when it turns negative": dict(hours="0-23", minutes="12,13", dog=True, cmin=0.15, cmax=0.35, fav=False, fmin=0.85, fmax=0.97,
                                                                                             pmodel=1, pbasis=0, look=45, half=14, hourk=100, edge=0.0),
    "Custom": None,
}


def find_preset(text: str) -> str:
    """The preset whose name starts with / contains ``text`` (case-insensitive), e.g. 'R2+' or 'auto'.  Raises ValueError when there is none or more than one."""
    t = text.strip().lower()
    names = [n for n, v in PRESETS.items() if v is not None]
    hits = [n for n in names if n.lower() == t] or [n for n in names if n.lower().startswith(t)] or [n for n in names if t in n.lower()]
    if len(hits) != 1:
        raise ValueError(f"preset '{text}' matches {len(hits)} presets; choose from: " + " | ".join(names))
    return hits[0]


def preset_params(v: dict) -> StrategyParams:
    return StrategyParams(hours=parse_int_list(v["hours"], 0, 23), minutes=parse_int_list(v["minutes"], 1, 14), cost_min=v["cmin"], cost_max=v["cmax"],
                          legs=tuple(n for n, on in (("DOG", v["dog"]), ("FAV", v["fav"])) if on), fav_min=v["fmin"], fav_max=v["fmax"])


def preset_config(name: str, start: date, end: date, capital: float = 1_000.0, auto: Optional[AutoConfig] = None, **overrides) -> BacktestConfig:
    """BacktestConfig of preset ``name`` (see ``find_preset``) over [start, end]; ``overrides`` replace any BacktestConfig field (slippage, kelly_mult, ...)."""
    v = PRESETS[find_preset(name)]
    core = parse_int_list(v["core"], 0, 23) if v.get("core") and v.get("corek", 0) > 0 else ()
    half = v.get("half", 0)
    kw = dict(start=start, end=end, capital=capital, params=preset_params(v), p_model="by_hour" if v["pmodel"] == 1 else "overall", p_basis="win_rate" if v["pbasis"] == 1 else "edge",
              hour_k=float(v.get("hourk", 40)), lookback_days=int(v.get("look", 45)), halflife_days=0.0 if half <= 0 else max(3.0, float(half)), core_hours=core, core_k=float(v.get("corek", 0)),
              near_k=float(v.get("near", 0)) if core else 0.0, near_width=int(v.get("nearw", 2)), auto=(auto or AutoConfig()) if v.get("auto") else None,
              ev_rules=tuple((int(d), float(t)) for d, t in v.get("evr", ())), ev_min_n=int(v.get("evn", DEFAULT_MIN_N)))
    if "edge" in v:                                      # a preset may carry its own conviction hurdle (PROJ-EV: any positive projected EV)
        kw["min_edge"] = float(v["edge"])
    kw.update(overrides)
    return BacktestConfig(**kw)
