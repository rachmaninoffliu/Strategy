"""
Strategy base class, the shared parameters, and the research context (no Qt).

A strategy only says how to SCORE every member at every bar (higher = expected to beat its factor-predicted return) and which bars it re-decides on. Everything else - universe, betas,
residuals, portfolio construction, hedging, costs, the backtest loop and the live engine - is shared, so the three strategies differ only in their alpha and share one GUI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import backtest as BT
from . import data as D
from . import factors as F
from . import features as X
from .portfolio import PortCfg

LAG = {"iex": 0, "sip_delayed": 1, "sip_realtime": 0}
REBAL_EVERY = {"15min": 1, "30min": 2, "1h": 4, "2h": 8}


@dataclass
class Param:
    key: str
    label: str
    default: object
    kind: str = "float"                 # float | int | choice | bool | text
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.1
    dec: int = 2
    choices: tuple = ()
    help: str = ""
    group: str = "Strategy"


COMMON = [
    Param("universe_n", "Universe: most liquid stocks", 100, "int", 30, 150, 10, help="Each month the N most liquid U.S. stocks (63-day median dollar volume, price >= $10), chosen with data up to the previous month end.", group="Universe & factor model"),
    Param("factor_model", "Factor model (beta extraction)", "etf", "choice", choices=("etf", "pca"), help="etf: SPY + the stock's sector SPDR (tradable hedge). pca: Avellaneda-Lee eigenportfolios (hedged by projection).", group="Universe & factor model"),
    Param("beta_days", "Beta window (days)", 126, "int", 40, 504, 21, help="Daily returns used for each morning's regression (ends at the previous close).", group="Universe & factor model"),
    Param("pca_k", "PCA factors", 15, "int", 1, 40, 1, group="Universe & factor model"),
    Param("rebalance", "Re-decide every", "1h", "choice", choices=("15min", "30min", "1h", "2h", "daily", "weekly", "monthly"), group="Trading"),
    Param("trade_time", "Trade at (daily / weekly / monthly)", "15:45", "text", help="ET time of the bar whose OPEN the orders fill at (09:30 ... 15:45).", group="Trading"),
    Param("feed", "Live data feed", "iex", "choice", choices=("iex", "sip_delayed", "sip_realtime"),
          help="iex = free real-time IEX bars (lag 0); sip_delayed = free SIP, 15 min late (the backtest delays every trade by one bar); sip_realtime = paid Algo Trader Plus.", group="Trading"),
    Param("method", "Portfolio", "quantile", "choice", choices=("quantile", "rank", "threshold"), group="Portfolio"),
    Param("q_in", "Enter: top / bottom fraction", 0.2, "float", 0.02, 0.5, 0.02, help="quantile method", group="Portfolio"),
    Param("q_out", "Hold until it leaves the top / bottom", 0.3, "float", 0.02, 0.5, 0.02, help="hysteresis (>= enter fraction)", group="Portfolio"),
    Param("gross", "Stock gross (x capital)", 1.0, "float", 0.1, 3.0, 0.1, help="1.0 = 50% long + 50% short (plus the ETF hedge)", group="Portfolio"),
    Param("max_w", "Max weight per name", 0.05, "float", 0.005, 0.2, 0.005, dec=3, group="Portfolio"),
    Param("hedge", "Factor hedge", "etf", "choice", choices=("etf", "project", "none"), group="Portfolio"),
    Param("max_gross", "Max gross: stocks + ETF hedges (x capital)", 1.9, "float", 0.2, 4.0, 0.1,
          help="the backtest scales the whole book down when a target needs more (the live engine does the same with its 'Max gross'); Reg-T allows 2x overnight", group="Portfolio"),
    Param("sector_neutral", "Rank within sector", False, "bool", group="Portfolio"),
    Param("flat_eod", "Flat overnight (intraday only)", False, "bool", group="Portfolio"),
    Param("band", "No-trade band", 0.0, "float", 0.0, 1.0, 0.05, help="skip a resize smaller than band x the average position", group="Portfolio"),
    Param("trade_rate", "Trade rate toward the target", 1.0, "float", 0.02, 1.0, 0.05,
          help="Garleanu-Pedersen (2013): each rebalance trades this fraction of the way from the current book to the target (1 = all the way). Lower = less turnover, slower signal capture",
          group="Portfolio"),
    Param("cost_model", "Stock trading cost", "spread", "choice", choices=("spread", "flat"),
          help="spread = each name's half spread at that bar, from a model fitted on SIP quotes 2017-2026 (a typical member at midday: ~1 bp in 2017-19, ~1.8 bp in "
               "2020-24, ~2.6 bp in 2025-26; +45% in the first hour, -27% in the last half hour, wider for less liquid names) x multiplier + slippage; flat = the flat cost below",
          group="Costs & period"),
    Param("cost_mult", "Spread-model multiplier", 1.0, "float", 0.0, 5.0, 0.25, group="Costs & period"),
    Param("slip_bp", "Slippage on top of the spread (bp)", 0.5, "float", 0.0, 10.0, 0.25, group="Costs & period"),
    Param("cost_bp", "Flat cost per side, stocks (bp)", 2.5, "float", 0.0, 20.0, 0.5, dec=1, group="Costs & period"),
    Param("etf_cost_bp", "Cost per side, ETFs (bp)", 0.5, "float", 0.0, 10.0, 0.1, dec=1, group="Costs & period"),
    Param("borrow_bp_yr", "Short borrow (bp / year)", 30.0, "float", 0.0, 500.0, 5.0, dec=0, group="Costs & period"),
    Param("fill", "Backtest fill price", "open", "choice", choices=("open", "vwap"),
          help="open = the next bar's first print (+ the per-side cost); vwap = that bar's VWAP - a stricter check for fast strategies whose open print may sit on the favourable side",
          group="Costs & period"),
    Param("start", "Backtest start", "2016-09-01", "text", group="Costs & period"),
    Param("split", "Out-of-sample from", "2023-01-01", "text", group="Costs & period"),
    Param("end", "Backtest end (blank = latest)", "", "text", group="Costs & period"),
]


class Strategy:
    KEY = ""                            # folder / file stem
    NAME = ""
    PREFIX = ""                         # environment-variable prefix of this strategy's Alpaca keys
    SUMMARY = ""
    PAPERS: list = []                   # (citation, url, what we use)
    HOWTO = ""
    PARAMS: list = []
    DEFAULTS: dict = {}                 # overrides of the shared defaults
    PRESETS: dict = {}
    FEATURES: list = []                 # features the score needs

    def __init__(self, values: Optional[dict] = None, folder: Optional[str] = None):
        self.v = {p.key: self.DEFAULTS.get(p.key, p.default) for p in self.all_params()}
        if values:
            self.v.update({k: val for k, val in values.items() if k in self.v})
        self.folder = folder or os.path.join(D.ROOT, self.KEY)

    @classmethod
    def all_params(cls) -> list:
        return list(cls.PARAMS) + COMMON

    # ── derived configs ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    @property
    def lag(self) -> int:
        return LAG.get(str(self.v["feed"]), 0)

    def factor_cfg(self) -> F.FactorCfg:
        return F.FactorCfg(model=self.v["factor_model"], beta_days=int(self.v["beta_days"]), pca_k=int(self.v["pca_k"]))

    def port_cfg(self) -> PortCfg:
        v = self.v
        return PortCfg(method=v["method"], q_in=float(v["q_in"]), q_out=max(float(v["q_out"]), float(v["q_in"])), s_open=float(v.get("s_open", 1.25)), s_close=float(v.get("s_close", 0.5)),
                       names=int(v.get("names", 40)), gross=float(v["gross"]), max_w=float(v["max_w"]), hedge=v["hedge"] if v["factor_model"] == "etf" or v["hedge"] != "etf" else "project",
                       sector_neutral=bool(v["sector_neutral"]), trade_rate=float(v["trade_rate"]))

    def bt_cfg(self) -> BT.BtCfg:
        v = self.v
        return BT.BtCfg(start=str(v["start"]), end=str(v["end"] or ""), split=str(v["split"]), lag=self.lag, cost_model=str(v["cost_model"]), cost_mult=float(v["cost_mult"]),
                        slip_bp=float(v["slip_bp"]), cost_bp=float(v["cost_bp"]), etf_cost_bp=float(v["etf_cost_bp"]), max_gross=float(v["max_gross"]),
                        borrow_bp_yr=float(v["borrow_bp_yr"]), flat_eod=bool(v["flat_eod"]), band=float(v["band"]), fill=str(v["fill"]))

    def trade_bar(self) -> int:
        hh, mm = [int(x) for x in str(self.v["trade_time"]).split(":")]
        return max(0, min(25, (hh * 60 + mm - 570) // 15))

    def decision_bars(self, p: D.Panel) -> np.ndarray:
        reb = str(self.v["rebalance"])
        if reb in REBAL_EVERY:
            return BT.schedule(p, every=REBAL_EVERY[reb])
        b = self.trade_bar() - 1 - self.lag                                                 # the data bar that is complete (and delivered) right before the trade bar opens
        return BT.schedule(p, bar_of_day=b, weekday=4 if reb == "weekly" else None, monthly=reb == "monthly")

    def warmup_days(self) -> int:
        return int(self.v["beta_days"]) + 30

    # ── the alpha ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    def score(self, ctx: "Context", log: Callable[[str], None] = lambda m: None) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        return self.NAME


# ── context ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
_PANELS: dict = {}
_RES: dict = {}


@dataclass
class Context:
    p: D.Panel
    res: F.Residuals
    member: np.ndarray                  # T x N bool
    member_d: np.ndarray                # days x N bool
    lag: int
    _feat: dict = field(default_factory=dict)

    def feature(self, name: str, **kw) -> np.ndarray:
        key = (name, tuple(sorted(kw.items())))
        if key not in self._feat:
            self._feat[key] = X.build(self.p, self.res, self.lag, names=[name], **kw)[name]
        return self._feat[key]

    def features(self, names: list, **kw) -> dict:
        todo = [n for n in names if (n, tuple(sorted(kw.items()))) not in self._feat]
        if todo:
            for n, a in X.build(self.p, self.res, self.lag, names=todo, **kw).items():
                self._feat[(n, tuple(sorted(kw.items())))] = a
        return {n: self._feat[(n, tuple(sorted(kw.items())))] for n in names}


def context(strat: Strategy, start=None, end=None, feed: str = "sip", log: Callable[[str], None] = print, extra: Optional[dict] = None, cache: bool = True,
            universe: Optional[pd.DataFrame] = None) -> Context:
    """Panel + residuals + membership for the strategy's universe between ``start`` (minus the warm-up) and ``end``."""
    n = int(strat.v["universe_n"])
    u = D.load_universe() if universe is None else universe
    if not len(u):
        raise FileNotFoundError("no universe yet - run:  python -m resarb.data universe")
    s0 = pd.Timestamp(start or strat.v["start"]) - pd.Timedelta(days=int(strat.warmup_days() * 1.5) + 10)
    s1 = pd.Timestamp(end) if end else None
    months = u[(u.month >= s0.to_period("M").to_timestamp()) & ((u.month <= s1) if s1 is not None else True) & (u["rank"] <= n)]
    syms = sorted(set(months.symbol))
    key = (tuple(syms), str(s0.date()), str(s1) if s1 is not None else "", feed)
    p = _PANELS.get(key) if cache and extra is None else None
    if p is None:
        log(f"loading {len(syms)} stocks + {len(D.ETFS)} ETFs ({feed}) from {s0:%Y-%m-%d} ...")
        p = D.load_panel(syms, s0, s1, feed=feed, extra=extra)
        if cache and extra is None:
            _PANELS.clear()
            _PANELS[key] = p
    member_d = D.membership_mask(p, n, u)
    fc = strat.factor_cfg()
    rkey = (id(p), fc.key(), n)
    res = _RES.get(rkey) if cache and extra is None else None
    if res is None:
        log(f"factor model '{fc.model}': betas over {fc.beta_days} days, residuals for {len(p.ts):,} bars ...")
        res = F.build(p, fc, member_d)
        if cache and extra is None:
            _RES.clear()
            _RES[rkey] = res
    member = member_d[p.day] & np.isfinite(res.e_cc) & np.isfinite(p.open)
    return Context(p=p, res=res, member=member, member_d=member_d, lag=strat.lag)


def backtest(strat: Strategy, ctx: Optional[Context] = None, log: Callable[[str], None] = print, stop: Callable[[], bool] = lambda: False) -> tuple:
    ctx = ctx or context(strat, log=log)
    log("scoring ...")
    sc = strat.score(ctx, log)
    log("simulating ...")
    bt = BT.run(ctx.p, ctx.res, sc, ctx.member, strat.decision_bars(ctx.p), strat.port_cfg(), strat.bt_cfg(), log=log, stop=stop)
    return bt, sc, ctx
