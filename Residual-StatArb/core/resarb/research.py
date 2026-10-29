"""
Signal research shared by the GUI's Research tab and the research scripts (no Qt): rank IC of a score (or of every feature) against the forward residual return at several horizons,
in-sample vs out-of-sample with Newey-West t-statistics, and the forward residual return by score decile.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import features as X
from .strategy import Context

HORIZONS = (1, 4, 13, 26, 130)


def _rows(ctx: Context, start: Optional[str], end: Optional[str], every: int = 1) -> np.ndarray:
    ts = ctx.p.ts
    i0 = int(np.searchsorted(ts, pd.Timestamp(start))) if start else 0
    i1 = int(np.searchsorted(ts, pd.Timestamp(end))) if end else len(ts)
    r = np.arange(i0, i1)
    return r[(ctx.p.bar[r] % max(every, 1)) == 0] if every > 1 else r


def _summ(ic: np.ndarray, day: np.ndarray, h: int) -> dict:
    d = X.daily_mean(ic, day - day.min(), int(day.max() - day.min() + 1)) if len(day) else np.array([])
    d = d[np.isfinite(d)]
    lags = max(2, int(math.ceil(h / 26)) + 1)
    return {"ic": float(np.mean(d)) if len(d) else np.nan, "t": X.nw_tstat(d, lags) if len(d) > 20 else np.nan, "days": int(len(d)),
            "ic_ir": float(np.mean(d) / np.std(d) * math.sqrt(252)) if len(d) > 20 and np.std(d) > 0 else np.nan}


def score_ic(ctx: Context, score: np.ndarray, start: str, split: str, end: Optional[str] = None, horizons=HORIZONS, every: int = 1,
             log: Callable[[str], None] = lambda m: None) -> pd.DataFrame:
    rows = _rows(ctx, start, end, every)
    sp = int(np.searchsorted(ctx.p.ts, pd.Timestamp(split)))
    mem = ctx.member[rows]
    fr = X.xs_rank(score[rows], mem)
    out = []
    for h in horizons:
        log(f"  IC at {h} bars ...")
        tg = X.forward(ctx.res, h, ctx.lag)[rows]
        ic = X.row_corr(fr, X.xs_rank(tg, mem))
        day = ctx.p.day[rows]
        ins, oos = rows < sp, rows >= sp
        a, b = _summ(ic[ins], day[ins], h), _summ(ic[oos], day[oos], h)
        out.append({"horizon_bars": h, "IS_ic": a["ic"], "IS_t": a["t"], "IS_icir": a["ic_ir"], "OOS_ic": b["ic"], "OOS_t": b["t"], "OOS_icir": b["ic_ir"], "IS_days": a["days"], "OOS_days": b["days"]})
    return pd.DataFrame(out)


def feature_ic(ctx: Context, names: list, start: str, split: str, end: Optional[str] = None, horizons=HORIZONS, every: int = 1,
               log: Callable[[str], None] = lambda m: None) -> pd.DataFrame:
    """Rank IC of every feature (sign as computed: a negative IC means 'reversal') at each horizon, IS and OOS."""
    rows = _rows(ctx, start, end, every)
    sp = int(np.searchsorted(ctx.p.ts, pd.Timestamp(split)))
    mem = ctx.member[rows]
    day = ctx.p.day[rows]
    ins, oos = rows < sp, rows >= sp
    tg = {}
    for h in horizons:
        log(f"  target {h} bars ...")
        tg[h] = X.xs_rank(X.forward(ctx.res, h, ctx.lag)[rows], mem)
    out = []
    feats = ctx.features([n for n in names if n not in X.MARKET_WIDE])
    for n, a in feats.items():
        log(f"  feature {n} ...")
        fr = X.xs_rank(a[rows].astype(float), mem)
        for h in horizons:
            ic = X.row_corr(fr, tg[h])
            s_is, s_oos = _summ(ic[ins], day[ins], h), _summ(ic[oos], day[oos], h)
            out.append({"feature": n, "horizon_bars": h, "IS_ic": s_is["ic"], "IS_t": s_is["t"], "OOS_ic": s_oos["ic"], "OOS_t": s_oos["t"]})
    return pd.DataFrame(out)


def deciles(ctx: Context, score: np.ndarray, h: int, start: str, split: str, end: Optional[str] = None, n: int = 10, every: int = 1) -> pd.DataFrame:
    """Mean forward residual return (bp) of each score decile, minus the cross-sectional mean, IS and OOS."""
    rows = _rows(ctx, start, end, every)
    sp = int(np.searchsorted(ctx.p.ts, pd.Timestamp(split)))
    mem = ctx.member[rows]
    pct = X.xs_rank(score[rows], mem) + 0.5
    tg = X.xs_demean(X.forward(ctx.res, h, ctx.lag)[rows], mem)
    q = np.floor(np.clip(pct, 0, 0.999999) * n)
    out = []
    for name, m in (("IS", rows < sp), ("OOS", rows >= sp)):
        qq, tt = q[m], tg[m]
        ok = np.isfinite(qq) & np.isfinite(tt)
        for k in range(n):
            s = ok & (qq == k)
            out.append({"sample": name, "decile": k + 1, "fwd_bp": float(np.mean(tt[s]) * 1e4) if s.any() else np.nan, "cells": int(s.sum())})
    return pd.DataFrame(out)
