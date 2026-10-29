"""
Signal stage: the Avellaneda-Lee residual / Ornstein-Uhlenbeck s-score applied to one 3x ETF against its 1x ETF, on any intraday time frame (stage A, cached).

A daily-reset 3x ETF is, within a session, three times its index since the previous close:  L_t/L_close - 1  ~  3 (B_t/B_close - 1).  The residual of L on beta x B is therefore the 'systematic factor'
model of the paper with ONE factor (the 1x ETF) and a known beta, and its cumulative sum X is the process the paper fits as an OU:  X(n+1) = a + b X(n) + zeta, kappa = -ln(b) per bar,
m = a/(1-b), sigma_eq = sqrt(var(zeta) / (1 - b^2)), s = (X - m) / sigma_eq.

Two definitions of X (``SigCfg.spread``):
  chain  X = running sum of the within-session bar residuals (the first bar of a session is open->close, the overnight gap is not in it) - the paper's construction;
  reset  X_t = (L_t/L_prev_close - 1) - beta (B_t/B_prev_close - 1): the exact premium / discount of the leveraged ETF to 'beta x the index since the reset'.
Everything is vectorised over sessions; the OU fit uses rolling sums, no loops over bars.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Optional

import numpy as np
import pandas as pd

from levpairs_data import NOMINAL_BETA, PairData

TD = 252


@dataclass(frozen=True)
class SigCfg:
    tf: int = 5                              # minutes per bar: 1 | 5 | 15 | 30
    spread: str = "chain"                    # chain | reset
    beta_mode: str = "rolling"               # rolling | fixed
    beta_days: int = 20                      # sessions of history for the rolling beta
    beta_shrink: float = 0.5                 # weight on the nominal 3.0 in the rolling beta
    beta_fixed: float = NOMINAL_BETA
    win_days: float = 5.0                    # OU estimation window, sessions
    s_mode: str = "ou"                       # ou (AR(1) fit as in the paper) | z (plain rolling z-score of X)
    min_obs_frac: float = 0.5                # fraction of the window that must be filled before a score is produced


@dataclass
class PairSignals:
    pair: tuple
    cfg: SigCfg
    dates: pd.DatetimeIndex
    nb: int
    s: np.ndarray            # (days, bars) s-score at the close of each bar (nan if undefined)
    hl_min: np.ndarray       # reversion half-life in minutes (nan if the fit does not revert)
    sig_eq_bp: np.ndarray    # equilibrium standard deviation of the spread, bp of the leveraged leg
    eps: np.ndarray          # (days, bars) bar residual return  rL - beta rB, close to close
    eps_gap: np.ndarray      # previous bar's close -> this bar's open (what the position held BEFORE a fill at this open earns)
    eps_body: np.ndarray     # this bar's open -> close (what a position filled at this bar's open earns)
    beta: np.ndarray         # (days,) hedge ratio (B notional per unit of L notional)
    X: np.ndarray            # the spread itself
    rB_day: np.ndarray       # (days, bars) B return since the previous close (for the move filter)
    L_close: np.ndarray
    B_close: np.ndarray
    L_open: Optional[np.ndarray] = None
    B_open: Optional[np.ndarray] = None


def sig_fields() -> set:
    return {f.name for f in fields(SigCfg)}


def bar_returns(close: np.ndarray, open_: np.ndarray, miss: Optional[np.ndarray] = None) -> np.ndarray:
    """Simple bar-to-bar returns inside a session; the first bar is open -> close (no overnight gap). A bar with no trade has return 0 by construction (price carried forward)."""
    r = np.empty_like(close)
    r[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    r[:, 0] = close[:, 0] / open_[:, 0] - 1.0
    if miss is not None:
        r[:, 0] = np.where(miss[:, 0], 0.0, r[:, 0])
    return r


def rolling_beta(rL: np.ndarray, rB: np.ndarray, cfg: SigCfg) -> np.ndarray:
    """Per-session hedge ratio from the PREVIOUS ``beta_days`` sessions only (regression through the origin of bar returns), shrunk toward the nominal 3.0."""
    n = rL.shape[0]
    if cfg.beta_mode == "fixed":
        return np.full(n, cfg.beta_fixed)
    sxy, sxx = (rL[:, 1:] * rB[:, 1:]).sum(axis=1), (rB[:, 1:] ** 2).sum(axis=1)
    cxy, cxx = np.concatenate([[0.0], np.cumsum(sxy)]), np.concatenate([[0.0], np.cumsum(sxx)])
    K = cfg.beta_days
    beta = np.full(n, cfg.beta_fixed)
    for d in range(5, n):
        lo = max(0, d - K)
        den = cxx[d] - cxx[lo]
        if den > 0:
            ols = (cxy[d] - cxy[lo]) / den
            beta[d] = (1 - cfg.beta_shrink) * ols + cfg.beta_shrink * cfg.beta_fixed
    return np.clip(beta, 1.5, 5.0)


def ou_scores(X: np.ndarray, nb: int, win_bars: int, min_obs: int, mode: str = "ou") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling s-score of the flattened spread X (days*bars). Returns (s, half-life in bars, sigma_eq) arrays of the same length. Transitions that cross a session boundary are excluded from the AR(1) fit."""
    N = len(X)
    Xs = pd.Series(X)
    if mode == "z":
        mu = Xs.rolling(win_bars, min_periods=min_obs).mean()
        sd = Xs.rolling(win_bars, min_periods=min_obs).std()
        s = ((Xs - mu) / sd.replace(0, np.nan)).to_numpy()
        return s, np.full(N, np.nan), sd.to_numpy()
    x = np.concatenate([[np.nan], X[:-1]])                       # X(n-1)
    inside = (np.arange(N) % nb) != 0                            # the transition into a session's first bar is overnight
    m = (inside & np.isfinite(x) & np.isfinite(X)).astype(float)
    xv, yv = np.where(m > 0, x, 0.0), np.where(m > 0, X, 0.0)
    R = lambda a: pd.Series(a).rolling(win_bars, min_periods=min_obs).sum().to_numpy()
    n, sx, sy, sxx, sxy, syy = R(m), R(xv), R(yv), R(xv * xv), R(xv * yv), R(yv * yv)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = sx / n, sy / n
        vx, vy, cv = sxx / n - mx * mx, syy / n - my * my, sxy / n - mx * my
        b = cv / vx
        a = my - b * mx
        vz = vy - cv * cv / vx                                   # variance of the AR(1) innovations
        ok = (b > 0) & (b < 1) & (vz > 0) & np.isfinite(b)
        mm = a / (1.0 - b)
        sig = np.sqrt(vz / (1.0 - b * b))
        s = np.where(ok, (X - mm) / sig, np.nan)
        hl = np.where(ok, math.log(2) / -np.log(np.where(ok, b, 0.5)), np.nan)
    return s, hl, np.where(ok, sig, np.nan)


def build_signals(d: PairData, cfg: SigCfg) -> PairSignals:
    n, nb = d.L_close.shape
    rL, rB = bar_returns(d.L_close, d.L_open, d.L_miss), bar_returns(d.B_close, d.B_open, d.B_miss)
    beta = rolling_beta(rL, rB, cfg)
    eps = rL - beta[:, None] * rB
    gap = lambda o, c: np.concatenate([np.zeros((o.shape[0], 1)), o[:, 1:] / c[:, :-1] - 1.0], axis=1)
    eps_gap = gap(d.L_open, d.L_close) - beta[:, None] * gap(d.B_open, d.B_close)
    eps_body = (d.L_close / d.L_open - 1.0) - beta[:, None] * (d.B_close / d.B_open - 1.0)
    prevL = np.concatenate([[d.L_open[0, 0]], d.L_close[:-1, -1]])
    prevB = np.concatenate([[d.B_open[0, 0]], d.B_close[:-1, -1]])
    rB_day = d.B_close / prevB[:, None] - 1.0
    if cfg.spread == "reset":
        X = (d.L_close / prevL[:, None] - 1.0) - beta[:, None] * rB_day
    else:
        X = np.cumsum(eps.ravel()).reshape(n, nb)
    win = max(int(round(cfg.win_days * nb)), 10)
    s, hl, sig = ou_scores(X.ravel(), nb, win, max(int(cfg.min_obs_frac * win), 5), cfg.s_mode)
    return PairSignals(pair=d.pair, cfg=cfg, dates=d.dates, nb=nb, s=s.reshape(n, nb), hl_min=(hl * d.tf).reshape(n, nb), sig_eq_bp=(sig * 1e4).reshape(n, nb), eps=eps, eps_gap=eps_gap, eps_body=eps_body, beta=beta, X=X, rB_day=rB_day,
                       L_close=d.L_close, B_close=d.B_close, L_open=d.L_open, B_open=d.B_open)
