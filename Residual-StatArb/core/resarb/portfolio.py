"""
Score -> target weights (no Qt). The SAME function decides in the backtest and in the live engine: it gets the scores of the decision bar, the members, and what is held right now
(simulated in the backtest, the broker's positions in live), and returns the stock weights to hold (fractions of the capital). The factor hedge is added on top.

Methods
  quantile   long the top q_in, short the bottom q_in of the members by score; a name already held stays until it leaves the top / bottom q_out (hysteresis cuts turnover); equal weight
  rank       weights proportional to the de-meaned cross-sectional rank (every member gets a position)
  threshold  Avellaneda-Lee: open long when score >= s_open, close it when score <= s_close; mirror for shorts; each open name gets gross / (2 * names)
Hedge
  etf        per $1 of stock, short (b_m - b_s) $ of SPY and b_s $ of the stock's sector ETF (factor model 'etf')
  project    remove the factor exposures (and the net dollar) from the stock weights by projecting them onto the null space of the loadings (any factor model; no ETFs traded)
  none
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factors import Residuals


@dataclass
class PortCfg:
    method: str = "quantile"
    q_in: float = 0.2
    q_out: float = 0.3
    s_open: float = 1.25
    s_close: float = 0.5
    names: int = 40                     # threshold method: gross is split over this many positions
    gross: float = 1.0                  # stock book gross as a fraction of capital (1.0 = 50% long + 50% short when balanced)
    max_w: float = 0.05
    hedge: str = "etf"
    sector_neutral: bool = False
    trade_rate: float = 1.0             # fraction of the way to the target traded at each rebalance (Garleanu-Pedersen); 1 = all the way


def _pct(score: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Percentile in (0, 1) among the valid names (NaN elsewhere)."""
    out = np.full(score.shape, np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = np.argsort(np.argsort(score[idx], kind="stable"), kind="stable")
    out[idx] = (order + 0.5) / len(idx)
    return out


def sector_demean(score: np.ndarray, ok: np.ndarray, sector_col: np.ndarray) -> np.ndarray:
    s = score.copy()
    for g in np.unique(sector_col[ok]):
        m = ok & (sector_col == g)
        if m.sum() >= 3:
            s[m] = s[m] - np.mean(s[m])
    return s


def decide(score: np.ndarray, member: np.ndarray, held: np.ndarray, cfg: PortCfg, sector_col: np.ndarray, tradable: np.ndarray = None) -> np.ndarray:
    """Target stock weights (N). ``held`` = current stock weights (same scale). ``tradable`` = names that may be traded now (have a price); others keep their held weight."""
    N = len(score)
    ok = member & np.isfinite(score)
    if tradable is not None:
        ok &= tradable
    sc = sector_demean(score, ok, sector_col) if cfg.sector_neutral else score
    w = np.zeros(N)
    if cfg.method == "quantile":
        pct = _pct(sc, ok)
        hl, hs = held > 0, held < 0
        lo = ok & ((pct >= 1 - cfg.q_in) | (hl & (pct >= 1 - cfg.q_out)))
        sh = ok & ((pct <= cfg.q_in) | (hs & (pct <= cfg.q_out)))
        lo &= ~sh
        nl, ns = lo.sum(), sh.sum()
        if nl and ns:
            w[lo] = 0.5 * cfg.gross / nl
            w[sh] = -0.5 * cfg.gross / ns
    elif cfg.method == "rank":
        pct = _pct(sc, ok)
        x = np.where(ok, pct - 0.5, 0.0)
        g = np.abs(x).sum()
        if g > 0:
            w = cfg.gross * x / g
    elif cfg.method == "threshold":
        per = cfg.gross / max(2 * cfg.names, 1)
        hl, hs = held > 0, held < 0
        keep_l = hl & ok & (sc > cfg.s_close)
        keep_s = hs & ok & (sc < -cfg.s_close)
        new_l = ok & ~hl & ~hs & (sc >= cfg.s_open)
        new_s = ok & ~hl & ~hs & (sc <= -cfg.s_open)
        L = keep_l | new_l
        S = keep_s | new_s
        cap = cfg.names
        nL, nS = L.sum(), S.sum()
        if nL > cap:                                                                        # too many: keep the held ones first, then the strongest new ones
            extra = np.where(new_l)[0]
            drop = extra[np.argsort(sc[extra])][: nL - cap]
            L[drop] = False
        if nS > cap:
            extra = np.where(new_s)[0]
            drop = extra[np.argsort(-sc[extra])][: nS - cap]
            S[drop] = False
        w[L] = per
        w[S] = -per
    else:
        raise ValueError(f"unknown method {cfg.method!r}")
    if cfg.max_w > 0:
        w = np.clip(w, -cfg.max_w, cfg.max_w)
    w = np.where(held * w < 0, 0.0, w)                                                     # never flip in one step: close now, re-open at the next decision (one order per name)
    if tradable is not None:
        w = np.where(tradable, w, held)                                                    # no price now: the position can't be changed
    return w


def approach(held: np.ndarray, target: np.ndarray, rate: float, gross: float = 1.0, tradable: np.ndarray = None) -> np.ndarray:
    """Garleanu & Pedersen (2013): trade only a fraction ``rate`` of the way from the held weights to the target each rebalance (rate 1 = jump to the target). A position whose
    target is zero and that has shrunk below 5% of an average position is closed outright (no dust)."""
    if rate >= 1.0:
        return target
    w = held + rate * (target - held)
    n = max(int((np.abs(target) > 0).sum()), 1)
    dust = (target == 0) & (np.abs(w) < 0.05 * gross / n)
    w = np.where(dust, 0.0, w)
    if tradable is not None:
        w = np.where(tradable, w, held)
    return w


def hedge_etf(w: np.ndarray, res: Residuals, d: int) -> np.ndarray:
    """ETF weights (E, in the order of res.etf_idx) that remove the market and sector exposure of the stock weights w on day d."""
    return -(w @ res.hedge_loadings(d))


def loadings(res: Residuals, d: int) -> np.ndarray:
    """N x K factor exposures for the projection hedge (+ a column of ones for dollar neutrality)."""
    if res.cfg.model == "pca" and res.pca_B is not None:
        B = np.nan_to_num(res.pca_B[d].astype(float))
    else:
        bm, bs = np.nan_to_num(res.b_m[d]), np.nan_to_num(res.b_s[d])
        secs = sorted(set(int(c) for c in res.sector_col if c >= 0))
        B = np.zeros((len(bm), 1 + len(secs)))
        B[:, 0] = bm
        for k, c in enumerate(secs):
            m = res.sector_col == c
            B[m, 1 + k] = bs[m]
    return np.c_[np.ones(len(B)), B]


def project(w: np.ndarray, res: Residuals, d: int) -> np.ndarray:
    """Make the selected names' weights factor- and dollar-neutral (least change), keeping the gross."""
    sel = np.where(w != 0)[0]
    if len(sel) < 3:
        return w
    B = loadings(res, d)[sel]
    B = B[:, np.abs(B).sum(0) > 1e-12]
    if B.shape[1] >= len(sel):
        return w
    x = w[sel]
    x2 = x - B @ np.linalg.lstsq(B, x, rcond=None)[0]
    g0, g1 = np.abs(x).sum(), np.abs(x2).sum()
    out = np.zeros_like(w)
    out[sel] = x2 * (g0 / g1 if g1 > 0 else 0.0)
    return out


def exposures(w: np.ndarray, h: np.ndarray, res: Residuals, d: int) -> dict:
    """Net exposures of a book (stock weights w + ETF hedge weights h) on day d: dollars, market beta, gross."""
    bm = np.nan_to_num(res.b_m[d]) if np.isfinite(res.b_m[d]).any() else np.zeros(len(w))
    beta = float(w @ bm + (h[0] if len(h) else 0.0) + (h[1:].sum() if len(h) > 1 else 0.0))
    return {"gross_stocks": float(np.abs(w).sum()), "gross_etf": float(np.abs(h).sum()), "net_stocks": float(w.sum()), "net_total": float(w.sum() + h.sum()),
            "beta_after_hedge": beta, "longs": int((w > 0).sum()), "shorts": int((w < 0).sum())}
