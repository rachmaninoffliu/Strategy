"""
Beta extraction and residuals (no Qt).

Two factor models, both from Avellaneda & Lee (2010), estimated every morning from DAILY returns that end at the previous close (nothing of the current day is used):

  * 'etf'  r_i = a + b_m * r_SPY + b_s * (r_SECTOR - r_SPY) + e_i       (the stock's sector SPDR; Communication stocks use SPY only before XLC existed, June 2018)
           tradable hedge: per $1 of stock, short (b_m - b_s) $ of SPY and b_s $ of the sector ETF.
  * 'pca'  the top k eigenportfolios of the trailing correlation matrix of the members' daily returns (Q = v / sigma), betas from a regression on their returns;
           the hedge is made of stocks (portfolio construction projects the weights onto the factor null space instead of trading ETFs).

The day's betas are applied to every 15-minute return of that day, which gives the residual ("alpha") returns all three strategies trade:
  e_cc  close-to-close residual of each bar (the first bar of a day includes the overnight move)
  e_oo  open-to-open residual (what a position entered at a bar's open earns until the next bar's open; the last bar of a day runs to the next day's open)
  e_on  overnight residual (first open vs previous close), e_d daily close-to-close residual
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import data as D


@dataclass
class FactorCfg:
    model: str = "etf"                  # 'etf' | 'pca'
    beta_days: int = 126                # regression window (daily returns)
    min_obs: int = 60
    pca_k: int = 15                     # eigenportfolios (PCA model)
    pca_corr_days: int = 252            # correlation window (PCA model)
    winsor: float = 0.20                # daily returns are clipped at +-20% for the estimation

    def key(self) -> str:
        return f"{self.model}-{self.beta_days}-{self.min_obs}-{self.pca_k if self.model == 'pca' else 0}-{self.pca_corr_days if self.model == 'pca' else 0}-{self.winsor}"


@dataclass
class Residuals:
    cfg: FactorCfg
    syms: list
    etf_idx: np.ndarray                 # column of each hedge ETF in the panel (SPY first, then the sector ETFs in D.SECTOR_ETF order)
    sector_col: np.ndarray              # per symbol: panel column of its sector ETF (-1: none)
    b_m: np.ndarray                     # D x N market beta (etf model)
    b_s: np.ndarray                     # D x N sector-excess beta (etf model)
    idio_var: np.ndarray                # D x N residual variance of the daily regression
    r2: np.ndarray                      # D x N
    e_cc: np.ndarray                    # T x N
    e_oo: np.ndarray                    # T x N
    e_on: np.ndarray                    # D x N
    e_d: np.ndarray                     # D x N
    r_cc: np.ndarray                    # T x N raw close-to-close bar returns
    r_oo: np.ndarray                    # T x N raw open-to-open returns
    pca_Q: Optional[np.ndarray] = None  # D x K x N eigenportfolio weights (pca model)
    pca_B: Optional[np.ndarray] = None  # D x N x K loadings (pca model)
    extra: dict = field(default_factory=dict)

    def hedge_loadings(self, d: int) -> np.ndarray:
        """N x E matrix: $ of each hedge ETF that carries the factor exposure of $1 of each stock on day d (etf model). Hedge = -(w @ L). Cached per day."""
        cache = self.extra.setdefault("_hl", {})
        if d in cache:
            return cache[d]
        N, E = len(self.syms), len(self.etf_idx)
        if "_hl_k" not in self.extra:
            pos = {int(c): k for k, c in enumerate(self.etf_idx)}
            self.extra["_hl_k"] = np.array([pos.get(int(c), 0) for c in self.sector_col])     # column of the stock's sector ETF (0 = SPY: no sector ETF)
        kk = self.extra["_hl_k"]
        bm, bs = np.nan_to_num(self.b_m[d]), np.nan_to_num(self.b_s[d])
        L = np.zeros((N, E))
        L[:, 0] = bm - bs
        np.add.at(L, (np.arange(N), kk), bs)
        if len(cache) > 4096:
            cache.clear()
        cache[d] = L
        return L


def bar_returns(p: D.Panel) -> tuple[np.ndarray, np.ndarray]:
    """Close-to-close and open-to-open returns of every bar (T x N). The open-to-open return of the last bar of a day runs to the next day's first open."""
    c, o = p.close, p.open
    with np.errstate(invalid="ignore", divide="ignore"):
        r_cc = np.full_like(c, np.nan)
        r_cc[1:] = c[1:] / c[:-1] - 1.0
        r_oo = np.full_like(o, np.nan)
        r_oo[:-1] = o[1:] / o[:-1] - 1.0
    return r_cc, r_oo


def daily_returns(p: D.Panel) -> tuple[np.ndarray, np.ndarray]:
    cd, od = p.daily_close(), p.daily_open()
    with np.errstate(invalid="ignore", divide="ignore"):
        rd = np.full_like(cd, np.nan)
        rd[1:] = cd[1:] / cd[:-1] - 1.0
        ron = np.full_like(od, np.nan)
        ron[1:] = od[1:] / cd[:-1] - 1.0
    return rd, ron


def _roll(x: np.ndarray, w: int) -> np.ndarray:
    """Sum over the previous w rows, EXCLUDING the current row (row d sees rows d-w .. d-1)."""
    cs = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x]), axis=0)                  # cs[k] = sum of rows 0 .. k-1
    d = np.arange(x.shape[0])
    return cs[d] - cs[np.maximum(d - w, 0)]


def etf_betas(rd: np.ndarray, rm: np.ndarray, rsec: np.ndarray, cfg: FactorCfg) -> tuple:
    """Rolling OLS of each stock's daily return on [1, r_m, r_sec - r_m] over the previous ``beta_days`` days. rd, rsec: D x N; rm: D. Returns b_m, b_s, idio_var, r2 (D x N)."""
    y = np.clip(rd, -cfg.winsor, cfg.winsor)
    s = rsec - rm[:, None]
    ok = np.isfinite(y) & np.isfinite(rm)[:, None]
    s = np.where(np.isfinite(s), s, 0.0)
    x0 = ok.astype(float)
    y0 = np.where(ok, y, 0.0)
    m0 = np.where(ok, rm[:, None], 0.0)
    s0 = np.where(ok, s, 0.0)
    W = cfg.beta_days
    n = _roll(x0, W)
    Sm, Ss, Sy = _roll(m0, W), _roll(s0, W), _roll(y0, W)
    Smm, Sss, Sms = _roll(m0 * m0, W), _roll(s0 * s0, W), _roll(m0 * s0, W)
    Smy, Ssy, Syy = _roll(m0 * y0, W), _roll(s0 * y0, W), _roll(y0 * y0, W)
    A = np.stack([np.stack([n, Sm, Ss], -1), np.stack([Sm, Smm, Sms], -1), np.stack([Ss, Sms, Sss], -1)], -2)       # D x N x 3 x 3
    A[..., 2, 2] += 1e-8 + 1e-3 * np.maximum(Smm, 1e-12)                                                        # a tiny ridge: a sector factor that is all zeros gives b_s = 0
    rhs = np.stack([Sy, Smy, Ssy], -1)
    good = n >= cfg.min_obs
    coef = np.full(rhs.shape, np.nan)
    if good.any():
        coef[good] = np.linalg.solve(A[good], rhs[good][..., None])[..., 0]
    a, bm, bs = coef[..., 0], coef[..., 1], coef[..., 2]
    sse = Syy - (coef * rhs).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ivar = np.maximum(sse, 0) / np.maximum(n - 3, 1)
        tss = Syy - Sy * Sy / np.maximum(n, 1)
        r2 = 1.0 - np.maximum(sse, 0) / np.where(tss > 0, tss, np.nan)
    ivar[~good] = np.nan
    r2[~good] = np.nan
    return bm, bs, ivar, r2


def pca_model(rd: np.ndarray, member_d: np.ndarray, cfg: FactorCfg) -> tuple:
    """Avellaneda-Lee PCA factors, re-estimated every day from the previous ``pca_corr_days`` days of the day's members. Returns Q (D x K x N eigenportfolio weights, zero for non-members),
    B (D x N x K betas from a ``beta_days`` regression on the factor returns), idio_var, r2."""
    Dn, N = rd.shape
    K = cfg.pca_k
    Q = np.zeros((Dn, K, N), np.float32)
    B = np.full((Dn, N, K), np.nan, np.float32)
    ivar = np.full((Dn, N), np.nan)
    r2 = np.full((Dn, N), np.nan)
    y_all = np.clip(rd, -cfg.winsor, cfg.winsor)
    for d in range(Dn):
        mem = np.where(member_d[d])[0]
        lo = d - cfg.pca_corr_days
        if lo < 0 or len(mem) < K + 5:
            continue
        Y = y_all[lo:d][:, mem]
        okc = np.isfinite(Y).mean(0) >= 0.9
        mem = mem[okc]
        Y = Y[:, okc]
        Y = np.where(np.isfinite(Y), Y, 0.0)
        mu, sd = Y.mean(0), Y.std(0)
        sd = np.where(sd > 0, sd, np.nan)
        Z = (Y - mu) / sd
        Z = np.nan_to_num(Z)
        C = (Z.T @ Z) / max(len(Z) - 1, 1)
        w, v = np.linalg.eigh(C)
        v = v[:, ::-1][:, :K]
        q = (v / np.nan_to_num(sd, nan=np.inf)[:, None]).T                             # K x n: $ in each member per unit of factor
        Q[d][:, mem] = q.astype(np.float32)
        Wb = cfg.beta_days
        F = y_all[max(d - Wb, 0):d][:, mem]
        F = np.where(np.isfinite(F), F, 0.0) @ q.T                                      # factor returns over the beta window
        Yb = y_all[max(d - Wb, 0):d]
        X = np.c_[np.ones(len(F)), F]
        XtX_inv = np.linalg.pinv(X.T @ X)
        for i in np.where(np.isfinite(Yb).sum(0) >= cfg.min_obs)[0]:
            yi = Yb[:, i]
            ok = np.isfinite(yi)
            if ok.all():
                coef = XtX_inv @ (X.T @ yi)
                res = yi - X @ coef
            else:
                Xi = X[ok]
                coef = np.linalg.lstsq(Xi, yi[ok], rcond=None)[0]
                res = yi[ok] - Xi @ coef
                yi = yi[ok]
            B[d, i] = coef[1:]
            ivar[d, i] = res.var(ddof=K + 1) if len(res) > K + 1 else np.nan
            v_ = yi.var()
            r2[d, i] = 1 - res.var() / v_ if v_ > 0 else np.nan
    return Q, B, ivar, r2


def build(p: D.Panel, cfg: FactorCfg, member_d: Optional[np.ndarray] = None) -> Residuals:
    """Betas for every day of the panel (from the previous days only) and the residual returns of every bar."""
    N = len(p.syms)
    etfs = [D.MARKET] + list(D.SECTOR_ETF.values())
    etf_idx = np.array([p.col[e] for e in etfs])
    sec = D.sectors()
    sector_col = np.array([p.col.get(D.SECTOR_ETF.get(str(sec.get(s)), ""), -1) if s not in D.ETFS else -1 for s in p.syms])
    rd, ron = daily_returns(p)
    r_cc, r_oo = bar_returns(p)
    m = p.col[D.MARKET]
    day = p.day
    if cfg.model == "etf":
        rsec = np.where(sector_col[None, :] >= 0, rd[:, np.maximum(sector_col, 0)], np.nan)
        bm, bs, ivar, r2 = etf_betas(rd, rd[:, m], rsec, cfg)
        bm_t, bs_t = bm[day], bs[day]
        sec_cc = np.where(sector_col[None, :] >= 0, r_cc[:, np.maximum(sector_col, 0)], 0.0)
        sec_oo = np.where(sector_col[None, :] >= 0, r_oo[:, np.maximum(sector_col, 0)], 0.0)
        sec_cc = np.where(np.isfinite(sec_cc), sec_cc, r_cc[:, [m]])                     # sector ETF missing (XLC before 2018-06): the sector term vanishes
        sec_oo = np.where(np.isfinite(sec_oo), sec_oo, r_oo[:, [m]])
        e_cc = r_cc - bm_t * r_cc[:, [m]] - bs_t * (sec_cc - r_cc[:, [m]])
        e_oo = r_oo - bm_t * r_oo[:, [m]] - bs_t * (sec_oo - r_oo[:, [m]])
        sd = np.where(sector_col[None, :] >= 0, rd[:, np.maximum(sector_col, 0)], np.nan)
        sd = np.where(np.isfinite(sd), sd, rd[:, [m]])
        so = np.where(sector_col[None, :] >= 0, ron[:, np.maximum(sector_col, 0)], np.nan)
        so = np.where(np.isfinite(so), so, ron[:, [m]])
        e_d = rd - bm * rd[:, [m]] - bs * (sd - rd[:, [m]])
        e_on = ron - bm * ron[:, [m]] - bs * (so - ron[:, [m]])
        Q = Bp = None
    elif cfg.model == "pca":
        if member_d is None:
            raise ValueError("the PCA model needs the membership mask")
        Q, Bp, ivar, r2 = pca_model(rd, member_d, cfg)
        f_cc = np.einsum("tn,tkn->tk", np.nan_to_num(r_cc), Q[day].astype(float))
        f_oo = np.einsum("tn,tkn->tk", np.nan_to_num(r_oo), Q[day].astype(float))
        f_d = np.einsum("dn,dkn->dk", np.nan_to_num(np.clip(rd, -cfg.winsor, cfg.winsor)), Q.astype(float))
        f_on = np.einsum("dn,dkn->dk", np.nan_to_num(np.clip(ron, -cfg.winsor, cfg.winsor)), Q.astype(float))
        Bt = Bp.astype(float)
        e_cc = r_cc - np.einsum("tnk,tk->tn", Bt[day], f_cc)
        e_oo = r_oo - np.einsum("tnk,tk->tn", Bt[day], f_oo)
        e_d = rd - np.einsum("dnk,dk->dn", Bt, f_d)
        e_on = ron - np.einsum("dnk,dk->dn", Bt, f_on)
        bm = np.full_like(rd, np.nan)
        bs = np.full_like(rd, np.nan)
    else:
        raise ValueError(f"unknown factor model {cfg.model!r}")
    is_etf = np.array([s in D.ETFS for s in p.syms])
    for a in (e_cc, e_oo):
        a[:, is_etf] = np.nan
    for a in (e_d, e_on):
        a[:, is_etf] = np.nan
    return Residuals(cfg=cfg, syms=list(p.syms), etf_idx=etf_idx, sector_col=sector_col, b_m=bm, b_s=bs, idio_var=ivar, r2=r2, e_cc=e_cc, e_oo=e_oo, e_on=e_on, e_d=e_d,
                     r_cc=r_cc, r_oo=r_oo, pca_Q=Q, pca_B=Bp)
