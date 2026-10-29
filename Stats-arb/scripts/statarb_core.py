"""
Statistical arbitrage core (no Qt, no network): Avellaneda & Lee (2008) residual mean-reversion, extended for today's market.

PAPER (the baseline, all of it reproduced here):
  * returns R_i are split into a systematic part (15 PCA eigenportfolios of the 252-day correlation matrix, or a sector ETF) and an idiosyncratic residual
  * on a 60-day window the cumulative residual X_k is fitted as an Ornstein-Uhlenbeck process (AR(1) regression  X_{n+1} = a + b X_n + zeta)
        kappa = -ln(b)*252   m = a/(1-b)   sigma_eq = sqrt(var(zeta)/(1-b^2))      only 'fast' reverters are traded: kappa > 252/30 = 8.4
  * the s-score  s = (X - m)/sigma_eq  (m centred across stocks) drives a bang-bang rule: open at |s| > 1.25, close longs at s > -0.50, close shorts at s < +0.75
  * 'trading time': returns are scaled by <V>/V so a big move on heavy volume is believed more (and faded less)
  * the book is held market neutral; costs 5 bp a side in the paper

WHAT IS ADDED FOR 2026 (each is a switch, so the report can show the marginal value of every one - see PortCfg / FactorCfg):
  * idiosyncratic-jump control   earnings reaction sessions come from the Yahoo earnings calendar (published weeks ahead, so usable point-in-time): no entries into the window
                                 before an announcement, positions are closed before it, and news / post-earnings days are not faded. Idiosyncratic variance is ~73% of
                                 single-stock variance now (Goldman, 2026) - this is where the left tail comes from.
  * crowding / style neutrality  the book is projected to zero exposure to market beta, the top PCA factors, the 11 sectors, 12-1 momentum, volatility and size
                                 (the quant unwinds of Jan / Mar / Jun-Jul 2026 were factor and crowding events, not single-name events)
  * geopolitical / macro         the book is neutral to each stock's estimated OIL (Hormuz), RATES (Fed hiking) and USD betas, and a shock score from SPY / VIXY / USO / TLT / GLD
                                 de-grosses the book for a few sessions after a macro shock
  * risk targeting               EWMA volatility target on the strategy's own P&L, capped at the Reg-T 2x overnight gross limit

EXECUTION AND LOOK-AHEAD:  signals use data through the CLOSE of day t, orders fill at the OPEN of day t+1, P&L is open-to-open. Nothing the model reads is later than the
close it is decided on (every rolling statistic that is compared with day t's value is shifted by one day; the earnings dates are scheduled in advance).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.linalg import eigh

TD = 252


# ══ configuration ═══════════════════════════════════════════════════════════════════
@dataclass
class FactorCfg:
    """Everything that changes the SIGNAL (stage A). Stage A is the expensive part, so its output is cached per config."""
    pca_window: int = 252
    est_window: int = 60
    n_factors: int = 15
    var_explained: Optional[float] = None       # e.g. 0.55 -> variable number of eigenportfolios (paper section 5.4)
    factor_mode: str = "pca"                    # 'pca' | 'etf' (own sector SPDR, paper section 5.2)
    volume_time: bool = False                   # paper section 6 trading-time returns
    vt_clip: tuple = (0.33, 3.0)
    sector_demean: bool = False
    kappa_min: float = 8.4
    min_price: float = 5.0
    min_adv: float = 50e6                       # 20-session median dollar volume
    max_abs_ret: float = 0.45                   # larger one-day moves are treated as data / corporate-action errors
    macro_window: int = 126
    start: str = "2017-01-03"
    # news detection (point-in-time: baselines are shifted one day; both are measured relative to that day's market-wide volume / gap so a market-wide spike is not 'news')
    news_vol_ratio: float = 2.5
    news_gap_z: float = 4.0


@dataclass
class PortCfg:
    """Everything that changes how signals become positions and P&L (stage B, cheap)."""
    w_rev: float = 1.0                          # capital share of the residual-reversion sleeve (Avellaneda-Lee)
    w_eap: float = 0.0                          # capital share of the earnings-announcement-premium sleeve (long announcers, sector-ETF hedged)
    eap_lo: int = 2                             # hold announcers whose reaction session is lo..hi sessions ahead of the decision close
    eap_hi: int = 10
    eap_pool: str = 'all'                       # 'all' | 'cheap' (OU s-score < 0) | 'notrich' (s < 1)
    eap_cap: float = 0.04                       # max weight of one name inside the sleeve
    eap_max_move: float = 0.20                  # skip names whose average earnings-day gap (last 8) exceeds this
    eap_min_names: int = 8
    eap_max_names: int = 0                      # 0 = every announcer; N = the N cheapest (lowest OU s-score) - keeps tickets above the minimum on small accounts
    eap_sector_cap: float = 0.30                # max share of the sleeve (at g=1) in one sector; the excess is left in cash
    eap_hedge: bool = True                      # short beta-weighted sector SPDRs against the announcers (market + sector neutral)
    s_open: float = 1.25
    s_close_long: float = 0.50
    s_close_short: float = 0.75
    s_stop: float = 3.5
    max_hold: int = 15
    max_pos_per_side: int = 60
    weight_pow: float = 1.0                     # 0 equal dollar, 1 inverse residual volatility
    min_vr: float = 0.0                         # only open when the signal session's volume ratio (vs 60-session median, market-adjusted) is at least this: event-driven liquidity provision
    vol_boost: float = 0.0                      # entry score = |s| * (1 + vol_boost * max(0, ln(volume ratio))): fade heavy-volume moves harder (the sign the 2008 paper did not find)
    allow_short: bool = True                    # False = long-only fade book, hedged with sector ETFs when hedge='etf'
    hedge: str = "none"                         # 'etf' = short beta-weighted sector SPDRs against the stock book
    tc_etf_bps: float = 0.5
    beta_win: int = 60
    name_cap: float = 0.04                      # max |w| as a share of gross
    neutral: bool = True
    n_pca_neutral: int = 5
    lam_dollar: float = 1e4
    lam_beta: float = 1e4
    lam_sector: float = 1e2
    lam_pca: float = 10.0
    lam_style: float = 1.0
    lam_macro: float = 1.0
    style_neutral: bool = True
    macro_neutral: bool = True
    event_filter: bool = True                   # no fading of news days; cool-off after earnings
    earn_filter: bool = True                    # no entries into / exit before the earnings window
    earn_entry_block: bool = True
    earn_exit: bool = True
    band: float = 0.20                          # no-trade band on continuing positions (share of the position)
    scale_band: float = 0.15                    # gross changes smaller than this are not traded
    tc_bps: float = 3.0                         # one-way, on traded notional
    borrow_bps: float = 50.0                    # annual, on short notional
    vol_target: float = 0.10
    vol_halflife: int = 20
    vol_warmup: int = 30
    g_init: float = 1.0
    g_min: float = 0.25
    g_max: float = 2.0                          # Reg-T overnight limit: long + short <= 2 x equity
    shock: bool = True
    shock_z0: float = 2.0
    shock_k: float = 0.25
    shock_min: float = 0.30
    shock_hold: int = 5
    start: str = "2017-06-01"
    end: Optional[str] = None
    capital: float = 100_000.0


# ══ stage A: signals ════════════════════════════════════════════════════════════════
@dataclass
class Signals:
    cfg: FactorCfg
    dates: pd.DatetimeIndex
    cols: list
    sector_idx: np.ndarray                      # (N,) int
    s: np.ndarray; kappa: np.ndarray; seq: np.ndarray; rvol: np.ndarray
    b_spy: np.ndarray; b_oil: np.ndarray; b_tlt: np.ndarray; b_usd: np.ndarray
    L: np.ndarray                               # (T, N, m) PCA loadings (nan where unused)
    mom_z: np.ndarray; vol_z: np.ndarray; size_z: np.ndarray
    vol_ratio: np.ndarray; gap_z: np.ndarray; news: np.ndarray        # vol_ratio = volume / its 60-session median, divided by that day's market-wide median ratio
    earn_react: np.ndarray; earn_soon: np.ndarray; earn_next: np.ndarray; post_earn: np.ndarray
    univ: np.ndarray
    regime: pd.DataFrame
    adv: np.ndarray = None                      # 20-session median dollar volume (for the cost tiers)
    b_sec: np.ndarray = None                    # beta of the stock to its own sector SPDR over the estimation window (for the ETF hedge)
    emove: np.ndarray = None                    # mean |earnings-day gap| over the stock's last 8 announcements (point-in-time)
    earn_ext: np.ndarray = None                 # reaction matrix extended by 12 future sessions (the published schedule)
    bad: np.ndarray = None                      # close-to-close data errors that were masked
    sector_names: list = field(default_factory=list)


def _rolling_median_shift(a: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(a).rolling(w, min_periods=max(10, w // 2)).median().shift(1).to_numpy()


def _xs_z(a: np.ndarray, mask: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Cross-sectional z-score per row over the masked names."""
    x = np.where(mask, a, np.nan)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    return np.clip((x - mu) / np.where(sd > 0, sd, np.nan), -clip, clip)


def ahead_any(flag: np.ndarray, h_lo: int, h_hi: int) -> np.ndarray:
    """out[d] = any(flag[d+h_lo .. d+h_hi]) - used with the earnings schedule, which is published in advance."""
    out = np.zeros_like(flag, dtype=bool)
    for h in range(h_lo, h_hi + 1):
        if h < flag.shape[0]:
            out[:-h] |= flag[h:]
    return out


def behind_any(flag: np.ndarray, k: int) -> np.ndarray:
    """out[d] = any(flag[d-k .. d])"""
    out = flag.copy()
    for h in range(1, k + 1):
        out[h:] |= flag[:-h]
    return out


def _prepare(panel, cfg: FactorCfg, earn_react: Optional[np.ndarray]) -> dict:
    from statarb_data import SECTOR_ETF
    C, O, V = panel.close.to_numpy(float), panel.open.to_numpy(float), panel.volume.to_numpy(float)
    T, N = C.shape
    cols = list(panel.close.columns)
    sectors = sorted(set(panel.sector.values))
    sector_idx = np.array([sectors.index(panel.sector[c]) for c in cols])
    Vs = np.where(V > 0, V, np.nan)
    vmed = _rolling_median_shift(Vs, 60)
    vmean = pd.DataFrame(Vs).rolling(60, min_periods=30).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = Vs / vmed
        mkt_vr = np.nanmedian(vol_ratio, axis=1, keepdims=True)
        vr_idio = vol_ratio / mkt_vr
    # a one-day move beyond max_abs_ret is a data / corporate-action artefact UNLESS the volume confirms it (SIVB -60% on 3x volume is real and must hurt)
    R = np.full_like(C, np.nan); R[1:] = C[1:] / C[:-1] - 1.0
    bad = np.isfinite(R) & (np.abs(R) > cfg.max_abs_ret) & ~(vr_idio >= 3.0)
    R[bad] = np.nan
    gap = np.full_like(C, np.nan); gap[1:] = O[1:] / C[:-1] - 1.0
    gap[bad | (np.abs(gap) > 0.9)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        mult = np.clip(vmean / Vs, cfg.vt_clip[0], cfg.vt_clip[1])
        gsd = pd.DataFrame(gap).rolling(60, min_periods=30).std().shift(1).to_numpy()
        mkt_gap = np.nanmedian(gap, axis=1, keepdims=True)
        gap_z = (gap - mkt_gap) / gsd
    mult = np.where(np.isfinite(mult), mult, 1.0)
    news = (vr_idio >= cfg.news_vol_ratio) | (np.abs(gap_z) >= cfg.news_gap_z)
    news = np.where(np.isnan(vr_idio) & np.isnan(gap_z), False, news)
    er_ext = np.zeros((T + 12, N), bool) if earn_react is None else earn_react
    er = er_ext[:T]
    emove = np.full((T, N), np.nan, np.float32)
    for j in range(N):
        ev = np.where(er[:, j])[0]
        if len(ev):
            ma = pd.Series(np.abs(gap[ev, j])).rolling(8, min_periods=3).mean().to_numpy()
            idx = np.searchsorted(ev, np.arange(T), side='right') - 1
            okk = idx >= 0
            emove[okk, j] = ma[idx[okk]]
    Rm = R * mult if cfg.volume_time else R.copy()
    nan_cum = np.cumsum(np.isnan(Rm), axis=0)
    adv = pd.DataFrame(C * np.where(np.isnan(V), 0, V)).rolling(20, min_periods=15).median().to_numpy()
    univ = (C >= cfg.min_price) & (adv >= cfg.min_adv)
    ec = panel.etf_close
    ER = ec.pct_change(fill_method=None).to_numpy(float)
    ecol = {c: i for i, c in enumerate(ec.columns)}
    return dict(C=C, R=R, Rm=Rm, nan_cum=nan_cum, univ=univ, adv=adv, bad=bad, ER=ER, macro_cols=[ecol[c] for c in ("SPY", "USO", "TLT", "UUP")],
                sec_etf_idx=np.array([ecol[SECTOR_ETF[sectors[k]]] for k in sector_idx]), sector_idx=sector_idx, sectors=sectors, cols=cols, T=T, N=N,
                vol_ratio=vr_idio, gap_z=gap_z, news=news, er=er, er_ext=er_ext, emove=emove, earn_soon=ahead_any(er_ext, 2, 6)[:T], earn_next=ahead_any(er_ext, 2, 3)[:T], post_earn=behind_any(er, 1),
                dates=panel.dates.values)


def _run_range(D: dict, cfg: FactorCfg, ta: int, tb: int, progress=None, cancelled=lambda: False) -> dict:
    """The expensive loop for decision sessions ta..tb-1. Returns arrays with tb-ta rows."""
    N = D["N"]
    nrow = tb - ta
    m_max = cfg.n_factors if cfg.var_explained is None else 40
    out = {k: np.full((nrow, N), np.nan, np.float32) for k in ("s", "kappa", "seq", "rvol", "b_spy", "b_oil", "b_tlt", "b_usd", "b_sec")}
    out["L"] = np.full((nrow, N, m_max), np.nan, np.float32)
    reg = {k: np.full(nrow, np.nan) for k in ("avg_corr", "absorb5", "top1", "n_names", "n_fast", "m_used", "idio_share")}
    Rm, R, nan_cum, univ, ER = D["Rm"], D["R"], D["nan_cum"], D["univ"], D["ER"]
    sector_idx, sectors, macro_cols, sec_etf_idx = D["sector_idx"], D["sectors"], D["macro_cols"], D["sec_etf_idx"]
    W, M = cfg.est_window, cfg.pca_window
    b_max = math.exp(-cfg.kappa_min / TD)
    for t in range(ta, tb):
        r = t - ta
        if cancelled():
            break
        if progress and r % 20 == 0:
            progress(r, nrow)
        lo = t - M + 1
        ok = ((nan_cum[t] - nan_cum[lo - 1]) == 0) & univ[t]
        idx = np.where(ok)[0]
        if len(idx) < 80:
            continue
        sd = Rm[lo:t + 1][:, idx].std(axis=0, ddof=1)
        idx = idx[sd > 1e-8]                                         # halted / constant-price names cannot be standardised
        n = len(idx)
        if n < 80:
            continue
        Y = Rm[lo:t + 1][:, idx]
        sd = Y.std(axis=0, ddof=1)
        Y60 = Rm[t - W + 1:t + 1][:, idx]
        if cfg.factor_mode == "pca":
            Z = (Y - Y.mean(0)) / sd
            G = Z @ Z.T / (M - 1)                                    # Gram trick: M x M instead of n x n (same non-zero spectrum)
            if cfg.var_explained is None:
                m = cfg.n_factors
                w, u = eigh(G, subset_by_index=[M - m, M - 1])
                w, u = w[::-1], u[:, ::-1]
            else:
                w, u = eigh(G)
                w, u = w[::-1], u[:, ::-1]
                m = max(3, min(int(np.searchsorted(np.cumsum(w) / n, cfg.var_explained) + 1), m_max))
                w, u = w[:m], u[:, :m]
            v = Z.T @ u / np.sqrt(np.maximum(w, 1e-12) * (M - 1))   # unit-norm eigenvectors of the correlation matrix
            Q = v / sd[:, None]
            X = np.column_stack([np.ones(W), Y60 @ Q])
            zs = Z.sum(axis=1)
            reg["avg_corr"][r] = ((zs @ zs) / (M - 1) - n) / (n * (n - 1))
            reg["absorb5"][r] = w[:5].sum() / n
            reg["top1"][r] = w[0] / n
            reg["m_used"][r] = m
            coef = np.linalg.lstsq(X, Y60, rcond=None)[0]
            eps = Y60 - X @ coef
            out["L"][r, idx, :m] = coef[1:].T
        else:
            Fe = ER[t - W + 1:t + 1][:, sec_etf_idx[idx]]
            xm, ym = Fe.mean(0), Y60.mean(0)
            bet = ((Fe - xm) * (Y60 - ym)).sum(0) / ((Fe - xm) ** 2).sum(0)
            eps = (Y60 - ym) - (Fe - xm) * bet
            reg["m_used"][r] = 1
        Fs = ER[t - W + 1:t + 1][:, sec_etf_idx[idx]]
        with np.errstate(invalid='ignore', divide='ignore'):
            out['b_sec'][r, idx] = ((Fs - Fs.mean(0)) * (Y60 - Y60.mean(0))).sum(0) / ((Fs - Fs.mean(0)) ** 2).sum(0)
        if cfg.sector_demean:
            sidx = sector_idx[idx]
            for k in range(len(sectors)):
                ck = np.where(sidx == k)[0]
                if len(ck) >= 4:
                    eps[:, ck] -= np.median(eps[:, ck], axis=1, keepdims=True)
        Xc = np.cumsum(eps, axis=0)
        x0, x1 = Xc[:-1], Xc[1:]
        x0m, x1m = x0.mean(0), x1.mean(0)
        v0 = ((x0 - x0m) ** 2).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = ((x0 - x0m) * (x1 - x1m)).sum(0) / v0
            a = x1m - b * x0m
            zeta = x1 - a - b * x0
            var_z = (zeta ** 2).sum(0) / (W - 3)
            valid = (b > 0) & (b < b_max) & np.isfinite(b) & (var_z > 0)
            kappa = -np.log(np.where(valid, b, np.nan)) * TD
            mm = a / (1.0 - b)
            seq = np.sqrt(var_z / (1.0 - b ** 2))
        if valid.sum() >= 5:
            mm_c = mm - np.nanmean(np.where(valid, mm, np.nan))        # paper (22): centre the equilibrium means across stocks
            s = (Xc[-1] - mm_c) / seq
        else:
            s = np.full(n, np.nan)
        out["s"][r, idx] = np.where(valid, s, np.nan); out["kappa"][r, idx] = kappa; out["seq"][r, idx] = seq; out["rvol"][r, idx] = eps.std(axis=0, ddof=1)
        reg["n_names"][r], reg["n_fast"][r] = n, int(valid.sum())
        reg["idio_share"][r] = float(np.mean(eps.var(axis=0, ddof=1) / Y60.var(axis=0, ddof=1)))
        mw = cfg.macro_window
        Xm = np.column_stack([np.ones(mw), ER[t - mw + 1:t + 1][:, macro_cols]])
        if np.isfinite(Xm).all():
            Ym = np.where(np.isfinite(R[t - mw + 1:t + 1][:, idx]), R[t - mw + 1:t + 1][:, idx], 0.0)
            cm = np.linalg.lstsq(Xm, Ym, rcond=None)[0]
            out["b_spy"][r, idx], out["b_oil"][r, idx], out["b_tlt"][r, idx], out["b_usd"][r, idx] = cm[1], cm[2], cm[3], cm[4]
    out["reg"] = reg
    return out


def _chunk_job(args):
    """Process-pool entry point: reload the inputs from the temp folder (pipes choke on 50 MB payloads on Windows), run one slice of sessions."""
    import os
    from statarb_data import Panel
    tmp, cfg, ta, tb = args
    long = pd.read_parquet(os.path.join(tmp, "panel.parquet"))
    sec = pd.read_parquet(os.path.join(tmp, "sectors.parquet")).set_index("symbol").sector
    er_path = os.path.join(tmp, "earn.npy")
    er = np.load(er_path) if os.path.exists(er_path) else None
    D = _prepare(Panel(long, sector_map=sec, stocks=list(sec.index)), cfg, er)
    return ta, _run_range(D, cfg, ta, tb)


def _panel_long(panel) -> pd.DataFrame:
    """Long frame that statarb_data.Panel can be rebuilt from inside a worker process."""
    parts = [getattr(panel, f + "_all").stack().rename(f) for f in ("open", "close", "volume")]
    return pd.concat(parts, axis=1).rename_axis(["date", "symbol"]).reset_index()


def build_signals(panel, cfg: FactorCfg = FactorCfg(), earnings: Optional[pd.DataFrame] = None, progress: Optional[Callable[[int, int], None]] = None,
                  cancelled: Callable[[], bool] = lambda: False, n_jobs: int = 1) -> Signals:
    """Stage A. ``earnings`` is the Yahoo earnings calendar (statarb_data.load_earnings()); without it the earnings filters are inert."""
    from statarb_data import earnings_reaction_matrix
    er = earnings_reaction_matrix(panel, earnings, extra=12) if earnings is not None and len(earnings) else None
    D = _prepare(panel, cfg, er)
    T, N = D["T"], D["N"]
    t0 = max(int(np.searchsorted(D["dates"], np.datetime64(cfg.start))), cfg.pca_window + 1)
    m_max = cfg.n_factors if cfg.var_explained is None else 40
    if n_jobs > 1 and (T - t0) > 120:
        import os, tempfile
        from concurrent.futures import ProcessPoolExecutor
        edges = np.linspace(t0, T, n_jobs + 1).astype(int)
        with tempfile.TemporaryDirectory() as tmp:
            _panel_long(panel).to_parquet(os.path.join(tmp, "panel.parquet"))
            pd.DataFrame({"symbol": panel.stocks, "sector": [panel.sector[c] for c in panel.stocks]}).to_parquet(os.path.join(tmp, "sectors.parquet"))
            if er is not None:
                np.save(os.path.join(tmp, "earn.npy"), er)
            with ProcessPoolExecutor(n_jobs) as ex:
                res = list(ex.map(_chunk_job, [(tmp, cfg, int(edges[i]), int(edges[i + 1])) for i in range(n_jobs)]))
        parts = sorted(res, key=lambda x: x[0])
    else:
        parts = [(t0, _run_range(D, cfg, t0, T, progress, cancelled))]
    full = {k: np.full((T, N), np.nan, np.float32) for k in ("s", "kappa", "seq", "rvol", "b_spy", "b_oil", "b_tlt", "b_usd", "b_sec")}
    L_ = np.full((T, N, m_max), np.nan, np.float32)
    reg = {k: np.full(T, np.nan) for k in ("avg_corr", "absorb5", "top1", "n_names", "n_fast", "m_used", "idio_share")}
    for ta, o in parts:
        nrow = o["s"].shape[0]
        for k in full:
            full[k][ta:ta + nrow] = o[k]
        L_[ta:ta + nrow] = o["L"]
        for k in reg:
            reg[k][ta:ta + nrow] = o["reg"][k]
    C, univ, adv = D["C"], D["univ"], D["adv"]
    mom = np.full_like(C, np.nan); mom[252:] = C[252 - 21:T - 21] / C[:T - 252] - 1.0
    vol60 = pd.DataFrame(D["R"]).rolling(60, min_periods=40).std().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mom_z, vol_z, size_z = (_xs_z(a, univ).astype(np.float32) for a in (mom, np.log(vol60), np.log(adv)))
    regime = pd.DataFrame(reg, index=panel.dates)
    regime["disp"] = pd.DataFrame(D["R"]).std(axis=1, skipna=True).to_numpy()
    regime = regime.join(shock_table(panel))
    return Signals(cfg=cfg, dates=panel.dates, cols=D["cols"], sector_idx=D["sector_idx"], s=full["s"], kappa=full["kappa"], seq=full["seq"], rvol=full["rvol"], b_spy=full["b_spy"],
                   b_oil=full["b_oil"], b_tlt=full["b_tlt"], b_usd=full["b_usd"], L=L_, mom_z=mom_z, vol_z=vol_z, size_z=size_z, vol_ratio=D["vol_ratio"].astype(np.float32),
                   gap_z=D["gap_z"].astype(np.float32), news=D["news"], earn_react=D["er"], earn_soon=D["earn_soon"], earn_next=D["earn_next"], post_earn=D["post_earn"], univ=univ,
                   regime=regime, adv=adv.astype(np.float32), bad=D["bad"], b_sec=full["b_sec"], emove=D["emove"], earn_ext=D["er_ext"], sector_names=D["sectors"])


def shock_table(panel) -> pd.DataFrame:
    """Macro / geopolitical shock score from tradable proxies: equity sell-off (SPY), fear (VIXY), oil (USO, Hormuz), rates (TLT, Fed), gold (GLD safe-haven bid).
    Each leg is its 3-session move in units of its own trailing-year volatility (shifted a day); the score is the largest leg and the multiplier is built from it in stage B."""
    ec = panel.etf_close
    r = ec.pct_change(fill_method=None)
    r3 = ec.pct_change(3, fill_method=None)
    sd = r.rolling(252, min_periods=120).std().shift(1) * math.sqrt(3)
    z = r3 / sd
    out = pd.DataFrame(index=ec.index)
    out["z_spy"] = -z["SPY"]
    out["z_vix"] = z["VIXY"]
    out["z_oil"] = z["USO"].abs()
    out["z_tlt"] = z["TLT"].abs()
    out["z_gld"] = z["GLD"]
    out["shock"] = out[["z_spy", "z_vix", "z_oil", "z_tlt", "z_gld"]].max(axis=1)
    out["spy_rv21"] = r["SPY"].rolling(21).std() * math.sqrt(TD)
    return out
