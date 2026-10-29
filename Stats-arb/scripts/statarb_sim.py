"""
Stage B of the stat-arb strategy: signals -> positions -> P&L (no Qt, no network).

``Book.decide`` is THE decision function: the backtest calls it once per session and the live engine calls the very same method on the latest row, so what is tested is what trades.

  1. state machine (paper section 4):  open long s < -1.25, open short s > +1.25, close long s > -0.50, close short s < +0.75, plus a model-failure stop (|s| > 3.5), a time stop,
     and the 2026 idiosyncratic-risk rules (no fading of news / post-earnings sessions, nothing opened into an earnings window, positions closed before it)
  2. risk-weighted size  1/residual-vol, half the gross long and half short
  3. soft projection onto the null space of the risk exposures:  w = w0 - B (diag(1/lam) + B'B)^-1 B' w0      B = dollar, SPY beta, top PCA factors, sectors, momentum, volatility,
     size, oil beta, rates beta, USD beta  (each column scaled to unit RMS; lam ~ how hard the constraint is: lam*n >> 1 is exact, lam*n << 1 is ignored)
  4. name cap, no-trade band on continuing positions, gross scaling with hysteresis

``simulate`` runs it through history: decisions at the CLOSE of day d, fills at the OPEN of d+1, P&L open-to-open, costs on traded notional, borrow on shorts, EWMA vol targeting that only
uses P&L realised before the decision, and a shock multiplier built from SPY / VIXY / USO / TLT / GLD.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from statarb_core import PortCfg, Signals, TD


# ══ the decision function ═══════════════════════════════════════════════════════════
@dataclass
class DayInputs:
    """One decision date's cross-section (all arrays length N)."""
    s: np.ndarray; rvol: np.ndarray; univ: np.ndarray
    news: np.ndarray; earn_soon: np.ndarray; earn_next: np.ndarray; post_earn: np.ndarray; vr: np.ndarray
    b_spy: np.ndarray; b_oil: np.ndarray; b_tlt: np.ndarray; b_usd: np.ndarray
    mom_z: np.ndarray; vol_z: np.ndarray; size_z: np.ndarray
    L: np.ndarray                                # (N, m) PCA loadings
    sector_idx: np.ndarray
    b_sec: np.ndarray = None                     # beta to the stock's own sector SPDR (ETF hedge)
    emove: np.ndarray = None                     # average |earnings-day gap| of the last 8 announcements
    eap_ahead: np.ndarray = None                 # announcement (reaction session) scheduled lo..hi sessions ahead

    @staticmethod
    def from_signals(sig: Signals, i: int, p: Optional[PortCfg] = None) -> "DayInputs":
        n = len(sig.cols)
        eap = np.zeros(n, bool)
        if p is not None and sig.earn_ext is not None and p.w_eap > 0:
            eap = sig.earn_ext[i + p.eap_lo:i + p.eap_hi + 1].any(axis=0)
        return DayInputs(s=sig.s[i], rvol=sig.rvol[i], univ=sig.univ[i], news=sig.news[i], earn_soon=sig.earn_soon[i], earn_next=sig.earn_next[i], post_earn=sig.post_earn[i], vr=sig.vol_ratio[i],
                         b_spy=sig.b_spy[i], b_oil=sig.b_oil[i], b_tlt=sig.b_tlt[i], b_usd=sig.b_usd[i], mom_z=sig.mom_z[i], vol_z=sig.vol_z[i], size_z=sig.size_z[i], L=sig.L[i],
                         sector_idx=sig.sector_idx, b_sec=None if sig.b_sec is None else sig.b_sec[i], emove=None if sig.emove is None else sig.emove[i], eap_ahead=eap)


class Book:
    """Position state across sessions. ``decide`` returns the target weights (fraction of equity, signed) for the next open."""
    MIN_SIDE = 3

    def __init__(self, n: int, p: PortCfg):
        self.n, self.p = n, p
        self.state = np.zeros(n, np.int8)
        self.hold = np.zeros(n, np.int32)
        self.entry_s = np.zeros(n)
        self.w_prev = np.zeros(n)
        self.g_applied: Optional[float] = None
        self.events: list = []                    # (kind, idx, s, reason)

    # ── persistence (live) ───────────────────────────────────────────────────────
    def to_dict(self, cols: list) -> dict:
        nz = np.where(self.state != 0)[0]
        return {"positions": {cols[i]: {"side": int(self.state[i]), "hold": int(self.hold[i]), "entry_s": float(self.entry_s[i]), "w": float(self.w_prev[i])} for i in nz},
                "g_applied": self.g_applied}

    def load_dict(self, d: dict, cols: list) -> None:
        col = {c: i for i, c in enumerate(cols)}
        self.state[:] = 0; self.hold[:] = 0; self.w_prev[:] = 0; self.entry_s[:] = 0
        for sym, v in (d.get("positions") or {}).items():
            if sym in col:
                i = col[sym]
                self.state[i], self.hold[i], self.entry_s[i], self.w_prev[i] = v["side"], v["hold"], v.get("entry_s", 0.0), v.get("w", 0.0)
        self.g_applied = d.get("g_applied")

    # ── 1. state machine ─────────────────────────────────────────────────────────
    def _update_states(self, x: DayInputs) -> None:
        p, st = self.p, self.state
        s = x.s
        ok = np.isfinite(s) & x.univ
        was_flat = st == 0                                            # a name closed today is not re-opened (or flipped) before tomorrow's decision
        self.hold[st != 0] += 1
        with np.errstate(invalid="ignore"):
            ex_l = (st > 0) & (~ok | (s > -p.s_close_long) | (s < -p.s_stop) | (self.hold >= p.max_hold))
            ex_s = (st < 0) & (~ok | (s < p.s_close_short) | (s > p.s_stop) | (self.hold >= p.max_hold))
        if p.earn_filter and p.earn_exit:
            ex_l |= (st > 0) & x.earn_next
            ex_s |= (st < 0) & x.earn_next
        for i in np.where(ex_l | ex_s)[0]:
            self.events.append(("exit", int(i), float(s[i]) if np.isfinite(s[i]) else float("nan"), "model/stop/time/earnings"))
        st[ex_l | ex_s] = 0
        self.hold[st == 0] = 0
        block = ~ok
        if p.event_filter:
            block = block | x.news | x.post_earn
        if p.earn_filter and p.earn_entry_block:
            block = block | x.earn_soon
        with np.errstate(invalid="ignore", divide="ignore"):
            boost = 1.0 + p.vol_boost * np.clip(np.log(np.where(x.vr > 0, x.vr, 1.0)), 0.0, 2.0) if p.vol_boost else 1.0
            sc = s * boost
            vr_ok = (x.vr >= p.min_vr) if p.min_vr > 0 else True
            cl = was_flat & ~block & vr_ok & (sc < -p.s_open)
            cs = was_flat & ~block & vr_ok & (sc > p.s_open) & p.allow_short
        for mask, sign in ((cl, 1), (cs, -1)):
            room = p.max_pos_per_side - int((st == sign).sum())
            idx = np.where(mask)[0]
            if room <= 0:
                continue
            if len(idx) > room:
                idx = idx[np.argsort(-np.abs(s[idx]))[:room]]
            for i in idx:
                st[i] = sign
                self.hold[i] = 0
                self.entry_s[i] = s[i]
                self.events.append(("entry", int(i), float(s[i]), "long" if sign > 0 else "short"))

    # ── 3. exposures ─────────────────────────────────────────────────────────────
    def _exposure_cols(self, x: DayInputs, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
        p = self.p
        cols, lam, names = [], [], []

        def add(v, l, nm):
            v = np.nan_to_num(np.asarray(v, float), nan=0.0)
            rms = math.sqrt(float(np.mean(v ** 2)))
            if rms > 1e-9:
                cols.append(v / rms); lam.append(l); names.append(nm)
        add(np.ones(len(a)), p.lam_dollar, "dollar")
        add(x.b_spy[a], p.lam_beta, "beta")
        npc = min(p.n_pca_neutral, x.L.shape[1])
        for f in range(npc):
            add(x.L[a, f], p.lam_pca, f"pca{f + 1}")
        sidx = x.sector_idx[a]
        for k in np.unique(sidx):
            add((sidx == k).astype(float), p.lam_sector, f"sector{k}")
        if p.style_neutral:
            add(x.mom_z[a], p.lam_style, "momentum"); add(x.vol_z[a], p.lam_style, "volatility"); add(x.size_z[a], p.lam_style, "size")
        if p.macro_neutral:
            add(x.b_oil[a], p.lam_macro, "oil"); add(x.b_tlt[a], p.lam_macro, "rates"); add(x.b_usd[a], p.lam_macro, "usd")
        if not cols:
            return np.zeros((len(a), 0)), np.zeros(0), names
        return np.column_stack(cols), np.array(lam), names

    def _project(self, w0: np.ndarray, B: np.ndarray, lam: np.ndarray) -> np.ndarray:
        if B.shape[1] == 0:
            return w0
        A = np.diag(1.0 / lam) + B.T @ B
        return w0 - B @ np.linalg.solve(A, B.T @ w0)

    def _project_bounded(self, w0: np.ndarray, B: np.ndarray, lam: np.ndarray, d: np.ndarray, cap: float) -> np.ndarray:
        """min ||w - w0||^2 + sum lam_k (b_k.w)^2   s.t.  d_i w_i in [0, cap]   (a name can be switched off by the exposures, never reversed)."""
        from scipy.optimize import lsq_linear
        n = len(w0)
        A = np.vstack([np.eye(n), np.sqrt(lam)[:, None] * B.T])
        b = np.concatenate([w0, np.zeros(len(lam))])
        lb = np.where(d > 0, 0.0, -cap)
        ub = np.where(d > 0, cap, 0.0)
        return lsq_linear(A, b, bounds=(lb, ub), method="bvls").x

    # ── 2-4. weights ─────────────────────────────────────────────────────────────
    def decide(self, x: DayInputs, g_target: float) -> tuple[np.ndarray, dict]:
        p = self.p
        self.events = []
        self._update_states(x)
        st = self.state
        a = np.where(st != 0)[0]
        info = {"n_long": int((st > 0).sum()), "n_short": int((st < 0).sum()), "events": list(self.events)}
        target = np.zeros(self.n)
        if info["n_long"] < self.MIN_SIDE or (p.allow_short and info["n_short"] < self.MIN_SIDE):
            self.w_prev = target
            info["note"] = "fewer than %d names on a side - flat" % self.MIN_SIDE
            return target, info
        d = st[a].astype(float)
        rv = np.where(np.isfinite(x.rvol[a]) & (x.rvol[a] > 0), x.rvol[a], np.nanmedian(x.rvol[a]))
        w0 = d * (1.0 / rv) ** p.weight_pow
        if p.allow_short:
            w0 = np.where(d > 0, 0.5 * w0 / w0[d > 0].sum(), 0.5 * w0 / (-w0[d < 0].sum()))
        else:
            w0 = w0 / w0.sum()
        w = w0.copy()
        cap = p.name_cap
        if p.neutral:
            B, lam, names = self._exposure_cols(x, a)
            w = self._project(w0, B, lam)
            if np.any(np.sign(w) != d) or np.any(np.abs(w) > cap):         # closed form broke a sign / cap constraint
                act = np.ones(len(a), bool)                                # cheap first: drop the names the projection reversed and project again (a few rounds) ...
                for _ in range(3):
                    flip = act & (np.sign(w) != d)
                    if not flip.any():
                        break
                    act &= ~flip
                    w = np.zeros(len(a))
                    if act.sum() >= 2 * self.MIN_SIDE:
                        w[act] = self._project(w0[act], B[act], lam)
                if np.any(np.sign(w[act]) != d[act]) or act.sum() < 2 * self.MIN_SIDE:   # ... then the exact bounded least squares on the full set
                    w = self._project_bounded(w0, B, lam, d, cap)
        w = np.where(np.sign(w) == d, w, 0.0)
        if not np.any(w):
            self.w_prev = target
            info["note"] = "neutralisation left no book - flat"
            return target, info
        w = w / np.abs(w).sum()
        for _ in range(4):                                                   # name cap (re-balance the two sides after clipping)
            if not (np.abs(w) > cap + 1e-12).any():
                break
            w = np.clip(w, -cap, cap)
            if p.allow_short and (w > 0).any() and (w < 0).any():
                lg, sh = w[w > 0].sum(), -w[w < 0].sum()
                m = min(lg, sh)
                w = np.where(w > 0, w * m / lg, w * m / sh)
            w = w / np.abs(w).sum()
        # gross scaling with hysteresis: a small change in the risk target is not worth a rebalance
        if self.g_applied is None or abs(g_target / max(self.g_applied, 1e-9) - 1.0) > p.scale_band:
            self.g_applied = g_target
        t = np.zeros(self.n)
        t[a] = w * self.g_applied
        # no-trade band on continuing positions
        prev = self.w_prev
        cont = (prev != 0) & (np.sign(prev) == np.sign(t)) & (t != 0)
        keep = cont & (np.abs(t - prev) < p.band * np.abs(t))
        t[keep] = prev[keep]
        # a name dropped by the projection (t == 0) while the state is still on is simply not held
        self.w_prev = t
        info["g_applied"] = self.g_applied
        info["n_held"] = int((t != 0).sum())
        return t, info


# ══ the sleeves combined ════════════════════════════════════════════════════════════
class Portfolio:
    """Both sleeves behind one ``decide``: the Avellaneda-Lee residual-reversion Book and the earnings-announcement-premium (EAP) sleeve, plus the sector-ETF hedge.
    Returns stock weights (N), ETF hedge weights (E) and an info dict. Used identically by ``simulate`` and the live engine."""

    def __init__(self, n: int, p: PortCfg, ecols: list, sec_e: np.ndarray, sector_idx: Optional[np.ndarray] = None):
        self.n, self.p, self.ecols, self.sec_e = n, p, list(ecols), np.asarray(sec_e)
        self.sector_idx = np.asarray(sector_idx) if sector_idx is not None else self.sec_e         # sector id per stock (the ETF index is a one-to-one label of it)
        self.etf_ok: Optional[np.ndarray] = None                                                    # (E,) bool: which sector ETFs have a price today (XLC only exists from mid-2018)
        self.book = Book(n, p)
        self.eap_w = np.zeros(n)
        self.g_eap: Optional[float] = None

    def to_dict(self, cols: list) -> dict:
        nz = np.where(self.eap_w != 0)[0]
        return {"rev": self.book.to_dict(cols), "eap": {"g_applied": self.g_eap, "w": {cols[i]: float(self.eap_w[i]) for i in nz}}}

    def load_dict(self, d: dict, cols: list) -> None:
        self.book.load_dict(d.get("rev") or {}, cols)
        e = d.get("eap") or {}
        col = {c: i for i, c in enumerate(cols)}
        self.eap_w[:] = 0
        for sym, w in (e.get("w") or {}).items():
            if sym in col:
                self.eap_w[col[sym]] = w
        self.g_eap = e.get("g_applied")

    def _eap(self, x: DayInputs, g: float) -> tuple[np.ndarray, list]:
        p = self.p
        sel = x.eap_ahead & x.univ
        if x.emove is not None:
            sel = sel & ~(x.emove > p.eap_max_move)
        if p.eap_pool == "cheap":
            sel = sel & np.isfinite(x.s) & (x.s < 0)
        elif p.eap_pool == "notrich":
            sel = sel & ~(x.s >= 1.0)
        idx = np.where(sel)[0]
        if p.eap_max_names and len(idx) > p.eap_max_names:                      # keep the cheapest N by the paper's own s-score (unfitted names count as neutral)
            idx = idx[np.argsort(np.nan_to_num(x.s[idx], nan=0.0))[:p.eap_max_names]]
        w = np.zeros(self.n)
        if len(idx) >= p.eap_min_names:
            v = np.minimum(np.ones(len(idx)) / len(idx), p.eap_cap)              # a REAL cap: with few announcers the book is simply smaller (the rest stays in cash)
            if p.eap_sector_cap < 1.0:
                sec = self.sector_idx[idx]
                for k in np.unique(sec):
                    m = sec == k
                    tot = v[m].sum()
                    if tot > p.eap_sector_cap:
                        v[m] *= p.eap_sector_cap / tot
            w[idx] = v
        if self.g_eap is None or abs(g / max(self.g_eap, 1e-9) - 1.0) > p.scale_band:
            self.g_eap = g
        t = w * self.g_eap
        ev = [("entry", int(i), float("nan"), "eap") for i in np.where((t != 0) & (self.eap_w == 0))[0]] + [("exit", int(i), float("nan"), "eap") for i in np.where((t == 0) & (self.eap_w != 0))[0]]
        self.eap_w = t
        return t, ev

    def decide(self, x: DayInputs, g_target: float) -> tuple[np.ndarray, np.ndarray, dict]:
        p = self.p
        stock = np.zeros(self.n)
        info: dict = {"events": [], "n_long": 0, "n_short": 0}
        T_rev = np.zeros(self.n)
        if p.w_rev > 0:
            T_rev, info = self.book.decide(x, g_target * p.w_rev)
            stock += T_rev
        T_eap = np.zeros(self.n)
        if p.w_eap > 0:
            T_eap, ev = self._eap(x, g_target * p.w_eap)
            stock += T_eap
            info["events"] = list(info.get("events", [])) + ev
        info["n_eap"] = int((T_eap != 0).sum())
        hedged = (T_eap if p.eap_hedge else 0.0) + (T_rev if p.hedge == "etf" else 0.0)
        etf = np.zeros(len(self.ecols))
        if np.any(hedged != 0) and x.b_sec is not None:
            sec_e = self.sec_e
            if self.etf_ok is not None and "SPY" in self.ecols:                                    # a sector whose ETF has no price yet is hedged with SPY instead
                sec_e = np.where(self.etf_ok[sec_e], sec_e, self.ecols.index("SPY"))
            etf = -np.bincount(sec_e, weights=hedged * np.nan_to_num(x.b_sec, nan=1.0), minlength=len(self.ecols))
        gross = float(np.abs(stock).sum() + np.abs(etf).sum())
        if gross > p.g_max + 1e-9:                                        # Reg-T: long + short <= g_max x equity
            k = p.g_max / gross
            stock, etf = stock * k, etf * k
            self.book.w_prev = self.book.w_prev * k
            self.eap_w = self.eap_w * k
            info["capped"] = k
        info["n_long"] = int((stock > 0).sum()); info["n_short"] = int((stock < 0).sum())
        info["gross_stock"] = float(np.abs(stock).sum()); info["gross_etf"] = float(np.abs(etf).sum())
        return stock, etf, info


# ══ the simulator ═══════════════════════════════════════════════════════════════════
@dataclass
class Result:
    cfg: PortCfg
    daily: pd.DataFrame                           # index = date of the OPEN the period starts at
    weights: np.ndarray                           # (T, N) weights held over [open d, open d+1]
    trades: pd.DataFrame
    cols: list
    notes: list = field(default_factory=list)

    @property
    def ret(self) -> pd.Series:
        return self.daily["ret"]


def shock_multiplier(regime: pd.DataFrame, p: PortCfg) -> pd.Series:
    if not p.shock:
        return pd.Series(1.0, index=regime.index)
    excess = (regime["shock"] - p.shock_z0).clip(lower=0).fillna(0.0)
    raw = (1.0 - p.shock_k * excess).clip(lower=p.shock_min, upper=1.0)
    return raw.rolling(p.shock_hold, min_periods=1).min()


def cost_vec(adv: np.ndarray, base_bps: float) -> np.ndarray:
    """One-way cost per name (fraction): the large-cap cost times a liquidity multiplier from the 20-session median dollar volume (>= $200M 1x, $50-200M 1.7x, $20-50M 2.7x, < $20M 4x)."""
    m = np.where(adv >= 2e8, 1.0, np.where(adv >= 5e7, 1.7, np.where(adv >= 2e7, 2.7, 4.0)))
    return m * base_bps / 1e4


def sector_betas(panel, sec_e: np.ndarray, ecols: list, win: int) -> np.ndarray:
    """Rolling beta of every stock's daily return on its own sector SPDR (known at the close of the row's date). Research helper; the engine uses Signals.b_sec."""
    r = panel.close.pct_change(fill_method=None)
    re = panel.etf_close.pct_change(fill_method=None)
    out = np.full(r.shape, np.nan)
    for e in np.unique(sec_e):
        cols = np.where(sec_e == e)[0]
        x = re.iloc[:, e]
        var = x.rolling(win, min_periods=win // 2).var()
        for j in cols:
            cov = r.iloc[:, j].rolling(win, min_periods=win // 2).cov(x)
            out[:, j] = (cov / var).to_numpy()
    return out


def sector_etf_index(sig: Signals, ecols: list) -> np.ndarray:
    from statarb_data import SECTOR_ETF
    return np.array([ecols.index(SECTOR_ETF[sig.sector_names[k]]) for k in sig.sector_idx])


def simulate(sig: Signals, panel, p: PortCfg = PortCfg(), progress: Optional[Callable[[int, int], None]] = None, cancelled: Callable[[], bool] = lambda: False) -> Result:
    O = panel.open.to_numpy(float)
    T, N = O.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        ro = np.full((T, N), 0.0)
        r = O[1:] / O[:-1] - 1.0
        vr_next = sig.vol_ratio[1:]
        r[~np.isfinite(r) | ((np.abs(r) > sig.cfg.max_abs_ret) & ~(vr_next >= 3.0))] = 0.0    # missing prices / unconfirmed corporate-action artefacts earn nothing
        r[np.abs(r) > 0.95] = 0.0
        ro[:-1] = r
    d0 = max(int(np.searchsorted(sig.dates.values, np.datetime64(p.start))), 0)
    first = np.where(np.isfinite(sig.s).any(axis=1))[0]
    d0 = max(d0, int(first[0]) if len(first) else T)
    d1 = T - 2 if p.end is None else min(T - 2, int(np.searchsorted(sig.dates.values, np.datetime64(p.end))))
    mult = shock_multiplier(sig.regime, p).to_numpy()
    eo = panel.etf_open
    ecols = list(eo.columns)
    sec_e = sector_etf_index(sig, ecols)
    with np.errstate(invalid="ignore", divide="ignore"):
        ero = np.zeros((T, len(ecols))); ero[:-1] = np.nan_to_num(eo.to_numpy(float)[1:] / eo.to_numpy(float)[:-1] - 1.0)
    port = Portfolio(N, p, ecols, sec_e, sig.sector_idx)
    etf_px_ok = np.isfinite(eo.to_numpy(float))
    re_ = panel.etf_close.pct_change(fill_method=None)
    etf_beta = (re_.rolling(60, min_periods=30).cov(re_["SPY"]).div(re_["SPY"].rolling(60, min_periods=30).var(), axis=0)).to_numpy() if "SPY" in ecols else np.ones((T, len(ecols)))
    H, Hh = np.zeros(N), np.zeros(len(ecols))
    alpha = 1.0 - 0.5 ** (1.0 / max(p.vol_halflife, 1))
    var_u, n_u = None, 0
    rows, trades, open_pos = [], [], {}
    cols = sig.cols
    bor, tce = p.borrow_bps / 1e4 / TD, p.tc_etf_bps / 1e4
    adv_arr = sig.adv if sig.adv is not None else np.full((T, N), 1e9, np.float32)
    W = np.zeros((T, N), np.float32)
    unit_gross = max(p.w_rev * (1.0 if p.hedge != "etf" else 2.0) + p.w_eap * (2.0 if p.eap_hedge else 1.0), 1e-9)       # gross per unit of the risk multiplier g (EAP long 1 + hedge ~1)
    g_cap = p.g_max / max(unit_gross, 1.0)
    for d in range(d0, d1 + 1):
        if cancelled():
            break
        if progress and (d - d0) % 50 == 0:
            progress(d - d0, d1 - d0 + 1)
        # 1. decision at the close of d: the vol estimate only knows P&L up to the open of d
        if var_u is not None and n_u >= p.vol_warmup and var_u > 0:
            g = float(np.clip(p.vol_target / math.sqrt(var_u * TD), p.g_min, g_cap))
        else:
            g = min(p.g_init, g_cap)
        g_t = float(min(g * mult[d], g_cap))
        x = DayInputs.from_signals(sig, d, p)
        port.etf_ok = etf_px_ok[d]
        T_w, Th, info = port.decide(x, g_t)
        # 2. realise [open d, open d+1] for the book that was set at the open of d
        gross_h = float(np.abs(H).sum() + np.abs(Hh).sum())
        pnl = float(H @ ro[d] + Hh @ ero[d])
        borrow = bor * float(-H[H < 0].sum()) + (25e-4 / TD) * float(-Hh[Hh < 0].sum())
        net = pnl - borrow
        if gross_h > 0.2:
            u = pnl / gross_h
            var_u = u * u if var_u is None else (1 - alpha) * var_u + alpha * u * u
            n_u += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            Hd = H * (1.0 + ro[d]) / (1.0 + net)
            Hhd = Hh * (1.0 + ero[d]) / (1.0 + net)
        # 3. trade at open d+1
        turn = float(np.abs(T_w - Hd).sum() + np.abs(Th - Hhd).sum())
        cost = float(np.abs(T_w - Hd) @ cost_vec(adv_arr[d], p.tc_bps)) + tce * float(np.abs(Th - Hhd).sum())
        ret = net - cost
        for kind, i, s_, why in info["events"]:
            px = O[d + 1, i]
            if kind == "entry":
                open_pos[i] = (sig.dates[d + 1], px, -1 if why == "short" else 1, s_, "eap" if why == "eap" else "reversion")
            elif i in open_pos:
                t0, p0, sd, s0, sleeve = open_pos.pop(i)
                if p0 > 0 and np.isfinite(px):
                    trades.append({"symbol": cols[i], "sleeve": sleeve, "side": "long" if sd > 0 else "short", "entry": t0, "exit": sig.dates[d + 1], "entry_px": p0, "exit_px": float(px),
                                   "ret": sd * (px / p0 - 1.0), "s_entry": s0, "days": int(np.busday_count(t0.date(), sig.dates[d + 1].date()))})
        H, Hh = T_w, Th
        W[d + 1] = H
        rows.append((sig.dates[d], ret, net, cost, borrow, turn, float(np.abs(H).sum() + np.abs(Hh).sum()), float(H[H > 0].sum()), float(H[H < 0].sum() + Hh[Hh < 0].sum()), info["n_long"], info["n_short"],
                     float(H @ np.nan_to_num(x.b_spy) + Hh @ np.nan_to_num(etf_beta[d], nan=1.0)), float(mult[d]), g_t, info.get("n_eap", 0), float(np.abs(Hh).sum())))
    daily = pd.DataFrame(rows, columns=["date", "ret", "gross_ret", "cost", "borrow", "turnover", "gross", "long", "short", "n_long", "n_short", "beta", "shock_mult", "g_target", "n_eap", "etf_gross"]).set_index("date")
    return Result(cfg=p, daily=daily, weights=W, trades=pd.DataFrame(trades), cols=cols)


# ══ metrics ═════════════════════════════════════════════════════════════════════════
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def psr(sr: float, n: int, skew: float, kurt: float, sr0: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado): P(true SR > sr0) given the sample's length, skew and kurtosis. ``sr`` is per-period (not annualised)."""
    den = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return _norm_cdf((sr - sr0) * math.sqrt(max(n - 1, 1)) / den)


def dsr(sr: float, n: int, skew: float, kurt: float, n_trials: int, sr_var: float) -> float:
    """Deflated Sharpe ratio: PSR against the Sharpe the best of ``n_trials`` luck-only strategies would show."""
    if n_trials <= 1:
        return psr(sr, n, skew, kurt, 0.0)
    g = 0.5772156649
    sr0 = math.sqrt(sr_var) * ((1 - g) * _norm_ppf(1 - 1.0 / n_trials) + g * _norm_ppf(1 - 1.0 / (n_trials * math.e)))
    return psr(sr, n, skew, kurt, sr0)


def metrics(ret: pd.Series, bench: Optional[pd.Series] = None, rf: float = 0.0, n_trials: int = 1, sr_var: float = 0.0) -> dict:
    r = ret.dropna()
    if len(r) < 20:
        return {"n": len(r)}
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(r) / TD
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    down = r[r < 0].std(ddof=1)
    sr_d = (mu - rf / TD) / sd if sd > 0 else 0.0
    skew, kurt = float(r.skew()), float(r.kurt() + 3.0)
    out = {"n": len(r), "years": yrs, "cagr": float(cagr), "mean_ann": float(mu * TD), "vol": float(sd * math.sqrt(TD)), "sharpe": float(sr_d * math.sqrt(TD)),
           "sortino": float((mu - rf / TD) / down * math.sqrt(TD)) if down > 0 else float("nan"), "maxdd": float(dd.min()), "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else float("nan"),
           "hit": float((r > 0).mean()), "skew": skew, "kurt": kurt, "best_day": float(r.max()), "worst_day": float(r.min()), "worst_21d": float(r.rolling(21).sum().min()),
           "psr0": psr(sr_d, len(r), skew, kurt, 0.0), "total": float(eq.iloc[-1] - 1),
           "dsr": dsr(sr_d, len(r), skew, kurt, n_trials, sr_var) if n_trials > 1 and sr_var > 0 else float("nan")}
    # Sharpe standard error (Lo 2002, iid): +- 1.96 * sqrt((1 + sr^2/2)/n) * sqrt(252)
    out["sharpe_se"] = float(math.sqrt((1 + sr_d ** 2 / 2) / len(r)) * math.sqrt(TD))
    if bench is not None:
        b = bench.reindex(r.index).dropna()
        rr = r.reindex(b.index)
        if len(b) > 30 and b.std() > 0:
            beta = float(np.cov(rr, b)[0, 1] / b.var())
            out.update(beta=beta, alpha_ann=float((rr.mean() - beta * b.mean()) * TD), corr=float(np.corrcoef(rr, b)[0, 1]), bench_sharpe=float(b.mean() / b.std() * math.sqrt(TD)),
                       bench_cagr=float((1 + b).prod() ** (TD / len(b)) - 1), bench_maxdd=float(((1 + b).cumprod() / (1 + b).cumprod().cummax() - 1).min()))
    return out


def yearly(ret: pd.Series) -> pd.DataFrame:
    rows = []
    for y, g in ret.groupby(ret.index.year):
        eq = (1 + g).cumprod()
        rows.append({"year": int(y), "return": float(eq.iloc[-1] - 1), "vol": float(g.std() * math.sqrt(TD)), "sharpe": float(g.mean() / g.std() * math.sqrt(TD)) if g.std() > 0 else float("nan"),
                     "maxdd": float((eq / eq.cummax() - 1).min()), "days": len(g)})
    return pd.DataFrame(rows).set_index("year")


def spy_open_to_open(panel) -> pd.Series:
    o = panel.etf_open["SPY"]
    return (o.shift(-1) / o - 1.0).dropna()


def fmt_metrics(m: dict) -> str:
    if m.get("n", 0) < 20:
        return "n/a"
    s = (f"CAGR {m['cagr']:+.1%} · vol {m['vol']:.1%} · Sharpe {m['sharpe']:.2f} (±{1.96 * m['sharpe_se']:.2f}) · Sortino {m['sortino']:.2f} · MaxDD {m['maxdd']:.1%} · "
         f"Calmar {m['calmar']:.2f} · PSR {m['psr0']:.0%}")
    if "beta" in m:
        s += f" · β {m['beta']:+.2f} · corr {m['corr']:+.2f}"
    return s
