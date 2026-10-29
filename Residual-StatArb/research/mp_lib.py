"""
Shared machinery for the momentum / prediction searches (mp_*.py): a cached context of aligned arrays, three honest simulators and a trial book.

Timing rules (the same as the MR study):
  * a 15:45 (15:30) decision sees the official daily bars of earlier sessions and today's 15-minute SIP bars that ended by then; it fills in the CLOSING auction;
  * an overnight position is sold in the next OPENING auction (official open), or - as a check - at the first trade / the 9:45 price;
  * slow (weekly / monthly) models decide on the previous session's close and fill in today's closing auction;
  * intraday positions enter at a 15-minute bar's close (paying the half spread) and leave in the closing auction.
No leverage: the weights of a sleeve sum to <= 1 at every moment. Two sleeves may share the same capital only when they never hold at the same time
(overnight close -> open, intraday open -> close).
"""
import math
import os
import time

import numpy as np
import pandas as pd
import scipy.stats as SS

from mp_data import SPLIT, START, Panel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
TD = 252


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)


class Ctx:
    def __init__(self):
        p = self.p = Panel()
        self.dates, self.syms, self.col = p.dates, p.syms, p.col
        self.T, self.N = p.C.shape
        self.O, self.H, self.L, self.C, self.V = p.O, p.H, p.L, p.C, p.V
        self.t0 = int(np.searchsorted(self.dates, pd.Timestamp(START)))
        self.ts = int(np.searchsorted(self.dates, pd.Timestamp(SPLIT)))
        with np.errstate(invalid="ignore", divide="ignore"):
            prev = self.lag(self.C)
            self.rc = self.C / prev - 1                       # close t-1 -> close t (dated t)
            self.ron = self.O / prev - 1                      # close t-1 -> open t (dated t)
            self.rid = self.C / self.O - 1                    # open t -> close t
        self.spy = self.col["SPY"]
        self._c = {}

    # helpers ------------------------------------------------------------------
    @staticmethod
    def lag(x, n=1):
        out = np.full_like(x, np.nan, dtype=float)
        out[n:] = x[:-n]
        return out

    @staticmethod
    def lead(x, n=1):
        out = np.full_like(x, np.nan, dtype=float)
        out[:-n] = x[n:]
        return out

    def j(self, s):
        return self.col[s]

    def cached(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def top(self, n):
        return self.cached(f"top{n}", lambda: self.p.top(n))

    def etf(self):
        return self.cached("etf", self.p.etf_mask)

    def snap(self, lead=1):
        return self.cached(f"snap{lead}", lambda: self.p.snap(lead))

    def slot(self, k):
        return self.cached(f"slot{k}", lambda: self.p.at_slot(k))

    def first(self):
        return self.cached("first", lambda: self.p.cube()[1].astype(float))

    def cumvol(self, lead=1):
        """T x N: session volume in the bars that ended by the decision time (lead bars before the close)."""
        def f():
            _, _, vo, nb = self.p.cube()
            out = np.full((self.T, self.N), np.nan)
            for t in range(self.T):
                k = int(nb[t]) - lead
                if k > 0:
                    out[t] = np.nansum(vo[t, :, :k], axis=1)
            out[out == 0] = np.nan
            return out
        return self.cached(f"cumvol{lead}", f)

    def beta(self, n=60):
        """T x N rolling beta to SPY of close-to-close returns over the n sessions ending yesterday."""
        def f():
            x = self.rc
            y = np.broadcast_to(self.rc[:, [self.spy]], x.shape)
            ok = np.isfinite(x) & np.isfinite(y)
            x0, y0 = np.where(ok, x, 0.0), np.where(ok, y, 0.0)

            def roll(a):
                c = np.cumsum(a, axis=0)
                c[n:] = c[n:] - c[:-n].copy()
                return c
            k = roll(ok.astype(float))
            sx, sy, sxy, syy = roll(x0), roll(y0), roll(x0 * y0), roll(y0 * y0)
            with np.errstate(invalid="ignore", divide="ignore"):
                b = (sxy - sx * sy / k) / (syy - sy * sy / k)
            b[k < n // 2] = np.nan
            return self.lag(b)
        return self.cached(f"beta{n}", f)

    def series(self, r):
        return pd.Series(r, index=self.dates)[self.t0:]

    def spy_ret(self):
        return self.series(np.nan_to_num(self.rc[:, self.spy]))


def topk(score, mask, k, largest=True):
    """Boolean T x N: the k best scores among mask & finite (ties by column order)."""
    cols = np.where(mask.any(0))[0]
    x = np.where(mask[:, cols] & np.isfinite(score[:, cols]), score[:, cols], np.nan)
    x = -x if largest else x
    kk = min(k, len(cols))
    order = np.argsort(np.where(np.isfinite(x), x, np.inf), axis=1, kind="stable")[:, :kk]
    sub = np.zeros(x.shape, dtype=bool)
    sub[np.arange(len(x))[:, None], order] = True
    sel = np.zeros(mask.shape, dtype=bool)
    sel[:, cols] = sub & np.isfinite(x)
    return sel


# ------------------------------------------------------------------------------ simulators
def overnight(cx: Ctx, sel, w=None, cost_in=1.0, cost_out=1.0, exit="open", k=None):
    """Hold the selected names close t -> exit on t+1. Weight 1/k each (or ``w`` rows summing to <= 1). Returns the daily return series dated t+1."""
    if exit == "open":
        px = cx.O
    elif exit == "first":
        px = np.where(np.isfinite(cx.first()), cx.first(), cx.O)
    elif exit == "945":
        px = np.where(np.isfinite(cx.slot(0)), cx.slot(0), cx.O)
    else:
        raise ValueError(exit)
    r = px / cx.lag(cx.C) - 1                                   # dated t+1
    r = np.where(np.isfinite(r), r, 0.0)
    if w is None:
        kk = k or int(sel.sum(1).max() or 1)
        w = sel / kk
    wl = cx.lag(np.asarray(w, float))
    wl = np.nan_to_num(wl)
    gross = wl.sum(1)
    out = (wl * r).sum(1) - gross * (cost_in + cost_out) / 1e4
    return cx.series(out)


def intraday(cx: Ctx, w, entry_px, cost_in=2.0, cost_out=1.0):
    """Hold weights ``w`` (decided on t, rows sum <= 1) from ``entry_px`` on t to the close of t. Dated t."""
    r = cx.C / entry_px - 1
    r = np.where(np.isfinite(r), r, 0.0)
    w = np.nan_to_num(np.asarray(w, float))
    out = (w * r).sum(1) - w.sum(1) * (cost_in + cost_out) / 1e4
    return cx.series(out)


def rebalance(cx: Ctx, target, rebal, cost=1.0, ret=None):
    """Multi-day holding: on each rebalance session t the book is set to ``target[t]`` at the close (rows sum <= 1), then drifts with the close-to-close returns.
    ``target`` must only use information available before the order is sent. Costs ``cost`` bp per side on the traded value. Dated by the session the return is earned."""
    R = np.where(np.isfinite(cx.rc), cx.rc, 0.0) if ret is None else ret
    T, N = R.shape
    w = np.zeros(N)
    out = np.zeros(T)
    turn = np.zeros(T)
    for t in range(cx.t0, T):
        if t > cx.t0:
            g = w * (1 + R[t])
            v = g.sum() + (1 - w.sum())                           # cash earns 0
            out[t] = v - 1
            w = g / v if v > 0 else w * 0
        if rebal[t]:
            tg = np.nan_to_num(target[t])
            tr = np.abs(tg - w).sum()
            out[t] -= tr * cost / 1e4
            turn[t] = tr
            w = tg.copy()
    s = cx.series(out)
    s.attrs["turnover"] = float(turn[cx.t0:].sum() / ((T - cx.t0) / TD))
    return s


# ------------------------------------------------------------------------------ statistics
def stats(r: pd.Series, bench: pd.Series = None) -> dict:
    r = r.dropna()
    out = {}
    for lab, x in (("all", r), ("is", r[r.index < SPLIT]), ("oos", r[r.index >= SPLIT])):
        if len(x) < 30:
            continue
        eq = (1 + x).cumprod()
        yrs = len(x) / TD
        sd = x.std()
        out[f"sh_{lab}"] = x.mean() / sd * math.sqrt(TD) if sd > 0 else np.nan
        out[f"cagr_{lab}"] = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
        out[f"mdd_{lab}"] = (eq / eq.cummax() - 1).min()
        out[f"vol_{lab}"] = sd * math.sqrt(TD)
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        out["beta"] = float(np.cov(r, b)[0, 1] / b.var()) if b.var() > 0 else np.nan
        out["corr_spy"] = float(np.corrcoef(r, b)[0, 1])
    return out


def psr(r: pd.Series, bench_sr_ann: float = 0.0) -> float:
    r = r.dropna()
    sr, b = r.mean() / r.std(), bench_sr_ann / math.sqrt(TD)
    sk, ku = SS.skew(r), SS.kurtosis(r, fisher=False)
    return float(SS.norm.cdf((sr - b) * math.sqrt(len(r) - 1) / math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))))


def dsr(r: pd.Series, n: int, var_ann: float) -> tuple:
    """Bailey & Lopez de Prado (2014): P(true Sharpe > expected max of n unskilled trials with this cross-trial variance)."""
    r = r.dropna()
    g = 0.5772156649
    v = var_ann / TD
    sr0 = math.sqrt(v) * ((1 - g) * SS.norm.ppf(1 - 1 / n) + g * SS.norm.ppf(1 - 1 / (n * math.e)))
    return psr(r, sr0 * math.sqrt(TD)), sr0 * math.sqrt(TD)


class Book:
    """Collects trials: one row of stats per variant plus its daily return series (for DSR and correlations)."""

    def __init__(self, name: str, cx: Ctx):
        self.name, self.cx = name, cx
        self.rows, self.rets = [], {}
        self.bench = cx.spy_ret()

    def add(self, r: pd.Series, label: str, **kw):
        st = stats(r, self.bench)
        st.update(kw)
        st["label"] = label
        if "turnover" in r.attrs:
            st.setdefault("turnover", r.attrs["turnover"])
        self.rows.append(st)
        self.rets[label] = r
        return st

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self):
        df = self.frame()
        df.to_csv(os.path.join(OUT, f"{self.name}.csv"), index=False)
        pd.DataFrame(self.rets).to_parquet(os.path.join(OUT, f"{self.name}_rets.parquet"))
        return df

    def show(self, sort="sh_is", n=40, cols=None):
        df = self.frame()
        cols = cols or ["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all", "beta"]
        cols = [c for c in cols if c in df]
        with pd.option_context("display.width", 260, "display.max_rows", 400, "display.max_colwidth", 90):
            print(df.sort_values(sort, ascending=False)[cols].head(n).round(3).to_string(index=False), flush=True)
