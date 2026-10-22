"""Exhaustive parameter sweep over the full grid in parallel (numba prange), TRAIN window only.

Test-window evaluation is deliberately NOT possible from this module: it refuses any window that is not the Train
window or a sub-window of it, so parameter selection cannot peek at out-of-sample data.
"""
import time
import numpy as np
import pandas as pd
from numba import njit, prange
from sim import *
from sim import _core

TRAIN_END = pd.Timestamp(TRAIN[1])
GRID_SHAPE = (len(RSI_LENS), len(RSI_ENTRIES), len(EXIT_LENS), len(TREND_LENS), len(MAX_HOLDS))
H1 = ("2010-02-11", "2015-12-31")
H2 = ("2016-01-01", "2021-12-31")


@njit(parallel=True, cache=True)
def _sweep(op, cl, pc, rsiB, sxB, stB, inwin, last, mode, dip_trend, comm, split, tick, qmode, rsi_entries, max_holds):
    nR = rsiB.shape[0]; nE = len(rsi_entries); nX = sxB.shape[0]; nT = stB.shape[0]; nH = len(max_holds)
    N = nR * nE * nX * nT * nH
    out = np.zeros((N, 20))
    for k in prange(N):
        h = k % nH; r = k // nH
        t = r % nT; r //= nT
        x = r % nX; r //= nX
        e = r % nE; r //= nE
        ri = r
        o, rets, a1, a2, b1, b2 = _core(op, cl, pc, rsiB[ri], sxB[x], stB[t], inwin, last, mode,
                                        rsi_entries[e], dip_trend, max_holds[h], comm, split, False, tick, qmode)
        for j in range(16):
            out[k, j] = o[j]
        out[k, 16] = a1; out[k, 17] = a2; out[k, 18] = b1; out[k, 19] = b2
    return out


def derive(out):
    """Vectorised metrics from the raw sums; returns dict of arrays shaped GRID_SHAPE."""
    nd = out[:, 0]; E = out[:, 1]; mdd = out[:, 2]; sS = out[:, 3]; sS2 = out[:, 4]; sNeg = out[:, 5]
    tr = out[:, 6]; w = out[:, 7]; gp = out[:, 8]; gl = out[:, 9]; ip = out[:, 10]; sB = out[:, 11]; sB2 = out[:, 12]; sSB = out[:, 13]
    n1 = out[:, 14]; n2 = out[:, 15]; a1 = out[:, 16]; a2 = out[:, 17]; b1 = out[:, 18]; b2 = out[:, 19]
    with np.errstate(all="ignore"):
        mean = sS / nd
        sd = np.sqrt(np.maximum(sS2 / nd - mean ** 2, 0))
        sharpe = mean / sd * np.sqrt(252)
        sortino = mean / np.sqrt(sNeg / nd) * np.sqrt(252)
        cagr = E ** (252.0 / nd) - 1
        calmar = cagr / mdd
        m1 = a1 / n1; sd1 = np.sqrt(np.maximum(a2 / n1 - m1 ** 2, 0)); sh1 = m1 / sd1 * np.sqrt(252)
        m2 = b1 / n2; sd2 = np.sqrt(np.maximum(b2 / n2 - m2 ** 2, 0)); sh2 = m2 / sd2 * np.sqrt(252)
        vS = nd * sS2 - sS ** 2; vB = nd * sB2 - sB ** 2
        corr = (nd * sSB - sS * sB) / np.sqrt(vS * vB)
        res = dict(sharpe=sharpe, sortino=sortino, cagr=cagr, mdd=mdd, calmar=calmar, sh1=sh1, sh2=sh2, trades=tr,
                   win=w / tr, pf=gp / gl, expo=ip / nd, corr=corr, final=E)
    return {k: v.reshape(GRID_SHAPE) for k, v in res.items()}


def sweep_market(m, mode, dip_trend=True, comm=0.001, win=TRAIN):
    s, e = pd.Timestamp(win[0]), pd.Timestamp(win[1])
    assert s >= pd.Timestamp(TRAIN[0]) and e <= TRAIN_END, "sweep refuses to look beyond the Train window"
    inwin, last = m.window(win)
    split = int(np.searchsorted(m.dates.values, np.datetime64("2016-01-01")))
    out = _sweep(m.op, m.cl, m.pc, m.rsi, m.sx, m.st, inwin, last, mode, bool(dip_trend), float(comm), split, 0.01, 2,
                 RSI_ENTRIES.astype(np.float64), MAX_HOLDS.astype(np.int64))
    return derive(out)


def trend_only(m, comm=0.001, win=TRAIN):
    rows = []
    for tl in TREND_LENS:
        mt = run_one(m, win, MODE_TREND, 2, 20, 5, int(tl), 10, True, comm=comm)
        mt["trend_len"] = int(tl)
        rows.append(mt)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys, os
    OUTD = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt\sweeps"
    os.makedirs(OUTD, exist_ok=True)
    markets = {"TQQQ": ("TQQQ", "QQQ", "tv"), "QQQ": ("QQQ", "QQQ", None), "SPXL": ("SPXL", "SPY", None), "SPY": ("SPY", "SPY", None)}
    t00 = time.time()
    for name, (tr, px, tsrc) in markets.items():
        m = Market(tr, px, src="yf", traded_src=tsrc)
        for key, mode, dt in (("dipT", MODE_DIP, True), ("dipF", MODE_DIP, False), ("td", MODE_TD, True)):
            t0 = time.time()
            r = sweep_market(m, mode, dt)
            np.savez_compressed(f"{OUTD}\{name}_{key}_train.npz", **r)
            print(f"{name:5s} {key}: {r['sharpe'].size:,} configs in {time.time()-t0:5.1f}s  best Sharpe {np.nanmax(r['sharpe']):.3f}  median {np.nanmedian(r['sharpe']):.3f}", flush=True)
        trend_only(m).to_csv(f"{OUTD}\{name}_trend_train.csv", index=False)
    print(f"done in {time.time()-t00:.0f}s")
