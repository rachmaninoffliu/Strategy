"""
Fill-price audit of the product decision. The research fills every order at the CLOSE of the 5-minute bar that produced the decision (a last trade, up to seconds old). A real marketable order sent a
few seconds after the bar end fills near the OPEN of the next bar. For a position change at decision j, with the signal-bar close C_j and the next-bar open O_j+1:

        fill price  F = C_j + alpha x (O_j+1 - C_j)          alpha = 0: the research assumption, alpha = 1: the next open

The old position earns C_j -> F, the new position earns F -> the close of the next bar (the official close for the last decision of the day); positions that do not change earn close-to-close.
Same signals, same costs (half the sampled quoted spread + 0.25 bp), no trading before 10:00.   Output: reports/fillmodel.log
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ia_audit as A
import ia_data as D
import ia_portfolio as P
import ia_study as S

SIG = {"TSLA": 2.46, "SMH": 1.18, "NVDA": 2.00}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def open_close_arrays(sym: str, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    b = D.load_bars(sym)
    g = {d: x for d, x in b.groupby(b.index.normalize())}
    return np.array([g[d].open.to_numpy() for d in days]), np.array([g[d].close.to_numpy() for d in days])


def pnl_alpha(dec: np.ndarray, Op: np.ndarray, Cl: np.ndarray, C: np.ndarray, alpha: float, cost_bp: float) -> np.ndarray:
    """Daily net bp (per lot) of a position path dec (days x 72 decisions) with fills at C_j + alpha (O_j+1 - C_j)."""
    n = dec.shape[0]
    hold = np.zeros((n, 73))
    hold[:, 1:] = dec                                          # hold[b] = position decided at the close of bar b-1, held from the fill in bar b
    gross = np.zeros(n)
    for b in range(1, 73):
        prev, cur = hold[:, b - 1], hold[:, b]
        base = Cl[:, b - 1]
        F = base + alpha * (Op[:, b] - base)
        end = Cl[:, b] if b <= 71 else C                       # the last decision (15:30) is held to the official close
        gross += np.where(prev == cur, cur * (end / base - 1.0), prev * (F / base - 1.0) + cur * (end / F - 1.0))
    lots = np.abs(dec - np.concatenate([np.zeros((n, 1)), dec[:, :-1]], axis=1)).sum(axis=1)
    return gross * 1e4 - lots * cost_bp


def main():
    sp = pd.read_csv(os.path.join(D.REPORTS, "spreads.csv")).set_index("symbol")
    series = {a: {} for a in ALPHAS}
    print("Sharpe 2019-26, ens k=3, no trading before 10:00, cost = half quoted spread + 0.25 bp per lot")
    for sym in SIG:
        dd, E = P.edges_for(sym)
        m = dd["days"].year.values >= 2019
        c = float(sp.spread_bp[sym]) / 2 + S.SLIP_BP
        e2 = np.nan_to_num(E["ens"]); e2[:, :5] = 0.0
        dec = A.path(e2, c, 3.0)
        Op, Cl = open_close_arrays(sym, dd["days"])
        C = dd["P"][:, 71] * (1 + dd["y"][:, 71] / 1e4)       # official close, from the research arrays
        # sanity: alpha = 0 must reproduce the research accounting exactly
        ref = A.net_j(dec, dd["nxt"], np.full(72, c))
        chk = pnl_alpha(dec, Op, Cl, C, 0.0, c)
        assert np.allclose(chk, ref, atol=1e-6), np.abs(chk - ref).max()
        row = {}
        for a in ALPHAS:
            net = pnl_alpha(dec, Op, Cl, C, a, c)
            series[a][sym] = pd.Series(net, index=dd["days"])[m]
            row[a] = A.sh(net[m])
        prev = np.concatenate([np.zeros((dec.shape[0], 1)), dec[:, :-1]], axis=1)
        d = dec - prev                                        # lots bought (+) / sold (-) at each decision
        gapb = (Op[:, 1:73] / Cl[:, 0:72] - 1.0) * 1e4         # close of the signal bar -> open of the next bar
        mm = (d != 0) & m[:, None]
        worse = (np.sign(d) * gapb)[mm]                        # positive = we paid more / received less than the signal-bar close
        print(f"{sym}: " + "  ".join(f"a={a:g}: {row[a]:5.2f}" for a in ALPHAS) + f" | move from the signal close to the next open, against us: {worse.mean():+.2f} bp (n={mm.sum():,})")
    w = pd.Series({s: 1 / SIG[s] for s in SIG})
    print("basket TSLA+SMH+NVDA, inverse-vol weights: " + "  ".join(f"a={a:g}: {A.sh(((pd.DataFrame(series[a]) * w).sum(axis=1) / w.sum()).values):5.2f}" for a in ALPHAS))


if __name__ == "__main__":
    main()
