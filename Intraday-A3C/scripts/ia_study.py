"""
Product screen for the 'intraday session-convergence' strategy (Demir et al. 2022 translated to US cash equities).

Paper -> equities:
  hourly contract with a hard session end        -> one regular trading day per product
  balancing price (volatile, known afterwards)   -> the OFFICIAL CLOSING AUCTION price (an MOC order, entered by 15:50 ET, settles at it, no spread)
  aggressor agent {Buy, Sell, Hold} on best quote -> marketable orders at the touch every 5 minutes (retail latency; no queue priority)
  v_max = 3 MW, q_high = 40 MW                    -> +-3 lots, 40 lots traded a day
  hindsight labels (eq. 6) for features / cloning -> y = (official close - price now) / price now  (what a Buy would earn if settled at the close)
  trading cost 0.232 EUR / MW                     -> half the quoted spread + a small slippage allowance, measured per product

Surrogate for the A3C agent: a walk-forward regressor of y from the paper's kind of state (distance to VWAP / open / previous close, lags, time to the deadline, volume pace, range position)
and a policy that holds  clip(floor(edge / (k * cost)), -3, +3)  lots. If this surrogate finds no edge net of cost, an RL agent on the same information will not either (it can only overfit).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ia_data as D

NB = 78                      # 5-minute bars in a regular day
NDEC = 72                    # decisions at the end of bars 0..71 = 9:35 .. 15:30; what is held after the last one is exited at the official close (MOC entered between 15:30 and 15:50)
VMAX, QHIGH = 3, 40
SLIP_BP = 0.25               # extra per-lot cost on top of half the quoted spread (latency, price drift between decision and fill)
FEATS = ["tod", "vwap_dist", "from_open", "from_prev", "gap", "ret1", "ret3", "ret6", "ret12", "rng_pos", "range_bp", "rvol12", "volpace", "r30", "prev_ret", "dow"]


def build_days(sym: str, bars: pd.DataFrame | None = None, daily: pd.DataFrame | None = None) -> dict:
    """Per-day arrays for one product: price at each decision, features, target, next-bar returns, settlement.
    `bars` / `daily` replace the cached SIP bars / official closes (feed-fidelity tests, the live runner's top-up)."""
    b = D.load_bars(sym) if bars is None else bars
    dc = D.load_daily() if daily is None else daily
    dc = dc[dc.symbol == sym].set_index("date").close
    b = b.assign(date=b.index.normalize())
    g = b.groupby("date")
    cnt = g.size()
    days = [d for d in cnt.index if cnt[d] == NB and d in dc.index]
    days = [d for d in days if d > dc.index[0]]
    P = np.full((len(days), NB), np.nan); V = P.copy(); VW = P.copy(); H = P.copy(); L = P.copy(); O0 = np.full(len(days), np.nan)
    for i, d in enumerate(days):
        x = g.get_group(d)
        P[i], V[i], VW[i], H[i], L[i], O0[i] = x.close.values, x.volume.values, x.vwap.values, x.high.values, x.low.values, x.open.values[0]
    C = dc.reindex(days).values
    prevC = dc.shift(1).reindex(days).values
    prev2C = dc.shift(2).reindex(days).values
    ok = np.isfinite(P).all(axis=1) & np.isfinite(C) & np.isfinite(prevC) & (V.sum(axis=1) > 0)
    keep = np.where(ok)[0]
    days = [days[i] for i in keep]
    P, V, VW, H, L, O0, C, prevC, prev2C = (a[keep] for a in (P, V, VW, H, L, O0, C, prevC, prev2C))
    cumv = np.cumsum(V, axis=1)
    vwap = np.cumsum(VW * V, axis=1) / np.maximum(cumv, 1)
    hi = np.maximum.accumulate(H, axis=1); lo = np.minimum.accumulate(L, axis=1)
    pace_ref = pd.DataFrame(cumv).rolling(20, min_periods=5).mean().shift(1).to_numpy()
    j = np.arange(NB)[None, :]
    def lag(k):
        out = np.zeros_like(P); out[:, k:] = (P[:, k:] / P[:, :-k] - 1) * 1e4; return out
    r1 = np.zeros_like(P); r1[:, 1:] = (P[:, 1:] / P[:, :-1] - 1) * 1e4
    rv = pd.DataFrame(r1.T).rolling(12, min_periods=3).std().T.to_numpy()
    feats = {
        "tod": np.broadcast_to(j, P.shape).astype(float),
        "vwap_dist": (P - vwap) / P * 1e4, "from_open": (P / O0[:, None] - 1) * 1e4, "from_prev": (P / prevC[:, None] - 1) * 1e4, "gap": np.broadcast_to(((O0 / prevC - 1) * 1e4)[:, None], P.shape),
        "ret1": r1, "ret3": lag(3), "ret6": lag(6), "ret12": lag(12), "rng_pos": (P - lo) / np.maximum(hi - lo, 1e-9), "range_bp": (hi - lo) / P * 1e4, "rvol12": np.nan_to_num(rv),
        "volpace": np.nan_to_num(cumv / pace_ref, nan=1.0, posinf=1.0), "r30": (P[:, [min(5, k) for k in range(NB)]] / O0[:, None] - 1) * 1e4,
        "prev_ret": np.broadcast_to(((prevC / prev2C - 1) * 1e4)[:, None], P.shape), "dow": np.broadcast_to(np.array([d.dayofweek for d in days])[:, None], P.shape).astype(float),
    }
    X = np.stack([feats[f][:, :NDEC] for f in FEATS], axis=2)                          # (days, decisions, features)
    y = (C[:, None] / P[:, :NDEC] - 1) * 1e4                                           # return to the official close if bought now
    nxt = np.zeros((len(days), NDEC))
    nxt[:, :-1] = (P[:, 1:NDEC] / P[:, :NDEC - 1] - 1) * 1e4                           # bar-to-bar return while the position is held
    nxt[:, -1] = (C / P[:, NDEC - 1] - 1) * 1e4                                        # last decision (15:30): held to the official close
    return dict(days=pd.DatetimeIndex(days), X=np.nan_to_num(X), y=y, nxt=nxt, P=P[:, :NDEC], vwapdist=feats["vwap_dist"][:, :NDEC], cumv=cumv)


def policy_pnl(edge: np.ndarray, nxt: np.ndarray, cost_bp: float, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """edge (days, decisions) in bp -> lots held after each decision. Returns (daily net bp, daily gross bp, daily lots traded)."""
    theta = k * cost_bp
    tgt = np.clip(np.sign(edge) * np.floor(np.abs(edge) / max(theta, 1e-9)), -VMAX, VMAX)
    nd = tgt.shape[0]
    pos = np.zeros_like(tgt)
    traded = np.zeros(nd)
    cur = np.zeros(nd)
    for j in range(tgt.shape[1]):
        want = tgt[:, j]
        step = want - cur
        room = np.maximum(QHIGH - traded, 0)
        step = np.sign(step) * np.minimum(np.abs(step), room)
        cur = cur + step
        traded += np.abs(step)
        pos[:, j] = cur
    gross = (pos * nxt).sum(axis=1)
    prev = np.concatenate([np.zeros((nd, 1)), pos[:, :-1]], axis=1)
    lots = np.abs(pos - prev).sum(axis=1)
    return gross - lots * cost_bp, gross, lots


class RidgeModel:
    """Standardised ridge; a class (not a closure) so a fitted model can be pickled for the live runner."""

    def __init__(self, Xtr, ytr):
        from sklearn.linear_model import Ridge
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        self.m = Ridge(alpha=1000.0).fit((Xtr - self.mu) / self.sd, ytr)

    def __call__(self, X):
        return self.m.predict((X - self.mu) / self.sd)


def fit_model(Xtr, ytr, kind="ridge"):
    """Fitted predictor (callable on a feature matrix)."""
    if kind == "ridge":
        return RidgeModel(Xtr, ytr)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=800, l2_regularization=10.0, random_state=0).fit(Xtr, ytr).predict


def fit_predict(Xtr, ytr, Xte, kind="ridge"):
    return fit_model(Xtr, ytr, kind)(Xte)


def walk_forward(dd: dict, kind: str, first_test_year: int = 2019, stride: int = 2) -> np.ndarray:
    """Expanding-window, one test year at a time: edge predictions (days, decisions) for every test day, NaN before the first test year."""
    years = dd["days"].year.values
    edge = np.full(dd["y"].shape, np.nan)
    nd, nt, nf = dd["X"].shape
    for Y in range(first_test_year, years.max() + 1):
        tr, te = np.where(years < Y)[0], np.where(years == Y)[0]
        if len(tr) < 400 or not len(te):
            continue
        Xtr = dd["X"][tr][:, ::stride].reshape(-1, nf); ytr = dd["y"][tr][:, ::stride].reshape(-1)
        Xte = dd["X"][te].reshape(-1, nf)
        edge[te] = fit_predict(Xtr, ytr, Xte, kind).reshape(len(te), nt)
    return edge


def stats(net: np.ndarray, gross: np.ndarray, lots: np.ndarray, days: pd.DatetimeIndex) -> dict:
    m = np.isfinite(net)
    n_, g_, l_ = net[m], gross[m], lots[m]
    if len(n_) < 50 or n_.std() == 0:
        return dict(days=int(m.sum()))
    sd = n_.std(ddof=1)
    return dict(days=int(m.sum()), net_bp_day=n_.mean(), gross_bp_day=g_.mean(), lots_day=l_.mean(), sharpe=n_.mean() / sd * np.sqrt(252), t=n_.mean() / sd * np.sqrt(len(n_)),
                gross_sharpe=g_.mean() / g_.std(ddof=1) * np.sqrt(252), pt_net_bp=n_.sum() / max(l_.sum(), 1e-9), pt_gross_bp=g_.sum() / max(l_.sum(), 1e-9), hit=(n_ > 0).mean(),
                ann_ret_pct=n_.mean() * 252 / 3 / 100)


def run_symbol(args):
    sym, half_spread_bp = args
    dd = build_days(sym)
    cost = half_spread_bp + SLIP_BP
    out = []
    for kind in ("ridge", "gbm"):
        edge = walk_forward(dd, kind)
        years = dd["days"].year.values
        te = years >= 2019
        # information content: correlation of the edge with the realised return to the close, by time of day buckets
        yy, ee = dd["y"][te].ravel(), edge[te].ravel()
        ic = float(np.corrcoef(ee, yy)[0, 1])
        for k in (1.0, 2.0, 4.0):
            net, gross, lots = policy_pnl(edge, dd["nxt"], cost, k)
            for label, mask in (("all 2019-26", te), ("2019-22", (years >= 2019) & (years <= 2022)), ("2023-26", years >= 2023)):
                s = stats(np.where(mask, net, np.nan), np.where(mask, gross, np.nan), lots, dd["days"])
                out.append(dict(symbol=sym, model=kind, k=k, period=label, ic=ic, cost_bp=cost, **s))
    # the paper's VWAP benchmark: buy when below the session VWAP, sell when above (+-1 lot)
    for k in (1.0,):
        edge = -dd["vwapdist"] * 1.0
        net, gross, lots = policy_pnl(edge, dd["nxt"], cost, 1e-9 + 5.0 * cost)         # theta = 5 cost: one lot per 5 costs of deviation
        years = dd["days"].year.values
        out.append(dict(symbol=sym, model="vwap_rule", k=5.0, period="all 2016-26", ic=np.nan, cost_bp=cost, **stats(net, gross, lots, dd["days"])))
    return out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    sp = pd.read_csv(os.path.join(D.REPORTS, "spreads.csv")).set_index("symbol")
    syms = [s for s in D.ALL if os.path.exists(D.bars_path(s))]
    jobs = [(s, float(sp.spread_bp.get(s, 2.0)) / 2.0) for s in syms]
    print("products:", len(jobs), flush=True)
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(run_symbol, jobs):
            rows += r
            print(r[0]["symbol"], "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(D.REPORTS, "product_screen.csv"), index=False)
    print("saved")
