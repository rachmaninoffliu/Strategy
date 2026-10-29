"""
The prediction family's feature table and walk-forward trainer as functions (mp_predict.py ran them at 15:45; mp_predict_check.py also needs 15:30).

build(cx, lead)  : one row per (session, name) of the top-150 + 59 ETFs with ~35 features known ``lead`` bars before the close, plus the targets
                   y_on (close -> next official open) and y_cc (close -> next close). Stock features are also ranked within the session (rk_*).
walk_forward(...): for each year Y = 2018 .. 2026, fit on every earlier row whose target is known before Y, predict Y. Fixed hyper-parameters.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from mp_lib import OUT, Ctx, log

YEARS = list(range(2018, 2027))
CACHE = os.path.join(OUT, "mp_cache")


def _roll_mean(x, n, mp=None):
    return pd.DataFrame(x).rolling(n, min_periods=mp or max(2, int(n * 0.75))).mean().to_numpy()


def _tag(lead):
    return {1: "1545", 2: "1530"}.get(lead, f"lead{lead}")


def build(cx: Ctx, lead: int = 1) -> pd.DataFrame:
    f = os.path.join(CACHE, f"features_{_tag(lead)}.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    C, O, H, L, V, T, N = cx.C, cx.O, cx.H, cx.L, cx.V, cx.T, cx.N
    prevC = cx.lag(C)
    cl, op, vo, nb = cx.p.cube()
    S, S15, S60 = cx.snap(lead), cx.snap(lead + 1), cx.snap(lead + 4)
    U = (cx.top(150) | cx.etf()) & np.isfinite(S) & np.isfinite(prevC)
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = np.where(np.isfinite(cx.rc), cx.rc, np.nan)
        ron = np.where(np.isfinite(cx.ron), cx.ron, np.nan)
        rid = np.where(np.isfinite(cx.rid), cx.rid, np.nan)
        hi, lo = np.full((T, N), np.nan), np.full((T, N), np.nan)
        for t in range(T):
            k = int(nb[t]) - lead
            seg = np.concatenate([op[t][:, None], cl[t, :, :k]], axis=1)
            hi[t] = np.nanmax(seg, 1)
            lo[t] = np.nanmin(seg, 1)
        d = S / prevC - 1
        beta = cx.beta(60)
        cv = cx.cumvol(lead)
        feats = {
            "d45": d, "res45": d - np.nan_to_num(beta, nan=1.0) * d[:, [cx.spy]], "id45": S / O - 1, "gap": O / prevC - 1, "hour": S / S60 - 1,
            "last15": S / S15 - 1, "ibs45": (S - lo) / (hi - lo), "vsurp": cv / pd.DataFrame(cv).rolling(20, min_periods=10).mean().shift(1).to_numpy(),
            "r1": cx.lag(rc), "r5": prevC / cx.lag(C, 6) - 1, "r21": prevC / cx.lag(C, 22) - 1, "r63": prevC / cx.lag(C, 64) - 1, "mom12_1": cx.lag(C, 22) / cx.lag(C, 253) - 1,
            "on5": _roll_mean(ron, 5), "on20": _roll_mean(ron, 20), "on60": _roll_mean(ron, 60), "id5": cx.lag(_roll_mean(rid, 5)), "id20": cx.lag(_roll_mean(rid, 20)),
            "id60": cx.lag(_roll_mean(rid, 60)), "rv20": cx.lag(pd.DataFrame(rc).rolling(20, min_periods=15).std().to_numpy()),
            "rv60": cx.lag(pd.DataFrame(rc).rolling(60, min_periods=45).std().to_numpy()), "range20": cx.lag(_roll_mean((H - L) / C, 20)), "beta60": beta,
            "hi52": prevC / cx.lag(pd.DataFrame(C).rolling(252, min_periods=200).max().to_numpy()), "c_sma20": prevC / cx.lag(_roll_mean(C, 20)) - 1,
            "c_sma200": prevC / cx.lag(_roll_mean(C, 200, 150)) - 1, "dvol": cx.lag(np.log(pd.DataFrame(C * V).rolling(63, min_periods=40).median().to_numpy())),
        }
        mkt = {
            "m_spy45": d[:, cx.spy], "m_spyrv20": feats["rv20"][:, cx.spy], "m_spytrend": feats["c_sma200"][:, cx.spy], "m_spyon20": feats["on20"][:, cx.spy],
            "m_breadth": np.nanmean(np.where(U, d > 0, np.nan), 1), "m_disp": np.nanstd(np.where(U, d, np.nan), 1), "m_dow": cx.dates.dayofweek.to_numpy().astype(float),
        }
        mon = cx.dates.to_period("M")
        pos_in_month = pd.Series(1, index=cx.dates).groupby(mon).cumsum().to_numpy()
        left_in_month = pd.Series(1, index=cx.dates)[::-1].groupby(mon[::-1]).cumsum()[::-1].to_numpy()
        mkt["m_tom"] = np.where(left_in_month <= 2, -left_in_month, pos_in_month).astype(float)
        y_on = cx.lead(O) / C - 1
        y_cc = cx.lead(C) / C - 1
    ti, ji = np.nonzero(U)
    df = pd.DataFrame({"t": ti, "j": ji})
    for k, v in feats.items():
        df[k] = v[ti, ji].astype(np.float32)
    for k, v in mkt.items():
        df[k] = v[ti].astype(np.float32)
    df["etf"] = cx.etf()[ti, ji].astype(np.float32)
    df["y_on"] = y_on[ti, ji]
    df["y_cc"] = y_cc[ti, ji]
    df["date"] = cx.dates[ti]
    stock = list(feats)
    df[[f"rk_{k}" for k in stock]] = df.groupby("t")[stock].rank(pct=True).astype(np.float32).to_numpy()
    df.to_parquet(f)
    return df


def columns(df: pd.DataFrame, kind: str) -> list:
    stock = [c for c in df.columns if c.startswith("rk_")]
    mkt = [c for c in df.columns if c.startswith("m_")] + ["etf"]
    return (stock if kind == "rank" else [c[3:] for c in stock]) + mkt


def walk_forward(df: pd.DataFrame, target: str, model: str, cols: list, rank_target: bool = True, key: str = None) -> np.ndarray:
    """Out-of-sample predictions for every row of 2018 on (NaN before). model: 'ridge' | 'hgb'. Cached under ``key``."""
    f = os.path.join(CACHE, f"pred_{key}.npy") if key else None
    if f and os.path.exists(f):
        return np.load(f)
    pred = np.full(len(df), np.nan)
    y = df.groupby("t")[target].rank(pct=True) - 0.5 if rank_target else df[target].clip(df[target].quantile(0.001), df[target].quantile(0.999))
    yr = df.date.dt.year.to_numpy()
    X = df[cols].to_numpy(np.float32)
    for Y in YEARS:
        te = yr == Y
        if not te.any():
            continue
        last_t = df.t[yr < Y].max()
        tr = (yr < Y) & y.notna().to_numpy() & (df.t < last_t).to_numpy()
        if model == "ridge":
            mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-9
            Xf = np.nan_to_num((X - mu) / sd, nan=0.0)
            pred[te] = Ridge(alpha=10.0).fit(Xf[tr], y[tr]).predict(Xf[te])
        else:
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0,
                                                early_stopping=False, random_state=0).fit(X[tr], y[tr])
            pred[te] = mdl.predict(X[te])
        log(f"  {key or model} {Y}: trained on {tr.sum():,} rows")
    if f:
        np.save(f, pred)
    return pred


def to_matrix(cx: Ctx, df: pd.DataFrame, v: np.ndarray) -> np.ndarray:
    m = np.full((cx.T, cx.N), np.nan)
    m[df.t.to_numpy(), df.j.to_numpy()] = v
    return m
