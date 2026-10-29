"""
Prediction family (P): machine-learned forecasts of the next return segment, used long-only and unlevered.

At 15:45 every name of the top-150 + 59 ETFs gets ~35 features known at that time (today's move to 15:45, last hour, volume surprise, IBS, gap, overnight / intraday
return persistence, momentum, volatility, beta, 52-week high, size, ETF flag, and market context: SPY's move, volatility, trend, breadth, dispersion, weekday,
turn of month). Stock features are ranked within the day.
Targets:  on = close -> next open (the overnight sleeve),  cc = close -> next close (a one-day hold).
Models:   ridge (rank features) and gradient-boosted trees (sklearn HistGradientBoosting, fixed hyper-parameters chosen a priori).
Walk-forward: for each year Y = 2018 .. 2026 the model is trained on every row before Y (expanding window) and predicts Y. In-sample = 2018-2022, out-of-sample = 2023-2026.
Portfolios: the k best predictions (k = 5 / 10 / 20), 1/k each, optionally only when SPY is above its 200-day mean or the model's own top-k forecast clears the costs.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from mp_lib import OUT, Book, Ctx, log, overnight, rebalance, topk

cx = Ctx()
C, O, H, L, V, T, N = cx.C, cx.O, cx.H, cx.L, cx.V, cx.T, cx.N
prevC = cx.lag(C)
cl, op, vo, nb = cx.p.cube()
S45, S1530, S1445 = cx.snap(1), cx.snap(2), cx.snap(5)
U = (cx.top(150) | cx.etf()) & np.isfinite(S45) & np.isfinite(prevC)


def roll_mean(x, n, mp=None):
    return pd.DataFrame(x).rolling(n, min_periods=mp or max(2, int(n * 0.75))).mean().to_numpy()


def build():
    f = os.path.join(OUT, "mp_cache", "features_1545.parquet")
    if os.path.exists(f):
        return pd.read_parquet(f)
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = np.where(np.isfinite(cx.rc), cx.rc, np.nan)
        ron = np.where(np.isfinite(cx.ron), cx.ron, np.nan)
        rid = np.where(np.isfinite(cx.rid), cx.rid, np.nan)
        hi45 = np.full((T, N), np.nan)
        lo45 = np.full((T, N), np.nan)
        for t in range(T):
            k = int(nb[t]) - 1
            seg = np.concatenate([op[t][:, None], cl[t, :, :k]], axis=1)
            hi45[t] = np.nanmax(seg, 1)
            lo45[t] = np.nanmin(seg, 1)
        d45 = S45 / prevC - 1
        beta = cx.beta(60)
        spy45 = d45[:, [cx.spy]]
        cv = cx.cumvol(1)
        feats = {
            "d45": d45, "res45": d45 - np.nan_to_num(beta, nan=1.0) * spy45, "id45": S45 / O - 1, "gap": O / prevC - 1, "hour": S45 / S1445 - 1,
            "last15": S45 / S1530 - 1, "ibs45": (S45 - lo45) / (hi45 - lo45), "vsurp": cv / pd.DataFrame(cv).rolling(20, min_periods=10).mean().shift(1).to_numpy(),
            "r1": cx.lag(rc), "r5": prevC / cx.lag(C, 6) - 1, "r21": prevC / cx.lag(C, 22) - 1, "r63": prevC / cx.lag(C, 64) - 1, "mom12_1": cx.lag(C, 22) / cx.lag(C, 253) - 1,
            "on5": roll_mean(ron, 5), "on20": roll_mean(ron, 20), "on60": roll_mean(ron, 60), "id5": cx.lag(roll_mean(rid, 5)), "id20": cx.lag(roll_mean(rid, 20)),
            "id60": cx.lag(roll_mean(rid, 60)), "rv20": cx.lag(pd.DataFrame(rc).rolling(20, min_periods=15).std().to_numpy()),
            "rv60": cx.lag(pd.DataFrame(rc).rolling(60, min_periods=45).std().to_numpy()), "range20": cx.lag(roll_mean((H - L) / C, 20)), "beta60": beta,
            "hi52": prevC / cx.lag(pd.DataFrame(C).rolling(252, min_periods=200).max().to_numpy()), "c_sma20": prevC / cx.lag(roll_mean(C, 20)) - 1,
            "c_sma200": prevC / cx.lag(roll_mean(C, 200, 150)) - 1, "dvol": cx.lag(np.log(pd.DataFrame(C * V).rolling(63, min_periods=40).median().to_numpy())),
        }
        mkt = {
            "m_spy45": d45[:, cx.spy], "m_spyrv20": feats["rv20"][:, cx.spy], "m_spytrend": feats["c_sma200"][:, cx.spy], "m_spyon20": feats["on20"][:, cx.spy],
            "m_breadth": np.nanmean(np.where(U, d45 > 0, np.nan), 1), "m_disp": np.nanstd(np.where(U, d45, np.nan), 1), "m_dow": cx.dates.dayofweek.to_numpy().astype(float),
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


df = build()
log(f"features: {len(df):,} rows, {df.t.nunique()} sessions")
STOCK = [c for c in df.columns if c.startswith("rk_")]
MKT = [c for c in df.columns if c.startswith("m_")] + ["etf"]
RAW = [c[3:] for c in STOCK]
YEARS = list(range(2018, 2027))


def walk_forward(target, model, cols, rank_target=True):
    key = f"{target}_{model}_{'rank' if rank_target else 'raw'}"
    f = os.path.join(OUT, "mp_cache", f"pred_{key}.npy")
    if os.path.exists(f):
        return np.load(f)
    pred = np.full(len(df), np.nan)
    y = df[target]
    if rank_target:
        y = df.groupby("t")[target].rank(pct=True) - 0.5
    else:
        y = y.clip(y.quantile(0.001), y.quantile(0.999))
    yr = df.date.dt.year.to_numpy()
    for Y in YEARS:
        tr = (yr < Y) & y.notna().to_numpy()
        last_t = df.t[yr < Y].max()
        tr &= (df.t < last_t).to_numpy()                                                  # the last training session's target lies in year Y
        te = yr == Y
        if not te.any():
            continue
        X = df[cols].to_numpy(np.float32)
        if model == "ridge":
            mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-9
            Xf = np.nan_to_num((X - mu) / sd, nan=0.0)                                   # standardised on the training rows; missing = the training mean
            mdl = Ridge(alpha=10.0).fit(Xf[tr], y[tr])
            pred[te] = mdl.predict(Xf[te])
        else:
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0,
                                                early_stopping=False, random_state=0).fit(X[tr], y[tr])
            pred[te] = mdl.predict(X[te])
        log(f"  {key} {Y}: trained on {tr.sum():,} rows")
    np.save(f, pred)
    return pred


bk = Book("mp_predict", cx)
bk.add(cx.spy_ret(), "SPY buy & hold", fam="bench")
sma200 = roll_mean(C, 200, 150)
spy_tr = (cx.lag(C[:, cx.spy]) > cx.lag(sma200[:, cx.spy]))[:, None]
t1 = int(np.searchsorted(cx.dates, pd.Timestamp("2018-01-01")))


def to_matrix(v):
    m = np.full((T, N), np.nan)
    m[df.t.to_numpy(), df.j.to_numpy()] = v
    return m


ic_rows = []
for target in ("y_on", "y_cc"):
    for model, cols in (("ridge", STOCK + MKT), ("hgb", STOCK + MKT), ("hgb_raw", RAW + MKT)):
        mname = "hgb" if model.startswith("hgb") else model
        for rank_target in (True, False):
            if model == "ridge" and not rank_target:
                continue
            pr = walk_forward(target, mname if model != "hgb_raw" else "hgbraw", cols, rank_target)
            P = to_matrix(pr)
            P[:t1] = np.nan
            ok = np.isfinite(pr) & df[target].notna().to_numpy()
            ic = pd.DataFrame({"t": df.t[ok], "p": pr[ok], "y": df[target][ok]}).groupby("t").apply(lambda g: g.p.corr(g.y, method="spearman"))
            dts = cx.dates[ic.index]
            ic_rows.append({"target": target, "model": model, "rank_target": rank_target, "ic_is": ic[dts < "2023-01-01"].mean(), "ic_oos": ic[dts >= "2023-01-01"].mean(),
                            "icir_is": ic[dts < "2023-01-01"].mean() / ic[dts < "2023-01-01"].std() * np.sqrt(252), "icir_oos": ic[dts >= "2023-01-01"].mean() / ic[dts >= "2023-01-01"].std() * np.sqrt(252)})
            tag = f"P {target[2:]} {model} {'rank' if rank_target else 'raw'}-target"
            for k in (5, 10, 20):
                sel = topk(P, U, k)
                gates = {"none": True, "spy_trend": spy_tr}
                if not rank_target:
                    mean_top = np.nanmean(np.where(sel, P, np.nan), 1)
                    gates["forecast>cost"] = (mean_top > 2e-4)[:, None]
                for gn, g in gates.items():
                    s = sel & g
                    if target == "y_on":
                        r = overnight(cx, s, k=k)
                    else:
                        daily = np.ones(T, bool)
                        r = rebalance(cx, s / k, daily)
                    r = r[r.index >= "2018-01-01"]
                    bk.add(r, f"{tag} top{k} gate={gn}", fam="P", target=target, model=model, k=k, gate=gn)
            log(f"{tag}: IC is {ic_rows[-1]['ic_is']:.4f} oos {ic_rows[-1]['ic_oos']:.4f}")
pd.DataFrame(ic_rows).to_csv(os.path.join(OUT, "mp_predict_ic.csv"), index=False)
print(pd.DataFrame(ic_rows).round(4).to_string(index=False))
# rule-based reference on the same 2018+ window
from mp_overnight_ref import reference  # noqa: E402
for lab, r in reference(cx).items():
    bk.add(r[r.index >= "2018-01-01"], lab, fam="ref")
bk.save()
bk.show(n=40, cols=["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all", "beta"])
