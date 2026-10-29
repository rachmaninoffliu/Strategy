"""
Does machine-learned prediction improve the time-shared book? The walk-forward overnight forecasts of mp_predict.py (2018 on) replace or blend with the
rule-based "today's move" ranking in the overnight leg; the intraday leg stays the QQQ noise area. Everything is compared on the same 2018-01 -> 2026-10 window.
"""
import os

import numpy as np
import pandas as pd

from mp_lib import OUT, Book, Ctx, overnight, topk
from mp_overnight_ref import signals, spy_trend
from mp_sleeves import noise_area

cx = Ctx()
bk = Book("mp_predict_combo", cx)
W0 = "2018-01-01"
live, S = signals(cx, 1)
tr = spy_trend(cx)
U = (cx.top(150) | cx.etf()) & live
f = pd.read_parquet(os.path.join(OUT, "mp_cache", "features_1545.parquet"), columns=["t", "j"])
ri = noise_area(cx, "QQQ", every=2, lookback=14)


def mat(v):
    m = np.full((cx.T, cx.N), np.nan)
    m[f.t.to_numpy(), f.j.to_numpy()] = v
    return m


def rk(x):
    return pd.DataFrame(np.where(U & np.isfinite(x), x, np.nan)).rank(axis=1, pct=True).to_numpy()


def add(lab, sig, k=20):
    ro = overnight(cx, topk(sig, U & tr, k), k=k)
    for nm, r in ((f"{lab} | overnight only", ro), (f"{lab} + QQQ noise area", (1 + ro) * (1 + ri) - 1)):
        bk.add(r[r.index >= W0], nm)


bk.add(cx.spy_ret()[cx.spy_ret().index >= W0], "SPY buy & hold")
add("RULE today's move top20", S["d45"])
for key in ("y_on_ridge_rank", "y_on_hgb_rank", "y_on_hgbraw_rank", "y_on_hgb_raw"):
    fn = os.path.join(OUT, "mp_cache", f"pred_{key}.npy")
    if not os.path.exists(fn):
        continue
    P = mat(np.load(fn))
    add(f"ML {key} top20", P)
    add(f"ML {key} + rule (rank average) top20", rk(P) + rk(S["d45"]))
    add(f"rule top20 among ML top half ({key})", np.where(rk(P) > 0.5, S["d45"], np.nan))
df = bk.save()
with pd.option_context("display.width", 260, "display.max_colwidth", 100):
    print(df[["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all", "beta"]].round(3).to_string(index=False))
