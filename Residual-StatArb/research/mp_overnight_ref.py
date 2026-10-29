"""The rule-based overnight-momentum leaders (mp_overnight.py), rebuilt for side-by-side comparison with the prediction models and the combinations."""
import numpy as np
import pandas as pd

from mp_lib import Ctx, overnight, topk


def signals(cx: Ctx, lead: int = 1) -> tuple:
    """(tradeable mask, {name: T x N signal}) at the decision time ``lead`` bars before the close."""
    C = cx.C
    prevC = cx.lag(C)
    S = cx.snap(lead)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = S / prevC - 1
        res = d - np.nan_to_num(cx.beta(60), nan=1.0) * d[:, [cx.spy]]
        cv = cx.cumvol(lead)
        vb = pd.DataFrame(cv).rolling(20, min_periods=10).mean().shift(1).to_numpy()
        vs = np.where(d > 0, cv / vb, np.nan)
    return np.isfinite(S) & np.isfinite(prevC), {"d45": d, "res45": res, "vsurp": vs}


def spy_trend(cx: Ctx) -> np.ndarray:
    C = cx.C
    sma200 = pd.Series(C[:, cx.spy]).rolling(200, min_periods=150).mean().to_numpy()
    return (cx.lag(C[:, cx.spy]) > cx.lag(sma200))[:, None]


def reference(cx: Ctx) -> dict:
    live, s = signals(cx)
    tr = spy_trend(cx)
    out = {}
    for un, um, sn, k in (("top150+etf", cx.top(150) | cx.etf(), "d45", 20), ("top150+etf", cx.top(150) | cx.etf(), "res45", 20),
                          ("top150", cx.top(150), "vsurp", 20), ("top150+etf", cx.top(150) | cx.etf(), "d45", 10)):
        out[f"REF {un} {sn} top{k} spy_trend"] = overnight(cx, topk(s[sn], um & live & tr, k), k=k)
    return out
