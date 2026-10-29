"""
Time-sharing the capital (PM idea): the overnight sleeve holds close -> open, the intraday sleeve holds only between the opening and the closing auction, so the
same equity can run both with gross <= 1x at every moment. Day return = (1 + overnight leg ending at today's open) x (1 + intraday leg) - 1.

Overnight candidates: the rule-based leaders (mp_overnight.py) and rank-blends of their signals.
Intraday candidates: the noise-area sleeve on QQQ / SPY / both / a few index ETFs, at the paper's settings and the neighbours.
"""
import itertools

import numpy as np
import pandas as pd

from mp_lib import Book, Ctx, log, overnight, topk
from mp_overnight_ref import signals, spy_trend
from mp_sleeves import noise_area

cx = Ctx()
bk = Book("mp_combo", cx)
bk.add(cx.spy_ret(), "SPY buy & hold", fam="bench")
live, S = signals(cx, 1)
tr = spy_trend(cx)
U150e = cx.top(150) | cx.etf()


def rk(x, m):
    return pd.DataFrame(np.where(m & np.isfinite(x), x, np.nan)).rank(axis=1, pct=True).to_numpy()


m = U150e & live
S["blend_d_v"] = rk(S["d45"], m) + rk(np.where(np.isfinite(S["vsurp"]), S["vsurp"], 0), m)
S["blend_d_r_v"] = rk(S["d45"], m) + rk(S["res45"], m) + rk(np.where(np.isfinite(S["vsurp"]), S["vsurp"], 0), m)
ON = {}
for sn in ("d45", "res45", "vsurp", "blend_d_v", "blend_d_r_v"):
    for k in (10, 20, 30):
        lab = f"ON top150+etf {sn} top{k} spy_trend"
        ON[lab] = overnight(cx, topk(S[sn], m & tr, k), k=k)
        bk.add(ON[lab], lab, fam="ON", sig=sn, k=k)
log(f"overnight: {len(ON)}")
ID = {}
for tks in (["QQQ"], ["SPY"], ["QQQ", "SPY"], ["QQQ", "SPY", "IWM", "DIA"], ["QQQ", "SMH", "XLK"]):
    for every, lb, mult in ((2, 14, 1.0), (1, 14, 1.0), (2, 20, 1.0), (2, 14, 1.25), (2, 10, 1.0)):
        lab = f"ID noise area {'+'.join(tks)} every {15 * every}m lb{lb} x{mult}"
        ID[lab] = noise_area(cx, tks, every=every, lookback=lb, mult=mult)
        bk.add(ID[lab], lab, fam="ID", n=len(tks), every=every, lb=lb, mult=mult)
log(f"intraday: {len(ID)}")
for (on, ro), (idl, ri) in itertools.product(ON.items(), ID.items()):
    r = (1 + ro) * (1 + ri) - 1
    bk.add(r, f"{on} || {idl}", fam="COMBO", on=on, id=idl, corr=float(np.corrcoef(ro, ri)[0, 1]))
df = bk.save()
log(f"{len(df)} rows")
with pd.option_context("display.width", 300, "display.max_rows", 400, "display.max_colwidth", 120):
    cols = ["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all", "beta"]
    print(df[df.fam == "ON"].sort_values("sh_is", ascending=False)[cols].round(3).head(15).to_string(index=False))
    print(df[df.fam == "ID"].sort_values("sh_is", ascending=False)[cols].round(3).head(15).to_string(index=False))
    print(df[df.fam == "COMBO"].sort_values("sh_is", ascending=False)[cols + ["corr"]].round(3).head(30).to_string(index=False))
    c = df[df.fam == "COMBO"]
    print(f"\ncombos: {len(c)}; Sharpe all > 2: {(c.sh_all > 2).sum()}, IS > 2: {(c.sh_is > 2).sum()}, OOS > 2: {(c.sh_oos > 2).sum()}, all three > 2: {((c.sh_all > 2) & (c.sh_is > 2) & (c.sh_oos > 2)).sum()}")
