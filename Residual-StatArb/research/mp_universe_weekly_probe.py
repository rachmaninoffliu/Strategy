"""
QR's probe (2026-10-28): top-150 vs top-500 by 63-session median dollar volume, re-ranked monthly vs weekly (point-in-time: the list known at the last session of a
month / week is used for the next one; price >= $10, >= 126 sessions of history, no ETPs, one share class) + the 59 ETFs.
The signal is a PROXY - today's move to the CLOSE (look-ahead; only the top-150's names have 15-minute bars) - so only the differences between rows mean anything.
Rule only, top-20, SPY > 200-day mean, overnight leg alone, 1 bp per auction side (and 3 bp).
"""
import numpy as np
import pandas as pd

from mp_data import _daily_proxy
from mp_lib import Ctx, overnight, topk
from mp_overnight_ref import spy_trend
from mr_data import universe
from resarb import data as D

cx = Ctx()
tr = spy_trend(cx)
daily = _daily_proxy(cx.p)
n0 = len(daily.syms)


def weekly(n):
    px = pd.DataFrame(daily.close, index=daily.dates, columns=daily.syms)
    med = pd.DataFrame(daily.close * daily.volume, index=daily.dates, columns=daily.syms).rolling(63, min_periods=40).median()
    hist = px.notna().cumsum()
    names = pd.read_parquet(f"{D.STATARB_DATA}/statarb_assets.parquet").drop_duplicates("symbol").set_index("symbol").name
    bad = D.etp_symbols(names) | set(D.ETFS) | {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO", "UUP", "VIXY", "ITA"} | set(D.DUP_CLASS)
    wk = daily.dates.to_period("W-FRI")
    we = np.r_[np.where(wk[1:] != wk[:-1])[0], len(wk) - 1]
    out = np.zeros(daily.close.shape, bool)
    for k, i in enumerate(we):
        row = med.iloc[i]
        ok = row.notna() & (px.iloc[i] >= 10) & (hist.iloc[i] >= 126) & ~row.index.isin(list(bad))
        cols = [daily.col[s] for s in row[ok].sort_values(ascending=False).head(n).index]
        out[we[k] + 1:(we[k + 1] + 1 if k + 1 < len(we) else len(wk))][:, cols] = True
    return out


def pad(u):
    o = np.zeros((cx.T, cx.N), bool)
    o[:, :u.shape[1]] = u
    return o


U = {"top150 monthly": pad(universe(daily, 150)), "top150 weekly": pad(weekly(150)), "top500 monthly": pad(universe(daily, 500)), "top500 weekly": pad(weekly(500))}
etf = cx.etf()
proxy = cx.rc
ok = np.isfinite(proxy) & np.isfinite(cx.lag(cx.C))
adv = cx.lag(pd.DataFrame(cx.C * cx.V).rolling(20, min_periods=10).mean().to_numpy())


def stats(r):
    r = r.dropna()
    a, b = r[r.index < "2023-01-01"], r[r.index >= "2023-01-01"]
    sh = lambda x: x.mean() / x.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    return dict(sh=sh(r), is_=sh(a), oos=sh(b), cagr=eq.iloc[-1] ** (252 / len(r)) - 1, mdd=(eq / eq.cummax() - 1).min(), night_bp=r[r != 0].mean() * 1e4)


rows = []
for name, u in U.items():
    m = u[cx.t0:]
    churn = np.mean([(m[i] & ~m[i - 1]).sum() for i in range(1, len(m)) if (m[i] != m[i - 1]).any()])
    for cost in (1.0, 3.0):
        sel = topk(proxy, ok & (u | etf) & tr, 20)
        r = overnight(cx, sel, k=20, cost_in=cost, cost_out=cost)
        rows.append(dict(universe=name + " + ETFs", cost_bp=cost, **stats(r), picks_adv_musd=np.nanmedian(np.where(sel, adv, np.nan)[cx.t0:]) / 1e6,
                         names_in_per_rerank=churn, names_ever=int(m.any(0).sum())))
df = pd.DataFrame(rows)
with pd.option_context("display.width", 250):
    print(df.round({"sh": 2, "is_": 2, "oos": 2, "cagr": 3, "mdd": 3, "night_bp": 1, "picks_adv_musd": 0, "names_in_per_rerank": 1}).to_string(index=False))
