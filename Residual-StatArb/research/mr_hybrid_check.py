"""
DA / QT: what the free 'sip+iex' live feed does to a 15:45 model. In live at 15:45:06 the delayed SIP has the bars up to the 15:00 bar; the 15:15 and 15:30 bars come from IEX
(IEX-only prints: a narrower high / low, a last price that can be seconds old, no bar at all when IEX had no trade). This rebuilds that view for every session since 2023 from the
SIP and IEX caches and runs the model on it, next to the true-SIP backtest: the share of identical trades and the Sharpe / CAGR it costs.

  python mr_hybrid_check.py [universe] [slots]
"""
import sys

import numpy as np
import pandas as pd

from mr_run import E, MRModel

from resarb import data as D  # noqa: E402
from resarb.mr.market import Market, decision_lead, session_closes  # noqa: E402

uni = sys.argv[1] if len(sys.argv) > 1 else "top50+etf"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
m = MRModel(name=f"{uni} k{k} @1545", universe=uni, minute=1545, entry_signal="ibs", entry_op="<", entry_value=0.2, trend_n=200, exit_signal="ibs", exit_op=">", exit_value=0.7,
            exit_sma_n=5, max_hold=10, slots=k, rank_by="signal")
START = "2023-01-03"
closes = session_closes("2015-06-01", "2026-12-31")


def hybrid(sym: str) -> pd.DataFrame:
    s = D.load_bars(sym, "sip")
    x = D.load_bars(sym, "iex")
    if not len(s):
        return s
    day = s.index.normalize()
    close = pd.DatetimeIndex(pd.Series(day).map(closes)).fillna(pd.NaT)
    close = pd.DatetimeIndex(np.where(close.isna(), day + pd.Timedelta(hours=16), close))
    seam = close - decision_lead(1545) - pd.Timedelta(minutes=45)                              # the last SIP bar start live can see at the decision (15:00 on a normal day)
    late = (s.index > seam) & (s.index >= pd.Timestamp(START))
    keep = s[~late]
    if not len(x):
        return keep                                                                             # no IEX history: live would see nothing after the seam
    xd = x.index.normalize()
    xc = pd.DatetimeIndex(pd.Series(xd).map(closes))
    xc = pd.DatetimeIndex(np.where(xc.isna(), xd + pd.Timedelta(hours=16), xc))
    xs = x[(x.index > xc - decision_lead(1545) - pd.Timedelta(minutes=45)) & (x.index >= pd.Timestamp(START))]
    return pd.concat([keep, xs]).sort_index()


syms = E.model_symbols(m)
have_iex = [s for s in syms if len(D.load_bars(s, "iex"))]
print(f"{len(syms)} symbols, {len(have_iex)} with IEX history", flush=True)
mk_sip = Market(syms, minute=1545, start="2015-06-01")
mk_hyb = Market(syms, minute=1545, start="2015-06-01", bars_fn=hybrid)
a = E.run(m, mk_sip, start="2016-09-01")
b = E.run(m, mk_hyb, start="2016-09-01")
t0 = pd.Timestamp(START)
ta, tb = a.trades[a.trades.entry_date >= t0], b.trades[b.trades.entry_date >= t0]
ka = set(zip(ta.entry_date, ta.symbol))
kb = set(zip(tb.entry_date, tb.symbol))
both = len(ka & kb)
print(f"entries since {START}: SIP {len(ka)}, hybrid {len(kb)}, same (day, ticker) {both} = {both / max(len(ka), 1):.1%} of SIP's")
j = mk_sip.dates >= t0
d = np.abs(mk_hyb.sP[j] - mk_sip.sP[j]) / mk_sip.sP[j]
print(f"snapshot last price, hybrid vs SIP: median |diff| {np.nanmedian(d) * 1e4:.1f} bp, 95th pct {np.nanpercentile(d, 95) * 1e4:.1f} bp")
ia = (mk_sip.sP - mk_sip.sL) / (mk_sip.sH - mk_sip.sL)
ib = (mk_hyb.sP - mk_hyb.sL) / (mk_hyb.sH - mk_hyb.sL)
di = np.abs(ia[j] - ib[j])
print(f"IBS hybrid vs SIP: median |diff| {np.nanmedian(di):.3f}, 95th pct {np.nanpercentile(di, 95):.3f}; IBS<0.2 agreement {np.nanmean((ia[j] < 0.2) == (ib[j] < 0.2)):.1%}")
oa, ob = a.returns[a.returns.index >= t0], b.returns[b.returns.index >= t0]
sh = lambda r: r.mean() / r.std() * np.sqrt(252)
cg = lambda r: (1 + r).prod() ** (252 / len(r)) - 1
print(f"since {START}: SIP Sharpe {sh(oa):.2f} CAGR {cg(oa):.1%} | hybrid Sharpe {sh(ob):.2f} CAGR {cg(ob):.1%}")
