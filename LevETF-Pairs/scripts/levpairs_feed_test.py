"""
Feed-fidelity test. This paper account may NOT query recent SIP data, only IEX, so the live signals are computed on IEX bars while the research used the consolidated SIP tape.
For each pair and bar size: how close are the IEX s-scores to the SIP s-scores, how many IEX bars are empty, and what does the strategy earn if it trades on IEX signals but is marked on SIP prices
(the consolidated market is where the orders would trade)?   Output: reports/feed_test.log
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

import levpairs_core as C
import levpairs_data as D
import levpairs_sim as S
from levpairs_ui.presets import DEFAULT_PRESET, cfgs


def run(tf: int, sc0: C.SigCfg, c: S.SimCfg) -> None:
    sc = replace(sc0, tf=tf)
    print(f"\n=== {tf}-minute bars (IEX history starts 2020-07-27)")
    sigs_sip, sigs_mix = {}, {}
    for p in D.PAIRS:
        name = D.pair_name(p)
        sip, iex = D.load_pairdata(p, tf, "sip", start="2020-08-03", max_missing=1.0), D.load_pairdata(p, tf, "iex", start="2020-08-03", max_missing=1.0)
        days = sip.dates.intersection(iex.dates)
        ix = lambda d_: np.array([list(d_.dates).index(x) for x in days])
        a, b = ix(sip), ix(iex)
        ss, si = C.build_signals(sip, sc), C.build_signals(iex, sc)
        s1, s2 = ss.s[a], si.s[b]
        ok = np.isfinite(s1) & np.isfinite(s2)
        big = ok & (np.abs(s1) > 1.5)
        miss = iex.L_miss[b].mean(), iex.B_miss[b].mean()
        print(f"{name}: sessions {len(days):,} | empty IEX bars: leveraged leg {miss[0]:.1%}, 1x leg {miss[1]:.1%} | s-score corr IEX vs SIP {np.corrcoef(s1[ok], s2[ok])[0, 1]:.3f} | "
              f"same side when |s_sip| > 1.5: {(np.sign(s1[big]) == np.sign(s2[big])).mean():.1%} | median |s diff| {np.median(np.abs(s1[ok] - s2[ok])):.2f}")
        # IEX signal, SIP marks: the decision comes from IEX, the P&L arrays from SIP
        sigs_sip[name] = replace(ss, dates=days, s=ss.s[a], hl_min=ss.hl_min[a], sig_eq_bp=ss.sig_eq_bp[a], eps=ss.eps[a], eps_gap=ss.eps_gap[a], eps_body=ss.eps_body[a], beta=ss.beta[a], X=ss.X[a], rB_day=ss.rB_day[a],
                                 L_close=ss.L_close[a], B_close=ss.B_close[a], L_open=ss.L_open[a], B_open=ss.B_open[a])
        sigs_mix[name] = replace(sigs_sip[name], s=si.s[b], hl_min=si.hl_min[b], sig_eq_bp=si.sig_eq_bp[b], beta=ss.beta[a])
    for label, sg in (("SIP signal, SIP marks", sigs_sip), ("IEX signal, SIP marks", sigs_mix)):
        res = S.simulate(sg, replace(c, start="2020-08-03"))
        m = S.metrics(res.ret)
        tr = res.trades
        print(f"  {label:24s} Sharpe {m.get('sharpe', float('nan')):6.2f} | round trips {len(tr):6,} | gross {tr.gross_bp.mean():+.2f} bp, net {tr.net_bp.mean():+.2f} bp per trip")


if __name__ == "__main__":
    sc, c = cfgs(DEFAULT_PRESET)
    c = replace(c, vol_on=False, g_init=1.0)
    print(f"preset: {DEFAULT_PRESET}")
    for tf in (5, 15, 30):
        run(tf, sc, c)
    print("\n=== 1-minute bars: share of empty IEX minutes")
    for sym in D.SYMBOLS:
        b = pd.read_parquet(D.path1(sym, "iex"))
        s = pd.read_parquet(D.path1(sym, "sip"))
        s = s[s.index >= b.index[0]]
        print(f"{sym}: IEX has a bar in {len(b.index.intersection(s.index)) / len(s):.1%} of the minutes the consolidated tape has one")
