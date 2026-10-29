"""
Valid's checks on the time-shared momentum book (overnight momentum + intraday noise area on the same capital):
  * the in-sample-selected combination, the a-priori one (simplest overnight signal + the paper's noise-area settings) and the blend;
  * costs: auctions 1 / 2 / 3 bp per side, QQQ half spread 0.5 / 1 / 2 bp;
  * the overnight exit at the first trade instead of the official open;
  * the free data path: the overnight decision at 15:30 on the delayed SIP feed, and the intraday decisions on real-time IEX bars (2024-01 -> 2026-10, the IEX cache);
  * year by year, rolling 12-month Sharpe, deflated Sharpe over every trial of the momentum / prediction search and of the MR study.
"""
import glob
import os

import numpy as np
import pandas as pd

from mp_lib import OUT, SPLIT, Book, Ctx, dsr, log, overnight, psr, stats, topk
from mp_overnight_ref import signals, spy_trend
from mp_sleeves import noise_area

cx = Ctx()
bk = Book("mp_combo_check", cx)
bench = cx.spy_ret()
bk.add(bench, "SPY buy & hold")
tr = spy_trend(cx)
U = cx.top(150) | cx.etf()


def rk(x, m):
    return pd.DataFrame(np.where(m & np.isfinite(x), x, np.nan)).rank(axis=1, pct=True).to_numpy()


def on_leg(sig, k, lead=1, cost=1.0, exit="open"):
    live, S = signals(cx, lead)
    m = U & live
    if sig == "blend_d_v":
        s = rk(S["d45"], m) + rk(np.where(np.isfinite(S["vsurp"]), S["vsurp"], 0), m)
    else:
        s = S[sig]
    return overnight(cx, topk(s, m & tr, k), k=k, cost_in=cost, cost_out=cost, exit=exit)


def combo(ro, ri):
    return (1 + ro) * (1 + ri) - 1


CANDS = {
    "IS-selected: d45 top20 + QQQ 30m lb20": (("d45", 20), dict(every=2, lookback=20)),
    "a priori: d45 top20 + QQQ 30m lb14 (paper settings)": (("d45", 20), dict(every=2, lookback=14)),
    "blend: d45+volume top20 + QQQ 30m lb14": (("blend_d_v", 20), dict(every=2, lookback=14)),
    "higher return: d45 top10 + QQQ 30m lb14": (("d45", 10), dict(every=2, lookback=14)),
}
years = {}
main = {}
for name, ((sig, k), nkw) in CANDS.items():
    ro = on_leg(sig, k)
    ri = noise_area(cx, "QQQ", **nkw)
    r = combo(ro, ri)
    main[name] = r
    years[name] = r
    bk.add(r, f"{name}", chk="base")
    bk.add(ro, f"{name} | overnight leg alone", chk="leg_on")
    bk.add(ri, f"{name} | intraday leg alone", chk="leg_id")
    for c in (2.0, 3.0):
        bk.add(combo(on_leg(sig, k, cost=c), noise_area(cx, "QQQ", half_bp=c / 2, auction_bp=c, **nkw)), f"{name} | auctions {c:.0f} bp, QQQ half spread {c / 2:.1f} bp", chk=f"cost{c:.0f}")
    bk.add(combo(on_leg(sig, k, exit="first"), ri), f"{name} | overnight exit at the first trade", chk="first")
    bk.add(combo(on_leg(sig, k, lead=2), ri), f"{name} | overnight decided at 15:30 (delayed SIP)", chk="1530")
    log(name)

# the free real-time path for the intraday leg: IEX bars for the session's decisions (2024-01 on)
iex = {}
for name, ((sig, k), nkw) in list(CANDS.items())[:2]:
    trades_s, trades_i = [], []
    rs = noise_area(cx, "QQQ", trades=trades_s, **nkw)
    ri = noise_area(cx, "QQQ", live_feed="iex", trades=trades_i, **nkw)
    have = ri.index[np.isin(ri.index, pd.DatetimeIndex(sorted({t["date"] for t in trades_i})))]
    w0 = pd.Timestamp("2024-01-02")
    a = pd.DataFrame(trades_s)
    b = pd.DataFrame(trades_i)
    a = a[a.date >= w0]
    days_s, days_i = set(a.date), set(b.date)
    both = days_s & days_i
    same_in = sum(1 for d in both if a[a.date == d].in_slot.min() == b[b.date == d].in_slot.min())
    seg_s, seg_i = rs[rs.index >= w0], ri[ri.index >= w0]
    iex[name] = {"sip_trade_days": len(days_s), "iex_trade_days": len(days_i), "both": len(both), "same_first_entry": same_in,
                 "sip_sh": seg_s.mean() / seg_s.std() * np.sqrt(252), "iex_sh": seg_i.mean() / seg_i.std() * np.sqrt(252),
                 "sip_ann": seg_s.mean() * 252, "iex_ann": seg_i.mean() * 252, "corr": float(np.corrcoef(seg_s, seg_i)[0, 1])}
print(pd.DataFrame(iex).T.round(3).to_string())

# deflated Sharpe: every trial of this search (+ the MR study's 1,871)
trial_sh = []
for f in glob.glob(os.path.join(OUT, "mp_*.csv")):
    if any(x in f for x in ("_check", "_ic", "mp_combo_check")):
        continue
    d = pd.read_csv(f)
    if "sh_is" in d:
        trial_sh += list(d.loc[~d.label.str.contains("buy & hold", na=False), "sh_is"].dropna())
n_mp = len(trial_sh)
combo_csv = pd.read_csv(os.path.join(OUT, "mp_combo.csv"))
v_combo = combo_csv[combo_csv.fam == "COMBO"].sh_is.var()
v_all = np.var(trial_sh)
rows = []
for name, r in main.items():
    ri_ = r[r.index < SPLIT]
    ro_ = r[r.index >= SPLIT]
    d1, e1 = dsr(ri_, n_mp + 1871, v_all)
    d2, e2 = dsr(ri_, int((combo_csv.fam == "COMBO").sum()) + 742, v_combo)
    rows.append({"model": name, "sh_is": stats(r)["sh_is"], "dsr_all_trials": d1, "exp_max_all": e1, "dsr_family": d2, "exp_max_family": e2,
                 "psr_oos>0": psr(ro_), "psr_oos>spy_sharpe": psr(ro_, 1.377), "psr_oos>2": psr(ro_, 2.0)})
print(f"\ntrials: {n_mp} in this search + 1,871 MR; IS Sharpe variance all {v_all:.3f}, combo family {v_combo:.3f}")
print(pd.DataFrame(rows).round(3).to_string(index=False))
df = bk.save()
with pd.option_context("display.width", 300, "display.max_rows", 400, "display.max_colwidth", 110):
    print(df[["label", "sh_all", "sh_is", "sh_oos", "cagr_all", "cagr_is", "cagr_oos", "vol_all", "mdd_all", "beta"]].round(3).to_string(index=False))
    y = pd.DataFrame({k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in years.items()})
    y["SPY"] = (1 + bench).groupby(bench.index.year).prod() - 1
    print((y * 100).round(1).T.to_string())
    roll = {k: v.rolling(252).mean() / v.rolling(252).std() * np.sqrt(252) for k, v in main.items()}
    rr = pd.DataFrame(roll).dropna()
    print("\nrolling 12-month Sharpe: share of windows above 2 / above 1 / below 0, min")
    for k in rr:
        print(f"  {k}: {(rr[k] > 2).mean():.0%} / {(rr[k] > 1).mean():.0%} / {(rr[k] < 0).mean():.0%}, min {rr[k].min():.2f}")
