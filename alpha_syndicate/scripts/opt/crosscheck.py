"""Train-only cross-checks for the leading candidates (no Test data touched)."""
import numpy as np
import pandas as pd
from sim import *
from sim import _core

SPEC = {"TQQQ": ("TQQQ", "QQQ", "tv"), "QQQ": ("QQQ", "QQQ", None), "SPXL": ("SPXL", "SPY", None), "SPY": ("SPY", "SPY", None)}
CANDS = {
    "A Dip 4/34/8/225/20": dict(mode=MODE_DIP, rl=4, re=34, el=8, tl=225, mh=20, dt=True),
    "B Dip 5/36/8/250/20": dict(mode=MODE_DIP, rl=5, re=36, el=8, tl=250, mh=20, dt=True),
    "C T+D 2/34/7/50/16": dict(mode=MODE_TD, rl=2, re=34, el=7, tl=50, mh=16, dt=True),
    "default Dip 2/20/5/200/10": dict(mode=MODE_DIP, rl=2, re=20, el=5, tl=200, mh=10, dt=True),
}
MK = {a: Market(*(SPEC[a][:2]), src="yf", traded_src=SPEC[a][2]) for a in SPEC}


def yearly(m, win, c, comm=0.001):
    mt, rets = run_one(m, win, c["mode"], c["rl"], c["re"], c["el"], c["tl"], c["mh"], c["dt"], comm=comm, series=True)
    inwin, _ = m.window(win)
    s = pd.Series(rets[inwin], index=m.dates[inwin])
    return (1 + s).groupby(s.index.year).prod() - 1


print("=== cross-asset Train metrics (comm 0.1%/side) ===")
for name, c in CANDS.items():
    print(f"\n{name}")
    for a in SPEC:
        mt = run_one(MK[a], TRAIN, c["mode"], c["rl"], c["re"], c["el"], c["tl"], c["mh"], c["dt"])
        print(f"   {a:5s} Sharpe {mt['sharpe']:.2f}  Calmar {mt['calmar']:.2f}  CAGR {mt['cagr']:.1%}  MDD {mt['mdd']:.1%}  trades {mt['trades']:3d}  win {mt['win']:.1%}  PF {mt['pf']:.2f}  expo {mt['exposure']:.1%}  corr {mt['corr']:.2f}")

print("\n=== cost stress: 0.2% per side (TQQQ / SPXL Sharpe, CAGR) ===")
for name, c in CANDS.items():
    o = []
    for a in ("TQQQ", "SPXL"):
        mt = run_one(MK[a], TRAIN, c["mode"], c["rl"], c["re"], c["el"], c["tl"], c["mh"], c["dt"], comm=0.002)
        o.append(f"{a} Sharpe {mt['sharpe']:.2f} CAGR {mt['cagr']:.1%}")
    print(f"{name:26s} " + " | ".join(o))

print("\n=== calendar-year returns on Train (TQQQ strategy vs buy&hold) for candidate A ===")
yA = yearly(MK["TQQQ"], TRAIN, CANDS["A Dip 4/34/8/225/20"])
yS = yearly(MK["SPXL"], TRAIN, CANDS["A Dip 4/34/8/225/20"])
bh = pd.Series(MK["TQQQ"].cl, index=MK["TQQQ"].dates); bh = bh[(bh.index >= "2010-02-11") & (bh.index <= "2021-12-31")]
bhy = bh.groupby(bh.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
print(pd.DataFrame({"TQQQ strat": yA, "SPXL strat": yS, "TQQQ B&H (approx)": bhy}).applymap(lambda v: f"{v:+.1%}").to_string())
print(f"positive years TQQQ strat: {(yA>0).sum()}/{len(yA)}  SPXL strat: {(yS>0).sum()}/{len(yS)}")

# ---- boundary / extension sanity (Train only): is the optimum pinned to the grid edge? ----
print("\n=== extension check for candidate A (Train, TQQQ & SPXL): max-hold and trend length beyond the grid ===")
def custom(m, c, rl, re, el, tl, mh):
    p = pd.read_csv(rf"{DATA}\yf_{m.proxy}_1D.csv", index_col="date", parse_dates=True)["Close"].values
    idx = pd.read_csv(rf"{DATA}\yf_{m.proxy}_1D.csv", index_col="date", parse_dates=True).index
    al = lambda a: pd.Series(a, index=idx).reindex(m.dates).values.copy()
    rsi = al(rsi_wilder(p, int(rl))); sx = al(sma(p, int(el))); st = al(sma(p, int(tl)))
    inwin, last = m.window(TRAIN)
    o, *_ = _core(m.op, m.cl, m.pc, rsi, sx, st, inwin, last, c["mode"], float(re), c["dt"], int(mh), 0.001, 0, False, 0.01, 2)
    return metrics(o)
cA = CANDS["A Dip 4/34/8/225/20"]
for label, mh, tl in [("hold 20 / trend 225 (A)", 20, 225), ("hold OFF(0) / trend 225", 0, 225), ("hold 30 / trend 225", 30, 225), ("hold 20 / trend 275", 20, 275), ("hold 20 / trend 300", 20, 300), ("hold OFF / trend 250", 0, 250)]:
    row = []
    for a in ("TQQQ", "SPXL"):
        mt = custom(MK[a], cA, 4, 34, 8, tl, mh)
        row.append(f"{a} Sharpe {mt['sharpe']:.2f} Calmar {mt['calmar']:.2f} CAGR {mt['cagr']:.1%} MDD {mt['mdd']:.1%} tr {mt['trades']}")
    print(f"{label:26s} " + " | ".join(row))
