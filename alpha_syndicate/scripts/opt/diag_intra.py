"""Train-only diagnostics to decide timeframe (daily vs 1h) and class (Dip vs Trend+Dip). No Test data touched."""
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
import sim
from sim import RSI_LENS, RSI_ENTRIES, EXIT_LENS, TREND_LENS, MAX_HOLDS, MODE_DIP, MODE_TREND, MODE_TD, TRAIN
import intra
from intra import IMarket, _core_intra, metrics_from_out

SW = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt\sweeps"
LEV = ["TQQQ", "SPXL"]
D = {a: {k: dict(np.load(f"{SW}\\{a}_{k}_train_1h.npz")) for k in ("dipT", "dipF", "td")} for a in LEV}


def passes(d):
    with np.errstate(invalid="ignore"):
        return ((d["trades"] >= 60) & (d["expo"] >= 0.05) & (d["cagr"] > 0) & (d["sh1"] > 0) & (d["sh2"] > 0)
                & (d["mdd"] <= 0.65) & np.isfinite(d["sharpe"]) & np.isfinite(d["calmar"]))


def rank_fn(v):
    s = np.sort(v)
    return lambda x: np.searchsorted(s, x, side="right") / len(s)


def top_class(variants, n=3):
    ok = {v: passes(D["TQQQ"][v]) & passes(D["SPXL"][v]) for v in variants}
    pool = {a: (np.concatenate([D[a][v]["sharpe"][ok[v]] for v in variants]), np.concatenate([D[a][v]["calmar"][ok[v]] for v in variants])) for a in LEV}
    rk = {a: (rank_fn(pool[a][0]), rank_fn(pool[a][1])) for a in LEV}
    rows = []
    for v in variants:
        sa = {a: np.where(ok[v], 0.5 * rk[a][0](D[a][v]["sharpe"]) + 0.5 * rk[a][1](D[a][v]["calmar"]), 0.0) for a in LEV}
        S = np.where(ok[v], 0.5 * (sa["TQQQ"] + sa["SPXL"]) - 0.5 * np.abs(sa["TQQQ"] - sa["SPXL"]), 0.0)
        Ssm = uniform_filter(S, size=3, mode="nearest")
        for idx in np.argsort(Ssm.ravel())[::-1][:400]:
            ii = np.unravel_index(idx, Ssm.shape); rows.append((Ssm[ii], v, ii))
    rows.sort(key=lambda t: -t[0]); seen = set(); out = []
    for sm, v, ii in rows:
        key = (v,) + tuple(int(x) // 2 for x in ii)
        if key in seen:
            continue
        seen.add(key); out.append((sm, v, ii))
        if len(out) >= n:
            break
    print(f"  passing both: { {v: int(ok[v].sum()) for v in variants} }")
    for sm, v, ii in out:
        ri, ei, xi, ti, hi = ii
        print(f"  {v} rsiLen {RSI_LENS[ri]} entry<{RSI_ENTRIES[ei]} exit{EXIT_LENS[xi]} trend{TREND_LENS[ti]} hold{MAX_HOLDS[hi]}d | S_smooth {sm:.3f}")
        for a in LEV:
            d = D[a][v]
            print(f"      {a:5s} Sharpe {d['sharpe'][ii]:.2f} (H1 {d['sh1'][ii]:.2f}/H2 {d['sh2'][ii]:.2f}) Calmar {d['calmar'][ii]:.2f} CAGR {d['cagr'][ii]:.1%} MDD {d['mdd'][ii]:.1%} trades {d['trades'][ii]:.0f} expo {d['expo'][ii]:.1%} corr {d['corr'][ii]:.2f} win {d['win'][ii]:.1%} PF {d['pf'][ii]:.2f}")
    return out


print("=== 1h, DIP class only (diversifier class; dipTrend on/off) -- compare with daily Dip A: TQQQ Sharpe .80/Calmar .55, SPXL .78/.45; B: .85/.66, .84/.50 ===")
top_class(["dipT", "dipF"], 4)
print("\n=== 1h, Trend+Dip class ===")
top_class(["td"], 2)

# ---- extension of the 1h Trend+Dip winner beyond the grid edges (entry<10 and trend 250 are the grid corners) ----
SPEC = {"TQQQ": ("TQQQ", "QQQ"), "SPXL": ("SPXL", "SPY")}
M = {a: IMarket(*SPEC[a], "60") for a in LEV}
dd = {a: pd.read_csv(rf"{intra.DATA}\yf_{SPEC[a][1]}_1D.csv", index_col="date", parse_dates=True)["Close"] for a in LEV}


def st_for(m, proxy_daily, n):
    s = sim.sma(proxy_daily.values.astype(np.float64), int(n))
    dpos = np.searchsorted(proxy_daily.index.values, m.t["date"].values, side="left") - 1
    return np.where(dpos >= 0, s[np.clip(dpos, 0, None)], np.nan)


def run_custom(a, mode, rl, re, el, tl, mh_days, dt=True):
    m = M[a]; inwin, last = m.window(TRAIN)
    ri = int(np.where(RSI_LENS == rl)[0][0]); xi = int(np.where(EXIT_LENS == el)[0][0])
    st = m.st[int(np.where(TREND_LENS == tl)[0][0])] if tl in TREND_LENS else st_for(m, dd[a], tl)
    o = _core_intra(m.op, m.cl, m.pc, m.rsi[ri], m.sx[xi], st, inwin, last, m.day, mode, float(re), dt, int(mh_days * m.bpd) if mh_days else 0, 0.001, 20160101, 0.01)
    return metrics_from_out(o)


def row(lab, f):
    s = []
    for a in LEV:
        x = f(a)
        s.append(f"{a} Sharpe {x['sharpe']:.2f} Calmar {x['calmar']:.2f} CAGR {x['cagr']:.1%} MDD {x['mdd']:.1%} expo {x['exposure']:.0%} corr {x['corr']:.2f} tr {x['trades']}")
    print(f"  {lab:34s} " + " | ".join(s))


print("\n=== extension check, 1h winner td rsi6/entry<10/exit17/trend250/hold18 ===")
row("winner (grid corner)", lambda a: run_custom(a, MODE_TD, 6, 10, 17, 250, 18))
for re in (2, 4, 6, 8, 12, 14):
    row(f"  entry<{re}", lambda a, re=re: run_custom(a, MODE_TD, 6, re, 17, 250, 18))
for tl in (200, 225, 275, 300, 350, 400):
    row(f"  trend {tl}", lambda a, tl=tl: run_custom(a, MODE_TD, 6, 10, 17, tl, 18))
row("1h TREND only 250", lambda a: run_custom(a, MODE_TREND, 6, 10, 17, 250, 18))
row("1h TREND only 200", lambda a: run_custom(a, MODE_TREND, 6, 10, 17, 200, 18))

print("\n=== like-for-like DAILY engine ===")
MK = {"TQQQ": sim.Market("TQQQ", "QQQ", src="yf", traded_src="tv"), "SPXL": sim.Market("SPXL", "SPY", src="yf")}
def drow(lab, **kw):
    s = []
    for a in LEV:
        x = sim.run_one(MK[a], TRAIN, **kw)
        s.append(f"{a} Sharpe {x['sharpe']:.2f} Calmar {x['calmar']:.2f} CAGR {x['cagr']:.1%} MDD {x['mdd']:.1%} expo {x['exposure']:.0%} corr {x['corr']:.2f} tr {x['trades']}")
    print(f"  {lab:34s} " + " | ".join(s))
drow("daily td 6/10/17/250/18", mode=MODE_TD, rsi_len=6, rsi_entry=10, exit_len=17, trend_len=250, max_hold=18)
drow("daily TREND 250", mode=MODE_TREND, rsi_len=6, rsi_entry=10, exit_len=17, trend_len=250, max_hold=18)
drow("daily TREND 200", mode=MODE_TREND, rsi_len=6, rsi_entry=10, exit_len=17, trend_len=200, max_hold=18)
drow("daily Dip A 4/34/8/225/20", mode=MODE_DIP, rsi_len=4, rsi_entry=34, exit_len=8, trend_len=225, max_hold=20)
