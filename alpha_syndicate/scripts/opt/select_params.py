"""Parameter selection exactly as written in data/opt/PROTOCOL.md (TRAIN data only)."""
import json
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter, uniform_filter1d
from sim import *
from sweep import sweep_market, GRID_SHAPE

SW = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt\sweeps"
LEV = ["TQQQ", "SPXL"]
CHK = ["QQQ", "SPY"]
SPEC = {"TQQQ": ("TQQQ", "QQQ", "tv"), "QQQ": ("QQQ", "QQQ", None), "SPXL": ("SPXL", "SPY", None), "SPY": ("SPY", "SPY", None)}
KEYS = ["dipT", "dipF", "td"]
NAMES = {"dipT": "Dip (dipTrend on)", "dipF": "Dip (dipTrend off)", "td": "Trend + Dip", "trend": "Trend"}


def load_all():
    D = {}
    for a in SPEC:
        D[a] = {k: dict(np.load(f"{SW}\\{a}_{k}_train.npz")) for k in KEYS}
        m = Market(SPEC[a][0], SPEC[a][1], src="yf", traded_src=SPEC[a][2])
        r = sweep_market(m, MODE_TREND)                       # trend mode ignores everything but trendLen
        D[a]["trend"] = {kk: v[0, 0, 0, :, 0] for kk, v in r.items()}   # 1-D over trendLen
    return D


def passes(d):
    with np.errstate(invalid="ignore"):
        return ((d["trades"] >= 60) & (d["expo"] >= 0.05) & (d["cagr"] > 0) & (d["sh1"] > 0) & (d["sh2"] > 0)
                & (d["mdd"] <= 0.65) & np.isfinite(d["sharpe"]) & np.isfinite(d["calmar"]))


def rank_fn(vals):
    s = np.sort(vals)
    return lambda x: np.searchsorted(s, x, side="right") / len(s)


def main():
    D = load_all()
    variants = KEYS + ["trend"]
    ok = {v: passes(D[LEV[0]][v]) & passes(D[LEV[1]][v]) for v in variants}
    npass = {v: int(ok[v].sum()) for v in variants}
    ntot = {v: int(ok[v].size) for v in variants}
    print("passing both leveraged assets (of total):", {NAMES[v]: f"{npass[v]:,}/{ntot[v]:,}" for v in variants})
    # pooled rank functions per asset (population = configs passing both filters, all variants)
    pool = {a: {"sh": np.concatenate([D[a][v]["sharpe"][ok[v]] for v in variants]),
                "ca": np.concatenate([D[a][v]["calmar"][ok[v]] for v in variants])} for a in LEV}
    rk = {a: (rank_fn(pool[a]["sh"]), rank_fn(pool[a]["ca"])) for a in LEV}
    S, Ssm, sc = {}, {}, {}
    for v in variants:
        sa = {}
        for a in LEV:
            sa[a] = np.where(ok[v], 0.5 * rk[a][0](D[a][v]["sharpe"]) + 0.5 * rk[a][1](D[a][v]["calmar"]), 0.0)
        S[v] = np.where(ok[v], 0.5 * (sa[LEV[0]] + sa[LEV[1]]) - 0.5 * np.abs(sa[LEV[0]] - sa[LEV[1]]), 0.0)
        sc[v] = sa
        Ssm[v] = uniform_filter1d(S[v], size=3, mode="nearest") if S[v].ndim == 1 else uniform_filter(S[v], size=3, mode="nearest")

    # candidate list: best smoothed cell per variant + overall top cells
    rows = []
    for v in variants:
        flat = np.argsort(Ssm[v].ravel())[::-1][:3000]
        for idx in flat:
            ii = np.unravel_index(idx, Ssm[v].shape)
            rows.append((Ssm[v][ii], v, ii))
    rows.sort(key=lambda t: -t[0])

    def params(v, ii):
        if v == "trend":
            return dict(mode="Trend", rsiLen=2, rsiEntry=20, exitLen=5, trendLen=int(TREND_LENS[ii[0]]), maxHold=10, dipTrend=True)
        ri, ei, xi, ti, hi = ii
        return dict(mode=("Dip" if v.startswith("dip") else "Trend + Dip"), rsiLen=int(RSI_LENS[ri]), rsiEntry=float(RSI_ENTRIES[ei]),
                    exitLen=int(EXIT_LENS[xi]), trendLen=int(TREND_LENS[ti]), maxHold=int(MAX_HOLDS[hi]), dipTrend=(v != "dipF"))

    def neigh(v, ii):
        if v == "trend":
            sl = (slice(max(ii[0] - 1, 0), ii[0] + 2),)
        else:
            sl = tuple(slice(max(i - 1, 0), i + 2) for i in ii)
        return sl

    def cell(a, v, ii):
        return {k: float(D[a][v][k][ii]) for k in ("sharpe", "calmar", "cagr", "mdd", "sh1", "sh2", "trades", "expo", "corr", "win", "pf", "sortino")}

    def plateau(v, ii):
        sl = neigh(v, ii)
        out = {}
        for a in LEV:
            sh = D[a][v]["sharpe"]; c = sh[ii]
            nb = sh[sl].ravel(); okn = ok[v][sl].ravel()
            out[a] = dict(frac_pass=float(okn.mean()), frac_ge75=float(np.mean(okn & (nb >= 0.75 * c))), peak_ratio=float(c / np.nanmedian(nb)), nb_min=float(np.nanmin(nb)))
        return out

    print("\n=== TOP PLATEAU CENTRES (distinct by variant, by smoothed cross-market score) ===")
    seen = set(); shown = 0; top = []
    for sm, v, ii in rows:
        key = (v,) + tuple(int(x) // 2 for x in ii)      # crude de-duplication of adjacent cells
        if key in seen:
            continue
        seen.add(key)
        top.append((sm, v, ii))
        shown += 1
        if shown >= 12:
            break
    for sm, v, ii in top:
        p = params(v, ii); pl = plateau(v, ii)
        t, s = cell("TQQQ", v, ii), cell("SPXL", v, ii)
        print(f"\n{NAMES[v]:18s} rsiLen {p['rsiLen']:2d} entry<{p['rsiEntry']:4.0f} exit SMA{p['exitLen']:2d} trend {p['trendLen']:3d} hold {p['maxHold']:2d}  | S_smooth {sm:.3f}  S {S[v][ii]:.3f}")
        for a, c in (("TQQQ", t), ("SPXL", s)):
            print(f"   {a:5s} Sharpe {c['sharpe']:.2f} (H1 {c['sh1']:.2f} / H2 {c['sh2']:.2f})  Calmar {c['calmar']:.2f}  CAGR {c['cagr']:.1%}  MDD {c['mdd']:.1%}  trades {c['trades']:.0f}  expo {c['expo']:.1%}  corr {c['corr']:.2f}  win {c['win']:.1%}  PF {c['pf']:.2f}   plateau: {pl[a]['frac_ge75']:.0%} of nbrs >=75% Sharpe, peak/median {pl[a]['peak_ratio']:.2f}")
    json.dump([dict(S_smooth=float(sm), variant=v, idx=[int(x) for x in ii], params=params(v, ii)) for sm, v, ii in top],
              open(r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt\train_candidates.json", "w"), indent=1)

    # --- context: default parameters and buy & hold on Train ---
    print("\n=== CONTEXT (Train) ===")
    for a in SPEC:
        m = Market(SPEC[a][0], SPEC[a][1], src="yf", traded_src=SPEC[a][2])
        d0 = run_one(m, TRAIN, MODE_DIP, 2, 20, 5, 200, 10, True)
        # buy & hold on the traded asset
        inwin, _ = m.window(TRAIN)
        r = np.diff(m.cl, prepend=m.cl[0]) / np.concatenate([[m.cl[0]], m.cl[:-1]]); r = r[inwin]
        eq = np.cumprod(1 + r); pk = np.maximum.accumulate(eq)
        bh = dict(sharpe=r.mean() / r.std() * np.sqrt(252), cagr=eq[-1] ** (252 / len(r)) - 1, mdd=float(((pk - eq) / pk).max()))
        print(f"{a:5s} default Dip: Sharpe {d0['sharpe']:.2f} Calmar {d0['calmar']:.2f} CAGR {d0['cagr']:.1%} MDD {d0['mdd']:.1%} trades {d0['trades']}   |   buy&hold: Sharpe {bh['sharpe']:.2f} CAGR {bh['cagr']:.1%} MDD {bh['mdd']:.1%}")
    return D, ok, S, Ssm, top, params, cell


if __name__ == "__main__":
    main()
