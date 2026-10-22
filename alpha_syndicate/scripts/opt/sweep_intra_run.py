"""1h Train-only exhaustive sweep + the same pre-registered selection as daily (TQQQ<-QQQ, SPXL<-SPY, 3 variants)."""
import time, json
import numpy as np
from scipy.ndimage import uniform_filter
from intra import *

OUT = r"C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\data\opt\sweeps"
SPEC = {"TQQQ": ("TQQQ", "QQQ"), "SPXL": ("SPXL", "SPY")}
VAR = {"dipT": (MODE_DIP, True), "dipF": (MODE_DIP, False), "td": (MODE_TD, True)}
t0 = time.time()
D = {}
for a, (tr, px) in SPEC.items():
    m = IMarket(tr, px, "60")
    print(a, "bars", len(m.op), "first", m.t["dt"].iloc[0], flush=True)
    D[a] = {}
    for k, (mode, dt) in VAR.items():
        d, raw = sweep_intra(m, mode, dt)
        D[a][k] = d
        np.savez_compressed(f"{OUT}\\{a}_{k}_train_1h.npz", **d)
        print(f"  {a} {k} done {time.time()-t0:.0f}s", flush=True)

LEV = list(SPEC)


def passes(d):
    with np.errstate(invalid="ignore"):
        return ((d["trades"] >= 60) & (d["expo"] >= 0.05) & (d["cagr"] > 0) & (d["sh1"] > 0) & (d["sh2"] > 0)
                & (d["mdd"] <= 0.65) & np.isfinite(d["sharpe"]) & np.isfinite(d["calmar"]))


ok = {v: passes(D["TQQQ"][v]) & passes(D["SPXL"][v]) for v in VAR}
print("passing both (of 336,960):", {v: int(ok[v].sum()) for v in VAR})
if sum(int(o.sum()) for o in ok.values()) == 0:
    print("NO 1h configuration passes the pre-registered filters on both leveraged assets -> intraday rejected")
    # show best raw Sharpe anyway for the record
    for a in LEV:
        for v in VAR:
            sh = np.where(np.isfinite(D[a][v]["sharpe"]), D[a][v]["sharpe"], -9)
            i = np.unravel_index(np.argmax(sh), sh.shape)
            print(f"  best raw {a} {v}: Sharpe {sh[i]:.2f} CAGR {D[a][v]['cagr'][i]:.1%} MDD {D[a][v]['mdd'][i]:.1%} trades {D[a][v]['trades'][i]:.0f} idx {i}")
    raise SystemExit

def rank_fn(vals):
    s = np.sort(vals)
    return lambda x: np.searchsorted(s, x, side="right") / len(s)

pool = {a: {"sh": np.concatenate([D[a][v]["sharpe"][ok[v]] for v in VAR]), "ca": np.concatenate([D[a][v]["calmar"][ok[v]] for v in VAR])} for a in LEV}
rk = {a: (rank_fn(pool[a]["sh"]), rank_fn(pool[a]["ca"])) for a in LEV}
S, Ssm = {}, {}
for v in VAR:
    sa = {a: np.where(ok[v], 0.5 * rk[a][0](D[a][v]["sharpe"]) + 0.5 * rk[a][1](D[a][v]["calmar"]), 0.0) for a in LEV}
    S[v] = np.where(ok[v], 0.5 * (sa["TQQQ"] + sa["SPXL"]) - 0.5 * np.abs(sa["TQQQ"] - sa["SPXL"]), 0.0)
    Ssm[v] = uniform_filter(S[v], size=3, mode="nearest")
rows = []
for v in VAR:
    for idx in np.argsort(Ssm[v].ravel())[::-1][:500]:
        ii = np.unravel_index(idx, Ssm[v].shape)
        rows.append((Ssm[v][ii], v, ii))
rows.sort(key=lambda t: -t[0])
print("\n=== TOP 1h plateau centres ===")
seen = set(); shown = 0
for sm, v, ii in rows:
    key = (v,) + tuple(int(x) // 2 for x in ii)
    if key in seen:
        continue
    seen.add(key); shown += 1
    ri, ei, xi, ti, hi = ii
    print(f"\n{v} rsiLen {RSI_LENS[ri]} entry<{RSI_ENTRIES[ei]} exit{EXIT_LENS[xi]} trend{TREND_LENS[ti]} hold{MAX_HOLDS[hi]}d | S_smooth {sm:.3f}")
    for a in LEV:
        d = D[a][v]
        print(f"   {a:5s} Sharpe {d['sharpe'][ii]:.2f} (H1 {d['sh1'][ii]:.2f}/H2 {d['sh2'][ii]:.2f}) Calmar {d['calmar'][ii]:.2f} CAGR {d['cagr'][ii]:.1%} MDD {d['mdd'][ii]:.1%} trades {d['trades'][ii]:.0f} expo {d['expo'][ii]:.1%} corr {d['corr'][ii]:.2f}")
    if shown >= 6:
        break
print("done", f"{time.time()-t0:.0f}s")
