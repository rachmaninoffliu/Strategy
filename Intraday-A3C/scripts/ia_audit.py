"""
Valid's audit of the product decision (reads the cached walk-forward edges, so it runs in a few minutes):
  A  slippage stress + block-bootstrap Sharpe interval
  B  drift / long-short / day-shuffle placebo / sign-flip / where in the day the P&L is earned
  C  cost of the first 30 minutes x1/x2/x3 (spreads were only sampled at 10:30, 13:00, 15:30) and a 'no trading before 10:00 / 10:30' variant
  D  equal-risk baskets
  E  ex-ante structural predictor: pre-2019 intraday vol / cost vs the 2019-26 net Sharpe
Output: reports/audit.log (stdout).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ia_data as D
import ia_portfolio as P
import ia_study as S

TOP = ["TSLA", "SMH", "NVDA", "AMZN", "QQQ"]


def sh(x) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def path(edge: np.ndarray, cost: float, k: float) -> np.ndarray:
    """Lots held after each decision (same rule as ia_study.policy_pnl)."""
    tgt = np.clip(np.sign(edge) * np.floor(np.abs(edge) / max(k * cost, 1e-9)), -S.VMAX, S.VMAX)
    pos = np.zeros_like(tgt); traded = np.zeros(tgt.shape[0]); cur = np.zeros(tgt.shape[0])
    for j in range(tgt.shape[1]):
        step = tgt[:, j] - cur
        step = np.sign(step) * np.minimum(np.abs(step), np.maximum(S.QHIGH - traded, 0))
        cur = cur + step; traded += np.abs(step); pos[:, j] = cur
    return pos


def net_j(pos: np.ndarray, nxt: np.ndarray, cj: np.ndarray) -> np.ndarray:
    prev = np.concatenate([np.zeros((pos.shape[0], 1)), pos[:, :-1]], axis=1)
    return (pos * nxt).sum(axis=1) - (np.abs(pos - prev) * cj[None, :]).sum(axis=1)


def main():
    sp = pd.read_csv(os.path.join(D.REPORTS, "spreads.csv")).set_index("symbol")
    rng = np.random.default_rng(1)
    data = {}
    for sym in TOP + ["SOXL"]:
        dd, E = P.edges_for(sym)
        data[sym] = (dd, E["ens"], float(sp.spread_bp[sym]) / 2, dd["days"].year.values >= 2019)

    print("== A. slippage stress (total per-lot cost = half spread + slip; policy thresholds move with cost) and 1000-draw 21-day block bootstrap")
    for sym, (dd, edge, half, m) in data.items():
        row = {}
        for slip in (0.25, 1.0, 2.0):
            c = half + slip
            net = net_j(path(edge, c, 3.0), dd["nxt"], np.full(72, c))
            row[f"slip {slip}"] = sh(net[m])
            if slip == 0.25:
                x = net[m]; n = len(x); bs = []
                for _ in range(1000):
                    idx = np.concatenate([np.arange(s, s + 21) % n for s in rng.integers(0, n, n // 21 + 1)])[:n]
                    bs.append(sh(x[idx]))
                row["boot5"], row["boot95"] = np.percentile(bs, 5), np.percentile(bs, 95)
                row["vol %/day of 3-lot capital"] = x.std() / 3 / 100
                row["worst day %"] = x.min() / 3 / 100
        print(sym, {k: round(float(v), 2) for k, v in row.items()})

    print("\n== B. drift / bias / placebo (day-shuffle keeps time-of-day structure and any long bias, destroys day-specific information; 100 shuffles)")
    for sym in TOP:
        dd, edge, half, m = data[sym]; c = half + S.SLIP_BP; nxt = dd["nxt"]
        pos = path(edge, c, 3.0); net = net_j(pos, nxt, np.full(72, c))
        longg = (np.clip(pos, 0, None) * nxt).sum(axis=1); shortg = (np.clip(pos, None, 0) * nxt).sum(axis=1)
        day_ret = nxt.sum(axis=1)
        flip = net_j(path(-edge, c, 3.0), nxt, np.full(72, c))
        idx = np.where(m)[0]; pl = []
        for _ in range(100):
            e2 = edge.copy(); e2[idx] = edge[rng.permutation(idx)]
            pl.append(sh(net_j(path(e2, c, 3.0), nxt, np.full(72, c))[m]))
        g = pos * nxt
        lab = {"09:35-10:30": range(0, 12), "10:30-12:00": range(12, 30), "12:00-14:00": range(30, 54), "14:00-15:00": range(54, 66), "15:05-16:00": range(66, 72)}
        print(f"{sym}: net Sharpe {sh(net[m]):.2f} | avg signed lots {pos[m].mean():+.2f}, avg |lots| {np.abs(pos[m]).mean():.2f} | gross bp/day long {longg[m].mean():+.1f} short {shortg[m].mean():+.1f} | "
              f"1-lot buy&hold {day_ret[m].mean():+.1f} bp/day | corr(P&L, day return) {np.corrcoef(net[m], day_ret[m])[0, 1]:+.2f}")
        print(f"   sign-flip {sh(flip[m]):.2f} | placebo mean {np.mean(pl):+.2f} p95 {np.percentile(pl, 95):+.2f} max {np.max(pl):+.2f}")
        print("   gross bp/day by interval:", {k: round(float(g[m][:, list(v)].sum(axis=1).mean()), 1) for k, v in lab.items()})

    print("\n== C. open-hour cost stress and no-open variants (Sharpe 2019-26)")
    store = {}
    for sym in TOP:
        dd, edge, half, m = data[sym]; base = half + S.SLIP_BP; nxt = dd["nxt"]; row = {}
        for mult in (1, 2, 3):
            cj = np.full(72, base); cj[:6] = base * mult
            row[f"open x{mult}"] = sh(net_j(path(edge, base, 3.0), nxt, cj)[m])
        for cut, lab in ((5, "no trade before 10:00"), (11, "no trade before 10:30")):
            e2 = edge.copy(); e2[:, :cut] = 0.0
            net = net_j(path(e2, base, 3.0), nxt, np.full(72, base)); row[lab] = sh(net[m])
            if cut == 5:
                store[sym] = pd.Series(net, index=dd["days"])[m]
        print(sym, {k: round(float(v), 2) for k, v in row.items()})

    print("\n== D. equal-risk baskets (inverse daily-vol weights), full-day policy and no-trade-before-10:00 policy")
    full = {s: pd.Series(net_j(path(data[s][1], data[s][2] + S.SLIP_BP, 3.0), data[s][0]["nxt"], np.full(72, data[s][2] + S.SLIP_BP)), index=data[s][0]["days"])[data[s][3]] for s in TOP}
    for label, st in (("full day", full), ("no trade < 10:00", store)):
        for names in (["TSLA", "SMH", "NVDA"], ["TSLA", "SMH"], ["TSLA", "SMH", "NVDA", "AMZN"]):
            df = pd.DataFrame({s: st[s] for s in names}); w = 1 / df.std(); b = (df * w).sum(axis=1) / w.sum()
            print(f"{label:17s} {'+'.join(names):22s} Sharpe {sh(b.values):.2f} | mean pairwise corr {df.corr().values[np.triu_indices(len(names), 1)].mean():.2f}"
                  + (" | by year " + str({int(y): round(sh(g.values), 2) for y, g in b.groupby(b.index.year)}) if len(names) == 3 and label != "full day" else ""))

    print("\n== E. ex-ante predictor: intraday 9:35->close vol (pre-2019 only) / cost vs 2019-26 net Sharpe")
    from scipy.stats import spearmanr
    dt = pd.read_csv(os.path.join(D.REPORTS, "decision_table.csv")).set_index("symbol")
    rows = []
    for sym in dt.index:
        dd = P.edges_for(sym)[0]; r = dd["nxt"].sum(axis=1); pre = dd["days"].year.values < 2019
        rows.append(dict(symbol=sym, vol_pre2019=r[pre].std(), cost=dt.cost_bp[sym], sharpe=dt.sharpe[sym]))
    x = pd.DataFrame(rows).set_index("symbol"); x["ratio"] = x.vol_pre2019 / x.cost
    rho, p = spearmanr(x.ratio, x.sharpe)
    print(x.sort_values("ratio", ascending=False).round(2).to_string())
    print(f"Spearman(vol/cost pre-2019, net Sharpe 2019-26) = {rho:+.2f}  p={p:.3f}")


if __name__ == "__main__":
    main()
