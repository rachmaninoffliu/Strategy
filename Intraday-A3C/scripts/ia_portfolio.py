"""
Pooled / selection-honest evaluation of the surrogate agent (the product decision).

1. edges  : walk-forward ridge and GBM edge forecasts per product (cached in data/edges_<SYM>_<model>.npy); first test year 2019
2. series : daily net / gross P&L in bp of one lot, per product and configuration
3. pools  : equal-risk portfolios (weight 1/sigma_i, sigma_i = std of the 10:00 return-to-close) over named product groups - no selection involved
4. select : the honest product-picking test - each year Y from 2021 pick the top-N products by trailing net Sharpe over 2019..Y-1 and trade them in Y
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import ia_data as D, ia_study as S

CONFIGS = {"ridge_k2": ("ridge", 2.0), "gbm_k4": ("gbm", 4.0), "ens_k3": ("ens", 3.0)}


def edges_for(sym):
    dd = S.build_days(sym)
    out = {}
    for kind in ("ridge", "gbm"):
        f = os.path.join(D.DATA, f"edges_{sym}_{kind}.npy")
        if os.path.exists(f):
            out[kind] = np.load(f)
        else:
            out[kind] = S.walk_forward(dd, kind)
            np.save(f, out[kind])
    out["ens"] = (out["ridge"] + out["gbm"]) / 2
    return dd, out


def series(args):
    sym, half = args
    dd, E = edges_for(sym)
    cost = half + S.SLIP_BP
    res = {"days": dd["days"], "sigma": float(dd["y"][:, 5].std())}
    for name, (kind, k) in CONFIGS.items():
        net, gross, lots = S.policy_pnl(E[kind], dd["nxt"], cost, k)
        res[name] = (net, gross, lots)
    return sym, res


def sharpe(x):
    x = x[np.isfinite(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(252) if len(x) > 50 and x.std() > 0 else np.nan


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    sp = pd.read_csv(os.path.join(D.REPORTS, "spreads.csv")).set_index("symbol")
    syms = [s for s in D.ALL if os.path.exists(D.bars_path(s))]
    jobs = [(s, float(sp.spread_bp.get(s, 2.0)) / 2.0) for s in syms]
    R = {}
    with ProcessPoolExecutor(8) as ex:
        for sym, r in ex.map(series, jobs):
            R[sym] = r
    all_days = sorted(set().union(*[set(r["days"]) for r in R.values()]))
    idx = pd.DatetimeIndex(all_days)
    def frame(name, which):
        return pd.DataFrame({s: pd.Series(r[name][which], index=r["days"]) for s, r in R.items()}).reindex(idx)
    sig = pd.Series({s: r["sigma"] for s, r in R.items()})
    groups = {"all 22": syms, "index ETFs": D.CANDIDATES["index ETF"], "sector/industry ETFs": D.CANDIDATES["sector / industry ETF"], "leveraged ETFs": D.CANDIDATES["leveraged ETF"],
              "mega-cap stocks": D.CANDIDATES["mega-cap stock"], "Nasdaq complex (QQQ XLK SMH NVDA TQQQ SOXL)": ["QQQ", "XLK", "SMH", "NVDA", "TQQQ", "SOXL"], "QQQ only": ["QQQ"], "SPY only": ["SPY"]}
    rows, pools = [], {}
    for cname in CONFIGS:
        net, gross = frame(cname, 0), frame(cname, 1)
        for gname, members in groups.items():
            w = 1.0 / sig[members]
            pn = (net[members] * w).sum(axis=1, min_count=1) / w.sum()
            pg = (gross[members] * w).sum(axis=1, min_count=1) / w.sum()
            pools[(cname, gname)] = pn
            yrs = pn.index.year
            for per, m in (("2019-26", yrs >= 2019), ("2019-22", (yrs >= 2019) & (yrs <= 2022)), ("2023-26", yrs >= 2023)):
                x = pn[m].dropna()
                rows.append(dict(config=cname, group=gname, period=per, sharpe=sharpe(pn[m].values), gross_sharpe=sharpe(pg[m].values), t=x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 50 else np.nan, n=len(x)))
    pd.DataFrame(rows).to_csv(os.path.join(D.REPORTS, "pools.csv"), index=False)
    # honest product selection: trailing-Sharpe top-N, then trade the next year
    sel_rows, sel_series = [], {}
    for cname in CONFIGS:
        net = frame(cname, 0)
        for N in (1, 3, 5):
            parts = []
            for Y in range(2021, idx.year.max() + 1):
                past = net[(net.index.year >= 2019) & (net.index.year < Y)]
                top = past.apply(sharpe).sort_values(ascending=False).head(N).index.tolist()
                cur = net[net.index.year == Y][top]
                w = 1.0 / sig[top]
                parts.append((cur * w).sum(axis=1, min_count=1) / w.sum())
            s = pd.concat(parts)
            sel_series[(cname, N)] = s
            sel_rows.append(dict(config=cname, top_n=N, sharpe=sharpe(s.values), t=s.mean() / s.std() * np.sqrt(len(s)), years="2021-26", n=len(s)))
    pd.DataFrame(sel_rows).to_csv(os.path.join(D.REPORTS, "selection.csv"), index=False)
    pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in pools.items()}).to_csv(os.path.join(D.REPORTS, "pool_daily_net_bp.csv"))
    print("done")
