"""Is there ANY hour x decision-minute x price-bucket cell with a net edge that appears in a discovery period and REPLICATES in a later one?
(A strategy that works across regimes needs such a cell; a single noisy winner is not enough.)

Cell = (UTC hour of window open, decision-minute group, side type, price bucket).  One signal per (market, cell-minute-group): the first qualifying minute.
Net EV/contract = win - (price + 2c slippage) - Polymarket US taker fee.  Multiple testing: Benjamini-Hochberg FDR over all cells with n >= MIN_N in the
discovery period; replication = same sign, one-sided p < 0.05 in the validation period for cells that passed discovery.  The null (no real cells) is simulated by
shuffling the period labels of whole days.

    python scan_cells.py DISC_START DISC_END VAL_START VAL_END      (UTC dates, end exclusive)
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm

from trader.data import load_rows
from trader.fees import fee_per_contract

MIN_N = 40
SLIP = 0.02
MIN_GROUPS = {"m2-5": (2, 3, 4, 5), "m6-9": (6, 7, 8, 9), "m10-11": (10, 11), "m12-13": (12, 13)}
BUCKETS = {"dog05-15": (0.05, 0.15), "dog15-25": (0.15, 0.25), "dog25-35": (0.25, 0.35), "fav65-85": (0.65, 0.85), "fav85-97": (0.85, 0.97)}


def build(rows: pd.DataFrame) -> pd.DataFrame:
    r = rows[(rows.bid > 0) & (rows.ask < 1) & (rows.ask >= rows.bid) & ((rows.ask - rows.bid) <= 0.06 + 1e-9)].copy()
    mid = (r.bid + r.ask) / 2
    # evaluate BOTH sides at every row; side type is implied by the price (underdog / middle / favourite)
    y = r.assign(cost=r.ask, win=r.yes)
    n = r.assign(cost=1 - r.bid, win=1 - r.yes)
    d = pd.concat([y, n], ignore_index=True)
    d = d[(d.cost >= 0.05) & (d.cost <= 0.97)]
    d["hour"] = pd.to_datetime(d.open_ts, unit="s", utc=True).dt.hour
    d["day"] = (d.open_ts // 86400).astype(int)
    px = np.minimum(d.cost + SLIP, 0.99)
    fee = {c: fee_per_contract("Polymarket US", float(c)) for c in np.unique(px.round(3))}
    d["pnl"] = d.win - px - px.round(3).map(fee)
    return d


def signals(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for mg, mins in MIN_GROUPS.items():
        for bk, (lo, hi) in BUCKETS.items():
            s = d[d.minute.isin(mins) & (d.cost >= lo) & (d.cost < hi)]
            # one signal per market per (minute group, bucket, side): the first minute; the two sides of a market are different cells only through their price bucket
            s = s.sort_values(["open_ts", "minute"]).drop_duplicates(["ticker"])         # first qualifying minute; dog/fav buckets are side-unique
            s = s.assign(mg=mg, bk=bk)
            out.append(s[["ticker", "open_ts", "day", "hour", "mg", "bk", "pnl", "win", "cost"]])
    return pd.concat(out, ignore_index=True)


def cell_table(s: pd.DataFrame) -> pd.DataFrame:
    g = s.groupby(["hour", "mg", "bk"]).pnl
    t = pd.DataFrame({"n": g.size(), "ev": g.mean(), "sd": g.std()})
    t["t"] = t.ev / (t.sd / np.sqrt(t.n))
    t["p1"] = 1 - norm.cdf(t.t)                                          # one-sided: edge > 0
    return t


def bh(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    ok = p[o] <= thr
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    sel = np.zeros(n, bool)
    sel[o[:k]] = True
    return sel


def run(s: pd.DataFrame, d0, d1, v0, v1, label=""):
    ts = lambda x: int(pd.Timestamp(x, tz="UTC").timestamp())
    disc = s[(s.open_ts >= ts(d0)) & (s.open_ts < ts(d1))]
    val = s[(s.open_ts >= ts(v0)) & (s.open_ts < ts(v1))]
    A, B = cell_table(disc), cell_table(val)
    A = A[A.n >= MIN_N]
    J = A.join(B, lsuffix="_d", rsuffix="_v", how="left")
    sel = bh(J.p1_d.values)
    J["disc_sig"] = sel
    J["rep"] = (J.ev_v > 0) & (J.p1_v < 0.05) & (J.n_v >= 20)
    hits = J[J.disc_sig]
    both = J[J.disc_sig & J.rep]
    pos_both = ((J.t_d > 2) & (J.t_v > 2)).sum()
    return dict(label=label, cells=len(J), disc_fdr=int(sel.sum()), replicated=len(both), both_t2=int(pos_both),
                median_ev_d=J.ev_d.median() * 100, median_ev_v=J.ev_v.median() * 100,
                frac_pos_d=(J.ev_d > 0).mean(), frac_pos_v=(J.ev_v > 0).mean(), best=both.sort_values("t_v", ascending=False).head(8)), J


if __name__ == "__main__":
    d0, d1, v0, v1 = sys.argv[1:5]
    rows = load_rows()
    d = build(rows)
    s = signals(d)
    res, J = run(s, d0, d1, v0, v1, "real")
    print(f"discovery {d0}..{d1} | validation {v0}..{v1}")
    print(f"cells with n>={MIN_N} in discovery: {res['cells']}; FDR(10%) discoveries: {res['disc_fdr']}; replicated in validation (same sign, p<.05): {res['replicated']}; cells with t>2 in BOTH: {res['both_t2']}")
    print(f"median net EV per cell: discovery {res['median_ev_d']:+.2f}c, validation {res['median_ev_v']:+.2f}c | share of cells with EV>0: {res['frac_pos_d']:.2f} / {res['frac_pos_v']:.2f}")
    # null: shuffle whole days between the two periods, rebuild the same statistic
    ts = lambda x: int(pd.Timestamp(x, tz="UTC").timestamp())
    both_days = s[((s.open_ts >= ts(d0)) & (s.open_ts < ts(d1))) | ((s.open_ts >= ts(v0)) & (s.open_ts < ts(v1)))].copy()
    days = np.sort(both_days.day.unique())
    n_disc_days = both_days[both_days.open_ts < ts(d1)].day.nunique()
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(40):
        perm = rng.permutation(days)
        dset = set(perm[:n_disc_days])
        x = both_days.assign(open_ts=np.where(both_days.day.isin(dset), ts(d0), ts(v0)))
        r2, _ = run(x, d0, d1, v0, v1)
        nulls.append((r2["disc_fdr"], r2["replicated"], r2["both_t2"]))
    nulls = np.array(nulls)
    print(f"null (period labels shuffled, 40 draws): FDR discoveries mean {nulls[:,0].mean():.1f}, replicated mean {nulls[:,1].mean():.2f} (max {nulls[:,1].max()}), both-t>2 mean {nulls[:,2].mean():.1f} (max {nulls[:,2].max()})")
    if len(res["best"]):
        print("\ncells that passed discovery AND replicated:")
        print(res["best"][["n_d", "ev_d", "t_d", "n_v", "ev_v", "t_v"]].assign(ev_d=lambda x: x.ev_d * 100, ev_v=lambda x: x.ev_v * 100).round(2).to_string())
