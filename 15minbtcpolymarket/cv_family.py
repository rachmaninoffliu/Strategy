"""Cross-validation of the *selection procedure* behind R1/R2 on the real Kalshi quotes (Dec 15 2025 -> now).

Strategy family F = {UTC window of 1-4 hours (24 starts)} x {decision-minute set} x {underdog price band}  (2,400 configs).
Per-signal P&L (flat 1 contract) = win - (cost + 2c slippage) - Polymarket-US taker fee, one signal per market.

 1. Anchored walk-forward CV : for each test month, pick the config with the best TRAIN t-stat (only data that settled before the
                               month starts), trade it untouched in the test month.  Also: rolling 2-month training window,
                               and a random-config null.
 2. CSCV / PBO               : combinatorially symmetric CV over 16 time blocks (Bailey-Borwein-Lopez de Prado-Zhu):
                               how often does the in-sample winner land below the median out of sample?
 3. Deflated Sharpe ratio    : is R2's Sharpe still significant after accounting for the number of configs that were tried?
 4. Weekly profile + change-point of R2 (fixed rule, shown for context - NOT out of sample).
"""
from __future__ import annotations

import itertools
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm

from trader.data import load_rows
from trader.fees import fee_per_contract

SLIP = 0.02
MIN_SETS = [(12,), (13,), (12, 13), (11, 12, 13), (10, 11, 12, 13)]
BANDS = [(0.05, 0.30), (0.15, 0.35), (0.05, 0.35), (0.10, 0.40), (0.20, 0.45)]
LENGTHS = (1, 2, 3, 4)
T0 = pd.Timestamp("2025-12-01", tz="UTC")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


# ── data ──────────────────────────────────────────────────────────────────────
def build_rows() -> pd.DataFrame:
    r = load_rows()
    r = r[r.minute.between(10, 13)].copy()
    ok = (r.bid > 0) & (r.ask < 1) & (r.ask >= r.bid) & ((r.ask - r.bid) <= 0.06 + 1e-9)
    r = r[ok].copy()
    mid = (r.bid + r.ask) / 2
    yd = mid < 0.5
    r["cost"] = np.where(yd, r.ask, 1 - r.bid).round(6)
    r["win"] = np.where(yd, r.yes, 1 - r.yes)
    px = np.minimum(r.cost + SLIP, 0.99)
    fee = {c: fee_per_contract("Polymarket US", c) for c in np.unique(px.round(4))}
    r["pnl"] = r.win - px - px.round(4).map(fee)
    dt = pd.to_datetime(r.open_ts, unit="s", utc=True)
    r["hour"] = dt.dt.hour
    r["month"] = (dt.dt.year - T0.year) * 12 + dt.dt.month - T0.month          # Dec-25 = 0, Jan-26 = 1, ...
    return r.sort_values(["open_ts", "minute"]).reset_index(drop=True)


def combo_signals(r: pd.DataFrame, minutes, band) -> pd.DataFrame:
    s = r[r.minute.isin(minutes) & (r.cost >= band[0] - 1e-9) & (r.cost <= band[1] + 1e-9)]
    return s.drop_duplicates("ticker")                       # rows are sorted by (open_ts, minute): first qualifying minute


def stat_tensor(r: pd.DataFrame, block: np.ndarray, B: int):
    """Returns (configs, Cfg[Nconf, 4, B]) with stats n, sum(pnl), sum(pnl^2), wins per config and block."""
    configs, mats = [], []
    for ms, band in itertools.product(MIN_SETS, BANDS):
        s = combo_signals(r, ms, band)
        b = block[s.index.values]
        key = b * 24 + s.hour.values
        T = np.zeros((4, B * 24))
        for i, w in enumerate((np.ones(len(s)), s.pnl.values, s.pnl.values ** 2, s.win.values.astype(float))):
            T[i] = np.bincount(key, weights=w, minlength=B * 24)
        T = T.reshape(4, B, 24)
        for L in LENGTHS:
            for h in range(24):
                hrs = [(h + i) % 24 for i in range(L)]
                mats.append(T[:, :, hrs].sum(2))
                configs.append(dict(hours=f"{h:02d}-{(h + L - 1) % 24:02d}", L=L, h=h, minutes="/".join(map(str, ms)), band=f"{band[0]:.2f}-{band[1]:.2f}"))
    return pd.DataFrame(configs), np.stack(mats)


def tstat(n, s, s2):
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / n
        var = (s2 - n * mean ** 2) / (n - 1)
        return np.where((n > 2) & (var > 0), mean / np.sqrt(var / n), np.nan)


def label(cfg: pd.Series) -> str:
    return f"{cfg.hours}h min{cfg.minutes} {cfg.band}"


# ── 1. walk-forward ───────────────────────────────────────────────────────────
def walk_forward(cfgs, C, first_test, train_len=None, min_n=40, metric="t", rng=None, random_pick=False):
    M = C.shape[2]
    out, tot = [], np.zeros(4)
    for k in range(first_test, M):
        lo = 0 if train_len is None else max(0, k - train_len)
        tr, te = C[:, :, lo:k].sum(2), C[:, :, k]
        t = tstat(tr[:, 0], tr[:, 1], tr[:, 2])
        score = t if metric == "t" else np.where(tr[:, 0] > 0, tr[:, 1] / np.maximum(tr[:, 0], 1), np.nan)
        elig = (tr[:, 0] >= min_n) & np.isfinite(score)
        if not elig.any():
            continue
        if random_pick:
            j = rng.choice(np.where(elig)[0])
        else:
            j = np.where(elig, score, -np.inf).argmax()
        n, s, s2, w = te[j]
        out.append(dict(month=k, cfg=label(cfgs.iloc[j]), train_n=int(tr[j, 0]), train_ev_c=tr[j, 1] / tr[j, 0] * 100, train_t=t[j],
                        test_n=int(n), test_ev_c=(s / n * 100) if n else np.nan, test_win=(w / n) if n else np.nan, test_pnl=s))
        tot += te[j]
    return pd.DataFrame(out), tot


def month_name(k):
    return (T0 + pd.DateOffset(months=k)).strftime("%b-%y")


def main():
    r = build_rows()
    print(f"rows {len(r):,}  | markets {r.ticker.nunique():,} | {pd.to_datetime(r.open_ts.min(), unit='s', utc=True):%Y-%m-%d} -> {pd.to_datetime(r.open_ts.max(), unit='s', utc=True):%Y-%m-%d}")
    M = int(r.month.max()) + 1
    cfgs, C = stat_tensor(r, r.month.values, M)
    print(f"strategy family: {len(cfgs):,} configs\n")

    # fixed R2 / R1 as members of the family
    def find(hours, minutes, band):
        m = (cfgs.hours == hours) & (cfgs.minutes == minutes) & (cfgs.band == band)
        return int(np.where(m)[0][0])
    j_r2 = find("22-23", "12/13", "0.15-0.35")
    j_r1 = find("22-23", "12/13", "0.05-0.30")

    print("=" * 110)
    print("1. ANCHORED WALK-FORWARD CV  (select best TRAIN t-stat with n>=40; trade it untouched in the next calendar month)")
    print("=" * 110)
    wf, tot = walk_forward(cfgs, C, first_test=3)
    wf["month"] = wf.month.map(month_name)
    print(wf.round(2).to_string(index=False))
    n, s, s2, w = tot
    print(f"POOLED out-of-fold: n={int(n)}  EV={s / n * 100:+.2f}c/contract  win={w / n:.3f}  t={tstat(n, s, s2):+.2f}  folds with profit: {(wf.test_pnl > 0).sum()}/{len(wf)}")

    print("\nBenchmarks over the same test months (Mar-26 .. Oct-26):")
    first = 3
    for nm, j in (("R2 fixed (hindsight)", j_r2), ("R1 fixed (hindsight)", j_r1)):
        te = C[j, :, first:].sum(1)
        print(f"  {nm:24s} n={int(te[0])} EV={te[1] / te[0] * 100:+.2f}c  t={tstat(te[0], te[1], te[2]):+.2f}")
    for lab, kw in (("rolling 2-month train window", dict(train_len=2)), ("rolling 3-month train window", dict(train_len=3)),
                    ("select by mean EV (n>=60)", dict(metric="mean", min_n=60))):
        wf2, tt = walk_forward(cfgs, C, first, **kw)
        print(f"  walk-forward, {lab:30s} n={int(tt[0])} EV={tt[1] / tt[0] * 100:+.2f}c  t={tstat(tt[0], tt[1], tt[2]):+.2f}  profitable folds {(wf2.test_pnl > 0).sum()}/{len(wf2)}")
    rng = np.random.default_rng(1)
    evs = []
    for _ in range(2000):
        _, tt = walk_forward(cfgs, C, first, random_pick=True, rng=rng)
        evs.append(tt[1] / tt[0] * 100 if tt[0] else np.nan)
    evs = np.array(evs)
    print(f"  random-config null (same folds): EV median {np.nanmedian(evs):+.2f}c, 5th {np.nanpercentile(evs, 5):+.2f}c, 95th {np.nanpercentile(evs, 95):+.2f}c")
    print(f"  walk-forward EV percentile within null: {(evs < s / n * 100).mean() * 100:.0f}%")

    # ── 2. CSCV / PBO ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("2. CSCV / PBO  (16 contiguous time blocks, all C(16,8)=12,870 IS/OOS splits; metric = t-stat of per-signal P&L)")
    print("=" * 110)
    tmin, tmax = r.open_ts.min(), r.open_ts.max() + 1
    B = 16
    blk = np.minimum(((r.open_ts.values - tmin) / (tmax - tmin) * B).astype(int), B - 1)
    cfgs_b, Cb = stat_tensor(r, blk, B)
    keep = Cb[:, 0, :].sum(1) >= 100
    for lab, mask in (("all configs", keep), ("configs containing 22-23 UTC", keep & cfgs_b.hours.str.contains("22|23").values)):
        X = Cb[mask]
        meta = cfgs_b[mask].reset_index(drop=True)
        lam, ev_is, ev_oos, picks = [], [], [], []
        for J in itertools.combinations(range(B), B // 2):
            Jn = [b for b in range(B) if b not in J]
            a, o = X[:, :, list(J)].sum(2), X[:, :, Jn].sum(2)
            ta = np.where(a[:, 0] >= 30, tstat(a[:, 0], a[:, 1], a[:, 2]), -np.inf)
            ta = np.nan_to_num(ta, nan=-np.inf)
            j = int(ta.argmax())
            to = np.nan_to_num(np.where(o[:, 0] >= 10, tstat(o[:, 0], o[:, 1], o[:, 2]), 0.0), nan=0.0)
            om = (to < to[j]).mean() + 0.5 * (to == to[j]).mean()
            om = min(max(om, 1e-3), 1 - 1e-3)
            lam.append(np.log(om / (1 - om)))
            ev_is.append(a[j, 1] / a[j, 0] * 100)
            ev_oos.append(o[j, 1] / max(o[j, 0], 1) * 100)
            picks.append(j)
        lam = np.array(lam)
        print(f"\n[{lab}: {len(meta)} configs]  PBO = P(IS-best ranks below OOS median) = {(lam <= 0).mean() * 100:.1f}%   median logit = {np.median(lam):+.2f}")
        print(f"   IS-best: mean IS EV {np.mean(ev_is):+.2f}c -> mean OOS EV {np.mean(ev_oos):+.2f}c   P(OOS EV > 0) = {(np.array(ev_oos) > 0).mean() * 100:.0f}%   slope OOS~IS = {np.polyfit(ev_is, ev_oos, 1)[0]:.2f}")
        top = pd.Series(picks).value_counts().head(5)
        print("   most frequently selected:", "; ".join(f"{label(meta.iloc[i])} ({c / len(picks) * 100:.0f}%)" for i, c in top.items()))

    # ── 3. deflated Sharpe ────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("3. DEFLATED SHARPE RATIO for R2 (per-signal Sharpe; trials = size of the family searched)")
    print("=" * 110)
    s2sig = combo_signals(r, (12, 13), (0.15, 0.35))
    s2sig = s2sig[s2sig.hour.isin([22, 23])]
    ts_regime = pd.Timestamp("2026-08-24", tz="UTC").timestamp()
    cfg_n = Cb[:, 0, :].sum(1)
    cfg_sr = (Cb[:, 1, :].sum(1) / np.maximum(cfg_n, 1)) / np.sqrt(np.maximum((Cb[:, 2, :].sum(1) - Cb[:, 1, :].sum(1) ** 2 / np.maximum(cfg_n, 1)) / np.maximum(cfg_n - 1, 1), 1e-12))
    V = np.nanvar(cfg_sr[cfg_n >= 100])
    g = 0.5772156649
    for lab, x in (("all 10 months", s2sig), ("since Aug 24 only", s2sig[s2sig.open_ts >= ts_regime])):
        p = x.pnl.values
        T = len(p)
        sr = p.mean() / p.std(ddof=1)
        sk = pd.Series(p).skew()
        ku = pd.Series(p).kurt() + 3
        for N in (20, 200, 2400):
            sr0 = np.sqrt(V) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e)))
            dsr = norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
            print(f"  {lab:18s} T={T:4d} per-signal SR={sr:+.3f}  skew={sk:+.2f}  trials N={N:4d}  expected-max SR0={sr0:.3f}  DSR (P true SR>0)={dsr:.3f}")

    # ── 4. weekly profile of R2 (context) ─────────────────────────────────────
    print("\n" + "=" * 110)
    print("4. R2 fixed rule: weekly profile and change-point (context, not out-of-sample)")
    print("=" * 110)
    x = s2sig.copy()
    x["wk"] = pd.to_datetime(x.open_ts, unit="s", utc=True).dt.to_period("W-SUN").dt.start_time
    wk = x.groupby("wk").agg(n=("pnl", "size"), win=("win", "mean"), ev_c=("pnl", lambda v: v.mean() * 100))
    wk["cum_c"] = (x.groupby("wk").pnl.sum().cumsum() * 100)
    print(wk.round(2).T.to_string())
    p = x.sort_values("open_ts").pnl.values
    cs = np.cumsum(p - p.mean())
    k = int(np.abs(cs).argmax())
    d_open = pd.to_datetime(x.sort_values("open_ts").open_ts.values[k], unit="s", utc=True)
    a, b = p[: k + 1], p[k + 1:]
    welch = (b.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"\nCUSUM change-point at {d_open:%Y-%m-%d}: before n={len(a)} EV={a.mean() * 100:+.2f}c | after n={len(b)} EV={b.mean() * 100:+.2f}c | Welch t={welch:+.2f}")
    rng = np.random.default_rng(0)
    mx = []
    for _ in range(3000):
        q = rng.permutation(p)
        c2 = np.abs(np.cumsum(q - q.mean())).max()
        mx.append(c2)
    print(f"CUSUM max={np.abs(cs).max():.2f}; permutation P(no change-point has CUSUM this large)={(np.array(mx) >= np.abs(cs).max()).mean():.4f} (iid null; a single regime shift is what this tests)")


if __name__ == "__main__":
    main()
