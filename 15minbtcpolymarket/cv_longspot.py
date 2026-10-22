"""Mimic the contract with historical BTC spot (Coinbase 1-minute candles) over as many years as we have, and ask:
 is the 22-23 UTC late-underdog edge a property of BTC's behaviour, of how Kalshi prices, or neither?

World:   for every 15-minute window (strike K = OHLC4 of the minute before the open, final F = OHLC4 of the last minute, YES iff F >= K),
         decision state at minute m: z = ln(S_m / K) / (sigma_m * sqrt(15 - m)),  sigma = EWMA 1-minute vol (half-life 90 min).
Pricing: g(z | minute) = isotonic fit of the REAL Kalshi mid on z, fitted on quotes BEFORE the regime start only (so it knows nothing about
         the late-August behaviour).  The synthetic underdog ask is g_dog + half spread; R2 mechanics then run over every window since 2024.
What this can answer: whether BTC outcomes beat the market's *typical* pricing in those hours (BTC-dynamics surprise).
What it cannot: whether Kalshi quotes in 2024-25 looked like g (no quotes exist before 2025-12-15).
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import cv_family as cvf
from trader.data import load_rows
from trader.fees import fee_per_contract

DATA = Path(__file__).parent / "data"
CUT = pd.Timestamp("2026-08-24", tz="UTC").timestamp()
SLIP, HALF_SPREAD = 0.02, 0.005
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def load_spot_all() -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in sorted(DATA.glob("cb_1m_long*.csv"))] + [pd.read_csv(DATA / "cb_1m.csv")]
    s = pd.concat(parts)
    s["ts"] = s.ts.astype("int64")
    s = s.drop_duplicates("ts").sort_values("ts").set_index("ts")
    grid = np.arange(s.index.min(), s.index.max() + 60, 60)
    s = s.reindex(grid)
    s["close"] = s.close.ffill(limit=5)
    for c in ("open", "high", "low"):
        s[c] = s[c].fillna(s.close)
    s["ohlc4"] = (s.open + s.high + s.low + s.close) / 4
    lr = np.log(s.close).diff()
    s["sig"] = np.sqrt((lr ** 2).ewm(halflife=90, min_periods=60).mean())
    return s


def build_windows(s: pd.DataFrame, minutes=(10, 11, 12, 13)) -> pd.DataFrame:
    g0 = int(s.index[0])
    first = (g0 // 900 + 2) * 900
    last = (int(s.index[-1]) // 900 - 2) * 900
    t0 = np.arange(first, last, 900, dtype="int64")
    ix = lambda t: ((t - g0) // 60).astype(int)
    K, F = s.ohlc4.values[ix(t0 - 60)], s.ohlc4.values[ix(t0 + 840)]
    close, sig = s.close.values, s.sig.values
    out = []
    for m in minutes:
        j = ix(t0 + 60 * m - 60)
        out.append(pd.DataFrame({"open_ts": t0, "minute": m, "K": K, "F": F, "S": close[j], "sig": sig[j]}))
    w = pd.concat(out, ignore_index=True)
    w = w.dropna()
    w["z"] = np.log(w.S / w.K) / (w.sig * np.sqrt(15 - w.minute))
    w["yes_proxy"] = (w.F >= w.K).astype(int)
    dt = pd.to_datetime(w.open_ts, unit="s", utc=True)
    w["hour"] = dt.dt.hour
    return w.sort_values(["open_ts", "minute"]).reset_index(drop=True)


def main():
    s = load_spot_all()
    print(f"spot minutes {len(s):,}  {pd.to_datetime(s.index[0], unit='s', utc=True):%Y-%m-%d} -> {pd.to_datetime(s.index[-1], unit='s', utc=True):%Y-%m-%d}")
    w = build_windows(s)
    print(f"windows x minutes: {len(w):,}  ({w.open_ts.nunique():,} windows)")

    # ── A. fidelity of the spot-reconstructed outcome vs the real Kalshi result ─────
    q = load_rows()
    q = q[q.minute.isin([10, 11, 12, 13])]
    q = q[(q.bid > 0) & (q.ask < 1) & (q.ask >= q.bid) & ((q.ask - q.bid) <= 0.06 + 1e-9)].copy()
    q["mid"] = (q.bid + q.ask) / 2
    m = q.merge(w, on=["open_ts", "minute"], how="inner")
    agree = (m.yes == m.yes_proxy)
    near = (m.z.abs() < 1.0)
    print("\n=== A. Spot-reconstructed outcome vs real Kalshi result (overlap, minutes 10-13) ===")
    print(f"  rows {len(m):,}: agreement {agree.mean() * 100:.1f}% overall; {agree[near].mean() * 100:.1f}% when |z|<1 (the underdog-relevant region, n={near.sum():,})")
    mid_dog = m.mid < 0.5
    dog_win_real = np.where(mid_dog, m.yes, 1 - m.yes)
    dog_win_prox = np.where(mid_dog, m.yes_proxy, 1 - m.yes_proxy)
    band = (np.minimum(m.mid, 1 - m.mid) >= 0.15) & (np.minimum(m.mid, 1 - m.mid) <= 0.35)
    print(f"  underdogs priced 15-35c: real win rate {dog_win_real[band].mean():.3f} vs proxy win rate {dog_win_prox[band].mean():.3f}  (mean price {np.minimum(m.mid, 1 - m.mid)[band].mean():.3f})")

    # ── B. pricing function g fitted on pre-regime quotes ───────────────────────
    g = {}
    for mn in (10, 11, 12, 13):
        tr = m[(m.minute == mn) & (m.open_ts < CUT)]
        g[mn] = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.005, y_max=0.995).fit(tr.z.values, tr.mid.values)
    m["g"] = [0.5] * len(m)
    for mn in g:
        i = m.minute == mn
        m.loc[i, "g"] = g[mn].predict(m.loc[i, "z"].values)
    res = (m.mid - m.g)
    print("\n=== B. Does g (fitted on quotes before 2026-08-24) price the real market consistently? ===")
    print(f"  MAE |mid - g| = {res.abs().mean() * 100:.2f}c overall; corr(mid, g) = {np.corrcoef(m.mid, m.g)[0, 1]:.3f}")
    m["per"] = np.where(m.open_ts < pd.Timestamp("2026-06-01", tz="UTC").timestamp(), "Dec-May", np.where(m.open_ts < CUT, "Jun-Aug23", "Aug24-now"))
    t = m.assign(res=res, hr2=m.hour.isin([22, 23]))
    print("  mean residual (mid - g, in cents) by period x [22-23 UTC | other hours]  (underdog-side residual sign: negative = market cheaper than g):")
    print(t.groupby(["per", "hr2"]).res.agg(["size", lambda x: x.mean() * 100, lambda x: x.std() * 100]).rename(columns={"<lambda_0>": "mean_c", "<lambda_1>": "sd_c"}).round(2).to_string())

    # ── C. decomposition of the real R2 edge: market pricing vs BTC outcome surprise ───────────
    r = m[(m.minute.isin([12, 13])) & (m.hour.isin([22, 23]))].copy()
    r["dog_yes"] = r.mid < 0.5
    r["cost"] = np.where(r.dog_yes, r.ask, 1 - r.bid)
    r["g_dog"] = np.where(r.dog_yes, r.g, 1 - r.g)
    r["win"] = np.where(r.dog_yes, r.yes, 1 - r.yes)
    r["win_p"] = np.where(r.dog_yes, r.yes_proxy, 1 - r.yes_proxy)
    r = r[(r.cost >= 0.15) & (r.cost <= 0.35)].sort_values(["open_ts", "minute"]).drop_duplicates("ticker")
    print("\n=== C. R2 signals (22-23 UTC, min 12/13, 15-35c), real quotes decomposed ===")
    print("   edge before costs = win - ask = (win - g_dog)  [BTC-outcome surprise vs typical pricing]  -  (ask - g_dog)  [how much cheaper than typical pricing the market quoted]")
    d = r.groupby("per").agg(n=("win", "size"), ask=("cost", "mean"), g_dog=("g_dog", "mean"), win_real=("win", "mean"), win_proxy=("win_p", "mean"))
    d["win_minus_g_c"] = (d.win_real - d.g_dog) * 100
    d["ask_minus_g_c"] = (d.ask - d.g_dog) * 100
    d["edge_c"] = (d.win_real - d.ask) * 100
    print(d.round(3).to_string())

    # ── D. synthetic world over the full spot history ───────────────────────────
    w2 = w.copy()
    w2["g"] = 0.5
    for mn in g:
        i = w2.minute == mn
        w2.loc[i, "g"] = g[mn].predict(w2.loc[i, "z"].values)
    dog_yes = w2.g < 0.5
    w2["cost"] = np.minimum(np.where(dog_yes, w2.g, 1 - w2.g) + HALF_SPREAD, 0.99).round(4)
    w2["win"] = np.where(dog_yes, w2.yes_proxy, 1 - w2.yes_proxy)
    px = np.minimum(w2.cost + SLIP, 0.99)
    fee = {c: fee_per_contract("Polymarket US", c) for c in np.unique(px.round(4))}
    w2["pnl"] = w2.win - px - px.round(4).map(fee)
    w2["ticker"] = w2.open_ts
    dt = pd.to_datetime(w2.open_ts, unit="s", utc=True)
    w2["ym"] = dt.dt.strftime("%Y-%m")
    w2["month"] = (dt.dt.year - 2024) * 12 + dt.dt.month - 1
    w2 = w2[(w2.cost >= 0.03)].reset_index(drop=True)

    sig2 = w2[w2.minute.isin([12, 13]) & w2.hour.isin([22, 23]) & (w2.cost >= 0.15) & (w2.cost <= 0.35)].drop_duplicates("ticker")
    mon = sig2.groupby("ym").agg(n=("pnl", "size"), win=("win", "mean"), price=("cost", "mean"), ev_c=("pnl", lambda x: x.mean() * 100))
    real = cvf.build_rows()
    rs = cvf.combo_signals(real, (12, 13), (0.15, 0.35))
    rs = rs[rs.hour.isin([22, 23])]
    rs["ym"] = pd.to_datetime(rs.open_ts, unit="s", utc=True).dt.strftime("%Y-%m")
    realm = rs.groupby("ym").agg(real_n=("pnl", "size"), real_ev_c=("pnl", lambda x: x.mean() * 100))
    tab = mon.join(realm, how="left")
    print("\n=== D. Synthetic R2 (price = g_dog + 0.5c, same costs) month by month, vs the REAL R2 where quotes exist ===")
    print(tab.round(2).T.to_string())
    ov = tab.dropna()
    if len(ov) > 4:
        print(f"  correlation of monthly EV, synthetic vs real (overlap {len(ov)} months): {np.corrcoef(ov.ev_c, ov.real_ev_c)[0, 1]:+.2f}")
    pre = sig2[sig2.open_ts < pd.Timestamp("2025-12-15", tz="UTC").timestamp()]
    mid_p = sig2[(sig2.open_ts >= pd.Timestamp("2025-12-15", tz="UTC").timestamp()) & (sig2.open_ts < CUT)]
    post = sig2[sig2.open_ts >= CUT]
    for lab, x in (("2024-01 .. 2025-12-14 (no Kalshi quotes exist)", pre), ("2025-12-15 .. 2026-08-23", mid_p), ("2026-08-24 .. now", post)):
        if len(x) > 5:
            print(f"  synthetic R2 {lab:48s}: n={len(x):5d} win={x.win.mean():.3f} price={x.cost.mean():.3f} EV={x.pnl.mean() * 100:+6.2f}c  t={x.pnl.mean() / x.pnl.std() * np.sqrt(len(x)):+5.2f}")

    # ── E. how often do 6-week stretches this good occur in 2.7 years of BTC (any 2-hour window)? ───────
    print("\n=== E. Base rate of 'hot' stretches in the synthetic world (every 2-hour UTC window, rolling 42-day windows, min 90 signals) ===")
    s2 = w2[w2.minute.isin([12, 13]) & (w2.cost >= 0.15) & (w2.cost <= 0.35)].drop_duplicates("ticker").copy()
    s2["day"] = (s2.open_ts // 86400).astype(int)
    d0, d1 = s2.day.min(), s2.day.max()
    rows_ = []
    for h in range(24):
        x = s2[s2.hour.isin([h, (h + 1) % 24])]
        by = x.groupby("day").pnl.agg(["size", "sum", lambda v: (v ** 2).sum()])
        by.columns = ["n", "S", "S2"]
        by = by.reindex(range(d0, d1 + 1), fill_value=0)
        cn, cs, cs2 = by.n.cumsum().values, by.S.cumsum().values, by.S2.cumsum().values
        for e in range(42, len(by), 7):
            n_, S_, S2_ = cn[e] - cn[e - 42], cs[e] - cs[e - 42], cs2[e] - cs2[e - 42]
            if n_ >= 90:
                mu = S_ / n_
                var = (S2_ - n_ * mu ** 2) / (n_ - 1)
                rows_.append((h, d0 + e, n_, mu * 100, mu / np.sqrt(var / n_)))
    E = pd.DataFrame(rows_, columns=["h", "day", "n", "ev_c", "t"])
    E["end"] = pd.to_datetime(E.day * 86400, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    print(f"  windows evaluated: {len(E):,} (24 hour-windows x {E.day.nunique()} weekly steps)")
    for thr in (5, 10, 15, 20):
        print(f"  stretches with EV >= +{thr}c: {(E.ev_c >= thr).sum()} ({(E.ev_c >= thr).mean() * 100:.2f}%); with t>=3: {((E.ev_c >= thr) & (E.t >= 3)).sum()}")
    best = E.sort_values("ev_c", ascending=False).head(12)[["h", "end", "n", "ev_c", "t"]]
    print("  best stretches found anywhere in the history (hour-window starting at h UTC, 42 days ending 'end'):")
    print(best.round(2).to_string(index=False))
    cur = E[(E.h == 22)].tail(8)[["end", "n", "ev_c", "t"]]
    print("  the 22-23 UTC window, latest rolling 42-day stretches:")
    print(cur.round(2).to_string(index=False))
    E.to_csv(DATA / "synthetic_hot_stretches.csv", index=False)

    # ── F. walk-forward CV of the strategy family inside the synthetic world ───────────────
    print("\n=== F. Anchored walk-forward CV of the strategy family on the synthetic 2.7-year world (train: all earlier months; test: next month) ===")
    sy = w2[w2.minute.isin([10, 11, 12, 13])].copy()
    sy = sy.sort_values(["open_ts", "minute"]).reset_index(drop=True)
    sy["ticker"] = sy.open_ts
    M = int(sy.month.max()) + 1
    cfgs, C = cvf.stat_tensor(sy, sy.month.values, M)
    first = 6
    wf, tot = cvf.walk_forward(cfgs, C, first_test=first, min_n=60)
    wf["month"] = wf.month.map(lambda k: (pd.Timestamp("2024-01-01") + pd.DateOffset(months=int(k))).strftime("%Y-%m"))
    print(wf[["month", "cfg", "train_n", "train_ev_c", "train_t", "test_n", "test_ev_c"]].round(2).to_string(index=False))
    n, S, S2, W = tot
    print(f"  POOLED OOF: n={int(n)} EV={S / n * 100:+.2f}c t={cvf.tstat(n, S, S2):+.2f}; folds >0: {(wf.test_pnl > 0).sum()}/{len(wf)}")
    fixed = cvf.stat_tensor  # noqa
    jr2 = np.where((cfgs.hours == "22-23") & (cfgs.minutes == "12/13") & (cfgs.band == "0.15-0.35"))[0][0]
    te = C[jr2, :, first:].sum(1)
    print(f"  R2 fixed over the same months: n={int(te[0])} EV={te[1] / te[0] * 100:+.2f}c t={cvf.tstat(te[0], te[1], te[2]):+.2f}")
    sel = wf.cfg.value_counts().head(5)
    print("  configs selected most often:", "; ".join(f"{k} x{v}" for k, v in sel.items()))


if __name__ == "__main__":
    main()
