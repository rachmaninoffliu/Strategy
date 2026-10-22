"""
Alpha Syndicate — HIGHER CAGR + HIGHER SHARPE THAN QQQ, NO LEVERAGE

TARGET: CAGR > 20.20% AND Sharpe > 0.94, with gross exposure <= 1.0 always.

THE ONLY MECHANISM THAT CAN DO THIS (QR):
  Unlevered, you cannot manufacture return by sizing up. Sharpe is
  scale-invariant and CAGR is capped by being <=100% invested. So the ONLY
  route to beating QQQ on BOTH dimensions without leverage is asymmetric
  timing: stay fully invested through most of the upside, and be out during
  enough of the downside that avoided losses exceed missed gains.

  Compounding makes this possible in principle: avoiding a -35% drawdown is
  worth more than missing an equivalent stretch of upside, because losses
  compound against you geometrically.

  This is a genuine mechanism, not a sizing trick. Whether it WORKS out of
  sample is the open question.

DESIGN — "vol cap", not "vol target":
  w = min(vol_target / realised_vol_{t-1}, 1.0)
  With a HIGH target, w = 1.0 almost always -- the book is fully invested and
  only de-risks when volatility genuinely spikes. This is the unlevered
  analogue of Moreira-Muir. No borrowing at any point.

HINDSIGHT CONTROL (Valid insisted):
  XLK buy & hold is included and LABELLED AS HINDSIGHT. It was the best sleeve
  of this window (CAGR 23.86%, Sharpe 0.99) and already beats QQQ on both --
  but only because we looked at the answer first. The earlier rules-based
  sector-momentum test scored 0.74, i.e. NO RULE reached XLK. It is in this
  table purely as a yardstick for how much "winning" is available from pure
  hindsight, so any genuine result can be judged against it.
  IT IS NOT A RECOMMENDATION.

ATTRIBUTION TEST:
  If a strategy passes, we check WHICH YEARS produced the edge. An edge that
  comes entirely from dodging one event (e.g. March 2020) is fragile and does
  not generalise. Reported per-year vs QQQ.

DSR n_trials = 32. This search is expensive and the cost is real: QQQ's own
DSR has fallen from 0.869 -> 0.836 across this programme purely because we
keep testing. Every row below is penalised accordingly.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
import talib
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRADING_DAYS = 252
N_TRIALS_DSR = 32
COST_BPS     = 3.0
VOL_LB       = 21
TREND_SMA    = 200
MAX_GROSS    = 1.00
BENCH        = "QQQ"

DATA   = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data/mhdm_etf_bars.parquet"
OUTDIR = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
os.makedirs(OUTDIR, exist_ok=True)


def load():
    p = pd.read_parquet(DATA)
    p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None).dt.normalize()
    px = {c: p.pivot_table(index="timestamp", columns="symbol", values=c)
              .sort_index().ffill() for c in ("open", "high", "low", "close")}
    idx = px["close"].dropna().index
    return {k: v.reindex(idx) for k, v in px.items()}


def sharpe_diff_test(ra, rb):
    df = pd.concat([pd.Series(ra), pd.Series(rb)], axis=1).dropna()
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    T = len(a)
    if T < 30:
        return np.nan, np.nan
    sa, sb = a.mean()/a.std(ddof=1), b.mean()/b.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1]
    th = (1.0/T) * (2*(1-rho) + 0.5*(sa**2 + sb**2 - 2*sa*sb*rho**2))
    if th <= 0:
        return np.nan, np.nan
    z = (sa-sb)/math.sqrt(th)
    return float(z), float(2*(1-norm.cdf(abs(z))))


def deflated_sharpe(r, n):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    T = len(r)
    if T < 30 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean()/r.std(ddof=1)
    g3 = float(pd.Series(r).skew()); g4 = float(pd.Series(r).kurtosis())+3.0
    gm, e = 0.5772156649, math.e
    sd = math.sqrt(max((1-g3*sr+(g4-1)/4*sr**2)/(T-1), 1e-18))
    sr0 = sd*((1-gm)*norm.ppf(1-1.0/n) + gm*norm.ppf(1-1.0/(n*e)))
    den = math.sqrt(max(1-g3*sr+(g4-1)/4*sr**2, 1e-18))
    return float(norm.cdf((sr-sr0)*math.sqrt(T-1)/den))


def metrics(eq, label, bench_r, extra=None):
    r = eq.pct_change().dropna()
    yrs = len(eq)/TRADING_DAYS
    vol = r.std()*np.sqrt(TRADING_DAYS)
    z, p = sharpe_diff_test(r, bench_r)
    m = {"label": label, "cagr": eq.iloc[-1]**(1/yrs)-1, "vol": vol,
         "sharpe": (r.mean()*TRADING_DAYS)/vol if vol > 0 else np.nan,
         "max_dd": (eq/eq.cummax()-1).min(),
         "dsr": deflated_sharpe(r, N_TRIALS_DSR), "p": p}
    if extra:
        m.update(extra)
    return m


def apply_weights(w, r, label):
    """Gross is hard-capped at 1.0. Raises if ever breached."""
    mx = float(w.max())
    if mx > MAX_GROSS + 1e-9:
        raise AssertionError(f"LEVERAGE BREACH in {label}: max gross {mx:.4f}")
    cost = w.diff().abs().fillna(0.0) * COST_BPS/1e4
    eq = (1 + w*r - cost).cumprod()
    return eq, mx, float(w.mean())


def vol_capped(px, sym, target, use_trend=False):
    """w = min(target / realised_vol, 1.0). Never levered."""
    c = px["close"][sym]
    r = c.pct_change().fillna(0.0)
    rv = (r.rolling(VOL_LB).std()*np.sqrt(TRADING_DAYS)).shift(1)
    w = (target/rv).clip(upper=MAX_GROSS).fillna(0.0)
    if use_trend:
        sma = pd.Series(talib.SMA(c.to_numpy(float), TREND_SMA), index=c.index)
        w = w.where(c.shift(1) > sma.shift(1), 0.0)
    return apply_weights(w, r, f"{sym}@{target}")


def main():
    print("="*104)
    print("NO LEVERAGE — can anything beat QQQ on BOTH CAGR and Sharpe?")
    print("="*104)

    px = load()
    idx = px["close"].index
    bench = px["close"][BENCH]/px["close"][BENCH].iloc[0]
    bench_r = bench.pct_change().dropna()
    b = metrics(bench, "QQQ buy & hold", bench_r, {"max_g": 1.0, "avg_g": 1.0})
    print(f"Window {idx[0].date()} -> {idx[-1].date()}")
    print(f"BAR: CAGR {b['cagr']*100:.2f}%  Sharpe {b['sharpe']:.2f}  "
          f"MaxDD {b['max_dd']*100:.2f}%")
    print(f"Constraint: gross <= {MAX_GROSS:.2f}, asserted in apply_weights()\n")

    runs = {"QQQ buy & hold": (bench, 1.0, 1.0)}
    for t in (0.25, 0.30, 0.35, 0.40):
        eq, mx, av = vol_capped(px, BENCH, t)
        runs[f"Vol-capped QQQ  target {t:.0%}"] = (eq, mx, av)
    for t in (0.30, 0.40):
        eq, mx, av = vol_capped(px, BENCH, t, use_trend=True)
        runs[f"Vol-cap + trend  target {t:.0%}"] = (eq, mx, av)

    xlk = px["close"]["XLK"]/px["close"]["XLK"].iloc[0]
    runs["XLK b&h  [HINDSIGHT ONLY]"] = (xlk, 1.0, 1.0)

    rows = []
    for k, (e, mx, av) in runs.items():
        e = (e/e.dropna().iloc[0]).dropna()
        rows.append(metrics(e, k, bench_r, {"max_g": mx, "avg_g": av}))

    print("="*104)
    print(f"{'':<32}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}{'MaxDD':>9}"
          f"{'DSR':>7}{'p':>8}{'avgGross':>10}")
    print("="*104)
    for m in rows:
        both = (m["cagr"] > b["cagr"] and m["sharpe"] > b["sharpe"]
                and m["label"] != "QQQ buy & hold")
        pstr = f"{m['p']:8.3f}" if pd.notna(m["p"]) else "     n/a"
        flag = "  <-- BOTH" if both else ""
        print(f"  {m['label']:<30}{m['cagr']*100:8.2f}%{m['vol']*100:7.2f}%"
              f"{m['sharpe']:8.2f}{m['max_dd']*100:8.2f}%{m['dsr']:7.3f}"
              f"{pstr}{m['avg_g']*100:9.1f}%{flag}")

    print("\n" + "="*104)
    print("LEVERAGE AUDIT")
    print("="*104)
    mx = max(m["max_g"] for m in rows)
    print(f"  Max gross across all strategies: {mx:.4f}  "
          f"| within constraint: {mx <= MAX_GROSS+1e-9}")

    # -- Attribution: is any edge concentrated in one year? --------------------
    winners = [m for m in rows if m["cagr"] > b["cagr"] and m["sharpe"] > b["sharpe"]
               and "HINDSIGHT" not in m["label"] and m["label"] != "QQQ buy & hold"]

    print("\n" + "="*104)
    print("VERDICT")
    print("="*104)
    if not winners:
        print("  NOTHING beat QQQ on both CAGR and Sharpe without leverage.")
        best_c = max((m for m in rows if "HINDSIGHT" not in m["label"]
                      and m["label"] != "QQQ buy & hold"), key=lambda x: x["cagr"])
        best_s = max((m for m in rows if "HINDSIGHT" not in m["label"]
                      and m["label"] != "QQQ buy & hold"), key=lambda x: x["sharpe"])
        print(f"    best CAGR:   {best_c['label']} {best_c['cagr']*100:.2f}% "
              f"(QQQ {b['cagr']*100:.2f}%)")
        print(f"    best Sharpe: {best_s['label']} {best_s['sharpe']:.2f} "
              f"(QQQ {b['sharpe']:.2f})")
    else:
        print(f"  {len(winners)} strategy/ies beat QQQ on BOTH, unlevered:\n")
        for m in winners:
            sig = pd.notna(m["p"]) and m["p"] < 0.05
            dsr_ok = m["dsr"] >= 0.95
            print(f"    {m['label']}")
            print(f"      CAGR {m['cagr']*100:.2f}% vs {b['cagr']*100:.2f}%  |  "
                  f"Sharpe {m['sharpe']:.2f} vs {b['sharpe']:.2f}")
            print(f"      p={m['p']:.3f} ({'SIGNIFICANT' if sig else 'not significant'})"
                  f"  DSR={m['dsr']:.3f} ({'pass' if dsr_ok else 'FAIL'})")

        print("\n" + "-"*104)
        print("  ATTRIBUTION — per-year excess CAGR vs QQQ (is the edge one event?)")
        print("-"*104)
        for m in winners:
            eq = (runs[m["label"]][0]).dropna()
            eq = eq/eq.iloc[0]
            print(f"\n  {m['label']}")
            hits = []
            for yr, g in eq.groupby(eq.index.year):
                if len(g) < 60:
                    continue
                sg = g/g.iloc[0]
                bg = bench.reindex(g.index).ffill()
                bg = bg/bg.iloc[0]
                d = (sg.iloc[-1] - bg.iloc[-1]) * 100
                hits.append((yr, d))
                bar = "+" * min(int(abs(d)/2), 20) if d > 0 else "-" * min(int(abs(d)/2), 20)
                print(f"    {yr}  {d:+7.2f}pp  {bar}")
            pos = sum(1 for _, d in hits if d > 0)
            top = max(hits, key=lambda x: abs(x[1]))
            tot = sum(d for _, d in hits)
            print(f"    -> beat QQQ in {pos}/{len(hits)} years | "
                  f"largest single year {top[0]}: {top[1]:+.2f}pp "
                  f"({abs(top[1]/tot)*100:.0f}% of total excess)" if tot != 0 else "")

    print("\n" + "="*104)
    print("HINDSIGHT YARDSTICK")
    print("="*104)
    h = [m for m in rows if "HINDSIGHT" in m["label"]][0]
    print(f"  XLK buy & hold: CAGR {h['cagr']*100:.2f}%  Sharpe {h['sharpe']:.2f}")
    print("  Beats QQQ on both -- but only because we picked the winner after")
    print("  seeing the data. The rules-based sector-momentum test scored 0.74")
    print("  Sharpe: NO RULE found XLK. Treat this as the hindsight ceiling,")
    print("  not an achievable strategy.")

    out = pd.DataFrame({k: v[0] for k, v in runs.items()})
    out.to_csv(os.path.join(OUTDIR, "nolev_beat_qqq_equity.csv"))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
    for k, (e, _, _) in runs.items():
        e = (e/e.dropna().iloc[0]).dropna()
        if k.startswith("QQQ"):
            st = dict(lw=2.6, color="black")
        elif "HINDSIGHT" in k:
            st = dict(lw=1.2, color="grey", ls=":")
        else:
            st = dict(lw=1.3)
        a1.plot(e.index, e, label=k, **st)
    a1.set_yscale("log"); a1.set_ylabel("Growth of $1 (log)")
    a1.set_title(f"No leverage — vs QQQ ({idx[0].date()} to {idx[-1].date()})")
    a1.legend(loc="upper left", fontsize=8); a1.grid(alpha=0.3)
    for k, (e, _, _) in runs.items():
        e = (e/e.dropna().iloc[0]).dropna()
        dd = e/e.cummax()-1
        st = dict(lw=2.2, color="black") if k.startswith("QQQ") else dict(lw=1.0)
        a2.plot(dd.index, dd*100, **st)
    a2.set_ylabel("Drawdown %"); a2.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUTDIR, "nolev_beat_qqq.png")
    plt.savefig(png, dpi=120); plt.close()
    print(f"\nSaved: {png}")


if __name__ == "__main__":
    main()
