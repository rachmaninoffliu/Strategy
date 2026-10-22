"""
Walk-forward & stability analysis — Supertrend SOXL (corrected engine)

Run AFTER the three engine fixes:
  1. entryPrice = open of the fill bar (Pine semantics), not the fill price
  2. supert frozen at entry for the dynamic TP
  3. mark-to-market from last_mark, not cl[i-1]  <- was suppressing results ~8x

Everything previously reported about sub-period stability was computed with
bug 3 present and is void.

THREE TESTS
-----------
  A. PER-YEAR STABILITY
     Is the edge present every year, or carried by a few?

  B. TRUE WALK-FORWARD ON atrPeriod
     atrPeriod is the fragile parameter (500 -> 2000 flipped the sign in
     earlier testing). This is the honest overfitting test: on each fold,
     pick the atrPeriod that was BEST on the prior 3 years, then trade it
     for the next year OOS. Compare against simply fixing atrPeriod=500.
     If in-sample selection does not beat the fixed choice, the parameter
     is not learnable and should stay fixed by convention.

  C. PARAMETER SENSITIVITY
     Does 500/5.0 sit on a plateau or a spike? A spike means the backtest
     is a lucky draw, not a robust edge.

Config = the user's stated settings: Always Fixed, atr 500, factor 5,
RR 4.5%, buyback 1.5%, trailFactor 0, price_perc 0.1, useDynamic OFF.
Realistic costs: 5bp/leg, conservative pierce fills, 50% equity.
"""

import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/scripts")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import supertrend_strategy as S

BPY = 26 * 252
OUT = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/reports"
DATA = "C:/Users/xinyu/PycharmProjects/pythonProject/alpha_syndicate/data/SOXL_15m_2016_split.parquet"

BASE = dict(strategyMode="Always Fixed", useDateFilter=False, equity_frac=0.5,
            factor=5.0, riskReward=0.045, buyback=0.015,
            trailFactor=0.0, price_perc=0.1, useDynamic=False,
            commission_bps=5.0, touch_fill=False)

ATR_GRID    = [250, 375, 500, 750, 1000, 1500]
FACTOR_GRID = [3.0, 4.0, 5.0, 6.0, 7.0]


def load_bars():
    p = pd.read_parquet(DATA)
    p["et"] = pd.to_datetime(p["et"], utc=True).dt.tz_convert("America/New_York")
    p["bar"] = p["et"].dt.tz_localize(None)
    return (p[p.symbol == "SOXL"].set_index("bar").sort_index()
            [["open", "high", "low", "close"]].ffill().dropna())


def run_cfg(bars, **over):
    cfg = S.Config(**{**BASE, **over})
    eq, tr = S.SupertrendStrategy(cfg).run(bars)
    return eq, tr


def seg_stats(tr, lo, hi, frac=0.5):
    """Compound a trade slice on a percent-of-equity basis."""
    s = tr[(tr.exit_time >= lo) & (tr.exit_time < hi)]
    if len(s) < 5:
        return None
    r = (s.return_pct / 100).to_numpy()
    m = np.cumprod(1 + frac * r)
    yrs = max((s.exit_time.max() - s.exit_time.min()).days / 365.25, 1e-6)
    dd = (pd.Series(np.r_[1.0, m]) / pd.Series(np.r_[1.0, m]).cummax() - 1).min()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(s), "mult": m[-1], "ret": (m[-1] - 1) * 100,
            "cagr": (m[-1] ** (1 / yrs) - 1) * 100 if m[-1] > 0 else -100,
            "dd": dd * 100, "pf": gp / gl if gl > 0 else np.nan,
            "avg": r.mean() * 100, "win": 100 * (r > 0).mean()}


def main():
    bars = load_bars()
    print("=" * 96)
    print("WALK-FORWARD & STABILITY — corrected engine")
    print("=" * 96)
    print(f"bars {len(bars):,} | {bars.index[0].date()} -> {bars.index[-1].date()}")
    print("config: Always Fixed, atr500 f5.0 RR4.5% bb1.5% trail0 pp0.1 dynOFF")
    print("costs : 5bp/leg, pierce fills, 50% equity\n")

    # ---------------- A. per-year stability -------------------------------
    eq, tr = run_cfg(bars, atrPeriod=500)
    print("=" * 96)
    print("A. PER-YEAR STABILITY")
    print("=" * 96)
    print(f"  {'year':<7}{'trades':>7}{'return':>10}{'MaxDD':>9}{'PF':>7}"
          f"{'avg%':>8}{'win%':>7}")
    yrly = []
    for y in range(2016, 2027):
        s = seg_stats(tr, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01"))
        if not s:
            continue
        yrly.append((y, s))
        bar = "+" * min(int(s["ret"] / 5), 24) if s["ret"] > 0 else "-" * min(int(-s["ret"] / 5), 24)
        print(f"  {y:<7}{s['n']:>7}{s['ret']:>9.1f}%{s['dd']:>8.1f}%{s['pf']:>7.2f}"
              f"{s['avg']:>8.3f}{s['win']:>7.1f}  {bar}")
    pos = sum(1 for _, s in yrly if s["ret"] > 0)
    pfs = [s["pf"] for _, s in yrly]
    print(f"\n  positive years {pos}/{len(yrly)} | PF>1 in "
          f"{sum(1 for x in pfs if x > 1)}/{len(pfs)} | median PF {np.median(pfs):.3f}")

    # ---------------- long/short attribution ------------------------------
    print("\n  attribution by side:")
    for side, g in tr.groupby("side"):
        r = (g.return_pct / 100).to_numpy()
        gp, gl = r[r > 0].sum(), -r[r < 0].sum()
        print(f"    {side:<6} n={len(g):5d}  avg={r.mean()*100:+.3f}%  "
              f"PF={gp/gl:.3f}  win={100*(r>0).mean():.1f}%")

    # ---------------- C. parameter sensitivity ----------------------------
    print("\n" + "=" * 96)
    print("C. PARAMETER SENSITIVITY — plateau or spike?")
    print("=" * 96)
    atr_res = {}
    print(f"  {'atrPeriod':<12}{'return':>12}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'n':>7}")
    for a in ATR_GRID:
        e, t = run_cfg(bars, atrPeriod=a)
        st = S.summarize(e, t, BPY)
        atr_res[a] = t
        print(f"  {a:<12}{st['total_return_pct']:>11,.0f}%{st['cagr_pct']:>8.2f}%"
              f"{st['sharpe']:>8.2f}{st['max_dd_pct']:>8.2f}%{st['trades']:>7}")

    print(f"\n  {'factor':<12}{'return':>12}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'n':>7}")
    for fc in FACTOR_GRID:
        e, t = run_cfg(bars, atrPeriod=500, factor=fc)
        st = S.summarize(e, t, BPY)
        print(f"  {fc:<12}{st['total_return_pct']:>11,.0f}%{st['cagr_pct']:>8.2f}%"
              f"{st['sharpe']:>8.2f}{st['max_dd_pct']:>8.2f}%{st['trades']:>7}")

    # ---------------- B. true walk-forward on atrPeriod -------------------
    print("\n" + "=" * 96)
    print("B. WALK-FORWARD — select atrPeriod on 3y train, trade it 1y OOS")
    print("=" * 96)
    print(f"  {'test yr':<9}{'picked':>8}{'train PF':>10}{'OOS ret':>10}"
          f"{'OOS PF':>8}   {'fixed-500 OOS':>14}")
    wf_mult, fx_mult, rows = 1.0, 1.0, []
    for ty in range(2019, 2027):
        tr0, tr1 = pd.Timestamp(f"{ty-3}-01-01"), pd.Timestamp(f"{ty}-01-01")
        te0, te1 = tr1, pd.Timestamp(f"{ty+1}-01-01")
        best, best_pf = None, -np.inf
        for a, t in atr_res.items():
            s = seg_stats(t, tr0, tr1)
            if s and s["pf"] > best_pf:
                best, best_pf = a, s["pf"]
        oos = seg_stats(atr_res[best], te0, te1)
        fix = seg_stats(atr_res[500], te0, te1)
        if not oos or not fix:
            continue
        wf_mult *= oos["mult"]; fx_mult *= fix["mult"]
        rows.append((ty, best, best_pf, oos, fix))
        print(f"  {ty:<9}{best:>8}{best_pf:>10.2f}{oos['ret']:>9.1f}%"
              f"{oos['pf']:>8.2f}   {fix['ret']:>13.1f}%")

    print(f"\n  walk-forward selected : {(wf_mult-1)*100:>10,.1f}% cumulative")
    print(f"  fixed atrPeriod=500   : {(fx_mult-1)*100:>10,.1f}% cumulative")
    picked = [b for _, b, _, _, _ in rows]
    hit = sum(1 for _, b, _, o, f in rows if o["ret"] > f["ret"])
    print(f"  selection beat fixed in {hit}/{len(rows)} folds | "
          f"params chosen: {sorted(set(picked))}")
    if fx_mult > wf_mult:
        print("\n  => In-sample selection FAILED to beat a fixed parameter.")
        print("     atrPeriod is NOT learnable from history. Keep it fixed at 500")
        print("     by convention; tuning it per-period destroys value.")
    else:
        print("\n  => Selection added value, but check fold count before trusting it.")

    # ---------------- plot -------------------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 9),
                                 gridspec_kw={"height_ratios": [2, 1]})
    ec = eq / eq.iloc[0]
    a1.plot(ec.index, ec, lw=1.3, color="tab:blue", label="Supertrend (5bp, pierce)")
    bh = bars["close"] / bars["close"].iloc[0]
    a1.plot(bh.index, bh, lw=1.6, color="black", alpha=.7, label="SOXL buy & hold")
    a1.set_yscale("log"); a1.set_ylabel("Growth of $1 (log)"); a1.grid(alpha=.3)
    a1.legend(loc="upper left"); a1.set_title("Supertrend SOXL — corrected engine")
    ys = [y for y, _ in yrly]; rs = [s["ret"] for _, s in yrly]
    a2.bar(ys, rs, color=["tab:green" if v > 0 else "tab:red" for v in rs])
    a2.axhline(0, color="black", lw=.8); a2.set_ylabel("annual return %")
    a2.grid(alpha=.3)
    plt.tight_layout()
    png = os.path.join(OUT, "supertrend_walkforward.png")
    plt.savefig(png, dpi=120); plt.close()
    print(f"\nSaved: {png}")


if __name__ == "__main__":
    main()
