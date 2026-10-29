"""
Valid / RM: the final numbers for the presets - full period, in-sample (2016-09 .. 2022), out-of-sample (2023 .. 2026-10), SPY next to each, years, the deflated Sharpe ratio over
EVERY variant tried (all grids and selection stages: reports/mr_grid_*.csv, reports/mr_select_*.csv), the probabilistic Sharpe of the out-of-sample part, cost and timing variants,
and a chart per preset (MeanReversion/reports/backtest_<preset>.png).

  python mr_final.py
"""
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy import stats as SS

from mr_run import E, market_for

from resarb.mr import presets as PR  # noqa: E402
from resarb.mr.models import MRModel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "MeanReversion", "reports")
os.makedirs(OUT, exist_ok=True)
SPLIT = pd.Timestamp("2023-01-01")

# ── the trial count and the spread of in-sample Sharpes across all trials ───────────────────────────────────────────────────────────────────────────
trials = []
for f in glob.glob(os.path.join(HERE, "reports", "mr_grid_*.csv")) + glob.glob(os.path.join(HERE, "reports", "mr_select_*.csv")):
    d = pd.read_csv(f)
    if "sh_is" in d:
        trials += list(d.sh_is.dropna())
N = len(trials) + 60                                                                          # + the triage families of mr_triage / mr_triage2 (not all stored as Sharpes)
V = float(np.var(trials))
print(f"trials: {N} (in-sample Sharpe across them: mean {np.mean(trials):.2f}, sd {math.sqrt(V):.2f})")


def dsr(r: pd.Series, n: int, var_ann: float) -> tuple:
    """Bailey & Lopez de Prado (2014): probability that the true Sharpe beats the expected maximum of n trials with this spread (all per-period)."""
    r = r.dropna()
    sr = r.mean() / r.std()
    v = var_ann / 252.0
    g = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * SS.norm.ppf(1 - 1 / n) + g * SS.norm.ppf(1 - 1 / (n * math.e)))
    sk, ku = SS.skew(r), SS.kurtosis(r, fisher=False)
    z = (sr - sr0) * math.sqrt(len(r) - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return float(SS.norm.cdf(z)), float(sr0 * math.sqrt(252))


def psr(r: pd.Series, bench_sr_ann: float = 0.0) -> float:
    r = r.dropna()
    sr, b = r.mean() / r.std(), bench_sr_ann / math.sqrt(252)
    sk, ku = SS.skew(r), SS.kurtosis(r, fisher=False)
    return float(SS.norm.cdf((sr - b) * math.sqrt(len(r) - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)))


def chart(res, name: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq = res.equity / res.equity.iloc[0]
    spy = (1 + res.bench.fillna(0)).cumprod()
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a.plot(eq.index, eq.values, label=name, color="#3d8bfd", lw=1.5)
    a.plot(spy.index, spy.values, label="SPY buy & hold", color="#8b98a9", lw=1.1)
    a.set_yscale("log"); a.axvspan(SPLIT, eq.index[-1], color="#d9a520", alpha=0.08); a.legend(loc="upper left"); a.grid(alpha=0.3)
    a.set_title(f"{name} vs SPY (out-of-sample shaded)")
    b.fill_between(eq.index, (eq / eq.cummax() - 1) * 100, 0, color="#e05561", alpha=0.5, label=name)
    b.plot(spy.index, (spy / spy.cummax() - 1) * 100, color="#8b98a9", lw=0.9, label="SPY")
    b.set_ylabel("drawdown %"); b.grid(alpha=0.3); b.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


rows, years = [], {}
for pname in PR.preset_names():
    base = PR.get(pname)
    variants = [("as preset", base)]
    if base.minute == 1545:
        variants += [("15:30 (free delayed SIP)", MRModel.from_dict({**base.to_dict(), "minute": 1530})), ("cost 2 bp", MRModel.from_dict({**base.to_dict(), "cost_auction_bp": 2.0})),
                     ("cost 3 bp", MRModel.from_dict({**base.to_dict(), "cost_auction_bp": 3.0}))]
    for vname, m in variants:
        r = E.run(m, market_for(m))
        s = r.stats
        a, i, o = s["all"], s["is"], s["oos"]
        d, sr0 = dsr(r.returns[r.returns.index < SPLIT], N, V)
        rows.append(dict(preset=pname, variant=vname, sharpe=a["sharpe"], sharpe_is=i["sharpe"], sharpe_oos=o["sharpe"], cagr=a["cagr"], cagr_is=i["cagr"], cagr_oos=o["cagr"],
                         vol=a["vol"], max_dd=a["max_dd"], spy_sharpe=s["spy_all"]["sharpe"], spy_cagr=s["spy_all"]["cagr"], spy_dd=s["spy_all"]["max_dd"],
                         spy_sharpe_oos=s["spy_oos"]["sharpe"], spy_cagr_oos=s["spy_oos"]["cagr"], trades=s["trades"], trades_yr=s["trades"] / (a["days"] / 252),
                         win=s.get("win_rate"), avg_ret=s.get("avg_ret"), avg_hold=s.get("avg_hold"), pf=s.get("profit_factor"), expo=s["avg_exposure"], max_expo=s["max_exposure"],
                         beta=s.get("beta_spy"), dsr_is=d, sr0=sr0, psr_oos=psr(r.returns[r.returns.index >= SPLIT]),
                         psr_oos_vs_spy=psr(r.returns[r.returns.index >= SPLIT], s["spy_oos"]["sharpe"])))
        print(f"{pname} [{vname}]: {E.summary(s)} · DSR(IS) {d:.2f} (bar {sr0:.2f}) · PSR(OOS>0) {rows[-1]['psr_oos']:.2f}", flush=True)
        if vname == "as preset":
            years[pname] = (s["by_year"], s["spy_by_year"])
            tag = m.name.replace(" ", "_").replace("<", "lt").replace(">", "gt").replace("+", "_")
            chart(r, m.name, os.path.join(OUT, f"backtest_{tag}.png"))
            r.trades.to_csv(os.path.join(OUT, f"trades_{tag}.csv"), index=False)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "reports", "mr_final.csv"), index=False)
pd.set_option("display.width", 250)
print(df[["preset", "variant", "sharpe", "sharpe_is", "sharpe_oos", "cagr", "cagr_oos", "max_dd", "vol", "spy_sharpe", "spy_cagr", "spy_dd", "trades_yr", "win", "expo", "max_expo",
          "beta", "dsr_is", "psr_oos", "psr_oos_vs_spy"]].round(3).to_string(index=False))
yr = pd.DataFrame({p: pd.Series(v[0]) for p, v in years.items()})
yr["SPY"] = pd.Series(next(iter(years.values()))[1])
print((yr * 100).round(1).to_string())
yr.to_csv(os.path.join(HERE, "reports", "mr_final_years.csv"))
