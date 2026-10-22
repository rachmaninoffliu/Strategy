"""Kelly sizing sweep: which multiplier / cap / min-edge / learning speed gives the best Sharpe (and the best RISK-ADJUSTED growth)?
Same data as the app (Jun 1 - Oct 8), Polymarket US fees, 2c slippage. In-sample: read plateaus, not the single best cell."""
import sys
import warnings
from datetime import date

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.strategy import StrategyParams

rows = load_rows()
lo, hi = data_range(rows)
START = date(2026, 6, 1)
PRESETS = {
    "R2": (StrategyParams(hours=(22, 23), cost_min=0.15, cost_max=0.35), "overall"),
    "R3": (StrategyParams(hours=(20, 21, 22, 23), cost_min=0.15, cost_max=0.35), "by_hour"),
    "R1": (StrategyParams(hours=(22, 23)), "overall"),
}
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def run(preset, **kw):
    pr, pm = PRESETS[preset]
    return run_backtest(rows, BacktestConfig(start=START, end=hi.date(), params=pr, p_model=pm, **kw))


MULTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]
CAPS = [0.01, 0.02, 0.03, 0.05, 0.10, 1.00]

# ── 1. multiplier x cap grid (R2) ─────────────────────────────────────────────
tab = {k: pd.DataFrame(index=[f"{m:.2f}" for m in MULTS], columns=[f"{c*100:.0f}%" for c in CAPS], dtype=float) for k in ("Sharpe", "Return%", "MaxDD%", "Calmar")}
for m in MULTS:
    for c in CAPS:
        mt = run("R2", kelly_mult=m, max_frac=c).metrics
        a, b = f"{m:.2f}", f"{c*100:.0f}%"
        tab["Sharpe"].loc[a, b] = mt["sharpe"]
        tab["Return%"].loc[a, b] = mt["total_return"] * 100
        tab["MaxDD%"].loc[a, b] = mt["max_drawdown"] * 100
        tab["Calmar"].loc[a, b] = (mt["total_return"] / abs(mt["max_drawdown"])) if mt["max_drawdown"] < 0 else np.nan
print("=== 1. R2: Kelly multiplier (rows) x per-trade cap (columns) ===")
for k, v in tab.items():
    print(f"\n-- {k} --")
    print(v.round(2).to_string())

# ── 2. other presets, same grid, best cells only ──────────────────────────────
print("\n=== 2. Same sweep on R1 and R3 (top 5 by Sharpe, and the default cell) ===")
for pn in ("R1", "R3"):
    res = []
    for m in MULTS:
        for c in CAPS:
            mt = run(pn, kelly_mult=m, max_frac=c).metrics
            res.append((m, c, mt["sharpe"], mt["total_return"] * 100, mt["max_drawdown"] * 100))
    d = pd.DataFrame(res, columns=["mult", "cap", "sharpe", "return%", "maxDD%"])
    print(f"\n{pn} top 5:"); print(d.sort_values("sharpe", ascending=False).head(5).round(2).to_string(index=False))
    print(f"{pn} default (0.25 / 2%):", d[(d.mult == 0.25) & (d.cap == 0.02)].round(2).to_string(index=False, header=False))

# ── 3. empirical growth-optimal multiplier (in-sample) + Monte-Carlo of risk ──
print("\n=== 3. Growth-optimal multiplier and Monte-Carlo risk (R2, trades resampled) ===")
t = run("R2", kelly_mult=1.0, max_frac=1.0).trades            # f_used == full Kelly: scale it ourselves
r = (t.pnl / t.stake).values
f = t.f_full.values
cs = np.arange(0.02, 2.0, 0.02)
growth = [np.mean(np.log1p(np.clip(c * f * r, -0.99, None))) for c in cs]
cstar = float(cs[int(np.argmax(growth))])
print(f"{len(t)} trades; mean full-Kelly f {f.mean()*100:.1f}%; in-sample growth-optimal multiplier c* = {cstar:.2f} x full Kelly (growth peaks there, falls beyond ~{2*cstar:.2f}x)")
rng = np.random.default_rng(1)
rows_mc = []
for c in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.0):
    for cap in (0.02, 1.0):
        finals, dds = [], []
        for _ in range(4000):
            ix = rng.integers(0, len(r), len(r))
            fr = np.minimum(c * f[ix], cap)
            path = np.cumprod(1 + fr * r[ix])
            peak = np.maximum.accumulate(np.r_[1.0, path])[1:]
            finals.append(path[-1]); dds.append(((path - peak) / peak).min())
        finals, dds = np.array(finals), np.array(dds)
        rows_mc.append((c, f"{cap*100:.0f}%", np.median(finals), np.percentile(finals, 5), np.mean(finals < 1) * 100, np.median(dds) * 100, np.mean(dds < -0.30) * 100))
M = pd.DataFrame(rows_mc, columns=["mult", "cap", "median final x", "5th pct final x", "P(loss)%", "median maxDD%", "P(DD>30%)%"])
print(M.round(2).to_string(index=False))

# ── 4. min edge and learning speed (prior strength, lookback) at the default sizing ──
print("\n=== 4. R2 at default sizing: min edge / prior strength / lookback ===")
res = []
for me in (0.0, 0.005, 0.01, 0.02, 0.03):
    mt = run("R2", min_edge=me).metrics
    res.append((f"min_edge {me*100:.1f}c", mt["trades"], mt["win_rate"] * 100, mt["total_return"] * 100, mt["max_drawdown"] * 100, mt["sharpe"]))
for pn in (5, 10, 20, 40, 80):
    mt = run("R2", prior_n=pn).metrics
    res.append((f"prior_n {pn}", mt["trades"], mt["win_rate"] * 100, mt["total_return"] * 100, mt["max_drawdown"] * 100, mt["sharpe"]))
for lb in (0, 30, 45, 60, 90):
    mt = run("R2", lookback_days=lb).metrics
    res.append((f"lookback {lb or 'all'}d", mt["trades"], mt["win_rate"] * 100, mt["total_return"] * 100, mt["max_drawdown"] * 100, mt["sharpe"]))
print(pd.DataFrame(res, columns=["setting", "trades", "win%", "return%", "maxDD%", "sharpe"]).round(2).to_string(index=False))
