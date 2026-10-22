"""Every GUI preset on the CURRENT engine (4c hurdle, decaying memory, throttle that scales the cap, GUI safety rules, default 0.25x / 2% / 2,000 sizing; Auto = AutoConfig defaults).
Windows: Jun 1 -> now (all hours complete), last 28 days, last 14 days, since the forward-test freeze (Oct 8), plus a 4c-slippage stress on Jun 1 -> now.  Learner is warm-started on all earlier history."""
import warnings
from datetime import date, timedelta

warnings.filterwarnings("ignore")
import pandas as pd

from trader.adaptive import AutoConfig
from trader.backtest import BacktestConfig, run_backtest
from trader.data import data_range, load_rows
from trader.gui import PRESETS
from trader.strategy import StrategyParams, parse_int_list

rows = load_rows()
lo, hi = data_range(rows)
end = hi.date()
GUI = dict(min_edge=0.04, dd_throttle=True, daily_loss_limit=0.10, max_drawdown_stop=0.25, min_history=30, slippage=0.02, compounding=True, kelly_mult=.25, max_frac=.02, max_contracts=2000)
WINDOWS = {"Jun1->now": date(2026, 6, 1), "last 28d": end - timedelta(days=27), "last 14d": end - timedelta(days=13), "since Oct 8": date(2026, 10, 8)}


def cfg_for(v, start, slip=0.02):
    params = StrategyParams(hours=parse_int_list(v["hours"], 0, 23), minutes=parse_int_list(v["minutes"], 1, 14), cost_min=v["cmin"], cost_max=v["cmax"],
                            legs=tuple(n for n, on in (("DOG", v["dog"]), ("FAV", v["fav"])) if on), fav_min=v["fmin"], fav_max=v["fmax"])
    core = parse_int_list(v["core"], 0, 23) if v.get("core") and v.get("corek", 0) > 0 else ()
    kw = dict(GUI, slippage=slip)
    hl = v.get("half", 0)
    return BacktestConfig(start=start, end=end, params=params, p_model="by_hour" if v["pmodel"] == 1 else "overall", p_basis="edge", hour_k=v.get("hourk", 40), lookback_days=v.get("look", 45),
                          halflife_days=0.0 if hl <= 0 else max(3.0, float(hl)), core_hours=core, core_k=v.get("corek", 0), auto=AutoConfig() if v.get("auto") else None, **kw)


out = []
for name, v in PRESETS.items():
    if v is None:
        continue
    rec = dict(preset=name.split(" · ")[0].split(" — ")[0][:34])
    for wn, st in WINDOWS.items():
        r = run_backtest(rows, cfg_for(v, st))
        m = r.metrics
        rec[wn] = f"{m.get('total_return', 0) * 100:+.0f}% ({m['trades']})"
        if wn == "Jun1->now":
            rec["Sharpe"], rec["maxDD"], rec["win"] = round(m.get("sharpe", float("nan")), 2), f"{m.get('max_drawdown', 0) * 100:.0f}%", f"{m.get('win_rate', 0) * 100:.0f}%"
            t = r.trades
            rec["EV c/ctr"] = round(float((t.pnl.sum() / t.contracts.sum()) * 100), 1) if len(t) else float("nan")
    r4 = run_backtest(rows, cfg_for(v, WINDOWS["Jun1->now"], slip=0.04))
    rec["4c slip: ret / Sharpe"] = f"{r4.metrics.get('total_return', 0) * 100:+.0f}% / {r4.metrics.get('sharpe', float('nan')):.1f}"
    out.append(rec)
    print(rec, flush=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n" + pd.DataFrame(out).to_string(index=False))
print("data through", hi)
