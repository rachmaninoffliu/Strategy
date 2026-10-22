"""Performance metrics from an equity series and a trade list (no Qt). One evaluation serves the Backtest AND the Live session, so the two are comparable."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
TRADING_DAYS = 252


# ── live session trades -> the backtest's trade schema ────────────────────────────
def _naive_et(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    return t


def trades_from_live(rows: list, capital: float) -> pd.DataFrame:
    """Engine 'trades' event rows -> the columns the backtest trade log has. Entry time is not known from a closing fill (NaT); equity is rebuilt from the P&L."""
    cols = ["side", "entry_time", "price", "qty", "exit_time", "exit_price", "exit_reason", "return_pct", "pnl_usd", "equity_after", "symbol"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame([{"side": str(r.get("side", "")).lower(), "entry_time": pd.NaT, "price": float(r["entry"]), "qty": float(r["qty"]), "exit_time": _naive_et(r["time"]),
                        "exit_price": float(r["exit"]), "exit_reason": "—", "return_pct": float(r["pct"]), "pnl_usd": float(r["pnl"]), "symbol": r.get("symbol", "")} for r in rows])
    df = df.sort_values("exit_time").reset_index(drop=True)
    df["equity_after"] = capital + df["pnl_usd"].cumsum()
    return df[cols]


def curve_from_trades(trades: pd.DataFrame, capital: float, start, end=None, end_equity: Optional[float] = None) -> pd.Series:
    """Step equity curve of a trade list: the capital at ``start``, the equity after each trade, and (optionally) the current equity at ``end`` (open P&L included)."""
    start = _naive_et(start)
    xs, ys = [start], [float(capital)]
    for t, e in zip(trades["exit_time"], trades["equity_after"]):
        xs.append(max(pd.Timestamp(t), xs[-1])); ys.append(float(e))
    if end is not None and end_equity is not None:
        xs.append(max(_naive_et(end), xs[-1])); ys.append(float(end_equity))
    return pd.Series(ys, index=pd.DatetimeIndex(xs), dtype=float)


# ── helpers ───────────────────────────────────────────────────────────────────────
def drawdown_series(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def daily_equity(eq: pd.Series, fill: bool = False) -> pd.Series:
    """Last equity of each calendar day. ``fill`` also adds the weekdays with no point (a trade-based curve is flat between trades)."""
    d = eq.groupby(eq.index.normalize()).last()
    if fill and len(d) > 1:
        d = d.reindex(pd.bdate_range(d.index[0], d.index[-1])).ffill()
    return d


def _runs(flag: np.ndarray):
    e = np.diff(np.r_[0, flag.astype(int), 0])
    return np.where(e == 1)[0], np.where(e == -1)[0]


def _streaks(win: np.ndarray) -> dict:
    if not len(win):
        return dict(max_consec_wins=0, max_consec_losses=0, avg_win_streak=0.0, avg_loss_streak=0.0, current_streak=0)
    ws, we = _runs(win)
    ls, le = _runs(~win)
    cur = int(we[-1] - ws[-1]) if len(we) and we[-1] == len(win) else -int(le[-1] - ls[-1]) if len(le) and le[-1] == len(win) else 0
    return dict(max_consec_wins=int((we - ws).max()) if len(ws) else 0, max_consec_losses=int((le - ls).max()) if len(ls) else 0,
                avg_win_streak=float((we - ws).mean()) if len(ws) else 0.0, avg_loss_streak=float((le - ls).mean()) if len(ls) else 0.0, current_streak=cur)


def _longest_underwater_days(eq: pd.Series, dd: pd.Series) -> float:
    s, e = _runs((dd < -1e-12).to_numpy())
    if not len(s):
        return 0.0
    idx = eq.index
    return float(max((idx[min(b, len(idx) - 1)] - idx[max(a - 1, 0)]).total_seconds() for a, b in zip(s, e)) / 86400.0)


def compute_metrics(eq: pd.Series, trades: pd.DataFrame, capital: float, bars_per_year: int = 26 * TRADING_DAYS, commission_bps: float = 0.0,
                    bh: Optional[pd.Series] = None, fill_days: bool = False) -> dict:
    """Flat metric dict. ``eq``: equity per bar (backtest) or a step curve (live). ``trades``: strategy trade log (return_pct in PERCENT, pnl_usd in $)."""
    m: dict = dict(initial_capital=float(capital), trades=int(len(trades)))
    if eq is None or len(eq) < 2:
        m.update(final_equity=float(capital), total_return=0.0, total_pnl=0.0, max_drawdown=0.0)
        return m
    eq = eq.astype(float)
    final = float(eq.iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400.0 / DAYS_PER_YEAR, 1e-9)
    dd = drawdown_series(eq)
    with np.errstate(over="ignore", invalid="ignore"):                                # a few minutes of live data annualised can overflow: that is inf, shown as '—', not a crash
        cagr = float(np.expm1(np.log(final / capital) / years)) if final > 0 else -1.0
    m.update(final_equity=final, total_pnl=final - capital, total_return=final / capital - 1.0, period_years=years,
             cagr=cagr, max_drawdown=float(dd.min()), max_drawdown_usd=float((eq - eq.cummax()).min()),
             max_dd_duration_days=_longest_underwater_days(eq, dd), avg_drawdown=float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0)

    day = daily_equity(eq, fill_days)
    dr = day.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    dday = drawdown_series(day)
    m["ulcer_index"] = float(np.sqrt((dday.values ** 2).mean()))
    if len(dr) > 2 and dr.std(ddof=1) > 0:
        vol = dr.std(ddof=1) * np.sqrt(TRADING_DAYS)
        dn = np.sqrt((np.minimum(dr.values, 0) ** 2).mean())
        m.update(daily_vol_ann=float(vol), sharpe=float(dr.mean() / dr.std(ddof=1) * np.sqrt(TRADING_DAYS)), sortino=float(dr.mean() / dn * np.sqrt(TRADING_DAYS)) if dn > 0 else float("nan"),
                 best_day=float(dr.max()), worst_day=float(dr.min()), pct_days_up=float((dr > 0).mean()))
    else:
        m.update(daily_vol_ann=float("nan"), sharpe=float("nan"), sortino=float("nan"), best_day=float("nan"), worst_day=float("nan"), pct_days_up=float("nan"))
    m["calmar"] = float(m["cagr"] / abs(m["max_drawdown"])) if m["max_drawdown"] < 0 and np.isfinite(m["cagr"]) else float("nan")
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()                       # the per-bar basis supertrend_strategy.summarize() uses (earlier research quotes it)
    bvol = r.std() * np.sqrt(bars_per_year) if len(r) > 2 else float("nan")
    m["sharpe_bar"] = float(r.mean() * bars_per_year / bvol) if np.isfinite(bvol) and bvol > 0 else float("nan")

    if bh is not None and len(bh) > 2:
        bh = bh.astype(float)
        by = max((bh.index[-1] - bh.index[0]).total_seconds() / 86400.0 / DAYS_PER_YEAR, 1e-9)
        m.update(bh_total_return=float(bh.iloc[-1] / bh.iloc[0] - 1.0), bh_cagr=float(np.expm1(np.log(bh.iloc[-1] / bh.iloc[0]) / by)), bh_max_drawdown=float(drawdown_series(bh).min()))

    n = len(trades)
    if n == 0:
        return m
    t = trades.reset_index(drop=True)
    ret = t["return_pct"].astype(float).to_numpy() / 100.0
    pnl = t["pnl_usd"].astype(float).to_numpy()
    win = ret > 0
    wins, losses = ret[win], ret[~win]
    gp, gl = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    m.update(wins=int(win.sum()), losses=int((~win).sum()), win_rate=float(win.mean()), avg_win=float(wins.mean()) if len(wins) else 0.0, avg_loss=float(losses.mean()) if len(losses) else 0.0,
             profit_factor=gp / gl if gl > 0 else float("inf") if gp > 0 else float("nan"), expectancy=float(ret.mean()), best_trade=float(ret.max()), worst_trade=float(ret.min()),
             best_trade_usd=float(pnl.max()), worst_trade_usd=float(pnl.min()))
    m["payoff_ratio"] = abs(m["avg_win"] / m["avg_loss"]) if m["avg_loss"] != 0 else float("nan")
    pr_g, pr_l = float(ret[ret > 0].sum()), float(-ret[ret < 0].sum())
    m["profit_factor_pct"] = pr_g / pr_l if pr_l > 0 else float("nan")                    # = supertrend_strategy.summarize()'s profit factor (sum of % returns)
    if n > 2 and ret.std(ddof=1) > 0:
        m["t_stat"] = float(ret.mean() / ret.std(ddof=1) * np.sqrt(n))
        bs = np.random.default_rng(0).choice(ret, (1500, n)).mean(axis=1)
        m["ret_ci_lo"], m["ret_ci_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    else:
        m["t_stat"] = m["ret_ci_lo"] = m["ret_ci_hi"] = float("nan")
    m.update(_streaks(win))

    side = t["side"].astype(str).str.lower()
    m["long_trades"], m["short_trades"] = int((side == "long").sum()), int((side == "short").sum())
    m["trades_per_year"] = n / years
    if commission_bps:
        m["commission_paid"] = float((t["qty"] * (t["price"] + t["exit_price"]) * commission_bps / 1e4).sum())
    if "entry_time" in t and t["entry_time"].notna().any():
        hold = (pd.to_datetime(t["exit_time"]) - pd.to_datetime(t["entry_time"])).dt.total_seconds() / 3600.0
        m.update(avg_hold_hours=float(hold.mean()), median_hold_hours=float(hold.median()), longest_hold_days=float(hold.max() / 24.0),
                 exposure=float(min(hold.sum() * 3600.0 / max((eq.index[-1] - eq.index[0]).total_seconds(), 1.0), 1.0)))
    before = t["equity_after"].astype(float).to_numpy() - pnl
    size = (t["qty"].astype(float) * t["price"].astype(float)).to_numpy() / np.where(before > 0, before, np.nan)
    m["avg_position_pct"] = float(np.nanmean(size)) if np.isfinite(size).any() else float("nan")
    reasons = t["exit_reason"].astype(str).value_counts(normalize=True) if "exit_reason" in t else pd.Series(dtype=float)
    for k in ("reversal", "trail", "target"):
        m[f"exit_{k}"] = float(reasons.get(k, 0.0))
    return m


METRIC_SECTIONS = [
    ("Returns", [("Initial capital", "initial_capital", "$"), ("Final equity", "final_equity", "$big"), ("Total P&L", "total_pnl", "$s"), ("Total return", "total_return", "%s"),
                 ("CAGR (annualised; extrapolated if < 1 year)", "cagr", "%"), ("Period (years)", "period_years", "2"), ("Buy & hold total return", "bh_total_return", "%s"),
                 ("Buy & hold CAGR", "bh_cagr", "%"), ("Buy & hold max drawdown", "bh_max_drawdown", "%"), ("Commission paid (estimate)", "commission_paid", "$")]),
    ("Risk", [("Max drawdown", "max_drawdown", "%"), ("Max drawdown ($)", "max_drawdown_usd", "$s"), ("Longest time under water (days)", "max_dd_duration_days", "1"),
              ("Average drawdown", "avg_drawdown", "%"), ("Ulcer index (daily)", "ulcer_index", "%"), ("Annualised volatility (daily)", "daily_vol_ann", "%"),
              ("Sharpe ratio (daily, √252)", "sharpe", "2"), ("Sharpe ratio (per bar - the basis of earlier research notes)", "sharpe_bar", "2"), ("Sortino ratio", "sortino", "2"),
              ("Calmar ratio (CAGR / max drawdown)", "calmar", "2"), ("Best day", "best_day", "%s"), ("Worst day", "worst_day", "%s"), ("% days up", "pct_days_up", "%")]),
    ("Trades", [("Trades", "trades", "0"), ("Wins", "wins", "0"), ("Losses", "losses", "0"), ("Win rate", "win_rate", "%"), ("Average win (per trade)", "avg_win", "%s"),
                ("Average loss (per trade)", "avg_loss", "%s"), ("Payoff ratio (avg win / avg loss)", "payoff_ratio", "2"), ("Profit factor ($)", "profit_factor", "2"),
                ("Profit factor (sum of % returns - the earlier-notes basis)", "profit_factor_pct", "2"), ("Expectancy (avg return per trade)", "expectancy", "%s"),
                ("95% CI of return per trade — low", "ret_ci_lo", "%s"), ("95% CI of return per trade — high", "ret_ci_hi", "%s"), ("t-stat of return per trade", "t_stat", "2"),
                ("Best trade", "best_trade", "%s"), ("Worst trade", "worst_trade", "%s")]),
    ("Streaks", [("Max consecutive wins", "max_consec_wins", "0"), ("Max consecutive losses", "max_consec_losses", "0"), ("Average win streak", "avg_win_streak", "2"),
                 ("Average loss streak", "avg_loss_streak", "2"), ("Current streak (+win / −loss)", "current_streak", "0")]),
    ("Activity and exits", [("Long trades", "long_trades", "0"), ("Short trades", "short_trades", "0"), ("Trades per year", "trades_per_year", "1"), ("Average hold (hours)", "avg_hold_hours", "1"),
                            ("Median hold (hours)", "median_hold_hours", "1"), ("Longest hold (days)", "longest_hold_days", "1"), ("Time in market", "exposure", "%"),
                            ("Average position size (% of equity)", "avg_position_pct", "%"), ("Exited by reversal (trend flip)", "exit_reversal", "%"),
                            ("Exited by trailing stop", "exit_trail", "%"), ("Exited by take-profit target", "exit_target", "%")]),
]


def fmt_metric(v, kind: str) -> str:
    from .theme import big_money, money, num, pct
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if kind == "$":
        return f"${money(v)}"
    if kind == "$big":
        return big_money(v)
    if kind == "$s":
        return f"{'-' if v < 0 else '+'}${money(abs(v))}"
    if kind == "%":
        return pct(v)
    if kind == "%s":
        return pct(v, 2, True)
    if kind == "0":
        return f"{int(v):,}"
    if kind == "1":
        return num(v, 1)
    return num(v, 2)


# ── tables derived from the curve / trade log ─────────────────────────────────────
def monthly_table(eq: pd.Series, capital: float) -> pd.DataFrame:
    """Returns by calendar month (rows = years, columns 1..12 and 'Year'), as fractions. NaN where the strategy had no data."""
    if eq is None or len(eq) < 2:
        return pd.DataFrame()
    me = eq.groupby([eq.index.year, eq.index.month]).last()
    prev = me.shift(1)
    prev.iloc[0] = float(eq.iloc[0])
    r = me / prev - 1.0
    tab = r.unstack(1).reindex(columns=range(1, 13))
    ye = eq.groupby(eq.index.year).last()
    yp = ye.shift(1)
    yp.iloc[0] = prev.iloc[0]
    tab["Year"] = ye / yp - 1.0
    return tab


def breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades grouped by side and by exit reason: where the money is made and lost."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["ret"] = t["return_pct"].astype(float) / 100.0
    t["hold_h"] = ((pd.to_datetime(t["exit_time"]) - pd.to_datetime(t["entry_time"])).dt.total_seconds() / 3600.0) if t["entry_time"].notna().any() else np.nan
    rows = []

    def add(label: str, g: pd.DataFrame, bold: bool = False):
        rows.append(dict(label=label, bold=bold, n=len(g), win=float((g.ret > 0).mean()), avg=float(g.ret.mean()), median=float(g.ret.median()), best=float(g.ret.max()), worst=float(g.ret.min()),
                         pnl=float(g.pnl_usd.sum()), hold=float(g.hold_h.mean()) if g.hold_h.notna().any() else float("nan")))

    add("All trades", t, True)
    for side, g in t.groupby(t.side.astype(str).str.lower()):
        add(f"{side.upper()}", g, True)
        for why, h in g.groupby("exit_reason"):
            add(f"   {side} · exit by {why}", h)
    return pd.DataFrame(rows)
