"""Backtest / live performance metrics from a trade list. All pure pandas/numpy."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


def _streaks(win: np.ndarray) -> dict:
    best_w = best_l = cur = 0
    runs_w: list[int] = []
    runs_l: list[int] = []
    prev = None
    for w in win:
        if w == prev:
            cur += 1
        else:
            if prev is not None:
                (runs_w if prev else runs_l).append(cur)
            cur, prev = 1, w
    if prev is not None:
        (runs_w if prev else runs_l).append(cur)
    best_w = max(runs_w, default=0)
    best_l = max(runs_l, default=0)
    current = (cur if prev else -cur) if prev is not None else 0
    return dict(max_consec_wins=best_w, max_consec_losses=best_l,
                avg_win_streak=float(np.mean(runs_w)) if runs_w else 0.0,
                avg_loss_streak=float(np.mean(runs_l)) if runs_l else 0.0,
                current_streak=int(current))


def equity_curve(trades: pd.DataFrame, capital: float, start_ts: int) -> pd.DataFrame:
    """Equity after each settled trade + the initial point. Columns: ts, equity."""
    if trades.empty:
        return pd.DataFrame({"ts": [start_ts], "equity": [capital]})
    t = trades.sort_values("settle_ts")
    return pd.DataFrame({"ts": np.r_[start_ts, t.settle_ts.values], "equity": np.r_[capital, t.equity_after.values]})


def drawdown_series(eq: pd.DataFrame) -> pd.Series:
    peak = eq.equity.cummax()
    return (eq.equity - peak) / peak


def evaluate_trades(trades: pd.DataFrame, capital: float, start_ts: int, end_ts: int, extra: dict | None = None):
    """(equity curve, drawdown series, metrics) - the one evaluation used by the backtest AND by live / paper sessions."""
    eq = equity_curve(trades, capital, start_ts)
    return eq, drawdown_series(eq), compute_metrics(trades, capital, start_ts, end_ts, extra)


def compute_metrics(trades: pd.DataFrame, capital: float, start_ts: int, end_ts: int, extra: dict | None = None) -> dict:
    """Returns a flat dict of metrics. ``trades`` needs: settle_ts, entry_ts, pnl, stake, fee, win, equity_after, f_full, f_used."""
    m: dict = dict(initial_capital=capital)
    days = max((end_ts - start_ts) / 86400.0, 1e-9)
    m["period_days"] = days
    n = len(trades)
    m["trades"] = n
    if extra:
        m.update(extra)
    if n == 0:
        m.update(final_equity=capital, total_return=0.0, total_pnl=0.0, max_drawdown=0.0)
        return m

    t = trades.sort_values("settle_ts").reset_index(drop=True)
    pnl = t.pnl.values.astype(float)
    win = (pnl > 0)
    final = float(t.equity_after.iloc[-1])
    eq = equity_curve(t, capital, start_ts)
    dd = drawdown_series(eq)

    # ── returns ─────────────────────────────────────────────────────────────
    m["final_equity"] = final
    m["total_pnl"] = final - capital
    m["total_return"] = final / capital - 1.0
    m["cagr"] = (final / capital) ** (DAYS_PER_YEAR / days) - 1.0 if final > 0 and days >= 1 else float("nan")
    m["fees_paid"] = float(t.fee.sum())

    # ── drawdown ────────────────────────────────────────────────────────────
    m["max_drawdown"] = float(dd.min())
    peak = eq.equity.cummax()
    m["max_drawdown_usd"] = float((eq.equity - peak).min())
    # longest time under water (days), including an unrecovered tail
    under = dd < -1e-12
    longest = 0.0
    start_i = None
    ts = eq.ts.values
    for i, u in enumerate(under.values):
        if u and start_i is None:
            start_i = i - 1 if i > 0 else i
        if (not u) and start_i is not None:
            longest = max(longest, (ts[i] - ts[start_i]) / 86400.0)
            start_i = None
    if start_i is not None:
        longest = max(longest, (end_ts - ts[start_i]) / 86400.0)
    m["max_dd_duration_days"] = float(longest)
    m["avg_drawdown"] = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    m["ulcer_index"] = float(np.sqrt((dd.values ** 2).mean()))

    # ── daily-return based ratios (calendar days, flat days = 0 return) ───────
    day0 = pd.Timestamp(start_ts, unit="s", tz="UTC").normalize()
    day1 = pd.Timestamp(end_ts, unit="s", tz="UTC").normalize()
    idx = pd.date_range(day0, day1, freq="D")
    s = pd.Series(eq.equity.values, index=pd.to_datetime(eq.ts.values, unit="s", utc=True))
    daily = s.groupby(s.index.normalize()).last().reindex(idx).ffill().fillna(capital)
    dr = daily.pct_change().dropna()
    m["daily_vol_ann"] = float(dr.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)) if len(dr) > 2 else float("nan")
    m["sharpe"] = float(dr.mean() / dr.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)) if len(dr) > 2 and dr.std(ddof=1) > 0 else float("nan")
    dn = dr[dr < 0]
    dd_dev = np.sqrt((np.minimum(dr.values, 0) ** 2).mean()) if len(dr) else 0.0
    m["sortino"] = float(dr.mean() / dd_dev * np.sqrt(DAYS_PER_YEAR)) if dd_dev > 0 else float("nan")
    m["calmar"] = float(m["cagr"] / abs(m["max_drawdown"])) if m["max_drawdown"] < 0 and np.isfinite(m["cagr"]) else float("nan")
    m["best_day"] = float(dr.max()) if len(dr) else 0.0
    m["worst_day"] = float(dr.min()) if len(dr) else 0.0
    m["pct_days_up"] = float((dr > 0).mean()) if len(dr) else 0.0

    # ── trade statistics ────────────────────────────────────────────────────
    wins, losses = pnl[win], pnl[~win]
    m["wins"], m["losses"] = int(win.sum()), int((~win).sum())
    m["win_rate"] = float(win.mean())
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    m["payoff_ratio"] = float(abs(m["avg_win"] / m["avg_loss"])) if m["avg_loss"] != 0 else float("nan")
    gp, gl = float(wins.sum()), float(-losses.sum())
    m["profit_factor"] = gp / gl if gl > 0 else float("inf") if gp > 0 else float("nan")
    m["expectancy"] = float(pnl.mean())
    ret = (t.pnl / t.stake.replace(0, np.nan)).dropna().values
    m["avg_return_per_trade"] = float(ret.mean()) if len(ret) else 0.0
    m["best_trade"] = float(pnl.max())
    m["worst_trade"] = float(pnl.min())
    m.update(_streaks(win))
    if len(ret) > 2 and ret.std(ddof=1) > 0:
        m["sqn"] = float(ret.mean() / ret.std(ddof=1) * np.sqrt(len(ret)))
        m["t_stat"] = m["sqn"]
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(ret, len(ret)).mean() for _ in range(1500)])
        m["ret_ci_lo"], m["ret_ci_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    else:
        m["sqn"] = m["t_stat"] = float("nan")
        m["ret_ci_lo"] = m["ret_ci_hi"] = float("nan")

    # ── sizing / exposure ───────────────────────────────────────────────────
    eq_before = t.equity_after.values - pnl
    stake_pct = t.stake.values / np.where(eq_before > 0, eq_before, np.nan)
    m["avg_stake_pct"] = float(np.nanmean(stake_pct))
    m["max_stake_pct"] = float(np.nanmax(stake_pct))
    m["avg_kelly_full"] = float(t.f_full.mean()) if "f_full" in t else float("nan")
    m["avg_kelly_used"] = float(t.f_used.mean()) if "f_used" in t else float("nan")
    hold = (t.settle_ts - t.entry_ts).clip(lower=0)
    m["avg_hold_min"] = float(hold.mean() / 60.0)
    m["exposure"] = float(hold.sum() / max(end_ts - start_ts, 1))
    m["trades_per_day"] = n / days
    return m
