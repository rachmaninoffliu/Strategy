"""
First-Hour Breakout Strategy on SOXL (Overnight Carry)
=======================================================
Logic:
    1. Setup: Isolate the 09:30-10:30 candle each day.
    2. Signal at 10:30:
       - LONG  if 1st-hour Close > Open
       - SHORT if 1st-hour Close < Open
    3. Execution at 10:30 bar open with 1:1 R/R:
       - LONG:  SL = 1st-hour Low,  TP = Entry + (Entry - SL)
       - SHORT: SL = 1st-hour High, TP = Entry - (SL - Entry)
    4. If TP/SL not hit by EOD, carry position overnight.
    5. Next day at 10:30, re-evaluate:
       - Same direction signal: update SL/TP to new candle levels.
       - Opposite signal: close position, flip to new direction.

Run:  python -m strategies.first_hour_breakout   (from Scripts/)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import DataFetcher

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_date: str
    exit_date: str
    direction: str       # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_price: float
    exit_reason: str     # "TP", "SL", "FLIP", "FINAL"
    exit_time: str
    quantity: float
    pnl: float
    hold_days: int
    equity_before: float
    equity_after: float


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy
# ═══════════════════════════════════════════════════════════════════════════════

def _build_first_hour_candle(day_bars: pd.DataFrame) -> dict | None:
    """Aggregate 15-min bars from 09:30 to 10:15 into a single first-hour candle.

    The four bars covering 09:30-10:30 are: 09:30, 09:45, 10:00, 10:15.
    Returns dict with open, high, low, close or None if bars are missing.
    """
    first_hour = day_bars.between_time("09:30", "10:15")
    if len(first_hour) < 4:
        return None
    return {
        "open": first_hour.iloc[0]["open"],
        "high": first_hour["high"].max(),
        "low": first_hour["low"].min(),
        "close": first_hour.iloc[-1]["close"],
    }


def _check_sl_tp(bar, direction: str, stop_loss: float, take_profit: float):
    """Check if a bar triggers SL or TP. Returns (exit_price, reason) or (None, None)."""
    bar_high, bar_low, bar_open = bar["high"], bar["low"], bar["open"]

    if direction == "LONG":
        sl_hit = bar_low <= stop_loss
        tp_hit = bar_high >= take_profit
        if sl_hit and tp_hit:
            if abs(bar_open - stop_loss) <= abs(bar_open - take_profit):
                return stop_loss, "SL"
            return take_profit, "TP"
        if sl_hit:
            return stop_loss, "SL"
        if tp_hit:
            return take_profit, "TP"
    else:
        sl_hit = bar_high >= stop_loss
        tp_hit = bar_low <= take_profit
        if sl_hit and tp_hit:
            if abs(bar_open - stop_loss) <= abs(bar_open - take_profit):
                return stop_loss, "SL"
            return take_profit, "TP"
        if sl_hit:
            return stop_loss, "SL"
        if tp_hit:
            return take_profit, "TP"

    return None, None


def _compute_levels(direction: str, entry_price: float, candle: dict):
    """Compute SL and TP for a direction given the first-hour candle.

    Returns (stop_loss, take_profit) or (None, None) if risk <= 0.
    """
    if direction == "LONG":
        sl = candle["low"]
        risk = entry_price - sl
        if risk <= 0:
            return None, None
        return sl, entry_price + risk
    else:
        sl = candle["high"]
        risk = sl - entry_price
        if risk <= 0:
            return None, None
        return sl, entry_price - risk


def run_first_hour_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    sizing_mode: str = "compound",
    fixed_shares: int | None = None,
    fixed_capital: float | None = None,
) -> tuple[list[Trade], pd.Series]:
    """Run the first-hour breakout strategy with overnight carry.

    Positions are NOT closed at EOD. Instead, the next day's first-hour
    candle determines whether to:
      - Update SL/TP (same direction signal)
      - Flip position (opposite direction signal)

    Args:
        df: 15-min OHLCV DataFrame with Eastern-timezone DatetimeIndex.
        initial_capital: Starting equity.
        sizing_mode: Position sizing method:
            - "compound": 100% of current equity each trade (compounds).
            - "fixed_shares": Trade a fixed number of shares every time.
            - "fixed_capital": Trade a fixed dollar amount every time.
        fixed_shares: Number of shares when sizing_mode="fixed_shares".
        fixed_capital: Dollar amount when sizing_mode="fixed_capital".

    Returns:
        Tuple of (trade_list, daily_equity Series).
    """
    if sizing_mode == "fixed_shares" and not fixed_shares:
        raise ValueError("fixed_shares must be set when sizing_mode='fixed_shares'")
    if sizing_mode == "fixed_capital" and not fixed_capital:
        raise ValueError("fixed_capital must be set when sizing_mode='fixed_capital'")

    equity = initial_capital
    trades: list[Trade] = []
    daily_equity: dict[str, float] = {}

    def _size_position(price: float) -> float:
        """Compute position quantity based on sizing mode."""
        if sizing_mode == "fixed_shares":
            return float(fixed_shares)
        elif sizing_mode == "fixed_capital":
            return fixed_capital / price
        else:  # compound
            return equity / price

    # Position state (persists across days)
    pos_dir: str | None = None       # "LONG" or "SHORT" or None
    pos_entry: float = 0.0
    pos_qty: float = 0.0
    pos_sl: float = 0.0
    pos_tp: float = 0.0
    pos_entry_date: str = ""
    pos_equity_before: float = 0.0

    def _close_position(exit_px: float, reason: str, exit_ts: str, date_str: str):
        """Record a trade and reset position state."""
        nonlocal equity, pos_dir, pos_entry, pos_qty, pos_sl, pos_tp
        nonlocal pos_entry_date, pos_equity_before

        if pos_dir == "LONG":
            pnl = (exit_px - pos_entry) * pos_qty
        else:
            pnl = (pos_entry - exit_px) * pos_qty

        equity_before = pos_equity_before
        equity += pnl

        entry_d = pd.Timestamp(pos_entry_date)
        exit_d = pd.Timestamp(date_str)
        hold = max(1, (exit_d - entry_d).days)

        trades.append(Trade(
            entry_date=pos_entry_date,
            exit_date=date_str,
            direction=pos_dir,
            entry_price=pos_entry,
            stop_loss=pos_sl,
            take_profit=pos_tp,
            exit_price=exit_px,
            exit_reason=reason,
            exit_time=exit_ts,
            quantity=pos_qty,
            pnl=pnl,
            hold_days=hold,
            equity_before=equity_before,
            equity_after=equity,
        ))

        pos_dir = None

    df = df.copy()
    df["date"] = df.index.date
    dates = sorted(df["date"].unique())

    for date in dates:
        day_bars = df[df["date"] == date].sort_index()
        date_str = str(date)

        candle = _build_first_hour_candle(day_bars)
        remaining = day_bars.between_time("10:30", "15:45")

        # ── If we have a position, scan pre-10:30 bars (09:30-10:15)
        # for SL/TP hits during the first hour itself
        if pos_dir is not None:
            first_hour_bars = day_bars.between_time("09:30", "10:15")
            for bar_ts, bar in first_hour_bars.iterrows():
                exit_px, reason = _check_sl_tp(bar, pos_dir, pos_sl, pos_tp)
                if exit_px is not None:
                    _close_position(exit_px, reason, str(bar_ts), date_str)
                    break

        # ── At 10:30: evaluate signal and manage position
        if candle is not None:
            s_signal = None
            if candle["close"] > candle["open"]:
                s_signal = "LONG"
            elif candle["close"] < candle["open"]:
                s_signal = "SHORT"

            if s_signal is not None and not remaining.empty:
                ref_price = remaining.iloc[0]["open"]  # 10:30 bar open

                if pos_dir is None:
                    # ── FLAT: enter new position
                    sl, tp = _compute_levels(s_signal, ref_price, candle)
                    if sl is not None:
                        pos_dir = s_signal
                        pos_entry = ref_price
                        pos_qty = _size_position(ref_price)
                        pos_sl = sl
                        pos_tp = tp
                        pos_entry_date = date_str
                        pos_equity_before = equity

                elif pos_dir == s_signal:
                    # ── SAME direction: update SL/TP based on new candle
                    sl, tp = _compute_levels(s_signal, ref_price, candle)
                    if sl is not None:
                        pos_sl = sl
                        pos_tp = tp

                else:
                    # ── OPPOSITE direction: flip position
                    _close_position(ref_price, "FLIP", str(remaining.index[0]), date_str)

                    sl, tp = _compute_levels(s_signal, ref_price, candle)
                    if sl is not None:
                        pos_dir = s_signal
                        pos_entry = ref_price
                        pos_qty = _size_position(ref_price)
                        pos_sl = sl
                        pos_tp = tp
                        pos_entry_date = date_str
                        pos_equity_before = equity

        # ── Scan remaining bars (10:30+) for SL/TP
        if pos_dir is not None and not remaining.empty:
            for bar_ts, bar in remaining.iterrows():
                exit_px, reason = _check_sl_tp(bar, pos_dir, pos_sl, pos_tp)
                if exit_px is not None:
                    _close_position(exit_px, reason, str(bar_ts), date_str)
                    break

        # ── Mark equity at end of day (use last known price)
        if pos_dir is not None and not day_bars.empty:
            last_close = day_bars.iloc[-1]["close"]
            if pos_dir == "LONG":
                unrealized = (last_close - pos_entry) * pos_qty
            else:
                unrealized = (pos_entry - last_close) * pos_qty
            daily_equity[date_str] = pos_equity_before + unrealized
        else:
            daily_equity[date_str] = equity

    # ── Close any remaining position at last bar's close
    if pos_dir is not None and not df.empty:
        last_bar = df.iloc[-1]
        _close_position(
            last_bar["close"], "FINAL",
            str(df.index[-1]), str(df.index[-1].date()),
        )

    eq_series = pd.Series(daily_equity, name="equity")
    eq_series.index = pd.to_datetime(eq_series.index)
    return trades, eq_series


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(trades: list[Trade], equity: pd.Series, initial: float) -> None:
    if not trades:
        print("  No trades generated.")
        return

    df = pd.DataFrame([t.__dict__ for t in trades])
    n = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]
    flat = df[df["pnl"] == 0]

    total_pnl = df["pnl"].sum()
    win_rate = len(wins) / n * 100
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    profit_factor = (
        wins["pnl"].sum() / abs(losses["pnl"].sum())
        if len(losses) > 0 and losses["pnl"].sum() != 0
        else float("inf")
    )

    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()

    n_days = (equity.index[-1] - equity.index[0]).days
    n_years = n_days / 365.25
    final = equity.iloc[-1]
    cagr = ((final / initial) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    longs = df[df["direction"] == "LONG"]
    shorts = df[df["direction"] == "SHORT"]
    long_wr = len(longs[longs["pnl"] > 0]) / len(longs) * 100 if len(longs) > 0 else 0
    short_wr = len(shorts[shorts["pnl"] > 0]) / len(shorts) * 100 if len(shorts) > 0 else 0

    tp_count = len(df[df["exit_reason"] == "TP"])
    sl_count = len(df[df["exit_reason"] == "SL"])
    flip_count = len(df[df["exit_reason"] == "FLIP"])
    final_count = len(df[df["exit_reason"] == "FINAL"])
    avg_hold = df["hold_days"].mean()

    sep = "=" * 60
    print(f"\n{sep}")
    print("  FIRST-HOUR BREAKOUT (OVERNIGHT CARRY) -- REPORT")
    print(sep)
    print(f"  Period:              {n_years:.1f} years ({n_days} days)")
    print(f"  Initial Capital:     ${initial:,.2f}")
    print(f"  Final Equity:        ${final:,.2f}")
    print(f"  Total P&L:           ${total_pnl:+,.2f}")
    print(f"  Total Return:        {(final/initial - 1)*100:+.2f}%")
    print(f"  CAGR:                {cagr:+.2f}%")
    print(f"  Max Drawdown:        {max_dd:.2f}%")
    print(f"  Sharpe Ratio:        {sharpe:.2f}")
    print(sep)
    print(f"  Total Trades:        {n}")
    print(f"  Win Rate:            {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L / {len(flat)}F)")
    print(f"  Avg Win:             ${avg_win:+,.2f}")
    print(f"  Avg Loss:            ${avg_loss:+,.2f}")
    print(f"  Profit Factor:       {profit_factor:.2f}")
    print(f"  Avg Hold Days:       {avg_hold:.1f}")
    print(sep)
    print(f"  LONG trades:         {len(longs)}  (WR: {long_wr:.1f}%)")
    print(f"  SHORT trades:        {len(shorts)}  (WR: {short_wr:.1f}%)")
    print(f"  Exits -- TP: {tp_count}  SL: {sl_count}  FLIP: {flip_count}  FINAL: {final_count}")
    print(sep)

    results = (df["pnl"] > 0).astype(int)
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for r in results:
        if r == 1:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)
    print(f"  Max Win Streak:      {max_win_streak}")
    print(f"  Max Loss Streak:     {max_loss_streak}")
    print(sep)


def plot_results(
    trades: list[Trade], equity: pd.Series, title: str
) -> go.Figure:
    if not trades:
        fig = go.Figure()
        fig.update_layout(title="No trades to display")
        return fig
    df = pd.DataFrame([t.__dict__ for t in trades])
    df["date"] = pd.to_datetime(df["exit_date"])

    peak = equity.cummax()
    dd_pct = (equity - peak) / peak * 100

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.50, 0.25, 0.25],
        subplot_titles=["Equity Curve", "Drawdown (%)", "Per-Trade P&L"],
    )

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Equity", line=dict(color="#2196F3", width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dd_pct.index, y=dd_pct.values,
        name="Drawdown", fill="tozeroy",
        line=dict(color="#E53935", width=1),
        hovertemplate="DD: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    colors = ["green" if p > 0 else "red" for p in df["pnl"]]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["pnl"],
        name="Trade P&L",
        marker_color=colors,
        hovertemplate="Date: %{x|%Y-%m-%d}<br>P&L: $%{y:,.2f}<extra></extra>",
    ), row=3, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=850, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        showlegend=True,
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    API_KEY    = "PKT6IGOQCNEYYFSQC5BZ3MJTLQ"
    SECRET_KEY = "GihTba9i4XTsDAqACPQ1GgY3HiMGzbRQTmyvAaoPPCFZ"
    SYMBOL     = "SOXL"
    START      = "2020-01-01"
    END        = "2026-03-30"
    CAPITAL    = 10_000.0

    cache_dir = Path(__file__).resolve().parent.parent / "cache"
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    # ── Fetch Data ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FETCHING SOXL 15-MIN DATA (for 1hr candle construction)")
    print("=" * 60)

    fetcher = DataFetcher(API_KEY, SECRET_KEY, cache_dir=cache_dir, feed="iex")
    data = fetcher.fetch(
        symbols=SYMBOL, start=START, end=END,
        timeframe_minutes=15, include_extended_hours=False,
        use_cache=True,
    )

    df = data.get(SYMBOL)
    if df is None or df.empty:
        print("  ERROR: No data for SOXL.")
        return

    print(f"  {SYMBOL}: {len(df):,} bars | {df.index.min().date()} -> {df.index.max().date()}")

    # ── Define sizing modes to compare ──────────────────────────────────
    # fixed_shares: ~200 shares ≈ $10K / $50 average SOXL price
    # fixed_capital: $10K per trade (no compounding)
    modes = [
        {"label": "Compound (100% equity)",
         "kwargs": dict(sizing_mode="compound")},
        {"label": "Fixed Shares (200)",
         "kwargs": dict(sizing_mode="fixed_shares", fixed_shares=200)},
        {"label": "Fixed Capital ($10K)",
         "kwargs": dict(sizing_mode="fixed_capital", fixed_capital=10_000.0)},
    ]

    all_equities: dict[str, pd.Series] = {}

    for mode in modes:
        print("\n" + "=" * 60)
        print(f"  MODE: {mode['label']}")
        print("=" * 60)

        trades, equity = run_first_hour_strategy(
            df, initial_capital=CAPITAL, **mode["kwargs"]
        )

        print(f"  Trades executed: {len(trades)}")
        print_report(trades, equity, CAPITAL)
        all_equities[mode["label"]] = equity

        # Save per-mode CSVs
        tag = mode["kwargs"]["sizing_mode"]
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(
            out_dir / f"first_hour_{tag}_trades.csv", index=False
        )
        equity.to_csv(out_dir / f"first_hour_{tag}_equity.csv")

    # ── Combined comparison chart ───────────────────────────────────────
    print("\n  Generating comparison chart ...")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=["Equity Curves by Sizing Mode", "Drawdown (%)"],
    )

    colors = {"Compound (100% equity)": "#2196F3",
              "Fixed Shares (200)": "#FF9800",
              "Fixed Capital ($10K)": "#4CAF50"}

    for label, eq in all_equities.items():
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=label, line=dict(color=colors.get(label, "#999"), width=2),
            hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}: $%{{y:,.2f}}<extra></extra>",
        ), row=1, col=1)

        peak = eq.cummax()
        dd_pct = (eq - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=dd_pct.index, y=dd_pct.values,
            name=f"{label} DD", line=dict(color=colors.get(label, "#999"), width=1),
            showlegend=False,
            hovertemplate=f"{label} DD: %{{y:.2f}}%<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text="First-Hour Breakout: SOXL | Sizing Mode Comparison",
            font=dict(size=18),
        ),
        height=750, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    chart_path = out_dir / "first_hour_sizing_comparison.html"
    fig.write_html(str(chart_path))
    print(f"  Chart saved -> {chart_path}")

    try:
        fig.show()
    except Exception:
        print("  (Open the chart HTML in browser manually.)")

    print(f"  Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
