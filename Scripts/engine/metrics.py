"""
Module 3: Performance Analytics & Interactive Plotting
======================================================
Computes institutional-grade risk/return metrics and renders an
interactive Plotly equity-curve chart with trade markers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class MetricsReport:
    """Container for all computed performance statistics."""
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    avg_drawdown_duration_days: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    period_years: float


class QuantMetrics:
    """Evaluates a backtest and produces metrics + interactive charts.

    Args:
        equity_curve: Daily equity DataFrame with column 'equity'.
        benchmark_prices: Raw benchmark OHLCV DataFrame (e.g., SPY).
        trade_ledger: DataFrame of fills from the backtester.
        risk_free_rate: Annualized risk-free rate (default 4%).
    """

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        benchmark_prices: pd.DataFrame,
        trade_ledger: pd.DataFrame,
        risk_free_rate: float = 0.04,
    ) -> None:
        self._equity = equity_curve["equity"]
        self._benchmark = benchmark_prices
        self._ledger = trade_ledger
        self._rf = risk_free_rate
        self._report: MetricsReport | None = None

    # ── metric computation ───────────────────────────────────────────────

    def compute(self) -> MetricsReport:
        """Calculate all performance metrics and return a MetricsReport."""
        eq = self._equity.dropna()
        if len(eq) < 2:
            raise ValueError("Equity curve has fewer than 2 data points.")

        start_val = eq.iloc[0]
        end_val = eq.iloc[-1]
        n_days = (eq.index[-1] - eq.index[0]).days
        n_years = n_days / 365.25

        # Returns
        total_ret = (end_val / start_val - 1) * 100
        cagr = ((end_val / start_val) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        # Drawdown
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = dd.min() * 100

        # Average drawdown duration
        in_dd = dd < 0
        dd_groups = (~in_dd).cumsum()
        durations = []
        for _, grp in in_dd.groupby(dd_groups):
            if grp.any():
                durations.append(grp.sum())
        avg_dd_dur = float(np.mean(durations)) if durations else 0.0

        # Daily returns for Sharpe / Sortino
        daily_ret = eq.pct_change().dropna()
        rf_daily = self._rf / 252
        excess = daily_ret - rf_daily

        sharpe = 0.0
        if excess.std() > 0:
            sharpe = (excess.mean() / excess.std()) * np.sqrt(252)

        sortino = 0.0
        downside = excess[excess < 0].std()
        if downside > 0:
            sortino = (excess.mean() / downside) * np.sqrt(252)

        # Trade-level metrics
        win_rate = 0.0
        profit_factor = 0.0
        total_trades = 0

        if not self._ledger.empty and "side" in self._ledger.columns:
            pnl = self._compute_trade_pnl()
            total_trades = len(pnl)
            if total_trades > 0:
                wins = pnl[pnl > 0]
                losses = pnl[pnl < 0]
                win_rate = len(wins) / total_trades * 100
                gross_profit = wins.sum() if len(wins) > 0 else 0
                gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
                profit_factor = (
                    gross_profit / gross_loss if gross_loss > 0 else float("inf")
                )

        self._report = MetricsReport(
            total_return_pct=total_ret,
            cagr_pct=cagr,
            max_drawdown_pct=max_dd,
            avg_drawdown_duration_days=avg_dd_dur,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            period_years=n_years,
        )
        return self._report

    def _compute_trade_pnl(self) -> pd.Series:
        """Match BUY/SELL pairs per symbol to compute per-trade P&L."""
        ledger = self._ledger.copy()
        pnl_list: list[float] = []

        for sym in ledger["symbol"].unique():
            sym_trades = ledger[ledger["symbol"] == sym].sort_values("timestamp")
            position_cost = 0.0
            position_qty = 0

            for _, row in sym_trades.iterrows():
                qty = row["quantity"]
                price = row["fill_price"]
                cost = row["cost"]

                if row["side"] == "BUY":
                    position_cost += price * qty + cost
                    position_qty += qty
                else:  # SELL
                    if position_qty > 0:
                        avg_entry = position_cost / position_qty
                        sell_qty = min(qty, position_qty)
                        trade_pnl = (price - avg_entry) * sell_qty - cost
                        pnl_list.append(trade_pnl)
                        position_qty -= sell_qty
                        position_cost = avg_entry * position_qty if position_qty > 0 else 0

        return pd.Series(pnl_list) if pnl_list else pd.Series(dtype=float)

    # ── formatted report ─────────────────────────────────────────────────

    def print_report(self) -> None:
        """Print a formatted performance summary to stdout."""
        r = self._report or self.compute()
        sep = "=" * 55
        print(f"\n{sep}")
        print("  PERFORMANCE REPORT")
        print(sep)
        print(f"  Period:              {r.period_years:.1f} years")
        print(f"  Total Return:        {r.total_return_pct:+.2f}%")
        print(f"  CAGR:                {r.cagr_pct:+.2f}%")
        print(f"  Max Drawdown:        {r.max_drawdown_pct:.2f}%")
        print(f"  Avg DD Duration:     {r.avg_drawdown_duration_days:.0f} days")
        print(f"  Sharpe Ratio:        {r.sharpe_ratio:.2f}  (rf={self._rf*100:.0f}%)")
        print(f"  Sortino Ratio:       {r.sortino_ratio:.2f}")
        print(f"  Total Trades:        {r.total_trades}")
        print(f"  Win Rate:            {r.win_rate_pct:.1f}%")
        print(f"  Profit Factor:       {r.profit_factor:.2f}")
        print(sep)

    # ── interactive plotly chart ─────────────────────────────────────────

    def plot_interactive_equity(self, title: str = "Strategy Performance") -> go.Figure:
        """Build and return an interactive Plotly figure.

        The chart overlays:
            - Strategy equity curve
            - Normalized Buy & Hold benchmark (scaled to starting capital)
            - Drawdown subplot
            - BUY (green) and SELL (red) trade markers

        Returns:
            A plotly Figure object.  Call fig.show() to open in browser.
        """
        eq = self._equity.dropna()
        start_val = eq.iloc[0]

        # Normalize benchmark to match starting capital
        bench = self._benchmark["close"].resample("D").last().dropna()
        # Align benchmark dates
        bench = bench.reindex(eq.index, method="ffill").dropna()
        bench_normalized = start_val * (bench / bench.iloc[0])

        # Drawdown series
        peak = eq.cummax()
        dd_pct = (eq - peak) / peak * 100

        # Build subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.75, 0.25],
            subplot_titles=["Equity Curve", "Drawdown (%)"],
        )

        # Strategy equity
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name="Strategy",
            line=dict(color="#2196F3", width=2),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        # Benchmark
        fig.add_trace(go.Scatter(
            x=bench_normalized.index, y=bench_normalized.values,
            name="Buy & Hold (normalized)",
            line=dict(color="gray", width=1.5, dash="dot"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>B&H: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        # Trade markers
        if not self._ledger.empty:
            buys = self._ledger[self._ledger["side"] == "BUY"]
            sells = self._ledger[self._ledger["side"] == "SELL"]

            # Map trade timestamps to closest equity date for Y position
            def _map_y(timestamps: pd.Series) -> list[float]:
                ys = []
                for t in timestamps:
                    idx = eq.index.get_indexer([t], method="nearest")[0]
                    ys.append(eq.iloc[idx] if idx >= 0 else np.nan)
                return ys

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["timestamp"], y=_map_y(buys["timestamp"]),
                    mode="markers", name="BUY",
                    marker=dict(color="green", size=7, symbol="triangle-up"),
                    hovertemplate=(
                        "BUY %{customdata[0]}<br>"
                        "Qty: %{customdata[1]}<br>"
                        "Price: $%{customdata[2]:.2f}<extra></extra>"
                    ),
                    customdata=buys[["symbol", "quantity", "fill_price"]].values,
                ), row=1, col=1)

            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["timestamp"], y=_map_y(sells["timestamp"]),
                    mode="markers", name="SELL",
                    marker=dict(color="red", size=7, symbol="triangle-down"),
                    hovertemplate=(
                        "SELL %{customdata[0]}<br>"
                        "Qty: %{customdata[1]}<br>"
                        "Price: $%{customdata[2]:.2f}<extra></extra>"
                    ),
                    customdata=sells[["symbol", "quantity", "fill_price"]].values,
                ), row=1, col=1)

        # Drawdown
        fig.add_trace(go.Scatter(
            x=dd_pct.index, y=dd_pct.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#E53935", width=1),
            hovertemplate="DD: %{y:.2f}%<extra></extra>",
        ), row=2, col=1)

        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            height=700,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

        return fig
