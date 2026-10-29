# Quant trading research

Independent projects, kept side by side. All of it is research code: **not financial advice, no performance is guaranteed, and the trading apps can send real orders if you arm them.** Start with paper accounts and dry runs.

| Folder | What it is |
|---|---|
| `alpha_syndicate/` | Systematic strategy research and backtests (momentum, SOXL blend, Supertrend). `scripts/supertrend_app.py` is a PyQt6 backtest / live-trading / signal-monitor app for the Supertrend SOXL strategy (Alpaca), with a replay harness that checks the live engine trades exactly like the live-accurate backtest. |
| `prediction_arb/` | Kalshi x Polymarket cross-venue arbitrage scanner and backtester (read-only, public endpoints). See `prediction_arb/README.md`. |
| `15minbtcpolymarket/` | Research and a PyQt6 trader for 15-minute BTC up/down markets. See `15minbtcpolymarket/TRADER_README.md`. |
| `Stats-arb/` | Avellaneda & Lee (2008) statistical arbitrage with 2026 risk overlays, GUI and Alpaca engine. See `Stats-arb/README.md` and `STATARB_REPORT.md`. |
| `LevETF-Pairs/` | The same residual / s-score construction on TQQQ/QQQ, SOXL/SMH, SPXL/SPY intraday bars, with a PyQt6 app and paper engine. See `LevETF-Pairs/README.md`. |
| `Residual-StatArb/` | Three U.S.-equity strategies on factor-model residuals (momentum, mean reversion, prediction) on one shared core, plus the `MeanReversion` and `MomentumBook` apps. See `Residual-StatArb/README.md`. |
| `Intraday-A3C/` | Demir et al. (2022) session-convergence idea ported to U.S. equities, with a dry-run paper runner. See `Intraday-A3C/DESIGN.md`. |
| `Scripts/` | Small early backtest engine and a first-hour breakout strategy. |
| `tv_editor_backups/` | Backups of TradingView Pine editor buffers. |

## What is not in this repository

- API keys, key files and tokens (never committed; the apps read them from your own machine).
- Files over GitHub's 100 MB limit: `prediction_arb/arb_data.db`, `15minbtcpolymarket/data/tape/KXBTC15M.csv` and `KXETH15M.csv`. Rebuild them with the collectors in each project.
- Cached market data and result pickles of `Stats-arb/`, `LevETF-Pairs/`, `Residual-StatArb/`, `Intraday-A3C/` and `Scripts/` (`.parquet`, `.pkl`, `.npy`, `.npz`, about 13 GB). Rebuild them with each project's download and research scripts.
- Private runtime records: live trade ledgers and session logs, paper-account state and journals, pre-flight reports and local GUI state.

## Running the tests

Each project has its own tests (`python tests/test_core.py`, `python test_trader.py`, and `python alpha_syndicate/scripts/test_supertrend_gui.py`, `test_supertrend_replay.py`). They use fake brokers / throw-away settings and send nothing to any account.
