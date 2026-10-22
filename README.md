# Quant trading research

Three independent projects, kept side by side. All of it is research code: **not financial advice, no performance is guaranteed, and the trading apps can send real orders if you arm them.** Start with paper accounts and dry runs.

| Folder | What it is |
|---|---|
| `alpha_syndicate/` | Systematic strategy research and backtests (momentum, SOXL blend, Supertrend). `scripts/supertrend_app.py` is a PyQt6 backtest / live-trading / signal-monitor app for the Supertrend SOXL strategy (Alpaca), with a replay harness that checks the live engine trades exactly like the live-accurate backtest. |
| `prediction_arb/` | Kalshi x Polymarket cross-venue arbitrage scanner and backtester (read-only, public endpoints). See `prediction_arb/README.md`. |
| `15minbtcpolymarket/` | Research and a PyQt6 trader for 15-minute BTC up/down markets. See `15minbtcpolymarket/TRADER_README.md`. |

## What is not in this repository

- API keys, key files and tokens (never committed; the apps read them from your own machine).
- Files over GitHub's 100 MB limit: `prediction_arb/arb_data.db`, `15minbtcpolymarket/data/tape/KXBTC15M.csv` and `KXETH15M.csv`. Rebuild them with the collectors in each project.
- Private runtime records: live trade ledgers and session logs, pre-flight reports and local GUI state.

## Running the tests

Each project has its own tests (`python tests/test_core.py`, `python test_trader.py`, and `python alpha_syndicate/scripts/test_supertrend_gui.py`, `test_supertrend_replay.py`). They use fake brokers / throw-away settings and send nothing to any account.
