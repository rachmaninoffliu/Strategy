# Stats-arb — statistical arbitrage (Avellaneda & Lee 2008) + 2026 risk overlays

Self-contained: everything this strategy needs lives in this folder.

| Path | What |
|---|---|
| `launch_statarb.bat` | Starts the GUI (Backtest + Alpaca live trading). Always starts on **Paper + DRY RUN**. |
| `STATARB_REPORT.md` | Team report: verdict, evidence, risk map, reviewer findings. Read first. |
| `scripts/` | Code: `statarb_data.py` (bars, earnings), `statarb_broad.py` (survivorship-reduced universe), `statarb_core.py` (signals), `statarb_sim.py` (positions, simulator, metrics), `statarb_bridge.py`, `statarb_live.py` (Alpaca engine), `statarb_report.py` (reproducible run), `statarb_app.py` + `statarb_ui/` (GUI), `test_statarb.py` (20 tests). `supertrend_ui/` is a minimal copy of the shared dark theme. |
| `data/` | Cached Alpaca daily bars, Yahoo earnings calendar, asset list, sector labels (re-creatable; ~160 MB). |
| `reports/` | Backtest runs (`final*_broad|core`), signal cache (`cache/`, several GB, safe to delete), live state/ledger and order logs once you trade (`statarb_state_paper_*.json`, `statarb_orders_paper.csv`). |
| `research/` | The exploratory scripts behind the report (IC studies, grids, event studies). Scratch-quality; `rlib.py` is their helper. |

Run: double-click `launch_statarb.bat`, or `python scripts\statarb_app.py`.
Tests: `cd scripts && python -m unittest test_statarb` (needs the Python 3.12 install with PyQt6, alpaca-py, pandas, scipy).
Reproduce the headline numbers: `cd scripts && python statarb_report.py --universe broad --trials 61`.
Status: Valid has not signed off for live capital; paper-trade only.
