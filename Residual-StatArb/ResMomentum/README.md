# Residual Momentum

Buys the stocks whose residual return (the part SPY and the sector ETF do not explain) was strongest over the formation window, shorts the weakest, and hedges market and sector beta with SPY and the sector SPDRs. Blitz, Huij & Martens (2011).

* **Run the app**: double-click `launch_resmomentum.bat` (PAPER + DRY RUN at start).
* **Its own paper account**: `setx RESMOM_ALPACA_API_KEY <key>` and `setx RESMOM_ALPACA_SECRET_KEY <secret>`, then open a new console. Without them the app runs DRY only (data + plans with the shared keys).
* **Headless session**: `run_live.bat` (DRY by default; the ARMED line is commented out). Save the GUI settings first (*File > Save settings* → `ResMomentum_settings.json` in this folder) - the runner reads them.
* **Files**: `strategy_resmom.py` (the alpha; everything else is shared in `../core/resarb`), `reports/` (journal, state, comparisons), `REPORT.md` (research, team review, verdict).
* Shared machinery, safety rules and the data plan: `../README.md`.
