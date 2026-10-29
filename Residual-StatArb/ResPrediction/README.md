# Residual Prediction

A walk-forward model (ridge regression or gradient-boosted trees) forecasts each member's residual return over the next horizon from residual-based features; long the best forecasts, short the worst, hedged. Krauss, Do & Huck (2017); Gu, Kelly & Xiu (2020).

* **Run the app**: double-click `launch_resprediction.bat` (PAPER + DRY RUN at start).
* **Its own paper account**: `setx RESPRED_ALPACA_API_KEY <key>` and `setx RESPRED_ALPACA_SECRET_KEY <secret>`, then open a new console. Without them the app runs DRY only (data + plans with the shared keys).
* **Headless session**: `run_live.bat` (DRY by default; the ARMED line is commented out). Save the GUI settings first (*File > Save settings* → `ResPrediction_settings.json` in this folder) - the runner reads them.
* **Files**: `strategy_respred.py` (the alpha; everything else is shared in `../core/resarb`), `reports/` (journal, state, comparisons), `REPORT.md` (research, team review, verdict), `models/` (walk-forward cache, the live model).
* Shared machinery, safety rules and the data plan: `../README.md`.
