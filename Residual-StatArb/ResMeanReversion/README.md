# Residual Mean Reversion

> This residual (long-short, hedged, up to 1.9x gross) version was **rejected**: its edge is below the spread out-of-sample. For mean reversion, use the newer **`../MeanReversion/`** app. It is long-only, unlevered, trades in the closing auction, and has a model picker and a full trade list (Sharpe 1.21 over 2016-2026, see its REPORT.md).

Fits an Ornstein-Uhlenbeck model to each stock's cumulative 15-minute residual and bets on the snap-back when the s-score is stretched (or ranks the residual return of the last k bars cross-sectionally); hedged with SPY and the sector SPDRs. Avellaneda & Lee (2010).

* **Run the app**: double-click `launch_resmeanreversion.bat` (PAPER + DRY RUN at start).
* **Its own paper account**: `setx RESMR_ALPACA_API_KEY <key>` and `setx RESMR_ALPACA_SECRET_KEY <secret>`, then open a new console. Without them the app runs DRY only (data + plans with the shared keys).
* **Headless session**: `run_live.bat` (DRY by default; the ARMED line is commented out). Save the GUI settings first (*File > Save settings* → `ResMeanReversion_settings.json` in this folder) - the runner reads them.
* **Files**: `strategy_resmr.py` (the alpha; everything else is shared in `../core/resarb`), `reports/` (journal, state, comparisons), `REPORT.md` (research, team review, verdict).
* Shared machinery, safety rules and the data plan: `../README.md`.
