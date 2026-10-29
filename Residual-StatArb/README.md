# Residual Stat-Arb: three strategies on factor-model residuals

Three U.S.-equity strategies that **extract beta and trade the residual**. Each one has its own folder, launcher, settings, reports and Alpaca paper account. All three share one core: universe, betas, residuals, features, portfolio construction, backtest, live engine and GUI.

| Folder | Strategy | Alpha (what is traded) | Main papers |
|---|---|---|---|
| `ResMomentum/` | Residual Momentum | long the strongest residual performers, short the weakest (formation window, skip the last weeks) | Blitz, Huij & Martens (2011); Lou, Polk & Skouras (2019) |
| `ResMeanReversion/` | Residual Mean Reversion | Ornstein-Uhlenbeck s-score of the cumulative residual (or cross-sectional residual reversal) | Avellaneda & Lee (2010); Yeo & Papanicolaou; Da, Liu & Schaumburg (2014); Lehmann (1990) |
| `ResPrediction/` | Residual Prediction | walk-forward model (ridge / gradient boosting) forecasting the next residual return from 28 features | Krauss, Do & Huck (2017); Gu, Kelly & Xiu (2020); Guijarro-Ordonez, Pelger & Zanotti (2021) |

Results, the team review and the verdict for each strategy are in its folder's `REPORT.md`. The cross-strategy summary is at the end of this file.

## How the residual is built (shared)
* **Universe**: at each month end, the N most liquid U.S. stocks become next month's members (default 100). The ranking uses 63-day median dollar volume, price ≥ $10, ≥ 126 sessions of history and one share class. It is point-in-time and survivorship-free: the history comes from the Stats-arb project's daily cache, delisted names included.
* **Beta extraction** (Avellaneda & Lee's ETF model): every morning, a regression of each stock's daily returns over the last 126 / 252 days, ending at the previous close: `r = a + b_m·r_SPY + b_s·(r_sector − r_SPY) + e`. Option `pca`: the top-k eigenportfolios of the members' correlation matrix.
* **Residual**: the day's betas are applied to every 15-minute return of the day; `e` is the alpha the strategies trade. The book hedges the factor exposure: per $1 of stock, short `b_m − b_s` $ of SPY and `b_s` $ of the sector SPDR (or project the weights onto the factor null space).
* **Bars**: 15-minute regular-hours bars (SIP, split + dividend adjusted), 2016 → now. Strategies re-decide every 15 min / 30 min / 1 h / 2 h / daily / weekly / monthly.

## Timing and the free Alpaca data plan
A decision made from bars up to *t* fills at the **open of the next bar**, in the backtest and in live. Alpaca's free plan gives real-time **IEX** bars (about 2-3% of U.S. volume, so prints can be seconds old) and **SIP** bars (all exchanges) only when they are older than 15 minutes. 15-minute *candles* are not the restriction. Setting **Live data feed**:
* `iex`: no delay, noisier bars. The study `research/r5_feedgap.py` measures what this costs each strategy.
* `sip_delayed`: accurate bars, but every trade happens one bar later. The backtest models that delay (lag 1).
* `sip_realtime`: Alpaca's paid plan.

## Folder layout
```
Residual-StatArb/
  core/resarb/        data.py factors.py features.py portfolio.py backtest.py strategy.py research.py live.py compare.py run.py   ui/ (PyQt6 GUI)
  core/tests/         test_resarb.py (synthetic data + a fake broker: no network, no orders)
  data/               bars15/ (SIP 15-min, one file per symbol), bars15_iex/, resarb_universe.parquet, resarb_sectors.parquet
  research/           r0_spreads.py r1_feature_ic.py r2_momentum.py r3_meanrev.py r4_predict.py r5_feedgap.py, reports/
  ResMomentum/        strategy_resmom.py  app.py  launch_resmomentum.bat  run_live.py  run_live.bat  REPORT.md  reports/
  ResMeanReversion/   strategy_resmr.py   app.py  launch_resmeanreversion.bat ...
  ResPrediction/      strategy_respred.py app.py  launch_resprediction.bat ...  models/
  MeanReversion/      app.py  launch_meanreversion.bat  run_live.py  run_live.bat  README.md  REPORT.md  reports/   (code: core/resarb/mr/, tests: core/tests/test_mr.py)
  MomentumBook/       app.py  launch_momentumbook.bat  run_live.py  run_live.bat  README.md  REPORT.md  models/  reports/   (code: core/resarb/mom/, tests: core/tests/test_mom.py)
```

## Using it
1. **Data**: `cd core` then `python -m resarb.data universe` (monthly members), `python -m resarb.data download` (15-min SIP bars, about 1 h the first time; Alpaca serves at most ~1,000 fifteen-minute bars per request), and optionally `python -m resarb.data iex`. Afterwards the GUI's *Update data* button tops up.
2. **GUI**: double-click `launch_<strategy>.bat` in the strategy's folder. Left: the settings, starting with the strategy's own. Tabs:
   * *Live trading*: connect, pre-flight, plan this bar (dry run), start/stop the 15-minute loop, train the model (prediction only), compare the last session with the backtest, KILL.
   * *Backtest*: KPIs, in-sample / out-of-sample, equity, cost sensitivity, years, names, exposure.
   * *Research*: rank IC by horizon, deciles, every feature.
   * *Signals*: the latest scores.
   * *How it works*.
3. **Headless**: `run_live.bat` runs a DRY-RUN session from 09:00 to 16:05 ET and journals every bar to `reports/journal_paper.jsonl`. Afterwards *Compare last session with the backtest* (or `python -m resarb.compare`) rebuilds the backtest's decisions for the same bars and reports target agreement and fill slippage.

## Safety (live)
* The app starts on **PAPER + DRY RUN**. ARM needs the typed phrase `ARM PAPER`. **Real money is disabled in code** (it needs `RESARB_ALLOW_LIVE=1`); Valid has signed off none of the three strategies.
* **Keys.** Every app in the project folder reads `Alpaca Kay.txt` / `Alpaca Key.txt` (lines `Key ID:` / `Secret Key:`, git-ignored) via `core/resarb/alpaca_keys.py`. A strategy's own `RESMOM_ALPACA_API_KEY` / `_SECRET_KEY` (`RESMR_…`, `RESPRED_…`) variables override it, giving that strategy a separate account.
* **One armed strategy per account.** An armed engine refuses the `~/.claude.json` fallback keys, and refuses an account another residual strategy has claimed (`data/resarb_accounts.json`): two strategies on one account would net and close each other's positions. The other apps in the folder (Supertrend, LevETF-Pairs, Stats-arb) use the same key file but not this registry, so don't arm them on the account at the same time.
* **Target-based engine.** Every rebalance cancels the engine's own open orders, reads the broker's positions, and trades only the difference to the target. An order whose answer was lost is looked up by its client id and self-corrects at the next bar.
* **Order rules.** Reducing orders go first, and no position flips sign in one order. Shorts only in easy-to-borrow names, whole shares, a minimum ticket, a gross cap (default 2.5x) and at most 150 orders per rebalance.
* **Limits.** Daily-loss halt (no new risk), drawdown stop (flatten, stay flat until reset), stale-data guard, HTTP timeouts, API throttle, and one engine per strategy (an operating-system lock).
* **Scope.** The engine trades only its own universe and the hedge ETFs; other symbols in the account are reported and never touched. KILL cancels and flattens the strategy's symbols.

## Results (2016-09 → 2026-10, top-100, ETF residuals, spread-based costs; in-sample → 2022, out-of-sample 2023 →)
| Strategy | Choice (in-sample best of N tried) | Sharpe IS / OOS / all | Before costs (all) | CAGR, vol, max DD | Turnover | Verdict |
|---|---|---|---|---|---|---|
| Residual Momentum | 3-1 month, monthly, top/bottom 10% (of 105) | 0.37 / **0.60** / 0.47 | 0.51 | 4.3%, 10.2%, -16% | 21x | paper forward test; no capital (DSR 0.05-0.43) |
| Residual Mean Reversion | last-hour reversal, daily at 15:45 (of 152) | 0.96 / **-0.20** / 0.47 | 1.13 | 4.3%, 10.2%, -17% | 543x | rejected: the edge is below the spread out-of-sample |
| Residual Prediction | ridge, 15 features, 5-day horizon, daily (of 25) | 0.41 / **-0.08** / 0.19 | 0.57 | 1.2%, 7.5%, -18% | 211x | rejected: no net edge; the feature research is the deliverable |

**`MeanReversion/`**: a separate, simpler app, built after these three. Long-only daily mean reversion (IBS < 0.2 in up-trends) on the top-50 stocks + 59 ETFs, decided at 15:45, traded in the closing auction, no leverage (gross ≤ 1x).
* Results: Sharpe **1.24 / 1.19 / 1.21** (IS / OOS / all, of ~1,870 variants), CAGR 15.8% vs SPY 15.0%, max drawdown -14% vs -34%. It does not beat SPY out-of-sample (SPY 1.39 / 21.5% there).
* Verdict: paper forward test.
* The app has a model picker, a full trade list, and a live engine proven identical to the backtest by replay tests. See `MeanReversion/README.md` and `MeanReversion/REPORT.md`.

**Momentum and prediction, second search** (`MOMENTUM_PREDICTION_REPORT.md`; research code `research/mp_*.py`; app `MomentumBook/`). The same brief and rules, 3,035 more variants:
* **Best model:** a time-shared momentum book. Overnight momentum (20 names bought in the closing auction and sold in the next opening auction, only when SPY is above its 200-day mean) and a QQQ intraday noise-area breakout run on the same capital, gross ≤ 1x.
* **Results:**
  * The day's 20 biggest winners: Sharpe **1.97 / 2.36 / 1.57** (full / IS / OOS), CAGR 34.6% vs SPY 15.0%, max drawdown -17%.
  * Names ranked by that rule blended with a walk-forward ML overnight forecast, decided at 15:30 (2018 on): **2.09 / 2.41 / 1.84**, CAGR 38%.
* **Shortfalls:** it does not reach Sharpe 2 out-of-sample, and it is cost-sensitive (3 bp per auction side → 1.3-1.4). On their own, slow momentum, ETF rotation and trend timing stay at or below 1.13, and ML at or below 1.86.

**`MomentumBook/`**: the app built from that search (code `core/resarb/mom/`, tests `core/tests/test_mom.py`).
* **Presets.** Traded: rule + ML forecast, decided at 15:30 on the free delayed SIP, QQQ intraday on IEX. Shadow: rule only, journaled next to it in live.
* **Results** (2018-01 → 2026-10, full top-150 universe, whole shares, 2% cash reserve, no leverage):
  * traded: Sharpe **2.03 / 2.40 / 1.74** (full / IS / OOS), CAGR 36.1% vs SPY 14.4%, max drawdown -15.8%;
  * shadow: 1.81 / 2.12 / 1.53.
* **App features.**
  * a full trade list of both legs;
  * one saved ML model per year (*Train ML models*);
  * a live engine proven identical to the backtest by replay tests;
  * a read-only dry run on the real paper account;
  * positions and trade history rebuilt from the broker's orders, next to the backtest's prices.
* **Update 2026-10-28: the Alpaca paper account has no auctions.** On the first day 6 of 20 MOC orders filled.
  * The models stay MOC / MOO, 1 bp per side. The live tab's *Auction orders* setting can send market DAY orders in their place: 5 s before the close and 3 min before the open. Only the order type differs.
  * Market orders priced at historical SIP quotes cut the book to Sharpe **0.46** at k = 20: **the edge lives in the auctions**. The paper test measures that cost on real fills. Ten extra technical ML features did not help significantly (`MomentumBook/REPORT.md` §6–7).
* **Verdict:** paper only, as an engine test. Real money needs real MOC / MOO routing. See `MomentumBook/README.md` and `MomentumBook/REPORT.md`.

**What the research found:**
* **Features that predict the residual.** Rank IC against the next residual return, *t* in-sample / out-of-sample:
  * short-term residual reversal (last bar's residual, IC -0.018, t -14.5 / -11.5), and its relatives: close location in the bar, distance from VWAP, 2-13-bar residual returns;
  * low-risk anomalies (idiosyncratic volatility, beta: negative);
  * volume surprise (positive);
  * residual momentum (positive, out-of-sample only).
* **The reversal is real but fast and cheap.** Its gross edge per trade is smaller than the 1-2.6 bp half spread, so every fast variant loses after costs. Gârleanu-Pedersen partial trading did not rescue it.
* **Data plan.** A 15-minute data delay (the free SIP plan) erases the fast signals but costs the slow ones nothing. IEX real-time bars keep the fast signals (score correlation 0.97 with SIP for mean reversion; 0.999 for momentum).
* **Margin.** Stocks plus ETF hedges use 1.5-1.6x of capital overnight on average (95th percentile 2.0-2.3x). Default stock gross is 0.8-0.9, and the live cap is 1.9x (Reg-T 2x overnight).
* **Data fixes.** Alpaca's adjusted bars needed four distribution fixes and the 2023-01-24 bad NYSE opening prints repaired, and TQQQ / SQQQ removed from the "stock" universe.
