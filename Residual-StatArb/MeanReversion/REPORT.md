# Mean Reversion: research report, team review, verdict

**Brief.** Mean reversion only, any timeframe. Target: Sharpe above 2 and beat SPY buy & hold. No leverage, with a rule for which signals get the capital. Every trade shown in the backtest, and live trading must run exactly the model picked there.

**Answer in one paragraph.** **Sharpe 2 was not reached by anything that survives honest testing**, in about 1,870 variants. The best robust model gets:
* full period (2016-09 → 2026-10): **Sharpe 1.21** (in-sample 1.24, out-of-sample 1.19), CAGR **15.8%** vs SPY 15.0%, max drawdown **-14.0%** vs -33.8%;
* the higher-return version: CAGR **18.7%** at Sharpe 1.18.

It beats SPY over the full period and in-sample. It does **not** beat SPY out-of-sample (2023 → 2026-10): SPY ran at Sharpe 1.39 and 21.5% a year there, against 1.19 and 17.3%. The model's lead over SPY comes mainly from 2020.

Everything here is long-only and unlevered (gross ≤ 1x equity, max 0.99x observed). Decisions use only bars that ended by the decision time, and fills are in the closing auction. The live engine is proven identical to the backtest by session-by-session replay tests.

## 1. Rules of the test (Valid's conditions, applied to every number below)
* **Decision timing.** A decision uses only data that exists at the decision time: the official daily bars of earlier sessions, plus today's **15-minute SIP bars that ended by then** (15:45 or 15:30). The order is market-on-close and fills at the **official close**; an "overnight" model sells in the next **opening auction**. On early-close days everything moves with the close.
  * Most published IBS / RSI(2) tests decide on the closing price they also trade at, which no one can do. We ran that close-decided variant too; it did not help here (ETF preset: Sharpe 1.11 vs 1.26 at 15:45). The timing that matters is the other way: deciding at 15:30 instead of 15:45 costs about 0.3 Sharpe.
* **No leverage.** At most `slots` positions, each (1 − 2% cash reserve) / slots of the equity, in whole shares at the decision price. The book can only be bought, never borrowed for. Maximum gross exposure over the whole test is 0.979x-0.989x.
  * Shorts were tested and dropped: no edge, and a losing short pushes the book above 1x.
* **Universe.** Point-in-time and survivorship-free: the 50 / 100 most liquid U.S. stocks of each month, delisted names included, plus 59 ETFs across asset classes.
* **Costs.** 1 bp per side in the auctions. An auction fill pays no spread, so this is conservative; 2 and 3 bp are also shown.
* **Samples.** In-sample 2016-09 → 2022; out-of-sample 2023 → 2026-10-26. Every choice was ranked on in-sample Sharpe. All ~1,870 variants count as trials in the deflated Sharpe ratio (DSR).

## 2. What was tried (QR idea generation → triage → grids → selection)
| Family | Idea / paper | Result (honest timing, net) |
|---|---|---|
| Overnight cross-sectional reversal, large caps | Lehmann 1990; Lou-Polk-Skouras | fails: today's losers do **not** bounce overnight |
| Overnight continuation of the day's winners | Berkman et al. 2012 (attention) | strong (Sharpe 1.30, CAGR 29.5%, max DD -45%), but it is momentum, not mean reversion; reported, not pursued |
| Intraday reversal of yesterday's winners | Heston-Korajczyk-Sadka | decayed out-of-sample |
| Residual (factor-stripped) 15-minute reversal | Avellaneda-Lee; Da-Liu-Schaumburg | rejected earlier (ResMeanReversion app): the edge is below the spread out-of-sample |
| Gap fades, volume-filtered reversal | — | no net edge after costs |
| **Daily oversold rules on large caps** (IBS, RSI(2), 1-5-day returns, z-scores; trend filter; 4 exit rules; 5 / 10 / 20 slots; 2 rankings) | Connors-Alvarez 2008; Pagonidis 2013 | 1,056 variants (`mr_grid.py stocks`). In-sample mean 0.23, sd 0.33. Best region: **IBS < 0.2 above the 200-day mean, out on IBS > 0.7 or a close above the 5-day mean** |
| Same grid on **59 ETFs** | Pagonidis (IBS on ETFs) | 528 variants (`mr_grid.py etf`). In-sample mean 0.41. **The same rule wins** |
| Stocks + ETFs together, slots 5-30, ranking, 15:30 vs 15:45 | diversification | 100 variants. **top-50 + ETFs at 15:45** is the stable plateau |
| Second entry condition (ret1 < 0, RSI2 < 30, z5 < -1, ret3 < 0), IBS threshold 0.1-0.3 | — | 30 variants; nothing beats plain IBS < 0.2 |
| Inverse-volatility sizing | risk parity per slot | drawdown about -10%, but a lower CAGR and no Sharpe gain |
| Overnight holding, looser exits, top-100 / 150 universes | — | all worse than the plateau |
| **Short side** (IBS > 0.8 / 0.9 below the 200-day mean) | — | in-sample ~0.3, out-of-sample negative everywhere, and the book drifts above 1x: dropped |

## 3. The presets (the model picker; `core/resarb/mr/presets.py`)
Common rule: enter when **IBS < 0.2** and the price is above its **200-day mean**. Exit on **IBS > 0.7**, a close above the **5-day mean**, or after **10 sessions**. Rank candidates by **the lowest IBS first**.

| Preset | Sharpe all / IS / OOS | CAGR (OOS) | Max DD | Vol | Beta | Exposure | Trades/yr | Win | DSR (IS) | PSR OOS > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Recommended**: top-50 + ETFs, 20 slots, 15:45 | **1.21 / 1.24 / 1.19** | 15.8% (17.3%) | **-14.0%** | 12.8% | 0.46 | 61% | 1,564 | 63% | 0.43 | 0.99 |
| Higher return: same, 10 slots, 15:45 | 1.18 / 1.25 / 1.09 | **18.7%** (18.7%) | -19.4% | 15.6% | 0.54 | 75% | 950 | 64% | 0.44 | 0.98 |
| ETFs only: 59 ETFs, 5 slots, 15:45 | **1.26** / 1.15 / **1.47** | 16.1% (18.3%) | -16.8% | 12.4% | 0.33 | 69% | 455 | 63% | 0.34 | 1.00 |
| Free feed, exact data: top-50 + ETFs, 20 slots, **15:30** | 0.91 / 0.98 / 0.80 | 11.6% (10.7%) | -16.0% | 13.0% | 0.48 | 61% | 1,540 | 63% | 0.21 | 0.94 |
| Reference: Connors RSI(2) < 10, top-100, 10 slots, 15:45 | 0.59 / 0.17 / 1.16 | 10.1% (25.9%) | -31.6% | 19.4% | 0.66 | 60% | 392 | 66% | 0.00 | 0.99 |
| **SPY buy & hold** | 0.88 / 0.66 / 1.39 | 15.0% (21.5%) | -33.8% | 17.6% | 1 | 100% | — | — | — | — |

**Cost and timing sensitivity** (recommended preset): Sharpe 1.21 at 1 bp, 1.10 at 2 bp, 0.97 at 3 bp; **0.91 at 15:30**. The last 15 minutes before the decision are worth about 0.3 Sharpe, for every preset.

**Year by year** (recommended vs SPY; 2016 from September, 2026 to 26 October):

| | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Recommended | 1.1% | 10.8% | 4.1% | 21.3% | **42.2%** | 21.3% | -1.5% | 1.2% | 10.3% | 21.1% | 35.5% |
| SPY | 4.0% | 21.7% | -5.0% | 31.1% | 18.5% | 28.6% | -18.2% | 26.2% | 24.9% | 17.7% | 13.1% |

Charts and full trade lists: `reports/backtest_<preset>.png`, `reports/trades_<preset>.csv`, `../research/reports/mr_final.csv`.

## 4. Team review

**DA 🗄️**
* **Data.**
  * Daily bars: Alpaca SIP, split- and dividend-adjusted, with the project's back-adjustments for distributions Alpaca missed.
  * Snapshots: the 15-minute SIP cache, 2016 → 2026-10-26.
  * 48 more ETFs downloaded; scanned for unadjusted distributions (`mr_etf_gapcheck.py`). The large overnight drops are real events: the March 2020 oil crash, the March 2023 bank run, the October 2024 China reversal.
* **Free hybrid live feed** (`mr_hybrid_check.py`). At 15:45 the free plan has SIP bars only up to the 15:00 bar, so the engine takes the last two bars from real-time IEX. Since 2023:
  * IBS < 0.2 agreement is 97%; the last price differs by a median 1.5 bp (95th percentile 31 bp);
  * **83% of the entries are identical**;
  * out-of-sample Sharpe 1.04 (14.5% CAGR) instead of 1.19 (17.3%).
* **The 15:30 / delayed-SIP option** is exactly the backtest's data, but costs 0.3 Sharpe.
* **Real-time SIP** (Alpaca's paid plan) gives the backtest's numbers at 15:45.

**QR 🔬**
* **Robustness.** The rule is the most robust thing found in this project: the same rule ranks first in two independent grids (stocks, ETFs). Its neighbours (IBS 0.15-0.25, 10-30 slots, rank by signal or calm-first) all land at Sharpe 1.1-1.2.
* **Cause.** It is the documented short-term reversal: liquidity providers are paid for absorbing late-day selling. The trend filter keeps it out of falling knives.
* **The ceiling.** It is a long-only beta-0.46 book with 61% exposure, so its Sharpe is bounded by the market's. Sharpe 2 at a 15% CAGR needs ≤ 7.5% volatility with no leverage. Nothing found gets close:
  * diversification across 109 names and asset classes already gives 12.8% volatility;
  * a beta hedge would cut the return more than the volatility.

**Dev ⚙️**
* **One decision function.** `decide()` is shared by backtest and live: the same Market (daily history + today's snapshot), signals, ranking, sizing and auctions.
* **Parity tests.** `core/tests/test_mr.py` replays backtests session by session through the live engine with a simulated broker: morning, decision, auction fills, after-close. It requires **identical orders, fills and final equity (to 1e-9)** for:
  * a rule-exit model at 15:30 on delayed SIP;
  * an overnight model at 15:45;
  * a trend + time-stop model;
  * the hybrid feed.

  Plus unit tests for no leverage, ranking, inverse-vol caps, early closes, feed / cut-off checks, dry run and the past-cut-off refusal. All pass.
* **The GUI.** The Backtest tab lists every trade with 19 columns, sortable, filterable, exportable. The Live tab trades the panel's model, says whether it is the last backtested one, and locks the panel while the loop runs.

**Valid 🛡️**
* **Look-ahead.** Clean: decisions use the snapshot, the close is used only as the fill price, and delistings are detected from past data only.
* **Multiple testing.** The recommended model's in-sample Sharpe 1.24 is **below** the expected best of 1,871 trials (1.31), so **DSR = 0.43**: in-sample alone does not prove it.
* **Out-of-sample.** The evidence is the out-of-sample Sharpe of 1.19 over 3.8 years (PSR 0.99 that it is above 0). That sample is not perfectly clean: the later selection stages showed out-of-sample columns, even though choices were ranked in-sample.
* **Concentration.** 2020 alone contributes most of the CAGR edge. Without 2020 the model compounds at about 13% a year against SPY's 14.5%.
* **Sign-off: NOT for capital. Approved for a paper forward test.**
  * Conditions: 3 months of paper trading, the "Check live vs backtest" tool showing ≥ 80% identical orders (hybrid feed) or ≥ 95% (real-time SIP), and a slippage review of the auction fills.

**RM 🔒**
* **Risk profile.** Unlevered by construction: max 0.99x, 2% cash reserve, whole shares. Max drawdown -14% against SPY's -34%, volatility 12.8%, beta 0.46.
* **Tail.** Tail risk is a gap-down through held positions: the 10-session time stop and the 20-way spread limit a single name to about 5%.
* **Live overlays** (not in the backtest; off by default so live = backtest):
  * the daily-loss halt;
  * the drawdown stop. Recommended setting: **25%**, well beyond anything the backtest saw, as a "the model is broken" stop.
* **Capital:** paper only until Valid's conditions are met.

**PM 👑**
* **Mandate.** The brief was Sharpe > 2 and beating SPY. The honest answer is no on Sharpe 2, and a qualified yes on SPY: better risk-adjusted over 10 years and in-sample, not out-of-sample.
* **Role.** As a portfolio sleeve it is a lower-volatility, half-beta equity exposure that does best in V-shaped sell-offs (2018, 2020, 2022) and lags in smooth bull years (2017, 2023). That is useful next to buy & hold, not instead of it.
* **Preset choice.** The recommended preset for a risk-adjusted mandate; the 10-slot version if beating SPY's CAGR matters more than drawdown.

**QT ⚡**
* **Execution.**
  * Entries and exits are market-on-close, sent before Alpaca's 15:50 cut-off (12:50 on early closes); overnight exits are market-on-open (OPG).
  * There is no spread to pay, so slippage risk is the auction itself: liquid large caps and ETFs, with small tickets of about $5k per slot at $100k.
  * The engine needs a margin account (Alpaca paper is 4x) only so that MOC buys can be queued while the same auction's MOC sells are pending; the book itself never exceeds 1x.
* **Today's dry run** against the paper account (read-only, not armed): pre-flight all green. The 15:45 decision of 2026-10-26 on the hybrid feed made 10 entries, **all identical (ticker and shares) to the decision rebuilt from the backtest's SIP data**; the SIP data had one more name (DIA, IBS 0.199).

## 5. Team recommendation
1. **Do not expect Sharpe 2.** No honest, unlevered mean-reversion model in this data gets there. Published Sharpe-2+ versions of these rules typically add leverage, use one ETF over a favourable window, or assume zero costs. Here 3 bp per side alone takes the recommended model from 1.21 to 0.97. The best supportable number is about 1.2.
2. **Paper-test the recommended preset at 15:45 with the `sip+iex` feed.** It is free, with 83% order agreement. If paper results matter more than $99/month, take Alpaca's real-time SIP plan instead: exact backtest data.
   * Steps: Backtest tab → Run; Live tab → Connect → Pre-flight → Plan now; then ARM with `ARM PAPER` and start the daily loop (or `run_live.bat`). It needs an account no other strategy has claimed.
3. **Valid re-reviews after 3 months** of paper journal + "Check live vs backtest".

## 6. Files
* `core/resarb/mr/`:
  * `models.py`: MRModel, signals, `decide()`;
  * `market.py`: daily bars, snapshots, early closes;
  * `engine.py`: the backtest and its stats;
  * `live.py`: the live engine (feeds, MOC / OPG, journal, book, KILL, pre-flight, live-vs-backtest check);
  * `presets.py`, `run.py` (headless), `ui/` (the GUI).
* `core/tests/test_mr.py`: replay parity and unit tests (FAKES ONLY).
* `research/`:
  * `mr_grid.py`, `mr_select.py` (stages A / A45 / B / C / D / E / F), `mr_final.py`;
  * `mr_hybrid_check.py`, `mr_etf_gapcheck.py`, `mr_live_dry.py`, `mr_gui_smoke.py`;
  * `mr_triage*.py`, `mr_ibs_check.py`;
  * `reports/`: every CSV and log.
