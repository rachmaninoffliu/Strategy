# Momentum and Prediction: the second search, team review, verdict

**Brief.** The same search as for mean reversion, for the other two strategies:
* Target: Sharpe above 2 and beat SPY buy & hold, any timeframe.
* No leverage.
* The same honest test rules (§1).

**Answer in one paragraph.** Momentum and prediction together clear the bar over the full history and in-sample, but **not out-of-sample**. The model runs two momentum sleeves on the **same capital at different times of day**:
* **overnight momentum**: buy 20 large caps / ETFs in the closing auction, sell them in the next opening auction;
* **intraday momentum on QQQ**: the published noise-area breakout, flat at the close.

The prediction family's contribution is a walk-forward machine-learned forecast of the overnight return, which improves which 20 names the overnight sleeve buys.

| Model | Sharpe (full / in-sample / out-of-sample) | CAGR | Max drawdown |
|---|---|---|---|
| Rule-based book (2016-09 → 2026-10) | **1.97 / 2.36 / 1.57** | **34.6%** | **-17%** |
| Rule + ML book, decided at 15:30 (2018-01 → 2026-10) | **2.09 / 2.41 / 1.84** | **38.3%** | **-16%** |
| SPY buy & hold (2016-09 → 2026-10) | 0.88 / 0.66 / 1.38 | 15.0% | -34% |

Both versions beat SPY in every window:
* out-of-sample (2023 → 2026-10), CAGR is 32-41% against 21%;
* they beat SPY's return in 8-9 of 9-11 calendar years.

Gross exposure never exceeds 1x, because the two sleeves never hold at the same time.

Where it falls short:
* **Sharpe 2 out-of-sample:** best 1.84-1.93 (the probability that the true out-of-sample Sharpe is above 2 is about 0.4).
* **Multiple testing:** across every trial of both studies, the in-sample Sharpe does not clear the deflated-Sharpe bar.
* **Costs:** 2 bp per auction side instead of 1 takes about 0.35 off the Sharpe.

On their own, the slow momentum families (≤ 1.13) and the ML predictors (≤ 1.86, 2018 on) do not reach the target.

## 1. Rules of the test (the same as the MR study)
* **Timing.**
  * A decision sees the official daily bars of earlier sessions plus today's 15-minute SIP bars that ended by the decision time. Early closes are taken from the NYSE calendar.
  * Fills:
    * closing auction at the official close;
    * opening auction at the official open;
    * an intraday fill at the 15-minute bar's close plus the half spread.
* **No leverage.** Each sleeve's weights sum to ≤ 1 at every moment. Two sleeves share capital only when one holds close → open and the other open → close.
* **Universe.** Point-in-time and survivorship-free:
  * the top-150 U.S. stocks by liquidity of each month (no leveraged / inverse ETPs);
  * the MR app's 59 ETFs;
  * the top-500 for the slow stock families.
* **Costs.** 1 bp per auction side. Half spread: QQQ / SPY 0.5 bp, other ETFs 1 bp, stocks 3 bp in the first hour. Sensitivities at 2 and 3 bp are shown.
* **Samples.** In-sample 2016-09 → 2022 (the walk-forward ML: 2018 → 2022); out-of-sample 2023 → 2026-10-23. Choices were ranked on in-sample Sharpe. Every variant is a trial: **3,035 here**, not counting ~140 robustness re-runs (plus the MR study's 1,871 for the deflated Sharpe ratio).

## 2. What was tried
| Family | Idea / paper | Variants | Result (honest timing, net) |
|---|---|---|---|
| M1 Stock momentum, long-only top-k (top-100 / 150 / 500, monthly / weekly, market / stock trend filters) | Jegadeesh-Titman 1993; George-Hwang 2004 (52-week high); Blitz-Huij-Martens 2011 (residual); Da-Gurun-Warachka 2014 (frog in the pan); Clenow 2015 | 720 | in-sample mean 0.51; best full-period **1.09**, but at 45% volatility and beta 1.34 (max DD -46%) |
| M2 ETF rotation / tactical allocation (relative + absolute momentum, canary, safe asset) | Faber 2007; Antonacci 2014 (dual momentum / GEM); Keller-Keuning 2017-18 (VAA / DAA) | 904 | in-sample mean 0.24; best **0.87**; none beats SPY's return in both halves |
| M3 Trend timing of SPY / QQQ (moving averages, time-series momentum, vol targeting ≤ 1x) | Moskowitz-Ooi-Pedersen 2012; Moreira-Muir 2017 | 76 | best **1.13** (QQQ, 10% vol target + 200-day mean; 11% CAGR); QQQ > 150-day mean 1.11 at 18.6% CAGR |
| **M4 Overnight momentum** (buy at the close, sell at the open; 8 signals × 4 filters × k) | Lou-Polk-Skouras 2019 ("tug of war": momentum is earned overnight); Berkman et al. 2012 (attention) | 736 | in-sample mean **1.14**, out-of-sample mean 0.92, 96% of variants positive out-of-sample. Best region: **today's move to 15:45, top-20, only when SPY > its 200-day mean**: 1.77 / 2.23 / 1.32, CAGR 27.5% |
| I1-I4 Intraday: hold open → close; market intraday momentum; stock intraday momentum / reversal at 10:00 | Gao-Han-Li-Zhou 2018 (JFE); Heston-Korajczyk-Sadka 2010 | 46 | no edge after costs (best 0.54; the last-half-hour version is negative out-of-sample) |
| **I5 Noise-area breakout** on SPY / QQQ (long-only, flat at the close) | Zarattini-Aziz-Barbon 2024 ("Beat the market") | 32 | **QQQ at the paper's settings: 0.97 / 0.93 / 1.04**, 5.5% CAGR at 5.7% volatility, max DD -5% |
| P Machine-learned prediction (ridge / gradient boosting on ~35 features, walk-forward by year) of the overnight and the next-day return | Gu-Kelly-Xiu 2020; Krauss-Do-Huck 2017 | 72 + 24 | overnight IC 0.065 / 0.047. Alone: best 1.86 (2018 on); next-day forecasts mean 0.58. **Blended with the rule inside the time-shared book: 2.09-2.13 / 2.39-2.41 / 1.84-1.93** |
| **Time-sharing**: an M4 overnight sleeve + an I5 intraday sleeve on the same capital | — | 375 | mean **1.84 / 2.12 / 1.55**, CAGR ~30%, max DD ~-16%; every combination beats SPY on return and drawdown in both halves |

**Why overnight momentum works (QR).** Large caps that rallied into 15:45 keep rising overnight and gap up at the open. Retail attention buys them at the open, and the same names give back 7.6 bp on average during the next session (`mp_overnight.py`).
* Selling in the **opening auction** captures the gap and avoids the reversal.
* Selling at 9:45 instead costs about 0.6 Sharpe: the edge is concentrated in the auction itself.
* The whole top-150 held overnight earns Sharpe 0.68: the overnight premium. Selecting the day's winners raises it to about 1.8.

## 3. The candidate: time-shared momentum book
**Rules** (`research/mp_combo_check.py`, "a priori"):
1. **15:45** (or 15:30): rank the top-150 stocks + 59 ETFs by today's move from yesterday's close to the 15:45 price. If SPY closed above its 200-day mean yesterday, buy the **top 20**, 1/20 of equity each, **market-on-close**. Otherwise stay in cash overnight.
2. **Next morning:** sell them all **market-on-open** (OPG).
3. **10:00 → 15:30, every 30 minutes:** if QQQ's last 15-minute bar closed above its noise band, buy QQQ with the whole equity at market. The band is max(open, yesterday's close) × (1 + the average absolute move from the open at that time of day over the last 14 sessions). Sell when it closes below max(band, session VWAP). Anything still held goes market-on-close.

The legs never overlap: the book is ≤ 1x at every moment.

| Model | Sharpe full / IS / OOS | CAGR (OOS) | Vol | Max DD | Beta |
|---|---|---|---|---|---|
| **A priori**: top-20 by today's move + QQQ noise area (paper settings) | **1.97 / 2.36 / 1.57** | **34.6%** (32.1%) | 15.7% | **-16.9%** | 0.32 |
| Same, decided at **15:30** (free delayed SIP = the backtest's own data) | **2.02 / 2.36 / 1.67** | 35.7% (34.6%) | 15.8% | -16.0% | 0.32 |
| In-sample-selected (noise look-back 20) | 1.94 / 2.44 / 1.41 | 34.0% (28.3%) | 15.8% | -17.2% | 0.32 |
| Blend (today's move + volume surprise) | 1.97 / 2.17 / **1.77** | 31.2% (32.8%) | 14.3% | -14.6% | 0.30 |
| Higher return: top-10 | 1.94 / 2.37 / 1.49 | **42.3%** (37.2%) | 19.1% | -20.7% | 0.35 |
| Overnight sleeve alone (top-20) | 1.77 / 2.23 / 1.32 | 27.5% (24.1%) | 14.3% | -22.3% | 0.21 |
| Intraday sleeve alone (QQQ noise area) | 0.97 / 0.93 / 1.04 | 5.5% (6.4%) | 5.7% | -5.4% | 0.10 |
| **SPY buy & hold** | 0.88 / 0.66 / 1.38 | 15.0% (21.3%) | 17.6% | -33.8% | 1 |
| QQQ buy & hold | 0.94 / 0.70 / 1.40 | 20.3% (30.0%) | 22.5% | -35.0% | 1.19 |

**Year by year** (a priori vs SPY; 2016 from September, 2026 to 23 October):

| | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Model | 12.0% | 31.0% | 17.7% | 23.3% | **109.1%** | 56.6% | 0.8% | 22.8% | 42.6% | 33.7% | 22.6% |
| SPY | 4.0% | 21.7% | -5.0% | 31.1% | 18.5% | 28.6% | -18.2% | 26.2% | 24.9% | 17.7% | 12.3% |

Without 2020: 28.2% a year at Sharpe 1.75, against SPY's 14.6% at 0.97. Equity and drawdown chart: `research/reports/mp_book.png`.
* **2022:** the book was flat because the SPY filter kept the overnight leg in cash for most of the year.
* **Deepest drawdown (-17%):** the 2026 one.

**Sensitivities** (a priori):
* **Costs.**
  * 1 bp per auction side → 1.97.
  * 2 bp, with the QQQ half spread at 1 bp → **1.62** (OOS 1.26).
  * 3 bp → **1.27** (OOS 0.95).
* **Overnight exit at the first trade** instead of the official open: 1.95.
* **Neighbours** (22 one-at-a-time changes: k, universe, trend filter, noise settings, SPY instead of QQQ): full **1.91 ± 0.07**, IS 2.31 ± 0.13, OOS 1.50 ± 0.10, CAGR 33%.
  * The worst is "no SPY filter": 1.70, with a -30% drawdown.
* **Rolling 12-month Sharpe:** above 2 in 50% of windows, above 1 in 85%, below 0 in 2% (minimum -0.39).

**Trading facts.**
* **Overnight sleeve:**
  * invested on 84% of nights, 20 names each time: about 4,240 round trips a year (≈ 40 auction orders a day);
  * ETFs are 16% of picks;
  * the picks trade a median $800M a day (10th percentile $250M).
* **Intraday sleeve:** trades on 78 days a year, 110 round trips, 43% winners, +5.0 bp on average.
* **Correlation and concentration:**
  * the two legs correlate at 0.05; the book's beta is 0.32;
  * the top-10 names contribute 21% of the profit (TSLA, AMD, MU, NVDA, AAL, SMCI, MSTR, GDX, BIDU, SHOP).

## 4. Prediction (machine learning)
**Set-up** (`mp_predict.py`, `mp_features.py`).
* **Rows and features.** One row per session and name of the top-150 + 59 ETFs: 540k rows. Each row has 35 features known at the decision time:
  * today's move, its beta-adjusted residual, the last hour, the last 15 minutes;
  * IBS, volume surprise, the gap;
  * 1 / 5 / 21 / 63-day returns, 12-1 momentum;
  * mean overnight and intraday returns over 5 / 20 / 60 days;
  * volatility, range, beta, 52-week high, distance from the 20 / 200-day means, liquidity, an ETF flag;
  * market context: SPY's move, volatility, trend and overnight drift, breadth, dispersion, weekday, turn of the month.
* **Targets:** the overnight return (close → next official open) and the next-day return (close → next close).
* **Models:** ridge and gradient-boosted trees (ranked or raw features, ranked or raw target), with hyper-parameters fixed in advance.
* **Walk-forward:** by calendar year. The model for year Y is trained only on rows whose target was known before Y. In-sample = 2018-2022, out-of-sample = 2023-2026.

**Information content** (mean daily rank IC, in-sample / out-of-sample):

| Model | Overnight return | Next-day return |
|---|---|---|
| Ridge (ranked features, ranked target) | **0.065 / 0.047** | 0.025 / 0.026 |
| Trees, ranked features | 0.056 / 0.039 | 0.020 / 0.028 |
| Trees, raw features | 0.057 / 0.044 | 0.024 / 0.025 |
| Trees, raw-return target | 0.021-0.032 / 0.014-0.021 | 0.010-0.014 / -0.001-0.010 |
| Single feature: 12-1 momentum | 0.054 / 0.044 | 0.022 / 0.024 |
| Single feature: today's move | 0.011 / -0.011 | -0.013 / -0.015 |

* **What the forecasts load on.** The overnight forecast is mostly 12-1 momentum, volatility / beta and overnight persistence (rank correlations with the forecast 0.30, 0.22, 0.14). It is negative on the last hour and the last 5 days.
* **Day-to-day ranking vs the extreme tail.** The models rank the whole cross-section much better than "today's move", whose edge lives only in the extreme top of the ranking. The next-day target carries less than half the information.

**As stand-alone strategies** (72 portfolios: the top 5 / 10 / 20 forecasts, with or without the SPY filter; 2018-01 → 2026-10):
* **Overnight forecasts:** mean Sharpe 1.32 (IS 1.28 / OOS 1.40). The best in-sample was trees on raw features, top-10 with the SPY filter: 1.69 / 1.88 / 1.60, CAGR 36%, max DD -30%.
* **Next-day forecasts:** mean 0.58. No stand-alone prediction strategy reaches Sharpe 2 over the full period. The best out-of-sample numbers (2.0-2.1, top-5 trees) were not the in-sample choices.

**Inside the time-shared book** (`mp_predict_combo.py`, 2018-01 → 2026-10, the same QQQ intraday leg). 12 ways to use the forecasts, ranked in-sample:

| Overnight ranking | Sharpe full / IS / OOS | CAGR (OOS) | Max DD |
|---|---|---|---|
| Rule only: today's move (the §3 model) | 1.88 / 2.21 / 1.57 | 34.7% (32.1%) | -16.9% |
| ML only: trees, raw features | 1.90 / 2.05 / 1.82 | 37.7% (43.5%) | -14.3% |
| **Rule + ML, rank average** | **2.13 / 2.39 / 1.93** | **38.7% (41.0%)** | **-14.0%** |
| Rule, only among the ML's top half | 2.13 / 2.41 / 1.90 | 37.3% (37.9%) | -14.1% |
| SPY buy & hold (same window) | 0.81 / 0.52 / 1.38 | 14.3% (21.3%) | -33.8% |

**Checks on the ML-enhanced book** (`mp_predict_check.py`, 2018-01 → 2026-10; the ML was in-sample-selected among 12 uses of the forecasts):

| Version | Sharpe full / IS / OOS | CAGR (OOS) | Max DD | 2 bp auctions | 3 bp auctions |
|---|---|---|---|---|---|
| 15:45, rule + trees, top-20 | **2.13 / 2.39 / 1.93** | 38.7% (41.0%) | -14.0% | 1.79 (OOS 1.62) | 1.45 (OOS 1.30) |
| **15:30, rule + trees retrained on 15:30 features** (free delayed SIP) | **2.09 / 2.41 / 1.84** | 38.3% (39.1%) | -16.1% | 1.76 (OOS 1.53) | 1.42 (OOS 1.22) |
| 15:45, a-priori ensemble (rule + ridge + both tree models) | 1.94 / 2.33 / 1.67 | 38.4% (39.4%) | -14.9% | 1.63 | 1.32 |
| 15:30, a-priori ensemble | 1.95 / 2.29 / 1.72 | 38.7% (40.6%) | -15.6% | 1.65 | 1.34 |
| 15:30, rule only (§3) | 1.93 / 2.21 / 1.67 | 36.1% (34.6%) | -16.0% | 1.61 | 1.28 |

Other checks:
* **Exit price:** a first-trade exit gives 2.10 / 2.07.
* **Number of names:** top-10 and top-30 give 2.03-2.08 and 1.92-2.06.

**Year by year** (15:30 rule + trees): 2018 +20.6%, 2019 +34.4%, 2020 +72.7%, 2021 +75.6%, 2022 +0.2%, 2023 +19.1%, 2024 +48.2%, 2025 +46.8%, 2026 +34.9%. It beats SPY in 8 of 9 years; 2023 is the exception (SPY +26.2%).

**What the ML adds, honestly.**
* **In-sample pick:** the trees-plus-rule blend lifts out-of-sample Sharpe from 1.57-1.67 (rule) to 1.84-1.93.
* **A-priori ensemble:** it lifts it only to 1.67-1.72.
* **Statistics:** the two books correlate at 0.90 daily. The out-of-sample Sharpe gain is +0.36 at 15:45 (block-bootstrap SE 0.20, 90% interval +0.04 → +0.68) and +0.18 at 15:30 (SE 0.21, interval -0.16 → +0.52). It is marginally significant at 15:45 and not significant at 15:30.
* **Deflated Sharpe over every trial:** 0.35 for the ML book. PSR that its out-of-sample Sharpe is above 2: 0.38-0.44.

## 5. Team review

**DA 🗄️**
* **Prices.**
  * The overnight result rests on the official open and close. Alpaca's daily open matches the first 9:30 trade to a median 0 bp (90th percentile 3.9 bp, mean -0.1 bp) for the names traded.
  * Selling at the first trade instead gives the same Sharpe (1.95 vs 1.97), so the result is not an artefact of which print the daily bar uses.
  * The leveraged / inverse ETPs are not in the universe.
* **Free data path.**
  * Overnight sleeve: decided at **15:30** on the free 15-minute-delayed SIP feed, which is the backtest's own data. Sharpe 2.02.
  * Intraday sleeve: needs bars as they close, i.e. real-time **IEX** for QQQ. On 2024-01 → 2026-10:
    * 199 of the 204 trade days have the same first entry as on SIP;
    * the leg makes 3.4% a year instead of 4.5% (Sharpe 0.51 vs 0.69, daily correlation 0.96).

**QR 🔬**
* **Robust and documented.**
  * Overnight momentum is the most robust family found in this project: 736 variants, in-sample mean 1.14, 96% positive out-of-sample.
  * It is a documented anomaly (momentum is earned overnight; attention-driven opening prices).
* **Diversification.**
  * The noise-area sleeve is a published rule run at its published settings.
  * Its return is earned in a different part of the day, which is why the legs are uncorrelated (0.05) and the Sharpe ratios add.
* **Prediction.** The ML models carry real cross-sectional information about the overnight return: IC 0.065 → 0.047 out-of-sample, mostly momentum, volatility and overnight persistence.
  * On their own they are no better than the simple rule.
  * Blended with it, they pick better overnight names: out-of-sample 1.84-1.93 vs 1.57-1.67.
  * This is the "prediction strategy" worth keeping.
* **Ceiling.** Out-of-sample, both legs are weaker than in-sample: overnight 2.23 → 1.32 (rule), the in-sample-selected noise setting 1.11 → 0.53. A sure Sharpe 2 out-of-sample would need a third uncorrelated return source; nothing else tested qualifies.

**Dev ⚙️**
* **Not built yet.** This is research code (`research/mp_*.py`). An app does not exist for it yet.
* **What a build needs** (to the same standard as the MeanReversion app):
  * one decision function shared by backtest and live;
  * MOC buys before 15:50, OPG sells at 9:28;
  * a 30-minute intraday loop on IEX bars;
  * a trade table with both legs;
  * replay parity tests with a simulated broker.
* **Reuse.** The MR engine's MOC / OPG / journal / account-registry pieces can be reused. The intraday loop is new.

**Valid 🛡️**
* **Look-ahead.** Clean:
  * signals use bars that ended by the decision time;
  * the 200-day filter uses yesterday's close;
  * the noise band uses the previous 14 sessions;
  * fills are at prices that exist after the decision.
* **Multiple testing.** Two ways to count:
  * **Every trial** of both studies (4,804-4,906 trials, cross-trial Sharpe variance 0.49): the expected best in-sample Sharpe of unskilled trials is 2.57-2.58, so the rule book's in-sample 2.36 has **DSR 0.30**. In-sample alone does not prove it.
  * **Within the overnight / time-sharing family** (variance 0.02): DSR 1.00.
  * The truth is between the two. The family is homogeneous (every variant is the same idea), so the all-trials figure is very conservative.
  * For the ML book (4,906 trials): DSR 0.35.
* **The ML version.** It is an in-sample pick among 12 uses of the forecasts. The ensemble fixed in advance does only slightly better than the rule (OOS 1.67-1.72 vs 1.57-1.67), and the pick's gain over the rule is not significant at 15:30. Valid accepts it as a candidate, not as proof that ML adds 0.3 Sharpe.
* **Out-of-sample.**
  * 3.8 years at Sharpe 1.57: PSR 0.999 that it is above 0, 0.65 that it beats SPY's 1.38, **0.20 that it is above 2**.
  * The 90% block-bootstrap interval (20-day blocks) is 0.80 → 2.36 out-of-sample and 1.48 → 2.50 over the full period.
  * Not perfectly clean: the combination stage displayed out-of-sample columns, although choices were ranked in-sample. The a priori model was fixed before those columns were read.
* **Concentration.** 2020 (+109%) inflates the CAGR, though without 2020 the model still compounds 28% a year at Sharpe 1.75.
* **Cost sensitivity is the main risk.** At 3 bp per auction side, out-of-sample Sharpe falls below 1.
* **Sign-off: NOT for capital. Approved for a paper forward test** once the app exists, on these conditions:
  * 3 months of paper trading;
  * live orders ≥ 95% identical to the backtest for the overnight leg (15:30, delayed SIP) and ≥ 80% for the intraday leg (IEX);
  * the measured auction slippage ≤ 1.5 bp per side.

**RM 🔒**
* **Risk profile.**
  * Unlevered: ≤ 1x at all times, 2% cash reserve, whole shares.
  * Volatility 15.7%, max drawdown -17% (SPY -34%), beta 0.32.
  * The worst rolling 12-month Sharpe was -0.39.
* **Tail.**
  * Overnight gap risk on 20 names at 5% each: a single -30% gap costs 1.5% of equity.
  * A market-wide gap down hits the overnight leg as a whole. The 200-day filter took it out for late 2018, March-May 2020, most of 2022 and the spring 2025 sell-off.
* **Operational risk** is higher than for MR: about 40 auction orders a day, and two hard cut-offs (OPG 9:28, MOC 15:50). A missed OPG leaves the overnight book in the market during the day.
  * Rule: a missed opening exit is sold at market at 9:45, and the day's intraday leg is skipped.
* **Live overlays:** a daily-loss halt and a 25% drawdown stop, off in the backtest.
* **Account rules:** a margin account, and ≥ $25k for the pattern-day-trader rule (the intraday leg day-trades).

**PM 👑**
* **Against the mandate.** The brief was Sharpe > 2 and beat SPY.
  * Full period: yes on both. Rule book 1.97-2.02 (34.6-35.7% vs 15.0%); rule + ML book 2.09-2.13.
  * In-sample: yes.
  * Out-of-sample: it beats SPY on return, drawdown and Sharpe (1.57-1.93 vs 1.38), but is **not** at 2.
* **Relative to the earlier work.** It is clearly better than the MeanReversion preset (1.21) and the earlier ResMomentum (0.60 OOS).
* **Choice.**
  * The **rule + ML book decided at 15:30**: free, exact data for the overnight leg, and the best out-of-sample evidence.
  * The rule-only book is kept as the fallback preset.
  * **Top-20** names, not top-10, for drawdown.
* **Disagreement with Dev.** Dev prefers to start with the rule-only book: no model files, no annual retraining, fewer ways for live to drift from the backtest. PM accepts that the ML adds operational risk, but wants the paper test on the version with the best chance of the Sharpe-2 mandate.
  * Agreed compromise: the app trades one preset and computes the other in shadow every day, so the paper period compares both.
* **Portfolio role.** It is a high-turnover, auction-dependent book. Capital should follow the paper evidence, not the backtest.

**QT ⚡**
* **Execution.**
  * Everything in the overnight leg is an auction order: MOC buys before 15:50, OPG sells before 9:28. No spread is paid; the risk is the auction price and Alpaca's handling of 40 auction orders a day.
  * The intraday leg is one liquid ETF at market (QQQ half spread ≈ 0.3 bp).
* **Capacity.** Since 2023 the picks' last 15-minute bar (which contains the closing cross) traded a median $118M (5th percentile $24M).
  * At $1M equity each order is $49k: 0.04% of the median, 0.2% of the 5th percentile.
  * At $10M it is 0.4% / 2%.
  * Capacity is not the constraint below about $10M.
* **Disagreement with Valid.** QT thinks 1 bp per side is conservative for auction fills of $5k tickets in names that trade $800M a day. Valid holds the 1.5 bp slippage gate because the edge per night (12 bp net) is thin relative to costs. Both agree the paper test settles it: compare each fill with the official open / close.

## 6. Team recommendation
1. **Sharpe 2 is reached over the full history and in-sample, not out-of-sample.**
   * The honest out-of-sample range is 1.6-1.9, from the time-shared momentum book (rule, or rule + ML forecast).
   * It beats SPY on return, drawdown and Sharpe in both halves.
   * Slow momentum, ETF rotation, trend timing and stand-alone ML do not reach it.
2. **Next step: build the time-shared book as an app to the MeanReversion standard:**
   * a trade list with both legs;
   * no leverage;
   * live = backtest replay tests, including the ML forecasts from a saved model file;
   * two presets: rule + ML (traded) and rule-only (shadow).

   Then paper-test it for 3 months on its own Alpaca paper account:
   * the overnight leg decided at 15:30 on the free delayed SIP feed;
   * the intraday leg on IEX;
   * the ML retrained once a year, as in the backtest.
3. **Valid re-reviews** after the paper period: order agreement, auction slippage, live vs backtest returns, ML vs shadow rule.

**Update 2026-10-27: the app is built** (`MomentumBook/`; results and verdict in `MomentumBook/REPORT.md`).
* Its backtest reproduces this study: the 15:45 rule book gives 1.98 / 2.37 / 1.58.
* On the full top-150 universe, with the 80 names that lacked intraday bars here filled, the traded preset gives **2.03 / 2.40 / 1.74** (rule + ML, 15:30, QQQ intraday on IEX). The rule-only shadow gives 1.81 / 2.12 / 1.53.

**Update 2026-10-28: the edge needs real auctions** (`MomentumBook/REPORT.md` §6).
* Alpaca's paper account does not run the auctions: 6 of 20 MOC orders filled on the first day.
* Market orders 30 s before the close and 60 s after the open cost **3.4 / 10 bp per side**, measured on SIP quotes of the picks. With those costs the book falls to **0.63** at k = 20 and **1.04** at k = 5; with auctions it is 2.04–2.09 at any k.
* Ten extra technical features (RSI, MACD, ADX, Bollinger, stochastic, ATR, SMA50/200) did not help significantly.

## 7. Files
* `research/mp_data.py`: the panel. Daily bars for 3,900 stocks + 59 ETFs, universes, and a 15-minute cube with calendar-aware early closes.
* `research/mp_lib.py`: the simulators (overnight, intraday, rebalance), stats, PSR / DSR, the trial book.
* `research/mp_slow.py`: M1-M3.
* `research/mp_overnight.py`, `research/mp_overnight_check.py`: M4 and its checks.
* `research/mp_intraday.py`, `research/mp_sleeves.py`: I1-I5, the noise-area sleeve.
* `research/mp_predict.py`, `research/mp_features.py`: the ML walk-forward (§4).
* `research/mp_predict_combo.py`, `research/mp_predict_check.py`: the ML inside the book, and its checks (15:30 retrain, ensemble, costs).
* `research/mp_combo.py`, `research/mp_combo_check.py`, `research/mp_combo_robust.py`: time-sharing, validation, neighbours.
* `research/reports/mp_book.png`: the equity and drawdown chart.
* `research/reports/mp_*.csv | .log | _rets.parquet`: every trial's statistics and daily returns.
* `research/mp_fill_bars.py`: downloads the 15-minute bars of the top-150 members that had none (used by the app).
