# Diversifier parameter selection - pre-registered protocol
Written 2026-10-07 BEFORE any sweep result was looked at. Engine: `scripts/opt/sim.py` (validated trade-for-trade against
TradingView: identical trade counts in 7/7 comparisons, win counts exact, final equity within 0.85% over 16 years).

## Data / windows
* Train = 2010-02-11 .. 2021-12-31 (selection uses ONLY this). Test = 2022-01-01 .. present (touched once, after the freeze).
* Sub-periods inside Train for stability: H1 = 2010-02-11..2015-12-31, H2 = 2016-01-01..2021-12-31.
* Costs 0.1% per side, no slippage, 100% of equity, long-only (as the research script).
* Markets: NDX family = TQQQ (and QQQ), signals from QQQ; SPX family = SPXL (and SPY), signals from SPY.
  The leveraged ETFs (TQQQ, SPXL) are the production assets; QQQ and SPY are cross-checks.

## Search space (exhaustive, daily bars)
rsiLen 2..14 (13) x rsiEntry 10..40 step 2 (16) x exitLen 3..20 (18) x trendLen 50..250 step 25 (9) x maxHold 2..20 step 2 (10)
= 336,960 per variant; variants: Dip with dipTrend=true, Dip with dipTrend=false, Trend+Dip; plus Trend (9 trendLen values).
Trials per asset ~1.01 million. Because the trial count is huge, selection is NOT "argmax": see smoothing + constraints.

## Selection rule (fixed in advance)
1. Hard filters, applied on BOTH leveraged assets (TQQQ and SPXL), Train only:
   trades >= 60, exposure >= 5%, CAGR > 0, Sharpe_H1 > 0 and Sharpe_H2 > 0, MDD <= 65%.
2. Score per asset a: percentile rank (among all filter-passing configs of that asset, all variants pooled) averaged over
   Sharpe and Calmar. Cross-market score S = mean(score_TQQQ, score_SPXL) - 0.5*|score_TQQQ - score_SPXL|
   (rewards configs that are good on BOTH markets and penalises disagreement).
3. Plateau smoothing: S_smooth = mean of S over the +-1-step neighbourhood (3^5 cells, edges clipped, failing cells count as 0).
   The winner is the config with the highest S_smooth, i.e. the centre of the best plateau, not the best single point.
4. Cross-checks on Train before freezing: QQQ and SPY (unlevered twins) must have Sharpe > 0 with the same signs; report
   the plateau stats (share of neighbours with Sharpe >= 75% of centre; peak-to-neighbourhood-median ratio).
5. Deflation: report Train Sharpe against the selection-luck benchmark SE*sqrt(2 ln N_eff).

## Freeze and single-shot Test
Parameters are written to `frozen_params.json` (with sha256 + timestamp) BEFORE the Test window is evaluated by any code.
Test is run once: TQQQ, SPXL, QQQ, SPY, with the frozen parameters and, for context, with the original default parameters
(Dip, RSI2<20, exit SMA5, hold 10, trend 200, dipTrend on) and with buy & hold.
Pass criteria (fixed now): Test Sharpe >= 50% of Train Sharpe on both leveraged assets, same sign on all four assets,
Test MDD <= 1.5x Train MDD. If it fails, the result is reported as a failure; no re-tuning on Test.
