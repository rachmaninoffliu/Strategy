# Diversifier - final parameters (Train/Test, 2026-10-07)

## Final set (frozen before Test, sha256 a10148de042fc4686b7344252976b505f9fe576576e02aeef4b4f0edf4f6841f)

| Input | Value |
|---|---|
| Timeframe | **Daily** |
| Mode | **Dip** |
| Proxy (signal) | QQQ for TQQQ/QQQ, SPY for SPXL/SPY (Auto-detect on) |
| rsiLen | **4** (Wilder RSI on the proxy) |
| rsiEntry | **34** (enter when RSI < 34) |
| exitLen | **8** (exit when proxy close > SMA 8) |
| trendLen | **225** (dipTrend ON: dips are only bought when proxy close > SMA 225) |
| maxHold | **20** days (safety cap, not binding: any value >= 16, OFF and 30 give identical Train results) |
| Costs assumed | 0.1% per side, 100% equity, long-only, fills at next open |

Files: `data/opt/frozen_params.json`, `data/opt/test_results.json`, engine `scripts/opt/sim.py`, protocol `data/opt/PROTOCOL.md`.

## Evidence

Train 2010-02-11..2021-12-31 (selection only), Test 2022-01-01..2026-10-06 (touched once, after the freeze).

| | Train Sharpe | Train Calmar | Train CAGR / MDD | Test Sharpe | Test Calmar | Test CAGR / MDD | Test B&H Sharpe / MDD |
|---|---|---|---|---|---|---|---|
| TQQQ | 0.80 | 0.55 | 22.3% / 40.5% | 0.76 | 0.59 | 19.6% / 33.1% | 0.56 / 81.1% |
| SPXL | 0.78 | 0.45 | 18.0% / 40.3% | 0.37 | 0.17 | 5.8% / 34.1% | 0.56 / 63.8% |
| QQQ  | 0.62 | 0.43 | 6.3% / 14.8% | 0.68 | 0.54 | 6.4% / 11.9% | 0.71 / 35.2% |
| SPY  | 0.59 | 0.34 | 4.8% / 14.1% | 0.24 | 0.12 | 1.5% / 13.2% | 0.69 / 25.4% |

Pre-registered criteria: (1) Test Sharpe >= 50% of Train on TQQQ and SPXL - **FAIL (SPXL 0.37 vs 0.39 required)**; (2) positive on all four - pass; (3) Test MDD <= 1.5x Train MDD - pass. **Formal verdict: FAIL, narrowly, on SPXL only. No re-tuning on Test.**

Original defaults (RSI2<20, SMA5, hold 10, trend 200) on Test: TQQQ 0.81, SPXL 0.63, QQQ 0.66, SPY 0.40 - the default is statistically indistinguishable from the optimised set on TQQQ/QQQ and was better on SPXL/SPY. The optimisation did not buy a measurable out-of-sample improvement.

Significance: Test Sharpe TQQQ 0.76 +- 0.52 (t = 1.45), SPXL 0.37 +- 0.47 (t = 0.78). Train Sharpe 0.80 is below the selection-luck benchmark (~1.7 for ~1M trials), so Train alone proves nothing; the Test is the evidence and it is weak-positive for TQQQ, inconclusive for SPXL.

## Timeframe / class decisions (Train only)
* 1h Dip class: best plateau SPXL Sharpe ~0.43 vs 0.78 daily, optimum pinned at grid edges -> daily wins.
* 1h Trend+Dip top (RSI<10, SMA17, trend 250) is Trend-250 in disguise (pure 1h Trend-250: TQQQ 0.95 / SPXL 0.82 Sharpe; RSI entry flat from <2 to <14): 91% exposure, corr 0.84 to B&H, 22% win rate. That is a leveraged trend-follower, not a diversifier; not eligible, and not tested (open lead, needs its own protocol and a fresh out-of-sample period).
* 15m has no Train coverage (data from 2020-08-03); 30m not run.

## Honest caveats
* Behaviour on the Test: strategy lost -15% in 2022 vs -80% B&H (TQQQ), but only +5% in 2023 (B&H +193%) - a dip-buyer with ~20% exposure gives up bull-market upside by design.
* SPXL/SPY side degraded OOS (2023 -9%); longest underwater 528 days (TQQQ) / 738 days (SPXL).
* Leveraged-ETF gap/decay risk is not in the backtest beyond the 0.1% commission; no slippage modelled.
* Exact frozen set not re-run in the TradingView Pine script (engine matched TradingView in 7/7 earlier comparisons incl. trade counts; enter the inputs above into "Diversifier Optimizer (Train-Test)" to double-check).
