# Residual Prediction - research report

## 1. Literature
| Paper | Finding used here |
|---|---|
| [Krauss, Do & Huck (2017), EJOR 259(2)](https://ideas.repec.org/a/eee/ejores/v259y2017i2p689-702.html) | Trees, random forests and deep nets on lagged returns predict next-day S&P 500 out-performance. Large gross returns, which shrink sharply after 2010 and after costs. |
| [Gu, Kelly & Xiu (2020), *Empirical asset pricing via machine learning*, RFS 33(5)](https://www.nber.org/papers/w25398) | All methods agree on the dominant predictors: momentum variants, liquidity and volatility. Trees and nets beat linear models through interactions. |
| [Guijarro-Ordonez, Pelger & Zanotti (2021), *Deep learning statistical arbitrage*](https://arxiv.org/abs/2106.04028) | Predict factor-model residuals, then trade them under constraints. High out-of-sample Sharpe on daily data (before realistic costs). |
| [Heston, Korajczyk & Sadka (2010)](https://ar5iv.arxiv.org/html/1005.3535); [Gao, Han, Li & Zhou (2018), JFE](https://profiles.wustl.edu/en/publications/market-intraday-momentum/); [Lou, Polk & Skouras (2019)](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf) | Intraday features: same-time-of-day continuation, market intraday momentum, the overnight / intraday split. |

## 2. Design (pre-registered)
* **Features**: 28 residual-based features (`core/resarb/features.py`), each from a paper. Each is measured at the close of bar t and ranked across members.
* **Target**: the ranked residual open-to-open return from the fill bar over h bars.
* **Model**: walk-forward ridge (linear, uses only cross-sectional features) or histogram gradient boosting (can use the market-state features through interactions). Fitted on the last 3 years, refitted quarterly, with an embargo of h + 1 + lag bars so no training target overlaps the forecast period.
* **Feature selection**: from the in-sample IC test only (r1). A feature counts as predictive when |t| ≥ 3 in-sample (Harvey-Liu-Zhu bar). Out-of-sample t-statistics are reported, never used to choose.
* **Hypothesis H3 (QR)**: the residual reversal family (rr_1 … rr_26, ou_s) dominates at short horizons and residual momentum at long ones. A combined model gives a higher IC than any single feature, but costs again decide the sign of the net result.

## 3. Data and test set-up
Same as the other two strategies:
* 306 symbols, 70,692 fifteen-minute SIP bars, 2016-01 → 2026-10-26, with the data fixes.
* Spread-based costs from 2017-2026 SIP quotes; fills at the next bar's open.
* In-sample 2016-09 → 2022, out-of-sample 2023 → 2026-10.
* Walk-forward: models are fitted on the previous 3 years and refitted quarterly, so the first forecasts start in 2018-08. The in-sample period is therefore 2018-08 → 2022.

## 4. Results
**Which features predict the residual?** (`r1_feature_ic.py`; rank IC against the forward residual return, Newey-West t, in-sample / out-of-sample)

| Feature (paper) | 1 bar ahead | 4 bars | 26 bars (1 day) |
|---|---|---|---|
| last bar's residual, rr_1 (Lehmann; Da-Liu-Schaumburg) | **-0.018 (t -14.5) / -0.021 (t -11.5)** | -0.018 (-14.6) / -0.013 (-7.6) | -0.009 (-8.3) / -0.006 (-4.2) |
| close location in the bar, cloc | -0.014 (-13.2) / -0.016 (-10.4) | -0.012 (-11.7) / -0.008 (-5.7) | -0.007 (-7.0) / -0.003 (-2.4) |
| distance from the day's VWAP | -0.007 (-6.1) / -0.012 (-7.1) | -0.013 (-8.2) / -0.009 (-4.1) | -0.009 (-2.9) / -0.005 (-1.3) |
| 13-bar residual, rr_13 | -0.006 (-5.2) / -0.008 (-4.8) | -0.011 (-7.1) / -0.008 (-3.5) | -0.012 (-4.1) / -0.008 (-2.3) |
| idiosyncratic volatility (Ang et al.) | -0.007 (-4.4) / -0.006 (-3.2) | -0.010 (-4.9) / -0.008 (-3.2) | -0.013 (-2.5) / -0.020 (-3.2) |
| volume surprise (Gervais et al.) | +0.003 (3.8) / +0.004 (3.3) | +0.003 (3.0) / +0.005 (3.8) | +0.003 (1.5) / +0.006 (2.2) |
| same time of day, 20 days (Heston-Korajczyk-Sadka) | +0.003 (3.0) / +0.006 (4.7) | ≈0 | ≈0 |
| residual momentum 12-1 (Blitz et al.) | ≈0 | +0.002 (0.9) / +0.007 (3.0) | +0.009 (1.6) / +0.013 (2.0) |
| overnight residual today (Lou-Polk-Skouras) | +0.005 (4.8) / +0.001 (0.8) | ≈0 | -0.007 (-2.1) / -0.004 (-0.9) |
| beta (betting against beta) | -0.005 (-3.7) / -0.006 (-3.0) | -0.005 (-2.4) / -0.007 (-2.2) | -0.011 (-2.1) / -0.018 (-2.5) |

The **15 features selected** on in-sample data (|t| ≥ 3 at 4, 13 or 26 bars) are: rr_1, rr_2, cloc, rr_4, vwap_dev, rr_13, id_today, ivol, rr_26, ou_s, r2_fit, range, ou_kappa, vol_surp and hks. Two caveats:
* **Delay**: the fast features (rr_1, cloc, vwap_dev) lose most of their out-of-sample power with a 15-minute data delay. ivol, vol_surp and momentum don't.
* **Beta extraction**: PCA residuals sharpen the reversal features (rr_1 t -17.0 against -14.5).

**Grid** (`r4_predict.py` + `r8_costaware.py`, **25 variants**: ridge with the selected or all features, horizons 4 / 13 / 26 / 130 bars, rebalanced hourly / daily / weekly, one gradient-boosting run, and partial trading):
* **Hourly rebalancing**: the forecasts are good before costs (in-sample gross Sharpe +1.26 at a 1-hour horizon) but turn over ~1,900x a year. Net in-sample -2.4, out-of-sample -5.0.
* **Daily rebalancing**: gross ≈0.8 in-sample, net 0.3-0.4.
* **Gradient boosting** (same settings as the chosen ridge): in-sample -0.61, out-of-sample +0.43. No better than ridge in-sample.

**Chosen (in-sample best)**: ridge on the 15 selected features, 5-day (130-bar) horizon, rebalanced daily at 15:45, top / bottom 20% (hold until 35%).

| | In-sample (2018-08 → 2022) | Out-of-sample | All |
|---|---|---|---|
| Sharpe (net) | **0.41** | **-0.08** | 0.19 |
| Sharpe before costs | 0.81 | | 0.57 |
| CAGR, volatility, max drawdown | 2.5% | -1.1% | 1.2%, 7.5%, -17.6% |
| Turnover, costs | | | 211x a year, 2.7% a year |

* **Selection tie**: the in-sample runner-up (ridge, all 28 features, 1-day horizon, daily) tied at in-sample 0.41 and made **+0.47 out-of-sample**. That the out-of-sample result flips between two equally good in-sample choices is the selection noise Valid warns about.
* **Robustness**:
  * Costs: Sharpe 0.56 at zero cost, 0.19 at 1x, 0.01 at 1.5x.
  * Fills and delay: VWAP fills give IS 0.46 / OOS -0.02; a one-bar delay gives IS 0.26 / OOS -0.10.
  * Variants: 50 names gives 0.02 / -0.45; PCA residuals give -0.56 / -1.07; yearly instead of quarterly refits give 0.16 / -0.14.
  * Statistics: deflated Sharpe 0.00 formally, 0.15 against noise; probability that the out-of-sample Sharpe is above zero: 0.44.
  * Free IEX feed: score correlation 0.83 with SIP. The bar-shape features (close location, range) differ most between feeds.

## 5. Team review
* **QR**: hypothesis H3 is confirmed on the features and rejected on the trading. Feature discovery works: about ten residual features predict out-of-sample with |t| > 3, and the reversal family dominates, as the Gu-Kelly-Xiu and Krauss papers found for returns. But the combined forecast's edge per trade is below the spread at any horizon where the forecast is strong. Slower horizons have too little forecast left to beat even ~1 bp.
* **Valid**: no look-ahead. Walk-forward with an embargo of horizon + 1 + lag bars; scores are unchanged when future data change; the live model scores with the saved weights. The result depends on the refit frequency, the feature set and the factor model, the typical signature of an overfit or absent edge. **Rejected for capital.**
* **RM**: 211x turnover; the costs (2.7% a year) eat ~70% of the gross. **No capital.** Paper only as an operations test of the ML pipeline: model age check, retrain, live scoring.
* **PM**: 0% allocation. The useful output is the feature table: it says where residual predictability lives (fast reversal, low-risk anomalies, volume) and how quickly it decays.
* **QT**: if this is ever revisited, the route is cheaper execution (passive orders around 15:45, where spreads are 27% tighter), not a better model. Needs the real-time feed: IEX or paid SIP.

## 6. Verdict
**No net edge out-of-sample.** The model and feature research are the deliverable; the strategy should not trade money.
