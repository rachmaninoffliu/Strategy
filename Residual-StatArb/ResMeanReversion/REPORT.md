# Residual Mean Reversion - research report

## 1. Literature
| Paper | Finding used here |
|---|---|
| [Avellaneda & Lee (2010), *Statistical arbitrage in the U.S. equities market*, Quantitative Finance 10(7)](https://math.nyu.edu/~avellane/AvellanedaLeeStatArb20090616.pdf) | ETF- or PCA-factor residuals, cumulative residual modelled as Ornstein-Uhlenbeck, s-score thresholds (open 1.25, close 0.5 / 0.75). Net Sharpe 1.1-1.4 over 1997-2007, falling after 2003. |
| [Yeo & Papanicolaou, *Risk control of mean-reversion time*](https://content.iospress.com/articles/risk-and-decision-analysis/rda132) | Trade only residuals that revert fast and fit well. |
| [Da, Liu & Schaumburg (2014), *A closer look at the short-term return reversal*, Management Science](https://newyorkfed.org/medialibrary/media/research/staff_reports/sr513.html) | The reversal lives in the residual (within-industry, non-fundamental) part: about 3x the plain reversal profit, 1982-2009. |
| [Lehmann (1990), QJE](https://www.nber.org/papers/w2533) | Weekly cross-sectional reversal. |
| [Heston, Korajczyk & Sadka (2010), J. Finance 65(4)](https://ar5iv.arxiv.org/html/1005.3535) | At 30-minute lags returns reverse, which they attribute to bid-ask bounce and thin resiliency. That is why fills must not be taken at the signal bar's close. |
| [Gatev, Goetzmann & Rouwenhorst (2006), RFS 19(3)](https://ideas.repec.org/p/nbr/nberwo/7032.html); [Krauss (2017) survey](https://ideas.repec.org/p/zbw/iwqwdp/092015.html) | Relative-value reversion; profits shrank after 2000. |

## 2. Design (pre-registered)
* **Residuals**: 15-minute residuals of the top-100 stocks (ETF model, 126-day betas).
* **`ou` mode**: an AR(1) fit of the cumulative residual over `ou_window` bars gives the s-score. Open at |s| ≥ `s_open` when the mean-reversion time is under half the window; close at |s| ≤ `s_close`. Each position is hedged with SPY plus the sector ETF.
* **`xs` mode**: rank the members by their residual return over the last k bars, buy the bottom fraction, short the top.
* **Hypothesis H2 (QR)**: gross reversal at intraday horizons is real (documented). Net of 2 bp per side on stocks plus the ETF hedge, with turnover of hundreds of times a year, it is likely negative. The daily Avellaneda-Lee signal in the Stats-arb project already had about zero net edge over 2017-2026. The fill-price check (bar VWAP) matters here: a reversal edge that only exists at the open print is not capturable.

## 3. Data and test set-up
* **Universe and bars**: 306 symbols, 70,692 fifteen-minute SIP bars, 2016-01 → 2026-10-26.
* **Data fixes**: unadjusted distributions are back-adjusted, the 2023-01-24 bad opening prints are replaced by the bar's VWAP, and TQQQ / SQQQ are excluded.
* **Costs**: each name's modelled half spread per bar plus 0.5 bp, from 2017-2026 SIP quotes; ≈1.0 / 1.8 / 2.6 bp at midday in 2017-19 / 2020-24 / 2025-26, +45% in the first hour.
* **Timing and samples**: fills at the open of the bar after the data. In-sample 2016-09 → 2022, out-of-sample 2023 → 2026-10; choices made on in-sample Sharpe only.

## 4. Results
**Feature evidence** (`r1_feature_ic.py`): the residual reversal is the strongest and most significant signal in the whole study.
* The last bar's residual return has a rank IC of **-0.018 against the next bar (t -14.5 in-sample, -0.021 / t -11.5 out-of-sample)**.
* The 2-, 4- and 13-bar residual returns, the close's location in the bar, and the distance from VWAP say the same thing.
* PCA residuals make it stronger (IC -0.020, OU s-score t -6.8 against -2.2 for ETF residuals).
* It is also **fast**. With a 15-minute data delay the next-bar IC falls to -0.0005 out-of-sample, and at the 4-bar horizon from -0.013 to -0.004.

**Grid** (`r3_meanrev.py` + `r8_costaware.py`, **152 variants**):
* **OU s-score** (window 26-260 bars, thresholds 1.25-2.0 / 0.25-0.75, 15 min / 1 h, with and without overnight).
* **Cross-sectional reversal** (k = 1-130 bars, 10% / 20% quantiles, 15 min / 1 h / daily).
* **Partial trading** (Gârleanu-Pedersen trade rate 0.1-0.5).
* **Gross edge**: 107 of the 124 round-1 variants are positive before costs. The best gross in-sample Sharpe is **+2.9** (reverse the last 15 minutes, every 15 minutes) at **11,000x turnover a year → net -19**. Only 9 of 124 are positive after costs.
* **Paper settings** (Avellaneda-Lee: 130-bar window, 1.25 / 0.5, every 15 min): in-sample gross -0.32, net -2.58.
* **Partial trading** lowered turnover, but the gross edge fell almost as fast; no variant is positive out-of-sample.

**Chosen (in-sample best)**: reverse the last hour's residual (k = 4 bars), decided at 15:30, traded at 15:45, held to the next day's 15:45. Enter the top / bottom 10%, hold until a name leaves 25%, ETF-hedged.

| | In-sample | Out-of-sample | All |
|---|---|---|---|
| Sharpe (net) | **0.96** | **-0.20** | 0.47 |
| Sharpe before costs | 1.62 | | 1.13 |
| CAGR (1.0x gross) | 9.0% | -3.0% | 4.3% |
| Volatility, max drawdown | | | 10.2%, -17.3% |
| Turnover, costs | | | 543x a year, 6.6% a year |

Years: 2016 +3.4%, 2017 -3.7%, 2018 +0.7%, 2019 +0.5%, **2020 +27.8%, 2021 +20.3%, 2022 +11.2%**, 2023 -0.7%, 2024 -8.2%, 2025 -2.8%, 2026 +0.8%. The money was made in the volatile 2020-2022 markets.

**Robustness**:
* **Costs**: Sharpe 1.11 at zero cost, 0.47 at 1x, 0.14 at 1.5x, -0.18 at 2x.
* **Fill**: VWAP fills give the same result (IS 0.99 / OOS -0.19), so it is not an open-print artefact.
* **Delay and feed**: a one-bar delay gives IS 0.49 / OOS -0.38. Over 2024-26: real-time SIP -0.23, free IEX -0.24 (score correlation 0.97 with SIP), free delayed SIP -0.38.
* **Variants**: 50 names gives IS 0.55 / OOS -0.72; PCA gives IS 0.16 / OOS -0.25.
* **Statistics**: deflated Sharpe 0.00 formally, 0.20 against noise; probability that the out-of-sample Sharpe is above zero: 0.35.
* **Capital use**: stocks plus ETF hedges average 1.63x overnight (95th percentile 2.27x), so the default stock gross is 0.8.
* The 5-day residual reversal (k = 130, daily) was steadier (in-sample 0.35, out-of-sample 0.71). It was **not** the in-sample choice; picking it now would be choosing on out-of-sample data.

## 5. Team review
* **QR**: hypothesis H2 is confirmed. Reversal in 15-minute to 1-day residuals is real and highly significant, but its gross profit per trade is below the spread. The edge decays within one bar, so the cost of crossing the spread can't be avoided by trading slower. Trading slower kills it.
* **Valid**: no look-ahead (scores are unchanged under future perturbation; live / backtest parity holds). The in-sample 0.96 is selection plus a regime (2020-22): it does not survive out-of-sample, costs x1.5, or a one-bar delay. **Rejected for capital. Not recommended even for paper**, except as an operations test.
* **RM**: high turnover (543x) makes the result hostage to spreads. The hedge is capital-heavy: today's book needed 1.8x of ETFs for 0.8x of stocks, and the engine scaled it down to the 1.9x cap. Overnight single-name gap risk on the reversal longs (yesterday's losers). **No capital.**
* **PM**: 0% allocation. Its only value is as a short-horizon feature inside a model, and the prediction strategy shows that doesn't pay either.
* **QT**: a real-time feed is mandatory (the delayed SIP drops the signal). IEX is fine for it, at a score correlation of 0.97. But no execution style can recover a 1-2 bp edge from 1-2.6 bp spreads with market orders. It would need passive limit orders, which the bar backtest cannot model honestly.

## 6. Verdict
**No edge after realistic costs out-of-sample.** The app, the engine and the research tools are complete, so the user can rerun it with other settings, for example PCA residuals or limit-order execution research. It must not trade money.
