# Residual Momentum - research report

## 1. Literature (what the idea rests on)
| Paper | Finding used here |
|---|---|
| [Blitz, Huij & Martens (2011), *Residual momentum*, J. Empirical Finance 18(3)](https://repub.eur.nl/pub/22252) | Ranking on the residual of a 36-month Fama-French regression, summed over months t-12..t-2 and divided by its volatility, gives about twice the risk-adjusted return of total-return momentum, with fewer crashes. The improvement comes from removing the momentum strategy's time-varying factor bets. |
| [Huij & Lansdorp (2017)](https://repub.eur.nl/pub/100801) | Out-of-sample confirmation in other markets and later years. |
| [Jegadeesh (1990)](https://alphaarchitect.com/quantitative-momentum-research-short-term-return-reversal/), [Lehmann (1990)](https://www.nber.org/papers/w2533) | The last week to month reverses, which is why the most recent month is skipped. |
| [Lou, Polk & Skouras (2019), *A tug of war*, JFE 134(1)](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf) | Momentum profits are earned overnight, and are stronger for large caps. Trading near the close keeps the overnight. |
| [Ehsani & Linnainmaa (2022), *Factor momentum and the momentum factor*, J. Finance 77(3)](https://www.nber.org/papers/w25551) | Stock momentum partly times other factors, so stripping the factors changes what momentum is. |

## 2. Design (pre-registered before any backtest)
* **Universe**: top-100 U.S. stocks by liquidity, chosen monthly and point-in-time.
* **Beta extraction**: every morning, SPY and sector-SPDR-excess betas from 252 daily returns.
* **Signal**: residual return over `form_days`, skipping the last `skip_days`, divided by its volatility × √n (a t-statistic).
* **Book**: long the top 20% and short the bottom 20%, equal weight; a name already held stays until it leaves the top / bottom `q_out`. Hedged with SPY plus the sector ETFs.
* **Trading**: the 15:45 bar of the rebalance day (monthly / weekly / daily).
* **Hypothesis H1 (QR)**: on top-100 large caps the 12-1 month residual momentum earns a positive but modest gross Sharpe (0.3-0.8). Turnover is low (≈5-10x a year), so costs barely matter. The literature's numbers come from broader universes and older samples; large caps after 2016 are the hardest case.

## 3. Data and test set-up
* **Universe and bars**: 306 symbols (292 stocks that were top-100 members at some point, plus 14 ETFs). 70,692 fifteen-minute SIP bars, 2016-01-04 → 2026-10-26, split and dividend adjusted.
* **Data fixes**: four distributions Alpaca leaves unadjusted (XLF→XLRE 2016, JCI→Adient 2016, 21st Century Fox→Fox Corp 2019, UTX→RTX 2020) are back-adjusted. The erroneous NYSE opening prints of 2023-01-24 are replaced by the bar's VWAP. TQQQ / SQQQ are excluded from the stock universe.
* **Costs**: each name's modelled half spread at that bar plus 0.5 bp (fitted on 2017-2026 SIP quotes; ≈1.0 bp midday for a typical member in 2017-19, 2.6 bp in 2025-26). Hedge ETFs pay 0.5 bp, shorts 30 bp a year.
* **Timing and samples**: fills at the open of the bar after the data. In-sample 2016-09 → 2022-12, out-of-sample 2023-01 → 2026-10. Choices are made on in-sample Sharpe only.

## 4. Results
**Feature evidence** (`research/r1_feature_ic.py`): the 12-1 residual momentum's rank IC against the next day's residual return is +0.0085 in-sample (t 1.6) and +0.013 out-of-sample (t 2.0). A 15-minute data delay changes nothing (+0.0084 / +0.0136): this signal is slow.

**Grid** (`r2_momentum.py` + `r8_costaware.py`, **105 variants**: formation 21-252 days × skip 0/5/21 × monthly / weekly / daily × three quantile rules, plus partial trading). The paper's 12-1 month, monthly setting earns Sharpe 0.07 in-sample and 0.29 out-of-sample (gross in-sample 0.10). Only 17% of the variants are positive in-sample, but 91% are positive out-of-sample. On these large caps residual momentum did not pay in 2016-2022 and did in 2023-2026.

**Chosen (in-sample best)**: 63-day formation, skip 21 days, monthly at 15:45, enter the top / bottom 10%, hold until a name leaves the top / bottom 25%, ETF-hedged.

| | In-sample 2016-09 → 2022 | Out-of-sample 2023 → 2026-10 | All |
|---|---|---|---|
| Sharpe (net) | 0.37 | **0.60** | 0.47 |
| CAGR at 1.0x stock gross | 3.0% | 6.5% | 4.3% |
| Volatility, max drawdown | | | 10.2%, -16.1% |
| Turnover, costs | | | 21x a year, 0.3% a year |
| Beta to SPY, names held | | | -0.005, ~26 |

Years: 2016 -3.4%, 2017 +11.0%, 2018 +10.8%, 2019 -2.9%, 2020 +8.3%, 2021 -2.3%, 2022 -1.0%, 2023 -5.0%, 2024 +4.8%, 2025 +5.6%, 2026 (to 10-26) +20.7%.

**Robustness** (`r6_robustness.py`, `r5_feedgap.py`, `r9_margin.py`):
* **Costs** barely matter: Sharpe 0.49 at zero cost, 0.47 at 1x, 0.41 at 3x.
* **Fill and delay**: fills at the bar VWAP give the same result (IS 0.36 / OOS 0.62); a one-bar delay (free delayed SIP) is identical.
* **Data feed**: IEX and SIP scores correlate 0.999.
* **Variants**: 50 names gives IS 0.63 / OOS 0.72; PCA residuals with a projection hedge give IS 0.51 / OOS 0.43.
* **Statistics**:
  * deflated Sharpe ratio after 105 trials: 0.43 formally, 0.05 against noise only;
  * probability that the out-of-sample Sharpe is above zero: 0.88.
* **Capital use**: stocks plus ETF hedges average 1.62x of capital overnight (95th percentile 2.04x). The default stock gross is therefore 0.9: about 4.0% a year at 9.5% volatility.

## 5. Team review
* **QR**: hypothesis H1 is half confirmed. The effect is positive and cheap to trade, but weaker than in the paper, and it lives in 2023-2026, not 2016-2022. The 3-1 window beating 12-1 on large caps is plausible: faster information diffusion in mega caps. But it was chosen from 105 tries.
* **Valid**: no look-ahead was found. Betas, residuals and the score are tested to be unchanged when later data change, and live / backtest targets match exactly in the parity test. The in-sample edge is not distinguishable from selection luck (DSR 0.05-0.43, below 0.95); the out-of-sample 0.60 is encouraging but rests on 3.8 years (t ≈ 1.2). **Not signed off for capital.** Signed off for a paper forward test.
* **RM**: the risk is a 10% volatility book with -16% max drawdown, beta about 0 and 10-26 names of at most 5% each. Single-name gap risk is about 1% of capital per earnings miss. The monthly rebalance makes the operations simple. Size stock gross 0.9 so stocks plus hedges stay under Reg-T's 2x overnight; the engine caps the total at 1.9x. **Approved for its own paper account only**: daily-loss halt 3%, drawdown stop 15%.
* **PM**: the only one of the three with an out-of-sample result worth forward testing; allocate 0% real capital. Forward-test for 6+ months alongside the earnings sleeve of the Stats-arb project; re-decide when out-of-sample t > 2.
* **QT**: trades once a month at 15:45 (the cheapest spread of the day). Market orders are fine at this turnover; use the free **delayed SIP** feed (the default), since IEX adds nothing for a monthly signal.

**As configured in the app** (research choice + stock gross 0.9 + the 1.9x cap on stocks plus hedges, which the backtest applies exactly like the live engine): Sharpe 0.46 (IS 0.37, OOS 0.59), CAGR 3.8%, vol 9.1%, max drawdown -14.5%, turnover 19x a year. Stocks plus hedges average 1.46x of capital overnight (95th percentile 1.84x).

**Live-engine fixes before the paper test (2026-10-26)**:
* Month ends now come from the NYSE calendar. The old rule treated the newest day in the data as a month end, so in live the monthly strategy would have rebalanced every day.
* A drawdown stop now flattens at the next bar instead of waiting for the next monthly rebalance.
* The warm-up now covers the beta window plus the formation window, so live momentum scores equal the backtest's.
* The ETF hedge is sized from the whole shares actually held.
* The backtest applies the same gross cap as the live engine.

Each fix has a test that fails without it. On real data, the engine decides on 2026-09-30 and not on 09-29, and its targets equal the backtest's to 3e-15 (`research/momentum_schedule_check.py`).

## 6. Verdict
**Paper forward test: yes. Real capital: no.** It is the strongest of the three: positive out-of-sample, robust to costs, fills, delay and factor model. But the in-sample record is flat and the out-of-sample run is too short for significance.
