# Intraday-A3C: session-convergence strategy for US cash equities

Source paper: Demir, Kok, Seeman, Mandic, Wiering, *"Algorithmic trading in intraday electricity markets with deep reinforcement learning"*, Applied Energy 2022 (S0306261922003348).
Status (2026-10-24): research complete, DRY-RUN paper runner built (`scripts/ia_live.py`, section 9). **No order has ever been sent, nothing is approved for live capital.** Valid clears a forward paper test only.

Correction made 2026-10-24 while building the runner: an earlier version of this note said the 72 decisions run 09:35-15:50 and that the 15:50 decision had to be frozen for the MOC. They run **09:35-15:30**; the position left by the last decision is simply entered into a market-on-close order right away (deadline 15:50). Backtest numbers were not affected (they use bar indices), and the "15:00-15:50" label of the last time-of-day bucket in the audit is now "15:05-16:00".

## 1. Verdict

| | |
|---|---|
| Does the paper's own idea (price converges to a volatile, known-afterwards settlement; fade deviations) transfer? | **No.** The VWAP-reversion rule loses net of cost on all 22 products (net Sharpe -0.6 to -3.1, mean -1.8). |
| What survives? | A walk-forward model of *the return from now to the official close* finds a trend-continuation edge, concentrated in high-volatility Nasdaq names, earned mostly in the first two hours. |
| Product | **TSLA** (if one line only), traded as an equal-risk basket **TSLA + SMH + NVDA** (net Sharpe 1.52 in the 2019-26 walk-forward, but the three names were picked after seeing results; plan on 0.8-1.0, my judgment between the 0.54 of the unselected 22-product pool and the ~1.1 of the honest year-by-year selection test). **Not** QQQ, SPY, TQQQ, UPRO, sector ETFs, AAPL, MSFT, META. |
| Live data | This paper account may **not** query recent SIP data, only IEX. The SIP-trained models run on IEX bars with almost no loss (edge correlation 0.98-0.99, basket Sharpe 1.27 vs 1.34 on 2022-26, `reports/feed_test.log`). |
| Next step | Forward paper test with the DRY-RUN runner, 60 trading days minimum, gate in section 7. |

## 2. Paper -> US equities

| Paper (electricity, German intraday) | Here (US cash equity, one product) |
|---|---|
| Hourly contract, trading closes at delivery | One regular session; every position is flat by the close, no overnight risk |
| Balancing price: volatile, known afterwards, leftover inventory settles at it | **Official closing-auction price** (Alpaca daily-bar close). A market-on-close order entered by 15:50 ET fills at it with no spread |
| Aggressor agent {Buy, Sell, Hold} at the best quote | Marketable orders at the touch every 5 minutes, 72 decision points 09:35-15:30 |
| v_max = 3 MW, q_high = 40 MW | +-3 lots held, at most 40 lots traded a day |
| Hindsight label (eq. 6): what a buy now earns if settled | y = (official close - price now) / price now |
| Trading cost 0.232 EUR/MW | Half the quoted spread (SIP, sampled) + 0.25 bp slippage per lot |
| Reward shaping from session low/high and settlement | Not used: the surrogate regresses y directly, the policy is a function of the edge |
| Rolling monthly retraining | Expanding window, refit once a year (tested); monthly refit is untested |
| Metrics: PnL > 0 share, PD, PT | Net Sharpe, t-stat, hit rate, net bp per lot traded (PT) |

**No A3C network was trained.** The agent is a supervised surrogate: walk-forward Ridge and HistGradientBoosting forecasts of y from the paper's kind of state, turned into a position by `hold = clip(floor(edge / (k * cost)), -3, +3)`, k = 3 on the ensemble mean. Rationale: if a regressor with the same information finds no net edge, an RL agent can only overfit it. The surrogate did find an edge, so an A3C agent is now a legitimate phase-2 experiment, with one rule: it must beat this surrogate net of cost on the same walk-forward folds, or it is not used.

State (16 features, all known at the decision time): time of day, distance to session VWAP, return from open, return from previous close, opening gap, 1/3/6/12-bar returns, position in the day range, range so far, 12-bar realised vol, volume pace vs the trailing 20-day mean at the same time, first-30-minute return, previous-day return, day of week.

## 3. Data and cost model

- 22 candidate products, Alpaca SIP 5-minute bars, split-adjusted, 2016-01 to 2026-10, regular hours only; days need all 78 bars. Official close from daily bars.
- Test period 2019-2026 (first model trained on 2016-18), one test year at a time, training strictly earlier.
- Spreads: SIP quotes, 1-minute windows at 10:30, 13:00, 15:30 ET on 3 days in October 2026. **Not sampled at the open.** Stress C below covers this.
- Fills: at the 5-minute bar close at the decision time, plus half spread and 0.25 bp. No queue position, no latency model beyond that slippage.

## 4. Evidence

Per-product net result (ensemble, k = 3, 2019-26, max 3 lots; full table `reports/decision_table.csv`):

| Product | net Sharpe | t | 2019-22 / 2023-26 | years positive | cost bp/lot | ADV $M |
|---|---|---|---|---|---|---|
| TSLA | 1.14 | 3.16 | 1.04 / 1.28 | 8/8 | 1.07 | 22,057 |
| SMH | 1.11 | 3.09 | 1.06 / 1.16 | 6/8 | 1.39 | 3,812 |
| NVDA | 1.04 | 2.90 | 0.58 / 1.60 | 6/8 | 0.71 | 29,835 |
| AMZN | 0.92 | 2.55 | 0.89 / 0.97 | 6/8 | 1.01 | 9,863 |
| SOXL | 0.82 | 2.26 | 0.95 / 0.69 | 6/8 | 1.57 | 6,330 |
| IWM | 0.61 | 1.70 | 0.52 / 0.76 | 6/8 | 0.43 | 7,981 |
| QQQ | 0.46 | 1.28 | 0.69 / 0.15 | 6/8 | 0.39 | 30,333 |
| SPY | 0.34 | 0.96 | 0.71 / -0.20 | 3/8 | 0.38 | 42,307 |
| TQQQ | 0.26 | 0.73 | 0.48 / -0.03 | 4/8 | 0.96 | 4,759 |
| AAPL, MSFT, META, XLK, XLF, XLE, GLD, TLT | -1.0 to 0.0 | | | | | |

Pools (equal risk, no hand selection): all 22 products 0.54; Nasdaq complex 0.77; mega-cap stocks 0.82; index ETFs 0.51; sector ETFs 0.02; leveraged ETFs 0.47.
Honest selection (each year pick top-N by trailing Sharpe, trade next year, 2021-26): 0.85-1.31 depending on model, N = 3 gives 0.95-1.31.

Valid's audit (`scripts/ia_audit.py`, `reports/audit.log`):

- **A. Slippage and sampling.** Total per-lot cost raised to 1 bp / 2 bp: TSLA 1.06 / 0.93, SMH 0.87 / 0.70, NVDA 0.74 / 0.54, **AMZN 0.20 / -0.20, QQQ -0.08 / -0.31**. Block-bootstrap 5-95%: TSLA 0.69-1.62, SMH 0.52-1.63, NVDA 0.48-1.64, QQQ -0.08-0.95.
- **B. Not just drift.** Both legs earn (TSLA gross bp/day: long 50.6, short 17.4); average signed position is +0.83 lots of an average 2.06, so there is a long bias, and daily P&L correlates +0.45 with the day's return (it earns on trend days). A day-shuffle placebo, which keeps the long bias and the time-of-day shape but removes day-specific information, gives mean Sharpe -0.17 and a maximum of +0.67 over 100 shuffles for TSLA (actual 1.14); SMH max +0.26 (actual 1.11); NVDA max +0.69 (actual 1.04); AMZN max +0.22 (actual 0.92); QQQ max +0.52 (actual 0.46, barely above the placebo 95th percentile of +0.28). Flipping the sign gives -1.0 (QQQ) to -2.3 (SMH).
- **C. Open-hour costs.** TSLA gross P&L is 26 bp/day from 09:35-10:30 and 24 bp/day from 10:30-12:00, so the unmeasured open spread matters. Tripling cost before 10:00: TSLA 0.90, SMH 0.73, NVDA 0.88. Not trading before 10:00 at all: TSLA 1.35, SMH 1.10, NVDA 0.89. The edge does not depend on the open.
- **D. Basket.** Equal-risk TSLA + SMH + NVDA 1.52 (mean pairwise correlation 0.28); with no trading before 10:00, 1.55. By year (no trading before 10:00): 2019 -0.32, 2020 2.39, 2021 2.96, 2022 0.50, 2023 1.43, 2024 2.17, 2025 2.03, 2026 1.34 (partial year).
- **E. Is the product choice hindsight?** Partly. Intraday volatility divided by cost, measured on 2016-18 only, rank-correlates with the 2019-26 net Sharpe at +0.60 (p = 0.003). It puts NVDA first and TSLA fifth, but also TQQQ, SOXL, QQQ and IWM, which delivered 0.26 / 0.82 / 0.46 / 0.61, and it ranks SMH low. So the volatility-to-cost story supports TSLA and NVDA, and SMH rests on the empirical result alone.
- **Multiple testing.** Best-of-22 one-sided p for the three leaders: TSLA ~0.02, SMH ~0.02, NVDA ~0.04 before counting the three model configurations and the design iterations, and ignoring return autocorrelation. Suggestive, not proven.

What did not work: VWAP reversion (the paper's behaviour) loses everywhere; simple rule tests (22 products x 7 entry times x 4 rules = 616 full-sample tests, held to the close) average net Sharpe +0.17 for open momentum, +0.11 for previous-day momentum, -0.08 for first-30-minute momentum and -0.58 for VWAP reversion; the model's edge comes from combining the features, not from any single rule.

## 5. Strategy specification (what a paper runner would do)

1. **Universe/lines:** TSLA, SMH, NVDA, one line each. Optional fourth line AMZN only after the gate, it is the first to die when costs rise.
2. **Clock:** decision every 5 minutes at the bar close, 09:35-15:30 ET. Recommended: **no new risk before 10:00** (first trade at the 10:00 decision; same Sharpe as the full day, avoids the unmeasured open spread).
3. **Signal:** `edge = mean(ridge, gbm)` of y in bp from the 16 features; model refit each January on all earlier data (expanding).
4. **Position:** `hold = clip(floor(edge / (3 * cost)), -3, +3)` lots, `cost = half spread (live quote) + 0.25 bp`; at most 40 lots traded per line per day; flat to the close.
5. **Exit:** whatever is held after the 15:30 decision goes into a market-on-close order immediately (deadline 15:50); the position then settles at the official close, 30 minutes of exposure after the last decision.
6. **Sizing:** inverse-volatility weights across lines. A line run at full 3 lots has daily P&L vol of 2.5% (TSLA), 2.0% (NVDA) and 1.2% (SMH) of its capital, so equal risk is roughly 23% / 29% / 48% of the capital deployed. Scale total deployed capital to a portfolio vol target (about 8% a year needs roughly 40% of equity at full lots; derived from the audit figures, confirm in the runner).

## 6. Team positions

- **QR:** the edge is real in the sense that matters (placebo, sign-flip, split-sample), but it is a trend effect in high-beta Nasdaq names, which is the opposite of the paper's mean reversion. Expect decay as in QQQ 2023-26.
- **DA:** closing price is Alpaca's consolidated daily close, treated as the official auction print; unverified against the exchange prints. Spreads sampled on 3 days only.
- **Dev:** DRY-RUN runner built and tested (section 9). Not built: order submission (deliberately), reconciliation against a broker, a GUI.
- **Valid:** cleared for forward paper test only. Open items: hindsight product selection, 22 x 3 configurations searched, fills at the bar close, unmeasured open spread, closing-print definition.
- **RM:** see section 7.
- **QT:** TSLA/NVDA/SMH are the three most liquid names with real depth; the risk is the first hour (6.6 lots/day traded at the open for TSLA) and halts (an MOC order that cannot enter leaves an overnight position).
- **PM:** a Sharpe ~1 intraday sleeve that is flat overnight is a different return source from the earnings-premium sleeve in `Stats-arb` (correlation between the two not measured) and is worth running at paper size; do not allocate capital before the gate.

## 7. Gate and risk limits

Paper test (Alpaca paper account, DRY RUN first, ARMED only on explicit instruction):

- Minimum 60 trading days, all three lines, no-new-risk-before-10:00 variant.
- **Pass** requires: realised cost per lot (vs the bar close, from consolidated quotes) within 1.5x of the model cost, shadow net Sharpe > 0.5 over the window, every day's position covered by a logged MOC intent, and the live-feed (IEX) edge correlating >= 0.90 with the SIP edge of the same day (backtest 0.98-0.99). Edge-sign agreement with the realised return to the close is reported but is not a gate: even in the backtest it is only 0.51-0.53, the edge is in the size of the moves, not the hit rate.
- **Fail** (stop): net Sharpe < -0.5 after 40 days, any overnight position, realised cost above 2.5x the model cost.

Risk limits (RM): max 3 lots per line, 40 lots traded per line per day, no margin (total deployed notional <= 100% of equity), daily loss halt at 1.0% of equity (flat everything, no new risk until next session), drawdown stop at 6% of the strategy's equity, no trading on halted/LULD names, no decisions on early-close days (the 78-bar models do not apply), kill switch flattens via market orders if MOC entry fails. Pattern day trader rules apply to a margin account under $25k equity: check current rules before any live account.

## 8. Reproduce

Run from `scripts/` in this order: `ia_data.py` (about 20 minutes, downloads the bars), `ia_spreads.py` (needs market hours for fresh quotes; cached in `reports/spreads.csv`), `ia_study.py` (product screen), `ia_rules.py`, `ia_portfolio.py` (caches the walk-forward edges in `data/`), `ia_decide.py`, `ia_audit.py`, `ia_feed_test.py` (needs the IEX history of TSLA/SMH/NVDA, fetched with `ia_data.fetch_bars(sym, 5, feed="iex")`). Logs and tables are in `reports/`.

## 9. The DRY-RUN paper runner (`scripts/ia_live.py`)

It decides every 5 minutes with the same features, models, cost rule and caps as the backtest, keeps a shadow book, and writes what it *would* send. **It has no order endpoint**: the only Alpaca calls are market-data reads and the paper account's clock, calendar and balance (`TradingClient(paper=True)`); `test_ia_live.py` fails if the source contains an order call. There is no ARM mode; adding one is a separate decision.

```
python scripts/ia_live.py status                  # paper account, data entitlements, models, kill file
python scripts/ia_live.py run                     # start before 09:30 ET; runs the session, logs MOC intents, settles itself after 16:20
python scripts/ia_live.py replay --date 2026-10-23 --feed iex     # any past session through the same code path
python scripts/ia_live.py settle --date YYYY-MM-DD                # re-mark a logged day on the SIP tape + official close
python scripts/ia_live.py report                  # the paper-test gate (section 7)
python scripts/ia_live.py train [--force] | refresh               # yearly model refit | extend the research parquets
python scripts/test_ia_live.py                    # 14 tests
python scripts/ia_fillmodel.py                    # fill-price audit (section 10), writes reports/fillmodel.log
python scripts/ia_replay_range.py 15 3            # last 15 sessions, IEX vs SIP, quotes for the last 3
```

Per session it writes `live/<date>/decisions.csv` (every decision: price, ridge/gbm/ensemble edge, cost, target, trade, position, notes), `intents.jsonl` (dry-run order intents and the MOC intents), `state.json` (restart-safe), and after settlement a row per line in `live/daily_summary.csv`. Touch `live/KILL` to make it flatten and stay flat.

Guards: previous-session closes must be present (otherwise it refuses to trade), a missing bar is carried forward and three in a row hold the position, early-close days are skipped, loss halt 1.0% of equity, drawdown stop 6%, 40-lot cap that never blocks a flatten, restart resumes positions and counters without repeating a decision.

**Live data.** This paper account is refused real-time SIP ("subscription does not permit querying recent SIP data"), only IEX works. The runner therefore feeds the SIP-trained models with IEX bars (`--feed iex`, default); a 5-minute bar with no IEX trade is carried forward. Fidelity (`reports/feed_test.log`, 2022-26, same rule): edge correlation with the SIP edge 0.99 / 0.98 / 0.99 (TSLA / SMH / NVDA), basket Sharpe 1.27 on IEX signals vs 1.34 on SIP signals. After the close, settlement re-marks every position on the consolidated tape and the official close, and replays the SIP signals through the same engine so the gap is measured each day. Real-time SIP would remove the gap; it is a paid data plan.

**Verified.** 14/14 tests (the fourteenth checks the output text, below): live features equal `ia_study` features (25 sessions x 20 decisions, max difference 0), live-trained models reproduce the cached walk-forward edges, the engine's position path and settled net bp equal the backtest rule on 5 sessions x 3 products, the cap, the 10:00 start, loss halt, kill, stale/missing bars, restart, settlement arithmetic, and a whole live day on a fake clock (waits, polls, decides, logs MOC intents before 15:50, settles after 16:20, resumes from mid-session, refuses stale history) that reproduces the replay.

**Replay of the last 15 sessions** (2026-10-05..23, `reports/replay_recent.log`, 2026 models trained on years before 2026 so out of sample): IEX-signal shadow P&L +$1,035 vs SIP-signal +$1,368 on $100,702 (40% deployed at full lots), daily P&L correlation 0.97, mean edge correlation 0.935 (minimum 0.61), 8.4 vs 8.0 lots per product-day. Fifteen days say nothing about Sharpe.

**The cost finding the paper test must settle.** On the last 3 replayed sessions the realised cost against the bar close, using the median consolidated quote 6 seconds after the bar end, was **2.69 bp per lot against the model's 1.06 bp (2.5x, which is the gate's stop level)**: half the quoted spread was 1.41 bp and the rest, about 1.3 bp, is price drift during the 6-second decision delay, five times the 0.25 bp allowance. SMH is the worst (3.35 bp vs 1.39) because its spread at 10:00 is several times the 10:30-15:30 sample behind the table. Re-running the backtest at the measured costs (policy re-thresholded): TSLA 1.18, SMH 0.65, NVDA 0.63, **basket 1.15 instead of 1.54**; at 1.5x the measured costs: 0.96 / 0.21 / 0.38, basket 0.75. So TSLA is the robust line, SMH the fragile one, and the 6-second latency is worth engineering down before real money. Three sessions is a small sample; the paper test measures it properly.

**What the output says.** Every decision row carries `action` (`BUY 3 lots = 105 sh TSLA @ 352.10`, `SELL ...`, empty when nothing is traded) and `position` (`LONG 2 lots = 70 sh TSLA`, `SHORT 1 lot = 35 sh NVDA`, `FLAT`), so a line is never just a signed integer; `replay` prints both next to the price and edge. The sign convention is the position's: positive edge = long the stock, negative = short.

**Not verified.** The live `run` has not been exercised against the real clock (the market is closed until Monday 2026-10-26 09:30 ET): only the simulated day above. How quickly real-time IEX publishes the bar that just ended is unknown (the loop polls up to 25 seconds, then carries the bar forward). The machine must stay awake from 09:25 to about 16:25 ET. There is no GUI and no broker reconciliation because nothing is sent.

## 10. Fill-price audit (does the backtest depend on where the order fills?)

The pairs project showed that a bar-based backtest can earn 8-12 bp a trade just by filling at the last print of the signal bar (the stale-print snap-back). The same question was put to this strategy: `scripts/ia_fillmodel.py` re-accounts the 2019-26 walk-forward positions with the fill price F = close + a x (next open - close), a = 0 (fill at the signal bar's close) .. 1 (fill at the next bar's open), same policy, same costs, same 10:00 first trade, MOC exit; a = 0 reproduces `ia_audit.net_j` exactly (asserted). Result (`reports/fillmodel.log`):

| Sharpe 2019-26 | a = 0 (close) | 0.5 | 1 (next open) | signal close -> next open, against us |
|---|---|---|---|---|
| TSLA | 1.35 | 1.35 | 1.36 | -0.03 bp (n = 18,071) |
| SMH | 1.10 | 1.09 | 1.09 | +0.00 bp (n = 12,689) |
| NVDA | 0.89 | 0.90 | 0.90 | -0.06 bp (n = 16,823) |
| basket (inverse-vol) | 1.54 | 1.54 | 1.55 | |

So the headline does not come from a favourable fill convention: for a single liquid stock the move from one bar's last print to the next bar's open is zero on average, and the spread of two assets, the thing that creates the artefact in the pairs, does not exist here. What the backtest does *not* capture is the latency between the bar's end and the order (section 9: about 1.3 bp of drift in 6 seconds on the last three replayed sessions, which is why the measured-cost basket is 1.15, not 1.54). That is a cost, not a fill-convention, problem, and it is what the paper test has to measure.
