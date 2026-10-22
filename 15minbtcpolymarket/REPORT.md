# BTC 15-minute Up/Down on Kalshi & Polymarket US — final research report (v2)

Date: 2026-10-07 · Read-only research, **no orders placed, no credentials used** · Code: this folder (`python test_btc15m.py`, 6 tests; app suite 65/65)

## 1. Decision: NO-GO for live capital

I looked for a strategy I would put money behind and did not find one. Every hypothesis that could be tested with public data was either
negative after costs, or failed an independent check. The one live lead (§3) is a transient, BTC-only anomaly that is already fading;
it is set up as a pre-registered forward test with a stop rule, not as a trading system.

| # | Hypothesis | Result (after fees, taker unless stated) |
|---|---|---|
| 1 | Buy the high-win-rate favourite (≥0.80/0.90, minutes 3–11) | BTC walk-forward **−0.2¢/contract, 92.9 % win** (226 trades); −3.6¢ with 0.5¢ slippage. Siblings (28 k trades): −0.5…−0.9¢, t ≈ −3.5 |
| 2 | Spot model (Coinbase BRTI proxy, vol, momentum) adds to the quote | No. Log-loss 0.483 vs market 0.470; z-term t = 0.4–2.0 |
| 3 | Sub-minute latency edge vs spot | No. Kalshi mid moves in the same second as Coinbase (corr peaks at lag 0) |
| 4 | Opening-minute mispricing | No. First prints average 0.503; gaps within noise |
| 5 | 15-min outcome mean-reversion | Decayed: −3.4 pts over 27 k markets, ≈ −1 pt in Aug–Oct, already priced at the open (t ≈ 1) |
| 6 | Time-of-day / weekend calibration | One BTC-only cell (22–23 UTC) — §3. Everything else flips sign between discovery and hold-out |
| 7 | Same rules on 8 thin sibling series (ETH, SOL, XRP, DOGE, BNB, HYPE, NEAR, ZEC; 46 k markets) | All negative: late fav −1.6¢ (t −4.9), late dog −1.9¢ (t −7.2) |
| 8 | Altcoin quotes lag BTC's quote (same venue) | No. BTC logit has coefficient −0.05 on siblings; trading disagreements −4.5¢ |
| 9 | Passive quoting (maker, no fee) on the trade tape | ETH (600 mkts, 12.9 k fills): **−1.3…−3.4¢**, t −9…−14, toxic fills. BTC (90 mkts, 2.2 k fills): **+0.2¢ / 0.0¢** (queue 1×), t ≈ 0 — break-even |
| 10 | Follow large aggressive trades 2–15 s later | ETH −2.2…−2.4¢ (t −5); BTC −1.6¢ (t −2.4) |
| 11 | Cross-venue Kalshi ↔ Polymarket US lock-in | **Untestable with public data**: Polymarket's public REST book and /bbo refresh every 30 s; real-time WebSocket needs API keys. Apparent arbs were frozen-book artefacts |

## 2. Why "high win rate" fails here

* The quote is a calibrated probability: mid vs realised frequency within ≈ ±1–2¢ in every bucket. Win rate is just the price.
* Crossing costs 2.25¢ at p = 0.5 (1.75¢ fee + 0.5¢ half-spread) and ≈ 0.4¢ at p = 0.95 — larger than the measured favourite premium (≈ +0.5…1¢, t ≈ 1.2).
* Alpha decays with liquidity (1.7 k → 2.4 M contracts per BTC window, Dec → Sep). Anything backtested before August is not representative.

## 3. The only live lead: BTC late-window underdog, UTC 22–23 ("R1") — transient, do not trade

Rule R1: window opens 22:00–23:59 UTC, minutes 12–13, buy the underdog at the ask (cost 5–30¢), hold to settlement.

| check | result |
|---|---|
| Since Aug 6 (n = 237) | **+13.5¢/contract, win 30.4 % at cost 16¢** (day-block bootstrap 95 % CI +8.3…+18.9¢) |
| Hold-out Sep 10→ | +17.9¢ (n = 103, t = 4.0) |
| **Trade-tape replay** (what same-side takers actually paid, 0–10 s delay) | paid 19¢ not 16¢; **EV still +10.4¢, t = 3.9** — quotes were real and fillable |
| Control (other hours, same rule, tape replay) | −1.8¢ (cost drag) |
| **Weekly profile** | wk Aug 3/10/17: −2/+5/+3¢ → Aug 24–Sep 27: +26/+16/+19/+22/+25¢ → Sep 28→: +7.6, +1.8¢ **(fading)** |
| **Out-of-time Jun 1–Aug 5** (n = 215) | **−1.4¢, win 14 %, t = −0.6 — the effect did not exist** |
| Cross-asset | **Absent** on all siblings (late-window slope in 20–23 UTC: BTC 0.81, others 1.02–1.15) |
| Mechanism | None found: no spot mean-reversion in those hours (β = −0.016, t = −0.7); implied/realised vol normal |

Reading: a ~5-week burst, BTC-only, unexplained, decaying — the profile of a participant's quirk that has since changed. Valid will not sign it off.
It is worth a **no-capital forward test** because the payoff if real is large.

### Forward test (pre-registered, `forward_test.py`)
Frozen 2026-10-07; evaluates only markets opened on/after 2026-10-08. SPRT H0: EV = 0 vs H1: +3¢ (σ 0.40/contract, α 0.05, β 0.20):
accept at LLR ≥ +2.77, reject at ≤ −1.56, or if paper P&L < −3¢/contract after 150 trades. At ≈ 3.8 qualifying trades/day: if the true EV is still ≈ +10¢ the test resolves in ≈ 45 days; if it has decayed to +3¢, ≈ 9 months — i.e. a "continue" verdict means the edge is not worth chasing.

## 4. Kelly sizing (applies only if R1 is accepted)

`f* = (p − c)/(1 − c)`, c = ask + fee. R1 inputs: tape price 0.191 + fee 0.011 ⇒ c ≈ 0.202; in-sample p = 0.304.
Shrink toward the market (50 %) and for regime decay ⇒ p ≈ 0.25 ⇒ full Kelly ≈ 6 %, half-Kelly ≈ 3 %. **Recommended: start at 0.5 % of bankroll per trade, hard cap 2 %,
one trade per window (≤ 4 per day), kill-switch at −15 % drawdown or an SPRT reject.** Win rate is ≈ 30 %, so expect long losing streaks (P(10 straight losses) ≈ 3 %).
For high-win-rate bets Kelly is the wrong tool without a verified edge: a 1¢ error in p moves f* by 13 % of bankroll at c = 0.925 (1.2 % at c = 0.15); with zero true edge even ¼-Kelly has a 21 % chance of a 30 % drawdown in 500 bets (`kelly_study.py`).

## 5. What would actually create an edge (needs things I cannot get from public data)

1. **Polymarket US API keys → real-time WebSocket**: only way to test cross-venue lock-ins (same BRTI, same price-to-beat, 99.996 % strike-chain agreement). Public data is 30 s stale.
2. **Second-level BRTI replication** (constituent exchange feeds) for the final-minute averaging window and for cancel-on-move maker logic — this is the HFT game; BTC passive quoting is already break-even naive.
3. **Maker economics** need live queue/fill data; Kalshi 15M series are `fee_type=quadratic` (taker-only; maker-fee types are named `quadratic_with_maker_fees`), Polymarket US pays a maker rebate (−0.0125·C·p(1−p)).

## 6. Market mechanics (verified)

| | Kalshi `KXBTC15M` (+11 sibling 15M series) | Polymarket US `btc-updown-15m-…` |
|---|---|---|
| Resolution | YES iff avg of 60 s BRTI before end ≥ avg of 60 s before start | identical rule/index/price-to-beat |
| Taker fee | `ceil_cent(0.07·C·p(1−p))` | `0.0695·C·p(1−p)` |
| Liquidity | BTC ≈ 2–3 M contracts/window, 5–10 k at touch; siblings 12–110 k | ≈ 15× thinner than Kalshi BTC |
| Data | live REST, 5 rps unauthenticated limit | public REST cached 30 s; history ≈ 13 days; label = bucket START |

## 7. Data & method

* Kalshi: 27.6 k BTC settled markets (Dec-25 →), 1-min bid/ask candles for ≈ 8.3 k (Jun-1 target hours + Jul-20 →), 8 sibling series × ≈ 5.7 k markets (Aug-6 →), trade tapes for 600 ETH / 220 SOL / 90 BTC markets (+ a partial XRP file, unused); Coinbase 1-min (426 k bars) as BRTI proxy (±1.1 bps).
* Discipline: discovery/hold-out split, one trade per market, window/market-clustered errors, slippage stress, walk-forward refits, out-of-time and cross-asset replication, tape replay for executability.
* Traps found: blank `expiration_value` on 2,520 early markets (use the strike chain); Kalshi 429 above ~5 rps and a 10 k-candle batch cap; Polymarket 30 s cache and +60 s timestamp convention (a look-ahead artefact in my first cross-venue result — caught and removed); liquidity grew ~1,500×.

## 8. Files
`collect*.py`, `tape.py`, `panel.py` (data) · `strategy.py` (calibrators, `decide()`, Kelly, walk-forward) · `run_bt*.py`, `analyze*.py`, `hypo*.py`, `sib_rules.py`, `leadlag_assets.py`, `mech.py`, `oot.py`, `stab.py` (research) ·
`maker_bt.py`, `maker_filter.py`, `flow_follow.py`, `verify_fills.py` (tape) · `forward_test.py` (pre-registered SPRT) · `kelly_study.py` · `live_logger.py`, `analyze_xv_hist.py` (cross-venue; Polymarket columns are 30 s-stale — do not use for claims) ·
`data/` (≈ 450 MB; `data/tape/` is ≈ 350 MB and can be deleted after reading this report).

## 9. Addendum (2026-10-09): can it be spread across the day, and can the win rate go up?
Tested with the GUI engine (Polymarket US fees, 2¢ slippage, quarter-Kelly, walk-forward probability, no look-ahead):
* **All-day does not work.** Underdog EV per contract is negative in most UTC hours (17–19 UTC t ≈ −3) and in every decision minute when pooled (t ≈ −10);
  the 85–97¢ favourite loses −1.7¢ pooled (≈ 89% win rate). All-day hour-adaptive Kelly: 6–10 trades/day, Sharpe 0.9–1.4, drawdown 34–47%, negative at ≥ 3¢ slippage.
* **A raw win-rate Kelly is unsafe across wide price ranges** (lost 99% all-day because cheap contracts borrowed the win rate of expensive ones). Fixed with a
  price-adjusted, skeptical probability (default).
* **Higher win rate within the good window works:** underdog 15–35¢ (R2) → 39% win rate, Sharpe 3.7, DD −12%, still 3.1 at 4¢ slippage (vs R1 0.7); positive in the
  post-burst weeks (+$3.6k on $10k, 32 trades). Caveat: band chosen after viewing this data; one regime.
* **More trades:** 20–23 UTC adaptive (R3) +60% more trades at lower Sharpe (2.9) and −20% DD; hours 20–21 lost money over the full sample.

### 9b. Addendum (2026-10-10): why no trades in Jul/Aug, and the Kelly sizing sweep
* **Flat Jul–Aug (R2):** signals existed (73 / 67) but the skeptical, price-adjusted probability gate needs a learned edge ≳ 3.8¢/contract (2¢ slippage + ~1.2¢ fee + 0.5¢ min edge). July's signals had a −4¢
  edge (20.5% win at 24.6¢) → gate shut (2 trades); betting all of them (fixed p = 30%) would have lost 56%. August turned +11¢ but an all-history learner opened only in September. A 45-day window starts ~1 week
  earlier and raises Sharpe on R1/R2/R3 (3.4→4.2, 3.9→4.6, 3.2→3.7); it is now the default.
* **Sizing sweep:** Sharpe is flat (4.1–4.7) across multipliers ≤ 0.5 and caps; sizing sets return vs drawdown. Quarter-Kelly / 2% cap: +239%, −12% DD, Monte-Carlo P(DD>30%) = 0.4%. 3% cap ≈ 2× return, P(DD>30%) 5.6%.
* **All-day (full data, 45d learner):** hour-adaptive underdog Sharpe 0.75 / −42% DD; underdog+favourite ≈ 0; favourite-only −3.4. Complete June–July history for hours 0–19 was backfilled for this.

### 9c. Addendum (2026-10-10): was 22-23 UTC / minute 12-13 a sensible choice?
Systematic scan (`explore_timeframe.py`, complete Jun 1 - Oct 10 data, underdog 15-35c, 2c slippage): 7 of 24 hours have EV > 0, none with t > 2 (3 have t < -2); 22-23 UTC is rank 1 of 24 two-hour windows (+6.65c, t 2.5);
windows of 3/4/6 hours dilute it (+3.9 / +3.3 / +2.2c). Decision minute inside 22-23: 12 = +8.1c (t 2.5), 13 = +6.0c, 11 = +1.9c, 14 = -9.2c, 2-10 = -2...-5c; other hours negative at all minutes. Permutation (hour labels
shifted within each day, 1,000 draws): P(best 2-hour window >= observed) = 1.3%. Stability: 22-23 was -3.5c (Jun-Aug 23), +23.6c (Aug 24-Sep 27), +10.5c (Sep 28-now, n = 39); neighbours 00-01 and 02-03 turned negative post-burst.
Forking paths (hours, minutes, band, lookback were all searched on this data) make 1.3% optimistic.

### 9d. Addendum (2026-10-10): full 10-month history and R1 vs R2
* Kalshi KXBTC15M launched 2025-12-15, so ~10 months is the maximum history (not 2 years). Evening hours 20-23 were backfilled to launch (early liquidity tiny: 1.7k contracts/market in Dec, 0.4M in May; median spread at the decision minute 5c in Dec, 1c Jan-Mar, 0.1-0.4c since June).
* **22-23 UTC underdog (15-35c) raw EV: Dec 15-May 31 -3.95c (n=339, t -1.7); Jun 1-Aug 23 -3.5c (171); Aug 24-now +19.8c (132, t +4.5).** The edge is a ~6-7 week regime; the earlier 8 months show none. Control hours 20-21 negative in most months.
* 10-month backtests (45d, 2c): R2 Sharpe 3.03 / +246% / -11%; R1 2.77 / +435% / -20%; R0 2.87 / +519% / -22%. 4c slippage: R2 2.90, R1 2.10. Dec-May alone: R2 29 trades -7%, R1/R0 none.
* R1 vs R2 (Jun 1-now): R1 higher return, R2 higher Sharpe in every window and more robust to slippage; with the raw win-rate probability R1 ranks higher (over-sizes cheap contracts).

### 9e. Addendum (2026-10-10): cross-validation, BTC-spot replica, better gate
* Scripts: `cv_family.py` (walk-forward + CSCV/PBO + deflated Sharpe on 2,400 rule configs), `cv_model.py` (quote + spot ML), `collect_spot_long.py`/`cv_longspot.py`/`cv_mechanism.py`/`cv_hours.py`/`cv_slots.py`/`cv_final.py` (2.7-year spot replica, mechanism, stress test, chart `cv_report.png`), `cv_gate.py`/`cv_gate2.py` (gate sensitivity).
* Rule selection has no out-of-sample value on the real quotes: walk-forward OOF EV -4.4c/contract (t -2.1), 1/8 months profitable; PBO 40%; IS +10.6c -> OOS -3.4c. R2 would not have been found by a CV procedure.
* Quote + BTC spot features add nothing (log-loss 0.45763 vs 0.45765 for the mid); all model-driven rules lose after costs.
* The 22-23 UTC effect is in BTC's price path, not Kalshi pricing: underdogs priced ~25c won 48.8% since Aug 24 (spot replica 50%); win rate = price before. 2.7 years of spot show no earlier episode (rolling-42d excess -12..+13c, now +22c). All hours show a milder shift (+4.0c vs +1.3c). EB-shrunk 22-23 estimate +7..+14c vs +21c observed; market repriced +3c since September.
* Replica caveat found and fixed: with a vol-only pricing function g, walk-forward "discovered" 13-14 UTC (+2.6c, t 4.0) - an artefact: the real market prices underdogs 2-4c above g at the US open (real EV -3.2c, t -1.3).
* Gate: min edge 0.5c -> 4c. 31 dead-regime months: 127 probing trades / -20% -> 0 trades / ~0%; live regime unchanged (10 months +247%, Sharpe 3.05, DD -11%).

### 9f. Addendum (2026-10-10, second pass): why it sits out, and a more future-proof design
* All-hours candle backfill (random 1/3 of Dec 15-May 31 days for hours 0-19; merged `data/candles.csv`). `scan_cells.py`: 422-467 hour x minute x price cells, discovery in Dec-Mar/May, validation later: 0 FDR discoveries, 0 replications, median cell -3.4..-4.2c (=costs), 11-15% of cells positive. The 8 months with no trades were genuinely edge-free; always-on R2 signals earned -2.2c (n=543).
* Siblings (ETH, SOL, XRP, DOGE, BNB, HYPE, NEAR, ZEC): no shift at 22-23 UTC since Aug 24 (win 17-27% vs price ~24%); BTC 48%. The effect is BTC-specific.
* Learner memory: exponential half-life beats a hard 45-day window (entry Aug 29 vs Sep 5, +363% vs +247%, same -11% DD; faster exit when a regime ends). Plateau 7-14 d with prior_n 20 and a 4c hurdle.
* Hour-agnostic all-day engines (hour shrinkage only) mostly lose in quiet stretches (-56..+18% with a hard window, median about -10%; -9..+3% with decay) and capture only +91..+157% of the regime (R2 hard-wired +351%). Core hours with a lighter shrinkage (k=10) and a very skeptical scout elsewhere (k=100) recover most of R2 (+301%) and can follow a move of the hot hour (simulated +7..+39% vs -12..-14%).
* Drawdown throttle (1/2 at -8%, 1/4 at -15%): in simulated regime endings DD -26% -> -19%, 10th pct -29% -> -19%; no cost while the edge persists.
* New: `trader/radar.py` + Regime radar tab, decay/core-hours/throttle in `SignalStats`/`BacktestConfig`/`LiveConfig`/GUI, pre-flight radar and core-aware checks. Scripts: `scan_cells.py`, `explore_holistic.py`, `explore_memory.py`, `project_scenarios.py`, `collect_hours_part.py`.
* Residual risk: a regime ending still costs ~10% of bankroll in the median simulated case (p10 -19%).

### 9g. Addendum (2026-10-10): Backtest/Live parity
* Live sizing now uses the shared assumed slippage (was the 3c order limit); backtest applies the live engine's daily-loss pause, drawdown stop and minimum-history rule (opt-in in the library, passed by the GUI); `evaluate_trades` is the single metrics path for backtest and live sessions; `trader/params.py` + the "Parameters in use" tab list backtest vs live values. Parity is proven by `test_live_engine_and_backtest_produce_identical_trades`.
* GUI safety: wheel-protected inputs, warning + reset for settings above the validated limits, stale key-file path ignored, settings location overridable for tests (the registry was polluted by earlier smoke runs).

### 9h. Addendum (2026-10-10): Auto mode (adaptive regime detection + equity-scaled sizing)
* New `trader/adaptive.py`, used identically by `run_backtest` and `LiveEngine`: three learners (half-lives 7/14/30 d) mixed by decayed predictive accuracy; regime label HOT/BUILDING/FADING/COLD; evidence-aware Kelly (edge - 0.25 SE); equity ladder (0.5x/3% -> 0.15x/1% from $500k) x regime scale x throttle; liquidity cap (10% of the last completed minute's Kalshi candle volume, ceiling 200k; the live endpoint matches the collected candles exactly). Parity test: identical live and backtest trades with Auto; Auto with neutral settings == the fixed engine.
* Verdict (Valid): **not an alpha source.** Real history Jun 1-Oct 10: Auto default +212% / Sharpe 4.1 / DD -11.6% vs fixed 0.25x/2% +223% / 4.6 / -11.7%. 60-day futures (10 paths each): at the same starting size Auto keeps about half the upside when the regime persists (+504% vs +994%) but bleeds less when quiet (-12% vs -16%), stops ~9 days sooner after the regime ends (last trade day 24 vs 33) and loses less after it (-7% vs -11%); it does not help when the regime fades or the hot hour moves. A multi-memory mixture did not beat a single 14-day memory (real history Sharpe 3.8 vs 4.6); regime scaling cost return without raising Sharpe; each is a position on the same return/risk frontier.
* Bug found: the drawdown throttle scaled only the Kelly multiplier, so it was inert whenever the stake cap bound (1.0x/5%/200k: halted by the 25% stop in 80% of persistent futures, +204% real history, -26% DD). Now scales the cap too: +560%, DD -19.7%, never halted in persistent futures; default 0.25x/2% sizing: median DD -14% -> -12%.
* State of the edge (refreshed Oct 11 01:30 UTC; an earlier draft of this bullet said '+1.5c last week' because the Oct 10 evening session had not been collected yet): net EV after 2c slippage + fees per contract, R2 signals: last 7 d +6.0c (se 9.6), 14 d +6.7c (7.1), 28 d +12.6c (5.5), 56 d +16.0c (4.0); weekly raw edge since Aug 24 was +32, +18, +36, +21, +23, +18c, i.e. weaker than the first weeks but still positive. Radar: 23 UTC TRADE (14 d +12c), 22 UTC WATCH (14 d +1.6c). Forward test (R1, frozen Oct 7): n=15, EV -4.2c (se 10.7c), LLR -0.16: CONTINUE.
* Caveats: one regime; Polymarket US depth unmeasured (volume proxy from Kalshi); volume exists only from June 2026; 10 paths per future. Scripts: `scen_auto.py`, `scen_auto2.py`, `ablate_auto.py`, `final_real_history.py`.

### 9i. Addendum (2026-10-12): first real LIVE trades, stale live quote, trade ledger
* The user's first LIVE session (Oct 11 evening, 1.0x/5%): 4 NO trades, all lost, $500 -> $449.48. Quoted NO 0.18/0.15/0.20/0.33 vs fills 0.08/0.11/0.08/0.24; Kalshi candles show the underdog price collapsing within the minute. Cause found: the live signal quote came from `GET /markets/{ticker}`, up to ~12 s behind the order book (seen: book 0.80/0.81 vs object 0.76/0.77; at minute 13 the book equalled the candle close, the object was 5c off). The live engine therefore took trades the backtest rules would not (3 of 4 outside the 15-35c band at the true current price). Fixed: quotes now come from the order book. n=4: no statistical claim, but live results before this fix are not evidence about the strategy.
* New: persistent trade ledger (`data/live/ledger.jsonl`), account trade cross-check (docs-only), `collect.py gapfill` so the newest evening is in the backtest data.

### 9j. Addendum (2026-10-12): following a moved hot hour
* Lowering the scout shrinkage everywhere speeds up hour discovery but loses real money in quiet hours (Jun-Oct: scout k 100/60/40/30 = +217/+182/+145/+80%). Adopted instead: near-core scouting (hours within 2 of the core use k=50, others 100): moved-by-2h futures first trade day ~10 instead of 25, median return +187% vs +93%; real history +202% vs +217%, DD -16% vs -12%. Far moves stay slow; a hot hour is only traded while the app runs in that hour. Radar now uses the engine's own shrinkage (it scouted with k=30 before). Scripts `scen_hours.py`, `replay_hours.py`.
