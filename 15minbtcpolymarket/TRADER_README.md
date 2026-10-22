# BTC 15-min Up/Down trader (GUI)

```
cd C:\Users\xinyu\PycharmProjects\pythonProject\15minbtcpolymarket
python trader_app.py          # or double-click run_trader.bat
```

## What it does
* **Live trading tab** – connects to your **Polymarket US** account with the key file (`Key ID: …` / `Secret Key: …`), takes an
  initial capital, and trades the selected strategy automatically. Each bet is sized with **Kelly**: `f* = (p − c)/(1 − c)`, where `p`
  is the strategy's **historical win probability** and `c` is the all-in cost per contract (price + slippage + fee). You then bet
  `Kelly multiplier × f*`, capped at "Max stake / trade". If `p ≤ c` the bet size is zero (no trade).
* **Backtest tab** – any date range, capital and settings: every trade, equity curve + drawdown, metrics (return, CAGR, max drawdown
  $/%/duration, Sharpe, Sortino, Calmar, win rate, profit factor, expectancy, max/avg consecutive wins and losses, stake sizing,
  bootstrap CI), a monthly table and a **By hour and leg** table that shows where the profit comes from. CSV export.
* The **left panel** (preset, rules, fees, slippage, Kelly settings) is shared, so a backtest runs exactly what live trades.

## Strategy presets (backtest Jun 1 – Oct 10 2026, 2¢ slippage, Polymarket US fees, quarter-Kelly, 2% cap, 45-day learner)
| Preset | Trades/day | Win rate | Return | Max DD | Sharpe | Sharpe @ 3¢ slip |
|---|---|---|---|---|---|---|
| **R2 · Higher win rate, 22–23 UTC, underdog 15–35¢ (default)** | 1.1 | **40%** | +239% | −12% | **4.6** | 4.4 |
| R3 · More trades, 20–23 UTC, hour-adaptive | 2.2 | 33% | +191% | −20% | 3.7 | 3.4 |
| R1 · Original, 22–23 UTC, underdog 5–30¢ | 1.2 | 34% | +435% | −20% | 4.2 | 3.7 |
| All-day underdog, hour-adaptive *(experimental)* | 6.4 | 26% | +9% | −42% | 0.75 | 0.9 |
| All-day underdog + favourite *(experimental)* | 10.3 | 51% | −22% | −55% | −0.1 | 0.4 |
| Favourite only, 85–97¢, minutes 5–11 *(experimental)* | 10.0 | 90% | −41% | −43% | −3.4 | −2.8 |

**Why not all day?** Outside the evening hours the underdog and favourite rules have no edge after fees and slippage; letting Kelly spread across all
hours just adds losing trades and ~40% drawdowns. **Why not a high-win-rate favourite strategy?** A ~90% win rate is simply the price; after
fees/slippage it earns −1 to −2¢ per contract. R2 raises the win rate (17% → 40%) by buying mid-priced underdogs, which is also far less
sensitive to slippage. All numbers are in-sample.

## R1 vs R2 — why R2 is the default (`python compare_r1_r2.py`)
Jun 1 – Oct 10, 2¢ slippage, default sizing, 45-day learner (Sharpe / return / max drawdown):
| | R1 (underdog 5–30¢) | R2 (15–35¢) | R0 (5–35¢) |
|---|---|---|---|
| Full period, price-adjusted p | 4.22 / +435% / −20% | **4.55** / +239% / **−12%** | 4.35 / +510% / −22% |
| Since Jul 20 | 5.39 / +435% / −20% | **6.35** / +277% / −11% | 5.58 / +514% / −22% |
| Since Sep 28 (post-burst) | 3.39 / +9% / −17% | **9.41** / +39% / −11% | 5.32 / +16% / −15% |
| Sharpe at 0¢ → 5¢ slippage | 5.3 → 2.7 | 4.9 → **4.1** | 5.4 → 2.9 |
| Win rate / avg price paid | 34% / 18.7¢ | **40%** / 27.2¢ | 33% / 20.3¢ |
R1 earns more *return* because it also buys cheaper contracts with bigger payoffs (its top 5 trades are 48% of its profit; best +$4.4k, worst −$1.1k), but that edge is the first to fade (since Sep 28 it made only +9%)
and it needs near-perfect fills (Polymarket US is ~15× thinner than Kalshi, so 3–4¢ slippage is realistic, where R2 is clearly better). With the **raw win-rate** probability instead, R1 *does* beat R2 on Sharpe (2.9–3.5 vs 1.9–3.2),
because that model over-sizes cheap contracts, which favours whichever preset buys more of them. Differences between the two are small relative to the noise (~150 trades, one regime); R0 is a middle option.

## How far back can I backtest?
Kalshi's BTC 15-minute series **launched on 2025-12-15** (nothing exists before it), so ~10 months is the maximum — not 2 years. Liquidity in the first months was tiny
(1.7k contracts per market in Dec, 0.4M in May vs 2.8M now), so quotes there are less reliable. The app only uses days that are downloaded: Jun 1 → now is complete for all hours; use
**Download older history** (left panel) to fetch the hours you have selected back to Dec 15, 2025. Polymarket US has only ~2 weeks of price history and its public data is cached, so it cannot extend the test;
the strategy's edge is a market-pricing effect, so it cannot be reconstructed from Bitcoin price history alone.

## Full-history result (Dec 15 2025 → Oct 10 2026; `python explore_longhistory.py`)
Raw edge of the 22–23 UTC underdog (15–35¢, 2¢ slippage + fees) by period: **Dec 15–May 31 (339 signals, thin liquidity): −3.95¢ (t −1.7); Jun 1–Aug 23 (171): −3.5¢; Aug 24–now (132): +19.8¢ (t +4.5).**
The edge exists only in the last ~6–7 weeks; control hours 20–21 were also negative. Full 10-month backtests (45-day learner, 2¢ slippage): R2 Sharpe 3.0 / +246% / −11% DD; R1 2.8 / +435% / −20%; R0 2.9 / +519% / −22%;
at 4¢ slippage R2 2.9 vs R1 2.1. In the 5.5 months before June the adaptive gate mostly stayed out (R2: 29 trades, −7%; R1 and R0: none) — that is the protection working, and ~7–11% of bankroll is the realistic cost of a "dead" regime
before it shuts off. Treat everything as **one regime**: keep stakes small and keep the forward test running.

## Backtest vs Live: one engine, one set of metrics, one parameter table (2026-10-10)
* **Shared settings (left panel)** drive Backtest, paper and LIVE alike: strategy rules, probability model and memory, Kelly sizing, drawdown throttle, fee model, **assumed slippage** and the **risk limits (daily loss pause, drawdown stop)**.
  *Backtest-only* (its own box): date range, initial capital, probability mode (history / fixed / hindsight), compounding. *Live-only* (its own box): session capital, mode (paper / LIVE), **order limit price** (ask + X¢, caps what a real order can pay), probability source (history / manual).
* **Fixed inconsistencies:** live bets used to be sized with the order limit (3¢) while backtest/paper used the assumed slippage (2¢) - now all three size with the assumed slippage and the order limit only caps the order. The daily-loss pause, drawdown stop and the 30-signal minimum history existed only in the live engine - `run_backtest` now applies the same rules window by window (`BacktestConfig.daily_loss_limit / max_drawdown_stop / min_history`; the GUI passes the shared values, library defaults stay off).
  Editing sizing / cost / risk fields no longer flips the preset label to "Custom" (only preset-defining fields do).
* **Same metrics:** both tabs use `PerformanceView` (same 8 KPI cards, same Metrics table, equity curve, trades, monthly, by-hour) fed through `metrics.evaluate_trades`; the Live tab feeds it the session's settled trades via `live.trades_frame` (same columns as backtest trades). The live-only operational cards (next decision, win probability, Kelly @ 16¢, open positions) sit in a separate strip.
* **Which parameters does each side use?** Tab **"Parameters in use"** (both tabs) lists every parameter, Backtest value next to Live value (`trader/params.py`): white = shared, yellow = differs on purpose, grey = one side only; the Live column shows the *running* session's frozen config. A one-line "this backtest / this session uses ..." summary sits under each control box.
* **Proof:** `test_live_engine_and_backtest_produce_identical_trades` replays the same data through `LiveEngine` and `run_backtest` (3 settings incl. risk limits and throttle): identical contracts, entry price, P&L, equity and metrics. `test_gui_smoke.py` asserts the shared values are identical in both configs.
* **Charts and pages (2026-10-10):** the equity chart (Backtest and Live) zooms with the mouse wheel (Shift+wheel = vertical only), pans by dragging, resets on double-click or the toolbar buttons, and shows full trade details when you hover a dot (click to pin). Every page is wrapped in a scroll area (`PageScroll`) so a small window scrolls instead of cutting text off; the settings panel sits behind a draggable splitter; tables never elide text and show the full cell on hover.
* **Safety fixes found on the way:** spin boxes / drop-downs ignore the mouse wheel unless focused (scrolling the panel used to change Kelly, stake cap, ... silently); the app warns (and offers one-click reset) when saved sizing is above the validated limits; a saved key-file path that no longer exists is ignored; `TRADER_SETTINGS_FILE` lets tests use a throw-away .ini - `QSettings(org, app)` ignores `setDefaultFormat/setPath`, so earlier smoke runs wrote into the real registry key `HKCU\Software\AlphaSyndicate\BTC15mTrader`.
* Note on numbers: the research tables elsewhere in this README were produced **without** the live risk limits (library defaults); the GUI backtest now applies them (10% daily / 25% drawdown by default), so GUI figures can differ slightly.

## Reconsidering the strategy: why it sat out for 8 months, and what changed (2026-10-10, second pass)
**Why it sat out - it was correct.** `scan_cells.py` tests every hour × decision-minute × price-bucket cell (both sides) on the real quotes: discovery Dec 15–Mar 31 → validation Apr 1–Aug 23 gives 422 cells, **0 FDR discoveries, 0 replications**, median cell **−4.2¢ / −3.4¢** (= trading costs) and only 11–15% of cells positive; discovery Dec 15–May 31 → validation Jun–Aug23: 467 cells, 0 and 0. The 2,400-rule walk-forward (−4.4¢), the quote+spot ML (no skill) and the 8 sibling crypto series (no shift since Aug 24; only BTC) say the same. Always-on, R2's own signals earned −2.2¢/contract before Sep 5 (543 signals). So there was nothing to trade; a strategy that stands aside at ~0 cost is doing its job. What *can* be improved is how fast it recognises a regime start/end, whether it can follow an edge that moves, and how much a regime ending costs.

| Change | Evidence (real engine, real quotes unless noted) |
|---|---|
| **Decaying memory** (half-life days instead of a hard 45-day window) | Dec 15→now R2: first live trade **Aug 29 vs Sep 5**, +363% vs +247%, same −11% DD, 0 quiet-month trades (`explore_memory.py`, plateau for half-lives 7–14 d). Regime ending (60-d bootstrap): −17%/DD −27% → −10%/−14% |
| **R2+ default: core 22–23 + skeptical all-hours scout** (by-hour shrinkage k=100, core k=10, half-life 14 d) | Jun 1→now +301%, Sharpe 5.1, DD −12.9% vs hard-wired +351% / DD −10.9%; scout trades in other hours were a wash (11 trades). Hot hour moves to 01–02 UTC (simulated): hard-wired R2 −12%…−14%, R2+ **+7…+39%**. A fully hour-agnostic engine gets only +91…+157% (must discover the hour; deeper DD) |
| **Drawdown throttle** (½ size at −8%, ¼ at −15%) | Regime ends at day 20: DD −26% → −19%, return +56% → +70%; quiet-future 10th pct −29% → −19%; persists +929% → +927% |
| **Regime radar** tab + pre-flight check | scouts all 24 hours with the shrunk estimator, warns when the active hours are cold and another is hot |

60-day forward projection (`project_scenarios.py`, 12 bootstrap draws per future from 47 regime days / 168 quiet days, learner warm-started on today's history; median return / median max DD):

| Future | R2, 45 d window (old) | R2, 10 d decay | R2+ no throttle | **R2+ (default)** |
|---|---|---|---|---|
| Regime persists | +967% / −13% | +967% / −13% | +929% / −13% | **+927% / −13%** |
| Fades (half the days quiet) | +186% / −20% | +104% / −17% | +138% / −18% | **+110% / −15%** |
| Ends at day 20 | +49% / −36% | +73% / −17% | +56% / −26% | **+70% / −19%** |
| Quiet (no regime) | −17% / −27% | −10% / −14% | −10% / −20% | **−11% / −16%** |
| Hot hour moves (22–23 → 01–02 UTC) | −20% / −27% | −14% / −16% | +39% / −27% | **+7% / −18%** |

**Limits, stated plainly:** one regime in 10 months; half-life, shrinkage k and the throttle thresholds were chosen from plateaus after seeing this data; the projection resamples observed days (it cannot invent a new regime); 12 draws per cell is noisy (e.g. the "fades" ordering); hour-level edges are statistically thin (~1 signal/hour/day), which is why the scout is skeptical. If the regime ends, expect to lose roughly 10% of bankroll before the engine stands down — not zero.

## Following a hot hour that moves (2026-10-12)
* **Limit:** one hour produces ~1.4 signals/day, so the evidence for a new hot hour accumulates slowly whatever the method. The only lever is how much prior weight (shrinkage k) the hour's own evidence must overcome, and a lighter k everywhere costs real money in quiet hours: Jun 1 → Oct 10 real history, scout k = 100 / 80 / 60 / 40 / 30 → +217% / +216% / +182% / +145% / +80% (Sharpe 4.44 / 4.55 / 4.39 / 3.90 / 2.88; trades outside 22–23: 11 / 18 / 36 / 62 / 95; their P&L +8% / +13% / −1% / −9% / −21%). No scout k beats 80–100 on the path that actually happened.
* **Chosen default (R2+ and AUTO): near-core scouting.** Hours within 2 of a core hour (20, 21, 0, 1 for core 22–23) use k = 50, everything farther k = 100 (`SignalStats(near_k, near_width)`, `BacktestConfig/LiveConfig.near_k/near_width`, GUI fields *Near-core shrinkage k / width*; 0 = off). 60-day bootstrap futures, 24 paths (`scen_hours.py`), R2+ now → near-core:
| Future | median return, k=100 → near k=50 | first trade in the moved hot hours | P&L from the new hours |
|---|---|---|---|
| hot hour moves 2 h earlier (20–21 UTC) | +93% → **+187%** | day 25.5 → **10** | +30% → +114% |
| moves 2 h later (00–01 UTC) | −10% → **+12%** | day 23 → 17.5 (never found: 12.5% → 0%) | +1% → +22% |
| moves 6 h earlier (16–17 UTC) | +12% → +11% | ~16 days either way (12.5% never found) | unchanged |
| regime persists / ends / quiet | +354 → +383 / +65 → +63 / −9.9 → −9.9 | – | – |
  Real Jun–Oct history (the hot hour did not move): **+202% / Sharpe 4.35 / DD −16%** vs +217% / 4.44 / −12% (32 vs 11 trades outside 22–23, +10% vs +8% P&L); regime-start timing unchanged (first regime trade Aug 29). k = 40 is faster (shift −2: +212%, first trade day 9) and costs more (+190% real); k = 60 is cheaper (+211% real) and slower. AUTO with near-core: +215% vs +209% real; simulated −2 h move +90% vs +51%, found in 100% vs 81% of paths.
  Far moves (≥ 3 h) stay slow by design; a hot hour is only traded while the app is running during that hour (run it around the clock to act on a move; the learner sees every hour's history either way).
* **Radar fix.** The Regime radar scouted with k = 30 while the engine used k = 100 (and 10 for the core), so a radar *TRADE* did not mean the engine would trade that hour. The radar now takes the engine's own settings (hour k, core, near-core, half-life) from the left panel / pre-flight config: *TRADE* = the engine's gate is open for a typical signal at that hour (`test_near_core_hours_..._radar_mirrors_the_engine`).

## Trade ledger and fresh data (2026-10-12)
* **Live history survives restarts.** `trader/ledger.py`: every position is appended to `data/live/ledger.jsonl` when it fills (`open` event) and when it settles (`settle` event); the Live tab shows it through **Trades shown** (this session / all recorded trades of this mode / both modes) with the same KPIs, metrics, equity curve, trades table (extra *Session* column) and chart tooltips as everything else. Positions left open by an earlier session are settled from Kalshi's result at the next start, on *Connect and test* and on *Reload history* (`Ledger.reconcile`, flagged "settled after a restart"). Old `session_*.csv` files are merged read-only (the ledger wins on duplicates), so trades from before the ledger existed appear too — that is how the four LIVE trades of the Oct 11 evening session were recovered.
  Each LIVE trade also stores the exchange's raw `avgPx` and the account's buying power right after settlement (audit trail), and *Connect and test* now lists open orders / positions and cross-checks the account's recent trades (`GET /v1/portfolio/activities`, read-only, built from the docs and tested against a mock only) against the ledger, warning about BTC-15m trades the ledger does not explain. Pooled equity = first recorded bankroll + cumulative P&L (no deposit/withdrawal knowledge).
* **The backtest always includes the newest data.** Found: the bulk candle download (`collect.py batch`) trails the newest windows by up to ~2 h, so the latest evening — exactly the 22–23 UTC core — was missing from the local data for hours (the markets list had it, the candles did not). Fixes: `collect.py gapfill` fetches per-market candles for recently settled markets (only stored once minute ≥ 13 exists; ~seconds after the close); `collect.py markets` is now incremental and merges instead of overwriting (3 s instead of ~40 s, no truncation if the network drops); the GUI refreshes in the background at start and before every backtest when the data is > 20 min behind (`data.needs_refresh`, at most one attempt per 5 min, checkbox *Refresh data first*); the **To** date follows the newest data until you pick one yourself (**→ latest** re-attaches it); a line shows the data's age; the run summary prints the newest data time.
* **Live quote source fixed (found through the four real LIVE trades).** The live engine read its signal quote from the market object (`GET /markets/{ticker}`), whose bid/ask update in steps and were seen **up to ~12 s behind the order book** (small sample: 9 comparisons; at minute 13 the book matched the candle close exactly, the object was 5¢ off) (e.g. book 0.80/0.81 while the object said 0.76/0.77; the book moved ~1¢/s). The 4 real trades of the Oct 11 evening were taken at quoted NO prices 0.18 / 0.15 / 0.20 / 0.33 but filled at 0.08 / 0.11 / 0.08 / 0.24, and the candle data shows the underdog's price collapsing inside that minute (for example NO 0.35 → 0.08 between minute 12 and 13): the engine bought underdogs that were already being repriced down, and 3 of 4 would not have passed the 15–35¢ band at the real current price (the backtest, which uses the candle-close quote, took only 2 trades that evening). `KalshiQuotes.quote` now reads the **order book** (best YES bid; YES ask = 1 − best NO bid) and falls back to the market object only if the book is unavailable (`test_live_quote_comes_from_the_order_book_...`). `quote_lag_probe.py` measures the old lag against candle closes. All four trades lost; four trades prove nothing statistically, but the mechanism (stale cheap underdog = adverse selection) is real and would have lowered the live edge below the backtest's.
* **Tests no longer touch your real files.** Sessions, ledger and pre-flight reports go to `TRADER_STATE_DIR` in tests (the GUI smoke test asserts the real `data/live` folder is unchanged). Earlier versions of that test deleted `session_*_paper.csv` and `preflight_*.txt` from the real `data/live` folder as "cleanup" — any such files from before this fix are gone (LIVE session files were never touched).

## Auto mode — regime detection + equity-scaled sizing (2026-10-10, third pass)
Tick **Auto** in the left panel (or choose the preset **AUTO · Adaptive regime + equity-scaled sizing**). At *every trade* — in the backtest, in paper and in LIVE, same code (`trader/adaptive.py`) — the engine sets, from the history it has at that moment:
1. **Memory / regime detection:** three learners of the same edge (half-lives 7 / 14 / 30 days), averaged with weights from how well each predicted the signals that settled recently (floor 5%). Stable regime → near-equal weights; right after a change the fast learner takes over (84 / 11 / 5% twenty days in). No parameter is chosen by past P&L.
2. **Regime label:** HOT (fast and slow edge both clear costs + hurdle) · BUILDING (only the fast one: new regime, thin evidence) · FADING (the slow one does, the fast one dropped) · COLD (neither → no trade). Size scale 1.0 / 0.5 / 0.4 / 0.
3. **Evidence-aware Kelly:** p = price + edge − z·SE (z = 0.25).
4. **Equity ladder** → Kelly multiplier and stake cap from the account's *current equity* (default 0.5×/3% from $0, 0.35×/2.5% from $25k, 0.25×/2% from $100k, 0.15×/1% from $500k; editable text, e.g. `0:1.0/5, 25000:0.5/3, 100000:0.35/2, 500000:0.2/1`), × regime scale × drawdown throttle (which now scales the multiplier **and** the cap).
5. **Liquidity cap:** max contracts = 10% of the contracts traded in the last completed minute (Kalshi candle volume; ~170k median at 22–23 UTC; the live endpoint returns exactly the numbers the backtest uses), ceiling 200,000. A failed live read falls back to the median of recent readings or a small default — never to unlimited.

With Auto on, Kelly multiplier / Max stake / Max contracts / lookback / half-life are greyed out; min edge, probability basis and hour model, costs, throttle and risk limits still apply. The Live tab shows an **Auto status** strip (regime, effective multiplier and cap, learner weights, hours with an edge now); trades record `regime`, `mult_used`, `cap_used` (trades table, chart tooltip, CSV). Pre-flight reports a "Learner gate (Auto)" row and checks the ladder's first step.

**Results (honest).** Real history Jun 1 → Oct 10, GUI safety rules, 2¢ slippage (`final_real_history.py`):
| | Return | Sharpe | Max DD | Max contracts |
|---|---|---|---|---|
| fixed 0.25×/2%/2k (R2+ default) | +223% | 4.6 | −11.7% | 2,000 |
| fixed 0.5×/3%/20k | +352% | 4.4 | −16.1% | 7,857 |
| **AUTO default** (ladder starts 0.5×/3%) | **+212%** | **4.1** | **−11.6%** | 4,258 |
| AUTO, aggressive ladder (starts 1.0×/5%) | +337% | 3.8 | −16.8% | 7,323 |
| fixed 1.0×/5%/200k (your sizing), new throttle | +560% | 3.8 | −19.7% | 19,768 |
| same, old throttle (scaled only the multiplier) | +204% | 3.0 | −26.4% (halted at the 25% stop after 60 trades) | 9,351 |

60-day futures (`scen_auto.py`, 10 bootstrap paths per future; median return / median max DD; the learner is warm-started on today's history):
| Future | fixed 0.25×/2% | fixed 0.5×/3% | **AUTO default** | fixed 1.0×/5%/200k | AUTO aggressive |
|---|---|---|---|---|---|
| Persists | +421% / −12% | +994% / −17% | +504% / −15% | +2,654% / −21% | +834% / −17% |
| Fades | +63% / −15% | +75% / −18% | +14% / −16% | +107% / −22% | −7% / −21% |
| Ends (day 20) | +56% / −15% | +90% / −19% | +61% / −16% | +116% / −25% | +95% / −19% |
| Quiet | −14% / −16% | −16% / −19% | −12% / −13% | −16% / −24% | −14% / −16% |
| Hot hour moves to 19–20 UTC | +12% / −18% | +12% / −21% | +14% / −15% | +4% / −25% | +17% / −20% |
| Paths halted by the 25% stop | 0% | 0–10% | 0% | 0–50% | 0–10% |

*Reading it:* Auto is a **different risk profile, not more alpha.** At the same starting size it keeps about half the upside when the regime persists, but bleeds less when there is none (−12% vs −16%), stops about 9 days sooner after a regime ends (last trade on day 24 vs 33 after the end; post-end loss −7% vs −11%), shrinks positions as the account grows and cannot size beyond the market's liquidity. It is no better when the regime fades or the hot hour moves (one hot hour among 24 is statistically invisible for weeks). The internals are dials on one frontier: a single 14-day memory earns more when the regime persists (+744%) and bleeds more when quiet (−16%); regime scaling off: +669% / −12.8%; z = 0: +585% / −14%; z = 0.5: +360% / −10% (`scen_auto2.py`, `ablate_auto.py`). Defaults sit at the conservative end because the regime has weakened since its first weeks (net edge after costs per contract: last 56 days +16¢, 28 days +13¢, 14 days +7¢, 7 days +6¢, against about +30¢ in the first weeks; data refreshed Oct 11 01:30 UTC).
*Caveats:* one regime; 10 paths per future (neighbouring cells within 10–20 points are noise); the fixed-sizing columns assume 2¢ slippage at any size (optimistic above a few thousand contracts); Polymarket US depth is unmeasured — the liquidity cap uses Kalshi volume as its proxy; volume exists only from June 2026 (earlier days: ceiling only).

**Two fixes found on the way.** (1) The drawdown throttle used to scale only the Kelly multiplier, so it did nothing whenever the stake cap was the binding limit (your 1.0×/5%/200k sizing hit the 25% stop in 80% of simulated persistent regimes and halted after ~50 trades; with the fix, 0% and ~10× the median return); it now scales the cap too (`strategy.throttled_sizing`, one code path for backtest and live). (2) A finished session no longer feeds the Live preview.
Tests: `test_auto_*` in `test_trader.py` (ladder parsing, regime labels, decide() incl. throttle/liquidity, **live = backtest parity with Auto**, Auto with neutral settings reduces exactly to the fixed engine, volume fallback, Kalshi volume endpoint), Auto flows in `test_gui_smoke.py`.

## Pre-flight check (go / no-go) — run before every session
**Start trading** now runs `trader/preflight.py` first (also the **Pre-flight check** button, the CLI `python preflight_cli.py [--live]`, and an automatic run at **21:45 UTC** while the app is open). It never sends an order.

| Area | Checks | Fails / warns when |
|---|---|---|
| Data | auto-updates from Kalshi if stale (≤1 update per 20 min), candle coverage of the last 28 days | history >36 h old (FAIL), >4 h (WARN), coverage <97% (WARN) |
| Forward test | detects new settled windows since the last check, re-evaluates the pre-registered R1 SPRT (same as `forward_test.py`; a boundary crossing is terminal) | REJECT → FAIL; CONTINUE → WARN + suggested (optional) 0.5% cap; ACCEPT → PASS |
| Edge now | net EV/contract of your preset over 28 / 14 days; learner gate open? | 28 d <0 → FAIL; below the 4¢ hurdle or 14 d <0 → WARN + cap; gate closed → WARN |
| Feed | Kalshi exchange status, live quote, PC clock vs server | exchange down / no quote / clock >20 s → FAIL |
| Risk | sizing vs validated limits (0.5× Kelly, 2% cap), tested preset, previous session ended by kill/halt | cap >5% or Kelly >1× → FAIL; above validated → WARN |
| Account (LIVE) | key file, buying power vs capital, clock skew, open orders / positions, exchange order validator (preview only) | auth error / buying power <50% of capital → FAIL |

Verdict **GO / CAUTION / NO-GO** is always about going live; paper is never blocked. In CAUTION the primary button is **"Continue to LIVE with my settings"** (your own sizing is used unchanged; a suggested 0.5% cap is only an optional secondary button while the forward test is unconfirmed); NO-GO disables the live button and offers "Run as paper instead". Reports are saved to `data/live/preflight_*.txt`; bookkeeping (last forward-test state, last session end) in `data/preflight_state.json`.
Tests: `python test_preflight.py` (unit + mock exchange), `python test_gui_smoke.py` (dialog + Start flow).

## Cross-validation and the BTC-spot replica (2026-10-10)
**Data:** Coinbase BTC-USD 1-minute candles Jan 2024 → now (`collect_spot_long.py`, 99.9% complete). Rebuilding each window from spot (strike/final = OHLC4 of the first/last minute) matches Kalshi's real result 96.9% of the time (94.5% near the strike); monthly R2 P&L from the replica correlates +0.74 with the real one.
**Replica pricing:** `g(z)` = isotonic fit of the real Kalshi mid on the spot z-score, fitted only on quotes *before* Aug 24 2026. Good for 22–23 UTC (real quotes within ~1¢ of g), **not** for the US-open hours 12–14 UTC (real market prices underdogs 2–4¢ above g), so replica P&L is only trusted where it was checked against real quotes.

| Test (scripts `cv_*.py`) | Result |
|---|---|
| Walk-forward CV of the *selection procedure* (2,400 configs = hours × minutes × band; train t-stat → trade next month) | **−4.4¢/contract, t −2.1, 1 of 8 months profitable** (random-config null median −2.6¢). Rolling 2-month window −1.9¢. R2 fixed over the same months: +3.1¢, t +1.5 |
| CSCV / PBO, 16 blocks | PBO 40% (34% among 22–23 configs); IS-best +10.6¢ → OOS −3.4¢, slope ≈ 0 |
| Deflated Sharpe, R2 | all 10 months ≈ 0; since Aug 24 only 0.99 (but the window was picked after the fact) |
| Change-point | ~Aug 21: −4.1¢ (n=503) → +19.6¢ (n=139), Welch t +5.1; all 7 weeks since Aug 24 positive, longest earlier positive streak 2 weeks |
| Models on quote + BTC spot features (LR, gradient boosting, ± hour), walk-forward | log-loss = market mid (0.45763 vs 0.45765) or worse; trading them −2.5¢ (t −3.5) → spot adds nothing to the quote |
| Where does the edge come from? | Underdogs the market priced ~25¢ won **48.8%** since Aug 24 (same 50% from spot alone); before: win rate = price. The quote was calibrated all along — **BTC's behaviour changed**, not Kalshi's pricing. A milder version appears in all hours (+4.0¢ vs +1.3¢ baseline) |
| 22–23 UTC underdog excess win-rate vs pricing, rolling 42 days, 2.7 years | wandered −12…+13¢; now +22¢, never seen before. Jan 2024–Dec 2025: −1.9¢ (n=1,399); Dec–Aug: +0.1¢; since Aug 24: +21.2¢ (n=100, se 4.9¢) |
| Hour concentration | all-hours +5.6¢ since Aug 24; empirical-Bayes shrinkage of the 22–23 figure → **+7…+14¢** (not +21¢); market has repriced underdogs +3¢ in 22–23 since September |
| Gate stress test (real engine, 31 months with no regime: 23 reconstructed + 8 real) | old gate (min edge 0.5¢): 127 probing trades, **−20%**, max DD −23%. min edge 4¢: 0 trades, ~0%; same live-regime capture (10 months +247% vs +200%, DD −11% vs −23%, ~1 week later start). Plateau 3–5¢ for R0/R1/R2 and at 4¢ slippage → **new default 4¢** |

**Reading:** the edge is real in BTC's own price path, but there is exactly one episode of it in 2.7 years and no precedent for how long such episodes last. Expect a fraction of the backtest return even if it persists (replica: +84% vs +246% real), and nothing if it ends.

## Why 22–23 UTC and minutes 12–13? (`python explore_timeframe.py`)
Scan on Jun 1 – Oct 10 (underdog 15–35¢, 2¢ slippage, Polymarket US fees, gate-independent EV per signal): 17 of 24 hours are ≤ 0; none is significant alone. **22–23 UTC is #1 of the 24
two-hour windows (+6.7¢, t = 2.5, 303 signals)**; 23–00 +2.8¢, 21–22 +2.4¢; wider windows dilute it (20–23 +2.8¢, 22–03 +2.2¢). Inside 22–23 the edge sits at **minute 12 (+8.1¢) and 13 (+6.0¢)**;
minute 11 ≈ 0, minute 14 −9¢, minutes 2–10 −2…−5¢; other hours are negative at every minute. A shuffle test puts the chance of a window this good at ~1.3% — but the hours, minutes and
price band were all searched on the same data, no mechanism explains them, 22–23 earned −3.8¢ in the 8 months before late August (see below), and the pattern is absent in 8 other crypto series. It is a hypothesis for forward testing,
not a proven structure. Other contract lengths (hourly/daily/5-min) were not tested (no usable history).

## Why there can be months with no trades (e.g. July–August)
The signals are there (73 in July, 67 in August) but the **probability gate** blocks them unless the strategy's own recent history beats the price
by more than costs (slippage 2¢ + fee ~1.2¢ + min edge 4¢ ≈ **7¢ per contract**; the min-edge hurdle was 0.5¢ before 2026-10-10 — see "Cross-validation" below). July's signals won 20.5% at an average 24.6¢ price (−4¢ edge), so the gate
stayed shut (2 trades); betting every one would have lost 56%. In August the edge turned positive (+11¢), but an all-history learner was still weighed
down by July and only opened in September. The default **45-day learning window** opens about a week earlier (19 bets the week of Aug 31 vs 4)
and lifts Sharpe (R2 3.9 → 4.6; R1 3.4 → 4.2; R3 3.2 → 3.7). A flat curve means "no edge at the time", not a bug. See the *How it works* tab.

## Kelly sizing sweep (R2, 45-day learner; run `python explore_kelly.py`)
* **Sharpe is almost flat** (4.1–4.7) for any Kelly multiplier up to ~0.5 and any cap, then falls (3.3 at full Kelly uncapped). The highest cells (multiplier 0.10–0.20, cap 1–2%, Sharpe 4.6–4.7) are
  within noise of the default (0.25 / 2% = 4.55). Sizing mostly sets *return vs drawdown*, not Sharpe.
* **Return and drawdown grow with size:** 0.25× / 2% → +239%, −12% DD; 0.25× / 3% → +425%, −16%; 0.25× / 5% → +559%, −14% (best Calmar, but see below); 0.50× / 5% → +665%, −23%; 1.0× uncapped → +777%, −41%.
* **Monte-Carlo (trades reshuffled, 4,000 paths):** P(drawdown > 30%) is 0.4% for 0.25× / 2%, 5.6% for 0.25× / 3%, 39% for 0.25× / 5%, 54% for 0.25× uncapped. The single historical path was kinder than a random ordering.
* In-sample growth-optimal multiplier is ~1.1× full Kelly, so quarter-Kelly is ~22% of "optimal" — deliberately conservative because the edge is noisy and fading.
* **Recommendation:** keep **0.25× Kelly with a 2% cap** (default). Raise the cap to 3% only if you accept ~6% odds of a 30% drawdown for roughly double the return.

## Probability model (what changed)
The **raw historical win rate** over-bets cheap contracts: a 6¢ contract is treated as if it won as often as a 30¢ one (this is what made the
all-day test lose 99%). The default **price-adjusted** basis uses `p = this trade's price + the strategy's historical edge (win − price)`,
shrunk toward zero (skeptical start: no bet until the history beats the price). It is still the strategy's historical probability, estimated
only from signals that already settled (no look-ahead), and every qualifying signal counts whether or not it was traded. The *By UTC hour*
model learns each hour separately and shrinks toward the overall estimate. Switch to **Raw historical win rate** in the panel if you prefer.

## The strategy ("late-window underdog")
In the chosen UTC hours, at minute 12 (then 13) after the 15-minute window opens, buy the **underdog** side (YES if mid < 0.5 else NO)
if its ask is inside the price band; hold to settlement. Signals are read from live **Kalshi** KXBTC15M quotes (the market it was
researched on; far deeper than Polymarket US and sub-second); the order goes to **Polymarket US** (same BRTI index and price-to-beat) as an
**IOC limit** order at `Kalshi ask + max slippage`, so it can never pay more. An optional *favourite* leg buys the other side at 85–97¢.
At most one trade per 15-minute window.

## Modes
| Mode | What happens |
|---|---|
| **Paper** (default) | Simulated fills at ask + slippage; **no orders**, no credentials needed |
| **LIVE** | Real IOC orders. Requires *Connect and test* first and typing `LIVE` in a confirm dialog |

Safeguards: 2% default stake cap, daily-loss pause (10%), max-drawdown halt (25%), buying-power cap, one trade per window, **KILL** button
(cancels open orders, halts), and the engine **halts instead of guessing** if an order's final state is ever unclear. Every settled live
trade logs the real account buying power next to the session equity. Trades are kept in the trade ledger `data/live/ledger.jsonl` (plus a per-session backup `data/live/session_*.csv`) and shown after a restart.

## Daily routine (recommended: preset R2, 0.25x Kelly, cap 1% for the first two weeks)
The strategy only acts in windows that **open between 22:00 and 23:59 UTC**, deciding at minute 12 (13) of each 15-minute window: decisions at **22:12/13, 22:27/28, 22:42/43, 22:57/58, 23:12/13, 23:27/28, 23:42/43, 23:57/58 UTC**; the last trade settles at 00:15 UTC.
| | UTC | US Eastern (EDT until Nov 1) | US Eastern (EST after Nov 1) |
|---|---|---|---|
| Open the app, **Update data**, Connect and test, Preview | 21:50 | 5:50 pm | 4:50 pm |
| Press **Start trading** | by 21:55 | 5:55 pm | 4:55 pm |
| Decisions | 22:12 – 23:58 | 6:12 – 7:58 pm | 5:12 – 6:58 pm |
| Last settlement; press **Stop** | 00:20 | 8:20 pm | 7:20 pm |
Keep the PC awake and online for the whole stretch (about 2.5 hours). Expect ~2–3 trades per evening (a signal on ~92% of evenings), a ~40% win rate, and losing streaks (11 in a row happened in the backtest).
Scale up (cap 2%) only after ~2 weeks of live results that match the paper/backtest behaviour; stop and re-evaluate if the engine halts (25% drawdown) or the edge fades (check `python forward_test.py`).

## First-run checklist
1. `Connect and test` – read-only calls (balance, book age, clock skew). Keep the PC clock within 30 s of real time.
2. `Preview test order` – validates the order format with the exchange **without placing an order**.
3. Run **Paper** through at least one 22–23 UTC session and read the log.
4. Only then LIVE, with a small capital (the order path was built from the Polymarket US docs and tested against a signature-verifying
   mock server, but has never been run against the real exchange).

## Files
`trader/` – `strategy.py` (rules, Kelly, throttle), `probmodel.py` (probability model), `adaptive.py` (Auto mode), `backtest.py`, `metrics.py`, `data.py`, `polyus.py`
(key file, Ed25519 signing, orders), `kalshi.py` (live quotes), `live.py` (engine), `gui.py`.
Tests: `python test_trader.py`, `python test_gui_smoke.py`. Headless paper run: `python paper_run.py --capital 1000 --minutes 30`.
Auto-mode research: `scen_auto.py` / `scen_auto2.py` (60-day futures; data/scenario_auto*.csv, data/scen_auto*.log), `ablate_auto.py`, `final_real_history.py`. Research for earlier versions: `explore_allday.py`, `explore_variants.py`, `explore_grid.py`, `explore_periods.py`, `diagnose_flat.py` (why quiet months), `explore_kelly.py` (sizing sweep).

## Important
The edge is a transient evening-hours anomaly (see `REPORT.md`; forward test in `forward_test.py`). All numbers above are in-sample, the
price band was chosen after looking at this data, and the recent weeks are weaker than the August–September burst. Nothing here is a
recommendation to trade real money. Your key file is only read from disk at "Connect"; the secret is never logged, displayed or sent anywhere
except as an Ed25519 signature in the headers of requests to `api.polymarket.us`.
