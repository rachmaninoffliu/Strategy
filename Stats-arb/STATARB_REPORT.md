# Statistical Arbitrage — Alpha Syndicate team report

*24 Oct 2026 · paper: Avellaneda & Lee (2008), "Statistical Arbitrage in the U.S. Equities Market" (SSRN 1197923)*
*Code `Stats-arb/scripts/statarb_*.py` · GUI `Stats-arb/launch_statarb.bat` · evidence `Stats-arb/reports/` · tests `cd scripts && python -m unittest test_statarb` (20 pass) · layout in `README.md`*

## 1. Verdict

| | Result |
|---|---|
| **The paper's strategy today** (PCA residuals → OU → s-score, bang-bang) | **No net edge.** Fills at the next open, 2017-06 → 2026-10: curated large-cap list Sharpe 0.7 before costs → 0.1 at 3 bp; survivorship-reduced universe 0.3 before costs → **negative** at 1.5 bp (5 bp, the paper's own cost, is worse). The paper already showed the Sharpe falling after 2002. |
| **Best thing we could validate** | The **earnings-announcement premium**, built with the paper's machinery (OU screen, sector-hedged, vol-targeted, shock multiplier, real caps): **net Sharpe 0.45 (SE 0.33), CAGR 2.7 %, vol 6.5 %, max DD −21 %, beta 0.07, corr to SPY 0.19** (in-sample 2017-21 ≈ 0.0, out-of-sample 2022-26 0.8). Block-bootstrap 90 % interval for the Sharpe: **[−0.11, 1.05]**. |
| **How much to believe it** | Not statistically distinguishable from zero: PSR 92 %, deflated Sharpe 0.16–0.47 after ~60 variants. Two years (2020, 2024) carry it; the 10 best days are 82 % of P&L; it breaks even at ~6 bp of costs; it is idle ~22 % of days. Independent validation expects **Sharpe 0.2–0.35 live**. |
| **Recommendation** | **Paper-trade it, do not size it.** No real money (Valid and RM both say no; conditions in §7). The 'max return, max Sharpe' brief cannot be met by this class of signal at retail costs and Reg-T leverage: honest ceiling ≈ Sharpe 0.5, low-single-digit % a year. |

> An earlier version of this work reported Sharpe 0.72 / CAGR 6.6 %. Valid and RM found why it was too good: the earnings calendar covered only 893 symbols (A–B tickers + the hand-picked list), so the "broad" run was the hand-picked large caps again; the 4 % name cap never bound (weights were re-normalised to 1/n, up to 12.5 % per name on thin days). With full calendars (1,482 of 1,496 live names) and real caps the number is 0.45. The first figure is not to be quoted.

## 2. The paper (reproduced) and why it fails now

Returns = systematic (15 PCA eigenportfolios of the 252-day correlation matrix, or the sector ETF) + residual. On 60 days the cumulative residual is fitted as OU: `X(n+1) = a + b X(n) + ζ`, κ = −ln b·252 (> 8.4), m = a/(1−b) (centred across stocks), σ_eq = √(var ζ/(1−b²)); s = (X−m)/σ_eq; open |s| > 1.25, close longs s > −0.50 / shorts s < 0.75; "trading-time" returns × ⟨V⟩/V; 5 bp a side. All implemented (`statarb_core.py`, `statarb_sim.Book`).

**Why it stopped working (measured, not assumed):**

* Information coefficient of −s vs the sector-demeaned forward return (core / broad): 1 day 0.0015 (t 1.0) / 0.0008; 5 days 0.0032 (t 2.3) / 0.0040 (t 4.1). The only large effect is the **overnight close→open reversal** (IC 0.013 / 0.009, t ≈ 10) — but as a long/short book it earns **0.6 bp a day gross** (≈ 1.5 %/yr at gross 1) against 1–2 bp of round-trip cost.
* Fade return by |s| bucket is 1–6 bp over 3 days in every liquidity tier; it does **not** grow in less liquid names (broad universe, 1,600 names a day).
* The sign of the paper's volume finding flipped: heavy-volume / news moves revert *more* today (6–10 bp), the opposite of the 2008 "believe the volume" result — but too small and sparse to pay costs (event-gated variants: Sharpe −0.4…+0.3, drawdowns to −90 %).
* Term structure (1 day … 12-1 months, sector-relative): no bucket with |t| > 2.
* Reversion-only with every 2026 overlay switched on (neutral to beta, PCA, sectors, momentum/vol/size, oil/rates/USD): Sharpe −0.4 — the projection removes factor exposure and with it the little alpha.

## 3. Today's market and the risk map

From press summaries (levels approximate): Fed **hiking**, inflation risks to the upside (FOMC minutes 7 Oct); **Iran / Hormuz** oil shocks since late February; chip and auto **tariffs**; Chinese **"quarantine" drills near Taiwan**; breadth at a dot-com-era low (Goldman, 28 Sep: median stock ~16 % below its high, index ~1 % from its record) with AI-led concentration; **idiosyncratic share of stock variance ~73 %** (58 % average); three sharp **quant drawdowns in 2026** (Jan junk squeeze, Mar, Jun–Jul momentum/AI-memory unwind). SPY 21-day realised vol only ~10 %.

| Risk | In the engine |
|---|---|
| Idiosyncratic jumps | Reversion sleeve: no fading of news / post-earnings sessions, no entry within 6 sessions of an announcement, exit 2–3 sessions before; risk-parity sizing. Earnings sleeve: holds the announcement **on purpose** but equal-weights ≥ 8 names, **true 4 % name cap and 30 % sector cap (cash if few report)**, skips names whose average earnings gap exceeds 20 %, hedges beta × sector SPDR. *RM:* gap carries no premium on average (+0.05 %, sd 5 %); ~80 % of the stock-leg P&L is pre-announcement drift; the hedge gives ~60 % of it back. |
| Crowding / factor unwinds | Projection onto the null space of SPY beta, top-5 PCA factors, 11 sectors, momentum, volatility, size (reversion sleeve). Stress windows: Dec-2018 −6.9 % (SPY −18.7 %), COVID +0.2 % (−20 %), 2022 −2.7 % (−22.7 %), yen-carry −2.5 %, Jan-2026 junk squeeze +2.2 %, Hormuz +2.6 %, Jun–Jul 2026 unwind −0.2 %. |
| Geopolitics / macro | Per-stock oil, rates, USD betas neutralised (reversion sleeve). **Shock multiplier** from 3-day z-moves of SPY↓, VIXY↑, \|USO\|, \|TLT\|, GLD↑: −25 % gross per σ above 2 for 5 sessions. Worth +0.06 Sharpe here (the 2026 events were known when it was designed — not clean out-of-sample). Today the multiplier is 0.8: a 2.8σ Treasury move on 22 Oct is still inside its 5-session hold. |
| Leverage / liquidity | Gross ≤ 1.6× (RM: 2.0× left zero Reg-T slack on 25 % of days); ADV floor $150M; liquidity-tiered costs (×1, 1.7, 2.7, 4); market-on-open only. |
| Decay | The default sleeve is the earnings premium; the paper's book stays available at weight 0. |

## 4. Method

* **Data (DA).** Alpaca daily bars 2016-01 → 2026-10-23 (SIP, split+dividend adjusted). *Core*: 431 hand-picked current large caps (hindsight-selected → flatters results). *Broad*: 3,757 names filtered each day by price ≥ $5 and 20-day median dollar volume, **including delisted tickers** (Alpaca keeps their bars) — but Alpaca's inactive-asset list stops around 2022, so there are no 2023-26 deaths (SIVB, FRC, TWTR, BBBY … are missing), and Yahoo earnings dates exist only for names that still trade (5 of 138 dead names). The earnings pool is therefore survivor-only; Valid measures the resulting bias at ~0.1–0.5 %/yr. Yahoo calendar: 2,540 symbols. 20 factor ETFs.
* **Execution model.** Signal on the close of *t* → market-on-open at *t+1*; P&L open-to-open; costs on |target − drifted holding| (1.5 bp one-way, tiered up for thinner names); short borrow 50 bp; ETF hedge 0.5 bp; real −60 % crashes kept when volume confirms (artefacts masked).
* **Validation.** Look-ahead tests (corrupt prices after *t* → nothing ≤ *t* changes), Valid's independent rebuild on data truncated at 2020-05 and 2023-06 (identical to 1e-16), 20 unit tests, IS 2017-21 / OOS 2022-26, ablation, cost, liquidity-band and neighbourhood tables, deflated Sharpe with ~60 trials.

## 5. Results (default preset, broad universe, ADV ≥ $150M, $100k)

*Note: the tables below were generated before `eap_max_names=100` became the default (it keeps tickets above the $250 minimum in peak earnings weeks, when 200–350 companies report). With it: Sharpe 0.46, CAGR 2.9 %, vol 6.6 %, DD −20.8 %, IS −0.02 / OOS 0.83 (GUI backtest). The re-runs with that setting and the complete earnings calendar (3,283 symbols) are `reports/statarb/final3_broad` (with ablation / cost tables) and `final4_broad`: Sharpe 0.465 (SE 0.33), CAGR 2.9 %, vol 6.6 %, DD −20.8 %, IS −0.02 / OOS 0.83, DSR 0.18. `final2_*` are the earlier runs the tables were taken from.*

| | All | In-sample 2017-21 | Out-of-sample 2022-26 |
|---|---|---|---|
| Sharpe (net) | **0.45** (SE 0.33) | −0.01 | 0.79 |
| CAGR / vol | 2.7 % / 6.5 % | −0.2 % / 5.6 % | 5.7 % / 7.3 % |
| Max drawdown | −20.7 % | −15.4 % | −9.5 % |

*By year:* 2017 +1.5 %, 2018 **−9.6 %**, 2019 −2.2 %, 2020 **+23.0 %**, 2021 **−10.3 %**, 2022 −5.5 %, 2023 +1.7 %, 2024 **+21.0 %**, 2025 +3.4 %, 2026 YTD +8.3 %. 16.5k trades, 56 % win rate, +56 bp average, 6 sessions average hold; median 19 names (peak earnings weeks 100–240; off-season flat).
*Risk ($100k):* 1-day VaR95 $555, VaR99 $1.2k, CVaR99 $1.7k; worst day −2.9 % (2021-03-03); avg gross 0.88× (max 1.6×); net beta 0.01; cost 1.0 %/yr.
*Costs:* Sharpe 0.58 at 0 bp, 0.45 at 1.5, 0.35 at 2.5, 0.22 at 4, 0.03 at 6.
*Ablation (Sharpe):* default 0.45 · paper rules 0.00 · reversion sleeve only −0.39 · **unhedged earnings sleeve 0.86** (that is equity beta, not alpha) · no OU screen 0.49 (**the paper's signal adds nothing to the surviving edge**) · no shock multiplier 0.39 · no vol target 0.40 · costs ×2 0.31 · ×0.5 0.52.
*Last 60 sessions:* −2.0 % (SPY +3.5 %). Core list, same settings: Sharpe 0.55, CAGR 1.9 %, vol 3.6 %, DD −7 % — the hindsight list is not much better once the caps are real.
*PM:* as an overlay on a fully-invested SPY it adds only ≈ +0.03 Sharpe (0.87 → 0.90 at 0.5× overlay) and +1.5 %/yr.
*Sensitivity RM/Valid flagged:* Sharpe vs ADV floor $50M/75/100/150/300/500M = 0.27/0.44/0.72/0.84/0.66/0.26 on the *incomplete-calendar* run — i.e. the result depends on where the liquidity line is drawn; schedule 1 session early −0.1…0.3; min-names 25 → 0.16; windows [3,10] → 0.38. The effect is announcement *timing* (placebo windows shifted 21+ sessions are negative), so it needs accurate dates, and live uses projected dates.

## 6. The team

* **QR 🔬** — The OU model is a better *screen* than a signal today. The overnight reversal is real (t ≈ 10) and I would love it to be tradeable; it is 0.6 bp/day. The earnings premium is the only positive: 81 % of it is pre-announcement drift, not the gap. I would not call 0.45 an edge.
* **DA 🗄️** — Alpaca serves bars for delisted tickers (big upgrade) but the asset list silently stops at ~2022 and Yahoo has no calendars for dead names: the "survivorship-free" label is only half true. Yahoo throttles silently — the first earnings pull returned 893 of 2,400 symbols without an error. Fixed with checkpoints, cool-downs and retries; coverage now 1,482/1,496 live names.
* **Dev ⚙️** — Backtest and live both call `Portfolio.decide` (parity tested). After the audit: real name/sector caps; hedge sized from the shares that will actually be held; staleness and incomplete-cache guards; broker positions are the truth at plan time; vol/drawdown state persisted; KILL closes min(ledger, broker) and forgets only what closed; retry-safe submits; incremental broad cache; arming a paper account can no longer carry over to LIVE.
* **Valid 🛡️ (independent agent)** — *Formal Level-1 REJECT* on universe/data; no look-ahead in the code (rebuilt on truncated data, identical); headline haircut 55–70 %; dry run OK, armed paper only after the live defects (stale-data liquidation, ARM-LIVE bypass, KILL, ledger-vs-fills, cap/hedge divergence) — **now fixed and tested** — not ready for real money: DSR < 0.95, edge in 3–4 years, no paper period.
* **RM 🔒 (independent agent)** — Approve for paper as an ops test; **reject $10–25k real** (the order logic built a naked ETF short at small account sizes — fixed) and larger. Proposed limits adopted: gross 1.6×, 4 % name cap, 30 % sector cap, ADV floor, daily-loss halt (−2.5 % default, enforced), drawdown stops (−8 % halve, −10 % flat, manual reset), kill triggers. Not yet automated: participation ≤ 0.5 % ADV check, stale-earnings-date detection, beta-band kill.
* **PM 👑** — Allocation to the reversion sleeve: 0. Earnings sleeve: paper only, zero capital until ≥ 60 live sessions track the shadow backtest within 2σ and the deflated Sharpe is revisited. As an overlay it is not worth the operational risk.
* **QT ⚡** — Market-on-open, plan after 16:25 ET, queue 19:05–09:25 ET. With 100–240 announcers at $100k each ticket is ~$350–500 (whole shares; the plan prints how much of the book is lost to 1-share rounding); the sleeve wants ≥ $250k. Opening-auction participation is trivial at $100k–$1M (0.2–1.6 % of an assumed reaction auction), not at $10M. A missed OPG window means waiting a day (no fallback by design).

*Disagreements kept on the table:* QR/QT on the overnight effect (statistically huge, economically nil); Valid vs RM on paper — Valid wants the engine defects closed first (done), RM accepts paper as an ops test now; PM vs QR on the premium (risk premium to hold vs "probably not an edge").

## 7. What would have to be true before real money

1. ≥ 60 sessions of PAPER (ARMED) trading whose P&L stays within 2σ of the shadow backtest and whose fills/hedge match the plan.
2. A calendar of announcement dates that is verified against issuers' own announcements (the Yahoo history is as-of-today).
3. A survivorship-complete earnings pool (needs a paid database with delisted names and point-in-time dates).
4. Deflated Sharpe > 0.95 on data not used to choose anything, or an economic reason strong enough to accept less.
5. Account ≥ $250k so 1-share rounding does not break the hedge; margin account with Reg-T 2× and no PDT constraint.

## 8. What could genuinely do better (not built)

Intraday residual reversion with auction-based execution and rebates; non-price information (analyst revisions, short interest, options skew, news NLP); a paid, point-in-time constituent + earnings database; portfolio margin; multi-asset stat-arb (ETF/futures basis, share-class pairs) where costs are an order of magnitude lower.

## 9. How to use

`Stats-arb/launch_statarb.bat` → starts on **Paper + DRY RUN**, default preset = the earnings sleeve on the broad universe. *Signals & risk* tab (today's s-scores, shock dashboard, 2026 risk map) → *Live trading*: Pre-flight → Plan next session (dry run) → [ARM + Submit, paper only] → Reconcile → KILL. *Backtest* tab: any window, IS/OOS split, Ablation, Cost & parameter checks. Reproduce the numbers: `python statarb_report.py --universe broad --trials 60`.
