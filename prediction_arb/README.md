# prediction_arb — Kalshi × Polymarket cross-venue arbitrage scanner / backtester

Read-only research tool (public endpoints, no keys, never places orders).

    pip install -r requirements.txt      # pyqt6 aiohttp pydantic matplotlib pandas numpy
    python main.py                       # launches the 4-tab GUI
    python tests/test_core.py            # 65 offline tests (hand-computed expectations)

## Tabs
1. **Live Arbitrage Scanner** – Type I parity / II MECE baskets / III strike-ladder subset arbs, filtered by APY hurdle,
   depth and match confidence; click a row for the manual order ticket (limit prices, size re-solver, deep links,
   both rulebooks side-by-side, "log as taken").
2. **Historical Opportunities & PnL** – replays the last 30/60/90 days (settled → realised PnL, open → locked-in
   unrealised PnL) + live-logged + manually logged positions, with locked-capital timeline.
3. **Backtesting Engine** – parameterised historical simulation with capital lock-up, C_max, fixed / fractional-Kelly
   sizing, fee + slippage models, equity vs 5 % cash, utilisation, drawdown, tear sheet + assumption flags.

## Hurdle
D = max((T_settle − T_now)/86400, 0.25);  R = (min payout − cost)/cost;  APY = (1+R)^(365.25/D) − 1 ≥ 7 %.
Capital is locked until the **last** leg settles (+ venue lag).

## Layout
    main.py  config.py  database.py (SQLite arb_data.db)
    core/       models.py pricing.py arb_detector.py scanner.py backtester.py
    connectors/ http_base.py kalshi_client.py polyus_client.py poly_client.py mapper.py
    ui/         common.py app_window.py tab_live.py tab_history.py tab_backtest.py tab_longshots.py
    tests/      test_core.py

## Things worth knowing (verified against the live APIs, Sept 2026)
* Kalshi prices are `*_dollars` strings; data settled before `/historical/cutoff` lives under `/historical/*`
  (newest-first, no time filter). Multivariate combo markets are excluded (`mve_filter=exclude`).
* Polymarket Gamma: plain `offset` pagination is rejected beyond ~2,500 rows (HTTP 422) -> the client uses `/events/keyset` (`after_cursor`); pages cap at 100 events; CLOB price-history windows are limited to ~14 days; `/book` asks come
  sorted worst-first; most markets are **fee-enabled** (`feeSchedule`), so "0 % taker" is not assumed.
* Polymarket history is price-only (no bid/ask) and noisy → synthetic spread + spike filter + mid-sum data gate.
* "Guaranteed" holds only if both venues resolve identically — always read the rules in the ticket.

## US tradability, links and price accuracy
* **Kalshi** is a CFTC-regulated exchange (US-tradable; some states restrict sports contracts - not detectable via the API).
* **Polymarket international (polymarket.com) prohibits US persons.** The default source is therefore **Polymarket US**
  (`gateway.polymarket.us`, CFTC-regulated, public market data, no key): `/v1/events` (nested markets + quotes),
  `/v1/markets/{slug}/book`, `/v1/markets/{slug}/bbo`. A market is one instrument with a long side (YES = long, NO = short,
  NO ask = 1 - best bid). Taker fee = 0.0695 x C x p x (1-p) rounded half-even to the cent. The international feed is an
  opt-in in the "Polymarket source" box, every row/leg is labelled "Poly INTL (not US)" and dropped while "US-tradable
  only" is ticked. Polymarket US lists no 24h volume; exact volume is fetched from `stats` for the top ~600 long-shot candidates.
* **Verified links** (opened in the browser): Kalshi `https://kalshi.com/events/{EVENT_TICKER}` (the series URL only
  reaches the *current* event); Polymarket US `https://polymarket.us/sports/{league}/{slug}` for two-team games and
  `https://polymarket.us/event/{slug}` otherwise; international `https://polymarket.com/event/{slug}`.
* **Price audit** (listing price vs live order book, whole universe): Kalshi 100 % exact, Polymarket US 100 % exact,
  international Gamma 89 % exact with errors up to 85 c (`bestBid/bestAsk` are unreliable for ~4 % of quotes). The ticket
  shows best ask *and* depth-weighted average fill, and "Refresh live quote" / "Refresh live book" re-price from fresh books
  (the Live ticket reports scan-vs-live cost drift and flags "EDGE GONE").
* **Limitations:** Polymarket price history / settled markets exist only on the international feed, so the History and
  Backtest tabs show a red "NOT US-TRADABLE DATA" banner; Polymarket US has no MECE-basket matching yet.

## Long-Shot Scanner (tab 4)
Finds markets whose YES is priced as *extremely unlikely* (default <= 3 % market-implied, adjustable) yet trade real
volume. Runs on the markets every scan already downloads (a Live scan fills it automatically; "Scan long-shots" is a
~30 s fast path). Per market: implied P(YES), YES bid/ask, NO ask, cash volume (Kalshi = contracts x last price, an
estimate), 24h volume, Kalshi open interest, days left, **net carry of buying NO to resolution** (after fees), carry
APY vs the 5 % T-bill, flags (NO unavailable / overdue / long-dated / fee drag / thin book / wide spread / inactive /
multi-outcome tail) and text tags (`absurd` = supernatural, `catastrophic` = war/nuclear/coup). The detail panel shows
the EV of buying NO if the *true* probability is 0.25x ... 5x the market's. **Not arbitrage**: the stake is lost if the
event happens, and "extremely unlikely" is the market's view, not a measured probability.
