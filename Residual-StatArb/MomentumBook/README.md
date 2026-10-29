# Momentum Book: backtest + live (Alpaca)

A long-only, unlevered book with two momentum legs on the same capital, never held at the same time:

* **Overnight leg.** At 15:30 the top-150 stocks + 59 ETFs are ranked. The 8 best are bought at the close, at most 3 from one sector, and sold at the next open, only when SPY's last close is above its 200-day mean.
* **Intraday leg.** QQQ noise-area breakout (Zarattini, Aziz & Barbon 2024), checked every 30 minutes on the **IEX** feed. QQQ is flat every night.

The overnight names are sold at the open before the intraday leg can buy, and the intraday position is sold at the same close that buys the overnight names. Gross exposure is ≤ 1x at every moment.

**Execution.** The models trade in the auctions: market-on-close (MOC) buys, market-on-open (MOO) sales, and a MOC sale of QQQ when it is still held after the last check. The backtest prices them at the official close / open plus 1 bp per side.

The live tab's **Auction orders** box replaces only those orders. The model, its decisions and its sizes stay the same, and the intraday checks' market orders are not affected:

* **MOC / MOO orders, as the model** (default).
* **market DAY orders: 5 s before the close, 3 min before the open.**
  * The 15:46 decision is queued and sent at **15:59:55**: the QQQ sale and the exits first, then the buys, in parallel (about 1 s for 20 orders). Nothing is sent in the last 2 seconds, because a day order after the close would trade at the next open.
  * Last night's names are sold with market orders sent at **09:27:00** and queued for the open. On a real Alpaca account an order received before 09:28 goes into the opening cross.
  * The fills are compared with the backtest's auction prices. In *Positions + trade history*, `diff` is what the market orders cost against the auctions; *Check live vs backtest* shows the cost of each order in bp.

The box is greyed out for a model that sends no auction order. Pre-flight checks the PC clock against Alpaca's, because these orders are timed to the second.

Alpaca's paper account does **not** run the auctions. It fills MOC / MOO orders at the quote in the last seconds, partly at random, and expires the rest (2026-10-28: 6 of 20 MOC buys filled, $20k of $96k). Use market DAY orders on paper.

The edge lives in the auctions. Market orders priced at historical quotes earned Sharpe 0.46 against 2.06 (`REPORT.md` §6), so the paper test measures that cost on real fills.

Research: `../MOMENTUM_PREDICTION_REPORT.md`. Results, team review and verdict: `REPORT.md`. Code: `../core/resarb/mom/`. Tests: `../core/tests/test_mom.py`.

## Presets (the model picker)
| Preset | Overnight ranking | Use |
|---|---|---|
| **Traded: rule + ML forecast, top 8 (max 3 per sector), 15:30, QQQ intraday on IEX** | rank of today's move + rank of the walk-forward ML forecast of the overnight return; the 8 best, at most 3 per sector | the book to trade |
| **Shadow: rule only, top 8 (max 3 per sector), 15:30, QQQ intraday on IEX** | today's move; the same k and cap | runs next to the traded book in the Live tab (journaled, never traded) |
| Research reference: rule only, 15:45, SIP intraday | today's move | the research's a-priori book; needs the paid real-time SIP plan live |
| Overnight leg only / Intraday leg only | — | each leg alone |

**Sector cap** (model panel; 0 = off): at most N overnight names per sector, in score order. A skipped name's place goes to the next-best name of another sector. The Traded and Shadow presets use k = 8 with a cap of 3 (`REPORT.md` §7). The other presets keep the research's k = 20 without a cap. Every field can be edited. The backtest runs the model in the panel, and the live tab trades exactly that model: the panel is locked while the loop runs, and the tab warns if the model was not backtested in this session.

## Using it
1. **Data.** *Update data* in the left panel tops up the universe, the daily bars, the 15-minute SIP bars of the universe, and the QQQ IEX bars.
2. **ML models.** The rule + ML book needs one model per year, fitted on the data before that year. *Backtest > Train ML models* fits the missing years (about 2 minutes for 2018–2026); headless: `python -m resarb.mom.ml ..\MomentumBook` from `core`. The files are kept in `models/`. **Every January, train the new year's model**; pre-flight fails without it.
3. **Backtest.** *Run backtest* shows:
   * KPIs next to SPY buy & hold (full, in-sample, out-of-sample);
   * equity and drawdown vs SPY;
   * the **full trade list**: leg, ticker, shares, entry/exit date and time, auction or market order, prices, value, ranking score and rank, today's move, ML forecast, exit reason, costs, P&L. It is filterable by leg and ticker and exportable to CSV;
   * tabs: years, legs, tickers, exposure, and the ML models used.
4. **Live trading** (paper; starts in DRY RUN):
   * *Connect*, then *Pre-flight check*.
   * *Plan overnight now* / *Intraday check now*: dry runs on today's data.
   * *Start the daily loop*.
   * To send orders: tick ARM and type `ARM PAPER`.
   * *Positions + trade history* reloads the open positions and every trade of the book from the **broker's order history**. It works on paper and real money, and does not need the journal. Each row shows the backtest's price and P&L for the same shares: buys at the official close, sales at the next official open, intraday orders at their check's bar close ± half spread. Partial fills are flagged. It also loads on Connect, after each plan and every 10 minutes.
   * *Check live vs backtest* rebuilds every journaled decision and check from the backtest's data and compares them. It then lists every order sent at the open or close against the official price, with the fill rate and cost in bp.
   * *Shadow report* compares the traded and shadow nights.
   * *Auction orders* (Mode box), for paper and real money alike: **MOC / MOO orders, as the model** (default) or **market DAY orders 5 s before the close and 3 min before the open** in their place. The headless runner uses the choice saved with `live_model.json`.
   * *Account*: **Paper** or **Live (real money)**. A live account can be connected and viewed (positions, orders, trade history). Sending real orders needs `RESARB_ALLOW_LIVE=1` and the typed phrase `ARM LIVE MONEY`. The tab always starts on Paper.
   * *Order history (broker)* lists every order of the book: time, ticker, side, shares, order type, leg, status, filled shares, fill price.
5. **Headless.** `run_live.bat` runs a DRY-RUN session (start before 08:45 ET) on `live_model.json`, which the GUI writes when its loop starts. The ARMED line is in the file, commented out.

## Daily cycle (ET)
| Time | Step |
|---|---|
| 08:45 | MOC/MOO: last night's names get market-on-open sells first; then the daily and 15-minute bars are topped up |
| 09:27:00 | market DAY orders: last night's names are sold (queued for the open) |
| 09:31:00 | MOC/MOO: anything the opening auction left is sold with a market order |
| 10:00:05, 10:30:05 … 15:30:05 | intraday check on the 30-minute IEX bar that just ended → market buy or sell of QQQ |
| ~15:36 | today's universe bars are read ahead |
| 15:46:06 | the overnight decision. MOC/MOO: QQQ (if still held) gets a market-on-close sell and the 20 names market-on-close buys (cut-off 15:50). Market DAY orders: the orders are queued |
| 15:59:55 | market DAY orders: QQQ sale, then exits, then the buys, sent in parallel. Nothing is sent in the last 2 seconds, because a day order after the close would trade at the next open (journaled as `close_orders_cut`) |
| 16:05 | fills journaled, overnight book rebuilt from the broker's positions, intraday round trips journaled |

On early-close days everything moves with the close: last check 12:30, decision 12:46, closing orders 12:59:55.

## Requirements and safety
* **Day trading.** A margin account with at least **$25,000** (pattern-day-trader rule); pre-flight checks it.
* **Data.** The free Alpaca plan is enough: delayed SIP for the 15:30 decision, real-time IEX for the intraday leg.
* **Safety.** PAPER + DRY RUN at start. ARM needs the typed phrase. Real money is disabled in code (`RESARB_ALLOW_LIVE=1`; Valid has not signed off).
* **Keys.** The project's `Alpaca Kay.txt`; `MB_ALPACA_API_KEY` / `MB_ALPACA_SECRET_KEY` select a separate paper account.
* **One strategy per account.** An account claimed by another strategy (MeanReversion, the residual apps) is refused. Run the book on its own paper account, or stop the other one first.
* **Only its own shares.** The book trades only the shares its own orders bought. It works them out from the broker's order history: its filled buys minus its filled sales, never more than the broker holds. Anything else in the account is never sold, never closed by KILL and not part of the book's capital. That covers other tickers, and also shares in the book's own tickers (AAPL, QQQ, ...) bought by hand or by another tool. With no position of its own, the morning step sends nothing.
* **KILL** disarms and blocks every order until a person clears it with *Reset drawdown base / clear KILL*. It then cancels the book's open orders and closes the book's own shares with market orders. It never closes a symbol whose earlier order could not be cancelled, which could double-sell. It refuses an account claimed by another strategy.
* **Never above 1x.** If the QQQ closing sale does not go out, no overnight buys are sent. If an exit fails, the lowest-ranked buys worth it are dropped.
* **Never after the close.** No order is sent in the last 2 seconds, and orders are not resubmitted automatically. After the close, the session's still-open orders are cancelled, so a late order cannot trade at the next open.
* **Never sold twice.** Every sale, including the decision's exits when they are sent, is netted against the book's open orders, read before the positions: an opening auction still running, or KILL orders queued for the next open.
  * Known, safe gap under MOC/MOO: an opening-auction order still open at 09:31 that later expires unfilled is not retried. Its name is then sold at the close. Nothing is sent while a KILL order is still open, and every order re-checks ARM and KILL.
* **One loop per account.** Every loop, armed or dry, holds the account's lock. A dry loop keeps its own step flags, so it never uses up an armed loop's steps for the day.
* **Missed steps.**
  * Started after the decision window: last night's book is sold before the close.
  * Under MOC/MOO, a missed decision also sells QQQ.
* **Optional live-only overlays:** daily-loss halt, drawdown stop.
