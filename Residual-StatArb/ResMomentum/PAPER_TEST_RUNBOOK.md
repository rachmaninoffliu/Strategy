# Residual Momentum - paper forward test runbook

**What this is**: a forward test of the research choice on Alpaca paper. Valid signed off paper only, RM approved paper only, PM allocates no real capital.
* Settings: 3-1 month residual momentum, top / bottom 10%, hold until a name leaves 25%, ETF-hedged, stock gross 0.9x, stocks + hedges capped at 1.9x.
* Backtest as configured: Sharpe 0.46 (in-sample 0.37, out-of-sample 0.59), 9% vol, -14.5% max drawdown.
* The book trades **once a month**, on the **last NYSE session of the month at the 15:45 ET bar**.
* Next rebalance days: **Fri 2026-10-30**, Mon 2026-11-30, Thu 2026-12-31, Fri 2027-01-29.

## 1-2. Keys: the project's key file
Every app in the project folder reads its Alpaca keys from **`Alpaca Kay.txt`** in `C:\Users\xinyu\PycharmProjects\pythonProject`. The file has two lines, `Key ID: …` and `Secret Key: …`, and is git-ignored. It holds the keys of paper account **PA38…DI**; nothing else to set up.

**That account is shared by every app in the folder.** While ResMomentum holds positions there:
* **Don't arm another app on it.** Of the residual strategies, the first armed order claims the account for ResMomentum (`../data/resarb_accounts.json`) and the other two are refused. But the Supertrend, LevETF-Pairs, Stats-arb and Intraday-A3C apps don't check that registry.
* **Supertrend's KILL closes everything in the account.**
* **LevETF-Pairs' 15:52 orphan close and its watchdog act on SPY**, which is ResMomentum's market hedge.

For an isolated test, open a second paper account (dashboard: paper account number at the top left → *Open New Paper Account* → generate its keys). Then set `RESMOM_ALPACA_API_KEY` / `RESMOM_ALPACA_SECRET_KEY` (they override the key file for ResMomentum only):
```powershell
[Environment]::SetEnvironmentVariable("RESMOM_ALPACA_API_KEY", (Read-Host "ResMomentum API key"), "User")
[Environment]::SetEnvironmentVariable("RESMOM_ALPACA_SECRET_KEY", (Read-Host "ResMomentum secret key"), "User")
```

## 3. Check the connection (any day, DRY RUN)
1. Double-click `launch_resmomentum.bat`. The app opens on **PAPER + DRY RUN**.
2. *Live trading → Connect*: the log says `connected (keys from the key file Alpaca Kay.txt)`. The account label shows PA38…DI (or your separate account if you set the RESMOM variables), and `keys: OWN`.
3. *Pre-flight check*: every line OK except `account owner: not claimed yet`. If it shows `keys … not the project's key file` or `arming: no own keys`, the key file wasn't found.
4. *Plan this bar now (dry run)*: shows the book the strategy would hold right now. Expect about 10 longs, 10 shorts and 6-10 ETF hedge orders, with beta after hedge ≈ 0. Nothing is sent.
5. *File → Save settings…* → `ResMomentum_settings.json` in this folder (the headless runner reads it). Check the left panel shows the research choice: formation 63, skip 21, monthly, 15:45, feed `sip_delayed`, gross 0.9, max gross 1.9.

## 4. The first watched session: Friday 2026-10-30 (all times ET)
| Time | What you do | What you should see |
|---|---|---|
| ~15:00 | Start the app, Connect, Pre-flight, *Update data*, *Plan this bar now* | a clean pre-flight; the dry plan's book |
| ~15:15 | Tick **ARM** → type `ARM PAPER` → *Start 15-minute loop* → confirm | log: `loop started (ARMED, paper, feed sip_delayed, lag 1)`; the SIP cache top-up takes ~1-2 min |
| 15:15:25, 15:30:25 | watch | log: `15:30 no rebalance at this bar (data 15:00; equity $…)` - the loop is alive, no orders |
| **15:45:25** | watch | log: `15:45 decision on 15:15 data: … orders` then `sent N / N orders`. N ≈ 26-30 (20 stocks + ETF hedges), all buys / sells to open. *Orders for this bar* lists them |
| 15:46-15:50 | compare the Alpaca dashboard's positions with *Target book* | same symbols and shares (paper fills ~10% of orders in random partial pieces; the rest arrive within seconds) |
| ~15:50 | *Stop loop*, untick ARM | log: `loop stopped` |
| after 16:20 | *Update data*, then *Compare last session with the backtest* | target correlation ≈ 1.0 (both use SIP data); fill slippage vs the backtest's 15:45 open is a few bp |

**If anything looks wrong:**
* Before 15:45: just *Stop loop*. Nothing has been sent.
* After the orders: **KILL** (type `KILL`) cancels the strategy's open orders and market-closes every position in its symbols, while the market is open (before 16:00).
* The Alpaca dashboard's *Close all positions* does the same for the whole account, which holds only this strategy.

Things that are expected, not errors: `positions outside this month's symbols are left alone` (if you put other trades in this account - don't), `target gross … > cap 1.90x - scaled down` (heavy-hedge months; the backtest applies the same cap).

**Headless alternative**:
1. Save the settings (step 3.5).
2. In `run_live.bat`, swap the comment marks so the `--arm "ARM PAPER"` line runs.
3. Start it at ~15:00 from a new console. It stops by itself at 16:05.

## 5. Between rebalances
* Nothing trades until **Mon 2026-11-30, 15:45**. The positions are held and drift.
* The daily-loss halt (3%) and drawdown stop (15% from the high) act only while the loop runs. They flatten at the next 15-minute bar, not the next month-end. For the first month, check the account in the dashboard once a week. If equity is down more than ~10% from the start, open the app and KILL, or ask for a review.
* After one clean rebalance you can run `run_live.bat` (ARMED) every trading day, e.g. from Task Scheduler at 09:00. The stops then act automatically. The PC must be awake, and on non-rebalance days the runner only reads data and the account.
* At each month end, members that left the top 100 are closed and new ones bought.

## 6. What paper cannot show
* Alpaca paper does **not** simulate dividends or borrow fees: longs receive no dividends, shorts pay none. The backtest uses dividend-adjusted prices and a 30 bp a year borrow cost, so paper P&L will drift from it by roughly the dividend difference between the two sides.
* Paper fills against the NBBO with no market impact. Real fills at 15:45 cost about the modelled half spread (~1.4-2 bp for these names at that time).
* The forward test answers two questions: does the engine trade exactly what the backtest says (the compare tool), and does the edge persist out-of-sample (months of returns)? The Sharpe needs years to become significant. PM's rule: reconsider capital only when the out-of-sample t-statistic exceeds 2 **and** Valid re-signs.

## 7. Files
* `reports/journal_paper.jsonl`: every decision (targets, held shares, prices), order and fill.
* `reports/state_paper.json`: high-water mark, day-start equity, drawdown-stop flag.
* `reports/compare_*_paper_<date>.csv`: the comparisons.
* `../data/resarb_accounts.json`: which strategy owns which account. The first armed order claims the new account for ResMomentum; the other strategies are then refused on it. If you delete the paper account later, delete its line here.
