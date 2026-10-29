# Month-long PAPER test: runbook

What this machinery does: the engine trades the pairs on your Alpaca **paper** account, sends **both legs at the same moment**, records **every order** with every timestamp, and every evening compares the day with the
backtest of the same session (research code, same configuration) so that you can see where live and backtest part ways: the decisions, the fills, the P&L, the latency.
It never touches a real-money account (there is no switch for one in `levpairs_daily.py` / `levpairs_watchdog.py`). Valid has signed off no capital; RM's verdict is in `LEVPAIRS_REPORT.md` section 6.

## 0. Read this first: what a month on IEX data can and cannot tell you

This account can read **real-time IEX** data only (a paid market-data plan would add real-time SIP; check Alpaca's current plans). IEX is 1-3% of all trades in these ETFs, and the strategy was researched on the consolidated tape:

| (one session, 10:00-14:00 ET, `reports/feed_gap.log`) | last trade before a 5-minute boundary, median age | first trade after it, median wait | trades per minute (IEX / SIP) |
|---|---|---|---|
| QQQ | 4.1 s (SIP 0.06) | 3.0 s (0.03) | 24 / 1,774 |
| TQQQ | 5.8 s (0.12) | 4.4 s (0.09) | 14 / 776 |
| SOXL | 3.5 s (0.09) | 6.3 s (0.04) | 17 / 1,835 |
| SPXL | 118 s (2.2) | 157 s (1.7) | 0.9 / 53 |

So the spread built from IEX bars is made of stale prints: IEX s-scores correlate only 0.2-0.57 with the consolidated ones, the rules trade about 8x as often (11 vs 3 trades on 2026-10-23), and a backtest run purely on IEX bars even looks
profitable (Sharpe +1.3 at the next-open fill, because the "next open" is itself a stale IEX print) while the same signals marked at real prices lose (Sharpe -13.7, `feed_test.log`). **A month of ARMED trading on IEX signals therefore measures the
plumbing, the fills and the size of the feed gap, not the strategy.** If you want the strategy itself tested live, get real-time SIP and run with `--feed sip`; everything below works the same.

## 1. One-time setup

1. Use a **dedicated** paper account (reset it) **with its own API keys**: RM's verdict assumes no other strategy trades TQQQ, QQQ, SOXL, SMH, SPXL or SPY there (the Intraday-A3C basket holds SMH; a Supertrend KILL closes everything). By default the keys are read from `~/.claude.json`, which is the account the Claude tools use - so the dedicated account's keys must be given to the job as the environment variables `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (Windows: `setx ALPACA_API_KEY "..."` and `setx ALPACA_SECRET_KEY "..."`, then open a new console; the variables win over `~/.claude.json`). **This is enforced:** an ARMED `levpairs_daily.py` job and a watchdog started with `--flatten` refuse to start without those variables (to use the shared account knowingly: `--shared-keys "SHARED ACCOUNT"`). Paper equity must stay at or above $25,000 (the pattern-day-trader rule: the strategy opens and closes positions every day).
2. The PC: set Windows **not to sleep or hibernate** (the job also asks Windows to stay awake while it runs, but it cannot wake a sleeping PC), keep the clock synchronised (Settings > Time: sync now; the pre-flight fails above 30 s of difference from the broker and warns above 2 s), keep it on mains power, and **do not close the console window, log off or reboot while a session runs**: that ends the engine without a flatten (Ctrl+C, Ctrl+Break and a plain `taskkill` do flatten; a closed window, "End task", a forced kill, logoff or a power cut cannot be caught - that is what the watchdog is for).
3. Keys: the same Alpaca paper keys the app uses. Bars: `python levpairs_data.py` (and `python levpairs_data.py iex`) once if `data/` is old; the evening job tops the bars up every day.
4. Rehearse in DRY RUN first: run `run_levpairs_day.bat` as it is (DRY RUN) for two or three sessions and read `reports\live_compare\REPORT_<date>.md`: the decisions compare, the latency is measured, nothing is sent.
5. Arm it: edit `ARGS` in `run_levpairs_day.bat` (the commented ARMED line: `--arm "ARM PAPER"`, `--accept-iex "ACCEPT IEX"` when on IEX, `--capital` = dollars to deploy, e.g. 20000). The engine's own caps still apply: gross 25% of equity, any single leg 10%, one entry per pair at a time, at most 40 round trips per pair per day, a daily-loss halt at 1% of the deployed capital.
6. Schedule (Windows Task Scheduler > Create Task): **`run_levpairs_day.bat`** weekdays at a local time before 09:25 ET (it waits for the open itself, so daylight-saving shifts cannot make it late) and **`run_levpairs_watchdog.bat`** weekdays before 09:30 ET. Easiest: stay logged in to Windows (a task set to "run whether the user is logged on or not" needs your Windows password stored and cannot show its console or the watchdog's pop-ups). In both tasks, **Conditions: untick "Start the task only if the computer is on AC power" and "Stop if the computer switches to battery power"; tick "Wake the computer to run this task". Settings: "Stop the task if it runs longer than" 12 hours or more (the default 72 h is fine, a short limit would kill the engine at its limit with positions open), "If the task is already running, do not start a new instance".**
7. The watchdog launcher starts in the DRY-day mode (`--no-engine-expected`: it does not alarm when no engine is running). **Before the first ARMED day** edit `run_levpairs_watchdog.bat` to the `--flatten "FLATTEN PAPER"` line (RM's recommendation: ON for a dedicated account; it then also alerts when no engine is running on a trading day). Leave `--close-orphans "CLOSE ORPHANS"` OFF: the armed day job already closes unexplained positions in the six ETFs at the 15:52 flatten (`--no-orphan-eod` switches that off).
8. **A notification that reaches you away from the PC** (the watchdog's pop-up window does not): set the environment variable `LEVPAIRS_NOTIFY_URL` to a URL that accepts an HTTP POST of the message text - the simplest is [ntfy](https://ntfy.sh): install the ntfy phone app, subscribe to a topic nobody can guess, then `setx LEVPAIRS_NOTIFY_URL "https://ntfy.sh/<your-secret-topic>"` (new console). Pushed: every watchdog alert (ENGINE DOWN, ENGINE NOT RUNNING, positions still open at 15:56:30, STILL HELD after a flatten, LATE), "the leftovers could NOT be cleared", "the session ended with an error", "the evening comparison failed", and one line per armed day (trips, P&L, matched trips, slippage against the NBBO mid). The message is the alert text itself (it can contain the positions); it passes through that service. The same alert goes out at most once in 10 minutes. It is untested against a real phone: send yourself a test with `python -c "import levpairs_notify as n; n.send('test', wait=True)"` from `scripts\`.

## 2. What a day looks like

| time (ET) | what happens |
|---|---|
| 09:25 | the job starts (an ARMED job without the account's own keys stops here). Leftovers of an earlier session (positions in the ledger, an order that was in flight) are closed at 09:30:30, once the market is open, before anything else (it stops with an error if it cannot); then pre-flight (account, shortable, margin, data, clock skew); it refuses to start if `reports\KILL` exists |
| 09:45 - 15:30 | every 5-minute bar, 4 s after it ends: signals -> decisions -> **both legs of a trade sent together** (market orders), confirmed, a failed leg unwinds the other. An order whose answer is lost is looked up by its client id (three tries); if it stays in doubt the same step waits 5 s, cancels this engine's open orders and closes what the two legs left, and the pair stays BROKEN. Nothing opens in the first 15 or last 30 minutes |
| 15:52 | hard flatten (retried until 15:58:30), then any unexplained position in the six ETFs is closed too (armed jobs), the engine stops |
| 09:40 - 16:10 | the watchdog watches every 20 s: a pop-up and a line in `ALERT_<date>.txt` if no engine is running on a trading day (09:40), if the heartbeat is lost with positions open (two looks 20 s apart, none of them right after a sleep), or if positions are still open at 15:56:30; with `--flatten` it closes what a dead engine left and leaves `levpairs_watchdog_paper.json` so that a revived engine reloads the flat ledger and stops for the day |
| 16:20 | `post`: the broker's record of the day (orders and fills) is written, the 1-minute SIP and IEX bars are topped up, the comparison runs |

## 3. What is recorded (all in `reports\`)

| file | content |
|---|---|
| `levpairs_fills_paper.jsonl` / `.csv` | **the order journal**: one row per order sent (entries, exits, unwinds, KILL / end-of-day closes, orphan closes), with the decision it belongs to (`trip`, `j`, `s`, `decided_at`), order id and client id, local submit / ack / seen times in ns, the broker's `submitted_at` / `filled_at`, status, fill price, quantity, the signal price and the slippage against it. The `.jsonl` is the record (an open Excel window locks the `.csv`; the engine keeps writing the `.jsonl`) |
| `levpairs_decisions_paper.jsonl` / `.csv` | every decision of every bar: s-score, half-life, prices, target, position, the text "ENTER long spread: BUY ... + SELL ...", wall-clock time |
| `levpairs_state_paper.json` | the ledger (restart-safe) |
| `levpairs_run_paper_<date>.json` | the exact configuration of the session (signal, simulation and live settings, capital) so the comparison rebuilds the same backtest |
| `levpairs_broker_orders_paper_<date>.csv`, `levpairs_broker_fills_paper_<date>.csv` | the **broker's own** record of the day, fetched after the close (the comparison trusts this over the journal) |
| `levpairs_daily_<date>.log`, `levpairs_daily_console.log` | what the job said |
| `ALERT_<date>.txt` | the watchdog's alerts (also shown as a pop-up window): **look at this file** |
| `levpairs_flatten_paper.lock`, `levpairs_watchdog_paper.json` | the cross-process closer lock (an operating-system lock: only one flatten at a time, from the engine or the watchdog, and it disappears with a dead process) and the watchdog's "I flattened this ledger" marker (a revived engine - even one that wakes up in the middle of a step - reloads the flat ledger and trades no more that day) |
| `live_compare\REPORT_<date>.md`, `<date>_trips.csv`, `<date>_legs.csv`, `<date>_decisions.csv`, `summary.csv`, `MONTH_TO_DATE.md` | the comparison, below (kept per mode: DRY RUN days get `_dry` files and never mix into the paper month) |

## 4. Reading the comparison (`reports\live_compare\REPORT_<date>.md`, `MONTH_TO_DATE.md`)

For every session it rebuilds the research backtest **twice with the session's own configuration**: on the final **SIP** bars (what the research says should happen) and on the final **IEX** bars (what the engine was fed), and lines the live trades up against them.

- **Round trips** (`<date>_trips.csv`): per trip *which product was bought and which sold* (`LONG TQQQ / SHORT QQQ`), the entry and exit bars, quantities, the four fill prices, the real fixed-share P&L in dollars and in bp of the 3x leg's notional, the twin's price-based P&L for the same trade, the difference, and the slippage of each fill against the twin's price - **judged only on comparable trips** (the same entry bar AND the same exit bar as the twin, exited in one go as many shares as entered; with IEX noise most others differ by timing, not by fill quality; they are still counted and priced in "all matched trips"). An exit split over several orders is priced with all of them. Rows are *matched*, *live only* (the engine traded, the SIP backtest did not: an IEX-noise trade or a failed entry) or *twin only* (the backtest traded, the engine did not).
- **Fills** (`<date>_legs.csv`): per order, slippage (bp, positive = worse than the reference) against (a) the signal bar's close (the optimistic backtest fill), (b) the twin's next-open price (the default backtest fill), (c) the SIP NBBO **mid** and (d) the **touch** (ask for a buy, bid for a sell) at the moment the broker accepted the order, (e) the first SIP trade after it; plus the quoted spread. This is the number that settles the "close vs next open" question for real orders.
- **Latency**: bar end -> decision (about 4 s by design: `decision_delay_s`, the wait for the IEX bar to appear), decision -> order sent, sent -> filled, and the local clock against the broker's. If the first number is much larger than 4 s the data arrived late.
- **Decisions** (`<date>_decisions.csv`): the live s-score and position after every bar against both twins: correlation, mean difference, and the share of bars on which the position agrees. On IEX signals I expect (not yet observed live) a low SIP-twin correlation (0.2-0.6) and an IEX-twin correlation near 1 (the same bars, read a few seconds after they ended: a clearly lower value would mean the bars changed after the engine read them).
- **Month to date**: totals and per-trip averages over all sessions.

What is normal, what is not: live P&L is *expected* to be lower than the SIP twin's gross (costs, slippage). A fill slippage vs the NBBO mid close to half the quoted spread is normal. Slippage vs the twin's next-open price far from 0 either way says the model's fill price is wrong. *Twin only* trades on many days = the engine is missing signals. Anything in `ALERT_<date>.txt` or an `entry failed` / `unwound` trip deserves a look at the day's log.

## 5. If something goes wrong

| situation | do |
|---|---|
| stop trading but keep the process | `python levpairs_daily.py kill` (creates `reports\KILL`; the engine exits its positions at its next decision and enters nothing; `unkill` before the next session) |
| the console of a running job | Ctrl+C: the engine flattens its ledger on the way out |
| the engine died with positions open | the watchdog alerts (and closes them if started with `--flatten "FLATTEN PAPER"`, including the legs of an entry that was in flight); or by hand after making sure no engine is running: `python levpairs_daily.py flatten --arm "ARM PAPER"` (no IEX phrase needed; `--orphans "CLOSE ORPHANS"` adds positions the ledger does not explain). The next morning's job closes any leftover by itself at 09:30:30 |
| what has it done today? | `python levpairs_daily.py status` |
| the evening job failed | `python levpairs_daily.py post --date YYYY-MM-DD --now --mode paper` (without `--mode paper` it reads the DRY RUN files; every step is independent and can be re-run) |
| a session is "BROKEN" / reconcile alarms | open the app (`launch_levpairs.bat`): Live tab > Reconcile, Clear 'broken' flags (the engine reads the broker first) |

Limits that remain (RM, `LEVPAIRS_REPORT.md` section 6): nothing here has run against the real clock or the real broker yet; Alpaca timeouts (now 5 s to connect, 20 s to read), rate-limit (429) handling, an order whose answer is lost (the engine then asks the broker for it by its client id) and paper-trading fill behaviour are unverified in real life; start watched (first 3-5 sessions, small `--capital`); an *unattended* month needs the watchdog running with `--flatten`, a PC that never sleeps, the phone notification working, and a look at `status`, `ALERT_<date>.txt` and the report every evening; RM asked for a week of clean watched sessions first.
