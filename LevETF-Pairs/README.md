# LevETF-Pairs - statistical arbitrage on TQQQ/QQQ, SOXL/SMH, SPXL/SPY

The Avellaneda-Lee residual / OU s-score construction (as in `../Stats-arb`) applied to each 3x ETF against its 1x ETF on 1, 5, 15 and 30-minute bars, with a PyQt6 app (backtest + live, same look and feel
as the other strategies) and an Alpaca paper / live engine.

**Result in one line:** with fills at the open of the bar after the signal and quoted-spread costs, all 1,458 rule sets lose money on every time frame; the gross reversion (under 1 bp per trade) is far smaller than the
5-11 bp a pair round trip costs. The write-up is `LEVPAIRS_REPORT.md`.

```
launch_levpairs.bat                  the app (always starts on PAPER + DRY RUN)
scripts/levpairs_data.py             Alpaca 1-minute bars (SIP + IEX), resampling, session grid, quoted-spread sample      python levpairs_data.py [iex|spreads]
scripts/levpairs_core.py             stage A: spread, rolling beta, rolling OU fit, s-score (vectorised)
scripts/levpairs_sim.py              stage B: the bar-by-bar state machine, costs on both legs, portfolio, vol target, metrics
scripts/levpairs_bridge.py           cached signals, one-call backtest, time-frame scan, ablation, cost / threshold checks
scripts/levpairs_live.py             session engine: DRY RUN shadow book or ARMED two-leg market orders, KILL, reconcile, replay
scripts/levpairs_ui/                 settings, live, backtest, signals, help tabs, presets
scripts/levpairs_scan.py             the 1,458-rule-set research scan (in-sample 2017-2021 ranks, 2022-2026 shown)
scripts/levpairs_feed_test.py        IEX signals vs SIP signals (the account cannot read real-time SIP)
scripts/levpairs_report.py           the tables behind the report
scripts/levpairs_compare.py          live vs backtest: the day's orders against the SIP and IEX backtests of the same session, fills vs NBBO, decisions, latency -> reports/live_compare/
scripts/levpairs_daily.py            the headless daily job of the month-long paper test: day / run / post / status / kill / flatten
scripts/levpairs_watchdog.py         an independent process that alerts (or closes) when the engine is dead with positions open
scripts/levpairs_notify.py           push notifications to your phone (set LEVPAIRS_NOTIFY_URL, ntfy style)
run_levpairs_day.bat / run_levpairs_watchdog.bat   what Windows Task Scheduler runs;   LIVE_TEST_RUNBOOK.md   setup, daily flow, files, how to read the comparison, emergencies
scripts/test_levpairs.py             112 tests (signals, rules, accounting, live engine with a fake broker incl. the RM failure scenarios, GUI offscreen);
scripts/test_levpairs_compare.py     12 tests, test_levpairs_ops.py 23 tests (daily job, watchdog, a whole session engine -> journal -> comparison)
data/                                cached bars   reports/   scan CSVs, logs, state files, cache
```

Safety: PAPER + DRY RUN by default; ARM needs a tick (a real-money account also a typed phrase); both legs are sent at the same moment, both confirmed, and a failed leg unwinds the other; everything is flat before the close; KILL flattens the
engine's own ledger only; orders are tagged `LP-…`. **Valid has not signed off.**
