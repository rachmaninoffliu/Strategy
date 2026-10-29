@echo off
REM ---------------------------------------------------------------
REM  The independent watchdog (levpairs_watchdog.py): looks every 20 s at the engine's heartbeat and at what the PAPER account holds.
REM  Schedule it with Windows Task Scheduler (weekdays, any local time BEFORE 09:30 ET); it ends itself at 16:10 ET.
REM  It only ALERTS (pop-up, reports\ALERT_<date>.txt, and a push to your phone if LEVPAIRS_NOTIFY_URL is set) until you edit ARGS.
REM
REM  ARMED days with a dedicated paper account (recommended once the dry rehearsal is done):  --flatten "FLATTEN PAPER"
REM     (it closes the ledger positions of a dead engine and the legs of an entry that was in flight; it needs the project's own keys:
REM      'Alpaca Key.txt' in the project folder or ALPACA_API_KEY / ALPACA_SECRET_KEY, or --shared-keys "SHARED ACCOUNT")
REM  DRY RUN days, when no engine is meant to trade:                                          --no-engine-expected
REM  Leave --close-orphans "CLOSE ORPHANS" OFF: the armed day job already closes unexplained positions in the six ETFs at 15:52.
REM ---------------------------------------------------------------
setlocal
set PY=C:\Users\xinyu\AppData\Local\Programs\Python\Python312\python.exe
set HERE=%~dp0
set ARGS=--no-engine-expected
REM set ARGS=--flatten "FLATTEN PAPER"

title Lev-ETF pairs - watchdog
cd /d "%HERE%scripts"
"%PY%" levpairs_watchdog.py %ARGS% >> "%HERE%reports\levpairs_watchdog_console.log" 2>&1
endlocal
