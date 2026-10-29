@echo off
REM ---------------------------------------------------------------
REM  The daily job of the month-long PAPER test (levpairs_daily.py day): waits for 09:25 ET, trades the session,
REM  flattens, waits for 16:20 ET, then syncs the broker's record, tops up the bars and writes the comparison report.
REM  Schedule THIS file with Windows Task Scheduler (weekdays, any local time BEFORE 09:25 ET - it waits for the open itself).
REM
REM  It is a DRY RUN until you edit ARGS below. To send orders to the PAPER account use the second line:
REM  (the IEX phrase is needed because this account can only read real-time IEX data - see LIVE_TEST_RUNBOOK.md)
REM ---------------------------------------------------------------
setlocal
set PY=C:\Users\xinyu\AppData\Local\Programs\Python\Python312\python.exe
set HERE=%~dp0
set ARGS=day
REM set ARGS=day --arm "ARM PAPER" --accept-iex "ACCEPT IEX" --capital 20000

title Lev-ETF pairs - daily paper job
cd /d "%HERE%scripts"
"%PY%" levpairs_daily.py %ARGS% >> "%HERE%reports\levpairs_daily_console.log" 2>&1
endlocal
