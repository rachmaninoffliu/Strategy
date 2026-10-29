@echo off
REM ---------------------------------------------------------------
REM  Momentum Book - headless session (start before 08:45 ET for the opening-auction sells, at least before 10:00 for the first intraday check; stops 16:10 ET).
REM  DRY RUN by default: plans and journals every check and decision, sends nothing.
REM  It trades live_model.json in this folder (written when the GUI's live loop starts - the model you backtested, with its shadow - or File > Save model).
REM  Keys: 'Alpaca Kay.txt' in the project folder; MB_ALPACA_API_KEY / MB_ALPACA_SECRET_KEY select a separate paper account.
REM  To send orders use the ARMED line instead.
REM ---------------------------------------------------------------
setlocal
set PY=C:\Users\xinyu\AppData\Local\Programs\Python\Python312\python.exe
cd /d "%~dp0"
"%PY%" run_live.py
REM "%PY%" run_live.py --arm "ARM PAPER"
endlocal
