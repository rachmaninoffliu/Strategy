@echo off
REM ---------------------------------------------------------------
REM  Mean Reversion - headless session (start before 15:30 ET, stops 16:10 ET). DRY RUN by default: plans and journals the decision, sends nothing.
REM  It trades live_model.json in this folder (written when the GUI's live loop starts - the model you backtested - or File > Save model).
REM  Keys: 'Alpaca Key.txt' in the project folder; MR_ALPACA_API_KEY / MR_ALPACA_SECRET_KEY select a separate paper account.
REM  To send orders use the ARMED line instead.
REM ---------------------------------------------------------------
setlocal
set PY=C:\Users\xinyu\AppData\Local\Programs\Python\Python312\python.exe
cd /d "%~dp0"
"%PY%" run_live.py
REM "%PY%" run_live.py --arm "ARM PAPER"
endlocal
