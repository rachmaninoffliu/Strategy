@echo off
REM ---------------------------------------------------------------
REM  Residual Prediction - headless session (09:00 - 16:05 ET). DRY RUN by default: plans and journals every bar, sends nothing.
REM  To trade its OWN paper account: set RESPRED_ALPACA_API_KEY and RESPRED_ALPACA_SECRET_KEY (setx, new console), save the GUI settings
REM  (File > Save settings -> ResPrediction_settings.json in this folder) and use the ARMED line instead.
REM ---------------------------------------------------------------
setlocal
set PY=C:\Users\xinyu\AppData\Local\Programs\Python\Python312\python.exe
cd /d "%~dp0"
"%PY%" run_live.py
REM "%PY%" run_live.py --arm "ARM PAPER"
endlocal
