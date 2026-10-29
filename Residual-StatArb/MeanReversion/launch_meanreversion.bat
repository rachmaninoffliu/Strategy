@echo off
REM ---------------------------------------------------------------
REM  Mean Reversion - Backtest + Live trading application
REM  Always starts on Paper + DRY RUN. Uses the real CPython install (not the WindowsApps stub).
REM ---------------------------------------------------------------
setlocal
set PY=C:\Users\xinyu\AppData\Local\Programs\Python\Python312\python.exe
set HERE=%~dp0
set APP=%HERE%app.py

title Mean Reversion
cd /d "%HERE%"

if not exist "%PY%" (
    echo.
    echo  ERROR: Python not found at: %PY%
    echo.
    pause
    exit /b 1
)

"%PY%" "%APP%"

if errorlevel 1 (
    echo.
    echo  The application exited with an error ^(code %errorlevel%^). The traceback is shown above.
    echo.
    pause
)
endlocal
