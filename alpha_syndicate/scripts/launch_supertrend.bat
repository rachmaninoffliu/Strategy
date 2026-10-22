@echo off
REM ---------------------------------------------------------------
REM  Supertrend SOXL/SOXS - Backtest + Live Trading application
REM  Launched from the desktop shortcut.
REM
REM  Uses the real CPython install rather than the WindowsApps stub,
REM  because the stub can redirect to the Microsoft Store and does not
REM  see the packages installed for this interpreter.
REM ---------------------------------------------------------------
setlocal
set PY=C:\Users\xinyu\AppData\Local\Programs\Python\Python312\python.exe
set APP=C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\scripts\supertrend_app.py

title Supertrend Trading App
cd /d C:\Users\xinyu\PycharmProjects\pythonProject\alpha_syndicate\scripts

if not exist "%PY%" (
    echo.
    echo  ERROR: Python not found at:
    echo    %PY%
    echo.
    pause
    exit /b 1
)
if not exist "%APP%" (
    echo.
    echo  ERROR: application not found at:
    echo    %APP%
    echo.
    pause
    exit /b 1
)

"%PY%" "%APP%"

if errorlevel 1 (
    echo.
    echo  The application exited with an error ^(code %errorlevel%^).
    echo  The traceback is shown above.
    echo.
    pause
)
endlocal
