@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title BTC 15-min Kalshi trader

rem ---------------------------------------------------------------------------------------------------------------
rem  Finds a REAL Python that has the packages the app needs and starts trader_app.py with it.
rem  Why not just "python": on this PC the first "python" on the PATH is the Microsoft Store alias in
rem  ...\WindowsApps, which is only a stub ("Python was not found") - it is skipped here.
rem  To force one interpreter: set the environment variable TRADER_PYTHON to its full path.
rem  Test only (no window):    run_trader.bat --check
rem ---------------------------------------------------------------------------------------------------------------
set "PY="
set "TRIED="

if defined TRADER_PYTHON call :try "%TRADER_PYTHON%"

rem 1) the Windows "py" launcher (the usual way to reach the right Python on Windows)
if not defined PY (
    for /f "usebackq delims=" %%P in (`py -3 -c "import sys; print(sys.executable)" 2^>nul`) do if not defined PY call :try "%%P"
)

rem 2) the standard per-user installs, newest first
if not defined PY (
    for /f "delims=" %%D in ('dir /b /ad /o-n "%LOCALAPPDATA%\Programs\Python\Python3*" 2^>nul') do if not defined PY call :try "%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
)

rem 3) Anaconda / Miniconda in the usual places
if not defined PY call :try "%USERPROFILE%\anaconda3\python.exe"
if not defined PY call :try "%USERPROFILE%\miniconda3\python.exe"
if not defined PY call :try "%ProgramData%\anaconda3\python.exe"

rem 4) whatever "python" is on the PATH - except the Microsoft Store stub
if not defined PY (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY (
        echo %%P | find /i "\WindowsApps\" >nul || call :try "%%P"
    )
)

if not defined PY goto :nopython

if /i "%~1"=="--check" (
    echo Using: %PY%
    exit /b 0
)

"%PY%" trader_app.py
if errorlevel 1 (
    echo.
    echo The app stopped with an error ^(see above^). Interpreter used: %PY%
    pause
)
exit /b %errorlevel%


:try
rem  usage: call :try "full\path\to\python.exe"  - accepts it only if it can import everything the app needs
if not exist "%~1" exit /b 1
set "TRIED=%TRIED%  %~1"
"%~1" -c "import PyQt6, pandas, numpy, matplotlib, requests, cryptography" >nul 2>&1
if errorlevel 1 exit /b 1
set "PY=%~1"
exit /b 0


:nopython
echo.
echo  No usable Python was found - the app needs Python 3.10+ with these packages:
echo      PyQt6  pandas  numpy  matplotlib  requests  cryptography
echo.
if defined TRIED (
    echo  Pythons that were found but are missing packages:
    echo     %TRIED%
    echo.
    echo  Fix: open a Command Prompt and run   py -3 -m pip install PyQt6 pandas numpy matplotlib requests cryptography
) else (
    echo  No Python was found at all. Install Python 3.12 from https://www.python.org/downloads/
    echo  ^(tick "Add python.exe to PATH" in the installer^), then run:
    echo      py -3 -m pip install PyQt6 pandas numpy matplotlib requests cryptography
)
echo.
echo  If Python is installed somewhere unusual, set TRADER_PYTHON to its full path and run this file again.
echo.
pause
exit /b 9009
