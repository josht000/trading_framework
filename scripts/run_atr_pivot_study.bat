@echo off
setlocal
set PY=C:\data\envs\trading_framework\Scripts\python.exe
cd /d "%~dp0.."
set RESULTS_TAG=lb13
set LOOKBACK_BARS=273

echo === main.py (backtest) ===
"%PY%" main.py || goto :fail
echo === rank_test.py (dev ranking experiments) ===
"%PY%" rank_test.py || goto :fail
echo === rank_final.py --charts ===
"%PY%" rank_final.py --charts || goto :fail
echo === partition_report.py ===
"%PY%" scripts\partition_report.py || goto :fail

if /i "%1"=="profile" (
  echo === main.py with cProfile ===
  set PROFILE=1
  "%PY%" main.py || goto :fail
)
echo Timing JSON/prof files: results\timing
exit /b 0

:fail
echo FAILED with errorlevel %errorlevel%
exit /b %errorlevel%
