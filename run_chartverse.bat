@echo off
REM Prompt the local ChartVerse-2B VLM with one of the resistance_dataset charts.
REM
REM Usage:
REM   run_chartverse.bat                                             (default chart below)
REM   run_chartverse.bat data\resistance_dataset\val\charts\14_RIOT_2024-02-27.png
REM   run_chartverse.bat "C:\full\path\to\chart.png"
REM
REM The instructions live in chartverse_prompt.txt (long text format) - edit that file
REM freely, then type /retry in the session to re-run the same chart with the new prompt
REM without reloading the model. Follow-up questions about the current chart are also
REM free: just type them at the ">" prompt.

setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "PY=C:\data\envs\trading_framework\Scripts\python.exe"
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "CHART=%~1"
if not defined CHART set "CHART=data\resistance_dataset\val\charts\13_PTON_2024-08-26.png"

REM Extra chartverse_cli.py flags, e.g.:  set "EXTRA=--no-chat --max-new-tokens 4096"
REM Add --no-ohlcv here to send only the image (no OHLCV table attached).
set "EXTRA="

"%PY%" "%ROOT%chartverse_cli.py" --image "%CHART%" --prompt-file "%ROOT%chartverse_prompt.txt" --ohlcv auto %EXTRA%
echo.
pause
