@echo off
REM Prompt ChartVerse-2B (GGUF Q4_K_M) through llama.cpp - much faster than the HF transformers path.
REM
REM Usage:
REM   run_chartverse_gguf.bat                                  (default chart, OHLCV attached)
REM   run_chartverse_gguf.bat <chart.png>
REM   run_chartverse_gguf.bat <chart.png> noohlcv              (image only)
REM
REM Edit chartverse_prompt.txt (long text) and re-run; the model reloads in a few seconds.
REM Prompt sent = chartverse_prompt.txt + (optionally) the sibling price_data csv.

setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "SNAP=C:\Users\jtham\.cache\huggingface\hub\models--prithivMLmods--ChartVerse-2B-GGUF\snapshots\aabfaf76f350d681e2e2203fc73749376537d653"
set "MODEL=%SNAP%\ChartVerse-2B.Q4_K_M.gguf"
set "MMPROJ=%SNAP%\ChartVerse-2B.mmproj-Q8_0.gguf"

set "CHART=%~1"
if not defined CHART set "CHART=data\resistance_dataset\val\charts\13_PTON_2024-08-26.png"
for %%F in ("%CHART%") do (
  set "STEM=%%~nF"
  set "CHARTDIR=%%~dpF"
)
set "CSV=%CHARTDIR%..\price_data\%STEM%.csv"

set "FULL=%TEMP%\chartverse_full_prompt.txt"
copy /y "chartverse_prompt.txt" "%FULL%" >nul
if /i not "%~2"=="noohlcv" if exist "%CSV%" (
  >>"%FULL%" echo.
  >>"%FULL%" echo.
  >>"%FULL%" echo OHLCV ^(date,open,high,low,close,volume^), oldest first; last row is the most recent completed bar:
  type "%CSV%" >>"%FULL%"
)

REM Tunables: sampling per Qwen3 guidance; --image-max-tokens bounds vision tokens (speed vs. price-label legibility).
REM -ngl 99 = all layers on GPU. If VRAM is tight (other apps), lower it or add --no-mmproj-offload.
llama-mtmd-cli -m "%MODEL%" --mmproj "%MMPROJ%" --image "%CHART%" -f "%FULL%" ^
  -ngl 99 -c 16384 -n 2048 --temp 0.6 --top-p 0.95 --repeat-penalty 1.05 ^
  --image-max-tokens 2048 --jinja --no-warmup
echo.
pause
