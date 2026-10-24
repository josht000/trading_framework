@echo off
setlocal
cd /d "%~dp0.."
set PROFILE_ARG=
if /i "%1"=="profile" set PROFILE_ARG=--profile
python run.py --strategy atr_pivot %PROFILE_ARG%
