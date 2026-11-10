@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ from https://python.org
    echo and make sure "Add python.exe to PATH" is checked during install.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment in .venv ...
    python -m venv .venv
)

REM Dependencies are installed only once (first run). To reinstall/update later,
REM delete .venv\.deps_installed or run:  run.bat --update
set NEED_INSTALL=0
if not exist ".venv\.deps_installed" set NEED_INSTALL=1
if /i "%~1"=="--update" set NEED_INSTALL=1

REM Slow connections: pip's default 15s timeout is too short. For a mirror, run first:
REM   set PIP_INDEX_URL=https://pypi.org/simple   (or any mirror you can reach)
REM Set outside the if-block below: %VAR% inside a block expands before "set" runs.
set PIP_OPTS=--default-timeout=120 --retries 10
if "%NEED_INSTALL%"=="1" (
    echo Installing dependencies ...
    ".venv\Scripts\python.exe" -m pip install %PIP_OPTS% -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Dependency install failed - see the error above.
        pause
        exit /b 1
    )
    echo ok> ".venv\.deps_installed"
)

echo.
echo Starting Anima Studio ...
REM pythonw.exe (not python.exe) runs with no console window at all, and
REM "start" launches it as its own detached process - so this setup
REM window closes itself right after, instead of a command prompt
REM staying open behind the app for as long as it's running.
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" desktop.py
) else (
    start "" pythonw desktop.py
)
