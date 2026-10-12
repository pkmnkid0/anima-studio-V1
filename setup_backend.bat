@echo off
setlocal

REM ==========================================================================
REM  The actual training engine Anima Studio's Train tab drives ships bundled
REM  already at training_backend\ (see THIRD_PARTY_NOTICES.md) - nothing to
REM  clone here anymore. This just runs ITS installer, which pulls in PyTorch,
REM  xformers, and the rest of the ML training stack into its own virtual
REM  environment (separate from Anima Studio's own lightweight dependencies).
REM
REM  Prefer doing this from the Train tab's "Set up backend" card instead -
REM  same installer, same output, but no terminal needed. This script is here
REM  for anyone who'd rather run it directly.
REM ==========================================================================

cd /d "%~dp0training_backend"

if not exist "install.bat" (
    echo training_backend\install.bat not found - is this a fresh checkout?
    echo If training_backend\ is missing entirely, redownload Anima Studio.
    pause
    exit /b 1
)

echo Running the bundled backend's installer. This downloads PyTorch and the
echo rest of the ML training stack, so it can take a while.
echo The first question it asks is "are you using this remotely? (y/n)" -
echo answer n (you're running this locally, even though that phrasing makes
echo it easy to answer backwards). Answering y sends you down a different
echo path meant for remote/cloud setups (ngrok tunnels, etc) that Anima
echo Studio doesn't use.
echo.

call install.bat

echo.
echo ==========================================================================
echo Done (or the installer above hit a prompt/error - scroll up to check).
echo.
echo To start training from Anima Studio:
echo   1. Start the backend server - run training_backend\run.bat, or use the
echo      Train tab's "Launch backend" button instead.
echo   2. In Anima Studio's Train tab, set "Backend URL" to
echo      http://127.0.0.1:8000 (the default) and click "Check connection".
echo   3. Fill in your settings and click "Start training".
echo ==========================================================================
pause
