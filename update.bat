@echo off
setlocal
cd /d "%~dp0"
REM Updates Anima Studio's code from your GitHub repository. Your data, projects, models
REM and installed environments are not touched; a backup of the old code is kept in data\update_backups.
REM   update.bat                 use the repository saved in App settings
REM   update.bat owner/name      set the repository first (e.g. update.bat yourname/anima-studio)
REM   update.bat --check         only look for an update
REM   update.bat --main          follow the latest commit on main instead of releases

if not exist ".venv\Scripts\python.exe" (
    echo Run run.bat once first so the app is set up, then try again.
    pause
    exit /b 1
)

echo Close Anima Studio before updating (files in use cannot be replaced).
tasklist /fi "imagename eq pythonw.exe" 2>nul | find /i "pythonw.exe" >nul
if not errorlevel 1 (
    echo.
    echo Anima Studio looks like it is still running. Close it, then press a key.
    pause >nul
)

".venv\Scripts\python.exe" update.py %*
set RC=%errorlevel%
echo.
if "%RC%"=="0" (echo Done. Start the app with run.bat.) else (echo Update did not finish - see above.)
pause
exit /b %RC%
