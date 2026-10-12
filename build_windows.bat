@echo off
setlocal

echo ===============================================
echo   Building Anima Studio.exe
echo ===============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python wasn't found on PATH. Install Python 3.10+ from python.org first.
    pause
    exit /b 1
)

echo Installing/checking dependencies (this can take a few minutes the first time)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency install failed - see the errors above.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo   Installing the bundled training backend's Python
echo   environment (torch, xformers, CUDA wheels, DeepSpeed, etc.)...
echo ===============================================
pushd training_backend
python installer.py local
if errorlevel 1 (
    echo.
    echo Backend install failed - see the errors above.
    popd
    pause
    exit /b 1
)
popd
echo Backend install finished.
echo.

echo.
echo Running PyInstaller...
python -m PyInstaller anima_studio.spec --noconfirm
if errorlevel 1 (
    echo.
    echo Build failed - see the errors above.
    pause
    exit /b 1
)

echo.
echo ===============================================
echo   Done. Your app is at:
echo   dist\Anima Studio\Anima Studio.exe
echo ===============================================
pause
