@echo off
REM DOE Toolkit Launcher
REM Double-click this file to start the application.

REM Set working directory to the folder containing this script.
REM This ensures src/ is found regardless of where the user extracted the zip.
cd /d "%~dp0"

REM Check the bundled environment exists
if not exist "env\python.exe" (
    echo ERROR: Bundled environment not found.
    echo Please re-download and extract DOE-Toolkit again.
    pause
    exit /b 1
)

REM Check app source exists
if not exist "src\ui\app.py" (
    echo ERROR: Application source not found.
    echo Please re-download and extract DOE-Toolkit again.
    pause
    exit /b 1
)

echo ============================================================
echo  DOE Toolkit - Design of Experiments Software
echo ============================================================
echo.
echo  Starting... your browser will open automatically.
echo  To stop the app, close this window.
echo.
echo ============================================================

REM Wait briefly then open browser once before launching Streamlit.
REM Using headless mode to prevent Streamlit from opening its own browser tab.
timeout /t 2 /nobreak >nul
start "" http://localhost:8501

REM Launch Streamlit using the bundled environment.
REM Run via "python -m streamlit" instead of the pip Scripts\streamlit.exe
REM launcher: pip shims embed the build machine's absolute interpreter path
REM and fail with "Fatal error in launcher" anywhere that path doesn't exist.
"env\python.exe" -m streamlit run src\ui\app.py ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --server.enableCORS false ^
    --theme.base light