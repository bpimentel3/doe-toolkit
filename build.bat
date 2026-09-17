@echo off
REM DOE Toolkit - Build Script (conda-pack approach)
REM
REM What this does:
REM   1. Packs the doe-toolkit conda environment into a self-contained folder
REM   2. Copies app source files
REM   3. Produces a dist\DOE-Toolkit folder ready to zip and distribute
REM
REM Requirements:
REM   - conda-pack installed in base environment (conda install conda-pack)
REM   - doe-toolkit conda environment exists with all dependencies

setlocal

set ENV_NAME=doe-toolkit
set OUTPUT_DIR=dist\DOE-Toolkit
set PACK_FILE=dist\doe-toolkit-env.tar.gz

echo ============================================================
echo  DOE Toolkit - Build Script
echo ============================================================
echo.

REM ── Step 1: Clean previous build ──────────────────────────────
echo [1/5] Cleaning previous build...
if exist dist (
    rmdir /s /q dist
    if errorlevel 1 (
        echo ERROR: Could not delete dist\ folder.
        echo Close any running instances of DOE-Toolkit and try again.
        pause
        exit /b 1
    )
)
mkdir dist
mkdir "%OUTPUT_DIR%"
echo       Done.
echo.

REM ── Step 2: Pack the conda environment ────────────────────────
echo [2/6] Packing conda environment "%ENV_NAME%"...
echo       This takes 3-8 minutes on first run.
echo.
conda-pack -n %ENV_NAME% -o "%PACK_FILE%" --ignore-missing-files
if errorlevel 1 (
    echo.
    echo ERROR: conda-pack failed.
    echo Make sure the "%ENV_NAME%" environment exists:
    echo   conda env list
    echo.
    echo If conda-pack is not installed in base:
    echo   conda install conda-pack
    pause
    exit /b 1
)
echo.
echo       Pack complete.
echo.

REM ── Step 3: Extract the environment into the output folder ─────
echo [3/6] Extracting environment into %OUTPUT_DIR%\env ...
mkdir "%OUTPUT_DIR%\env"
tar -xzf "%PACK_FILE%" -C "%OUTPUT_DIR%\env"
if errorlevel 1 (
    echo ERROR: Failed to extract environment.
    echo Make sure 'tar' is available (Windows 10 build 17063 or later).
    pause
    exit /b 1
)
echo       Extraction complete.
echo.

REM ── Step 4: Unpack the conda environment (fixes shebangs etc.) ─
echo [4/6] Finalising environment...
"%OUTPUT_DIR%\env\Scripts\conda-unpack.exe"
if errorlevel 1 (
    echo WARNING: conda-unpack returned an error. Continuing anyway.
)
echo       Done.
echo.

REM ── Step 5: Copy application source and launcher ──────────────
echo [5/6] Copying application files...

REM Copy source code
xcopy /e /i /q src "%OUTPUT_DIR%\src"

REM Copy Streamlit config if present
if exist .streamlit (
    xcopy /e /i /q .streamlit "%OUTPUT_DIR%\.streamlit"
)

REM Copy the user-facing launcher
copy DOE-Toolkit.bat "%OUTPUT_DIR%\DOE-Toolkit.bat"

REM Copy supporting docs
if exist LICENSE.txt   copy LICENSE.txt   "%OUTPUT_DIR%\LICENSE.txt"
if exist QUICKSTART.md copy QUICKSTART.md "%OUTPUT_DIR%\QUICKSTART.md"

echo       Done.
echo.

REM ── Step 6: Generate third-party license notices ──────────────
echo [6/6] Generating third-party license notices...
if exist "%OUTPUT_DIR%\env" (
    python "%~dp0tools\license_audit.py" --env "%OUTPUT_DIR%\env" --emit-notices
    if errorlevel 1 (
        echo       WARNING: license audit flagged components; continuing build.
    )
    if exist "%OUTPUT_DIR%\env\THIRD_PARTY_NOTICES.txt" (
        copy /y "%OUTPUT_DIR%\env\THIRD_PARTY_NOTICES.txt" "%OUTPUT_DIR%\THIRD_PARTY_NOTICES.txt" >nul
    ) else (
        echo       WARNING: THIRD_PARTY_NOTICES.txt not generated.
    )
)
echo       Done.
echo.

REM ── Clean up the intermediate tar file ────────────────────────
del "%PACK_FILE%"

REM ── Summary ───────────────────────────────────────────────────
echo ============================================================
echo  BUILD SUCCESSFUL
echo ============================================================
echo.
echo  Output folder : %OUTPUT_DIR%
echo.
echo  To test:
echo    cd %OUTPUT_DIR%
echo    DOE-Toolkit.bat
echo.
echo  To distribute:
echo    Zip the entire %OUTPUT_DIR% folder.
echo    Users extract and double-click DOE-Toolkit.bat.
echo.
echo  Approximate size: 500-700 MB uncompressed, ~150 MB zipped.
echo ============================================================
echo.
pause