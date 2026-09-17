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

REM === Step 1: Clean previous build ===
echo [1/6] Cleaning previous build...
if not exist dist goto :step1_clean_done
rmdir /s /q dist
if errorlevel 1 goto :step1_failed
:step1_clean_done
mkdir dist
mkdir "%OUTPUT_DIR%"
echo       Done.
echo.

REM === Step 2: Verify conda-pack, then pack the conda environment ===
echo [2/6] Packing conda environment "%ENV_NAME%"...
echo       Checking conda-pack version (0.9.2+ required)...
python "%~dp0tools\check_conda_pack.py"
if errorlevel 1 goto :pack_version_failed
echo       This takes 3-8 minutes on first run.
echo.
conda-pack -n "%ENV_NAME%" -o "%PACK_FILE%" --ignore-missing-files
if errorlevel 1 goto :pack_failed
echo.
echo       Pack complete.
echo.

REM === Step 3: Extract the environment into the output folder ===
echo [3/6] Extracting environment into "%OUTPUT_DIR%\env" ...
mkdir "%OUTPUT_DIR%\env"
tar -xzf "%PACK_FILE%" -C "%OUTPUT_DIR%\env"
if errorlevel 1 goto :extract_failed
echo       Extraction complete.
echo.

REM === Step 4: Unpack the conda environment (fixes shebangs etc.) ===
echo [4/6] Finalising environment...
"%OUTPUT_DIR%\env\Scripts\conda-unpack.exe"
if errorlevel 1 echo WARNING: conda-unpack returned an error. Continuing anyway.
echo       Done.
echo.

REM === Step 5: Copy application source and launcher ===
echo [5/6] Copying application files...
xcopy /e /i /q src "%OUTPUT_DIR%\src"
if exist .streamlit xcopy /e /i /q .streamlit "%OUTPUT_DIR%\.streamlit"
copy DOE-Toolkit.bat "%OUTPUT_DIR%\DOE-Toolkit.bat"
if exist LICENSE.txt   copy LICENSE.txt   "%OUTPUT_DIR%\LICENSE.txt"
if exist QUICKSTART.md copy QUICKSTART.md "%OUTPUT_DIR%\QUICKSTART.md"
echo       Done.
echo.

REM === Step 6: Generate third-party license notices ===
echo [6/6] Generating third-party license notices...
if not exist "%OUTPUT_DIR%\env" goto :step6_no_notice
python "%~dp0tools\license_audit.py" --env "%OUTPUT_DIR%\env" --emit-notices
if errorlevel 1 echo       WARNING: license audit flagged components; continuing build.
if not exist "%OUTPUT_DIR%\env\THIRD_PARTY_NOTICES.txt" goto :step6_no_notice
copy /y "%OUTPUT_DIR%\env\THIRD_PARTY_NOTICES.txt" "%OUTPUT_DIR%\THIRD_PARTY_NOTICES.txt" >nul
if not exist "%OUTPUT_DIR%\THIRD_PARTY_NOTICES.txt" echo       WARNING: could not copy notices file.
goto :step6_done
:step6_no_notice
echo       WARNING: THIRD_PARTY_NOTICES.txt not generated.
:step6_done
echo       Done.
echo.

REM === Validate the packed environment ===
echo [6/6] Validating packed environment...
"%OUTPUT_DIR%\env\python.exe" -c "import streamlit.watcher.util as u; assert u._WINDOWS_EXTENDED_PATH_PREFIX == '\\\\?\\', 'streamlit util.py corrupted during packaging'"
if errorlevel 1 goto :verify_failed
echo       Packed environment OK.
echo.

REM === Clean up the intermediate tar file ===
del "%PACK_FILE%"

REM === Summary ===
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
exit /b 0

REM === Error handlers (top-level labels only) ===
:step1_failed
echo ERROR: Could not delete dist\ folder.
echo Close any running instances of DOE-Toolkit and try again.
pause
exit /b 1

:pack_failed
echo ERROR: conda-pack failed.
echo Make sure the "%ENV_NAME%" environment exists:
echo   conda env list
echo.
echo If conda-pack is not installed in base:
echo   conda install conda-pack
pause
exit /b 1

:pack_version_failed
echo ERROR: conda-pack too old or missing.
echo This build requires conda-pack 0.9.2+.
echo Older versions corrupt Python source files in the packaged
echo environment on Windows (streamlit breaks on launch).
echo.
echo Upgrade with:
echo   conda install -n base -c conda-forge conda-pack=0.9.2 --freeze-installed
pause
exit /b 1

:verify_failed
echo ERROR: packed environment validation failed.
echo streamlit was corrupted during packaging - the known conda-pack
echo prefix-rewrite bug on Windows. Ensure conda-pack is 0.9.2+ in base,
echo then rebuild.
pause
exit /b 1

:extract_failed
echo ERROR: Failed to extract environment.
echo Make sure 'tar' is available (Windows 10 build 17063 or later).
pause
exit /b 1
