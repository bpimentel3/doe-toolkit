# DOE Toolkit - Build Script (conda-pack approach)
#
# What this does:
#   1. Packs the doe-toolkit conda environment into a self-contained folder
#   2. Copies app source files
#   3. Produces a dist\DOE-Toolkit folder ready to zip and distribute
#
# Requirements:
#   - conda-pack installed in base environment (conda install conda-pack)
#   - doe-toolkit conda environment exists with all dependencies

$ErrorActionPreference = "Stop"

$EnvName   = "doe-toolkit"
# OPTIONAL: absolute path to the doe-toolkit conda environment.
# Leave as $null normally — the environment is located by name on any machine.
# Set this only if conda cannot find the environment by name here.
$EnvPath   = $null
$OutputDir = "dist\DOE-Toolkit"
$PackFile  = "dist\doe-toolkit-env.tar.gz"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DOE Toolkit - Build Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Clean previous build ──────────────────────────────────────
Write-Host "[1/6] Cleaning previous build..." -ForegroundColor Yellow
if (Test-Path "dist") {
    try {
        Remove-Item -Recurse -Force "dist" -ErrorAction Stop
    } catch {
        Write-Host ""
        Write-Host "ERROR: Could not delete dist\ folder." -ForegroundColor Red
        Write-Host "Close any running instances of DOE-Toolkit and try again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null
Write-Host "      Done." -ForegroundColor Green
Write-Host ""

# ── Step 2: Pack the conda environment ────────────────────────────────
Write-Host "[2/6] Packing conda environment '$EnvName'..." -ForegroundColor Yellow
Write-Host "      This takes 3-8 minutes on first run." -ForegroundColor Gray
Write-Host ""

# Locate the environment by name first (works on any machine).
conda-pack -n $EnvName -o $PackFile --ignore-missing-files

# Fallback: if the name lookup failed, retry with an explicitly
# configured absolute path (set $EnvPath at the top of this script).
if ($LASTEXITCODE -ne 0 -and $EnvPath -and (Test-Path $EnvPath)) {
    Write-Host "      Environment '$EnvName' not found by name; retrying with configured path..." -ForegroundColor Yellow
    conda-pack -p $EnvPath -o $PackFile --ignore-missing-files
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: conda-pack failed." -ForegroundColor Red
    Write-Host "The '$EnvName' environment could not be found. Verify it exists:" -ForegroundColor Yellow
    Write-Host "  conda env list" -ForegroundColor Gray
    Write-Host ""
    Write-Host "If the environment lives in a non-standard location, set `$EnvPath at the top of this script." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "If conda-pack is not installed in base:" -ForegroundColor Yellow
    Write-Host "  conda install conda-pack" -ForegroundColor Gray
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""
Write-Host "      Pack complete." -ForegroundColor Green
Write-Host ""

# ── Step 3: Extract the environment ───────────────────────────────────
Write-Host "[3/6] Extracting environment into $OutputDir\env ..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "$OutputDir\env" | Out-Null

tar -xzf $PackFile -C "$OutputDir\env"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to extract environment." -ForegroundColor Red
    Write-Host "Make sure 'tar' is available (Windows 10 build 17063 or later)." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "      Extraction complete." -ForegroundColor Green
Write-Host ""

# ── Step 4: Unpack (fix shebangs/paths inside the env) ────────────────
Write-Host "[4/6] Finalising environment..." -ForegroundColor Yellow
& "$OutputDir\env\Scripts\conda-unpack.exe"
if ($LASTEXITCODE -ne 0) {
    Write-Host "      WARNING: conda-unpack returned an error. Continuing anyway." -ForegroundColor Yellow
} else {
    Write-Host "      Done." -ForegroundColor Green
}
Write-Host ""

# ── Step 5: Copy application source and launcher ──────────────────────
Write-Host "[5/6] Copying application files..." -ForegroundColor Yellow

# Source code
Copy-Item -Recurse -Force "src" "$OutputDir\src"

# Remove __pycache__ directories and *.tmp files from copied source
Get-ChildItem -Path "$OutputDir\src" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path "$OutputDir\src" -Recurse -File -Filter "*.tmp" | Remove-Item -Force

# Streamlit config
if (Test-Path ".streamlit") {
    Copy-Item -Recurse -Force ".streamlit" "$OutputDir\.streamlit"
}

# User-facing launcher
Copy-Item -Force "DOE-Toolkit.bat" "$OutputDir\DOE-Toolkit.bat"

# Supporting docs
foreach ($doc in @("LICENSE.txt", "QUICKSTART.md")) {
    if (Test-Path $doc) { Copy-Item -Force $doc "$OutputDir\$doc" }
}

Write-Host "      Done." -ForegroundColor Green
Write-Host ""

# ── Step 6: Generate third-party license notices ──────────────────────
Write-Host "[6/6] Generating third-party license notices..." -ForegroundColor Yellow

if (Test-Path "$OutputDir\env") {
    python "$PSScriptRoot\tools\license_audit.py" --env "$OutputDir\env" --emit-notices
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      WARNING: license audit flagged components; continuing build." -ForegroundColor Yellow
    }
    if (Test-Path "$OutputDir\env\THIRD_PARTY_NOTICES.txt") {
        Copy-Item -Force "$OutputDir\env\THIRD_PARTY_NOTICES.txt" "$OutputDir\THIRD_PARTY_NOTICES.txt"
    } else {
        Write-Host "      WARNING: THIRD_PARTY_NOTICES.txt not generated." -ForegroundColor Yellow
    }
}
Write-Host "      Done." -ForegroundColor Green
Write-Host ""

# ── Clean up intermediate tar ─────────────────────────────────────────
Remove-Item -Force $PackFile

# ── Summary ───────────────────────────────────────────────────────────
Write-Host "============================================================" -ForegroundColor Green
Write-Host " BUILD SUCCESSFUL" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Output folder : $OutputDir" -ForegroundColor Cyan
Write-Host ""
Write-Host " To test:" -ForegroundColor Yellow
Write-Host "   cd $OutputDir" -ForegroundColor Gray
Write-Host "   .\DOE-Toolkit.bat" -ForegroundColor Gray
Write-Host ""
Write-Host " To distribute:" -ForegroundColor Yellow
Write-Host "   Zip the entire $OutputDir folder." -ForegroundColor Gray
Write-Host "   Users extract and double-click DOE-Toolkit.bat." -ForegroundColor Gray
Write-Host ""
Write-Host " Approximate size: 500-700 MB uncompressed, ~150 MB zipped." -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"