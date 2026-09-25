# DOE Toolkit - Quick Rebuild Script
#
# Use this after source-only changes (no dependency updates).
# Skips conda-pack entirely — only re-copies src/, .streamlit/, and DOE-Toolkit.bat.
#
# Use build.ps1 instead if you have added or updated conda dependencies.

$ErrorActionPreference = "Stop"

$OutputDir = "dist\DOE-Toolkit"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DOE Toolkit - Quick Rebuild (source only)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Verify a prior full build exists ──────────────────────────────────
if (-not (Test-Path "$OutputDir\env\python.exe")) {
    Write-Host "ERROR: No existing build found at $OutputDir." -ForegroundColor Red
    Write-Host "Run build.ps1 first to create a full build." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# ── Re-copy source ─────────────────────────────────────────────────────
Write-Host "[1/3] Updating source files..." -ForegroundColor Yellow

Remove-Item -Recurse -Force "$OutputDir\src"
Copy-Item -Recurse -Force "src" "$OutputDir\src"

Get-ChildItem -Path "$OutputDir\src" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path "$OutputDir\src" -Recurse -File -Filter "*.tmp" | Remove-Item -Force

Write-Host "      Done." -ForegroundColor Green
Write-Host ""

# ── Re-copy Streamlit config ───────────────────────────────────────────
Write-Host "[2/3] Updating Streamlit config..." -ForegroundColor Yellow

if (Test-Path ".streamlit") {
    Remove-Item -Recurse -Force "$OutputDir\.streamlit" -ErrorAction SilentlyContinue
    Copy-Item -Recurse -Force ".streamlit" "$OutputDir\.streamlit"
    Write-Host "      Done." -ForegroundColor Green
} else {
    Write-Host "      No .streamlit config found, skipping." -ForegroundColor Gray
}
Write-Host ""

# ── Re-copy launcher ───────────────────────────────────────────────────
Write-Host "[3/3] Updating launcher..." -ForegroundColor Yellow
Copy-Item -Force "DOE-Toolkit.bat" "$OutputDir\DOE-Toolkit.bat"
Write-Host "      Done." -ForegroundColor Green
Write-Host ""

# ── Re-copy in-app docs ────────────────────────────────────────────────
if (Test-Path "docs") {
    Remove-Item -Recurse -Force "$OutputDir\docs" -ErrorAction SilentlyContinue
    Copy-Item -Recurse -Force "docs" "$OutputDir\docs"
}

# ── Summary ───────────────────────────────────────────────────────────
Write-Host "============================================================" -ForegroundColor Green
Write-Host " QUICK REBUILD SUCCESSFUL" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Output folder : $OutputDir" -ForegroundColor Cyan
Write-Host ""
Write-Host " To test:" -ForegroundColor Yellow
Write-Host "   cd $OutputDir" -ForegroundColor Gray
Write-Host "   .\DOE-Toolkit.bat" -ForegroundColor Gray
Write-Host ""
Write-Host " NOTE: If you added or changed dependencies, run build.ps1 instead." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
