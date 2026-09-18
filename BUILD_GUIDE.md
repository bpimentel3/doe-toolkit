# DOE Toolkit - Build Guide

## Overview

DOE Toolkit is distributed as a **self-contained folder** built with `conda-pack`.
Users receive a zip file, extract it, and double-click `DOE-Toolkit.bat`. No Python
installation required on their machine.

This replaces the previous PyInstaller approach, which had an unresolvable recursive
spawn bug when used with Streamlit.

---

## Prerequisites (Developer Machine Only)

- Anaconda or Miniconda installed
- `doe-toolkit` conda environment with all dependencies
- `conda-pack` **0.9.2 or newer** in the **base** environment (older versions corrupt
  Python source files in the packed environment on Windows — see troubleshooting)

Verify:
```powershell
conda activate base
conda list | Select-String conda-pack
# Should show: conda-pack  0.9.2  ...  conda-forge
```

0.9.2 is only on `conda-forge` (the `defaults` channel tops out at 0.9.1, which
still has the corrupting bug). Install/upgrade with:
```powershell
conda install -n base -c conda-forge conda-pack=0.9.2 --freeze-installed
```

The build scripts fail fast with a clear message if the version is too old, and also
validate the packed environment (streamlit integrity) as the last step.

---

## Building

### Option A: PowerShell (recommended)
```powershell
cd <path-to-repo>\doe-toolkit
.\build.ps1
```

### Option B: Command Prompt
```cmd
cd <path-to-repo>\doe-toolkit
build.bat
```

Replace `<path-to-repo>` with the folder where you cloned the repository.

Both scripts do the same thing. Build time is **3-8 minutes** on first run
(conda-pack compresses ~500 MB of dependencies).

---

## What the Build Produces

```
dist\DOE-Toolkit\
├── DOE-Toolkit.bat     ← users double-click this
├── src\                ← app source code
│   └── ui\app.py
├── .streamlit\         ← Streamlit config
├── env\                ← bundled Python + all dependencies
│   └── python.exe
├── THIRD_PARTY_NOTICES.txt  ← license map of every bundled package (auto-generated)
├── LICENSE.txt
└── QUICKSTART.md
```

Total size: ~500-700 MB uncompressed, ~150 MB zipped.

The build runs `tools/license_audit.py` against the packed environment and
writes `THIRD_PARTY_NOTICES.txt` so each redistributed package's license and
origin travel with the app. To inspect before shipping:
```powershell
python tools\license_audit.py --env dist\DOE-Toolkit\env
```

---

## Testing the Build

```powershell
cd dist\DOE-Toolkit
.\DOE-Toolkit.bat
```

Expected output in the console window:
```
============================================================
 DOE Toolkit - Design of Experiments Software
============================================================

 Starting... your browser will open automatically.
 To stop the app, close this window.

============================================================
```

Browser should open to `http://localhost:8501` within ~10 seconds.

---

## Distributing

```powershell
# Create zip from project root
Compress-Archive -Path dist\DOE-Toolkit -DestinationPath dist\DOE-Toolkit-v0.2.0-win64.zip
```

Share `DOE-Toolkit-v0.2.0-win64.zip`. Users:
1. Extract the zip (right-click → Extract All)
2. Double-click `DOE-Toolkit.bat`
3. Browser opens with the app

---

## How It Works

`DOE-Toolkit.bat` runs:
```bat
"env\python.exe" -m streamlit run src\ui\app.py
```

Streamlit is launched through the bundled `env\python.exe` with the `streamlit`
module rather than the pip-generated `Scripts\streamlit.exe` launcher. Pip
launchers embed the build machine's absolute interpreter path and break with
`Fatal error in launcher: Unable to create process using ...` once the zip is
extracted anywhere that path doesn't exist. `python -m streamlit` uses only
paths relative to the app folder, so the app is fully relocatable.

---

## Troubleshooting

### conda-pack fails with "environment not found"
```powershell
conda env list   # verify doe-toolkit exists
conda activate doe-toolkit
conda env list   # should show * next to doe-toolkit
```

If the environment exists but conda still can't find it by name, set an
absolute path override at the top of `build.ps1`:
```powershell
$EnvPath = "C:\full\path\to\your\envs\doe-toolkit"
```

### tar extraction fails
`tar` ships with Windows 10 build 17063 and later.
```powershell
tar --version   # verify it exists
```

### Browser doesn't open
Streamlit opens the browser automatically. If it doesn't:
1. Check the console window for errors
2. Open `http://localhost:8501` manually in your browser

### App works in dev but not in the build
Check that `src\` was copied correctly into `dist\DOE-Toolkit\`:
```powershell
ls dist\DOE-Toolkit\src\ui\app.py   # should exist
```

### "conda-pack too old" or validation fails
conda-pack ≤ 0.9.1 mangles Python source files during unpack on Windows: its
prefix rewriting strips the extended-path sequence `\\?\` from pip-installed
source files (e.g. Streamlit's `watcher/util.py`), producing
`SyntaxError: unterminated string literal` on launch.

Fix: make sure base has conda-pack 0.9.2+ from conda-forge, then rebuild.
```powershell
conda install -n base -c conda-forge conda-pack=0.9.2 --freeze-installed
```

---

## Rebuilding

The build script always does a clean build (deletes `dist\` first).
Just re-run `build.ps1` or `build.bat`.

---

## Development (Not for Distribution)

For day-to-day development, skip the build entirely:
```powershell
conda activate doe-toolkit
streamlit run src/ui/app.py
```