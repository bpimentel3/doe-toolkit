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
- `conda-pack` installed in the **base** environment

Verify:
```powershell
conda activate base
conda list | Select-String conda-pack
# Should show: conda-pack  0.8.x  ...
```

If missing:
```powershell
conda install conda-pack
```

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
│   └── Scripts\
│       └── streamlit.exe
├── LICENSE.txt
└── QUICKSTART.md
```

Total size: ~500-700 MB uncompressed, ~150 MB zipped.

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
Compress-Archive -Path dist\DOE-Toolkit -DestinationPath dist\DOE-Toolkit-v0.1.0-win64.zip
```

Share `DOE-Toolkit-v0.1.0-win64.zip`. Users:
1. Extract the zip (right-click → Extract All)
2. Double-click `DOE-Toolkit.bat`
3. Browser opens with the app

---

## How It Works

`DOE-Toolkit.bat` runs:
```bat
"env\Scripts\streamlit.exe" run src\ui\app.py
```

`env\Scripts\streamlit.exe` is a real Windows executable inside the bundled
environment — not `sys.executable` pointing back at itself. This avoids the
recursive spawn issue that PyInstaller had.

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