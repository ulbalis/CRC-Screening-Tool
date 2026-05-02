@echo off
setlocal enabledelayedexpansion

REM ===========================================
REM CRC Screening Tool — Setup and Launch (Windows)
REM ===========================================
REM
REM This script does everything needed to run the tool:
REM   1. Finds a compatible Python (3.10 or 3.11)
REM   2. Creates a virtual environment (if needed)
REM   3. Installs required packages (if needed)
REM   4. Registers the Jupyter kernel
REM   5. Cleans up stale files
REM   6. Clears notebook outputs
REM   7. Launches the notebook in your browser
REM
REM Usage:
REM   Double-click this file, or open Command Prompt and run:
REM     setup.bat
REM
REM ===========================================

echo.
echo ===========================================
echo   CRC Screening Tool — Setup and Launch
echo ===========================================
echo.

REM ───────────────────────────────────────────
REM Locate this script's directory
REM ───────────────────────────────────────────
set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "KERNEL_NAME=crc_screening"
set "NOTEBOOK=CRC_Screening_Tool.ipynb"

cd /d "%SCRIPT_DIR%"

REM ───────────────────────────────────────────
REM Step 1: Find Python 3.10 or 3.11
REM ───────────────────────────────────────────
echo [Step 1] Looking for Python 3.10 or 3.11...
echo.

set "PYTHON_CMD="

REM Check candidates in order of preference
for %%C in (python3.11 python3.10 python3 python py) do (
    where %%C >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=2" %%V in ('%%C --version 2^>^&1') do (
            for /f "tokens=1,2 delims=." %%A in ("%%V") do (
                if "%%A"=="3" (
                    if "%%B"=="10" (
                        set "PYTHON_CMD=%%C"
                        echo   Found: %%C ^(%%V^)
                        goto :found_python
                    )
                    if "%%B"=="11" (
                        set "PYTHON_CMD=%%C"
                        echo   Found: %%C ^(%%V^)
                        goto :found_python
                    )
                )
            )
        )
    )
)

REM Also check py launcher which is common on Windows
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3.11 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3.11"
        echo   Found: py -3.11
        goto :found_python
    )
    py -3.10 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3.10"
        echo   Found: py -3.10
        goto :found_python
    )
)

echo   X No compatible Python found.
echo.
echo   This tool requires Python 3.10 or 3.11.
echo.
echo   Download from: https://www.python.org/downloads/
echo   Choose Python 3.11.x (not 3.12 or 3.13).
echo.
echo   IMPORTANT: During installation, check the box
echo   "Add Python to PATH"
echo.
echo   After installing, run this script again.
echo.
pause
exit /b 1

:found_python
echo.

REM ───────────────────────────────────────────
REM Step 2: Create virtual environment
REM ───────────────────────────────────────────
if exist "%VENV_DIR%" (
    echo [Step 2] Virtual environment already exists.

    if not exist "%VENV_DIR%\Scripts\python.exe" (
        echo   Warning: Broken venv detected. Recreating...
        rmdir /s /q "%VENV_DIR%"
    )
)

if not exist "%VENV_DIR%" (
    echo [Step 2] Creating virtual environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"

    if !errorlevel! neq 0 (
        echo   X Failed to create virtual environment.
        echo   Make sure Python 3.10 or 3.11 is properly installed.
        pause
        exit /b 1
    )
    echo   Created at: %VENV_DIR%
)

echo.

REM ───────────────────────────────────────────
REM Step 3: Install packages
REM ───────────────────────────────────────────
echo [Step 3] Checking packages...

"%VENV_DIR%\Scripts\python.exe" -c "import tensorflow; import openslide; import ipywidgets; import pandas; import matplotlib; import plotly; import sklearn" >nul 2>&1
if !errorlevel! neq 0 (
    echo   Installing required packages...
    echo   (This may take a few minutes on first run^)
    echo.

    "%VENV_DIR%\Scripts\pip.exe" install --upgrade pip --quiet

    "%VENV_DIR%\Scripts\pip.exe" install ^
        tensorflow==2.15.1 ^
        keras==2.15.0 ^
        numpy==1.26.4 ^
        pillow ^
        matplotlib ^
        pandas ^
        plotly==5.23.0 ^
        openslide-python ^
        openslide-bin ^
        ipykernel==7.1.0 ^
        ipywidgets==8.1.7 ^
        ipyfilechooser ^
        ipyevents ^
        notebook ^
        scikit-learn ^
        --quiet

    if !errorlevel! neq 0 (
        echo.
        echo   X Package installation failed.
        echo   Check your internet connection and try again.
        pause
        exit /b 1
    )

    REM Verify OpenSlide native library
    "%VENV_DIR%\Scripts\python.exe" -c "import openslide" >nul 2>&1
    if !errorlevel! neq 0 (
        echo   Warning: OpenSlide native library not detected.
        echo   Tile export and slide viewing may not work.
        echo   Visit https://openslide.org for installation help.
    )

    echo   All packages installed.
) else (
    echo   All packages already installed.
)

echo.

REM ───────────────────────────────────────────
REM Step 4: Register Jupyter kernel
REM ───────────────────────────────────────────
echo [Step 4] Registering Jupyter kernel...

"%VENV_DIR%\Scripts\python.exe" -m ipykernel install ^
    --user ^
    --name "%KERNEL_NAME%" ^
    --display-name "CRC Screening Tool (Python 3)" >nul 2>&1

if !errorlevel! equ 0 (
    echo   Kernel registered: %KERNEL_NAME%
    echo   Python: %VENV_DIR%\Scripts\python.exe
) else (
    echo   Warning: Kernel registration failed.
    echo   You may need to select the kernel manually in Jupyter.
)

echo.

REM ───────────────────────────────────────────
REM Step 5: Clean up stale files
REM ───────────────────────────────────────────
echo [Step 5] Cleaning up...

for /d /r "%SCRIPT_DIR%\utils" %%D in (__pycache__) do (
    if exist "%%D" rmdir /s /q "%%D"
)
if exist "%SCRIPT_DIR%\iframe_figures" rmdir /s /q "%SCRIPT_DIR%\iframe_figures"

echo   Stale files removed.
echo.

REM ───────────────────────────────────────────
REM Step 6: Clear notebook outputs
REM ───────────────────────────────────────────
echo [Step 6] Clearing notebook outputs...

"%VENV_DIR%\Scripts\jupyter.exe" nbconvert ^
    --ClearOutputPreprocessor.enabled=True ^
    --to notebook ^
    --inplace ^
    "%SCRIPT_DIR%\%NOTEBOOK%" >nul 2>&1

if !errorlevel! equ 0 (
    echo   Notebook outputs cleared.
) else (
    echo   Warning: Could not clear notebook outputs (non-critical^).
)

echo.

REM ───────────────────────────────────────────
REM Step 7: Launch notebook
REM ───────────────────────────────────────────
echo [Step 7] Launching notebook...
echo.
echo ===========================================
echo   The notebook will open in your browser.
echo   To stop the server, press Ctrl+C here
echo   or close this window.
echo ===========================================
echo.

"%VENV_DIR%\Scripts\jupyter.exe" notebook "%SCRIPT_DIR%\%NOTEBOOK%" --quiet

pause