#!/bin/bash

# ===========================================
# CRC Screening Tool — Setup & Launch (macOS)
# ===========================================
#
# This script does everything needed to run the tool:
#   1. Finds a compatible Python (3.10 or 3.11)
#   2. Creates a virtual environment (if needed)
#   3. Installs required packages (if needed)
#   4. Registers the Jupyter kernel
#   5. Cleans up stale files
#   6. Clears notebook outputs
#   7. Launches the notebook in your browser
#
# Usage:
#   cd /path/to/CRC_Screening_Tool
#   bash setup.sh
#
# ===========================================

echo ""
echo "==========================================="
echo "  CRC Screening Tool — Setup & Launch"
echo "==========================================="
echo ""

# ───────────────────────────────────────────
# Locate this script's directory
# ───────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
KERNEL_NAME="crc_screening"
NOTEBOOK="CRC_Screening_Tool.ipynb"

cd "$SCRIPT_DIR"

# ───────────────────────────────────────────
# Step 1: Find Python 3.10 or 3.11
# ───────────────────────────────────────────
echo "[Step 1] Looking for Python 3.10 or 3.11..."
echo ""

PYTHON_CMD=""

for candidate in python3.11 python3.10 python3 python; do
    if command -v "$candidate" &> /dev/null; then
        PY_VERSION=$("$candidate" --version 2>&1 | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

        if [ "$PY_MAJOR" = "3" ] && ([ "$PY_MINOR" = "10" ] || [ "$PY_MINOR" = "11" ]); then
            PYTHON_CMD="$candidate"
            echo "  Found: $candidate ($PY_VERSION)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "   No compatible Python found."
    echo ""
    echo "  This tool requires Python 3.10 or 3.11."
    echo ""
    echo "  To install with Homebrew:"
    echo "    brew install python@3.11"
    echo ""
    echo "  To install with pyenv:"
    echo "    pyenv install 3.11.14"
    echo "    pyenv local 3.11.14"
    echo ""
    echo "  After installing, run this script again."
    exit 1
fi

echo ""

# ───────────────────────────────────────────
# Step 2: Create virtual environment
# ───────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[Step 2] Virtual environment already exists."

    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo "    Broken venv detected (moved folder?). Recreating..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[Step 2] Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"

    if [ $? -ne 0 ]; then
        echo "   Failed to create virtual environment."
        echo "  Make sure Python 3.10 or 3.11 is properly installed."
        exit 1
    fi
    echo "  Created at: $VENV_DIR"
fi

echo ""

# ───────────────────────────────────────────
# Step 3: Install packages
# ───────────────────────────────────────────
echo "[Step 3] Checking packages..."

NEEDS_INSTALL=0
"$VENV_DIR/bin/python" -c \
    "import tensorflow; import openslide; import ipywidgets; import pandas; import matplotlib; import plotly; import sklearn" \
    2>/dev/null || NEEDS_INSTALL=1

if [ "$NEEDS_INSTALL" = "1" ]; then
    echo "  Installing required packages..."
    echo "  (This may take a few minutes on first run)"
    echo ""

    "$VENV_DIR/bin/pip" install --upgrade pip --quiet

    "$VENV_DIR/bin/pip" install \
        tensorflow==2.15.1 \
        keras==2.15.0 \
        numpy==1.26.4 \
        pillow \
        matplotlib \
        pandas \
        plotly==5.23.0 \
        openslide-python \
        openslide-bin \
        ipykernel==7.1.0 \
        ipywidgets==8.1.7 \
        ipyfilechooser \
        ipyevents \
        notebook \
        scikit-learn \
        --quiet

    if [ $? -ne 0 ]; then
        echo ""
        echo "   Package installation failed."
        echo "  Check your internet connection and try again."
        exit 1
    fi

    echo "   All packages installed."
else
    echo "   All packages already installed."
fi

echo ""

# ───────────────────────────────────────────
# Step 4: Register Jupyter kernel
# ───────────────────────────────────────────
echo "[Step 4] Registering Jupyter kernel..."

"$VENV_DIR/bin/python" -m ipykernel install \
    --user \
    --name "$KERNEL_NAME" \
    --display-name "CRC Screening Tool (Python 3)" \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo "   Kernel registered: $KERNEL_NAME"
    echo "     Python: $VENV_DIR/bin/python"
else
    echo "    Kernel registration failed."
    echo "     You may need to select the kernel manually in Jupyter."
fi

echo ""

# ───────────────────────────────────────────
# Step 5: Clean up stale files
# ───────────────────────────────────────────
echo "[Step 5] Cleaning up..."

find "$SCRIPT_DIR/utils" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
rm -rf "$SCRIPT_DIR/iframe_figures" 2>/dev/null

echo "   Stale files removed."
echo ""

# ───────────────────────────────────────────
# Step 6: Clear notebook outputs
# ───────────────────────────────────────────
echo "[Step 6] Clearing notebook outputs..."

"$VENV_DIR/bin/jupyter" nbconvert \
    --ClearOutputPreprocessor.enabled=True \
    --to notebook \
    --inplace \
    "$SCRIPT_DIR/$NOTEBOOK" \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo "    Notebook outputs cleared."
else
    echo "   Could not clear notebook outputs (non-critical)."
fi

echo ""

# ───────────────────────────────────────────
# Step 7: Launch notebook
# ───────────────────────────────────────────
echo "[Step 7] Launching notebook..."
echo ""
echo "==========================================="
echo "  The notebook will open in your browser."
echo "  To stop the server, press Ctrl+C here."
echo "==========================================="
echo ""

"$VENV_DIR/bin/jupyter" notebook "$SCRIPT_DIR/$NOTEBOOK"