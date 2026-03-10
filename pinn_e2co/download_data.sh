#!/usr/bin/env bash
# ==============================================================================
# Download PINN-E2CO data from Google Drive to /workspace/data/
#
# Usage:
#   bash download_data.sh <GOOGLE_DRIVE_FOLDER_ID>
#   bash download_data.sh <GOOGLE_DRIVE_ZIP_LINK>
#
# Examples:
#   # If data/ folder is shared as a Google Drive folder:
#   bash download_data.sh 1aBcDeFgHiJkLmNoPqRsT
#
#   # If data/ is zipped and shared as a single file:
#   bash download_data.sh --zip 1aBcDeFgHiJkLmNoPqRsT
#
#   # Direct full URL also works:
#   bash download_data.sh "https://drive.google.com/drive/folders/1aBcD..."
#   bash download_data.sh --zip "https://drive.google.com/file/d/1aBcD.../view"
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$WORKSPACE/data"

REQUIRED_FILES=(
    "states_norm_slt.mat"
    "controls_norm_slt.mat"
    "rate_norm_slt.mat"
    "TRUE_PERM_64by220.mat"
)

# ---- Helpers ----

usage() {
    echo "Usage:"
    echo "  bash $0 <FOLDER_ID_OR_URL>           Download individual files from a Drive folder"
    echo "  bash $0 --zip <FILE_ID_OR_URL>        Download a zip archive and extract"
    echo ""
    echo "  The Google Drive folder/file must be shared with 'Anyone with the link'."
    echo ""
    echo "  To get the ID:"
    echo "    Folder: https://drive.google.com/drive/folders/<THIS_IS_THE_ID>"
    echo "    File:   https://drive.google.com/file/d/<THIS_IS_THE_ID>/view"
    exit 1
}

extract_id() {
    local input="$1"
    # Already a bare ID (no slashes, no dots)
    if [[ ! "$input" =~ [/\.] ]]; then
        echo "$input"
        return
    fi
    # Extract from folder URL: .../folders/<ID>...
    if [[ "$input" =~ folders/([a-zA-Z0-9_-]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi
    # Extract from file URL: .../d/<ID>/...
    if [[ "$input" =~ /d/([a-zA-Z0-9_-]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi
    # Extract from export/open URL: ...id=<ID>...
    if [[ "$input" =~ id=([a-zA-Z0-9_-]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return
    fi
    echo "$input"
}

install_gdown() {
    if command -v gdown &> /dev/null; then
        return
    fi
    echo "[*] Installing gdown..."
    pip install --quiet gdown
}

download_file_by_id() {
    local file_id="$1"
    local output_path="$2"
    # gdown handles the Google Drive confirmation page automatically
    gdown "$file_id" -O "$output_path" --fuzzy 2>&1 || {
        echo "  gdown failed, trying curl fallback..."
        curl -L "https://drive.google.com/uc?export=download&id=${file_id}&confirm=t" -o "$output_path"
    }
}

verify_data() {
    echo ""
    echo "[*] Verifying downloaded files..."
    local all_ok=true
    for f in "${REQUIRED_FILES[@]}"; do
        if [ -f "$DATA_DIR/$f" ]; then
            local size
            size=$(du -h "$DATA_DIR/$f" | cut -f1)
            echo "  OK  $f ($size)"
        else
            echo "  MISSING  $f"
            all_ok=false
        fi
    done

    if [ "$all_ok" = true ]; then
        echo ""
        echo "All data files present. Ready to train:"
        echo "  cd $SCRIPT_DIR && python pinn_train.py --epochs 5"
    else
        echo ""
        echo "WARNING: Some files are missing. Check your Google Drive share settings."
        echo "The folder/file must be shared with 'Anyone with the link'."
    fi
}

# ---- Parse args ----

MODE="folder"
DRIVE_INPUT=""

if [ $# -eq 0 ]; then
    usage
fi

if [ "$1" = "--zip" ]; then
    MODE="zip"
    if [ $# -lt 2 ]; then
        echo "Error: --zip requires a file ID or URL"
        usage
    fi
    DRIVE_INPUT="$2"
elif [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
else
    DRIVE_INPUT="$1"
fi

DRIVE_ID=$(extract_id "$DRIVE_INPUT")

echo "============================================"
echo "  PINN-E2CO Data Download"
echo "  Mode:      $MODE"
echo "  Drive ID:  $DRIVE_ID"
echo "  Target:    $DATA_DIR"
echo "============================================"

mkdir -p "$DATA_DIR"
install_gdown

# ---- Download ----

if [ "$MODE" = "zip" ]; then
    echo ""
    echo "[*] Downloading zip archive..."
    TMP_ZIP="$WORKSPACE/_data_download.zip"
    download_file_by_id "$DRIVE_ID" "$TMP_ZIP"

    echo "[*] Extracting to $DATA_DIR..."
    # Try to extract smartly: if zip contains a top-level folder, flatten it
    unzip -o "$TMP_ZIP" -d "$WORKSPACE/_data_tmp" > /dev/null 2>&1

    # Check if extracted into a subfolder
    EXTRACTED_DIRS=("$WORKSPACE"/_data_tmp/*/)
    if [ ${#EXTRACTED_DIRS[@]} -eq 1 ] && [ -d "${EXTRACTED_DIRS[0]}" ]; then
        # Single subfolder — move its contents to data/
        echo "  Detected subfolder: $(basename "${EXTRACTED_DIRS[0]}")"
        cp -r "${EXTRACTED_DIRS[0]}"* "$DATA_DIR/" 2>/dev/null || true
        cp -r "${EXTRACTED_DIRS[0]}".* "$DATA_DIR/" 2>/dev/null || true
    else
        # Files directly in the tmp dir
        cp -r "$WORKSPACE"/_data_tmp/* "$DATA_DIR/" 2>/dev/null || true
    fi

    rm -rf "$TMP_ZIP" "$WORKSPACE/_data_tmp"
    echo "  Done."

else
    echo ""
    echo "[*] Downloading folder contents..."
    # gdown can download entire Google Drive folders
    gdown "$DRIVE_ID" -O "$DATA_DIR" --folder --remaining-ok 2>&1 || {
        echo ""
        echo "Folder download failed. Alternatives:"
        echo "  1. Zip the data/ folder on Drive, share the zip, then run:"
        echo "     bash $0 --zip <ZIP_FILE_ID>"
        echo ""
        echo "  2. Download individual files by ID:"
        echo "     gdown <FILE_ID> -O $DATA_DIR/filename.mat"
        echo ""
        echo "  3. Make sure the folder is shared with 'Anyone with the link'"
        exit 1
    }

    # gdown --folder may create a subfolder with the Drive folder name
    # If data files ended up in a subfolder, move them up
    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$DATA_DIR/$f" ]; then
            # Search one level down
            found=$(find "$DATA_DIR" -maxdepth 2 -name "$f" -type f 2>/dev/null | head -1)
            if [ -n "$found" ]; then
                echo "  Moving $f from subfolder..."
                mv "$found" "$DATA_DIR/$f"
            fi
        fi
    done

    # Clean up empty subdirs
    find "$DATA_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    echo "  Done."
fi

verify_data
