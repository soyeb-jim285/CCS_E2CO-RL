#!/bin/bash
# Zip outputs (plots + logs) excluding large checkpoint files.
# Usage: bash zip_outputs.sh [output_name]
#   Creates: /workspace/outputs_light.zip (or custom name)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../outputs"
ZIP_NAME="${1:-outputs_light.zip}"
ZIP_PATH="$SCRIPT_DIR/../$ZIP_NAME"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: outputs directory not found at $OUTPUT_DIR"
    exit 1
fi

cd "$OUTPUT_DIR/.."

zip -r "$ZIP_PATH" outputs/ \
    -x "outputs/checkpoints/*" \
    -x "*.pt" \
    -x "*.pth" \
    -x "*.h5" \
    -x "*.ckpt"

echo ""
echo "Created: $ZIP_PATH"
echo "Size: $(du -h "$ZIP_PATH" | cut -f1)"
echo ""
echo "Included:"
zipinfo -1 "$ZIP_PATH" | head -30
COUNT=$(zipinfo -1 "$ZIP_PATH" | wc -l)
[ "$COUNT" -gt 30 ] && echo "... and $((COUNT - 30)) more files"
