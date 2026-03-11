#!/bin/bash
# Zip all output results except model checkpoints.
# Usage: bash versions/zip_results.sh [output_name]
#
# Creates a zip with logs, plots, metrics, and comparison results
# but excludes heavy checkpoint files (.pt).

OUT_NAME="${1:-results_$(date +%Y%m%d_%H%M%S).zip}"

zip -r "$OUT_NAME" \
    outputs_v*/logs/ \
    outputs_v*/plots/ \
    outputs_v*.log \
    outputs_comparison/ \
    -x "outputs_v*/checkpoints/*" \
    -x "*.pt"

echo ""
echo "Created: $OUT_NAME"
echo "Size: $(du -h "$OUT_NAME" | cut -f1)"
