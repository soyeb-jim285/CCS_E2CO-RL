#!/bin/bash
# Run all 8 E2CO versions in parallel (2 waves of 4).
# Usage: cd /path/to/CCS_E2CO-RL && bash versions/run_all.sh [--epochs 200]
#
# Assumes:
#   - data/ directory exists with .mat files
#   - GPU with >= 24GB VRAM (each version uses ~2-3GB)
#   - torchdiffeq installed (for V4)

set -e

# Parse optional epoch override
EPOCHS="${1:-200}"
if [ "$1" = "--epochs" ]; then
    EPOCHS="$2"
fi

echo "=========================================="
echo "E2CO Multi-Version Comparison"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Common args
COMMON="--epochs $EPOCHS --data_dir data/ --seed 1010"

echo ""
echo "Wave 1: V0-V3 (simpler architectures)"
echo "=========================================="

python -m versions.v0_baseline.train \
    $COMMON --output_dir outputs_v0_baseline/ --batch_size 4 \
    2>&1 | tee outputs_v0_baseline.log &
PID_V0=$!

python -m versions.v1_physics_constrained.train \
    $COMMON --output_dir outputs_v1_physics_constrained/ --batch_size 32 \
    2>&1 | tee outputs_v1_physics_constrained.log &
PID_V1=$!

python -m versions.v2_physics_encoder.train \
    $COMMON --output_dir outputs_v2_physics_encoder/ --batch_size 32 \
    2>&1 | tee outputs_v2_physics_encoder.log &
PID_V2=$!

python -m versions.v3_physics_corrector.train \
    $COMMON --output_dir outputs_v3_physics_corrector/ --batch_size 32 \
    --n_corrector_iterations 2 --corrector_alpha 0.1 \
    2>&1 | tee outputs_v3_physics_corrector.log &
PID_V3=$!

echo "Waiting for Wave 1 (PIDs: $PID_V0 $PID_V1 $PID_V2 $PID_V3)..."
wait $PID_V0 $PID_V1 $PID_V2 $PID_V3
echo "Wave 1 complete."

echo ""
echo "Wave 2: V4-V7 (complex architectures)"
echo "=========================================="

python -m versions.v4_neural_ode.train \
    $COMMON --output_dir outputs_v4_neural_ode/ --batch_size 32 \
    --ode_method dopri5 --no_compile \
    2>&1 | tee outputs_v4_neural_ode.log &
PID_V4=$!

python -m versions.v5_coordinate_pinn.train \
    $COMMON --output_dir outputs_v5_coordinate_pinn/ --batch_size 32 \
    --n_collocation_points 256 --lambda_consistency 0.1 \
    2>&1 | tee outputs_v5_coordinate_pinn.log &
PID_V5=$!

python -m versions.v6_fno_decoder.train \
    $COMMON --output_dir outputs_v6_fno_decoder/ --batch_size 32 \
    --fno_modes 12 --fno_width 32 \
    2>&1 | tee outputs_v6_fno_decoder.log &
PID_V6=$!

python -m versions.v7_deeponet.train \
    $COMMON --output_dir outputs_v7_deeponet/ --batch_size 32 \
    --n_basis 64 --trunk_dim 128 \
    2>&1 | tee outputs_v7_deeponet.log &
PID_V7=$!

echo "Waiting for Wave 2 (PIDs: $PID_V4 $PID_V5 $PID_V6 $PID_V7)..."
wait $PID_V4 $PID_V5 $PID_V6 $PID_V7
echo "Wave 2 complete."

echo ""
echo "=========================================="
echo "Generating comparison report..."
echo "=========================================="

python -m versions.compare --base_dir . --out_dir outputs_comparison/

echo ""
echo "All done! Results in outputs_comparison/"
