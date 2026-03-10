#!/bin/bash
# Run all 8 E2CO versions in parallel (3 waves of 3-3-2).
# Outputs go to log files; terminal shows a live progress dashboard.
#
# Usage: cd /path/to/CCS_E2CO-RL && bash versions/run_all.sh [--epochs 200]
#
# Assumes:
#   - data/ directory exists with .mat files
#   - GPU with >= 24GB VRAM (each wave uses ~6-9GB)
#   - torchdiffeq installed (for V4)

# No set -e: background jobs may fail individually; we check after each wave

# Parse optional epoch override
EPOCHS="${1:-200}"
if [ "$1" = "--epochs" ]; then
    EPOCHS="$2"
fi

COMMON="--epochs $EPOCHS --data_dir data/ --seed 1010"

# ── Progress monitor ─────────────────────────────────────────────
# Reads each version's loss_history.csv to show epoch progress
monitor_progress() {
    local wave_name="$1"
    shift
    local versions=("$@")

    while true; do
        sleep 10
        # Check if any background job from this wave is still running
        local any_alive=false
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                any_alive=true
                break
            fi
        done
        if ! $any_alive; then
            break
        fi

        # Build status line
        local status=""
        for vname in "${versions[@]}"; do
            local csv="outputs_${vname}/logs/loss_history.csv"
            local epoch="--"
            local loss="--"
            if [ -f "$csv" ]; then
                local last_line
                last_line=$(tail -1 "$csv" 2>/dev/null)
                if [ -n "$last_line" ] && ! echo "$last_line" | grep -q "^epoch"; then
                    epoch=$(echo "$last_line" | cut -d',' -f1)
                    loss=$(echo "$last_line" | cut -d',' -f2 | head -c 10)
                fi
            fi
            local short="${vname%%_*}"
            short="${short^^}"
            status+="  ${short}: ${epoch}/${EPOCHS} (${loss})"
        done
        # Print on single line with carriage return
        printf "\r\033[K[$wave_name]%s" "$status"
    done
    printf "\r\033[K"
}

echo "=========================================="
echo " E2CO Multi-Version Comparison"
echo " Epochs: $EPOCHS | 3 waves (3-3-2)"
echo "=========================================="

# ── Wave 1: V0, V1, V2 ──────────────────────────────────────────
echo ""
echo "── Wave 1/3: V0 Baseline, V1 Physics, V2 Physics-Encoder ──"
PIDS=()

python -m versions.v0_baseline.train \
    $COMMON --output_dir outputs_v0_baseline/ --batch_size 4 \
    > outputs_v0_baseline.log 2>&1 &
PIDS+=($!)

python -m versions.v1_physics_constrained.train \
    $COMMON --output_dir outputs_v1_physics_constrained/ --batch_size 32 \
    --no_adaptive_weights \
    > outputs_v1_physics_constrained.log 2>&1 &
PIDS+=($!)

python -m versions.v2_physics_encoder.train \
    $COMMON --output_dir outputs_v2_physics_encoder/ --batch_size 32 \
    --no_adaptive_weights \
    > outputs_v2_physics_encoder.log 2>&1 &
PIDS+=($!)

monitor_progress "Wave 1" "v0_baseline" "v1_physics_constrained" "v2_physics_encoder" &
MON_PID=$!

echo "  Running V0, V1, V2 in parallel (logs: outputs_v*.log)"
wait "${PIDS[@]}" 2>/dev/null
kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
printf "\r\033[K"

# Check for failures
FAILED=""
for i in 0 1 2; do
    if ! wait "${PIDS[$i]}" 2>/dev/null; then
        case $i in
            0) FAILED+=" V0" ;;
            1) FAILED+=" V1" ;;
            2) FAILED+=" V2" ;;
        esac
    fi
done
if [ -n "$FAILED" ]; then
    echo "  WARNING: Failed:$FAILED (check logs)"
else
    echo "  Wave 1 complete."
fi

# ── Wave 2: V3, V4, V5 ──────────────────────────────────────────
echo ""
echo "── Wave 2/3: V3 Corrector, V4 Neural-ODE, V5 Coord-PINN ──"
PIDS=()

python -m versions.v3_physics_corrector.train \
    $COMMON --output_dir outputs_v3_physics_corrector/ --batch_size 32 \
    --no_adaptive_weights \
    --n_corrector_iterations 2 --corrector_alpha 0.1 \
    > outputs_v3_physics_corrector.log 2>&1 &
PIDS+=($!)

python -m versions.v4_neural_ode.train \
    $COMMON --output_dir outputs_v4_neural_ode/ --batch_size 32 \
    --no_adaptive_weights \
    --ode_method dopri5 --no_compile \
    > outputs_v4_neural_ode.log 2>&1 &
PIDS+=($!)

python -m versions.v5_coordinate_pinn.train \
    $COMMON --output_dir outputs_v5_coordinate_pinn/ --batch_size 32 \
    --no_adaptive_weights \
    --n_collocation_points 256 --lambda_consistency 0.1 \
    > outputs_v5_coordinate_pinn.log 2>&1 &
PIDS+=($!)

monitor_progress "Wave 2" "v3_physics_corrector" "v4_neural_ode" "v5_coordinate_pinn" &
MON_PID=$!

echo "  Running V3, V4, V5 in parallel (logs: outputs_v*.log)"
wait "${PIDS[@]}" 2>/dev/null
kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
printf "\r\033[K"

FAILED=""
for i in 0 1 2; do
    if ! wait "${PIDS[$i]}" 2>/dev/null; then
        case $i in
            0) FAILED+=" V3" ;;
            1) FAILED+=" V4" ;;
            2) FAILED+=" V5" ;;
        esac
    fi
done
if [ -n "$FAILED" ]; then
    echo "  WARNING: Failed:$FAILED (check logs)"
else
    echo "  Wave 2 complete."
fi

# ── Wave 3: V6, V7 ──────────────────────────────────────────────
echo ""
echo "── Wave 3/3: V6 FNO, V7 DeepONet ──"
PIDS=()

python -m versions.v6_fno_decoder.train \
    $COMMON --output_dir outputs_v6_fno_decoder/ --batch_size 32 \
    --no_adaptive_weights \
    --fno_modes 12 --fno_width 32 \
    > outputs_v6_fno_decoder.log 2>&1 &
PIDS+=($!)

python -m versions.v7_deeponet.train \
    $COMMON --output_dir outputs_v7_deeponet/ --batch_size 32 \
    --no_adaptive_weights \
    --n_basis 64 --trunk_dim 128 \
    > outputs_v7_deeponet.log 2>&1 &
PIDS+=($!)

monitor_progress "Wave 3" "v6_fno_decoder" "v7_deeponet" &
MON_PID=$!

echo "  Running V6, V7 in parallel (logs: outputs_v*.log)"
wait "${PIDS[@]}" 2>/dev/null
kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
printf "\r\033[K"

FAILED=""
for i in 0 1; do
    if ! wait "${PIDS[$i]}" 2>/dev/null; then
        case $i in
            0) FAILED+=" V6" ;;
            1) FAILED+=" V7" ;;
        esac
    fi
done
if [ -n "$FAILED" ]; then
    echo "  WARNING: Failed:$FAILED (check logs)"
else
    echo "  Wave 3 complete."
fi

# ── Comparison ───────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Generating comparison report..."
echo "=========================================="

python -m versions.compare --base_dir . --out_dir outputs_comparison/

echo ""
echo "=========================================="
echo " All done! Results in outputs_comparison/"
echo "=========================================="
echo ""
echo " Per-version logs:  outputs_v*.log"
echo " Per-version plots: outputs_v*/plots/"
echo " Comparison:        outputs_comparison/"
