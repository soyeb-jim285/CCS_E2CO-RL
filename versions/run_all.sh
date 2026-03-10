#!/bin/bash
# Run all E2CO versions in parallel (6 waves).
# Outputs go to log files; terminal shows a live progress dashboard.
#
# Usage: cd /path/to/CCS_E2CO-RL && bash versions/run_all.sh [--epochs 200] [--new-only]
#
# Assumes:
#   - data/ directory exists with .mat files
#   - GPU with >= 24GB VRAM (each wave uses ~6-9GB)
#   - torchdiffeq installed (for V4, V14)

# No set -e: background jobs may fail individually; we check after each wave

# Parse arguments
EPOCHS=200
NEW_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --new-only)
            NEW_ONLY=true
            shift
            ;;
        *)
            # Legacy: first positional arg is epochs
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                EPOCHS="$1"
            fi
            shift
            ;;
    esac
done

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

if $NEW_ONLY; then
    echo "=========================================="
    echo " E2CO V8-V16 (new versions only)"
    echo " Epochs: $EPOCHS | 3 waves (3-3-3)"
    echo "=========================================="

    # Check V0 checkpoint exists (needed by V13)
    if [ ! -f "outputs_v0_baseline/checkpoints/best_model.pt" ]; then
        echo ""
        echo "WARNING: V0 checkpoint not found at outputs_v0_baseline/checkpoints/best_model.pt"
        echo "  V13 (physics fine-tuning) will fall back to training from scratch."
    fi
else
    echo "=========================================="
    echo " E2CO Multi-Version Comparison (V0-V16)"
    echo " Epochs: $EPOCHS | 6 waves"
    echo "=========================================="
fi

# ── Waves 1-3: V0-V7 (skip if --new-only) ───────────────────────
if ! $NEW_ONLY; then

# ── Wave 1: V0, V1, V2 ──────────────────────────────────────────
echo ""
echo "── Wave 1/6: V0 Baseline, V1 Physics, V2 Physics-Encoder ──"
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
echo "── Wave 2/6: V3 Corrector, V4 Neural-ODE, V5 Coord-PINN ──"
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
echo "── Wave 3/6: V6 FNO, V7 DeepONet ──"
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

fi  # end if ! $NEW_ONLY

# ── Wave 4: V8, V9, V10 (PINN variants) ─────────────────────────
echo ""
echo "── Wave 4/6: V8 Curriculum-PINN, V9 Pressure-Only, V10 Enhanced-PINN ──"
PIDS=()

python -m versions.v8_curriculum_pinn.train \
    $COMMON --output_dir outputs_v8_curriculum_pinn/ --batch_size 32 \
    --no_adaptive_weights \
    --n_collocation_points 256 --lambda_consistency 0.1 \
    --curriculum_start 60 --curriculum_end 120 \
    > outputs_v8_curriculum_pinn.log 2>&1 &
PIDS+=($!)

python -m versions.v9_pressure_only_pinn.train \
    $COMMON --output_dir outputs_v9_pressure_only_pinn/ --batch_size 32 \
    --no_adaptive_weights \
    --n_collocation_points 256 --lambda_consistency 0.1 \
    > outputs_v9_pressure_only_pinn.log 2>&1 &
PIDS+=($!)

python -m versions.v10_enhanced_pinn.train \
    $COMMON --output_dir outputs_v10_enhanced_pinn/ --batch_size 32 \
    --no_adaptive_weights \
    --n_collocation_points 1024 --lambda_consistency 0.1 \
    --curriculum_start 60 --curriculum_end 120 \
    --siren_omega0 30.0 --fourier_L 4 \
    > outputs_v10_enhanced_pinn.log 2>&1 &
PIDS+=($!)

monitor_progress "Wave 4" "v8_curriculum_pinn" "v9_pressure_only_pinn" "v10_enhanced_pinn" &
MON_PID=$!

echo "  Running V8, V9, V10 in parallel (logs: outputs_v*.log)"
wait "${PIDS[@]}" 2>/dev/null
kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
printf "\r\033[K"

FAILED=""
for i in 0 1 2; do
    if ! wait "${PIDS[$i]}" 2>/dev/null; then
        case $i in
            0) FAILED+=" V8" ;;
            1) FAILED+=" V9" ;;
            2) FAILED+=" V10" ;;
        esac
    fi
done
if [ -n "$FAILED" ]; then
    echo "  WARNING: Failed:$FAILED (check logs)"
else
    echo "  Wave 4 complete."
fi

# ── Wave 5: V11, V12, V13 (masked/dual/finetune) ────────────────
echo ""
echo "── Wave 5/6: V11 Well-Masked, V12 Dual-Latent, V13 Physics-Finetune ──"
PIDS=()

python -m versions.v11_well_masked_physics.train \
    $COMMON --output_dir outputs_v11_well_masked_physics/ --batch_size 32 \
    --no_adaptive_weights \
    --well_mask_radius 3 \
    > outputs_v11_well_masked_physics.log 2>&1 &
PIDS+=($!)

python -m versions.v12_dual_latent.train \
    $COMMON --output_dir outputs_v12_dual_latent/ --batch_size 32 \
    --no_adaptive_weights \
    --n_collocation_points 256 --lambda_consistency 0.1 \
    --latent_data_dim 12 --latent_phys_dim 8 \
    > outputs_v12_dual_latent.log 2>&1 &
PIDS+=($!)

python -m versions.v13_physics_finetune.train \
    $COMMON --output_dir outputs_v13_physics_finetune/ --batch_size 32 \
    --no_adaptive_weights \
    --finetune_epochs 100 \
    --pretrained_path outputs_v0_baseline/checkpoints/best_model.pt \
    --encoder_lr_scale 0.1 \
    > outputs_v13_physics_finetune.log 2>&1 &
PIDS+=($!)

monitor_progress "Wave 5" "v11_well_masked_physics" "v12_dual_latent" "v13_physics_finetune" &
MON_PID=$!

echo "  Running V11, V12, V13 in parallel (logs: outputs_v*.log)"
wait "${PIDS[@]}" 2>/dev/null
kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
printf "\r\033[K"

FAILED=""
for i in 0 1 2; do
    if ! wait "${PIDS[$i]}" 2>/dev/null; then
        case $i in
            0) FAILED+=" V11" ;;
            1) FAILED+=" V12" ;;
            2) FAILED+=" V13" ;;
        esac
    fi
done
if [ -n "$FAILED" ]; then
    echo "  WARNING: Failed:$FAILED (check logs)"
else
    echo "  Wave 5 complete."
fi

# ── Wave 6: V14, V15, V16 (ODE/PINO/obs-weighted) ───────────────
echo ""
echo "── Wave 6/6: V14 Structured-ODE, V15 PINO, V16 Obs-Weighted ──"
PIDS=()

python -m versions.v14_latent_physics.train \
    $COMMON --output_dir outputs_v14_latent_physics/ --batch_size 32 \
    --no_adaptive_weights \
    --ode_method dopri5 --no_compile \
    --lambda_eigen_reg 0.001 \
    > outputs_v14_latent_physics.log 2>&1 &
PIDS+=($!)

python -m versions.v15_pino.train \
    $COMMON --output_dir outputs_v15_pino/ --batch_size 32 \
    --no_adaptive_weights \
    --fno_modes 12 --fno_width 32 \
    --n_collocation_points 256 --lambda_consistency 0.1 \
    --curriculum_start 60 --curriculum_end 120 \
    > outputs_v15_pino.log 2>&1 &
PIDS+=($!)

python -m versions.v16_obs_weighted_physics.train \
    $COMMON --output_dir outputs_v16_obs_weighted_physics/ --batch_size 32 \
    --no_adaptive_weights \
    --n_collocation_points 256 --lambda_consistency 0.1 \
    --well_influence_radius 5 \
    > outputs_v16_obs_weighted_physics.log 2>&1 &
PIDS+=($!)

monitor_progress "Wave 6" "v14_latent_physics" "v15_pino" "v16_obs_weighted_physics" &
MON_PID=$!

echo "  Running V14, V15, V16 in parallel (logs: outputs_v*.log)"
wait "${PIDS[@]}" 2>/dev/null
kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
printf "\r\033[K"

FAILED=""
for i in 0 1 2; do
    if ! wait "${PIDS[$i]}" 2>/dev/null; then
        case $i in
            0) FAILED+=" V14" ;;
            1) FAILED+=" V15" ;;
            2) FAILED+=" V16" ;;
        esac
    fi
done
if [ -n "$FAILED" ]; then
    echo "  WARNING: Failed:$FAILED (check logs)"
else
    echo "  Wave 6 complete."
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
