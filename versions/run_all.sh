#!/bin/bash
# Run all E2CO versions with a parallel job queue.
# Instead of rigid waves, runs up to N jobs concurrently — when one finishes,
# the next starts immediately.
#
# Usage: cd /path/to/CCS_E2CO-RL && bash versions/run_all.sh [OPTIONS]
#
# Options:
#   --epochs N       Training epochs (default: 200)
#   --jobs N         Max concurrent jobs (default: 3)
#   --new-only       Skip V0-V16, run only V17-V25
#   --base_dir DIR   Base directory (default: .)
#
# Assumes:
#   - data/ directory exists with .mat files
#   - GPU with >= 24GB VRAM
#   - torchdiffeq installed (for V4, V14, V17, V22)

# Parse arguments
EPOCHS=200
MAX_JOBS=3
NEW_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)   EPOCHS="$2"; shift 2 ;;
        --jobs)     MAX_JOBS="$2"; shift 2 ;;
        --new-only) NEW_ONLY=true; shift ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then EPOCHS="$1"; fi
            shift ;;
    esac
done

COMMON="--epochs $EPOCHS --data_dir data/ --seed 1010 --no_checkpoints"
V0_CKPT="outputs_v0_baseline/checkpoints/best_model.pt"

# ── Job definitions ──────────────────────────────────────────────
# Each job: "version_name|needs_v0|command_args"
#   needs_v0=1 means it waits for V0 checkpoint before starting

ALL_JOBS=()

if ! $NEW_ONLY; then
ALL_JOBS+=(
    "v0_baseline|0|--output_dir outputs_v0_baseline/ --batch_size 4"
    "v1_physics_constrained|0|--output_dir outputs_v1_physics_constrained/ --batch_size 32 --no_adaptive_weights"
    "v2_physics_encoder|0|--output_dir outputs_v2_physics_encoder/ --batch_size 32 --no_adaptive_weights"
    "v3_physics_corrector|0|--output_dir outputs_v3_physics_corrector/ --batch_size 32 --no_adaptive_weights --n_corrector_iterations 2 --corrector_alpha 0.1"
    "v4_neural_ode|0|--output_dir outputs_v4_neural_ode/ --batch_size 32 --no_adaptive_weights --ode_method dopri5 --no_compile"
    "v5_coordinate_pinn|0|--output_dir outputs_v5_coordinate_pinn/ --batch_size 32 --no_adaptive_weights --n_collocation_points 256 --lambda_consistency 0.1"
    "v6_fno_decoder|0|--output_dir outputs_v6_fno_decoder/ --batch_size 32 --no_adaptive_weights --fno_modes 12 --fno_width 32"
    "v7_deeponet|0|--output_dir outputs_v7_deeponet/ --batch_size 32 --no_adaptive_weights --n_basis 64 --trunk_dim 128"
    "v8_curriculum_pinn|0|--output_dir outputs_v8_curriculum_pinn/ --batch_size 32 --no_adaptive_weights --n_collocation_points 256 --lambda_consistency 0.1 --curriculum_start 60 --curriculum_end 120"
    "v9_pressure_only_pinn|0|--output_dir outputs_v9_pressure_only_pinn/ --batch_size 32 --no_adaptive_weights --n_collocation_points 256 --lambda_consistency 0.1"
    "v10_enhanced_pinn|0|--output_dir outputs_v10_enhanced_pinn/ --batch_size 32 --no_adaptive_weights --n_collocation_points 1024 --lambda_consistency 0.1 --curriculum_start 60 --curriculum_end 120 --siren_omega0 30.0 --fourier_L 4"
    "v11_well_masked_physics|0|--output_dir outputs_v11_well_masked_physics/ --batch_size 32 --no_adaptive_weights --well_mask_radius 3"
    "v12_dual_latent|0|--output_dir outputs_v12_dual_latent/ --batch_size 32 --no_adaptive_weights --n_collocation_points 256 --lambda_consistency 0.1 --latent_data_dim 12 --latent_phys_dim 8"
    "v13_physics_finetune|1|--output_dir outputs_v13_physics_finetune/ --batch_size 32 --no_adaptive_weights --finetune_epochs 100 --pretrained_path $V0_CKPT --encoder_lr_scale 0.1"
    "v14_latent_physics|0|--output_dir outputs_v14_latent_physics/ --batch_size 32 --no_adaptive_weights --ode_method dopri5 --no_compile --lambda_eigen_reg 0.001"
    "v15_pino|0|--output_dir outputs_v15_pino/ --batch_size 32 --no_adaptive_weights --fno_modes 12 --fno_width 32 --n_collocation_points 256 --lambda_consistency 0.1 --curriculum_start 60 --curriculum_end 120"
    "v16_obs_weighted_physics|0|--output_dir outputs_v16_obs_weighted_physics/ --batch_size 32 --no_adaptive_weights --n_collocation_points 256 --lambda_consistency 0.1 --well_influence_radius 5"
)
fi

# V17-V22: all need V0 checkpoint
ALL_JOBS+=(
    "v17_finetune_ode|1|--output_dir outputs_v17_finetune_ode/ --batch_size 32 --no_adaptive_weights --no_compile --ode_method dopri5 --lambda_eigen_reg 0.001 --pretrained_path $V0_CKPT --encoder_lr_scale 0.1"
    "v18_finetune_fno|1|--output_dir outputs_v18_finetune_fno/ --batch_size 32 --no_adaptive_weights --no_compile --fno_modes 12 --fno_width 32 --pretrained_path $V0_CKPT --encoder_lr_scale 0.1"
    "v19_perwell_mlp|1|--output_dir outputs_v19_perwell_mlp/ --batch_size 32 --no_adaptive_weights --lambda_yobs 5.0 --pretrained_path $V0_CKPT --encoder_lr_scale 0.1"
    "v20_multistep_rollout|1|--output_dir outputs_v20_multistep_rollout/ --batch_size 16 --no_adaptive_weights --nsteps 6 --pretrained_path $V0_CKPT --encoder_lr_scale 0.1"
    "v21_well_features|1|--output_dir outputs_v21_well_features/ --batch_size 32 --no_adaptive_weights --pretrained_path $V0_CKPT --encoder_lr_scale 0.1"
    "v22_full_combo|1|--output_dir outputs_v22_full_combo/ --batch_size 8 --no_adaptive_weights --no_compile --nsteps 6 --ode_method euler --fno_modes 12 --fno_width 32 --lambda_eigen_reg 0.001 --pretrained_path $V0_CKPT --encoder_lr_scale 0.05"
)

# V23-V25: no V0 dependency
ALL_JOBS+=(
    "v23_baseline_bs4|0|--output_dir outputs_v23_baseline_bs4/ --batch_size 4"
    "v24_baseline_bs8|0|--output_dir outputs_v24_baseline_bs8/ --batch_size 8"
    "v25_baseline_bs32|0|--output_dir outputs_v25_baseline_bs32/ --batch_size 32"
)

TOTAL_JOBS=${#ALL_JOBS[@]}

# ── Pre-flight checks ────────────────────────────────────────────
if $NEW_ONLY && [ ! -f "$V0_CKPT" ]; then
    # Check if any new-only job actually needs V0
    has_v0_dep=false
    for job in "${ALL_JOBS[@]}"; do
        IFS='|' read -r name needs_v0 args <<< "$job"
        if [[ "$needs_v0" == "1" ]]; then has_v0_dep=true; break; fi
    done
    if $has_v0_dep; then
        echo "WARNING: V0 checkpoint not found at $V0_CKPT"
        echo "  V0-dependent jobs (V13, V17-V22) will wait, but won't start"
        echo "  without it. V0-independent jobs (V23-V25) will run normally."
    fi
fi

echo "=========================================="
if $NEW_ONLY; then
    echo " E2CO V17-V25 (new versions only)"
else
    echo " E2CO V0-V25 (all versions)"
fi
echo " Epochs: $EPOCHS | Jobs: $TOTAL_JOBS | Max parallel: $MAX_JOBS"
echo "=========================================="
echo ""

# ── Job queue engine ─────────────────────────────────────────────
# Arrays indexed by job position
declare -A JOB_PID        # PID of running job
declare -A JOB_NAME       # version name
declare -A JOB_STATUS     # pending | waiting_v0 | running | done | failed
declare -A JOB_NEEDS_V0   # 0 or 1
declare -A JOB_ARGS       # command args
declare -A JOB_START_TIME # start timestamp

# Initialize all jobs
for i in "${!ALL_JOBS[@]}"; do
    IFS='|' read -r name needs_v0 args <<< "${ALL_JOBS[$i]}"
    JOB_NAME[$i]="$name"
    JOB_NEEDS_V0[$i]="$needs_v0"
    JOB_ARGS[$i]="$args"
    JOB_STATUS[$i]="pending"
done

V0_DONE=false
RUNNING_COUNT=0
COMPLETED=0
FAILED_LIST=""

# Get short name from version name (e.g., v17_finetune_ode -> V17)
short_name() {
    local v="${1%%_*}"
    echo "${v^^}"
}

# Get epoch progress from loss_history.csv
get_progress() {
    local vname="$1"
    local csv="outputs_${vname}/logs/loss_history.csv"
    local epoch="--"
    local loss="--"
    if [ -f "$csv" ]; then
        local last_line
        last_line=$(tail -1 "$csv" 2>/dev/null)
        if [ -n "$last_line" ] && ! echo "$last_line" | grep -q "^epoch"; then
            epoch=$(echo "$last_line" | cut -d',' -f1)
            loss=$(echo "$last_line" | cut -d',' -f2 | head -c 8)
        fi
    fi
    echo "${epoch}/${EPOCHS}(${loss})"
}

# Start a job
start_job() {
    local idx="$1"
    local name="${JOB_NAME[$idx]}"
    local args="${JOB_ARGS[$idx]}"

    python -m "versions.${name}.train" $COMMON $args \
        > "outputs_${name}.log" 2>&1 &
    JOB_PID[$idx]=$!
    JOB_STATUS[$idx]="running"
    JOB_START_TIME[$idx]=$(date +%s)
    RUNNING_COUNT=$((RUNNING_COUNT + 1))

    echo "  START  $(short_name "$name") ($name) [PID ${JOB_PID[$idx]}]"
}

# Check if V0 checkpoint is available
check_v0_ready() {
    if $V0_DONE; then return 0; fi
    if [ -f "$V0_CKPT" ]; then
        V0_DONE=true
        return 0
    fi
    return 1
}

# Print status line
print_status() {
    local status=""
    for i in "${!ALL_JOBS[@]}"; do
        if [[ "${JOB_STATUS[$i]}" == "running" ]]; then
            local sn=$(short_name "${JOB_NAME[$i]}")
            local prog=$(get_progress "${JOB_NAME[$i]}")
            status+="  ${sn}:${prog}"
        fi
    done
    if [ -n "$status" ]; then
        printf "\r\033[K[%d/%d done, %d running]%s" "$COMPLETED" "$TOTAL_JOBS" "$RUNNING_COUNT" "$status"
    fi
}

# ── Main loop ────────────────────────────────────────────────────
while [ "$COMPLETED" -lt "$TOTAL_JOBS" ]; do

    # Check for finished jobs
    for i in "${!ALL_JOBS[@]}"; do
        if [[ "${JOB_STATUS[$i]}" == "running" ]]; then
            if ! kill -0 "${JOB_PID[$i]}" 2>/dev/null; then
                # Job finished — check exit code
                wait "${JOB_PID[$i]}" 2>/dev/null
                exit_code=$?
                RUNNING_COUNT=$((RUNNING_COUNT - 1))
                COMPLETED=$((COMPLETED + 1))

                local_name="${JOB_NAME[$i]}"
                local_short=$(short_name "$local_name")
                elapsed=$(( $(date +%s) - ${JOB_START_TIME[$i]} ))
                elapsed_min=$(( elapsed / 60 ))

                if [ "$exit_code" -eq 0 ]; then
                    JOB_STATUS[$i]="done"
                    printf "\r\033[K"
                    echo "  DONE   ${local_short} ($local_name) [${elapsed_min}m]"

                    # If V0 just finished, mark it
                    if [[ "$local_name" == "v0_baseline" ]]; then
                        check_v0_ready
                    fi
                else
                    JOB_STATUS[$i]="failed"
                    FAILED_LIST+=" ${local_short}"
                    printf "\r\033[K"
                    echo "  FAIL   ${local_short} ($local_name) [exit=$exit_code, log: outputs_${local_name}.log]"
                fi
            fi
        fi
    done

    # Start new jobs if we have capacity
    check_v0_ready
    for i in "${!ALL_JOBS[@]}"; do
        if [ "$RUNNING_COUNT" -ge "$MAX_JOBS" ]; then break; fi

        if [[ "${JOB_STATUS[$i]}" == "pending" ]]; then
            if [[ "${JOB_NEEDS_V0[$i]}" == "1" ]]; then
                if $V0_DONE; then
                    start_job "$i"
                else
                    JOB_STATUS[$i]="waiting_v0"
                fi
            else
                start_job "$i"
            fi
        elif [[ "${JOB_STATUS[$i]}" == "waiting_v0" ]] && $V0_DONE; then
            start_job "$i"
        fi
    done

    # If nothing is running and we still have jobs, they must be waiting for V0
    if [ "$RUNNING_COUNT" -eq 0 ] && [ "$COMPLETED" -lt "$TOTAL_JOBS" ]; then
        echo ""
        echo "ERROR: All remaining jobs need V0 checkpoint but V0 is not available."
        echo "  Missing: $V0_CKPT"
        FAILED_LIST+=" (blocked)"
        break
    fi

    # Status update + short sleep
    print_status
    sleep 5
done

printf "\r\033[K"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Generating comparison report..."
echo "=========================================="

python -m versions.compare --base_dir . --out_dir outputs_comparison/

echo ""
echo "=========================================="
if [ -n "$FAILED_LIST" ]; then
    echo " Done! ($COMPLETED/$TOTAL_JOBS succeeded)"
    echo " Failed:$FAILED_LIST"
else
    echo " All done! $TOTAL_JOBS/$TOTAL_JOBS succeeded"
fi
echo "=========================================="
echo ""
echo " Per-version logs:  outputs_v*.log"
echo " Per-version plots: outputs_v*/plots/"
echo " Comparison:        outputs_comparison/"
