"""Deep analysis of PINN-E2CO training results from outputs_light.zip."""

import csv
import os
import numpy as np

# ---- Load CSV ----
CSV_PATH = "outputs_analysis/outputs/logs/loss_history.csv"

rows = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

epochs = [int(row['epoch']) for row in rows]
n_epochs = len(epochs)

# ---- Helper ----
def get_col(name, default=None):
    vals = []
    for row in rows:
        v = row.get(name, '')
        if v == '':
            vals.append(default)
        else:
            vals.append(float(v))
    return vals

# ---- Extract losses ----
train_loss = get_col('train_loss')
eval_loss = get_col('eval_loss')

loss_names = ['rec_t0', 'rec_t1', 'l2_reg', 'trans', 'yobs',
              'pressure_pde', 'mass_conservation', 'darcy_flux']
sub_losses = {name: get_col(f'train_{name}') for name in loss_names}

sigma_names = ['rec_t0', 'rec_t1', 'l2_reg', 'trans', 'yobs',
               'pressure_pde', 'mass_conservation', 'darcy_flux']
sigmas = {name: get_col(f'sigma_{name}') for name in sigma_names}

# ---- Report ----
print("=" * 70)
print("          PINN-E2CO TRAINING ANALYSIS REPORT")
print("=" * 70)

# 1. Training overview
print(f"\n{'─' * 70}")
print("1. TRAINING OVERVIEW")
print(f"{'─' * 70}")
print(f"  Total epochs:     {n_epochs}")
print(f"  Initial loss:     {train_loss[0]:.2f}")
print(f"  Final loss:       {train_loss[-1]:.4f}")
print(f"  Best loss:        {min(train_loss):.4f} (epoch {epochs[train_loss.index(min(train_loss))]})")
print(f"  Reduction factor: {train_loss[0] / train_loss[-1]:.1f}x")

# Eval losses
eval_vals = [(e, v) for e, v in zip(epochs, eval_loss) if v is not None]
if eval_vals:
    best_eval = min(eval_vals, key=lambda x: x[1])
    print(f"\n  Eval loss (final): {eval_vals[-1][1]:.4f}")
    print(f"  Eval loss (best):  {best_eval[1]:.4f} (epoch {best_eval[0]})")

# 2. Loss spike detection
print(f"\n{'─' * 70}")
print("2. STABILITY ANALYSIS")
print(f"{'─' * 70}")

spikes = []
for i in range(1, len(train_loss)):
    if train_loss[i] > train_loss[i-1] * 5:  # 5x jump
        spikes.append((epochs[i], train_loss[i-1], train_loss[i]))

if spikes:
    print("  ⚠ LOSS SPIKES DETECTED:")
    for ep, before, after in spikes:
        print(f"    Epoch {ep}: {before:.4f} → {after:.4f} ({after/before:.1f}x jump)")
else:
    print("  No major loss spikes detected.")

# Check if loss is still decreasing at end
last_10 = train_loss[-10:]
if last_10[-1] < last_10[0]:
    pct = (1 - last_10[-1] / last_10[0]) * 100
    print(f"\n  Loss still decreasing in last 10 epochs: {pct:.1f}% reduction")
    print("  → Model has NOT converged, more epochs may help.")
else:
    print(f"\n  Loss plateaued or increasing in last 10 epochs.")

# 3. Sub-loss analysis
print(f"\n{'─' * 70}")
print("3. SUB-LOSS BREAKDOWN (Final Epoch)")
print(f"{'─' * 70}")
print(f"  {'Loss Term':<22} {'Initial':>12} {'Final':>12} {'Reduction':>10}")
print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*10}")
for name in loss_names:
    vals = sub_losses[name]
    init_v = vals[0] if vals[0] is not None else 0
    final_v = vals[-1] if vals[-1] is not None else 0
    if init_v > 0 and final_v > 0:
        ratio = f"{init_v/final_v:.1f}x"
    else:
        ratio = "N/A"
    print(f"  {name:<22} {init_v:>12.4f} {final_v:>12.6f} {ratio:>10}")

# Identify dominant losses
final_vals = {name: sub_losses[name][-1] for name in loss_names if sub_losses[name][-1] is not None}
sorted_losses = sorted(final_vals.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Dominant losses (final epoch, sorted):")
for name, val in sorted_losses:
    pct = val / sum(final_vals.values()) * 100
    bar = '█' * int(pct / 2)
    print(f"    {name:<22} {val:>12.4f}  ({pct:5.1f}%) {bar}")

# 4. Adaptive weight analysis
print(f"\n{'─' * 70}")
print("4. ADAPTIVE WEIGHT ANALYSIS (σ values)")
print(f"{'─' * 70}")
print(f"  Higher σ → LOWER effective weight (weight = 1/(2σ²))")
print(f"  {'Loss Term':<22} {'Final σ':>10} {'Eff. Weight':>12} {'Status'}")
print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*20}")

final_sigmas = {}
for name in sigma_names:
    vals = sigmas[name]
    # Get the last non-None value
    last_val = None
    for v in reversed(vals):
        if v is not None:
            last_val = v
            break
    if last_val is not None:
        final_sigmas[name] = last_val
        eff_weight = 1.0 / (2.0 * last_val ** 2)
        if last_val > 10:
            status = "⚠ SUPPRESSED"
        elif last_val > 3:
            status = "↓ Downweighted"
        elif last_val < 0.3:
            status = "↑ Amplified"
        else:
            status = "~ Normal"
        print(f"  {name:<22} {last_val:>10.3f} {eff_weight:>12.6f} {status}")

# 5. Key problems
print(f"\n{'─' * 70}")
print("5. CRITICAL ISSUES IDENTIFIED")
print(f"{'─' * 70}")

issues = []

# Check reconstruction dominance
if 'rec_t0' in final_vals and 'rec_t1' in final_vals:
    rec_total = final_vals['rec_t0'] + final_vals['rec_t1']
    total = sum(final_vals.values())
    rec_pct = rec_total / total * 100
    if rec_pct > 95:
        issues.append(
            f"RECONSTRUCTION DOMINANCE: rec_t0 + rec_t1 = {rec_pct:.1f}% of total loss.\n"
            f"    The reconstruction losses (~{rec_total:.0f}) are orders of magnitude larger\n"
            f"    than physics losses (~{final_vals.get('pressure_pde', 0):.4f}). The physics\n"
            f"    losses have essentially NO influence on training."
        )

# Check adaptive weight collapse
if final_sigmas:
    max_sigma_name = max(final_sigmas, key=final_sigmas.get)
    max_sigma = final_sigmas[max_sigma_name]
    if max_sigma > 15:
        issues.append(
            f"ADAPTIVE WEIGHT COLLAPSE: σ_{max_sigma_name} = {max_sigma:.1f}\n"
            f"    Effective weight = {1/(2*max_sigma**2):.6f}. The adaptive weighting\n"
            f"    is suppressing {max_sigma_name} instead of balancing losses.\n"
            f"    This often indicates the learning rate for adaptive params is too high,\n"
            f"    or that the loss magnitudes are too different to balance automatically."
        )

# Check pressure bias from error histogram observation
if 'darcy_flux' in final_vals:
    darcy = final_vals['darcy_flux']
    if darcy > 100:
        issues.append(
            f"HIGH DARCY FLUX LOSS: {darcy:.1f}\n"
            f"    Indicates predicted pressure gradients diverge significantly from true.\n"
            f"    Consistent with the ~200 psia systematic pressure bias visible in plots."
        )

# Check training instability
if spikes:
    issues.append(
        f"TRAINING INSTABILITY: {len(spikes)} major loss spike(s) detected.\n"
        f"    The spike at epoch ~175 caused total loss to jump from ~15 to ~2000+.\n"
        f"    This is likely caused by the adaptive weights diverging (σ_trans hit 39).\n"
        f"    The model partially recovered but likely settled in a worse minimum."
        )

# Check if model hasn't converged
if last_10[-1] < last_10[0] * 0.95:
    issues.append(
        f"NOT CONVERGED: Loss decreased {(1-last_10[-1]/last_10[0])*100:.1f}% in last 10 epochs.\n"
        f"    Training should continue for more epochs (try 500-1000)."
    )

for i, issue in enumerate(issues, 1):
    print(f"\n  [{i}] {issue}")

# 6. Performance metrics from plots (observed)
print(f"\n{'─' * 70}")
print("6. EVALUATION METRICS (from plots)")
print(f"{'─' * 70}")

print("""
  SATURATION (normalized [0,1]):
    Case 77 RMSE: 0.3569    MAE: 0.2116
    Per-timestep RMSE: 0.21 (t=0) → 0.47 (t=19)  ← grows 2.2x
    Error distribution: bimodal, spike at -1.0 (predicting 0 where true ~1)

  PRESSURE (psia):
    Case 77 RMSE: 208.28    MAE: 188.69
    Per-timestep RMSE: ~200-250 psia (nearly flat)
    Error distribution: strong POSITIVE bias, centered ~+250 psia
    → Model systematically UNDER-PREDICTS pressure drop

  WELL OUTPUTS (case 77):
    Well 1 (water): RMSE=241.72, MAE=210.28  ← NEGATIVE rates predicted!
    Well 5 (water): RMSE=359.70, MAE=195.20  ← better shape, offset errors
    Well 11 (gas):  RMSE=135.15, MAE=106.32  ← both near zero (no signal)

  R² CROSS-PLOTS (cumulative):
    Well 1: R² = -89.132   ← CATASTROPHIC (worse than mean prediction)
    Well 5: R² =   0.166   ← Poor (should be > 0.9 for a good model)
""")

# 7. Convergence analysis
print(f"{'─' * 70}")
print("7. CONVERGENCE RATE")
print(f"{'─' * 70}")

milestones = [1, 10, 50, 100, 150, 200]
print(f"  {'Epoch':>6} {'Total Loss':>12} {'rec_t0':>12} {'rec_t1':>12} {'pressure_pde':>14} {'darcy_flux':>12}")
for ep in milestones:
    idx = ep - 1
    if idx < len(rows):
        tl = train_loss[idx]
        r0 = sub_losses['rec_t0'][idx] or 0
        r1 = sub_losses['rec_t1'][idx] or 0
        pp = sub_losses['pressure_pde'][idx] or 0
        df = sub_losses['darcy_flux'][idx] or 0
        print(f"  {ep:>6} {tl:>12.2f} {r0:>12.2f} {r1:>12.2f} {pp:>14.4f} {df:>12.2f}")

# 8. Recommendations
print(f"\n{'─' * 70}")
print("8. RECOMMENDATIONS")
print(f"{'─' * 70}")
print("""
  The model trained for 200 epochs but shows SEVERE issues:

  A) IMMEDIATE FIX — Disable adaptive weights:
     --no_adaptive_weights
     The adaptive weighting is the likely cause of the instability spike at
     epoch ~175 and the systematic pressure bias. The σ_trans grew to 39,
     effectively zeroing out the transition loss, which broke sequential
     prediction.

  B) REDUCE PHYSICS LOSS STRENGTH — Current lambdas too aggressive:
     --lambda_pressure_pde 0.001 --lambda_mass_conservation 0.001
     --lambda_darcy_flux 0.001
     The physics losses have very different magnitudes from data losses.
     Start with 10x lower lambdas and increase gradually.

  C) INCREASE EPOCHS — Model hasn't converged:
     --epochs 500
     Loss was still decreasing at epoch 200.

  D) REDUCE BATCH SIZE — Original used batch_size=4:
     --batch_size 4  (or 8 for GPU utilization)
     batch_size=32 may be too large for this small dataset, causing
     poor generalization. The original MS-E2C used batch_size=4.

  E) CHECK PRESSURE CHANNEL — Possible data issue:
     The pressure error histogram shows a strong +250 psia bias across
     ALL timesteps (not growing over time). This suggests the model may
     be learning a wrong offset for the pressure channel. Check:
     - Pressure normalization (p_min=2200, p_max=4069.2)
     - Whether channel 0 is pressure or saturation in the data
     - Decoder output activation (should not clip pressure)

  SUGGESTED NEXT RUN:
     python pinn_train.py --epochs 500 --batch_size 8 \\
       --no_adaptive_weights \\
       --lambda_pressure_pde 0.001 \\
       --lambda_mass_conservation 0.001 \\
       --lambda_darcy_flux 0.001
""")

print("=" * 70)
print("Analysis complete.")
print("=" * 70)
