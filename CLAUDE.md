# CLAUDE.md — Project Instructions for Claude Code

## Project
Physics-Constrained Embed-to-Control (PC-E2CO) for CO2 storage surrogate modeling.
Based on the MS-E2C architecture with added physics-based loss regularization.

## Key Directories
- `pinn_e2co/` — Main codebase (model, training, evaluation, physics losses)
- `data/` — Training data (.mat files, gitignored, ~835MB)
- `outputs/` — Training outputs (plots, logs, checkpoints — gitignored)

## Commit Rules
- NEVER add Co-Authored-By or any co-author attribution to commits
- Keep commit messages concise and descriptive

## Architecture
- This is NOT a true PINN — it is a physics-constrained surrogate model
- The neural network (encoder/decoder/transition) is purely data-driven
- Physics is enforced only through loss function regularization terms
- Correct terminology: "physics-constrained E2CO" or "PC-E2CO"

## Physics Losses (in loss function only)
1. Pressure diffusion PDE residual (finite differences)
2. CO2 mass conservation residual (finite differences)
3. Darcy flux consistency

## Tech Stack
- PyTorch 2.0+, mixed precision (AMP), torch.compile
- Training on RunPod GPU instances
- Data: 64x64 grid, 2 channels (CO2 mole fraction + pressure), 20 timesteps
