# %% [markdown]
# # PINN-E2CO Training Script
# Physics-Informed Neural Network variant of MS-E2C for CO2 storage.
# Can be run as a script (`python pinn_train.py`) or as a Jupyter notebook.
#
# Expected RunPod layout:
#   /workspace/
#   ├── pinn_e2co/        ← this code (zip and upload)
#   │   ├── pinn_train.py ← run this
#   │   └── ...
#   ├── data/             ← upload .mat files here
#   └── outputs/          ← created automatically
#
# Run:  cd /workspace/pinn_e2co && python pinn_train.py --epochs 5

# %% [code] Configuration
import sys
import os

# Anchor all relative paths to this script's directory (pinn_e2co/).
# This means "../data/" always resolves to the sibling "data/" directory
# regardless of where you invoke the script from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Add parent directory to path so we can import pinn_e2co as a package
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

import torch
import numpy as np

# GPU performance: enable before anything else
torch.backends.cudnn.benchmark = True  # auto-tune convolution algorithms (fixed input size)
torch.set_float32_matmul_precision('high')  # use TF32 on Ampere+ GPUs

from pinn_e2co.config import PINNConfig
from pinn_e2co.model import PINNE2CO
from pinn_e2co.physics_loss import PINNLoss
from pinn_e2co.data_loader import PINNDataLoader
from pinn_e2co.trainer import PINNTrainer
from pinn_e2co.evaluator import Evaluator
from pinn_e2co.utils import set_seed, ensure_dirs

# Parse CLI args (when run as script) or use defaults (when run as notebook)
try:
    cfg = PINNConfig.from_args()
except SystemExit:
    # Running in notebook — use defaults
    cfg = PINNConfig()

print(f"Working directory: {os.getcwd()}")
print(f"Data directory:    {os.path.abspath(cfg.data_dir)}")
print(f"Output directory:  {os.path.abspath(cfg.output_dir)}")
print(f"Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
      f"lr={cfg.learning_rate}, latent_dim={cfg.latent_dim}, nsteps={cfg.nsteps}")
print(f"Performance: AMP={cfg.use_amp}, compile={cfg.use_compile}, "
      f"grad_accum={cfg.gradient_accumulation_steps}")
print(f"Physics lambdas: pressure_pde={cfg.lambda_pressure_pde}, "
      f"mass_conservation={cfg.lambda_mass_conservation}, "
      f"darcy_flux={cfg.lambda_darcy_flux}")
print(f"Adaptive weights: {cfg.use_adaptive_weights}")

# %% [code] Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

set_seed(cfg.seed)
ensure_dirs(cfg.output_dir, cfg.checkpoint_dir, cfg.log_dir, cfg.plot_dir)

# %% [code] Load Data
print("Loading data...")
data_loader = PINNDataLoader(cfg)
train_data, eval_data, perm = data_loader.load_all(device)

print(f"Train samples: {train_data['num_train']}")
print(f"Eval samples: {eval_data['num_eval']}")
print(f"Permeability shape: {perm.shape}")

# %% [code] Build Model + Loss
print("Building model...")
model = PINNE2CO(
    latent_dim=cfg.latent_dim,
    u_dim=cfg.u_dim,
    num_prod=cfg.num_prod,
    num_inj=cfg.num_inj,
    input_shape=cfg.input_shape,
    nsteps=cfg.nsteps,
).to(device)

loss_fn = PINNLoss(cfg, perm, device).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# %% [code] Training
trainer = PINNTrainer(model, loss_fn, cfg, device)
trainer.try_resume()

print("Starting training...")
trainer.train(train_data, eval_data)

# %% [code] Evaluation
print("Running evaluation...")

# Load best model
best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
if os.path.exists(best_path):
    print(f"Loading best model from {best_path}")
    PINNE2CO.load_checkpoint(best_path, model)
    model.to(device)

# Load test data
test_data, perm_test = data_loader.load_test_data(device)

# Run sequential evaluation
evaluator = Evaluator(model, loss_fn, cfg, device)
results = evaluator.run_sequential_eval(test_data, perm_test)

# Generate all plots
evaluator.generate_all_plots(test_data, perm_test, results)

print("Done! Check outputs in:", os.path.abspath(cfg.plot_dir))
