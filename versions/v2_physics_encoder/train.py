"""V2: Physics Encoder E2CO — Training entry point.

Run as: python -m versions.v2_physics_encoder.train [--epochs N] [--batch_size N] ...
"""

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from versions.shared.config import BaseConfig
from versions.shared.utils import set_seed, ensure_dirs
from versions.shared.data_loader import VersionDataLoader
from versions.shared.trainer import BaseTrainer
from versions.shared.evaluator import BaseEvaluator

from versions.v2_physics_encoder.model import PhysicsEncoderE2CO
from versions.v2_physics_encoder.loss import PhysicsLoss

# --- Config ---
cfg = BaseConfig.from_args()
cfg.version_name = "v2_physics_encoder"
cfg.batch_size = cfg.batch_size if cfg.batch_size != BaseConfig.batch_size else 32

print(f"=== V2: Physics Encoder E2CO ===")
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

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

set_seed(cfg.seed)
ensure_dirs(cfg.output_dir, cfg.checkpoint_dir, cfg.log_dir, cfg.plot_dir)

# --- Load Data ---
print("Loading data...")
data_loader = VersionDataLoader(cfg)
train_data, eval_data, perm = data_loader.load_all(device)

print(f"Train samples: {train_data['num_train']}")
print(f"Eval samples: {eval_data['num_eval']}")
print(f"Permeability shape: {perm.shape}")

# --- Build Model + Loss ---
print("Building model...")
model = PhysicsEncoderE2CO(
    latent_dim=cfg.latent_dim,
    u_dim=cfg.u_dim,
    num_prod=cfg.num_prod,
    num_inj=cfg.num_inj,
    input_shape=cfg.input_shape,
    nsteps=cfg.nsteps,
).to(device)

loss_fn = PhysicsLoss(cfg, perm, device).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# --- Training ---
trainer = BaseTrainer(model, loss_fn, cfg, device)
trainer.try_resume()

print("Starting training...")
train_start = time.time()
trainer.train(train_data, eval_data)
train_time = time.time() - train_start

# --- Evaluation ---
print("Running evaluation...")

# Load best model
best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
if os.path.exists(best_path):
    print(f"Loading best model from {best_path}")
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)

# Load test data
test_data, perm_test = data_loader.load_test_data(device)

# Run sequential evaluation
evaluator = BaseEvaluator(model, loss_fn, cfg, device)
results = evaluator.run_sequential_eval(test_data, perm_test)

# Save metrics and plots
evaluator.save_summary_metrics(test_data, results, train_time=train_time)
evaluator.generate_all_plots(test_data, perm_test, results)

print("Done! Check outputs in:", os.path.abspath(cfg.plot_dir))
