"""V15 PINO E2CO training script."""

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

from versions.v15_pino.model import PINOE2CO
from versions.v15_pino.loss import PINOLoss


def add_extra_args(parser):
    parser.add_argument("--fno_modes", type=int, default=12)
    parser.add_argument("--fno_width", type=int, default=32)
    parser.add_argument("--n_collocation_points", type=int, default=256)
    parser.add_argument("--lambda_consistency", type=float, default=0.1)
    parser.add_argument("--curriculum_start", type=int, default=60)
    parser.add_argument("--curriculum_end", type=int, default=120)
    return {"fno_modes": 12, "fno_width": 32, "n_collocation_points": 256,
            "lambda_consistency": 0.1, "curriculum_start": 60, "curriculum_end": 120}


cfg = BaseConfig.from_args(extra_args_fn=add_extra_args)
cfg.version_name = "v15_pino"
cfg.use_compile = False  # autograd incompatible
cfg.use_amp = False      # cfloat + autograd conflicts

print(f"=== V15 PINO E2CO ===")
print(f"Working directory: {os.getcwd()}")
print(f"Data directory:    {os.path.abspath(cfg.data_dir)}")
print(f"Output directory:  {os.path.abspath(cfg.output_dir)}")
print(f"Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
      f"lr={cfg.learning_rate}, latent_dim={cfg.latent_dim}, nsteps={cfg.nsteps}")
print(f"FNO: modes={cfg.fno_modes}, width={cfg.fno_width}")
print(f"Performance: AMP={cfg.use_amp}, compile={cfg.use_compile}")
print(f"Collocation points: {cfg.n_collocation_points}")
print(f"Curriculum: start={cfg.curriculum_start}, end={cfg.curriculum_end}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

set_seed(cfg.seed)
ensure_dirs(cfg.output_dir, cfg.checkpoint_dir, cfg.log_dir, cfg.plot_dir)

print("Loading data...")
data_loader = VersionDataLoader(cfg)
train_data, eval_data, perm = data_loader.load_all(device)

print(f"Train samples: {train_data['num_train']}")
print(f"Eval samples: {eval_data['num_eval']}")

print("Building model...")
model = PINOE2CO(
    latent_dim=cfg.latent_dim,
    u_dim=cfg.u_dim,
    num_prod=cfg.num_prod,
    num_inj=cfg.num_inj,
    input_shape=cfg.input_shape,
    nsteps=cfg.nsteps,
    fno_modes=cfg.fno_modes,
    fno_width=cfg.fno_width,
    n_collocation_points=cfg.n_collocation_points,
).to(device)

loss_fn = PINOLoss(
    cfg, perm, device,
    lambda_consistency=cfg.lambda_consistency,
    curriculum_start=cfg.curriculum_start,
    curriculum_end=cfg.curriculum_end,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

trainer = BaseTrainer(model, loss_fn, cfg, device)
trainer.try_resume()

print("Starting training...")
train_start = time.time()
trainer.train(train_data, eval_data)
train_time = time.time() - train_start
print(f"Training completed in {train_time:.1f}s")

print("Running evaluation...")
best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
if os.path.exists(best_path):
    print(f"Loading best model from {best_path}")
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)

test_data, perm_test = data_loader.load_test_data(device)
evaluator = BaseEvaluator(model, loss_fn, cfg, device)
results = evaluator.run_sequential_eval(test_data, perm_test)
evaluator.save_summary_metrics(test_data, results, train_time=train_time)
evaluator.generate_all_plots(test_data, perm_test, results)

print("Done! Check outputs in:", os.path.abspath(cfg.plot_dir))
