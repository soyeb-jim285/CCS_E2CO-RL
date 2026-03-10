"""V13 Physics Fine-Tuning E2CO training script.

Two-phase training:
1. Load pretrained V0 baseline weights (if available)
2. Fine-tune with physics losses using differential learning rates
"""

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.optim as optim

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from versions.shared.config import BaseConfig
from versions.shared.utils import set_seed, ensure_dirs
from versions.shared.data_loader import VersionDataLoader
from versions.shared.trainer import BaseTrainer
from versions.shared.evaluator import BaseEvaluator

from versions.v13_physics_finetune.model import PhysicsConstrainedE2CO
from versions.v13_physics_finetune.loss import PhysicsFinetuneLoss


def add_extra_args(parser):
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--pretrained_path", type=str,
                        default="outputs_v0_baseline/checkpoints/best_model.pt")
    parser.add_argument("--encoder_lr_scale", type=float, default=0.1)
    return {"finetune_epochs": 100,
            "pretrained_path": "outputs_v0_baseline/checkpoints/best_model.pt",
            "encoder_lr_scale": 0.1}


cfg = BaseConfig.from_args(extra_args_fn=add_extra_args)
cfg.version_name = "v13_physics_finetune"

print(f"=== V13 Physics Fine-Tuning E2CO ===")
print(f"Working directory: {os.getcwd()}")
print(f"Data directory:    {os.path.abspath(cfg.data_dir)}")
print(f"Output directory:  {os.path.abspath(cfg.output_dir)}")
print(f"Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
      f"lr={cfg.learning_rate}, latent_dim={cfg.latent_dim}, nsteps={cfg.nsteps}")
print(f"Performance: AMP={cfg.use_amp}, compile={cfg.use_compile}")
print(f"Pretrained path: {cfg.pretrained_path}")
print(f"Encoder LR scale: {cfg.encoder_lr_scale}")

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
model = PhysicsConstrainedE2CO(
    latent_dim=cfg.latent_dim,
    u_dim=cfg.u_dim,
    num_prod=cfg.num_prod,
    num_inj=cfg.num_inj,
    input_shape=cfg.input_shape,
    nsteps=cfg.nsteps,
).to(device)

# Try to load pretrained V0 weights
pretrained_loaded = False
if os.path.exists(cfg.pretrained_path):
    print(f"Loading pretrained V0 weights from {cfg.pretrained_path}")
    ckpt = torch.load(cfg.pretrained_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    pretrained_loaded = True
    print("  Pretrained weights loaded successfully.")
else:
    print(f"WARNING: Pretrained checkpoint not found at {cfg.pretrained_path}")
    print("  Falling back to training from scratch with physics losses.")

loss_fn = PhysicsFinetuneLoss(cfg, perm, device).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Custom trainer with differential learning rates
class FinetuneTrainer(BaseTrainer):
    def __init__(self, model, loss_fn, cfg, device, encoder_lr_scale=0.1):
        # Skip BaseTrainer.__init__'s optimizer creation — we'll make our own
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device

        # Differential learning rates
        encoder_params = list(model.encoder.parameters())
        transition_params = list(model.transition.parameters())
        decoder_params = list(model.decoder.parameters())

        encoder_ids = set(id(p) for p in encoder_params)
        transition_ids = set(id(p) for p in transition_params)
        decoder_ids = set(id(p) for p in decoder_params)

        param_groups = [
            {'params': encoder_params, 'lr': cfg.learning_rate * encoder_lr_scale},
            {'params': decoder_params, 'lr': cfg.learning_rate},
            {'params': transition_params, 'lr': cfg.learning_rate * encoder_lr_scale},
        ]
        if hasattr(loss_fn, 'adaptive') and loss_fn.adaptive is not None:
            param_groups.append({
                'params': loss_fn.adaptive.parameters(),
                'lr': cfg.adaptive_lr,
            })
        self.optimizer = optim.Adam(param_groups)

        self.use_amp = cfg.use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        if cfg.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile failed: {e}")

        self.start_epoch = 0
        self.best_loss = 1e9
        self.loss_history = []

        self.writer = None
        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=cfg.log_dir)
            except ImportError:
                pass

        self.csv_path = os.path.join(cfg.log_dir, "loss_history.csv")


if pretrained_loaded:
    print(f"Fine-tuning with differential LR: "
          f"encoder={cfg.learning_rate * cfg.encoder_lr_scale:.2e}, "
          f"decoder={cfg.learning_rate:.2e}, "
          f"transition={cfg.learning_rate * cfg.encoder_lr_scale:.2e}")
    trainer = FinetuneTrainer(model, loss_fn, cfg, device,
                               encoder_lr_scale=cfg.encoder_lr_scale)
else:
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
