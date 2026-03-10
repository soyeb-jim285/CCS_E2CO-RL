"""Utility functions: seeding, metrics, denormalization."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 1010):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # NOTE: we do NOT set deterministic=True or benchmark=False here.
        # cudnn.benchmark=True is set in pinn_train.py for performance.
        # For reproducibility at the cost of speed, uncomment these:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# --------------- Metrics ---------------

def compute_rmse(pred, true):
    """Element-wise RMSE over all dimensions except batch."""
    pred_np = _to_numpy(pred)
    true_np = _to_numpy(true)
    return np.sqrt(np.mean((pred_np - true_np) ** 2))


def compute_mae(pred, true):
    pred_np = _to_numpy(pred)
    true_np = _to_numpy(true)
    return np.mean(np.abs(pred_np - true_np))


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --------------- Denormalization ---------------

def denormalize_pressure(p_norm, p_min=2200.0, p_max=4069.2):
    """Convert normalized pressure [0,1] → physical [p_min, p_max] psia."""
    return p_norm * (p_max - p_min) + p_min


def denormalize_saturation(s_norm, s_min=0.0, s_max=1.0):
    """Convert normalized saturation [0,1] → physical [s_min, s_max]."""
    return s_norm * (s_max - s_min) + s_min


def denormalize_rates(yobs, num_prod=5, num_inj=4,
                      Q_max_w=3151.0, Q_max_g=1.2e6,
                      p_min=2200.0, p_max=4069.2):
    """Denormalize observation vector: water rates, gas rates, injector BHP.

    yobs shape: (..., 2*num_prod + num_inj) = (..., 14)
    Returns a copy with physical units.
    """
    out = yobs.clone() if isinstance(yobs, torch.Tensor) else yobs.copy()
    # Water production rates [0, Q_max_w] STB/Day
    out[..., :num_prod] = out[..., :num_prod] * Q_max_w
    # Gas production rates [0, Q_max_g] ft³/Day
    out[..., num_prod:2 * num_prod] = out[..., num_prod:2 * num_prod] * Q_max_g
    # Injector BHP [p_min, p_max] psia
    out[..., 2 * num_prod:] = out[..., 2 * num_prod:] * (p_max - p_min) + p_min
    return out
