"""Base configuration dataclass for all versions."""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BaseConfig:
    # --- Version identification ---
    version_name: str = "base"

    # --- Data paths ---
    data_dir: str = "data/"
    output_dir: str = "outputs/"
    checkpoint_dir: str = ""  # auto-derived
    log_dir: str = ""         # auto-derived
    plot_dir: str = ""        # auto-derived

    # --- Data files ---
    state_file: str = "states_norm_slt.mat"
    ctrl_file: str = "controls_norm_slt.mat"
    yobs_file: str = "rate_norm_slt.mat"
    perm_file: str = "TRUE_PERM_64by220.mat"
    cond: str = "SC"

    # --- Model architecture ---
    latent_dim: int = 20
    u_dim: int = 9
    num_prod: int = 5
    num_inj: int = 4
    n_channels: int = 2
    Nx: int = 64
    Ny: int = 64
    nsteps: int = 2
    input_shape: Tuple[int, int, int] = (2, 64, 64)

    # --- Training ---
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 2e-4
    seed: int = 1010

    # --- Performance ---
    use_amp: bool = True
    use_compile: bool = True
    use_tensorboard: bool = True
    gradient_accumulation_steps: int = 1
    eval_every: int = 5
    eval_batch_size: int = 256

    # --- Checkpoint / resume ---
    checkpoint_every: int = 20
    resume: bool = True

    # --- Physical constants ---
    p_min: float = 2200.0
    p_max: float = 4069.2
    Q_min: float = 0.0
    Q_max_w: float = 3151.0
    Q_max_g: float = 1.2e6

    # --- Well locations (row, col) ---
    inj_loc: List[List[int]] = field(
        default_factory=lambda: [[18, 2], [48, 20], [46, 32], [27, 53]]
    )
    prod_loc: List[List[int]] = field(
        default_factory=lambda: [[4, 16], [26, 16], [30, 31], [36, 45], [7, 49]]
    )

    # --- Physics loss lambdas (used by physics versions) ---
    lambda_rec_t0: float = 1.0
    lambda_rec_t1: float = 1.0
    lambda_l2_reg: float = 1.0
    lambda_trans: float = 1.0
    lambda_yobs: float = 1.0
    lambda_pressure_pde: float = 0.01
    lambda_mass_conservation: float = 0.01
    lambda_darcy_flux: float = 0.01

    # --- Adaptive weighting ---
    use_adaptive_weights: bool = True
    adaptive_lr: float = 1e-3

    # --- Physics PDE params ---
    porosity: float = 0.2
    total_compressibility: float = 1e-5
    dt_physical: float = 20.0

    # --- Evaluation ---
    test_cases_per_batch: int = 25
    ind_case: List[int] = field(
        default_factory=lambda: [10, 25, 77, 97]
    )
    eval_case: int = 77

    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = self.output_dir + "checkpoints/"
        if not self.log_dir:
            self.log_dir = self.output_dir + "logs/"
        if not self.plot_dir:
            self.plot_dir = self.output_dir + "plots/"

    @classmethod
    def from_args(cls, args=None, extra_args_fn=None):
        """Create config from CLI arguments with optional extra args."""
        parser = argparse.ArgumentParser(description="E2CO Version Training")
        parser.add_argument("--data_dir", type=str, default=cls.data_dir)
        parser.add_argument("--output_dir", type=str, default=cls.output_dir)
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--batch_size", type=int, default=cls.batch_size)
        parser.add_argument("--learning_rate", type=float, default=cls.learning_rate)
        parser.add_argument("--latent_dim", type=int, default=cls.latent_dim)
        parser.add_argument("--nsteps", type=int, default=cls.nsteps)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--checkpoint_every", type=int, default=cls.checkpoint_every)
        parser.add_argument("--eval_every", type=int, default=cls.eval_every)
        parser.add_argument("--no_resume", action="store_true")
        parser.add_argument("--no_adaptive_weights", action="store_true")
        parser.add_argument("--no_amp", action="store_true")
        parser.add_argument("--no_compile", action="store_true")
        parser.add_argument("--grad_accum", type=int, default=cls.gradient_accumulation_steps)
        parser.add_argument("--lambda_pressure_pde", type=float, default=cls.lambda_pressure_pde)
        parser.add_argument("--lambda_mass_conservation", type=float, default=cls.lambda_mass_conservation)
        parser.add_argument("--lambda_darcy_flux", type=float, default=cls.lambda_darcy_flux)
        parser.add_argument("--n_channels", type=int, default=cls.n_channels)

        # Allow version-specific extra args
        extra_defaults = {}
        if extra_args_fn is not None:
            extra_defaults = extra_args_fn(parser)

        parsed = parser.parse_args(args)

        out = parsed.output_dir
        if not out.endswith("/"):
            out += "/"

        cfg = cls(
            data_dir=parsed.data_dir,
            output_dir=out,
            epochs=parsed.epochs,
            batch_size=parsed.batch_size,
            learning_rate=parsed.learning_rate,
            latent_dim=parsed.latent_dim,
            nsteps=parsed.nsteps,
            seed=parsed.seed,
            checkpoint_every=parsed.checkpoint_every,
            eval_every=parsed.eval_every,
            resume=not parsed.no_resume,
            use_adaptive_weights=not parsed.no_adaptive_weights,
            use_amp=not parsed.no_amp,
            use_compile=not parsed.no_compile,
            gradient_accumulation_steps=parsed.grad_accum,
            lambda_pressure_pde=parsed.lambda_pressure_pde,
            lambda_mass_conservation=parsed.lambda_mass_conservation,
            lambda_darcy_flux=parsed.lambda_darcy_flux,
            n_channels=parsed.n_channels,
            input_shape=(parsed.n_channels, 64, 64),
        )

        # Apply extra args
        for key, default_val in extra_defaults.items():
            val = getattr(parsed, key, default_val)
            setattr(cfg, key, val)

        return cfg
