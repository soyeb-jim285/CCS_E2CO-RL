"""V1 Physics-Constrained loss — 5 data + 3 physics losses with adaptive weighting."""

from collections import OrderedDict

import torch
import torch.nn as nn


def _reconstruction_loss(x, x_rec):
    v = 0.1
    diff = x.reshape(x.size(0), -1) - x_rec.reshape(x_rec.size(0), -1)
    return torch.mean(torch.sum(diff * diff * (0.5 / v), dim=-1))


def _l2_reg_loss(z):
    return torch.mean(torch.sum(0.5 * z * z, dim=-1))


class AdaptiveLossWeights(nn.Module):
    """Learnable loss weights via log-variance parameterization.

    total = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)
    Params: log_sigma_i (one per loss term).
    """
    LOSS_NAMES = [
        'rec_t0', 'rec_t1', 'l2_reg', 'trans',
        'yobs', 'pressure_pde', 'mass_conservation', 'darcy_flux',
    ]

    def __init__(self, num_terms=8):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(num_terms))

    def forward(self, losses_dict):
        """Combine losses with adaptive weighting. Returns total_loss scalar."""
        total = torch.zeros(1, device=self.log_sigmas.device)
        for i, (name, loss_val) in enumerate(losses_dict.items()):
            log_s = self.log_sigmas[i]
            precision = torch.exp(-2 * log_s)
            total = total + (0.5 * precision * loss_val + log_s)
        return total.squeeze(0)

    def get_weight_info(self, losses_dict):
        """Get detailed weight info — only call at logging time."""
        info = {}
        with torch.no_grad():
            for i, (name, loss_val) in enumerate(losses_dict.items()):
                log_s = self.log_sigmas[i]
                info[name] = {
                    'raw_loss': loss_val.item(),
                    'sigma': torch.exp(log_s).item(),
                }
        return info

    def get_sigmas(self):
        return torch.exp(self.log_sigmas).detach().cpu().numpy()


class PhysicsLoss(nn.Module):
    """Combined data-driven + physics-informed loss for PC-E2CO.

    8 loss terms:
      1. rec_t0:             reconstruction at t=0
      2. rec_t1:             reconstruction at t=1..K
      3. l2_reg:             0.5 * ||z0||^2
      4. trans:              0.5 * ||z_true - z_pred||^2
      5. yobs:               observation match
      6. pressure_pde:       pressure diffusion PDE residual
      7. mass_conservation:  CO2 mass conservation residual
      8. darcy_flux:         Darcy flux consistency
    """

    def __init__(self, cfg, perm_log, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Permeability as buffer: k = 10^(log_perm) [mD]
        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)  # (1, Nx, Ny)

        # Precompute harmonic transmissibility (used by pressure_pde + mass_conservation)
        self.register_buffer('tran_x',
            2.0 / (1.0 / self.perm[:, 1:, :] + 1.0 / self.perm[:, :-1, :]))
        self.register_buffer('tran_y',
            2.0 / (1.0 / self.perm[:, :, 1:] + 1.0 / self.perm[:, :, :-1]))

        # Precompute inverse-sum transmissibility for Darcy flux
        self.register_buffer('darcy_tran_x',
            1.0 / (1.0 / self.perm[:, 1:, :] + 1.0 / self.perm[:, :-1, :]))
        self.register_buffer('darcy_tran_y',
            1.0 / (1.0 / self.perm[:, :, 1:] + 1.0 / self.perm[:, :, :-1]))

        # Precompute physics constants
        self._phi_ct_over_dt = cfg.porosity * cfg.total_compressibility / cfg.dt_physical
        self._phi_over_dt = cfg.porosity / cfg.dt_physical

        # Adaptive weights
        self.use_adaptive = cfg.use_adaptive_weights
        if self.use_adaptive:
            self.adaptive = AdaptiveLossWeights(num_terms=8)
        else:
            self.adaptive = None

        # Static lambdas as tensor for fast weighted sum
        self._lambda_keys = [
            'rec_t0', 'rec_t1', 'l2_reg', 'trans',
            'yobs', 'pressure_pde', 'mass_conservation', 'darcy_flux',
        ]
        lambda_vals = [
            cfg.lambda_rec_t0, cfg.lambda_rec_t1, cfg.lambda_l2_reg,
            cfg.lambda_trans, cfg.lambda_yobs, cfg.lambda_pressure_pde,
            cfg.lambda_mass_conservation, cfg.lambda_darcy_flux,
        ]
        self.register_buffer('_lambdas', torch.tensor(lambda_vals, dtype=torch.float32))

    def forward(self, pred):
        """Compute all 8 loss terms. Returns (total_loss, losses_stack).

        losses_stack is a 1D tensor with 8 elements.
        """
        x0 = pred['x0']
        x0_rec = pred['x0_rec']
        z0 = pred['z0']
        X_next_pred = pred['X_next_pred']
        X_next = pred['X_next']
        Z_next_pred = pred['Z_next_pred']
        Z_next = pred['Z_next']
        Y_next_pred = pred['Y_next_pred']
        Y = pred['Y']

        # 1. Reconstruction at t=0
        loss_rec_t0 = _reconstruction_loss(x0, x0_rec)

        # 2. Reconstruction at t=1..K
        loss_rec_t1 = _reconstruction_loss(X_next[0], X_next_pred[0])
        for i in range(1, len(X_next)):
            loss_rec_t1 = loss_rec_t1 + _reconstruction_loss(X_next[i], X_next_pred[i])

        # 3. L2 reg
        loss_l2_reg = _l2_reg_loss(z0)

        # 4. Transition
        loss_trans = _l2_reg_loss(Z_next[0] - Z_next_pred[0])
        for i in range(1, len(Z_next)):
            loss_trans = loss_trans + _l2_reg_loss(Z_next[i] - Z_next_pred[i])

        # 5. Observation
        loss_yobs = _l2_reg_loss(Y[0] - Y_next_pred[0])
        for i in range(1, len(Y)):
            loss_yobs = loss_yobs + _l2_reg_loss(Y[i] - Y_next_pred[i])

        # 6. Pressure PDE
        x_ref = x0
        loss_pde = self._pressure_pde_residual(x_ref, X_next_pred[0])
        for i in range(1, len(X_next_pred)):
            loss_pde = loss_pde + self._pressure_pde_residual(X_next[i - 1], X_next_pred[i])

        # 7. Mass conservation
        x_ref = x0
        loss_mass = self._mass_conservation_residual(x_ref, X_next_pred[0])
        for i in range(1, len(X_next_pred)):
            loss_mass = loss_mass + self._mass_conservation_residual(X_next_pred[i - 1], X_next_pred[i])

        # 8. Darcy flux
        loss_darcy = self._darcy_flux_loss(X_next[0], X_next_pred[0])
        for i in range(1, len(X_next)):
            loss_darcy = loss_darcy + self._darcy_flux_loss(X_next[i], X_next_pred[i])

        # Stack for fast weighted sum
        losses_stack = torch.stack([
            loss_rec_t0, loss_rec_t1, loss_l2_reg, loss_trans,
            loss_yobs, loss_pde, loss_mass, loss_darcy,
        ])

        if self.use_adaptive and self.adaptive is not None:
            losses_dict = OrderedDict(zip(self._lambda_keys, losses_stack))
            total_loss = self.adaptive(losses_dict)
        else:
            total_loss = (self._lambdas * losses_stack).sum()

        return total_loss, losses_stack

    def losses_to_dict(self, losses_stack):
        """Convert losses stack to dict with .item() calls — only for logging."""
        vals = losses_stack.detach()
        return {name: vals[i].item() for i, name in enumerate(self._lambda_keys)}

    @property
    def num_loss_terms(self):
        return 8

    # ---- Physics loss functions ----

    def _pressure_pde_residual(self, x_t, x_t1):
        """Pressure diffusion PDE residual on interior 62x62 cells."""
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]

        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)

        flux_x = self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]

        flux_y = self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]

        residual = dp_dt[:, :, 1:-1, 1:-1] - (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])
        return torch.mean(residual * residual)

    def _mass_conservation_residual(self, x_t, x_t1):
        """CO2 mass conservation residual on interior 62x62 cells."""
        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        p_t1 = x_t1[:, -1:, :, :]

        accum = self._phi_over_dt * (c_t1 - c_t)

        vx = -self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        vy = -self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])

        cx_avg = 0.5 * (c_t1[:, :, 1:, :] + c_t1[:, :, :-1, :])
        cy_avg = 0.5 * (c_t1[:, :, :, 1:] + c_t1[:, :, :, :-1])

        flux_x = cx_avg * vx
        flux_y = cy_avg * vy

        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]

        residual = accum[:, :, 1:-1, 1:-1] + (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])
        return torch.mean(residual * residual)

    def _darcy_flux_loss(self, x_true, x_pred):
        """Darcy flux consistency — uses precomputed transmissibility buffers."""
        p = x_true[:, -1:, :, :]
        p_pred = x_pred[:, -1:, :, :]

        tx = self.darcy_tran_x
        ty = self.darcy_tran_y

        flux_x = (p[:, :, 1:, :] - p[:, :, :-1, :]) * tx
        flux_y = (p[:, :, :, 1:] - p[:, :, :, :-1]) * ty
        flux_x_pred = (p_pred[:, :, 1:, :] - p_pred[:, :, :-1, :]) * tx
        flux_y_pred = (p_pred[:, :, :, 1:] - p_pred[:, :, :, :-1]) * ty

        loss_x = torch.sum(torch.abs(flux_x - flux_x_pred).reshape(p.size(0), -1), dim=-1)
        loss_y = torch.sum(torch.abs(flux_y - flux_y_pred).reshape(p.size(0), -1), dim=-1)

        return torch.mean(loss_x + loss_y)

    # ---- Checkpointing ----

    def get_adaptive_state(self):
        if self.adaptive is not None:
            return self.adaptive.state_dict()
        return None

    def load_adaptive_state(self, state):
        if self.adaptive is not None and state is not None:
            self.adaptive.load_state_dict(state)

    # ---- Visualization ----

    def compute_physics_residuals(self, x_t, x_t1):
        """Per-pixel residuals for visualization (call with no_grad)."""
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]
        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)
        flux_x = self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        flux_y = self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]
        pres_residual = dp_dt[:, :, 1:-1, 1:-1] - (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])

        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        accum = self._phi_over_dt * (c_t1 - c_t)
        vx = -self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        vy = -self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        cx_avg = 0.5 * (c_t1[:, :, 1:, :] + c_t1[:, :, :-1, :])
        cy_avg = 0.5 * (c_t1[:, :, :, 1:] + c_t1[:, :, :, :-1])
        mdiv_x = (cx_avg * vx)[:, :, 1:, :] - (cx_avg * vx)[:, :, :-1, :]
        mdiv_y = (cy_avg * vy)[:, :, :, 1:] - (cy_avg * vy)[:, :, :, :-1]
        mass_residual = accum[:, :, 1:-1, 1:-1] + (mdiv_x[:, :, :, 1:-1] + mdiv_y[:, :, 1:-1, :])

        return pres_residual, mass_residual
