"""V6 FNO Decoder loss — 5 data + 3 physics losses with spectral derivatives."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def _reconstruction_loss(x, x_rec):
    v = 0.1
    diff = x.reshape(x.size(0), -1) - x_rec.reshape(x_rec.size(0), -1)
    return torch.mean(torch.sum(diff * diff * (0.5 / v), dim=-1))


def _l2_reg_loss(z):
    return torch.mean(torch.sum(0.5 * z * z, dim=-1))


class AdaptiveLossWeights(nn.Module):
    """Learnable loss weights via log-variance parameterization."""

    LOSS_NAMES = [
        'rec_t0', 'rec_t1', 'l2_reg', 'trans',
        'yobs', 'pressure_pde', 'mass_conservation', 'darcy_flux',
    ]

    def __init__(self, num_terms=8):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(num_terms))

    def forward(self, losses_dict):
        total = torch.zeros(1, device=self.log_sigmas.device)
        for i, (name, loss_val) in enumerate(losses_dict.items()):
            log_s = self.log_sigmas[i]
            precision = torch.exp(-2 * log_s)
            total = total + (0.5 * precision * loss_val + log_s)
        return total.squeeze(0)

    def get_weight_info(self, losses_dict):
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


class SpectralPhysicsLoss(nn.Module):
    """Combined data-driven + physics-informed loss with spectral derivatives.

    8 loss terms:
      1. rec_t0:             reconstruction at t=0
      2. rec_t1:             reconstruction at t=1..K
      3. l2_reg:             0.5 * ||z0||^2
      4. trans:              0.5 * ||z_true - z_pred||^2
      5. yobs:               observation match
      6. pressure_pde:       pressure diffusion PDE residual (spectral derivatives)
      7. mass_conservation:  CO2 mass conservation residual (spectral derivatives)
      8. darcy_flux:         Darcy flux consistency (spectral derivatives)
    """

    def __init__(self, cfg, perm_log, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Permeability as buffer: k = 10^(log_perm) [mD]
        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)  # (1, Nx, Ny)

        # Precompute physics constants
        self._phi_ct_over_dt = cfg.porosity * cfg.total_compressibility / cfg.dt_physical
        self._phi_over_dt = cfg.porosity / cfg.dt_physical

        # Precompute wavenumber grids for spectral derivatives
        Nx, Ny = cfg.Nx, cfg.Ny
        kx = torch.fft.fftfreq(Nx, d=1.0).to(device) * 2 * np.pi  # (Nx,)
        ky = torch.fft.rfftfreq(Ny, d=1.0).to(device) * 2 * np.pi  # (Ny//2+1,)
        # Make 2D grids for broadcasting with rfft2 output
        kx_grid = kx.unsqueeze(-1).expand(Nx, Ny // 2 + 1)  # (Nx, Ny//2+1)
        ky_grid = ky.unsqueeze(0).expand(Nx, Ny // 2 + 1)   # (Nx, Ny//2+1)
        self.register_buffer('kx_grid', kx_grid)
        self.register_buffer('ky_grid', ky_grid)

        # Adaptive weights
        self.use_adaptive = cfg.use_adaptive_weights
        if self.use_adaptive:
            self.adaptive = AdaptiveLossWeights(num_terms=8)
        else:
            self.adaptive = None

        # Static lambdas
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

    def _spectral_derivative_x(self, field):
        """Compute df/dx via spectral method. field: (B, 1, Nx, Ny)."""
        f_hat = torch.fft.rfft2(field)
        df_hat = 1j * self.kx_grid * f_hat
        return torch.fft.irfft2(df_hat, s=(field.shape[-2], field.shape[-1]))

    def _spectral_derivative_y(self, field):
        """Compute df/dy via spectral method. field: (B, 1, Nx, Ny)."""
        f_hat = torch.fft.rfft2(field)
        df_hat = 1j * self.ky_grid * f_hat
        return torch.fft.irfft2(df_hat, s=(field.shape[-2], field.shape[-1]))

    def _spectral_laplacian(self, field):
        """Compute d2f/dx2 + d2f/dy2 via spectral method."""
        f_hat = torch.fft.rfft2(field)
        lap_hat = -(self.kx_grid ** 2 + self.ky_grid ** 2) * f_hat
        return torch.fft.irfft2(lap_hat, s=(field.shape[-2], field.shape[-1]))

    def forward(self, pred):
        """Compute all 8 loss terms. Returns (total_loss, losses_stack)."""
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

        # 6. Pressure PDE (spectral)
        x_ref = x0
        loss_pde = self._pressure_pde_residual(x_ref, X_next_pred[0])
        for i in range(1, len(X_next_pred)):
            loss_pde = loss_pde + self._pressure_pde_residual(X_next[i - 1], X_next_pred[i])

        # 7. Mass conservation (spectral)
        x_ref = x0
        loss_mass = self._mass_conservation_residual(x_ref, X_next_pred[0])
        for i in range(1, len(X_next_pred)):
            loss_mass = loss_mass + self._mass_conservation_residual(X_next_pred[i - 1], X_next_pred[i])

        # 8. Darcy flux (spectral)
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

    # ---- Physics loss functions (spectral derivatives) ----

    def _pressure_pde_residual(self, x_t, x_t1):
        """Pressure diffusion PDE residual using spectral Laplacian."""
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]

        # Time derivative: phi * ct / dt * (p_{t+1} - p_t)
        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)

        # Spectral Laplacian of pressure at t+1, scaled by permeability
        # For variable coefficient: div(k * grad(p)) approximated as k * laplacian(p)
        # (valid when k varies slowly relative to p)
        k_lap_p = self.perm * self._spectral_laplacian(p_t1)

        residual = dp_dt - k_lap_p
        return torch.mean(residual * residual)

    def _mass_conservation_residual(self, x_t, x_t1):
        """CO2 mass conservation residual using spectral derivatives."""
        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        p_t1 = x_t1[:, -1:, :, :]

        # Accumulation
        accum = self._phi_over_dt * (c_t1 - c_t)

        # Velocity from Darcy's law: v = -k * grad(p)
        vx = -self.perm * self._spectral_derivative_x(p_t1)
        vy = -self.perm * self._spectral_derivative_y(p_t1)

        # Advective flux divergence: div(c * v) = c * div(v) + v . grad(c)
        dc_dx = self._spectral_derivative_x(c_t1)
        dc_dy = self._spectral_derivative_y(c_t1)
        dvx_dx = self._spectral_derivative_x(vx)
        dvy_dy = self._spectral_derivative_y(vy)

        flux_div = c_t1 * (dvx_dx + dvy_dy) + vx * dc_dx + vy * dc_dy

        residual = accum + flux_div
        return torch.mean(residual * residual)

    def _darcy_flux_loss(self, x_true, x_pred):
        """Darcy flux consistency using spectral derivatives."""
        p = x_true[:, -1:, :, :]
        p_pred = x_pred[:, -1:, :, :]

        # True fluxes
        flux_x_true = -self.perm * self._spectral_derivative_x(p)
        flux_y_true = -self.perm * self._spectral_derivative_y(p)

        # Predicted fluxes
        flux_x_pred = -self.perm * self._spectral_derivative_x(p_pred)
        flux_y_pred = -self.perm * self._spectral_derivative_y(p_pred)

        loss_x = torch.sum(torch.abs(flux_x_true - flux_x_pred).reshape(p.size(0), -1), dim=-1)
        loss_y = torch.sum(torch.abs(flux_y_true - flux_y_pred).reshape(p.size(0), -1), dim=-1)

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
        k_lap_p = self.perm * self._spectral_laplacian(p_t1)
        pres_residual = dp_dt - k_lap_p

        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        vx = -self.perm * self._spectral_derivative_x(p_t1)
        vy = -self.perm * self._spectral_derivative_y(p_t1)
        dc_dx = self._spectral_derivative_x(c_t1)
        dc_dy = self._spectral_derivative_y(c_t1)
        dvx_dx = self._spectral_derivative_x(vx)
        dvy_dy = self._spectral_derivative_y(vy)
        accum = self._phi_over_dt * (c_t1 - c_t)
        flux_div = c_t1 * (dvx_dx + dvy_dy) + vx * dc_dx + vy * dc_dy
        mass_residual = accum + flux_div

        return pres_residual, mass_residual
