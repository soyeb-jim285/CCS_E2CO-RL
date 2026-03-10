"""V7 DeepONet loss — 5 data + 3 physics losses with autograd or finite-difference fallback."""

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


class DeepONetPhysicsLoss(nn.Module):
    """Combined data-driven + physics-informed loss for DeepONet E2CO.

    8 loss terms:
      1. rec_t0:             reconstruction at t=0
      2. rec_t1:             reconstruction at t=1..K
      3. l2_reg:             0.5 * ||z0||^2
      4. trans:              0.5 * ||z_true - z_pred||^2
      5. yobs:               observation match
      6. pressure_pde:       pressure diffusion PDE residual
      7. mass_conservation:  CO2 mass conservation residual
      8. darcy_flux:         Darcy flux consistency

    Physics losses use autograd when 'grad_coords' is available in pred dict,
    otherwise fall back to finite differences.
    """

    def __init__(self, cfg, perm_log, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Permeability as buffer: k = 10^(log_perm) [mD]
        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)  # (1, Nx, Ny)

        # Precompute harmonic transmissibility (finite-difference fallback)
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

    def forward(self, pred):
        """Compute all 8 loss terms. Returns (total_loss, losses_stack).

        If pred contains 'grad_coords', uses autograd for physics losses.
        Otherwise falls back to finite differences.
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

        # Check if autograd coordinates are available
        use_autograd = 'grad_coords' in pred

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

        # Physics losses
        if use_autograd:
            grad_coords = pred['grad_coords']
            # 6. Pressure PDE (autograd)
            x_ref = x0
            loss_pde = self._pressure_pde_residual_autograd(x_ref, X_next_pred[0], grad_coords)
            for i in range(1, len(X_next_pred)):
                loss_pde = loss_pde + self._pressure_pde_residual_autograd(
                    X_next[i - 1], X_next_pred[i], grad_coords)

            # 7. Mass conservation (autograd)
            x_ref = x0
            loss_mass = self._mass_conservation_residual_autograd(x_ref, X_next_pred[0], grad_coords)
            for i in range(1, len(X_next_pred)):
                loss_mass = loss_mass + self._mass_conservation_residual_autograd(
                    X_next_pred[i - 1], X_next_pred[i], grad_coords)

            # 8. Darcy flux (autograd)
            loss_darcy = self._darcy_flux_loss_autograd(X_next[0], X_next_pred[0], grad_coords)
            for i in range(1, len(X_next)):
                loss_darcy = loss_darcy + self._darcy_flux_loss_autograd(
                    X_next[i], X_next_pred[i], grad_coords)
        else:
            # Fall back to finite differences
            # 6. Pressure PDE
            x_ref = x0
            loss_pde = self._pressure_pde_residual_fd(x_ref, X_next_pred[0])
            for i in range(1, len(X_next_pred)):
                loss_pde = loss_pde + self._pressure_pde_residual_fd(X_next[i - 1], X_next_pred[i])

            # 7. Mass conservation
            x_ref = x0
            loss_mass = self._mass_conservation_residual_fd(x_ref, X_next_pred[0])
            for i in range(1, len(X_next_pred)):
                loss_mass = loss_mass + self._mass_conservation_residual_fd(
                    X_next_pred[i - 1], X_next_pred[i])

            # 8. Darcy flux
            loss_darcy = self._darcy_flux_loss_fd(X_next[0], X_next_pred[0])
            for i in range(1, len(X_next)):
                loss_darcy = loss_darcy + self._darcy_flux_loss_fd(X_next[i], X_next_pred[i])

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

    # ---- Autograd physics loss functions ----

    def _compute_spatial_grads(self, field, coords):
        """Compute df/dx, df/dy via autograd w.r.t. coordinates.

        field: (B, 1, Nx, Ny) - must be computed from coords
        coords: (Nx*Ny, 2) - with requires_grad=True

        Returns: (df_dx, df_dy) each (B, 1, Nx, Ny)
        """
        B = field.shape[0]
        Nx, Ny = field.shape[-2], field.shape[-1]

        # Flatten field for autograd: (B, Nx*Ny)
        field_flat = field.view(B, -1)

        # Sum over batch for autograd (compute grad for each spatial point)
        field_sum = field_flat.sum(dim=0)  # (Nx*Ny,)

        grads = torch.autograd.grad(
            outputs=field_sum.sum(),
            inputs=coords,
            create_graph=True,
            retain_graph=True,
        )[0]  # (Nx*Ny, 2)

        df_dx = grads[:, 0].view(1, 1, Nx, Ny).expand(B, 1, Nx, Ny)
        df_dy = grads[:, 1].view(1, 1, Nx, Ny).expand(B, 1, Nx, Ny)
        return df_dx, df_dy

    def _pressure_pde_residual_autograd(self, x_t, x_t1, coords):
        """Pressure diffusion PDE residual using autograd derivatives."""
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]

        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)

        # Spatial derivatives via autograd
        dp_dx, dp_dy = self._compute_spatial_grads(p_t1, coords)

        # Second derivatives: d2p/dx2, d2p/dy2
        kp_x = self.perm * dp_dx
        kp_y = self.perm * dp_dy

        dkpx_dx, _ = self._compute_spatial_grads(kp_x, coords)
        _, dkpy_dy = self._compute_spatial_grads(kp_y, coords)

        residual = dp_dt - (dkpx_dx + dkpy_dy)
        return torch.mean(residual * residual)

    def _mass_conservation_residual_autograd(self, x_t, x_t1, coords):
        """CO2 mass conservation residual using autograd derivatives."""
        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        p_t1 = x_t1[:, -1:, :, :]

        accum = self._phi_over_dt * (c_t1 - c_t)

        # Velocity: v = -k * grad(p)
        dp_dx, dp_dy = self._compute_spatial_grads(p_t1, coords)
        vx = -self.perm * dp_dx
        vy = -self.perm * dp_dy

        # Advective flux: c * v
        flux_x = c_t1 * vx
        flux_y = c_t1 * vy

        # Divergence of flux
        dflux_x_dx, _ = self._compute_spatial_grads(flux_x, coords)
        _, dflux_y_dy = self._compute_spatial_grads(flux_y, coords)

        residual = accum + (dflux_x_dx + dflux_y_dy)
        return torch.mean(residual * residual)

    def _darcy_flux_loss_autograd(self, x_true, x_pred, coords):
        """Darcy flux consistency using autograd derivatives."""
        p = x_true[:, -1:, :, :]
        p_pred = x_pred[:, -1:, :, :]

        # True fluxes
        dp_dx, dp_dy = self._compute_spatial_grads(p, coords)
        flux_x_true = -self.perm * dp_dx
        flux_y_true = -self.perm * dp_dy

        # Predicted fluxes
        dpp_dx, dpp_dy = self._compute_spatial_grads(p_pred, coords)
        flux_x_pred = -self.perm * dpp_dx
        flux_y_pred = -self.perm * dpp_dy

        loss_x = torch.sum(torch.abs(flux_x_true - flux_x_pred).reshape(p.size(0), -1), dim=-1)
        loss_y = torch.sum(torch.abs(flux_y_true - flux_y_pred).reshape(p.size(0), -1), dim=-1)

        return torch.mean(loss_x + loss_y)

    # ---- Finite-difference fallback physics losses ----

    def _pressure_pde_residual_fd(self, x_t, x_t1):
        """Pressure diffusion PDE residual on interior 62x62 cells (finite differences)."""
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]

        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)

        flux_x = self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]

        flux_y = self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]

        residual = dp_dt[:, :, 1:-1, 1:-1] - (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])
        return torch.mean(residual * residual)

    def _mass_conservation_residual_fd(self, x_t, x_t1):
        """CO2 mass conservation residual on interior 62x62 cells (finite differences)."""
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

    def _darcy_flux_loss_fd(self, x_true, x_pred):
        """Darcy flux consistency (finite differences)."""
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
        """Per-pixel residuals for visualization (call with no_grad). Uses finite differences."""
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
