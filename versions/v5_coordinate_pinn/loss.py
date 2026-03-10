"""V5 Coordinate PINN loss — 9 terms: 5 data + 3 autograd physics + 1 consistency."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLossWeights(nn.Module):
    """Learnable loss weights via log-variance parameterization.

    total = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)
    """

    def __init__(self, num_terms=9):
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


def _reconstruction_loss(x, x_rec):
    v = 0.1
    diff = x.reshape(x.size(0), -1) - x_rec.reshape(x_rec.size(0), -1)
    return torch.mean(torch.sum(diff * diff * (0.5 / v), dim=-1))


def _l2_reg_loss(z):
    return torch.mean(torch.sum(0.5 * z * z, dim=-1))


class CoordinatePINNLoss(nn.Module):
    """Combined data + autograd physics + consistency loss for V5 Coordinate PINN.

    9 loss terms:
      1. rec_t0:                      reconstruction at t=0
      2. rec_t1:                      reconstruction at t=1..K
      3. l2_reg:                      0.5 * ||z0||^2
      4. trans:                       0.5 * ||z_true - z_pred||^2
      5. yobs:                        observation match
      6. pressure_pde_autograd:       pressure PDE via autograd at collocation pts
      7. mass_conservation_autograd:  mass conservation via autograd at collocation pts
      8. darcy_flux_autograd:         Darcy flux via autograd at collocation pts
      9. consistency:                 CNN decoder vs PINN decoder agreement
    """

    _lambda_keys = [
        'rec_t0', 'rec_t1', 'l2_reg', 'trans', 'yobs',
        'pressure_pde_autograd', 'mass_conservation_autograd',
        'darcy_flux_autograd', 'consistency',
    ]

    def __init__(self, cfg, perm_log, device, lambda_consistency=0.1):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.lambda_consistency = lambda_consistency

        # Permeability field as buffer (for grid_sample at collocation points)
        # perm_log: (1, Nx, Ny) -> perm_linear: (1, 1, Nx, Ny) for grid_sample
        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)  # (1, Nx, Ny)
        self.register_buffer('perm_4d', perm_linear.unsqueeze(0))  # (1, 1, Nx, Ny)

        # Physics constants
        self._phi_ct_over_dt = cfg.porosity * cfg.total_compressibility / cfg.dt_physical
        self._phi_over_dt = cfg.porosity / cfg.dt_physical

        # Adaptive weights
        self.use_adaptive = cfg.use_adaptive_weights
        if self.use_adaptive:
            self.adaptive = AdaptiveLossWeights(num_terms=9)
        else:
            self.adaptive = None

        # Static lambdas
        lambda_vals = [
            cfg.lambda_rec_t0, cfg.lambda_rec_t1, cfg.lambda_l2_reg,
            cfg.lambda_trans, cfg.lambda_yobs,
            cfg.lambda_pressure_pde, cfg.lambda_mass_conservation,
            cfg.lambda_darcy_flux, lambda_consistency,
        ]
        self.register_buffer('_lambdas', torch.tensor(lambda_vals, dtype=torch.float32))

    def forward(self, pred):
        """Compute all 9 loss terms. Returns (total_loss, losses_stack)."""
        x0 = pred['x0']
        x0_rec = pred['x0_rec']
        z0 = pred['z0']
        X_next_pred = pred['X_next_pred']
        X_next = pred['X_next']
        Z_next_pred = pred['Z_next_pred']
        Z_next = pred['Z_next']
        Y_next_pred = pred['Y_next_pred']
        Y = pred['Y']
        pinn_preds = pred['pinn_pred']
        collocation_coords = pred['collocation_coords']

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

        # 6-8. Autograd physics losses at collocation points
        loss_pde_ag = torch.tensor(0.0, device=self.device)
        loss_mass_ag = torch.tensor(0.0, device=self.device)
        loss_darcy_ag = torch.tensor(0.0, device=self.device)

        for i_step in range(len(pinn_preds)):
            pinn_out = pinn_preds[i_step]       # (B, N, n_ch)
            coords = collocation_coords[i_step]  # (B, N, 2), requires_grad=True

            pde_res, mass_res, darcy_res = self._autograd_physics(
                pinn_out, coords)
            loss_pde_ag = loss_pde_ag + pde_res
            loss_mass_ag = loss_mass_ag + mass_res
            loss_darcy_ag = loss_darcy_ag + darcy_res

        # 9. Consistency loss: CNN decoder vs PINN decoder at collocation points
        loss_consistency = torch.tensor(0.0, device=self.device)
        for i_step in range(len(pinn_preds)):
            pinn_out = pinn_preds[i_step]       # (B, N, n_ch)
            coords = collocation_coords[i_step]  # (B, N, 2)
            cnn_out = X_next_pred[i_step]        # (B, n_ch, H, W)

            # Sample CNN output at collocation points using grid_sample
            # coords are in [-1, 1], grid_sample expects (B, N, 1, 2)
            grid = coords.detach().unsqueeze(2)  # (B, N, 1, 2)
            cnn_sampled = F.grid_sample(
                cnn_out, grid, mode='bilinear', padding_mode='border',
                align_corners=True
            )  # (B, n_ch, N, 1)
            cnn_sampled = cnn_sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, n_ch)

            loss_consistency = loss_consistency + F.mse_loss(
                pinn_out, cnn_sampled.detach())

        # Stack all 9 losses
        losses_stack = torch.stack([
            loss_rec_t0, loss_rec_t1, loss_l2_reg, loss_trans, loss_yobs,
            loss_pde_ag, loss_mass_ag, loss_darcy_ag, loss_consistency,
        ])

        if self.use_adaptive and self.adaptive is not None:
            losses_dict = OrderedDict(zip(self._lambda_keys, losses_stack))
            total_loss = self.adaptive(losses_dict)
        else:
            total_loss = (self._lambdas * losses_stack).sum()

        return total_loss, losses_stack

    @torch.amp.autocast('cuda', enabled=False)
    def _autograd_physics(self, pinn_out, coords):
        """Compute autograd-based physics residuals at collocation points.

        Args:
            pinn_out: (B, N, n_ch) — PINN decoder output (ch0=saturation, ch1=pressure)
            coords: (B, N, 2) — collocation coordinates with requires_grad=True

        Returns:
            (pressure_pde_residual, mass_conservation_residual, darcy_flux_residual)
        """
        # Don't cast coords — they must be the SAME tensor that pinn_out was computed from
        # (casting creates a new tensor that breaks the autograd graph)
        # pinn_out is already float32 from the PINN decoder's autocast-disabled forward
        B, N, _ = pinn_out.shape

        # Extract pressure and saturation
        saturation = pinn_out[..., 0]   # (B, N)
        pressure = pinn_out[..., 1]     # (B, N)

        # Compute spatial derivatives via autograd
        dp = torch.autograd.grad(
            pressure.sum(), coords, create_graph=True, retain_graph=True
        )[0]  # (B, N, 2)
        dp_dx = dp[..., 0]  # (B, N)
        dp_dy = dp[..., 1]  # (B, N)

        ds = torch.autograd.grad(
            saturation.sum(), coords, create_graph=True, retain_graph=True
        )[0]  # (B, N, 2)
        ds_dx = ds[..., 0]  # (B, N)
        ds_dy = ds[..., 1]  # (B, N)

        # Get permeability at collocation points via grid_sample
        # coords: (B, N, 2) in [-1, 1]
        grid = coords.unsqueeze(2)  # (B, N, 1, 2)
        perm_4d = self.perm_4d.expand(B, -1, -1, -1)  # (B, 1, Nx, Ny)
        perm_at_pts = F.grid_sample(
            perm_4d, grid, mode='bilinear', padding_mode='border',
            align_corners=True
        )  # (B, 1, N, 1)
        k = perm_at_pts.squeeze(-1).squeeze(1)  # (B, N)

        # Second derivatives for Laplacian (pressure PDE)
        dp_dx_sum = dp_dx.sum()
        dp_dy_sum = dp_dy.sum()

        d2p_dx2 = torch.autograd.grad(
            dp_dx_sum, coords, create_graph=True, retain_graph=True
        )[0][..., 0]  # (B, N)
        d2p_dy2 = torch.autograd.grad(
            dp_dy_sum, coords, create_graph=True, retain_graph=True
        )[0][..., 1]  # (B, N)

        # 6. Pressure PDE residual: nabla . (k * nabla p) ~ 0
        # Simplified: k * (d2p/dx2 + d2p/dy2) as steady-state approximation
        pde_residual = k * (d2p_dx2 + d2p_dy2)
        loss_pde = torch.mean(pde_residual ** 2)

        # 7. Mass conservation: phi * ds/dt + nabla . (s * v) ~ 0
        # At collocation points we approximate with spatial terms only
        # v = -k * nabla(p), so flux = s * (-k * nabla p)
        # div(s * v) = ds_dx * (-k * dp_dx) + ds_dy * (-k * dp_dy) + s * (-k * laplacian_p)
        flux_div = (ds_dx * (-k * dp_dx) + ds_dy * (-k * dp_dy)
                    + saturation * (-k * (d2p_dx2 + d2p_dy2)))
        loss_mass = torch.mean(flux_div ** 2)

        # 8. Darcy flux: v + k * nabla(p) = 0 (enforced implicitly)
        # Since we don't have explicit velocity, use gradient magnitude consistency
        darcy_residual = k * (dp_dx ** 2 + dp_dy ** 2)
        # Minimize variance of Darcy flux magnitude across collocation points
        darcy_mean = darcy_residual.mean(dim=-1, keepdim=True)
        loss_darcy = torch.mean((darcy_residual - darcy_mean) ** 2)

        return loss_pde, loss_mass, loss_darcy

    def losses_to_dict(self, losses_stack):
        """Convert losses stack to dict with .item() calls — only for logging."""
        vals = losses_stack.detach()
        return {name: vals[i].item() for i, name in enumerate(self._lambda_keys)}

    @property
    def num_loss_terms(self):
        return 9

    def get_adaptive_state(self):
        if self.adaptive is not None:
            return self.adaptive.state_dict()
        return None

    def load_adaptive_state(self, state):
        if self.adaptive is not None and state is not None:
            self.adaptive.load_state_dict(state)

    def compute_physics_residuals(self, x_t, x_t1):
        """Per-pixel residuals for visualization (finite-difference, same as V1).
        Uses the perm buffer for transmissibility computation."""
        tran_x = 2.0 / (1.0 / self.perm[:, 1:, :] + 1.0 / self.perm[:, :-1, :])
        tran_y = 2.0 / (1.0 / self.perm[:, :, 1:] + 1.0 / self.perm[:, :, :-1])

        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]
        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)
        flux_x = tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        flux_y = tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]
        pres_residual = dp_dt[:, :, 1:-1, 1:-1] - (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])

        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        accum = self._phi_over_dt * (c_t1 - c_t)
        vx = -tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        vy = -tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        cx_avg = 0.5 * (c_t1[:, :, 1:, :] + c_t1[:, :, :-1, :])
        cy_avg = 0.5 * (c_t1[:, :, :, 1:] + c_t1[:, :, :, :-1])
        mdiv_x = (cx_avg * vx)[:, :, 1:, :] - (cx_avg * vx)[:, :, :-1, :]
        mdiv_y = (cy_avg * vy)[:, :, :, 1:] - (cy_avg * vy)[:, :, :, :-1]
        mass_residual = accum[:, :, 1:-1, 1:-1] + (mdiv_x[:, :, :, 1:-1] + mdiv_y[:, :, 1:-1, :])

        return pres_residual, mass_residual
