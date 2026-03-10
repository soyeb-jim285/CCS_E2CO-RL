"""V3: Physics Corrector E2CO — Jacobi PDE corrector after decoder.

Same encoder/decoder/transition as V0 baseline, but adds a PhysicsCorrectorLayer
that iteratively corrects pressure using PDE residuals after decoding.
"""

import torch
import torch.nn as nn

from versions.shared.layers import (
    conv_bn_relu, ResidualConv, dconv_bn_nolinear, fc_bn_relu
)


def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)


def fc_bn_relu_2arg(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU()
    )


def create_trans_encoder(total_input_dim):
    return nn.Sequential(
        fc_bn_relu_2arg(total_input_dim, 200),
        fc_bn_relu_2arg(200, 200),
        fc_bn_relu_2arg(200, total_input_dim - 1)
    )


class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape, sigma=0.0):
        super().__init__()
        self.sigma = sigma
        self.fc_layers = nn.Sequential(
            conv_bn_relu(input_shape[0], 16, 3, 3, stride=2),
            conv_bn_relu(16, 32, 3, 3, stride=1),
            conv_bn_relu(32, 64, 3, 3, stride=2),
            conv_bn_relu(64, 128, 3, 3, stride=1)
        )
        self.fc_layers.apply(weights_init)

        self.res_layers = nn.Sequential(
            ResidualConv(128, 128, 3, 3, stride=1),
            ResidualConv(128, 128, 3, 3, stride=1),
            ResidualConv(128, 128, 3, 3, stride=1)
        )
        self.res_layers.apply(weights_init)

        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(
            128 * int(input_shape[1] / 4) * int(input_shape[2] / 4),
            latent_dim
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = self.res_layers(x)
        x = self.flatten(x)
        xi_mean = self.fc_mean(x)
        return xi_mean


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim,
                      int(input_shape[1] * input_shape[2] / 16 * 128)),
            nn.ReLU()
        )
        self.fc_layers.apply(weights_init)

        self.upsample_layers = nn.Sequential(
            ResidualConv(128, 128, 3, 3),
            ResidualConv(128, 128, 3, 3),
            ResidualConv(128, 128, 3, 3)
        )
        self.upsample_layers.apply(weights_init)

        self.deconv_layers = nn.Sequential(
            dconv_bn_nolinear(128, 64, 3, 3, stride=(1, 1), padding=1),
            dconv_bn_nolinear(64, 32, 2, 2, stride=(2, 2)),
            dconv_bn_nolinear(32, 16, 3, 3, stride=(1, 1), padding=1),
            dconv_bn_nolinear(16, 16, 2, 2, stride=(2, 2)),
            nn.Conv2d(16, input_shape[0], kernel_size=(3, 3), padding='same')
        )
        self.deconv_layers.apply(weights_init)

    def forward(self, z):
        x = self.fc_layers(z)
        x = x.view(-1, 128,
                    int(self.input_shape[1] / 4),
                    int(self.input_shape[2] / 4))
        x = self.upsample_layers(x)
        y = self.deconv_layers(x)
        return y


class LinearTransitionModel(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prod, num_inj):
        super().__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim
        self.num_prod = num_prod
        self.num_inj = num_inj
        self.obs_dim = num_prod * 2 + num_inj

        self.trans_encoder = create_trans_encoder(latent_dim + 1)
        self.trans_encoder.apply(weights_init)

        self.At_layer = nn.Linear(latent_dim, latent_dim * latent_dim)
        self.At_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(latent_dim, latent_dim * u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(latent_dim, self.obs_dim * latent_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(latent_dim, self.obs_dim * u_dim)
        self.Dt_layer.apply(weights_init)

    def forward_nsteps(self, zt, dt, U):
        zt_expand = torch.cat([zt, dt], dim=-1)
        hz = self.trans_encoder(zt_expand)

        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.obs_dim, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.obs_dim, self.u_dim)

        Zt_k = []
        Yt_k = []
        for ut in U:
            ut_dt = ut * dt
            zt = (torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1)
                  + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))
            yt = (torch.bmm(Ct, zt.unsqueeze(-1)).squeeze(-1)
                  + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1))
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut):
        zt_expand = torch.cat([zt, dt], dim=-1)
        hz = self.trans_encoder(zt_expand)

        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.obs_dim, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.obs_dim, self.u_dim)

        ut_dt = ut * dt
        zt1 = (torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1)
               + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))
        yt1 = (torch.bmm(Ct, zt1.unsqueeze(-1)).squeeze(-1)
               + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1))
        return zt1, yt1


class PhysicsCorrectorLayer(nn.Module):
    """Jacobi PDE corrector: iteratively corrects pressure after decoding.

    For n_iterations:
        p = p - alpha * PDE_residual(p, p_prev, transmissibility)
    Only corrects pressure channel; saturation passes through.
    """
    def __init__(self, perm_log, device, n_iterations=2, alpha=0.1,
                 porosity=0.2, total_compressibility=1e-5, dt_physical=20.0):
        super().__init__()
        self.n_iterations = n_iterations
        self.alpha = alpha

        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)
        self.register_buffer('tran_x',
            2.0 / (1.0 / self.perm[:, 1:, :] + 1.0 / self.perm[:, :-1, :]))
        self.register_buffer('tran_y',
            2.0 / (1.0 / self.perm[:, :, 1:] + 1.0 / self.perm[:, :, :-1]))
        self._phi_ct_over_dt = porosity * total_compressibility / dt_physical

    def forward(self, x_decoded, x_prev):
        """x_decoded: (B, 2, 64, 64), x_prev: (B, 2, 64, 64)"""
        sat = x_decoded[:, 0:1, :, :]  # pass through
        p = x_decoded[:, 1:2, :, :].clone()
        p_prev = x_prev[:, 1:2, :, :]

        for _ in range(self.n_iterations):
            # Compute PDE residual
            dp_dt = self._phi_ct_over_dt * (p - p_prev)
            flux_x = self.tran_x * (p[:, :, 1:, :] - p[:, :, :-1, :])
            div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]
            flux_y = self.tran_y * (p[:, :, :, 1:] - p[:, :, :, :-1])
            div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]

            residual = torch.zeros_like(p)
            residual[:, :, 1:-1, 1:-1] = dp_dt[:, :, 1:-1, 1:-1] - (
                div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])

            p = p - self.alpha * residual
            p = torch.clamp(p, 0.0, 1.0)  # clamp normalized pressure

        return torch.cat([sat, p], dim=1)


class PhysicsCorrectorE2CO(nn.Module):
    """V3: Physics Corrector E2CO.

    Same encoder/decoder/transition as baseline V0, but adds a
    PhysicsCorrectorLayer that iteratively corrects pressure using
    PDE residuals after decoding.
    """

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0, perm_log=None, device=None,
                 n_corrector_iterations=2, corrector_alpha=0.1,
                 porosity=0.2, total_compressibility=1e-5, dt_physical=20.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps
        self.n_corrector_iterations = n_corrector_iterations
        self.corrector_alpha = corrector_alpha

        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = Decoder(latent_dim, input_shape)
        self.transition = LinearTransitionModel(latent_dim, u_dim,
                                                 num_prod, num_inj)

        # Physics corrector layer (requires permeability field)
        if perm_log is not None and device is not None:
            self.corrector = PhysicsCorrectorLayer(
                perm_log, device,
                n_iterations=n_corrector_iterations,
                alpha=corrector_alpha,
                porosity=porosity,
                total_compressibility=total_compressibility,
                dt_physical=dt_physical,
            )
        else:
            self.corrector = None

    def forward(self, inputs):
        """Multi-step forward for training.

        Args:
            inputs: (X, U, Y, dt) where
                X = list of K+1 state tensors [x0, x1, ..., xK]
                U = list of K control tensors
                Y = list of K observation tensors
                dt = time step tensor (batch, 1)

        Returns:
            dict with keys: x0, x0_rec, z0, X_next_pred, X_next,
                           Z_next_pred, Z_next, Y_next_pred, Y
        """
        X, U, Y, dt = inputs

        x0 = X[0]
        z0 = self.encoder(x0)
        x0_rec = self.decoder(z0)

        Z_next_pred, Y_next_pred = self.transition.forward_nsteps(z0, dt, U)

        X_next_pred = []
        Z_next = []
        X_next = X[1:]

        # Track previous state for corrector
        x_prev = x0

        for i_step in range(len(Z_next_pred)):
            z_next_pred = Z_next_pred[i_step]
            x_next_pred = self.decoder(z_next_pred)

            # Apply physics corrector
            if self.corrector is not None:
                x_next_pred = self.corrector(x_next_pred, x_prev)

            z_next_true = self.encoder(X[i_step + 1])

            X_next_pred.append(x_next_pred)
            Z_next.append(z_next_true)

            # Update x_prev for next corrector step
            x_prev = X[i_step + 1]

        return {
            'x0': x0,
            'x0_rec': x0_rec,
            'z0': z0,
            'X_next_pred': X_next_pred,
            'X_next': X_next,
            'Z_next_pred': Z_next_pred,
            'Z_next': Z_next,
            'Y_next_pred': Y_next_pred,
            'Y': Y,
        }

    def predict(self, xt, ut, yt, dt):
        """Single-step prediction for sequential eval."""
        self.eval()
        with torch.no_grad():
            zt = self.encoder(xt)
            zt_next, yt_next = self.transition(zt, dt, ut)
            xt_next_pred = self.decoder(zt_next)

            # Apply physics corrector with xt as previous state
            if self.corrector is not None:
                xt_next_pred = self.corrector(xt_next_pred, xt)

        return xt_next_pred, yt_next

    def predict_latent(self, zt, dt, ut):
        """Latent-space single-step (used by RL)."""
        self.eval()
        with torch.no_grad():
            zt_next, yt_next = self.transition(zt, dt, ut)
        return zt_next, yt_next
