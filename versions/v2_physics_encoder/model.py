"""V2: Physics Encoder E2CO — Sobel gradient preprocessing before encoding.

The encoder receives 6 channels (original 2 + 4 spatial gradients) instead of 2.
The decoder still outputs 2 channels (mole fraction + pressure).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SobelPreprocessor(nn.Module):
    """Fixed (non-trainable) Sobel kernels for spatial gradients.
    Input: (B, 2, 64, 64) -> Output: (B, 6, 64, 64)
    Channels: [mole_frac, pressure, d_mole/dx, d_mole/dy, dp/dx, dp/dy]
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # x: (B, 2, 64, 64)
        grads = []
        for ch in range(x.shape[1]):
            ch_data = x[:, ch:ch+1, :, :]
            grad_x = F.conv2d(ch_data, self.sobel_x, padding=1)
            grad_y = F.conv2d(ch_data, self.sobel_y, padding=1)
            grads.extend([grad_x, grad_y])
        return torch.cat([x] + grads, dim=1)  # (B, 6, 64, 64)


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


class PhysicsEncoderE2CO(nn.Module):
    """V2: Physics Encoder E2CO.

    Adds a fixed Sobel preprocessor before encoding, giving the encoder
    spatial gradient information (6 channels instead of 2).
    Decoder still outputs 2 channels.
    """

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps
        self.input_shape = input_shape  # (2, 64, 64) — original

        # Sobel preprocessor: (2, 64, 64) -> (6, 64, 64)
        self.preprocessor = SobelPreprocessor()

        # Encoder takes 6 channels
        encoder_input_shape = (6, input_shape[1], input_shape[2])
        self.encoder = Encoder(latent_dim, encoder_input_shape, sigma=sigma)

        # Decoder outputs 2 channels (original)
        self.decoder = Decoder(latent_dim, input_shape)

        self.transition = LinearTransitionModel(latent_dim, u_dim,
                                                 num_prod, num_inj)

    def forward(self, inputs):
        """Multi-step forward for training.

        Args:
            inputs: (X, U, Y, dt) where
                X = list of K+1 state tensors [x0, x1, ..., xK]  (B, 2, 64, 64)
                U = list of K control tensors
                Y = list of K observation tensors
                dt = time step tensor (batch, 1)

        Returns:
            dict with keys: x0, x0_rec, z0, X_next_pred, X_next,
                           Z_next_pred, Z_next, Y_next_pred, Y
        """
        X, U, Y, dt = inputs

        x0 = X[0]
        x0_preprocessed = self.preprocessor(x0)
        z0 = self.encoder(x0_preprocessed)
        x0_rec = self.decoder(z0)

        Z_next_pred, Y_next_pred = self.transition.forward_nsteps(z0, dt, U)

        X_next_pred = []
        Z_next = []
        X_next = X[1:]

        for i_step in range(len(Z_next_pred)):
            z_next_pred = Z_next_pred[i_step]
            x_next_pred = self.decoder(z_next_pred)

            # Preprocess X_next before encoding for Z_next
            x_next_preprocessed = self.preprocessor(X[i_step + 1])
            z_next_true = self.encoder(x_next_preprocessed)

            X_next_pred.append(x_next_pred)
            Z_next.append(z_next_true)

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
            xt_preprocessed = self.preprocessor(xt)
            zt = self.encoder(xt_preprocessed)
            zt_next, yt_next = self.transition(zt, dt, ut)
            xt_next_pred = self.decoder(zt_next)
        return xt_next_pred, yt_next

    def predict_latent(self, zt, dt, ut):
        """Latent-space single-step (used by RL)."""
        self.eval()
        with torch.no_grad():
            zt_next, yt_next = self.transition(zt, dt, ut)
        return zt_next, yt_next
