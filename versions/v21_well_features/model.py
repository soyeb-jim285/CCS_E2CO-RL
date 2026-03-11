"""V21 Well-Location Feature Attention — encoder feature extraction at well coordinates.

Bypasses information bottleneck by extracting encoder feature maps at well locations
and feeding them directly to per-well MLP observation heads.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

from versions.v0_baseline.model import (
    Decoder, weights_init, fc_bn_relu_2arg, create_trans_encoder
)
from versions.shared.layers import (
    conv_bn_relu, ResidualConv, dconv_bn_nolinear, fc_bn_relu
)


class WellFeatureEncoder(nn.Module):
    """Modified V0 Encoder that returns both z and intermediate feature maps."""

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
        features = self.res_layers(x)  # (B, 128, 16, 16)
        z = self.fc_mean(self.flatten(features))
        return z, features

    def encode_z_only(self, x):
        """For compatibility — just return z without features."""
        z, _ = self.forward(x)
        return z


class WellFeatureExtractor(nn.Module):
    """Bilinear samples feature map at well locations."""

    def __init__(self, prod_loc, feature_map_size=16, input_size=64, feature_channels=128):
        super().__init__()
        self.num_wells = len(prod_loc)
        self.feature_channels = feature_channels

        # Convert well locations from 64x64 grid to normalized [-1, 1] coords
        # for F.grid_sample. Feature map is input_size/4 = 16x16.
        scale = input_size / feature_map_size  # 4.0
        grid_coords = []
        for row, col in prod_loc:
            # Convert to feature map coordinates
            feat_row = row / scale
            feat_col = col / scale
            # Normalize to [-1, 1] for grid_sample
            # grid_sample expects (x, y) = (col_norm, row_norm)
            x_norm = 2.0 * feat_col / (feature_map_size - 1) - 1.0
            y_norm = 2.0 * feat_row / (feature_map_size - 1) - 1.0
            grid_coords.append([x_norm, y_norm])

        # Shape: (1, num_wells, 1, 2) for grid_sample
        grid = torch.tensor(grid_coords, dtype=torch.float32).view(1, self.num_wells, 1, 2)
        self.register_buffer('grid', grid)

    def forward(self, features):
        """Extract features at well locations.

        Args:
            features: (B, 128, 16, 16) encoder feature maps

        Returns:
            (B, num_wells, 128) features at each well location
        """
        B = features.shape[0]
        # Expand grid to batch size
        grid = self.grid.expand(B, -1, -1, -1)
        # grid_sample: input (B, C, H, W), grid (B, N, 1, 2) -> (B, C, N, 1)
        sampled = F.grid_sample(features, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        # (B, 128, num_wells, 1) -> (B, num_wells, 128)
        return sampled.squeeze(-1).permute(0, 2, 1)


class WellAwareTransition(nn.Module):
    """V0 dynamics + per-well MLP obs with feature map context."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, feature_channels=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim
        self.num_prod = num_prod
        self.num_inj = num_inj
        self.obs_dim = num_prod * 2 + num_inj  # 14

        # Same z dynamics as V0
        self.trans_encoder = create_trans_encoder(latent_dim + 1)
        self.trans_encoder.apply(weights_init)
        self.At_layer = nn.Linear(latent_dim, latent_dim * latent_dim)
        self.At_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(latent_dim, latent_dim * u_dim)
        self.Bt_layer.apply(weights_init)

        # Per-well MLP: input = cat(z, u, well_features) = 20 + 9 + 128 = 157
        well_input_dim = latent_dim + u_dim + feature_channels

        self.well_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(well_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            for _ in range(num_prod)
        ])

        # Injector head (no spatial features — injectors are separate)
        self.inj_head = nn.Linear(latent_dim + u_dim, num_inj)

    def forward_nsteps(self, z0, dt, U, well_features):
        """Multi-step with well features from x0 encoder (constant across steps).

        Args:
            z0: (B, latent_dim)
            dt: (B, 1)
            U: list of (B, u_dim) control tensors
            well_features: (B, num_prod, 128) from WellFeatureExtractor
        """
        zt_expand = torch.cat([z0, dt], dim=-1)
        hz = self.trans_encoder(zt_expand)

        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        zt = z0
        Zt_k = []
        Yt_k = []
        for ut in U:
            ut_dt = ut * dt
            zt = (torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1)
                  + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))

            # Per-well observation with spatial features
            zu = torch.cat([zt, ut], dim=-1)
            well_outputs = []
            for i, head in enumerate(self.well_heads):
                # Concatenate z, u, and well-specific features
                well_feat_i = well_features[:, i, :]  # (B, 128)
                head_input = torch.cat([zu, well_feat_i], dim=-1)
                well_outputs.append(head(head_input))
            inj_output = self.inj_head(zu)
            yt = torch.cat(well_outputs + [inj_output], dim=-1)

            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut, well_features):
        """Single-step with well features."""
        zt_expand = torch.cat([zt, dt], dim=-1)
        hz = self.trans_encoder(zt_expand)

        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        ut_dt = ut * dt
        zt1 = (torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1)
               + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))

        zu = torch.cat([zt1, ut], dim=-1)
        well_outputs = []
        for i, head in enumerate(self.well_heads):
            well_feat_i = well_features[:, i, :]
            head_input = torch.cat([zu, well_feat_i], dim=-1)
            well_outputs.append(head(head_input))
        inj_output = self.inj_head(zu)
        yt1 = torch.cat(well_outputs + [inj_output], dim=-1)

        return zt1, yt1


class WellFeatureE2CO(nn.Module):
    """V21 Well Feature E2CO — encoder features at well locations bypass bottleneck."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0,
                 prod_loc=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps

        if prod_loc is None:
            prod_loc = [[4, 16], [26, 16], [30, 31], [36, 45], [7, 49]]

        self.encoder = WellFeatureEncoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = Decoder(latent_dim, input_shape)
        self.well_extractor = WellFeatureExtractor(
            prod_loc, feature_map_size=16,
            input_size=input_shape[1], feature_channels=128)
        self.transition = WellAwareTransition(latent_dim, u_dim,
                                               num_prod, num_inj,
                                               feature_channels=128)

    def forward(self, inputs):
        X, U, Y, dt = inputs
        x0 = X[0]

        z0, feat0 = self.encoder(x0)
        x0_rec = self.decoder(z0)
        well_features = self.well_extractor(feat0)

        Z_next_pred, Y_next_pred = self.transition.forward_nsteps(
            z0, dt, U, well_features)

        X_next_pred = []
        Z_next = []
        X_next = X[1:]

        for i_step in range(len(Z_next_pred)):
            z_next_pred = Z_next_pred[i_step]
            x_next_pred = self.decoder(z_next_pred)
            z_next_true = self.encoder.encode_z_only(X[i_step + 1])
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
        self.eval()
        with torch.no_grad():
            zt, feat = self.encoder(xt)
            well_features = self.well_extractor(feat)
            zt_next, yt_next = self.transition(zt, dt, ut, well_features)
            xt_next_pred = self.decoder(zt_next)
        return xt_next_pred, yt_next

    def predict_latent(self, zt, dt, ut):
        """Latent-space single-step — no well features available."""
        self.eval()
        with torch.no_grad():
            # Use zero well features as fallback
            B = zt.shape[0]
            dummy_feats = torch.zeros(B, self.transition.num_prod, 128,
                                     device=zt.device)
            zt_next, yt_next = self.transition(zt, dt, ut, dummy_feats)
        return zt_next, yt_next

    def save_checkpoint(self, path, optimizer=None, epoch=0, best_loss=1e9,
                        adaptive_weights=None):
        state = {
            'model_state': self.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }
        if optimizer is not None:
            state['optimizer_state'] = optimizer.state_dict()
        if adaptive_weights is not None:
            state['adaptive_weights'] = adaptive_weights
        torch.save(state, path)

    @staticmethod
    def load_checkpoint(path, model, optimizer=None):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        return ckpt.get('epoch', 0), ckpt.get('best_loss', 1e9), ckpt.get('adaptive_weights', None)

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir):
        pattern = os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")
        files = sorted(glob.glob(pattern))
        return files[-1] if files else None
