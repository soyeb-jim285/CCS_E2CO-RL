"""V22 Full Combo E2CO — combines all winning ingredients.

- WellFeatureEncoder from V21 (returns z + features)
- FNODecoder from V6 (spectral decoder)
- WellFeatureExtractor from V21
- FullComboODETransition: StructuredODEFunc from V14 + per-well MLP obs from V21
"""

import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    raise ImportError("V22 requires torchdiffeq: pip install torchdiffeq")

from versions.v21_well_features.model import WellFeatureEncoder, WellFeatureExtractor
from versions.v6_fno_decoder.model import FNODecoder
from versions.v14_latent_physics.model import StructuredODEFunc


class FullComboODETransition(nn.Module):
    """ODE dynamics (A=-LL^T) + per-well MLP obs with feature map context."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj,
                 feature_channels=128, method='euler'):
        super().__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim
        self.num_prod = num_prod
        self.num_inj = num_inj
        self.obs_dim = num_prod * 2 + num_inj  # 14
        self.method = method

        # ODE dynamics from V14
        self.ode_func = StructuredODEFunc(latent_dim, u_dim)

        # Per-well MLP: input = cat(z, u, well_features) = 20 + 9 + 128 = 157
        well_input_dim = latent_dim + u_dim + feature_channels

        self.well_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(well_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            for _ in range(num_prod)
        ])

        # Injector head
        self.inj_head = nn.Linear(latent_dim + u_dim, num_inj)

    def forward_nsteps(self, z0, dt, U, well_features):
        zt = z0
        Zt_k = []
        Yt_k = []
        for ut in U:
            # ODE integration for z dynamics
            self.ode_func.u = ut
            t_span = torch.tensor([0.0, 1.0], device=z0.device, dtype=z0.dtype)
            zt = odeint(self.ode_func, zt, t_span, method=self.method,
                        rtol=1e-3, atol=1e-3)[-1]

            # Per-well observation with spatial features
            zu = torch.cat([zt, ut], dim=-1)
            well_outputs = []
            for i, head in enumerate(self.well_heads):
                well_feat_i = well_features[:, i, :]
                head_input = torch.cat([zu, well_feat_i], dim=-1)
                well_outputs.append(head(head_input))
            inj_output = self.inj_head(zu)
            yt = torch.cat(well_outputs + [inj_output], dim=-1)

            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut, well_features):
        self.ode_func.u = ut
        t_span = torch.tensor([0.0, 1.0], device=zt.device, dtype=zt.dtype)
        zt1 = odeint(self.ode_func, zt, t_span, method=self.method,
                     rtol=1e-3, atol=1e-3)[-1]

        zu = torch.cat([zt1, ut], dim=-1)
        well_outputs = []
        for i, head in enumerate(self.well_heads):
            well_feat_i = well_features[:, i, :]
            head_input = torch.cat([zu, well_feat_i], dim=-1)
            well_outputs.append(head(head_input))
        inj_output = self.inj_head(zu)
        yt1 = torch.cat(well_outputs + [inj_output], dim=-1)

        return zt1, yt1


class FullComboE2CO(nn.Module):
    """V22 Full Combo E2CO — ODE dynamics + FNO decoder + well feature obs."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0,
                 fno_modes=12, fno_width=32, ode_method='euler',
                 prod_loc=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps

        if prod_loc is None:
            prod_loc = [[4, 16], [26, 16], [30, 31], [36, 45], [7, 49]]

        self.encoder = WellFeatureEncoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = FNODecoder(latent_dim, input_shape,
                                  modes=fno_modes, width=fno_width)
        self.well_extractor = WellFeatureExtractor(
            prod_loc, feature_map_size=16,
            input_size=input_shape[1], feature_channels=128)
        self.transition = FullComboODETransition(
            latent_dim, u_dim, num_prod, num_inj,
            feature_channels=128, method=ode_method)

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
        self.eval()
        with torch.no_grad():
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
