"""V19 Per-Well MLP Observation Heads — nonlinear per-well observation model.

Replaces the linear C/D observation model with independent MLPs per well.
Encoder, decoder, and z dynamics (A/B) reuse V0 architecture.
"""

import os
import glob
import torch
import torch.nn as nn

from versions.v0_baseline.model import (
    Encoder, Decoder, weights_init, fc_bn_relu_2arg, create_trans_encoder
)


class PerWellMLPTransition(nn.Module):
    """Transition with V0 z dynamics (A/B) + per-well MLP observation heads."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj):
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

        # Per-well MLP heads: input = cat(z, u) = latent_dim + u_dim
        zu_dim = latent_dim + u_dim  # 20 + 9 = 29

        # 5 production wells: each outputs [water_rate, gas_rate]
        self.well_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(zu_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )
            for _ in range(num_prod)
        ])

        # 1 injector head: outputs BHP for all injectors
        self.inj_head = nn.Linear(zu_dim, num_inj)

    def forward_nsteps(self, zt, dt, U):
        zt_expand = torch.cat([zt, dt], dim=-1)
        hz = self.trans_encoder(zt_expand)

        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        Zt_k = []
        Yt_k = []
        for ut in U:
            ut_dt = ut * dt
            zt = (torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1)
                  + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))

            # Observation via per-well MLPs
            zu = torch.cat([zt, ut], dim=-1)
            well_outputs = [head(zu) for head in self.well_heads]
            inj_output = self.inj_head(zu)
            yt = torch.cat(well_outputs + [inj_output], dim=-1)

            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut):
        zt_expand = torch.cat([zt, dt], dim=-1)
        hz = self.trans_encoder(zt_expand)

        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)

        ut_dt = ut * dt
        zt1 = (torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1)
               + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))

        zu = torch.cat([zt1, ut], dim=-1)
        well_outputs = [head(zu) for head in self.well_heads]
        inj_output = self.inj_head(zu)
        yt1 = torch.cat(well_outputs + [inj_output], dim=-1)

        return zt1, yt1


class PerWellMLPE2CO(nn.Module):
    """V19 Per-Well MLP E2CO — V0 encoder/decoder + per-well MLP observation."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps

        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = Decoder(latent_dim, input_shape)
        self.transition = PerWellMLPTransition(latent_dim, u_dim,
                                                num_prod, num_inj)

    def forward(self, inputs):
        X, U, Y, dt = inputs
        x0 = X[0]
        z0 = self.encoder(x0)
        x0_rec = self.decoder(z0)

        Z_next_pred, Y_next_pred = self.transition.forward_nsteps(z0, dt, U)

        X_next_pred = []
        Z_next = []
        X_next = X[1:]

        for i_step in range(len(Z_next_pred)):
            z_next_pred = Z_next_pred[i_step]
            x_next_pred = self.decoder(z_next_pred)
            z_next_true = self.encoder(X[i_step + 1])
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
            zt = self.encoder(xt)
            zt_next, yt_next = self.transition(zt, dt, ut)
            xt_next_pred = self.decoder(zt_next)
        return xt_next_pred, yt_next

    def predict_latent(self, zt, dt, ut):
        self.eval()
        with torch.no_grad():
            zt_next, yt_next = self.transition(zt, dt, ut)
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
