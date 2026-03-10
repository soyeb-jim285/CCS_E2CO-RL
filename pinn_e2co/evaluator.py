"""Evaluation and visualization — matches SC notebook plots + new PINN metrics."""

import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'

from .utils import (compute_rmse, compute_mae, denormalize_pressure,
                    denormalize_saturation, denormalize_rates)


class Evaluator:
    """Generates all plots: SC-notebook-matching + new PINN-specific."""

    def __init__(self, model, loss_fn, cfg, device):
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device

    def run_sequential_eval(self, test_data, perm):
        """Run sequential prediction on test cases, return results dict."""
        cfg = self.cfg
        model = self.model
        model.eval()

        num_case = test_data['num_case']
        num_tstep = test_data['num_tstep']
        Nx, Ny = cfg.Nx, cfg.Ny
        num_prod = cfg.num_prod
        num_inj = cfg.num_inj

        sat_pred = torch.zeros((num_case, num_tstep, 1, Nx, Ny),
                               dtype=torch.float32, device=self.device)
        pres_pred = torch.zeros((num_case, num_tstep, 1, Nx, Ny),
                                dtype=torch.float32, device=self.device)
        yobs_pred = torch.zeros((num_case, num_tstep, num_prod * 2 + num_inj),
                                dtype=torch.float32, device=self.device)

        state_pred = torch.cat((sat_pred, pres_pred), dim=2)
        state_t_seq = test_data['state_t_seq'].clone()
        bhp_seq = test_data['bhp_seq']
        yobs_t_seq = test_data['yobs_t_seq']
        indt_del = test_data['indt_del']

        with torch.no_grad():
            for i_tstep in range(num_tstep):
                state_pred[:, i_tstep, ...] = state_t_seq
                dt_seq = torch.tensor(
                    np.ones((num_case, 1)) * indt_del[i_tstep],
                    dtype=torch.float32, device=self.device)
                xt_next, yt_next = model.predict(
                    state_t_seq, bhp_seq[:, i_tstep, :],
                    yobs_t_seq[:, i_tstep, :], dt_seq)
                state_t_seq = xt_next
                yobs_pred[:, i_tstep, :] = yt_next

        return {
            'state_pred': state_pred,
            'yobs_pred': yobs_pred,
        }

    def generate_all_plots(self, test_data, perm, results):
        """Generate all SC-matching + new plots."""
        plot_dir = self.cfg.plot_dir
        os.makedirs(plot_dir, exist_ok=True)

        state_pred = results['state_pred']
        yobs_pred = results['yobs_pred']

        # Build ground truth
        sat_seq_true = test_data['sat_seq_true']
        pres_seq_true = test_data['pres_seq_true']
        num_case = test_data['num_case']
        Nx, Ny = self.cfg.Nx, self.cfg.Ny

        state_seq_true = torch.zeros((num_case, 2, 21, Nx, Ny))
        state_seq_true[:, 0, :, :] = sat_seq_true
        state_seq_true[:, 1, :, :] = pres_seq_true

        yobs_seq_true = torch.swapaxes(
            torch.cat((
                test_data['sat_seq_true'][:, :1, :, :].squeeze(2).squeeze(2),  # dummy
            ), dim=1), 1, 2) if False else None
        # Reload yobs true from test_data
        # yobs_seq_true is already in test_data via yobs_t_seq
        yobs_seq_true_raw = test_data['yobs_t_seq'].clone()

        # SC-matching plots
        self._plot_permeability_field(perm, plot_dir)
        self._plot_saturation_maps(state_pred, state_seq_true, test_data, plot_dir)
        self._plot_pressure_maps(state_pred, state_seq_true, test_data, plot_dir)
        self._plot_well_outputs(yobs_pred, yobs_seq_true_raw, test_data, plot_dir)
        self._plot_r2_crossplots(yobs_pred, yobs_seq_true_raw, test_data, plot_dir)

        # New PINN-specific plots
        self._plot_training_curves(plot_dir)
        self._plot_per_timestep_errors(state_pred, state_seq_true, test_data, plot_dir)
        self._plot_error_histograms(state_pred, state_seq_true, test_data, plot_dir)
        self._plot_physics_residuals(state_pred, test_data, plot_dir)
        self._plot_adaptive_weights(plot_dir)

        print(f"All plots saved to {plot_dir}")

    # ===== SC-Notebook Matching Plots =====

    def _plot_permeability_field(self, perm, plot_dir):
        """Permeability field with well locations (SC notebook cell 26)."""
        import scipy.io as scio
        cfg = self.cfg
        Inj_loc = np.array(cfg.inj_loc)
        Prod_loc = np.array(cfg.prod_loc)

        m = perm.squeeze().numpy() if isinstance(perm, torch.Tensor) else perm.squeeze()

        fig = plt.figure()
        plt.imshow(m)
        plt.axis('off')
        for i_inj in range(len(Inj_loc)):
            plt.plot(Inj_loc[i_inj, 0], Inj_loc[i_inj, 1],
                     marker='*', color='black', markersize=10)
            plt.text(Inj_loc[i_inj, 0] + 4, Inj_loc[i_inj, 1],
                     'I%d' % (i_inj + 1), ha="center", va="center",
                     color="black", fontsize=16)
        for i_prod in range(len(Prod_loc)):
            plt.plot(Prod_loc[i_prod, 0], Prod_loc[i_prod, 1],
                     marker='.', color='black', markersize=10)
            plt.text(Prod_loc[i_prod, 0] + 4, Prod_loc[i_prod, 1],
                     'P%d' % (i_prod + 1), ha="center", va="center",
                     color="black", fontsize=16)
        cbar = plt.colorbar(orientation="vertical")
        cbar.ax.set_ylabel('log(K), mD')
        plt.savefig(os.path.join(plot_dir, 'permeability_field.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_saturation_maps(self, state_pred, state_seq_true, test_data, plot_dir):
        """Saturation maps: 3 rows (pred/true/|error|), 10 cols."""
        cfg = self.cfg
        ind_case = cfg.ind_case
        t_steps = test_data['t_steps']
        dt_val = test_data['dt_val']
        num_tstep = test_data['num_tstep']

        s_max, s_min = 1.0, 0.0
        s_diff = s_max - s_min

        # Denormalize saturation (channel 0)
        sat_pred_plot = state_pred[:, :, 0, :, :].cpu() * s_diff + s_min

        divide = 2
        for k_idx, k in enumerate(ind_case):
            fig = plt.figure(figsize=(16, 5))
            for i_tstep in range(num_tstep // divide):
                t_idx = i_tstep * divide

                # Predicted
                ax = plt.subplot(3, num_tstep // divide, i_tstep + 1)
                plt.imshow(sat_pred_plot[k, t_idx, :, :].detach().numpy())
                plt.title(f't={t_steps[t_idx] * dt_val}')
                plt.xticks([])
                plt.yticks([])
                plt.clim([0.0, 1.0])
                if i_tstep == 9:
                    cbar = plt.colorbar(fraction=0.046)
                    cbar.ax.set_ylabel('fraction')

                # True
                plt.subplot(3, num_tstep // divide, i_tstep + 1 + num_tstep // divide)
                plt.imshow(state_seq_true[k, 0, t_idx, :].numpy())
                plt.xticks([])
                plt.yticks([])
                plt.clim([0.0, 1.0])
                if i_tstep == 9:
                    cbar = plt.colorbar(fraction=0.046)
                    cbar.ax.set_ylabel('fraction')

                # Error
                plt.subplot(3, num_tstep // divide, i_tstep + 1 + 2 * num_tstep // divide)
                error = torch.abs(
                    state_seq_true[k, 0, t_idx, ...] - sat_pred_plot[k, t_idx, :, :].detach())
                plt.imshow(error.numpy())
                plt.xticks([])
                plt.yticks([])
                plt.clim([0, 0.15])
                if i_tstep == 9:
                    cbar = plt.colorbar(fraction=0.046)
                    cbar.ax.set_ylabel('fraction')

            # Add RMSE/MAE annotation
            sat_p = sat_pred_plot[k].detach().numpy()
            sat_t = (state_seq_true[k, 0, :num_tstep, ...]).numpy()
            rmse = np.sqrt(np.mean((sat_p - sat_t) ** 2))
            mae = np.mean(np.abs(sat_p - sat_t))
            fig.suptitle(f'Case {k} | RMSE={rmse:.4f}, MAE={mae:.4f}', fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'saturation_case_{k}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_pressure_maps(self, state_pred, state_seq_true, test_data, plot_dir):
        """Pressure maps: 3 rows (pred/true/|error|), 10 cols."""
        cfg = self.cfg
        ind_case = cfg.ind_case
        t_steps = test_data['t_steps']
        dt_val = test_data['dt_val']
        num_tstep = test_data['num_tstep']

        p_min, p_max = cfg.p_min, cfg.p_max
        p_diff = p_max - p_min

        # Denormalize pressure (channel 1/-1)
        pres_pred_plot = state_pred[:, :, 1, :, :].cpu() * p_diff + p_min
        pres_true_plot = state_seq_true[:, 1, :, :] * p_diff + p_min

        divide = 2
        for k_idx, k in enumerate(ind_case):
            fig = plt.figure(figsize=(16, 5))
            for i_tstep in range(num_tstep // divide):
                t_idx = i_tstep * divide

                # Predicted
                plt.subplot(3, num_tstep // divide, i_tstep + 1)
                plt.imshow(pres_pred_plot[k, t_idx, :, :].detach().numpy())
                plt.title(f't={t_steps[t_idx] * dt_val}')
                plt.xticks([])
                plt.yticks([])
                plt.clim([2200, 2500])
                if i_tstep == 9:
                    cbar = plt.colorbar(fraction=0.046)
                    cbar.ax.set_ylabel('in psia')

                # True
                plt.subplot(3, num_tstep // divide, i_tstep + 1 + num_tstep // divide)
                plt.imshow(pres_true_plot[k, t_idx, ...].numpy())
                plt.xticks([])
                plt.yticks([])
                plt.clim([2200, 2500])
                if i_tstep == 9:
                    cbar = plt.colorbar(fraction=0.046)
                    cbar.ax.set_ylabel('in psia')

                # Relative error
                plt.subplot(3, num_tstep // divide, i_tstep + 1 + 2 * num_tstep // divide)
                rel_err = np.abs(
                    pres_true_plot[k, t_idx, ...].numpy()
                    - pres_pred_plot[k, t_idx, :, :].detach().numpy()
                ) / pres_true_plot[k, t_idx, ...].numpy()
                plt.imshow(rel_err)
                plt.xticks([])
                plt.yticks([])
                if i_tstep == 9:
                    cbar = plt.colorbar(fraction=0.046)
                    cbar.ax.set_ylabel('fraction')

            # RMSE/MAE annotation
            pp = pres_pred_plot[k].detach().numpy()
            pt = pres_true_plot[k, :num_tstep, ...].numpy()
            rmse = np.sqrt(np.mean((pp - pt) ** 2))
            mae = np.mean(np.abs(pp - pt))
            fig.suptitle(f'Case {k} | RMSE={rmse:.2f} psia, MAE={mae:.2f} psia', fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'pressure_case_{k}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_well_outputs(self, yobs_pred, yobs_seq_true, test_data, plot_dir):
        """Well output plots: case 77, linewidth=5.0."""
        cfg = self.cfg
        num_prod = cfg.num_prod
        p_min, p_max = cfg.p_min, cfg.p_max
        p_diff = p_max - p_min
        Q_max_w = cfg.Q_max_w
        Q_max_g = cfg.Q_max_g

        # Denormalize
        yobs_pred_denorm = yobs_pred.clone().cpu().detach()
        yobs_true_denorm = yobs_seq_true.clone().cpu().detach()

        yobs_pred_denorm[:, :, :num_prod] *= Q_max_w
        yobs_true_denorm[:, :, :num_prod] *= Q_max_w
        yobs_pred_denorm[:, :, num_prod:2 * num_prod] *= Q_max_g
        yobs_true_denorm[:, :, num_prod:2 * num_prod] *= Q_max_g
        yobs_pred_denorm[:, :, 2 * num_prod:] = yobs_pred_denorm[:, :, 2 * num_prod:] * p_diff + p_min
        yobs_true_denorm[:, :, 2 * num_prod:] = yobs_true_denorm[:, :, 2 * num_prod:] * p_diff + p_min

        eval_case = cfg.eval_case
        num_wells = yobs_pred_denorm.shape[-1]

        for i_well in range(num_wells):
            fig = plt.figure(figsize=(16, 5))
            pred_vals = yobs_pred_denorm[eval_case, :, i_well].numpy()
            true_vals = yobs_true_denorm[eval_case, :, i_well].numpy()

            plt.plot(pred_vals, linewidth=5.0)
            plt.plot(true_vals, linewidth=5.0)
            plt.xticks([])
            plt.yticks(fontsize=20)

            if i_well < num_prod:
                plt.ylabel('STB/Day', fontsize=20)
            elif i_well > num_prod * 2:
                # Matches SC notebook cell 77: i_well>num_prod*2
                plt.ylabel('psia', fontsize=20)
            else:
                plt.ylabel('ft^3/Day', fontsize=20)
                plt.ylim((-100000, 1500000))

            plt.legend(['prediction', 'true'], fontsize=20)

            # RMSE/MAE annotation
            rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
            mae = np.mean(np.abs(pred_vals - true_vals))
            plt.title(f'Well {i_well + 1} | RMSE={rmse:.2f}, MAE={mae:.2f}', fontsize=14)

            plt.savefig(os.path.join(plot_dir, f'well_{i_well + 1}_case{eval_case}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_r2_crossplots(self, yobs_pred, yobs_seq_true, test_data, plot_dir):
        """R2 cross-plots for cumulative production."""
        from sklearn.metrics import r2_score
        cfg = self.cfg
        num_prod = cfg.num_prod
        p_diff = cfg.p_max - cfg.p_min
        Q_max_w = cfg.Q_max_w
        Q_max_g = cfg.Q_max_g

        # Denormalize
        yobs_pred_d = yobs_pred.clone().cpu().detach()
        yobs_true_d = yobs_seq_true.clone().cpu().detach()

        yobs_pred_d[:, :, :num_prod] *= Q_max_w
        yobs_true_d[:, :, :num_prod] *= Q_max_w
        yobs_pred_d[:, :, num_prod:2 * num_prod] *= Q_max_g
        yobs_true_d[:, :, num_prod:2 * num_prod] *= Q_max_g
        yobs_pred_d[:, :, 2 * num_prod:] = yobs_pred_d[:, :, 2 * num_prod:] * p_diff + cfg.p_min
        yobs_true_d[:, :, 2 * num_prod:] = yobs_true_d[:, :, 2 * num_prod:] * p_diff + cfg.p_min

        # Cumulative production
        cum_pred = 100 * torch.sum(yobs_pred_d[:, :, :num_prod * 2], dim=1).numpy()
        cum_true = 100 * torch.sum(yobs_true_d[:, :, :num_prod * 2], dim=1).numpy()

        for i_well in range(cum_pred.shape[-1]):
            fig = plt.figure(figsize=(16, 8))

            if i_well < num_prod:
                r2 = r2_score(cum_true[:, i_well] / 1000, cum_pred[:, i_well] / 1000)
                plt.scatter(cum_true[:, i_well] / 1000, cum_pred[:, i_well] / 1000)
                plt.plot(cum_true[:, i_well] / 1000, cum_true[:, i_well] / 1000, color='r')
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('Cumulative water (reference), 1000 STB', fontsize=20)
                plt.ylabel('Cumulative water (E2CO), 1000 STB', fontsize=20)
                plt.figtext(0.40, 0.77, f'R2 = {round(r2, 3)}', fontsize=20)
                vmin = np.min(cum_pred[:, i_well] / 1000)
                vmax = np.max(cum_pred[:, i_well] / 1000)
                plt.xlim((vmin, vmax))
                plt.ylim((vmin, vmax))

                # RMSE annotation
                rmse = np.sqrt(np.mean((cum_true[:, i_well] / 1000 - cum_pred[:, i_well] / 1000) ** 2))
                mae = np.mean(np.abs(cum_true[:, i_well] / 1000 - cum_pred[:, i_well] / 1000))
                plt.figtext(0.40, 0.72, f'RMSE={rmse:.2f}, MAE={mae:.2f}', fontsize=14)

            else:
                r2 = r2_score(cum_true[:, i_well] / 1000, cum_pred[:, i_well] / 1000)
                plt.scatter(cum_true[:, i_well] / 1e6, cum_pred[:, i_well] / 1e6)
                plt.plot(cum_true[:, i_well] / 1e6, cum_true[:, i_well] / 1e6, color='r')
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.xlabel('Cumulative gas (reference), 10^6 ft^3', fontsize=20)
                plt.ylabel('Cumulative gas (E2CO), 10^6 ft^3', fontsize=20)
                plt.figtext(0.40, 0.77, f'R2 = {round(r2, 3)}', fontsize=20)
                vmin = np.min(cum_pred[:, i_well] / 1e6)
                vmax = np.max(cum_pred[:, i_well] / 1e6)
                plt.xlim((vmin, vmax))
                plt.ylim((vmin, vmax))

                rmse = np.sqrt(np.mean((cum_true[:, i_well] / 1e6 - cum_pred[:, i_well] / 1e6) ** 2))
                mae = np.mean(np.abs(cum_true[:, i_well] / 1e6 - cum_pred[:, i_well] / 1e6))
                plt.figtext(0.40, 0.72, f'RMSE={rmse:.2f}, MAE={mae:.2f}', fontsize=14)

            plt.axis('square')
            plt.savefig(os.path.join(plot_dir, f'r2_well_{i_well + 1}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ===== New PINN-Specific Plots =====

    def _plot_training_curves(self, plot_dir):
        """Plot training loss curves from CSV."""
        csv_path = os.path.join(self.cfg.log_dir, "loss_history.csv")
        if not os.path.exists(csv_path):
            print("No loss_history.csv found, skipping training curves.")
            return

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return

        epochs = [int(r['epoch']) for r in rows]

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # --- Top panel: Total train + eval loss ---
        ax = axes[0]
        train_vals = [float(r['train_loss']) for r in rows]
        ax.plot(epochs, train_vals, 'b-', linewidth=2, label='Train Total')

        # Eval loss: only plot epochs where eval was actually run
        eval_epochs, eval_vals = [], []
        for r in rows:
            v = r.get('eval_loss', '')
            if v != '' and v is not None:
                try:
                    eval_vals.append(float(v))
                    eval_epochs.append(int(r['epoch']))
                except (ValueError, TypeError):
                    pass
        if eval_vals:
            ax.plot(eval_epochs, eval_vals, 'ro-', markersize=5,
                    linewidth=2, label='Eval Total')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # --- Bottom panel: Sub-losses ---
        ax = axes[1]
        sub_loss_keys = [k for k in rows[0].keys()
                         if k.startswith('train_') and k != 'train_loss'
                         and k != 'train_total']
        for key in sorted(sub_loss_keys):
            vals, eps = [], []
            for r in rows:
                v = r.get(key, '')
                if v != '' and v is not None:
                    try:
                        vals.append(float(v))
                        eps.append(int(r['epoch']))
                    except (ValueError, TypeError):
                        pass
            if vals:
                ax.plot(eps, vals, linewidth=1.5,
                        label=key.replace('train_', ''))

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Sub-Losses', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right', ncol=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'training_curves.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_per_timestep_errors(self, state_pred, state_seq_true, test_data, plot_dir):
        """Per-timestep RMSE/MAE bar charts for saturation and pressure."""
        cfg = self.cfg
        num_tstep = test_data['num_tstep']
        p_diff = cfg.p_max - cfg.p_min

        sat_pred = state_pred[:, :, 0, :, :].cpu().detach().numpy()
        sat_true = state_seq_true[:, 0, :num_tstep, ...].numpy()
        pres_pred = state_pred[:, :, 1, :, :].cpu().detach().numpy() * p_diff + cfg.p_min
        pres_true = state_seq_true[:, 1, :num_tstep, ...].numpy() * p_diff + cfg.p_min

        sat_rmse = np.zeros(num_tstep)
        sat_mae = np.zeros(num_tstep)
        pres_rmse = np.zeros(num_tstep)
        pres_mae = np.zeros(num_tstep)

        for t in range(num_tstep):
            sat_rmse[t] = np.sqrt(np.mean((sat_pred[:, t] - sat_true[:, t]) ** 2))
            sat_mae[t] = np.mean(np.abs(sat_pred[:, t] - sat_true[:, t]))
            pres_rmse[t] = np.sqrt(np.mean((pres_pred[:, t] - pres_true[:, t]) ** 2))
            pres_mae[t] = np.mean(np.abs(pres_pred[:, t] - pres_true[:, t]))

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        x = np.arange(num_tstep)

        axes[0, 0].bar(x, sat_rmse, color='steelblue')
        axes[0, 0].set_title('Saturation RMSE per Timestep')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('RMSE')

        axes[0, 1].bar(x, sat_mae, color='darkorange')
        axes[0, 1].set_title('Saturation MAE per Timestep')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('MAE')

        axes[1, 0].bar(x, pres_rmse, color='steelblue')
        axes[1, 0].set_title('Pressure RMSE per Timestep (psia)')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('RMSE (psia)')

        axes[1, 1].bar(x, pres_mae, color='darkorange')
        axes[1, 1].set_title('Pressure MAE per Timestep (psia)')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('MAE (psia)')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'per_timestep_errors.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_error_histograms(self, state_pred, state_seq_true, test_data, plot_dir):
        """Pixel-wise error histograms for saturation and pressure."""
        cfg = self.cfg
        num_tstep = test_data['num_tstep']
        p_diff = cfg.p_max - cfg.p_min

        sat_err = (state_pred[:, :, 0, :, :].cpu().detach().numpy()
                   - state_seq_true[:, 0, :num_tstep, ...].numpy()).flatten()
        pres_err = ((state_pred[:, :, 1, :, :].cpu().detach().numpy() * p_diff + cfg.p_min)
                    - (state_seq_true[:, 1, :num_tstep, ...].numpy() * p_diff + cfg.p_min)).flatten()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(sat_err, bins=100, color='steelblue', alpha=0.7, density=True)
        axes[0].set_title('Saturation Error Distribution')
        axes[0].set_xlabel('Error')
        axes[0].set_ylabel('Density')
        axes[0].axvline(0, color='r', linestyle='--', alpha=0.5)

        axes[1].hist(pres_err, bins=100, color='darkorange', alpha=0.7, density=True)
        axes[1].set_title('Pressure Error Distribution (psia)')
        axes[1].set_xlabel('Error (psia)')
        axes[1].set_ylabel('Density')
        axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'error_histograms.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_physics_residuals(self, state_pred, test_data, plot_dir):
        """Physics residual heatmaps at selected timesteps."""
        if self.loss_fn is None:
            return

        cfg = self.cfg
        ind_case = cfg.ind_case
        # Pick timesteps: 0, 5, 10, 15, 19
        sel_tsteps = [0, 5, 10, 15, 19]

        for case_idx in ind_case[:2]:  # just first 2 cases
            fig, axes = plt.subplots(2, len(sel_tsteps), figsize=(20, 8))

            for col, t in enumerate(sel_tsteps):
                if t == 0:
                    continue  # can't compute residual at t=0

                x_t = state_pred[case_idx:case_idx + 1, t - 1, ...].to(self.device)
                x_t1 = state_pred[case_idx:case_idx + 1, t, ...].to(self.device)

                with torch.no_grad():
                    pres_res, mass_res = self.loss_fn.compute_physics_residuals(x_t, x_t1)

                ax = axes[0, col]
                im = ax.imshow(pres_res[0, 0].cpu().numpy(), cmap='RdBu_r')
                ax.set_title(f'Pressure PDE t={t}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)

                ax = axes[1, col]
                im = ax.imshow(mass_res[0, 0].cpu().numpy(), cmap='RdBu_r')
                ax.set_title(f'Mass Cons. t={t}')
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)

            # Handle t=0 column
            for row in range(2):
                axes[row, 0].text(0.5, 0.5, 'N/A\n(t=0)',
                                  ha='center', va='center',
                                  transform=axes[row, 0].transAxes)
                axes[row, 0].set_xticks([])
                axes[row, 0].set_yticks([])

            fig.suptitle(f'Physics Residuals — Case {case_idx}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'physics_residuals_case_{case_idx}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_adaptive_weights(self, plot_dir):
        """Plot adaptive weight (sigma) evolution over epochs."""
        csv_path = os.path.join(self.cfg.log_dir, "loss_history.csv")
        if not os.path.exists(csv_path):
            return

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return

        sigma_keys = [k for k in rows[0].keys() if k.startswith('sigma_')]
        if not sigma_keys:
            print("No adaptive weight data found, skipping sigma plot.")
            return

        epochs = [int(r['epoch']) for r in rows]

        fig = plt.figure(figsize=(14, 6))
        for key in sigma_keys:
            try:
                vals = [float(r.get(key, 1.0)) for r in rows]
                plt.plot(epochs, vals, label=key.replace('sigma_', ''), linewidth=2)
            except (ValueError, TypeError):
                pass

        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Sigma', fontsize=14)
        plt.title('Adaptive Loss Weight Evolution', fontsize=16)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, 'adaptive_weights.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
