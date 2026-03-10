"""Data loading — identical for all versions. Copied from pinn_e2co/data_loader.py."""

import numpy as np
import h5py
import scipy.io as scio
import torch


class VersionDataLoader:
    """Loads and splits data identically for all versions."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = None

    def load_all(self, device):
        """Load all data, return train/eval tensors + permeability field."""
        self.device = device
        cfg = self.cfg

        Mole_slt, SAT_slt, PRES_slt, BHP_slt, Yobs_slt, num_t_slt, Nx, Ny, \
            num_well, num_prod, num_inj = self._prepare_data()

        STATE_train, BHP_train, Yobs_train, \
            STATE_eval, BHP_eval, Yobs_eval = self._train_split_data(
                Mole_slt, SAT_slt, PRES_slt, BHP_slt, Yobs_slt,
                num_t_slt, Nx, Ny, num_well, num_prod, num_inj)

        num_train = STATE_train[0].shape[0]
        num_eval = STATE_eval[0].shape[0]
        dt_train = torch.ones((num_train, 1), dtype=torch.float32, device=device)
        dt_eval = torch.ones((num_eval, 1), dtype=torch.float32, device=device)

        perm = self._load_permeability()

        train_data = {
            'STATE': STATE_train,
            'BHP': BHP_train,
            'Yobs': Yobs_train,
            'dt': dt_train,
            'num_train': num_train,
        }
        eval_data = {
            'STATE': STATE_eval,
            'BHP': BHP_eval,
            'Yobs': Yobs_eval,
            'dt': dt_eval,
            'num_eval': num_eval,
        }
        return train_data, eval_data, perm

    def load_test_data(self, device):
        """Load full data for sequential evaluation (100 test cases)."""
        self.device = device
        cfg = self.cfg

        hf = h5py.File(cfg.data_dir + cfg.state_file, 'r')
        sat = torch.tensor(
            np.array(hf.get('Mole_frac_norm_slt')).transpose((3, 2, 1, 0)),
            dtype=torch.float32)
        pres = torch.tensor(
            np.array(hf.get('Psim_norm_slt')).transpose((3, 2, 1, 0)),
            dtype=torch.float32)
        hf.close()

        hf = h5py.File(cfg.data_dir + cfg.ctrl_file, 'r')
        bhp0 = torch.tensor(
            np.array(hf.get('Pwf_norm_slt')).transpose((2, 1, 0)),
            dtype=torch.float32)
        rate0 = torch.tensor(
            np.array(hf.get('Qinj_norm_slt')).transpose((2, 1, 0)),
            dtype=torch.float32)
        hf.close()
        bhp = torch.cat((bhp0, rate0), dim=1)

        hf = h5py.File(cfg.data_dir + cfg.yobs_file, 'r')
        Qrate_w = torch.tensor(
            np.array(hf.get('Qpro_w_norm_slt')).transpose((2, 1, 0)),
            dtype=torch.float32)
        Qrate_g = torch.tensor(
            np.array(hf.get('Qpro_g_norm_slt')).transpose((2, 1, 0)),
            dtype=torch.float32)
        BHP_inj = torch.tensor(
            np.array(hf.get('BHPinj_norm_slt')).transpose((2, 1, 0)),
            dtype=torch.float32)
        hf.close()
        yobs = torch.cat((Qrate_w, Qrate_g, BHP_inj), dim=1)

        test_case0 = np.zeros((25, 4))
        a = np.array(range(75, 400, 100))[np.newaxis, :]
        b = np.array(range(25))[:, np.newaxis]
        test_case = (test_case0 + a + b).T.reshape(100).astype(int)

        num_tstep = 20
        tmp1 = np.array(range(num_tstep))
        bhp_tt1 = bhp[:, :, tmp1]
        bhp_t = torch.swapaxes(bhp_tt1, 1, 2).to(device)
        bhp_seq = bhp_t[test_case, :, :]

        sat_t_seq = sat[test_case, 0:1, ...].to(device)
        pres_t_seq = pres[test_case, 0:1, ...].to(device)
        state_t_seq = torch.cat((sat_t_seq, pres_t_seq), dim=1)

        yobs_t_seq = torch.swapaxes(yobs[test_case, ...], 1, 2).to(device)

        sat_seq_true = sat[test_case, ...]
        pres_seq_true = pres[test_case, ...]

        perm = self._load_permeability()

        t_steps = np.arange(0, 200, 200 // num_tstep)
        dt = 10
        t_steps1 = (t_steps + dt).astype(int)
        indt_del = t_steps1 - t_steps
        indt_del = indt_del / max(indt_del)

        test_data = {
            'state_t_seq': state_t_seq,
            'bhp_seq': bhp_seq,
            'yobs_t_seq': yobs_t_seq,
            'sat_seq_true': sat_seq_true,
            'pres_seq_true': pres_seq_true,
            'test_case': test_case,
            'num_tstep': num_tstep,
            'num_case': len(test_case),
            't_steps': t_steps,
            'indt_del': indt_del,
            'dt_val': dt,
        }
        return test_data, perm

    def _load_permeability(self):
        cfg = self.cfg
        perm_data = scio.loadmat(cfg.data_dir + cfg.perm_file)
        m_full = perm_data['TRUE_PERM'].astype(np.float32)
        m = m_full[:cfg.Nx, 120:cfg.Ny + 120]
        m = torch.tensor(m, dtype=torch.float32).reshape(1, cfg.Nx, cfg.Ny)
        return m

    def _prepare_data(self):
        cfg = self.cfg
        Ksteps = cfg.nsteps

        hf = h5py.File(cfg.data_dir + cfg.state_file, 'r')
        mole = np.array(hf.get('Mole_frac_norm_slt')).transpose((3, 2, 1, 0))
        sat = np.array(hf.get('Sg_norm_slt')).transpose((3, 2, 1, 0))
        pres = np.array(hf.get('Psim_norm_slt')).transpose((3, 2, 1, 0))
        hf.close()
        n_sample, steps_slt, Nx, Ny = mole.shape

        hf = h5py.File(cfg.data_dir + cfg.ctrl_file, 'r')
        bhp0 = np.array(hf.get('Pwf_norm_slt')).transpose((2, 1, 0))
        rate0 = np.array(hf.get('Qinj_norm_slt')).transpose((2, 1, 0))
        hf.close()
        bhp = np.concatenate((bhp0, rate0), axis=1)

        hf = h5py.File(cfg.data_dir + cfg.yobs_file, 'r')
        if cfg.cond == 'RC':
            Qrate_w = np.array(hf.get('Qpro_w_RC_norm_slt')).transpose((2, 1, 0))
            Qrate_g = np.array(hf.get('Qpro_g_RC_norm_slt')).transpose((2, 1, 0))
        else:
            Qrate_w = np.array(hf.get('Qpro_w_norm_slt')).transpose((2, 1, 0))
            Qrate_g = np.array(hf.get('Qpro_g_norm_slt')).transpose((2, 1, 0))
        BHP_inj = np.array(hf.get('BHPinj_norm_slt')).transpose((2, 1, 0))
        hf.close()
        yobs = np.concatenate((Qrate_w, Qrate_g, BHP_inj), axis=1)

        n_sample, num_well, steps_ctrl = bhp.shape
        _, num_prod, _ = bhp0.shape
        _, num_inj, _ = rate0.shape

        Mole_slt, SAT_slt, PRES_slt, BHP_slt, Yobs_slt = [], [], [], [], []

        indt = np.array(range(0, steps_slt - (Ksteps - 1)))

        for k in range(Ksteps):
            indt_k = indt + k
            mole_t_slt = mole[:, indt_k, :, :]
            sat_t_slt = sat[:, indt_k, :, :]
            pres_t_slt = pres[:, indt_k, :, :]
            num_t_slt = sat_t_slt.shape[1]

            if k < Ksteps - 1:
                bhp_t_slt = np.swapaxes(bhp[:, :, indt_k], 1, 2)
                yobs_t_slt = np.swapaxes(yobs[:, :, indt_k], 1, 2)

            Mole_slt.append(mole_t_slt)
            SAT_slt.append(sat_t_slt)
            PRES_slt.append(pres_t_slt)
            if k < Ksteps - 1:
                BHP_slt.append(bhp_t_slt)
                Yobs_slt.append(yobs_t_slt)

        return (Mole_slt, SAT_slt, PRES_slt, BHP_slt, Yobs_slt,
                num_t_slt, Nx, Ny, num_well, num_prod, num_inj)

    def _train_split_data(self, Mole_slt, SAT_slt, PRES_slt, BHP_slt,
                          Yobs_slt, num_t_slt, Nx, Ny, num_well,
                          num_prod, num_inj):
        cfg = self.cfg
        device = self.device

        num_all = Mole_slt[0].shape[0]
        split_ratio = int(num_all / 100)
        num_run_per_case = 75
        num_run_eval = 100 - num_run_per_case

        num_train = num_run_per_case * split_ratio * num_t_slt
        shuffle_ind_train = np.random.default_rng(seed=cfg.seed).permutation(num_train)
        num_eval = num_run_eval * split_ratio * num_t_slt
        shuffle_ind_eval = np.random.default_rng(seed=cfg.seed).permutation(num_eval)

        STATE_train, BHP_train, Yobs_train = [], [], []
        STATE_eval, BHP_eval, Yobs_eval = [], [], []

        for i_step in range(len(SAT_slt)):
            mole_t_train = np.zeros((num_run_per_case * split_ratio, num_t_slt, Nx, Ny))
            pres_t_train = np.zeros((num_run_per_case * split_ratio, num_t_slt, Nx, Ny))
            bhp_t_train = np.zeros((num_run_per_case * split_ratio, num_t_slt, num_well))
            yobs_t_train = np.zeros((num_run_per_case * split_ratio, num_t_slt,
                                     2 * num_prod + num_inj))

            mole_t_eval = np.zeros((num_run_eval * split_ratio, num_t_slt, Nx, Ny))
            pres_t_eval = np.zeros((num_run_eval * split_ratio, num_t_slt, Nx, Ny))
            bhp_t_eval = np.zeros((num_run_eval * split_ratio, num_t_slt, num_well))
            yobs_t_eval = np.zeros((num_run_eval * split_ratio, num_t_slt,
                                    2 * num_prod + num_inj))

            for k in range(split_ratio):
                ind0 = k * num_run_per_case
                mole_t_train[ind0:ind0 + num_run_per_case, ...] = \
                    Mole_slt[i_step][k * 100:k * 100 + num_run_per_case, ...]
                pres_t_train[ind0:ind0 + num_run_per_case, ...] = \
                    PRES_slt[i_step][k * 100:k * 100 + num_run_per_case, ...]
                if i_step < len(SAT_slt) - 1:
                    bhp_t_train[ind0:ind0 + num_run_per_case, ...] = \
                        BHP_slt[i_step][k * 100:k * 100 + num_run_per_case, ...]
                    yobs_t_train[ind0:ind0 + num_run_per_case, ...] = \
                        Yobs_slt[i_step][k * 100:k * 100 + num_run_per_case, ...]

                ind1 = k * num_run_eval
                mole_t_eval[ind1:ind1 + num_run_eval, ...] = \
                    Mole_slt[i_step][k * 100 + num_run_per_case:k * 100 + 100, ...]
                pres_t_eval[ind1:ind1 + num_run_eval, ...] = \
                    PRES_slt[i_step][k * 100 + num_run_per_case:k * 100 + 100, ...]
                if i_step < len(SAT_slt) - 1:
                    bhp_t_eval[ind1:ind1 + num_run_eval, ...] = \
                        BHP_slt[i_step][k * 100 + num_run_per_case:k * 100 + 100, ...]
                    yobs_t_eval[ind1:ind1 + num_run_eval, ...] = \
                        Yobs_slt[i_step][k * 100 + num_run_per_case:k * 100 + 100, ...]

            Mole_t_train = mole_t_train.reshape(
                (num_run_per_case * split_ratio * num_t_slt, 1, Nx, Ny))
            PRES_t_train = pres_t_train.reshape(
                (num_run_per_case * split_ratio * num_t_slt, 1, Nx, Ny))
            Mole_t_eval = mole_t_eval.reshape(
                (num_run_eval * split_ratio * num_t_slt, 1, Nx, Ny))
            PRES_t_eval = pres_t_eval.reshape(
                (num_run_eval * split_ratio * num_t_slt, 1, Nx, Ny))

            STATE_t_train = torch.tensor(
                np.concatenate((Mole_t_train, PRES_t_train), axis=1),
                dtype=torch.float32).to(device)
            STATE_t_eval = torch.tensor(
                np.concatenate((Mole_t_eval, PRES_t_eval), axis=1),
                dtype=torch.float32).to(device)

            STATE_t_train = STATE_t_train[shuffle_ind_train, ...]
            STATE_t_eval = STATE_t_eval[shuffle_ind_eval, ...]

            if i_step < len(SAT_slt) - 1:
                BHP_t_train = torch.tensor(
                    bhp_t_train.reshape(
                        (num_run_per_case * split_ratio * num_t_slt, num_well)),
                    dtype=torch.float32).to(device)
                Yobs_t_train = torch.tensor(
                    yobs_t_train.reshape(
                        (num_run_per_case * split_ratio * num_t_slt,
                         2 * num_prod + num_inj)),
                    dtype=torch.float32).to(device)
                BHP_t_eval = torch.tensor(
                    bhp_t_eval.reshape(
                        (num_run_eval * split_ratio * num_t_slt, num_well)),
                    dtype=torch.float32).to(device)
                Yobs_t_eval = torch.tensor(
                    yobs_t_eval.reshape(
                        (num_run_eval * split_ratio * num_t_slt,
                         2 * num_prod + num_inj)),
                    dtype=torch.float32).to(device)

                BHP_t_train = BHP_t_train[shuffle_ind_train, ...]
                Yobs_t_train = Yobs_t_train[shuffle_ind_train, ...]
                BHP_t_eval = BHP_t_eval[shuffle_ind_eval, ...]
                Yobs_t_eval = Yobs_t_eval[shuffle_ind_eval, ...]

            STATE_train.append(STATE_t_train)
            STATE_eval.append(STATE_t_eval)
            if i_step < len(SAT_slt) - 1:
                BHP_train.append(BHP_t_train)
                BHP_eval.append(BHP_t_eval)
                Yobs_train.append(Yobs_t_train)
                Yobs_eval.append(Yobs_t_eval)

        return STATE_train, BHP_train, Yobs_train, STATE_eval, BHP_eval, Yobs_eval
