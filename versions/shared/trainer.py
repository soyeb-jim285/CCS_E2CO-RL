"""Generic training loop for all versions."""

import os
import csv
import glob
import time
import torch
import torch.optim as optim


class BaseTrainer:
    """Handles training, evaluation, checkpointing, and resume.

    Works with any model/loss_fn that follows the interface contract:
    - model.forward(inputs) -> pred_dict
    - model.predict(xt, ut, yt, dt) -> (xt_next, yt_next)
    - loss_fn.forward(pred_dict) -> (total_loss, losses_stack)
    - loss_fn.losses_to_dict(stack) -> dict
    - loss_fn.num_loss_terms -> int
    - loss_fn.get_adaptive_state() -> dict or None
    - loss_fn.load_adaptive_state(state)
    """

    def __init__(self, model, loss_fn, cfg, device):
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device

        # Optimizer
        param_groups = [{'params': model.parameters(), 'lr': cfg.learning_rate}]
        if hasattr(loss_fn, 'adaptive') and loss_fn.adaptive is not None:
            param_groups.append({
                'params': loss_fn.adaptive.parameters(),
                'lr': cfg.adaptive_lr,
            })
        self.optimizer = optim.Adam(param_groups)

        # AMP scaler
        self.use_amp = cfg.use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # torch.compile
        if cfg.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile failed, continuing without it: {e}")

        self.start_epoch = 0
        self.best_loss = 1e9
        self.loss_history = []

        # TensorBoard
        self.writer = None
        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=cfg.log_dir)
            except ImportError:
                pass

        self.csv_path = os.path.join(cfg.log_dir, "loss_history.csv")

    def try_resume(self):
        if not self.cfg.resume:
            return
        latest = self._find_latest_checkpoint()
        if latest is None:
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Resuming from {latest}")
        ckpt = torch.load(latest, map_location='cpu', weights_only=False)

        # Handle compiled model
        model = self.model
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt['model_state'])

        if 'optimizer_state' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.start_epoch = ckpt.get('epoch', 0)
        self.best_loss = ckpt.get('best_loss', 1e9)

        adaptive_state = ckpt.get('adaptive_weights', None)
        if adaptive_state is not None:
            self.loss_fn.load_adaptive_state(adaptive_state)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r') as f:
                self.loss_history = list(csv.DictReader(f))
            print(f"  Loaded {len(self.loss_history)} epochs of history.")

        print(f"  Resuming at epoch {self.start_epoch + 1}, best_loss={self.best_loss:.6f}")

    def train(self, train_data, eval_data):
        cfg = self.cfg
        STATE_train = train_data['STATE']
        BHP_train = train_data['BHP']
        Yobs_train = train_data['Yobs']
        dt_train = train_data['dt']
        num_train = train_data['num_train']
        batch_size = cfg.batch_size
        num_batch = num_train // batch_size
        grad_accum = cfg.gradient_accumulation_steps
        eval_every = cfg.eval_every
        num_loss_terms = self.loss_fn.num_loss_terms

        amp_dtype = torch.float16 if self.use_amp else torch.float32

        print(f"Training: {num_train} samples, {num_batch} batches/epoch, "
              f"epochs {self.start_epoch + 1} to {cfg.epochs}")
        print(f"  batch_size={batch_size}, grad_accum={grad_accum}, "
              f"effective_batch={batch_size * grad_accum}")
        print(f"  AMP={self.use_amp}, eval_every={eval_every} epochs")

        for e in range(self.start_epoch, cfg.epochs):
            epoch_start = time.time()
            self.model.train()

            epoch_loss_sum = torch.zeros(1, device=self.device)
            epoch_subloss_sum = torch.zeros(num_loss_terms, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)

            for ib in range(num_batch):
                ind0 = ib * batch_size

                X_batch = [state[ind0:ind0 + batch_size] for state in STATE_train]
                U_batch = [bhp[ind0:ind0 + batch_size] for bhp in BHP_train]
                Y_batch = [yobs[ind0:ind0 + batch_size] for yobs in Yobs_train]
                dt_batch = dt_train[ind0:ind0 + batch_size]

                inputs = (X_batch, U_batch, Y_batch, dt_batch)

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
                    pred = self.model(inputs)
                    total_loss, losses_stack = self.loss_fn(pred)

                scaled_loss = total_loss / grad_accum
                self.scaler.scale(scaled_loss).backward()

                if (ib + 1) % grad_accum == 0 or (ib + 1) == num_batch:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss_sum += total_loss.detach()
                epoch_subloss_sum += losses_stack.detach()

                if ib % 500 == 0:
                    print(f"Epoch {e + 1}/{cfg.epochs}, Batch {ib + 1}/{num_batch}, "
                          f"Loss {total_loss.item():.6f}")

            avg_loss = (epoch_loss_sum / num_batch).item()
            avg_sublosses = (epoch_subloss_sum / num_batch).cpu()

            eval_loss = None
            if (e + 1) % eval_every == 0 or (e + 1) == cfg.epochs:
                eval_loss = self._evaluate(eval_data)

            elapsed = time.time() - epoch_start
            print("=" * 60)
            eval_str = f", Eval loss {eval_loss:.6f}" if eval_loss is not None else ""
            print(f"Epoch {e + 1}/{cfg.epochs}, "
                  f"Train loss {avg_loss:.6f}{eval_str}, "
                  f"Time {elapsed:.1f}s")
            print("=" * 60)

            subloss_dict = self.loss_fn.losses_to_dict(avg_sublosses)
            log_entry = {'epoch': e + 1, 'train_loss': avg_loss}
            if eval_loss is not None:
                log_entry['eval_loss'] = eval_loss
            log_entry.update({f"train_{k}": v for k, v in subloss_dict.items()})

            if hasattr(self.loss_fn, 'adaptive') and self.loss_fn.adaptive is not None:
                sigmas = self.loss_fn.adaptive.get_sigmas()
                for i, name in enumerate(self.loss_fn._lambda_keys):
                    if i < len(sigmas):
                        log_entry[f"sigma_{name}"] = float(sigmas[i])

            self.loss_history.append(log_entry)
            self._write_csv()

            if self.writer is not None:
                self.writer.add_scalar('loss/train_total', avg_loss, e + 1)
                if eval_loss is not None:
                    self.writer.add_scalar('loss/eval_total', eval_loss, e + 1)
                for k, v in subloss_dict.items():
                    self.writer.add_scalar(f'loss/train_{k}', v, e + 1)

            check_loss = eval_loss if eval_loss is not None else avg_loss
            if check_loss < self.best_loss:
                self.best_loss = check_loss
                self._save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, "best_model.pt"),
                    e + 1)
                print(f"  New best model saved (loss={check_loss:.6f})")

            if (e + 1) % cfg.checkpoint_every == 0:
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir, f"ckpt_epoch_{e + 1:04d}.pt")
                self._save_checkpoint(ckpt_path, e + 1)
                print(f"  Checkpoint saved: {ckpt_path}")

        if self.writer is not None:
            self.writer.close()

        print("Training complete.")

    def _evaluate(self, eval_data):
        self.model.eval()
        cfg = self.cfg
        eval_batch_size = cfg.eval_batch_size
        amp_dtype = torch.float16 if self.use_amp else torch.float32

        STATE_eval = eval_data['STATE']
        BHP_eval = eval_data['BHP']
        Yobs_eval = eval_data['Yobs']
        dt_eval = eval_data['dt']
        num_eval = eval_data['num_eval']

        total_loss_sum = 0.0
        num_eval_batches = max(1, num_eval // eval_batch_size)

        with torch.inference_mode():
            for ib in range(num_eval_batches):
                ind0 = ib * eval_batch_size
                ind1 = min(ind0 + eval_batch_size, num_eval)

                X_batch = [state[ind0:ind1] for state in STATE_eval]
                U_batch = [bhp[ind0:ind1] for bhp in BHP_eval]
                Y_batch = [yobs[ind0:ind1] for yobs in Yobs_eval]
                dt_batch = dt_eval[ind0:ind1]

                inputs = (X_batch, U_batch, Y_batch, dt_batch)

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
                    pred = self.model(inputs)
                    loss, _ = self.loss_fn(pred)

                total_loss_sum += loss.item() * (ind1 - ind0)

        return total_loss_sum / num_eval

    def _save_checkpoint(self, path, epoch):
        model = self.model
        if hasattr(model, '_orig_mod'):
            model_state = model._orig_mod.state_dict()
        else:
            model_state = model.state_dict()

        state = {
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_loss': self.best_loss,
        }
        adaptive_state = self.loss_fn.get_adaptive_state()
        if adaptive_state is not None:
            state['adaptive_weights'] = adaptive_state
        torch.save(state, path)

    def _find_latest_checkpoint(self):
        pattern = os.path.join(self.cfg.checkpoint_dir, "ckpt_epoch_*.pt")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        return files[-1]

    def _write_csv(self):
        if not self.loss_history:
            return
        latest = self.loss_history[-1]
        all_keys = set()
        for entry in self.loss_history:
            all_keys.update(entry.keys())
        all_keys = sorted(all_keys)
        if 'epoch' in all_keys:
            all_keys.remove('epoch')
            all_keys = ['epoch'] + all_keys

        file_exists = os.path.exists(self.csv_path) and len(self.loss_history) > 1
        mode = 'a' if file_exists else 'w'
        with open(self.csv_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
                for entry in self.loss_history:
                    writer.writerow(entry)
            else:
                writer.writerow(latest)
