"""Training loop — optimized for GPU throughput."""

import os
import csv
import time
import torch
import torch.optim as optim

from .model import PINNE2CO
from .physics_loss import PINNLoss


class PINNTrainer:
    """Handles training, evaluation, checkpointing, and resume."""

    def __init__(self, model, loss_fn, cfg, device):
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device

        # Optimizer
        param_groups = [{'params': model.parameters(), 'lr': cfg.learning_rate}]
        if cfg.use_adaptive_weights and loss_fn.adaptive is not None:
            param_groups.append({
                'params': loss_fn.adaptive.parameters(),
                'lr': cfg.adaptive_lr,
            })
        self.optimizer = optim.Adam(param_groups)

        # AMP scaler for mixed precision
        self.use_amp = cfg.use_amp and device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # torch.compile (PyTorch 2.0+)
        if cfg.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile failed, continuing without it: {e}")

        self.start_epoch = 0
        self.best_loss = 1e9
        self.loss_history = []

        # TensorBoard (lazy init)
        self.writer = None
        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=cfg.log_dir)
            except ImportError:
                pass

        self.csv_path = os.path.join(cfg.log_dir, "loss_history.csv")

    def try_resume(self):
        """Attempt to resume from latest checkpoint."""
        if not self.cfg.resume:
            return
        latest = PINNE2CO.find_latest_checkpoint(self.cfg.checkpoint_dir)
        if latest is None:
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Resuming from {latest}")
        epoch, best_loss, adaptive_state = PINNE2CO.load_checkpoint(
            latest, self.model, self.optimizer)
        self.start_epoch = epoch
        self.best_loss = best_loss

        if adaptive_state is not None:
            self.loss_fn.load_adaptive_state(adaptive_state)

        # Move optimizer states to device
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
        """Full training loop — optimized for GPU throughput."""
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

        amp_dtype = torch.float16 if self.use_amp else torch.float32

        print(f"Training: {num_train} samples, {num_batch} batches/epoch, "
              f"epochs {self.start_epoch + 1} to {cfg.epochs}")
        print(f"  batch_size={batch_size}, grad_accum={grad_accum}, "
              f"effective_batch={batch_size * grad_accum}")
        print(f"  AMP={self.use_amp}, eval_every={eval_every} epochs")

        for e in range(self.start_epoch, cfg.epochs):
            epoch_start = time.time()
            self.model.train()

            # Accumulate losses on GPU — NO .item() calls during training
            epoch_loss_sum = torch.zeros(1, device=self.device)
            epoch_subloss_sum = torch.zeros(8, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)

            for ib in range(num_batch):
                ind0 = ib * batch_size

                X_batch = [state[ind0:ind0 + batch_size] for state in STATE_train]
                U_batch = [bhp[ind0:ind0 + batch_size] for bhp in BHP_train]
                Y_batch = [yobs[ind0:ind0 + batch_size] for yobs in Yobs_train]
                dt_batch = dt_train[ind0:ind0 + batch_size]

                inputs = (X_batch, U_batch, Y_batch, dt_batch)

                # Forward with AMP
                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=amp_dtype):
                    pred = self.model(inputs)
                    total_loss, losses_stack = self.loss_fn(pred)

                # Scale for gradient accumulation
                scaled_loss = total_loss / grad_accum

                # Backward with AMP
                self.scaler.scale(scaled_loss).backward()

                # Step every grad_accum batches or at end of epoch
                if (ib + 1) % grad_accum == 0 or (ib + 1) == num_batch:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                # Accumulate on GPU (no sync)
                epoch_loss_sum += total_loss.detach()
                epoch_subloss_sum += losses_stack.detach()

                # Print progress (syncs only occasionally)
                if ib % 500 == 0:
                    print(f"Epoch {e + 1}/{cfg.epochs}, Batch {ib + 1}/{num_batch}, "
                          f"Loss {total_loss.item():.6f}")

            # Now sync once per epoch
            avg_loss = (epoch_loss_sum / num_batch).item()
            avg_sublosses = (epoch_subloss_sum / num_batch).cpu()

            # Eval (not every epoch — expensive)
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

            # Log — convert to dict here (single sync point)
            subloss_dict = self.loss_fn.losses_to_dict(avg_sublosses)
            log_entry = {'epoch': e + 1, 'train_loss': avg_loss}
            if eval_loss is not None:
                log_entry['eval_loss'] = eval_loss
            log_entry.update({f"train_{k}": v for k, v in subloss_dict.items()})

            if cfg.use_adaptive_weights and self.loss_fn.adaptive is not None:
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

            # Checkpoint
            check_loss = eval_loss if eval_loss is not None else avg_loss
            if check_loss < self.best_loss:
                self.best_loss = check_loss
                best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
                self.model.save_checkpoint(
                    best_path, self.optimizer, e + 1, self.best_loss,
                    self.loss_fn.get_adaptive_state())
                print(f"  New best model saved (loss={check_loss:.6f})")

            if (e + 1) % cfg.checkpoint_every == 0:
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir, f"ckpt_epoch_{e + 1:04d}.pt")
                self.model.save_checkpoint(
                    ckpt_path, self.optimizer, e + 1, self.best_loss,
                    self.loss_fn.get_adaptive_state())
                print(f"  Checkpoint saved: {ckpt_path}")

        if self.writer is not None:
            self.writer.close()

        print("Training complete.")

    def _evaluate(self, eval_data):
        """Run evaluation in batches to avoid OOM, return total loss."""
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

    def _write_csv(self):
        """Append-mode CSV write (don't rewrite entire file)."""
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
                # Write all history if starting fresh
                for entry in self.loss_history:
                    writer.writerow(entry)
            else:
                writer.writerow(latest)
