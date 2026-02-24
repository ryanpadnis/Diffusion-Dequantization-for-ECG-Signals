"""
Robust diffusion trainer with Accelerate, checkpointing, and validation.
Handles everything: train/val splits, logging, progress bars, AWS compatibility.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import psutil
import os


class DiffusionTrainer:
    """Complete trainer for conditional diffusion models.
    
    Features:
    - Automatic train/val split
    - Accelerate for distributed training
    - Checkpointing (resume from interruptions)
    - Progress bars with tqdm
    - Validation during training
    - Logging to tensorboard
    - AWS-ready (spot instances, multi-GPU)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        checkpoints_dir: Optional[Path] = None,
        samples_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: Optional[str] = None,
        validation_split: float = 0.1,
        save_every_n_epochs: int = 1,
        validate_every_n_epochs: int = 1,
        log_advanced_metrics: bool = True,
        advanced_metrics_every_n_steps: int = 50,
        energy_curve_every_n_steps: int = 200,
        ema_config: Optional[Dict[str, Any]] = None,
        total_training_steps: Optional[int] = None,
        loss_components_every_n_steps: int = 50,
        early_stop_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: ConditionalDiffuser model
            optimizer: PyTorch optimizer
            lr_scheduler: Learning rate scheduler (optional)
            checkpoints_dir: Where to save checkpoints
            samples_dir: Where to save generated samples
            logs_dir: Where to save logs
            gradient_accumulation_steps: Accumulate gradients for larger effective batch
            mixed_precision: None, 'fp16', or 'bf16' (requires GPU)
            validation_split: Fraction of data for validation (0.0-1.0)
            save_every_n_epochs: Save checkpoint every N epochs
            validate_every_n_epochs: Run validation every N epochs
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.validation_split = validation_split
        self.save_every_n_epochs = save_every_n_epochs
        self.validate_every_n_epochs = validate_every_n_epochs
        
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else Path("checkpoints")
        self.samples_dir = Path(samples_dir) if samples_dir else Path("samples")
        self.logs_dir = Path(logs_dir) if logs_dir else Path("logs")
        
        for dir_path in [self.checkpoints_dir, self.samples_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="tensorboard",
            project_dir=str(self.logs_dir),
        )
        
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        self.log_advanced_metrics = bool(log_advanced_metrics)
        self.advanced_metrics_every_n_steps = int(advanced_metrics_every_n_steps)
        self.energy_curve_every_n_steps = int(energy_curve_every_n_steps)

        self.loss_components_every_n_steps = int(loss_components_every_n_steps)
        self.total_training_steps = int(total_training_steps) if total_training_steps is not None else None

        self.ema_config = ema_config if isinstance(ema_config, dict) else {}
        self._ema = None

        # Optional early stopping on moving-average validation loss.
        # Controlled via early_stop_config={enabled, window_epochs/window, patience, min_delta}.
        self.early_stop_config = early_stop_config if isinstance(early_stop_config, dict) else {}
        
        print(f"[Trainer] Initialized on device: {self.accelerator.device}")
        print(f"  - Checkpoints: {self.checkpoints_dir}")
        print(f"  - Samples: {self.samples_dir}")
        print(f"  - Logs: {self.logs_dir}")
        print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  - Validation split: {validation_split:.1%}")


    @staticmethod
    def _batch_stats(x: torch.Tensor, prefix: str) -> Dict[str, float]:
        xf = x.detach().to(torch.float32)
        return {
            f"{prefix}/mean": float(xf.mean().item()),
            f"{prefix}/std": float(xf.std(unbiased=False).item()),
            f"{prefix}/min": float(xf.min().item()),
            f"{prefix}/max": float(xf.max().item()),
            f"{prefix}/mean_abs": float(xf.abs().mean().item()),
            f"{prefix}/mean_sq": float((xf * xf).mean().item()),
            f"{prefix}/rms": float(torch.sqrt((xf * xf).mean() + 1e-12).item()),
        }

    def _get_tb_writer(self):
        for tracker in getattr(self.accelerator, "trackers", []):
            if getattr(tracker, "name", None) == "tensorboard" and hasattr(tracker, "writer"):
                return tracker.writer
        return None

    def _log_energy_curve(self, cond_batch: torch.Tensor, real_batch: torch.Tensor, step: int) -> None:
        writer = self._get_tb_writer()
        if writer is None:
            return

        try:
            import matplotlib.pyplot as plt

            # Both are [B, 1, H, W]; make a single curve over W by averaging over B,1,H.
            cond_energy_t = (cond_batch.detach().to(torch.float32) ** 2).mean(dim=(0, 1, 2)).cpu().numpy()
            real_energy_t = (real_batch.detach().to(torch.float32) ** 2).mean(dim=(0, 1, 2)).cpu().numpy()

            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            ax.plot(real_energy_t, label="real_energy_t", linewidth=1.0)
            ax.plot(cond_energy_t, label="cond_energy_t", linewidth=1.0, alpha=0.8)
            ax.set_title("Energy over time frames (mean over batch/freq)")
            ax.set_xlabel("time frame")
            ax.set_ylabel("mean square")
            ax.legend(loc="upper right")
            fig.tight_layout()

            writer.add_figure("train/energy_over_time", fig, global_step=step)
            plt.close(fig)
        except Exception:
            return
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        stats = {}
        
        # CPU memory
        process = psutil.Process(os.getpid())
        stats['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        # GPU memory (CUDA)
        if torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        # MPS memory (Mac - no direct access; expose a numeric flag)
        if self.accelerator.device.type == 'mps':
            stats['mps_active'] = 1.0
        
        return stats
    
    def prepare_dataloaders(
        self,
        cond_data: torch.Tensor,
        real_data: torch.Tensor,
        batch_size: int,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """Split data and create train/val dataloaders."""
        dataset = TensorDataset(cond_data, real_data)
        
        val_size = int(len(dataset) * self.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Disable pin_memory if data is already on CUDA (avoids "cannot pin cuda tensor" error)
        use_pin_memory = cond_data.device.type == 'cpu'
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory
        ) if val_size > 0 else None
        
        print(f"[Trainer] Dataset split:")
        print(f"  - Train: {train_size} samples ({train_size/len(dataset):.1%})")
        print(f"  - Val: {val_size} samples ({val_size/len(dataset):.1%})")
        
        return train_loader, val_loader
    
    def fit(
        self,
        cond_data: torch.Tensor,
        real_data: torch.Tensor,
        epochs: int,
        batch_size: int = 16,
        num_workers: int = 0,
        resume_from: Optional[Path] = None
    ):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"[Trainer] Starting training for {epochs} epochs")
        print(f"{'='*60}\n")
        
        train_loader, val_loader = self.prepare_dataloaders(
            cond_data, real_data, batch_size, num_workers
        )
        
        self.model, self.optimizer, train_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader
        )

        # Initialize EMA on the prepared (wrapped) model.
        if self.ema_config.get('enabled'):
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
                decay = float(self.ema_config.get('decay', 0.999))
                self._ema = _ModelEMA(unwrapped, decay=decay)
                if self.accelerator.is_main_process:
                    print(f"[Trainer] EMA enabled (decay={decay})")
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"[Trainer] EMA init failed; continuing without EMA: {e}")
                self._ema = None
        
        if val_loader:
            val_loader = self.accelerator.prepare(val_loader)
        
        if self.lr_scheduler:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        
        if resume_from:
            self._load_checkpoint(resume_from)
        
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("diffusion_training")
            print(f"[Trainer] Tensorboard logs: {self.logs_dir / 'diffusion_training'}")

        # Moving-average early stop state.
        val_ma_history: list[float] = []
        best_val_ma: float | None = None
        bad_windows: int = 0
        es_enabled = bool(self.early_stop_config.get('enabled', False))
        es_window = int(self.early_stop_config.get('window_epochs', self.early_stop_config.get('window', 5)) or 5)
        es_window = max(1, es_window)
        es_patience = int(self.early_stop_config.get('patience', 1) or 1)
        es_patience = max(1, es_patience)
        es_min_delta = float(self.early_stop_config.get('min_delta', 0.0) or 0.0)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch(train_loader, epoch, epochs)
            
            val_loss = None
            if val_loader and (epoch + 1) % self.validate_every_n_epochs == 0:
                val_loss = self.validate_epoch(val_loader, epoch)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.accelerator.is_main_process:
                        self._save_checkpoint(
                            self.checkpoints_dir / "best_model.pt",
                            is_best=True
                        )

                # Early stopping (moving-average validation loss)
                if es_enabled and (val_loss is not None):
                    try:
                        val_ma_history.append(float(val_loss))
                        if len(val_ma_history) >= es_window:
                            ma = sum(val_ma_history[-es_window:]) / float(es_window)
                            if best_val_ma is None or (ma <= (best_val_ma - es_min_delta)):
                                best_val_ma = float(ma)
                                bad_windows = 0
                            else:
                                bad_windows += 1

                            if self.accelerator.is_main_process:
                                print(
                                    f"[early_stop] epoch={epoch+1} val_ma({es_window})={ma:.6f} "
                                    f"best_ma={best_val_ma:.6f} bad_windows={bad_windows}/{es_patience}"
                                )

                            if bad_windows >= es_patience:
                                if self.accelerator.is_main_process:
                                    print(
                                        f"[early_stop] Stopping early: val_ma({es_window}) did not improve "
                                        f"by >= {es_min_delta:g} for {bad_windows} window(s)."
                                    )
                                break
                    except Exception as e:
                        if self.accelerator.is_main_process:
                            print(f"[early_stop] Warning: failed to compute early stop criterion: {e}")
            
            if self.accelerator.is_main_process and (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint(
                    self.checkpoints_dir / f"checkpoint_epoch_{epoch+1}.pt"
                )
            
            if self.accelerator.is_main_process:
                print(f"\n[Epoch {epoch+1}/{epochs}] Summary:")
                print(f"  Train Loss: {train_loss:.6f}")
                if val_loss is not None:
                    print(f"  Val Loss: {val_loss:.6f}")
                    print(f"  Best Val Loss: {self.best_val_loss:.6f}")
                print()
        
        print(f"\n{'='*60}")
        print(f"[Trainer] Training complete!")
        print(f"  - Best validation loss: {self.best_val_loss:.6f}")
        print(f"  - Total steps: {self.global_step}")
        print(f"{'='*60}\n")
        
        if self.accelerator.is_main_process:
            # Flush tensorboard logs before ending
            writer = self._get_tb_writer()
            if writer:
                writer.flush()
                print(f"[Trainer] Flushed tensorboard logs to: {self.logs_dir / 'diffusion_training'}")
            self.accelerator.end_training()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Accumulate batch-wise statistics so we can log once per epoch.
        stats_sums: Dict[str, float] = {}
        last_cond_batch: Optional[torch.Tensor] = None
        last_real_batch: Optional[torch.Tensor] = None
        last_lr: Optional[float] = None
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, (cond_batch, real_batch) in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                # Provide model the current training progress for dynamic loss schedules.
                try:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    if hasattr(unwrapped, 'set_training_progress'):
                        unwrapped.set_training_progress(step=int(self.global_step), total_steps=self.total_training_steps)
                except Exception:
                    pass

                loss = self.model(real_batch, cond_batch)
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # EMA update on optimizer step.
                if self._ema is not None and self.accelerator.sync_gradients:
                    try:
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        # Optional linear schedule for EMA decay.
                        ema_decay = float(self.ema_config.get('decay', 0.999))
                        sched = self.ema_config.get('schedule') if isinstance(self.ema_config.get('schedule'), dict) else None
                        if sched and isinstance(self.global_step, int):
                            if str(sched.get('type', 'linear')).lower() == 'linear':
                                start = float(sched.get('decay_start', ema_decay))
                                end = float(sched.get('decay_end', ema_decay))
                                start_step = int(sched.get('start_step', 0))
                                end_step = sched.get('end_step', None)
                                if end_step is None:
                                    end_step = int(self.total_training_steps) if self.total_training_steps is not None else start_step
                                end_step = int(end_step)
                                if end_step > start_step:
                                    if self.global_step <= start_step:
                                        ema_decay = start
                                    elif self.global_step >= end_step:
                                        ema_decay = end
                                    else:
                                        t = (float(self.global_step) - float(start_step)) / (float(end_step) - float(start_step))
                                        ema_decay = float(start + t * (end - start))
                        self._ema.update(unwrapped, decay=float(ema_decay))
                    except Exception:
                        pass
            
            total_loss += loss.detach().item()
            num_batches += 1
            
            current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
            last_lr = float(current_lr)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'step': self.global_step
            })

            # Print/log low/high bucket loss components if the model provides them.
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
                comps = getattr(unwrapped, 'last_loss_components', None)
                if isinstance(comps, dict) and self.accelerator.is_local_main_process:
                    low = comps.get('loss_low_bucket')
                    high = comps.get('loss_high_bucket')
                    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'low': f'{float(low):.4f}',
                            'high': f'{float(high):.4f}',
                            'lr': f'{current_lr:.2e}',
                            'step': self.global_step,
                        })

                if isinstance(comps, dict) and self.accelerator.is_main_process:
                    every = max(1, int(self.loss_components_every_n_steps))
                    if self.accelerator.sync_gradients and (int(self.global_step) % every == 0):
                        low = comps.get('loss_low_bucket')
                        high = comps.get('loss_high_bucket')
                        cutoff = comps.get('bucket_cutoff_bins')
                        wlow = comps.get('mean_weight_low_bucket')
                        whigh = comps.get('mean_weight_high_bucket')
                        msg = f"[loss buckets] step={self.global_step} cutoff_bins={cutoff} low={low:.6f} high={high:.6f}"
                        if isinstance(wlow, (int, float)) and isinstance(whigh, (int, float)):
                            msg += f" w_low~{float(wlow):.3f} w_high~{float(whigh):.3f}"
                        print(msg)
            except Exception:
                pass

            if self.log_advanced_metrics:
                # Aggregate cheap scalar stats once per batch; logged at end of epoch.
                batch_stats = {}
                batch_stats.update(self._batch_stats(cond_batch, "train/cond"))
                batch_stats.update(self._batch_stats(real_batch, "train/real"))
                for k, v in batch_stats.items():
                    stats_sums[k] = stats_sums.get(k, 0.0) + float(v)

                # Keep last batch around for the epoch-level energy curve figure.
                last_cond_batch = cond_batch
                last_real_batch = real_batch
            
            if self.accelerator.sync_gradients:
                self.global_step += 1
        
        avg_loss = total_loss / num_batches

        # Log once per epoch (TensorBoard step = epoch index).
        if self.accelerator.is_main_process:
            epoch_step = int(epoch + 1)
            memory_stats = self._get_memory_stats()

            metrics: Dict[str, float] = {
                "train/loss": float(avg_loss),
                "train/learning_rate": float(last_lr) if last_lr is not None else 0.0,
                "train/epoch": float(epoch_step),
                **{
                    f"memory/{k}": float(v)
                    for k, v in memory_stats.items()
                    if isinstance(v, (int, float, bool))
                },
            }

            if self.log_advanced_metrics and num_batches > 0:
                for k, v in stats_sums.items():
                    metrics[k] = float(v) / float(num_batches)

            self.accelerator.log(metrics, step=epoch_step)

            if self.log_advanced_metrics and last_cond_batch is not None and last_real_batch is not None:
                self._log_energy_curve(last_cond_batch, last_real_batch, step=epoch_step)
            
            # Flush tensorboard logs after each epoch
            writer = self._get_tb_writer()
            if writer:
                writer.flush()

        return avg_loss
    
    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1} [Val]",
            disable=not self.accelerator.is_local_main_process
        )
        
        for cond_batch, real_batch in progress_bar:
            loss = self.model(real_batch, cond_batch)
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        
        if self.accelerator.is_main_process:
            epoch_step = int(epoch + 1)
            memory_stats = self._get_memory_stats()
            metrics = {
                "val/loss": avg_loss,
                "val/epoch": float(epoch_step),
                **{f"memory/{k}": v for k, v in memory_stats.items()},
            }
            self.accelerator.log(metrics, step=epoch_step)
        
        return avg_loss
    
    def _save_checkpoint(self, path: Path, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }

        if self._ema is not None:
            try:
                checkpoint['ema_state_dict'] = self._ema.state_dict()
            except Exception:
                pass
        
        if self.lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"[Trainer] {'Best model' if is_best else 'Checkpoint'} saved: {path}")
    
    def _load_checkpoint(self, path: Path):
        """Load checkpoint and resume training."""
        print(f"[Trainer] Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.accelerator.device)
        
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self._ema is not None and isinstance(checkpoint.get('ema_state_dict'), dict):
            try:
                self._ema.load_state_dict(checkpoint['ema_state_dict'])
                if self.accelerator.is_main_process:
                    print("[Trainer] Restored EMA state")
            except Exception:
                pass
        
        print(f"[Trainer] Resumed from epoch {self.current_epoch}, step {self.global_step}")


class _ModelEMA:
    """Lightweight EMA for model parameters (CPU/GPU agnostic)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name] = p.detach().clone()

    def update(self, model: nn.Module, decay: float | None = None) -> None:
        d = float(self.decay if decay is None else decay)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name not in self.shadow:
                    continue
                if not p.requires_grad:
                    continue
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def state_dict(self) -> Dict[str, Any]:
        return {
            'decay': float(self.decay),
            'shadow': {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state.get('decay', self.decay))
        shadow = state.get('shadow', {})
        if isinstance(shadow, dict):
            self.shadow = {k: v.detach().clone() for k, v in shadow.items() if torch.is_tensor(v)}
