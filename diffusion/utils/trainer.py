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
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
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
        
        if val_loader:
            val_loader = self.accelerator.prepare(val_loader)
        
        if self.lr_scheduler:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        
        if resume_from:
            self._load_checkpoint(resume_from)
        
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("diffusion_training")
        
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
                loss = self.model(real_batch, cond_batch)
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.detach().item()
            num_batches += 1
            
            current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
            last_lr = float(current_lr)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'step': self.global_step
            })

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
        
        print(f"[Trainer] Resumed from epoch {self.current_epoch}, step {self.global_step}")
