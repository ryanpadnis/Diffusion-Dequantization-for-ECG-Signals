""" 
Diffuser class following HuggingFace pattern with Ray Train and Accelerate support.
Based on: https://huggingface.co/docs/diffusers/en/tutorials/basic_training

Extensible design for adding:
- Different UNet architectures
- Various schedulers (DDPM, DDIM, etc.)
- Custom samplers
- Mixed precision training
- Distributed training (Ray Train + Accelerate)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from diffusion.utils.torch_utils import resolve_torch_dtype


class ConditionalDiffuser(nn.Module):
    """Extensible conditional diffusion model.
    
    Features:
    - Condition encoder: 4-bit → cross-attention features
    - Flexible UNet backbone (can swap architectures)
    - Multiple scheduler support (DDPM, DDIM)
    - Compatible with Accelerate and Ray Train
    - Easy to extend with new components
    
    Training: noisy_16bit + cond_4bit → predict_noise → loss
    Inference: noise + cond_4bit → denoise → generated_16bit
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        unet_type: Literal['conditional', 'custom'] = 'conditional',
        scheduler_type: Literal['ddpm', 'ddim'] = 'ddpm'
    ):
        super().__init__()
        self.config = config
        self.unet_type = unet_type
        self.scheduler_type = scheduler_type

        # Cache for optional frequency-weighted loss.
        self._loss_freq_weight_cache: torch.Tensor | None = None
        self._loss_freq_weight_cache_h: int | None = None

        # Training progress (optional; set by trainer for dynamic schedules).
        self._training_step: int | None = None
        self._training_total_steps: int | None = None

        # Introspection for debugging/logging.
        self.last_loss_components: Dict[str, float] | None = None
        
        # Build UNet backbone
        self.unet = self._build_unet(unet_type, config)
        
        # Build scheduler
        self.noise_scheduler = self._build_scheduler(scheduler_type, config)
        
        # Build condition encoder
        self.cond_encoder = self._build_condition_encoder(config)
    
    def _build_unet(self, unet_type: str, config: Dict[str, Any]) -> nn.Module:
        """Build UNet based on type (extensible for new architectures)."""
        if unet_type == 'conditional':
            return UNet2DConditionModel(
                sample_size=config['image_size'],
                in_channels=config['in_channels'],
                out_channels=config['out_channels'],
                layers_per_block=2,
                block_out_channels=(128, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=512,
            )
        else:
            raise ValueError(f"Unknown unet_type: {unet_type}")
    
    def _build_scheduler(self, scheduler_type: str, config: Dict[str, Any]):
        """Build scheduler (extensible for new schedulers)."""
        num_timesteps = config.get('num_noising_steps', 1000)
        beta_schedule = config.get('noise_schedule_type', None) or config.get('beta_type', 'linear')

        # diffusers uses `prediction_type` to interpret model output:
        #   - 'epsilon': model predicts noise
        #   - 'sample': model predicts x0 (x_start)
        pred = str(config.get('prediction_type', 'epsilon') or 'epsilon').strip().lower()
        if pred in {'x0', 'x_start', 'xstart', 'start', 'sample'}:
            pred = 'sample'
        elif pred in {'eps', 'epsilon', 'noise'}:
            pred = 'epsilon'
        else:
            raise ValueError(f"Unknown prediction_type/objective: {pred!r}. Use 'epsilon' or 'sample'.")
        
        if scheduler_type == 'ddpm':
            return DDPMScheduler(
                num_train_timesteps=num_timesteps,
                beta_schedule=beta_schedule,
                prediction_type=pred,
            )
        elif scheduler_type == 'ddim':
            return DDIMScheduler(
                num_train_timesteps=num_timesteps,
                beta_schedule=beta_schedule,
                prediction_type=pred,
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    
    def _build_condition_encoder(self, config: Dict[str, Any]) -> nn.Module:
        """Build condition encoder: 4-bit input → cross-attention features."""
        cross_attn_dim = 512  # Must match UNet cross_attention_dim

        cond_in_channels = int(config.get('cond_in_channels', config.get('in_channels', 1)) or 1)
        cond_in_channels = max(1, cond_in_channels)
        
        return nn.Sequential(
            nn.Conv2d(cond_in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, cross_attn_dim, kernel_size=1),
        )
    
    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Encode condition to cross-attention format.
        
        Args:
            condition: [B, 1, H, W]
        
        Returns:
            cond_embed: [B, H*W, cross_attention_dim]
        """
        # Get features: [B, C, H, W]
        cond_features = self.cond_encoder(condition)
        B, C, H, W = cond_features.shape
        
        # Reshape for cross-attention: [B, H*W, C]
        cond_embed = cond_features.view(B, C, H * W).permute(0, 2, 1)
        
        return cond_embed

    def set_training_progress(self, *, step: int | None, total_steps: int | None = None) -> None:
        """Set training progress for dynamic schedules (non-persistent).

        The trainer should call this once per optimizer step (or batch).
        """
        self._training_step = int(step) if step is not None else None
        self._training_total_steps = int(total_steps) if total_steps is not None else None

    @staticmethod
    def _linear_schedule_value(*, start: float, end: float, step: int, start_step: int, end_step: int) -> float:
        if end_step <= start_step:
            return float(end)
        if step <= start_step:
            return float(start)
        if step >= end_step:
            return float(end)
        t = (float(step) - float(start_step)) / (float(end_step) - float(start_step))
        return float(start + t * (end - start))

    def _get_loss_freq_weights(self, *, h: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        """Return a broadcastable weight tensor [1, 1, H, 1] or None.

        Controlled by config['loss_freq_weighting'].
        Designed for STFT-like representations where axis H corresponds to frequency bins.
        """
        cfg = self.config.get('loss_freq_weighting', None)
        if not isinstance(cfg, dict) or not cfg:
            return None

        if self._loss_freq_weight_cache is not None and self._loss_freq_weight_cache_h == int(h):
            return self._loss_freq_weight_cache.to(device=device, dtype=dtype)

        weight_type = str(cfg.get('type', 'lowpass')).lower()
        H = int(h)
        if H <= 0:
            return None

        # Allow optional linear schedule for the bucket weights over training steps.
        schedule = cfg.get('schedule') if isinstance(cfg.get('schedule'), dict) else None
        step = self._training_step
        total_steps = self._training_total_steps

        def _sched(key: str, default: float) -> float:
            base = float(cfg.get(key, default))
            if not schedule or step is None:
                return base
            stype = str(schedule.get('type', 'linear')).lower()
            if stype != 'linear':
                return base
            start_step = int(schedule.get('start_step', 0))
            end_step = schedule.get('end_step', None)
            if end_step is None:
                end_step = int(total_steps) if total_steps is not None else int(start_step)
            else:
                end_step = int(end_step)
            start_key = f"{key}_start"
            end_key = f"{key}_end"
            start_val = float(schedule.get(start_key, base))
            end_val = float(schedule.get(end_key, base))
            if stype == 'linear':
                return self._linear_schedule_value(
                    start=start_val,
                    end=end_val,
                    step=int(step),
                    start_step=int(start_step),
                    end_step=int(end_step),
                )
            return base

        w = torch.ones((H,), dtype=torch.float32)

        if weight_type == 'lowpass':
            cutoff_bins = int(cfg.get('cutoff_bins', max(1, H // 4)))
            cutoff_bins = max(1, min(H, cutoff_bins))
            low_weight = float(_sched('low_weight', 1.0))
            high_weight = float(_sched('high_weight', 0.25))
            w[:cutoff_bins] = low_weight
            w[cutoff_bins:] = high_weight
        elif weight_type in ('exp', 'exponential'):
            alpha = float(cfg.get('alpha', 0.03))
            idx = torch.arange(H, dtype=torch.float32)
            w = torch.exp(-alpha * idx)
        else:
            raise ValueError(f"Unknown loss_freq_weighting.type: {weight_type!r}")

        # Normalize so mean weight is ~1 (keeps loss scale comparable).
        w = w / (torch.mean(w) + 1e-12)
        w4 = w.view(1, 1, H, 1)
        self._loss_freq_weight_cache = w4
        self._loss_freq_weight_cache_h = H
        return w4.to(device=device, dtype=dtype)
    
    def forward(self, clean_images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Training forward pass (follows HF tutorial pattern).
        
        Args:
            clean_images: Target spectrogram, [B, 1, H, W]
            condition: Condition spectrogram, [B, 1, H, W]
        
        Returns:
            loss: Diffusion loss
        """
        batch_size = clean_images.shape[0]
        device = clean_images.device
        
        # Sample noise
        noise = torch.randn_like(clean_images)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device, dtype=torch.int64
        )
        
        # Add noise (forward diffusion)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # Encode condition
        cond_embed = self.encode_condition(condition)
        
        # Predict (configured) target from noisy input.
        # For prediction_type='epsilon' this is noise; for 'sample' this is x0.
        model_pred = self.unet(
            noisy_images,
            timesteps,
            encoder_hidden_states=cond_embed,
            return_dict=False
        )[0]

        pred = str(self.config.get('prediction_type', 'epsilon') or 'epsilon').strip().lower()
        if pred in {'x0', 'x_start', 'xstart', 'start', 'sample'}:
            pred = 'sample'
        elif pred in {'eps', 'epsilon', 'noise'}:
            pred = 'epsilon'
        else:
            raise ValueError(f"Unknown prediction_type/objective: {pred!r}. Use 'epsilon' or 'sample'.")

        target = noise if pred == 'epsilon' else clean_images

        # Base loss map (per-pixel), configurable for robustness.
        # Note: classic hinge loss is for classification margins; here we're regressing noise.
        # We still support a hinge-*like* epsilon-insensitive regression loss.
        loss_type = str(self.config.get('loss_type', 'mse') or 'mse').strip().lower()
        err = model_pred - target
        if loss_type in {'mse', 'l2'}:
            loss_map = err ** 2
        elif loss_type in {'l1', 'mae'}:
            loss_map = err.abs()
        elif loss_type in {'huber', 'smooth_l1', 'smoothl1'}:
            beta = float(self.config.get('loss_huber_beta', 1.0) or 1.0)
            # reduction='none' to keep a map matching [B,1,H,W]
            loss_map = F.smooth_l1_loss(model_pred, target, reduction='none', beta=beta)
        elif loss_type in {'hinge', 'eps_hinge', 'epsilon_hinge', 'epsilon_insensitive'}:
            # Epsilon-insensitive (SVR-style) hinge-like regression loss:
            #   max(0, |err| - eps)
            eps = float(self.config.get('loss_hinge_epsilon', 0.0) or 0.0)
            loss_map = torch.clamp(err.abs() - eps, min=0.0)
        elif loss_type in {'charbonnier', 'robust_l1', 'robustl1'}:
            # Charbonnier loss: sqrt(err^2 + eps^2)
            eps = float(self.config.get('loss_charbonnier_epsilon', 1e-3) or 1e-3)
            loss_map = torch.sqrt(err * err + (eps * eps))
        else:
            raise ValueError(
                f"Unknown loss_type: {loss_type!r}. Use: mse, l1, huber, hinge (epsilon-insensitive), charbonnier"
            )

        # Optional frequency-weighted loss to emphasize lower frequency bins.
        # Applies weights across the H dimension (assumed frequency axis).
        weights = self._get_loss_freq_weights(
            h=int(noise.shape[-2]),
            device=noise.device,
            dtype=model_pred.dtype,
        )

        # Optional: phase-channel weighting (phase is undefined where magnitude is tiny).
        # Applies a per-pixel weight for phase channels based on the (normalized) magnitude channel.
        phase_w_cfg = self.config.get('phase_loss_weighting', None)
        phase_weight_map = None
        if (
            bool(self.config.get('use_phase_channel', False))
            and isinstance(phase_w_cfg, dict)
            and bool(phase_w_cfg.get('enabled', False))
            and clean_images.ndim == 4
            and int(clean_images.shape[1]) >= 2
        ):
            # magnitude channel is normalized to [-1,1] -> map to [0,1]
            mag01 = torch.clamp((clean_images[:, :1].detach().to(torch.float32) + 1.0) / 2.0, 0.0, 1.0)
            gamma = float(phase_w_cfg.get('mag_gamma', 1.0) or 1.0)
            if not math.isfinite(gamma):
                gamma = 1.0
            mask = mag01 ** float(max(0.0, gamma))
            floor = float(phase_w_cfg.get('mag_floor', 0.0) or 0.0)
            if floor > 0:
                mask = torch.clamp(mask, min=floor)
            w_phase = float(phase_w_cfg.get('weight', 0.25) or 0.25)

            phase_weight_map = torch.ones_like(loss_map, dtype=loss_map.dtype, device=loss_map.device)
            phase_weight_map[:, 1:, :, :] = (w_phase * mask.to(dtype=loss_map.dtype))

        weighted = loss_map
        if weights is not None:
            weighted = weighted * weights
        if phase_weight_map is not None:
            weighted = weighted * phase_weight_map
        loss = torch.mean(weighted)

        # Always compute two-bucket diagnostics when configured.
        self.last_loss_components = None
        cfg = self.config.get('loss_freq_weighting', None)
        if isinstance(cfg, dict) and cfg:
            H = int(loss_map.shape[-2])
            cutoff = int(cfg.get('cutoff_bins', max(1, H // 4)))
            cutoff = max(1, min(H, cutoff))
            low_bucket = torch.mean(loss_map[:, :, :cutoff, :]).detach()
            high_bucket = torch.mean(loss_map[:, :, cutoff:, :]).detach() if cutoff < H else torch.tensor(0.0, device=low_bucket.device)
            out: Dict[str, float] = {
                'loss_total': float(loss.detach().item()),
                'loss_low_bucket': float(low_bucket.item()),
                'loss_high_bucket': float(high_bucket.item()),
                'bucket_cutoff_bins': float(cutoff),
                'loss_type': loss_type,
            }
            if weights is not None:
                # Weighted contributions (approx): mean(err2 * w) restricted by bucket.
                w = weights.detach().to(loss_map.device, dtype=loss_map.dtype)
                low_w = float(w[0, 0, :cutoff, 0].mean().item())
                high_w = float(w[0, 0, cutoff:, 0].mean().item()) if cutoff < H else 0.0
                out['mean_weight_low_bucket'] = low_w
                out['mean_weight_high_bucket'] = high_w
            self.last_loss_components = out
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        use_ddim: bool = True
    ) -> torch.Tensor:
        """Generate samples from condition.
        
        Args:
            condition: Condition spectrogram, [B, 1, H, W]
            num_inference_steps: Denoising steps
            generator: For reproducibility
            use_ddim: Use DDIM for faster sampling
        
        Returns:
            Generated samples, [B, 1, H, W]
        """
        device = condition.device
        batch_size = condition.shape[0]
        
        # Choose scheduler
        if use_ddim and self.scheduler_type == 'ddpm':
            # Create DDIM from DDPM config for faster sampling
            scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        else:
            scheduler = self.noise_scheduler
        
        scheduler.set_timesteps(num_inference_steps, device=device)

        H, W = self.config.get('image_size', (16, 128))
        
        # Start from noise
        image = torch.randn(
            (batch_size, 1, H, W),
            device=device,
            generator=generator
        )
        
        # Encode condition once
        cond_embed = self.encode_condition(condition)
        
        # Iterative denoising
        for t in scheduler.timesteps:
            # Predict noise
            noise_pred = self.unet(
                image,
                t,
                encoder_hidden_states=cond_embed,
                return_dict=False
            )[0]
            
            # Remove noise
            image = scheduler.step(noise_pred, t, image).prev_sample
        
        return image
    
    def save_checkpoint(self, path: Path, **kwargs):
        """Save checkpoint (Ray Train compatible)."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'unet_type': self.unet_type,
            'scheduler_type': self.scheduler_type,
            **kwargs
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    @classmethod
    def from_checkpoint(cls, path: Path, device: str = 'cpu'):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            checkpoint['config'],
            unet_type=checkpoint.get('unet_type', 'conditional'),
            scheduler_type=checkpoint.get('scheduler_type', 'ddpm')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint


# Factory functions for Ray Train
def create_diffuser(config: Dict[str, Any]) -> ConditionalDiffuser:
    """Factory for Ray Train compatibility."""
    # Optional: phase channel (mag + phase) doubles channel count.
    # Keep backwards compatibility by auto-setting channels when flag is enabled.
    if bool(config.get('use_phase_channel', False)):
        rep = str(config.get('phase_channel_representation', 'angle') or 'angle').strip().lower()
        desired_c = 3 if rep == 'sincos' else 2

        if int(config.get('in_channels', 1) or 1) == 1:
            config['in_channels'] = desired_c
        if int(config.get('out_channels', 1) or 1) == 1:
            config['out_channels'] = desired_c
        if 'cond_in_channels' not in config:
            config['cond_in_channels'] = int(config.get('in_channels', desired_c) or desired_c)

    scheduler_type = config.get('scheduler_type', config.get('sampler_type', 'ddpm'))
    model = ConditionalDiffuser(
        config,
        unet_type=config.get('unet_type', 'conditional'),
        scheduler_type=scheduler_type,
    )
    device = torch.device(config.get('device', 'cpu'))
    dtype = resolve_torch_dtype(config, device=device)
    model = model.to(device=device, dtype=dtype)
    return model


def get_optimizer(model: ConditionalDiffuser, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config (extensible for different optimizer types)."""
    optimizer_type = config.get('optimizer_type', 'adamw').lower()
    lr = config.get('learning_rate', 1e-4)

    if optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(config.get('adam_beta1', 0.95), config.get('adam_beta2', 0.999)),
            weight_decay=config.get('adam_weight_decay', 1e-6),
            eps=config.get('adam_epsilon', 1e-8),
        )
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(config.get('adam_beta1', 0.95), config.get('adam_beta2', 0.999)),
            eps=config.get('adam_epsilon', 1e-8),
        )
    if optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('sgd_momentum', 0.9),
            weight_decay=config.get('adam_weight_decay', 1e-6),
        )

    raise ValueError(f"Unknown optimizer_type: {optimizer_type}. Use: adamw, adam, sgd")


def get_lr_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], num_training_steps: int):
    """Create learning rate scheduler."""
    from diffusers.optimization import get_cosine_schedule_with_warmup
    
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.get('lr_warmup_steps', 500),
        num_training_steps=num_training_steps
    )
    
  

