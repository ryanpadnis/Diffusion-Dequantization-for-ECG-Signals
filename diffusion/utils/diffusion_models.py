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
        beta_schedule = config.get('noise_schedule', 'linear')
        
        if scheduler_type == 'ddpm':
            return DDPMScheduler(
                num_train_timesteps=num_timesteps,
                beta_schedule=beta_schedule,
            )
        elif scheduler_type == 'ddim':
            return DDIMScheduler(
                num_train_timesteps=num_timesteps,
                beta_schedule=beta_schedule,
            )
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    
    def _build_condition_encoder(self, config: Dict[str, Any]) -> nn.Module:
        """Build condition encoder: 4-bit input → cross-attention features."""
        cross_attn_dim = 512  # Must match UNet cross_attention_dim
        
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
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
        
        # Predict noise
        noise_pred = self.unet(
            noisy_images,
            timesteps,
            encoder_hidden_states=cond_embed,
            return_dict=False
        )[0]
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
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
    
  

