"""
Configuration for diffusion model training.
"""

import os
from pathlib import Path
import diffusion.settings as settings
import torch


class DiffusionConfig:
    """Configuration for diffusion training."""
    __version__ = "2026-02-05-v4"  # Hardcoded version to debug stale code

    # Run metadata
    version = "V1" #change this between runs 
    run_name = ""

    # Immutable project paths
    diffusion_root = settings.DIFFUSION_ROOT
    project_root = settings.PROJECT_ROOT
    data_root = settings.PROJECT_ROOT / "data"
    raw_data_dir = data_root / "data" / "processed"
    raw_data_path = raw_data_dir / "arrhythmia_chunks.pt"
    
    # Version specific (include results dir)
    results_dir = diffusion_root / "results" / version
    checkpoint_dir = results_dir / "checkpoints"
    logs_dir = results_dir / "logs"
    samples_dir = results_dir / "samples"
    data_dir = results_dir / "data"

    # Diffusion Model Parameters
    unet_type = "conditional"
    scheduler_type = "ddpm"
    num_noising_steps = 100
    image_size = (128, 64)  # (height, width)
    in_channels = 1
    out_channels = 1

    # Conditioning
    bit_size = 4  # Condition bit depth
    real_bit_size = 16  # Real data bit depth
    quantizer_type = "uniform"
    transform_type = "stft"

    # Quantizer range computation (percentile clipping to avoid outliers dominating range)
    # Example: upper=99.5 means values above the 99.5th percentile clip to range_max.
    quantile_clip_lower = 0.0
    quantile_clip_upper = 99.5

    force_preprocess = False  # Set to True or use --force-preprocess flag to regenerate from raw dataset
    
    # Data pipeline config
    pipeline_config = {
        "transform": transform_type,
        # Natural 128x64 STFT for 3600-sample chunks (no cropping):
        # freq_bins = n_fft//2 + 1 = 128
        # time_frames = 1 + floor(L / hop_length) = 64 when L in [3591, 3647]
        "n_fft": 254,
        "hop_length": 57,
        "win_length": 254,
        "onesided": True,
        "center": True,
    }
    
    # Training parameters
    learning_rate = 1e-4
    batch_size = 16
    num_epochs = 50
    epochs = 50
    gradient_accumulation_steps = 1
    num_workers = 0
    mixed_precision = "bf16"  # Use bfloat16 for numerical stability (prevents NaN losses)
    max_samples = None  # No limit - use all available samples
    max_batches = None  # No limit - train on full dataset per epoch 
    
    # Checkpointing and validation
    save_every_n_epochs = 1
    validate_every_n_epochs = 1
    validation_split = 0.1

    # Hold out a fixed test set that is never used in training/validation.
    # By default, reserve the last N samples (stable across runs).
    test_holdout_count = 500
    test_holdout_from_end = True
    
    # Optimizer
    optimizer_type = 'adamw'  # 'adamw', 'adam', or 'sgd'
    adam_beta1 = 0.95
    adam_beta2 = 0.999
    adam_weight_decay = 1e-6
    adam_epsilon = 1e-8
    sgd_momentum = 0.9
    
    # LR Scheduler
    lr_warmup_steps = None  # None = warmup for first epoch automatically
    
    # Sampling
    num_trajectories = 16  # Number of diverse samples per condition
    
    device = (
        "cuda"
        if hasattr(torch, 'cuda') and torch.cuda.is_available()
        else 'cpu'
    )

    # Tensor dtype for model + data on device.
    # Use bfloat16 instead of float16 for better numerical stability
    torch_dtype = "bfloat16"
    
    # Random seed
    random_seed = 42
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert config to dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }

