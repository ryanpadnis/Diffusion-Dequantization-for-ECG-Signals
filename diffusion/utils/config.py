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
    # Mirroring V2/20260213_151839 baseline (STFT 128x64).
    version = "V8"  # change this between runs
    run_name = ""  # Descriptive name for this run (used in results dir)

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

    # What the model predicts.
    # - 'epsilon' (default): predict noise (classic DDPM objective)
    # - 'sample': predict x0 / x_start (denoised sample)
    # Note: this is forwarded to diffusers schedulers as prediction_type.
    prediction_type = "sample"
    beta_type = "linear"  # 'linear', 'cosine', or 'sigmoid'
    # Alias for clarity: diffusers calls this the "beta_schedule".
    # Keep beta_type for backwards compatibility.
    noise_schedule_type = "linear"  # if set, overrides beta_type
    num_noising_steps = 2500
    # For STFT: image_size = (freq_bins, time_frames)
    # Use power-of-two dimensions for the UNet.
    # 128 freq bins comes from n_fft=254 with onesided=True (254//2 + 1 = 128)
    image_size = (128, 64)
    in_channels = 1
    out_channels = 1

    # Conditioning
    bit_size = 4  # Condition bit depth
    real_bit_size = 11  # Real data bit depth

    # If True, treat `raw_data_path` waveforms as already quantized to `real_bit_size`.
    # Training/analysis then use the raw waveform as the target time-domain signal.
    # (Conditioning still uses `bit_size` quantization.)
    assume_raw_is_real_bits = True
    quantizer_type = "uniform"
    transform_type = "stft"

    # Quantizer range computation (percentile clipping to avoid outliers dominating range)
    # Example: upper=99.5 means values above the 99.5th percentile clip to range_max.
    quantile_clip_lower = 0.0
    quantile_clip_upper = 100.0

    # Spectrogram magnitude normalization.
    # - 'minmax': per-sample (cond) min/max over (F,T) then map to [-1,1]
    # - 'absmax'/'peak': per-sample (cond) peak magnitude then map to [-1,1] (less sensitive to noisy mins)
    # - 'zscore': per-sample (cond) mean + std*clamp_sigma -> clamp to [-1,1]
    #             avoids R-peak spikes dominating the scale; denorm = x * (std*sigma) + mean
    # - 'none': no normalization — raw STFT magnitudes passed directly to model
    #           WARNING: model input will be in [0, ~600] range; only use for debugging
    mag_norm_mode = "zscore"
    mag_norm_clamp_sigma = 4.5  # std devs that map to ±1; values beyond are clamped
    mag_norm_epsilon = 1e-8

    # Optional: include STFT phase as an additional channel.
    # If enabled, tensors become [B, 2, F, T] where:
    #   channel 0: magnitude normalized to [-1, 1] (per-sample, condition-derived)
    #   channel 1: phase angle normalized to [-1, 1] via (phase_radians / pi)
    # The diffusion model then predicts noise for BOTH channels.
    use_phase_channel = False

    # Phase channel representation when use_phase_channel=True:
    # - 'angle': single channel in [-1,1] via (phase_rad / pi)
    # - 'sincos': two channels (sin, cos), giving total channels = 1(mag) + 2 = 3
    phase_channel_representation = 'angle'

    # When use_phase_channel=True, control which phase is used for ISTFT inversion.
    # - 'auto': use predicted phase if available, else fall back to condition phase
    # - 'predicted': require predicted phase
    # - 'condition': always use condition phase (useful for ablations)
    phase_inversion_source = 'auto'

    # Optional: help phase training by down-weighting phase where magnitude is tiny.
    # This weights the diffusion *noise prediction loss* per-pixel for phase channels.
    # - enabled: apply weighting
    # - weight: overall multiplier on phase loss
    # - mag_gamma: exponent for magnitude-based mask (higher -> focus on strong bins)
    # - mag_floor: minimum mask value to avoid all-zero gradients
    phase_loss_weighting = {
        'enabled': False,
        'weight': 0.25,
        'mag_gamma': 1.0,
        'mag_floor': 0.0,
    }

    # Time-domain quantization range policy
    # - 'per_sample': each waveform gets its own symmetric range (recommended to avoid global range bias)
    # - 'global': a single symmetric range computed from the training split
    time_quantization_mode = 'per_sample'

    force_preprocess = False  # Set to True or use --force-preprocess flag to regenerate from raw dataset
    
    # Data pipeline config (V2 baseline)
    pipeline_config = {
        "transform": transform_type,
        "n_fft": 254,
        "win_length": 254,
        "hop_length": 57,
        "onesided": True,
        "center": True,
    }

    # Optional: time-domain low-pass applied right before training preprocessing.
    # Uses the same ideal rFFT masking approach as diffusion/analysis/analysis.py.
    # Applies to the raw waveform BEFORE time quantization and STFT.
    pre_lowpass = {
        "enabled": False,
        "cutoff_hz": 40.0,
        "sample_rate_hz": 360.0,
    }

    # Optional: frequency-weighted diffusion loss.
    # Two buckets: [0:cutoff_bins) is "low", [cutoff_bins:H) is "other".
    # The trainer will print bucket losses periodically.
    """loss_freq_weighting = {
        "type": "lowpass",
        "cutoff_bins": 16,
        "low_weight": 1.0,
        "high_weight": 0.2,
        # Dynamic schedule (linear) for weights over training steps.
        "schedule": {
            "type": "linear",
            "start_step": 0,
            # end_step omitted -> uses total_training_steps
            "high_weight_start": 0.2,
            "high_weight_end": 0.05,
        },
    }"""

    # Base diffusion loss type (robust losses can help with sharp transitions).
    # Options: 'mse' | 'l1' | 'huber' | 'hinge' (epsilon-insensitive) | 'charbonnier'
    # Recommended default: huber.
    loss_type = "mse"
    loss_huber_beta = 1.0
    # Epsilon-insensitive hinge-like regression loss: max(0, |err| - eps)
    loss_hinge_epsilon = 0.0
    # Charbonnier: sqrt(err^2 + eps^2)
    loss_charbonnier_epsilon = 1e-3

    # Optional EMA of model weights (for sampling stability).
    """ema_config = {
        "enabled": True,
        "decay": 0.999,
        "schedule": {
            "type": "linear",
            "start_step": 0,
            # end_step omitted -> uses total_training_steps
            "decay_start": 0.995,
            "decay_end": 0.9999,
        },
    }"""
    
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

    # Early stopping / pruning (based on moving-average validation loss)
    # Window=5 matches the requested 5-epoch moving average.
    # When enabled, training stops when the moving average over the most recent
    # `window_epochs` validation losses fails to improve for `patience` checks.
    early_stop_config = {
        "enabled": True,
        "window_epochs": 5,
        "patience": 3,
        "min_delta": 0.001,
    }

    # How often to print/log low vs high frequency loss buckets.
    loss_components_every_n_steps = 50

    # Hold out a fixed test set that is never used in training/validation.
    # By default, reserve the last N samples (stable across runs).
    test_holdout_count = 500
    test_holdout_from_end = True
    
    # Optimizer
    optimizer_type = 'adamw'  # 'adamw', 'adam', or 'sgd'

    
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
        out = {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }
        out["__version__"] = getattr(cls, "__version__", "")
        return out

