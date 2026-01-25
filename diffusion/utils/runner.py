"""Runnable training runner with direct torch transforms.

Orchestrates: load raw data → transform (on device) → quantize → train → save.
Uses apply() directly for GPU speed instead of sklearn pipeline numpy conversion.
"""

import os
from pathlib import Path
from typing import Optional

import torch

from data.preprocess.transform import get_transform, AVAILABLE_TRANSFORMS
from data.preprocess.quantize import UniformQuantizer, compute_range_from_tensor


def _resolve_torch_dtype(config: dict, device: str) -> torch.dtype:
    dtype_val = config.get('torch_dtype', 'float32')
    if isinstance(dtype_val, torch.dtype):
        dtype = dtype_val
    else:
        dtype = getattr(torch, str(dtype_val), torch.float32)

    dev = torch.device(device)
    if dev.type == 'cpu' and dtype == torch.float16:
        return torch.float32
    return dtype


def construct_train_filename(quantizer_type: str, transform_type: str, bits: int, cond: bool = False) -> str:
    """Construct filename for processed training data based on parameters.
    
    Args:
        quantizer_type: Type of quantizer (e.g., 'uniform')
        transform_type: Type of transform (e.g., 'stft')
        bits: Number of bits for quantization
        cond: If True, use 'train_cond' prefix; otherwise use 'train_data'
    """
    prefix = "train_cond" if cond else "train_data"
    return f"{prefix}_{quantizer_type}_{transform_type}_{bits}bit.pt"





def load_data(data_path: Path) -> torch.Tensor:
    """Load raw data (signals or chunks) from .pt file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    data = torch.load(data_path)
    
    # Handle both dict and direct tensor formats
    if isinstance(data, dict):
        signals = data.get("signals")
        if signals is None:
            signals = data.get("chunks")
        if signals is None:
            raise KeyError("Expected key 'signals' or 'chunks' in loaded data")
    else:
        signals = data

    return signals


def create_transform(transform_config: dict, device: str):
    """Create transform object from pipeline_config using registry.
    
    Prefers `pipeline_config` inside the overall training config for consistency
    across training and sampling.
    """
    device_obj = torch.device(device)
    pipeline_config = transform_config.get('pipeline_config', {})
    transform_name = pipeline_config.get('transform', 'stft')

    if transform_name == 'stft':
        return get_transform(
            'stft',
            n_fft=pipeline_config.get('n_fft', 30),
            hop_length=pipeline_config.get('hop_length', 64),
            win_length=pipeline_config.get('win_length', None),
            onesided=pipeline_config.get('onesided', True),
            center=pipeline_config.get('center', False),
            device=device_obj,
        )
    else:
        return get_transform(transform_name, device=device_obj)


def create_quantizer(config: dict, signals: torch.Tensor) -> UniformQuantizer:
    """Create quantizer from config and compute range from signals."""
    bits = config.get('bits', 4)

    # Allow percentile clipping to avoid extreme outliers.
    lower_pct = float(config.get('quantile_clip_lower', 0.0))
    upper_pct = float(config.get('quantile_clip_upper', 100.0))

    # Compute robust range using percentiles on the transformed signals
    try:
        range_min, range_max = compute_range_from_tensor(signals, lower_pct, upper_pct)
    except Exception:
        # Fallback to min/max
        range_min = float(signals.min().item())
        range_max = float(signals.max().item())

    # STFT magnitudes are non-negative; anchoring at 0 often prevents crushing low-energy bins
    if config.get('transform_type', 'stft') == 'stft':
        range_min = 0.0 if lower_pct <= 0.0 else range_min

    # Add small margin to avoid edge effects
    margin = (range_max - range_min) * 0.01 if (range_max - range_min) != 0 else 0.0
    range_max = range_max + margin

    q = UniformQuantizer(bits=bits, range_min=range_min, range_max=range_max)
    # Attach metadata for debugging
    q._meta = {'lower_pct': lower_pct, 'upper_pct': upper_pct}
    return q


def create_time_quantizer(config: dict, signals_time: torch.Tensor, bits: int) -> UniformQuantizer:
    """Create a uniform quantizer for time-domain signals.

    Uses percentile clipping and a symmetric range around 0 for stability.
    """
    lower_pct = float(config.get('quantile_clip_lower', 0.0))
    upper_pct = float(config.get('quantile_clip_upper', 100.0))

    lo, hi = compute_range_from_tensor(signals_time, lower_pct, upper_pct)
    peak = max(abs(lo), abs(hi))
    # Avoid degenerate range
    if peak <= 0:
        peak = float(signals_time.abs().max().item())
    if peak <= 0:
        peak = 1.0

    q = UniformQuantizer(bits=bits, range_min=-peak, range_max=peak)
    q._meta = {'lower_pct': lower_pct, 'upper_pct': upper_pct, 'symmetric': True}
    return q


def build_diffuser(config: dict):
    """Build diffuser model, optimizer, and scheduler from config."""
    from diffusion.utils.model import create_diffuser, get_optimizer, get_lr_scheduler
    diffuser = create_diffuser(config)
    optimizer = get_optimizer(diffuser, config)
    lr_scheduler = None
    
    print(f"[build_diffuser] Created ConditionalDiffuser")
    print(f"  - Image size: {config.get('image_size', (16, 128))}")
    print(f"  - Channels: {config.get('in_channels', 1)}")
    print(f"  - Optimizer: {type(optimizer).__name__}")
    
    return diffuser, optimizer, lr_scheduler


def train_diffuser(config: dict):
    """Main training orchestrator: load/create condition (4-bit) and real data (16-bit).
    
    Loads or processes two datasets:
    - train_cond: Condition dataset at low bit depth (4-bit)
    - train_data: Real data at high bit depth (16-bit)
    
    Config keys:
    - raw_data_path: Path to raw data .pt file
    - data_dir: Directory to store processed data
    - bit_size: Bit depth for condition data (default: 4)
    - real_bit_size: Bit depth for real data (default: 16)
    - quantizer_type, transform_type, device, etc.
    """

    epochs = config.get('epochs', 1)
    device = config.get('device', 'cpu')
    
    print(f"[train_diffuser] Device: {device}")

    # Get paths and processing parameters from config
    data_dir = Path(config.get('data_dir'))
    raw_data_path = Path(config.get('raw_data_path'))
    quantizer_type = config.get('quantizer_type', 'uniform')
    transform_type = config.get('transform_type', 'stft')
    cond_bits = config.get('bit_size', 4)  # Condition dataset bit depth
    real_bits = config.get('real_bit_size', 16)  # Real data bit depth
    
    # Get directories from config
    checkpoint_dir = Path(config.get('checkpoint_dir'))
    samples_dir = Path(config.get('samples_dir'))
    logs_dir = Path(config.get('logs_dir'))

    # Keep TensorBoard runs clean by writing each run to its own log subdir.
    # This prevents mixed/overlaid curves from multiple runs.
    from datetime import datetime
    run_id = str(config.get('run_id') or datetime.now().strftime('%Y%m%d_%H%M%S'))
    logs_dir = logs_dir / run_id
    config['logs_dir'] = str(logs_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Save config for later loading by sampler
    import pickle
    config_path = Path(config.get('results_dir')) / 'config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"[train_diffuser] Config saved to: {config_path}")
    
    # Real data versus conditonal (hgih vs low bit depth)
    cond_filename = construct_train_filename(quantizer_type, transform_type, cond_bits, cond=True)
    real_filename = construct_train_filename(quantizer_type, transform_type, real_bits, cond=False)
    
    cond_data_path = data_dir / cond_filename
    real_data_path = data_dir / real_filename
  
    force_preprocess = bool(config.get('force_preprocess', False))

    if (not force_preprocess) and cond_data_path.exists() and real_data_path.exists():
        # Load pre-processed data directly
        print(f"[train_diffuser] Loading pre-processed datasets:")
        print(f"  - Condition (4-bit): {cond_data_path}")
        cond_data = load_data(cond_data_path).to(device)
        print(f"  - Real data (16-bit): {real_data_path}")
        real_data = load_data(real_data_path).to(device)
        
        # Channel dimension for 2-d uent
        if cond_data.ndim == 3:
            cond_data = cond_data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        if real_data.ndim == 3:
            real_data = real_data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        
        print(f"[train_diffuser] Condition shape: {cond_data.shape}")
        print(f"[train_diffuser] Real data shape: {real_data.shape}")
        
    elif raw_data_path.exists():
        # Load raw data and process to both bit depths
        print(f"[train_diffuser] Loading raw data from: {raw_data_path}")
        signals = load_data(raw_data_path)
        signals = signals.to(device)
        print(f"[train_diffuser] Loaded raw signals shape: {signals.shape}")

        # Quantize in TIME DOMAIN first to create degraded (4-bit) and target (16-bit) waveforms.
        print(f"[train_diffuser] Time-domain quantization: {cond_bits}-bit condition, {real_bits}-bit target")
        cond_time_quantizer = create_time_quantizer(config, signals, bits=cond_bits)
        real_time_quantizer = create_time_quantizer(config, signals, bits=real_bits)

        # Build simulated quantized waveforms (values lie on discrete quantizer levels)
        cond_time = cond_time_quantizer.quantize(signals)
        real_time = real_time_quantizer.quantize(signals)

        # Apply transform (STFT) to both waveforms to get matrices (spectrogram magnitudes)
        transform_cond = create_transform(config, device)
        transform_real = create_transform(config, device)

        cond_mag = transform_cond.apply(cond_time)
        real_mag = transform_real.apply(real_time)
        print(f"[train_diffuser] After transform shapes: cond {cond_mag.shape}, real {real_mag.shape}")

        cond_data = cond_mag
        real_data = real_mag
        
    
        if cond_data.ndim == 3:
            cond_data = cond_data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        if real_data.ndim == 3:
            real_data = real_data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        
        print(f"[train_diffuser] Condition data shape: {cond_data.shape}")
        print(f"[train_diffuser] Real data shape: {real_data.shape}")
        
        print(f"[train_diffuser] Saving condition spectrograms to: {cond_data_path}")
        torch.save({'signals': cond_data.to(torch.float32).cpu()}, cond_data_path)
        print(f"[train_diffuser] Saving real spectrograms to: {real_data_path}")
        torch.save({'signals': real_data.to(torch.float32).cpu()}, real_data_path)
        print(f"[train_diffuser] Saved both datasets")
    else:
        raise ValueError(f"Raw data not found at {raw_data_path}")
    
    print(f"[train_diffuser] Computing magnitude ranges (full dataset)")
    cond_mag_min = float(cond_data.min().item())
    cond_mag_max = float(cond_data.max().item())
    real_mag_min = float(real_data.min().item())
    real_mag_max = float(real_data.max().item())

    # Limit dataset size for faster local testing (training subset only)
    max_samples = config.get('max_samples')
    if max_samples and max_samples < len(cond_data):
        print(f"[train_diffuser] Using subset for training: {max_samples}/{len(cond_data)} samples")
        indices = torch.randperm(len(cond_data))[:max_samples]
        cond_data = cond_data[indices]
        real_data = real_data[indices]

    # Limit to specific number of batches for quick testing (training subset only)
    batch_size = config.get('batch_size', 16)
    max_batches = config.get('max_batches')
    if max_batches:
        max_samples_for_batches = max_batches * batch_size
        if max_samples_for_batches < len(cond_data):
            print(f"[train_diffuser] Limiting training to {max_batches} batches ({max_samples_for_batches} samples)")
            cond_data = cond_data[:max_samples_for_batches]
            real_data = real_data[:max_samples_for_batches]

    # Normalize SPECTROGRAM MAGNITUDES to [-1, 1] using the full-dataset ranges
    print(f"[train_diffuser] Normalizing spectrogram magnitudes to [-1, 1]")

    cond_data = cond_data.to(torch.float32)
    real_data = real_data.to(torch.float32)

    cond_data = (cond_data - cond_mag_min) / (cond_mag_max - cond_mag_min + 1e-8)
    cond_data = cond_data * 2.0 - 1.0
    real_data = (real_data - real_mag_min) / (real_mag_max - real_mag_min + 1e-8)
    real_data = real_data * 2.0 - 1.0

    # Cast tensors to specified dtype 
    train_dtype = _resolve_torch_dtype(config, device)
    cond_data = cond_data.to(dtype=train_dtype)
    real_data = real_data.to(dtype=train_dtype)

    print(f"  - Condition mag range (full): [{cond_mag_min:.6f}, {cond_mag_max:.6f}]")
    print(f"  - Real mag range (full):      [{real_mag_min:.6f}, {real_mag_max:.6f}]")
    print(f"  - Normalized condition range: [{cond_data.min():.4f}, {cond_data.max():.4f}]")
    print(f"  - Normalized real range:      [{real_data.min():.4f}, {real_data.max():.4f}]")
    
    normalizer_params = {
        'cond_time_range_min': (cond_time_quantizer.range_min if 'cond_time_quantizer' in locals() else None),
        'cond_time_range_max': (cond_time_quantizer.range_max if 'cond_time_quantizer' in locals() else None),
        'real_time_range_min': (real_time_quantizer.range_min if 'real_time_quantizer' in locals() else None),
        'real_time_range_max': (real_time_quantizer.range_max if 'real_time_quantizer' in locals() else None),
        'time_quant_clip_lower': (getattr(cond_time_quantizer, '_meta', {}).get('lower_pct') if 'cond_time_quantizer' in locals() else None),
        'time_quant_clip_upper': (getattr(cond_time_quantizer, '_meta', {}).get('upper_pct') if 'cond_time_quantizer' in locals() else None),
        'cond_mag_min': cond_mag_min,
        'cond_mag_max': cond_mag_max,
        'real_mag_min': real_mag_min,
        'real_mag_max': real_mag_max,
        'cond_bits': int(cond_bits),
        'real_bits': int(real_bits),
        'transform_type': transform_type,
        'pipeline_config': config.get('pipeline_config'),
    }
    normalizer_path = data_dir / 'normalizer_params.pkl'
    with open(normalizer_path, 'wb') as f:
        pickle.dump(normalizer_params, f)
    print(f"[train_diffuser] Normalizer params saved to: {normalizer_path}")
    
    # Build diffuser and train
    diffuser, optimizer, lr_scheduler = build_diffuser(config)
    
    from diffusion.utils.trainer import DiffusionTrainer
    from diffusion.utils.model import get_lr_scheduler
    
    # Calculate total training steps for lr scheduler
    batch_size = config.get('batch_size', 16)
    num_samples = len(cond_data)
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
 
    warmup_steps = config.get('lr_warmup_steps')
    if warmup_steps is None:
        warmup_steps = steps_per_epoch
    print(f"[train_diffuser] LR warmup: {warmup_steps} steps ({warmup_steps/steps_per_epoch:.1f} epochs)")
    config['lr_warmup_steps'] = warmup_steps

    lr_scheduler = get_lr_scheduler(optimizer, config, total_steps)
    
    trainer = DiffusionTrainer(
        model=diffuser,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoints_dir=checkpoint_dir,
        samples_dir=samples_dir,
        logs_dir=logs_dir,
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        mixed_precision=config.get('mixed_precision'),
        validation_split=config.get('validation_split', 0.1),
        save_every_n_epochs=config.get('save_every_n_epochs', 1),
        validate_every_n_epochs=config.get('validate_every_n_epochs', 1),
        log_advanced_metrics=config.get('log_advanced_metrics', True),
        advanced_metrics_every_n_steps=config.get('advanced_metrics_every_n_steps', 50),
        energy_curve_every_n_steps=config.get('energy_curve_every_n_steps', 200),
    )
    
    trainer.fit(
        cond_data=cond_data,
        real_data=real_data,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=config.get('num_workers', 0),
    )
    print(f"[train_diffuser] Outputs saved to:")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Samples:     {samples_dir}")
    print(f"  - Logs:        {logs_dir}")



if __name__ == '__main__':
    from diffusion.utils.config import DiffusionConfig
    config = DiffusionConfig.to_dict()
    train_diffuser(config)
