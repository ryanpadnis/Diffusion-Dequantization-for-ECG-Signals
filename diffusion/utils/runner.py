"""Runnable training runner with direct torch transforms.

Orchestrates: load raw data → transform (on device) → quantize → train → save.
Uses apply() directly for GPU speed instead of sklearn pipeline numpy conversion.
"""

import os
from pathlib import Path
from typing import Optional

import torch

from diffusion import settings as diff_settings
from diffusion.utils.trainer import DiffusionTrainer
from data.preprocess.transform import get_transform, AVAILABLE_TRANSFORMS
from data.preprocess.quantize import UniformQuantizer



def load_config(config_path: Optional[Path]) -> dict:
    """Load config from JSON, merging with defaults."""
    if config_path is None:
        return diff_settings.DEFAULT_PIPELINE_CONFIG.copy()
    with open(config_path, "r") as f:
        import json
        user_config = json.load(f)
    config = diff_settings.DEFAULT_PIPELINE_CONFIG.copy()
    config.update(user_config)
    return config


def load_data(data_path: Path) -> torch.Tensor:
    """Load raw data (signals or chunks) from .pt file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    data = torch.load(data_path)
    signals = data.get("signals") or data.get("chunks")
    if signals is None:
        raise KeyError("Expected key 'signals' or 'chunks' in loaded data")

    return signals


def create_transform(transform_config: dict, device: str):
    """Create transform object from config using registry.
    
    Config keys:
    - transform: 'stft', 'dft', or 'haar' (default: 'stft')
    - n_fft: STFT window size (default: 30)
    - hop_length: STFT hop length (default: 64)
    - onesided: STFT onesided (default: True)
    
    Returns transform ready for apply() on device.
    """
    transform_name = transform_config.get('transform', 'stft')
    device_obj = torch.device(device)
    
    # Get transform from registry
    if transform_name == 'stft':
        return get_transform('stft',
            n_fft=transform_config.get('n_fft', 30),
            hop_length=transform_config.get('hop_length', 64),
            onesided=transform_config.get('onesided', True),
            device=device_obj
        )
    else:
        return get_transform(transform_name, device=device_obj)


def create_quantizer(config: dict, signals: torch.Tensor) -> UniformQuantizer:
    """Create quantizer from config and compute range from signals."""
    bits = config.get('bits', 4)
    range_min = signals.min().item()
    range_max = signals.max().item()
    
    # Add small margin to avoid edge effects
    margin = (range_max - range_min) * 0.01
    range_min -= margin
    range_max += margin
    
    return UniformQuantizer(bits=bits, range_min=range_min, range_max=range_max)


def build_diffuser(config: dict):
    """Build diffuser model from config. (Stub - fill in with actual model)"""
    # TODO: Implement actual diffuser model construction
    print(f"[build_diffuser] Config: {config}")
    return None


def train_diffuser(config: dict):
    """Main training orchestrator: load → transform (on device) → quantize → train → save.
    
    Uses apply() directly for GPU speed instead of sklearn pipeline numpy conversion.
    
    Config keys (one of):
    - raw_data_path: Path to raw data .pt file (chunks/signals) - will apply transform & quantization
    - train_data_path: Path to pre-processed train data .pt file - load directly
    - epochs: Number of training epochs (default: 1)
    - run_name: Run name (default: from settings)
    - model_name: Model name (default: from settings)
    - device: Device to train on (default: 'cpu')
    - transform, n_fft, hop_length, onesided: Transform config (only for raw_data_path)
    - bits: Quantization bits (default: 4, only for raw_data_path)
    """
    
    epochs = config.get('epochs', 1)
    run_name = config.get('run_name', diff_settings.DEFAULT_RUN_NAME)
    model_name = config.get('model_name', diff_settings.DEFAULT_MODEL_NAME)
    device = config.get('device', 'cpu')
    
    print(f"[train_diffuser] Config: {config}")
    print(f"[train_diffuser] Device: {device}")
  
    # Load data: either raw or pre-processed
    train_data_path = config.get('data_path')
    raw_data_path = config.get('raw_data_path')
    
    if os.path.exists(train_data_path):
        # Load pre-processed data directly
        print(f"[train_diffuser] Loading pre-processed train data from: {train_data_path}")
        signals_dequantized = load_data(Path(train_data_path))
        signals_dequantized = signals_dequantized.to(device)
        print(f"[train_diffuser] Loaded train data shape: {signals_dequantized.shape}")
        
    elif os.path.exists(raw_data_path):
        # Load raw data and apply pipeline
        print(f"[train_diffuser] Loading raw data from: {raw_data_path}")
        signals = load_data(Path(raw_data_path))
        signals = signals.to(device)
        print(f"[train_diffuser] Loaded raw signals shape: {signals.shape}")
        
        # Apply transform directly on device
        transform = create_transform(config, device)
        signals_transformed = transform.apply(signals)
        print(f"[train_diffuser] After transform shape: {signals_transformed.shape}")
        
        # Apply quantization
        quantizer = create_quantizer(config, signals_transformed)
        signals_quantized = quantizer.quantize(signals_transformed)
        signals_dequantized = quantizer.dequantize(signals_quantized)
        print(f"[train_diffuser] After quantization shape: {signals_dequantized.shape}")
    else:
        raise ValueError("Config must have either 'train_data_path' or 'raw_data_path'")
    
    # Build diffuser and train
    diffuser = build_diffuser(config)
    
    trainer = DiffusionTrainer(
        model_name=model_name,
        run_name=run_name,
        checkpoints_dir=diff_settings.CHECKPOINTS_DIR,
        samples_dir=diff_settings.SAMPLES_DIR,
        logs_dir=diff_settings.LOGS_DIR,
        device=device,
    )
    
   
    trainer.fit(signals_dequantized, epochs=epochs)
    print(f"[train_diffuser] Outputs saved to:")
    print(f"  - Checkpoints: {diff_settings.CHECKPOINTS_DIR}")
    print(f"  - Samples:     {diff_settings.SAMPLES_DIR}")
    print(f"  - Logs:        {diff_settings.LOGS_DIR}")



if __name__ == '__main__':
    from diffusion.utils.config import DiffusionConfig
    config = vars(DiffusionConfig)
    train_diffuser(config)
