"""Diffusion project paths and defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DIFFUSION_ROOT = Path(__file__).parent


# Defaults
DEFAULT_MODEL_NAME = "unet"
DEFAULT_RUN_NAME = "run1"

# Pipeline config defaults
DEFAULT_PIPELINE_CONFIG = {
    "transform": "stft",      # 'stft', 'dft', or 'haar'
    "n_fft": 254,            # STFT window size
    "hop_length": 57,        # STFT hop length
    "win_length": 254,       # STFT window length
    "center": True,          # Centered padding
    "onesided": True,        # STFT one-sided spectrum
}


