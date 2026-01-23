"""Diffusion project paths and defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DIFFUSION_ROOT = Path(__file__).parent

# Output directories
CHECKPOINTS_DIR = DIFFUSION_ROOT / "checkpoints"
SAMPLES_DIR = DIFFUSION_ROOT / "samples"
LOGS_DIR = DIFFUSION_ROOT / "logs"

# Defaults
DEFAULT_MODEL_NAME = "unet"
DEFAULT_RUN_NAME = "run1"

# Pipeline config defaults
DEFAULT_PIPELINE_CONFIG = {
    "transform": "stft",      # 'stft', 'dft', or 'haar'
    "n_fft": 30,              # STFT window size
    "hop_length": 64,         # STFT hop length
    "onesided": True,         # STFT one-sided spectrum
}

# Ensure directories exist
for _dir in (CHECKPOINTS_DIR, SAMPLES_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
