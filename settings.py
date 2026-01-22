"""Project settings and paths."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data directories
RAW_DIR = PROJECT_ROOT / "data" / "data"/ "raw"
PROCESSED_DIR = PROJECT_ROOT / "data"/ "data" / "processed"

# Input files
ART_SIGNALS_FILE = RAW_DIR / "art_signals.parquet"
ART_CHUNKS_FILE = RAW_DIR / "art_chunks.pt"

# Output files
STFT_LINEAR_OUTPUT = PROCESSED_DIR / "signals_linear_stft.pt"
STFT_UNIFORM_OUTPUT = PROCESSED_DIR / "signals_uniform_stft.pt"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
