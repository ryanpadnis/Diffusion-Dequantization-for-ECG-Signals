"""Project settings and paths."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
RAW_DIR = PROJECT_ROOT / "data" / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT /"data" / "data" / "processed"

# Input files
ART_SIGNALS_FILE = RAW_DIR / "art_signals.parquet"
ART_CHUNKS_FILE = RAW_DIR / "art_chunks.pt"
PREPROC_TRANSFORMS_FILE = PROCESSED_DIR / "signals_transforms.pt"

# Output files
STFT_LINEAR_OUTPUT = PROCESSED_DIR / "signals_linear_stft.pt"
STFT_UNIFORM_OUTPUT = PROCESSED_DIR / "signals_uniform_stft.pt"

# Preprocessing options and output naming
DEFAULT_BITS = [4]
DEFAULT_QUANTIZERS = ["uniform"]
DEFAULT_TRANSFORMS = ["stft"]
OUTPUT_FILENAME_PATTERN = "signals_{quantizer}_{transform}_{bits}bit.pt"

def output_path(quantizer: str, transform: str, bits: int) -> Path:
	"""Generate output file path for given settings."""
	return PROCESSED_DIR / OUTPUT_FILENAME_PATTERN.format(
		quantizer=quantizer,
		transform=transform,
		bits=bits,
	)

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
