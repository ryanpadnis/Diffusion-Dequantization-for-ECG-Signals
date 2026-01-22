# EE269Project

Diffusion Models for generating higher bit depth vital signs signals from VitalDB data.

## Setup

### Requirements
- Python 3.9+
- `uv` package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. Clone the repository:
```bash
git clone <repo>
cd EE269Project
```

2. Install dependencies using `uv`:
```bash
uv sync
```

This creates a `.venv` environment and installs all dependencies from `pyproject.toml` and `uv.lock`.

### Running Scripts

All scripts use `uv run` (no activation needed):

```bash
# Load VitalDB records
uv run python data/load/load.py

# Chunk signals into sequences
uv run python data/preprocess/chunk.py

# Compute STFT
uv run python data/preprocess/transform.py

# Visualize signals
uv run python data/visualize_chunks.py
```

## Data Pipeline

1. **Load** (`data/load/load.py`) - Download ART signals from VitalDB
   - Saves to `data/raw/art_signals.parquet`

2. **Chunk** (`data/preprocess/chunk.py`) - Split into 512-sample sequences
   - Saves to `data/raw/art_chunks.pt`

3. **Transform** (`data/preprocess/transform.py`) - Compute STFT
   - Saves to `data/raw/train_stft.pt`, etc.

## Dependencies

- `vitaldb` - VitalDB signal library
- `torch` - Deep learning framework
- `polars` - Fast data processing
- `pandas`, `numpy`, `scipy`, `matplotlib`

See `pyproject.toml` for full dependency list.

## Project Structure

```
EE269Project/
├── data/
│   ├── load/load.py           # Download VitalDB records
│   ├── preprocess/
│   │   ├── chunk.py           # Create fixed-length chunks
│   │   ├── quantize.py        # Quantization utilities
│   │   └── transform.py       # STFT computation
│   ├── plot_vital_signs.py    # Visualization
│   └── visualize_chunks.py    # Chunk visualization
├── diffusion/                 # Diffusion model code
├── pyproject.toml            # Project configuration & dependencies
├── uv.lock                   # Locked dependency versions
└── README.md
```
