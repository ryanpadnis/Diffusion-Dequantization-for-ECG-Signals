# EE269Project

Diffusion Models for generating higher bit depth vital signs signals from arrhythmia data.

## Setup

### Requirements
- Python 3.10+
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
# Load VitalDB records from kaggle
uv run python data/load/download_kaggle_mitdb.py

# Chunk signals into sequences
uv run python data/load/create_chunks.py

# Process with quantizing and then STFT
uv run python data/preprocess/process.py

```

## Data Pipeline

1. **Download** (`uv run python data/load/download_kaggle_mitdb.py`) - Download VitalDB signals converted into CSVs from Kaggle
   - Saves to `data/data/raw/mitdb_kaggle`

2. **Chunk** (`v run python data/load/create_chunks.py`) - Split into 512-sample sequences
   - Saves to `data/data/processed/arrhythmia_chunks.pt`

3. **Process** (`data/preprocess/process.py`) - Quantize and compute STFT
   - Saves to `data/data/processed/signals_{description}.pt`, etc. 

## Dependencies

- `vitaldb` - VitalDB signal library
- `torch` - Deep learning framework
- `polars` - Fast data processing
- `pandas`, `numpy`, `scipy`, `matplotlib`

See `pyproject.toml` for full dependency list.

### Adding Dependencies

To add a new package (with automatic conflict resolution):
```bash
uv add <package>          # Add regular dependency
uv add --dev <package>    # Add dev-only dependency
```

To sync environment after modifying `pyproject.toml`:
```bash
uv sync
```

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
