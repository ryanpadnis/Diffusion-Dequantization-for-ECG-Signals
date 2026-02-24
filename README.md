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

1. **Download** (`uv run python data/pipeline/download.py`) - Download VitalDB signals converted into CSVs from Kaggle
   - Saves to `data/data/raw/mitdb_kaggle`

2. **Normalize** (`uv run python data/pipeline/normalize.py`) - Normalizes signal by selected method for selected lead from each csv. 
   - Saves to `data/data/processed/normalized` each as its own .pt

3. **Chunk** (`uv run python data/pipeline/chunk.py`) - Split into 3600-sample (default, but can specify otherwise) sequences
   - Saves to `data/data/processed/chunks`
   - Can visualize these by running `uv run python data/pipeline/visualize.py`

4. **Quantize** (`uv run python data/pipeline/quantize.py`) - Quantize by specified method
   - Saves to `data/data/processed/quantized`

5. **Transform** (`uv run python data/pipeline/transform.py`) - Perform specified transform on input quantized data
   - Saves to `data/data/processed/transformed`

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
