"""Normalize MITDB ECG signals per-recording and save as PyTorch tensors with stats."""

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from data import settings


SUPPORTED_LEADS = ["MLII", "V5", "V2", "V1"]


def _iter_kaggle_ekg_csvs(kaggle_dir: Path) -> list[Path]:
    kaggle_dir = Path(kaggle_dir)
    if not kaggle_dir.exists():
        raise FileNotFoundError(f"Directory not found: {kaggle_dir}")
    return sorted(kaggle_dir.glob("*_ekg.csv"))


def normalize_signal(signal: torch.Tensor, method: str = "zscore") -> tuple[torch.Tensor, dict]:
    """
    Normalize a 1D signal tensor.

    Methods:
        zscore  — mean=0, std=1. Good default for ECG, handles inter-patient gain variance.
        robust  — median=0, IQR-scaled. Better when recordings have artifact spikes.
        minmax  — scales to [-1, 1]. Sensitive to outliers; generally avoid for raw ECG.

    Returns:
        normalized signal tensor, dict of stats needed to invert the normalization
    """
    if method == "zscore":
        mean = signal.mean()
        std = signal.std()
        std = std if std > 1e-8 else torch.tensor(1.0)  # guard against flat signal
        normalized = (signal - mean) / std
        stats = {"method": method, "mean": mean.item(), "std": std.item()}

    elif method == "robust":
        median = signal.median()
        q75 = torch.quantile(signal, 0.75)
        q25 = torch.quantile(signal, 0.25)
        iqr = q75 - q25
        iqr = iqr if iqr > 1e-8 else torch.tensor(1.0)
        normalized = (signal - median) / iqr
        stats = {"method": method, "median": median.item(), "iqr": iqr.item()}

    elif method == "minmax":
        lo = signal.min()
        hi = signal.max()
        rng = hi - lo
        rng = rng if rng > 1e-8 else torch.tensor(1.0)
        normalized = 2.0 * (signal - lo) / rng - 1.0  # -> [-1, 1]
        stats = {"method": method, "min": lo.item(), "max": hi.item()}

    else:
        raise ValueError(f"Unknown normalization method: {method}. Choose zscore, robust, or minmax.")

    return normalized, stats


def normalize_kaggle_ekg_csv(
    csv_path: Path,
    lead_name: str = "MLII",
    method: str = "zscore",
    torch_dtype=torch.float32,
) -> tuple[str, torch.Tensor | None, dict | None]:
    """Load one CSV, select lead, normalize. Returns (record_name, tensor, stats)."""
    df = pd.read_csv(csv_path)

    if lead_name not in df.columns:
        print(f"  Skipping {csv_path.name}: lead '{lead_name}' not found. Available: {list(df.columns)}")
        return csv_path.stem, None, None

    signal = torch.tensor(df[lead_name].values, dtype=torch_dtype)
    normalized, stats = normalize_signal(signal, method=method)
    stats["lead"] = lead_name
    stats["n_samples"] = len(signal)

    return csv_path.stem, normalized, stats
    

def normalize_kaggle_ekg_signals(
    kaggle_dir: Path,
    output_dir: Path,
    lead_name: str = "MLII",
    method: str = "zscore",
    torch_dtype=torch.float32,
):
    """
    Normalize all EKG CSVs and save:
        <output_dir>/<record_name>_normalized.pt   — 1D float tensor, shape [N]
        <output_dir>/normalization_stats.json       — all per-recording stats
    """
    csvs = _iter_kaggle_ekg_csvs(kaggle_dir)
    print(f"Found {len(csvs)} EKG CSVs in {kaggle_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for csv_path in tqdm(csvs, desc="Normalizing"):
        record_name, normalized, stats = normalize_kaggle_ekg_csv(
            csv_path, lead_name=lead_name, method=method, torch_dtype=torch_dtype
        )
        if normalized is None:
            continue

        # Save normalized signal as .pt
        out_path = output_dir / f"{record_name}_normalized.pt"
        torch.save(normalized, out_path)
        all_stats[record_name] = stats

        print(f"  {record_name}: {stats}")

    # Save all stats together — needed later for inversion and for Lloyd-Max fitting
    stats_path = output_dir / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved {len(all_stats)} normalized signals to {output_dir}")
    print(f"Normalization stats saved to {stats_path}")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize MITDB ECG signals.")
    parser.add_argument("--kaggle-dir", type=str, default=str(settings.RAW_DIR / "mitdb_kaggle"))
    parser.add_argument("--output-dir", type=str, default=str(settings.PROCESSED_DIR / "normalized"))
    parser.add_argument("--lead", type=str, default="MLII", help="ECG lead to extract (MLII, V5, V2, V1)")
    parser.add_argument(
        "--method",
        type=str,
        default="zscore",
        choices=["zscore", "robust", "minmax"],
        help="Normalization method. zscore recommended for ECG.",
    )
    args = parser.parse_args()

    normalize_kaggle_ekg_signals(
        kaggle_dir=Path(args.kaggle_dir),
        output_dir=Path(args.output_dir),
        lead_name=args.lead,
        method=args.method,
    )