"""Chunk normalized MITDB signals into fixed sequences and store as PyTorch tensors."""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from data import settings


def _iter_normalized_pts(normalized_dir: Path) -> list[Path]:
    normalized_dir = Path(normalized_dir)
    if not normalized_dir.exists():
        raise FileNotFoundError(f"Normalized directory not found: {normalized_dir}")
    return sorted(normalized_dir.glob("*_normalized.pt"))


def chunk_normalized_signal(pt_path: Path, sequence_length: int = 3600, stride: int = 3600) -> tuple[str, list[torch.Tensor]]:
    """Load a normalized .pt signal and chunk it into fixed-length tensors."""
    signal = torch.load(pt_path)  # 1D tensor, shape [N]
    record_name = pt_path.stem.replace("_normalized", "")

    chunks = []
    for i in range(0, len(signal) - sequence_length, stride):
        chunks.append(signal[i : i + sequence_length])

    print(f"  {record_name}: {len(signal)} samples → {len(chunks)} chunks of {sequence_length}")
    return record_name, chunks


def chunk_normalized_signals(
    normalized_dir: Path,
    raw_dir: Path,
    sequence_length: int = 3600,
    stride: int = 3600,
):
    """Chunk all normalized .pt signals and save arrhythmia/non-arrhythmia tensors."""
    pt_files = _iter_normalized_pts(normalized_dir)
    print(f"Found {len(pt_files)} normalized signals in {normalized_dir}")

    arrhythmia_chunks, arrhythmia_names = [], []
    non_arrhythmia_chunks, non_arrhythmia_names = [], []

    for pt_path in tqdm(pt_files, desc="Chunking"):
        record_name, chunks = chunk_normalized_signal(pt_path, sequence_length, stride)
        if not chunks:
            continue

        # Derive the original record ID (e.g. "100" from "100_ekg_normalized")
        record_id = record_name.split("_")[0]
        ann_path = Path(raw_dir) / f"{record_id}_annotations_1.csv"

        if ann_path.exists():
            arrhythmia_chunks.extend(chunks)
            arrhythmia_names.extend([record_name] * len(chunks))
        else:
            non_arrhythmia_chunks.extend(chunks)
            non_arrhythmia_names.extend([record_name] * len(chunks))

    output_dir = settings.PROCESSED_DIR / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    if arrhythmia_chunks:
        ar_tensor = torch.stack(arrhythmia_chunks)
        torch.save({"chunks": ar_tensor, "record_names": arrhythmia_names}, output_dir / "arrhythmia_chunks.pt")
        print(f"Saved arrhythmia chunks: {ar_tensor.shape} → {output_dir / 'arrhythmia_chunks.pt'}")
    else:
        print("No arrhythmia chunks found.")

    if non_arrhythmia_chunks:
        nonar_tensor = torch.stack(non_arrhythmia_chunks)
        torch.save({"chunks": nonar_tensor, "record_names": non_arrhythmia_names}, output_dir / "non_arrhythmia_chunks.pt")
        print(f"Saved non-arrhythmia chunks: {nonar_tensor.shape} → {output_dir / 'non_arrhythmia_chunks.pt'}")
    else:
        print("No non-arrhythmia chunks found.")

    total = arrhythmia_chunks + non_arrhythmia_chunks
    if total:
        print(f"Final dataset shape: {torch.stack(total).shape}")
    else:
        print("No chunks produced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk normalized MITDB ECG signals.")
    parser.add_argument("--normalized-dir", type=str, default=str(settings.PROCESSED_DIR / "normalized"))
    parser.add_argument("--raw-dir", type=str, default=str(settings.RAW_DIR / "mitdb_kaggle"), help="Raw CSV dir, used to check for annotation files")
    parser.add_argument("--sequence-length", type=int, default=3600)
    parser.add_argument("--stride", type=int, default=3600)
    args = parser.parse_args()

    chunk_normalized_signals(
        normalized_dir=Path(args.normalized_dir),
        raw_dir=Path(args.raw_dir),
        sequence_length=args.sequence_length,
        stride=args.stride,
    )