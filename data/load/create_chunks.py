"""Chunk MITDB signals into fixed sequences and store as PyTorch tensors."""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from data import settings


def _iter_kaggle_ekg_csvs(kaggle_dir: Path) -> list[Path]:
    kaggle_dir = Path(kaggle_dir)
    if not kaggle_dir.exists():
        raise FileNotFoundError(f"Kaggle MITDB directory not found: {kaggle_dir}")
    return sorted(kaggle_dir.glob("*_ekg.csv"))
    


def chunk_kaggle_ekg_csv(csv_path: Path, sequence_length=3600, stride=3600, lead_name="MLII", torch_dtype=torch.float32):
    """Load a Kaggle MITDB EKG CSV and chunk the specified lead."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path.name} shape: {df.shape}")
    lead_priority = [lead_name, "MLII", "V5", "V2", "V1"]
    found_lead = None
    for lead in lead_priority:
        if lead in df.columns:
            found_lead = lead
            break
    if not found_lead:
        print(f"  Skipping {csv_path.name}: no recognized lead found. Available: {df.columns}")
        return csv_path.stem, []
    print(f"  Using lead: {found_lead}")
    ecg = df[found_lead].values
    chunks = []
    for i in range(0, len(ecg) - sequence_length, stride):
        chunk = ecg[i:i + sequence_length]
        chunks.append(torch.tensor(chunk, dtype=torch_dtype))
    print(f"  Chunks: {len(chunks)}; Chunk shape: {chunks[0].shape if chunks else 'N/A'}")
    return csv_path.stem, chunks



def chunk_kaggle_ekg_signals(kaggle_dir: Path, sequence_length=3600, stride=3600, lead_name="MLII", torch_dtype=torch.float32):
    """Chunk all Kaggle MITDB EKG CSVs and print shapes."""
    import torch
    csvs = _iter_kaggle_ekg_csvs(kaggle_dir)
    print(f"Found {len(csvs)} EKG CSVs in {kaggle_dir}")
    arythmia_chunks = []
    non_arythmia_chunks = []
    arythmia_names = []
    non_arythmia_names = []
    for csv_path in tqdm(csvs, desc="Chunking Kaggle EKGs"):
        record_name, chunks = chunk_kaggle_ekg_csv(csv_path, sequence_length, stride, lead_name, torch_dtype)
        if not chunks:
            continue
        # Heuristic: if annotation file exists for this record, treat as arythmia
        ann_path = csv_path.parent / f"{csv_path.name.split('_')[0]}_annotations_1.csv"
        if ann_path.exists():
            arythmia_chunks.extend(chunks)
            arythmia_names.extend([record_name]*len(chunks))
        else:
            non_arythmia_chunks.extend(chunks)
            non_arythmia_names.extend([record_name]*len(chunks))
    # Save
    processed_dir = settings.PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    if arythmia_chunks:
        ar_chunks_tensor = torch.stack(arythmia_chunks)
        torch.save({"chunks": ar_chunks_tensor, "record_names": arythmia_names}, processed_dir/"arythmia_chunks.pt")
        print(f"Saved arythmia chunks: {ar_chunks_tensor.shape} to {processed_dir/'arythmia_chunks.pt'}")
    else:
        print("No arythmia chunks found.")
    if non_arythmia_chunks:
        nonar_chunks_tensor = torch.stack(non_arythmia_chunks)
        torch.save({"chunks": nonar_chunks_tensor, "record_names": non_arythmia_names}, processed_dir/"non_arythmia_chunks.pt")
        print(f"Saved non-arythmia chunks: {nonar_chunks_tensor.shape} to {processed_dir/'non_arythmia_chunks.pt'}")
    else:
        print("No non-arythmia chunks found.")
    # Print concatenated dataset shape
    all_chunks = arythmia_chunks + non_arythmia_chunks
    if all_chunks:
        all_tensor = torch.stack(all_chunks)
        print(f"Final concatenated dataset shape: {all_tensor.shape}")
    else:
        print("No chunks found in total.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", type=str, default=str(settings.RAW_DIR / "mitdb_kaggle"), help="Directory with Kaggle MITDB EKG CSVs")
    parser.add_argument("--lead", type=str, default="MLII", help="ECG lead name to use (e.g., MLII, V5)")
    parser.add_argument("--sequence-length", type=int, default=3600)
    parser.add_argument("--stride", type=int, default=3600)
    args = parser.parse_args()

    chunk_kaggle_ekg_signals(
        Path(args.kaggle_dir),
        sequence_length=args.sequence_length,
        stride=args.stride,
        lead_name=args.lead,
        torch_dtype=torch.float32,
    )

