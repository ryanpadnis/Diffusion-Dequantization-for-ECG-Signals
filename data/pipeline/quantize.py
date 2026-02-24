"""Quantize normalized signal chunks to a lower bit depth."""

import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from data.utils import settings
from data.utils.quantizers import UniformQuantizer, compute_range_from_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantize_chunks(
    input_file: Path,
    bits: int = 4,
    quantizer_name: str = "uniform",
    torch_dtype=torch.float32,
    device=DEVICE,
):
    input_file = Path(input_file)
    data = torch.load(input_file)
    signals = data["chunks"].to(dtype=torch_dtype)
    print(f"Loaded {signals.shape} from {input_file}")

    # Fit quantizer globally across all signals
    range_min, range_max = compute_range_from_tensor(signals)
    print(f"Global range: [{range_min:.4f}, {range_max:.4f}]")

    if quantizer_name == "uniform":
        quantizer = UniformQuantizer(bits=bits, range_min=range_min, range_max=range_max)
    else:
        raise ValueError(f"Unknown quantizer: {quantizer_name}. Available: uniform")

    quantized = []
    for signal in tqdm(signals, desc=f"Quantizing ({quantizer_name}, {bits}-bit)"):
        signal = signal.to(device=device)
        quantized.append(quantizer.quantize(signal).cpu())

    quantized_tensor = torch.stack(quantized)

    # Save quantized chunks
    output_dir = settings.PROCESSED_DIR / "quantized"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem  # e.g. "arrhythmia_chunks"
    out_file = output_dir / f"{stem}_{quantizer_name}_{bits}bit.pt"
    torch.save({"chunks": quantized_tensor, "record_names": data["record_names"]}, out_file)
    print(f"Saved quantized chunks: {quantized_tensor.shape} → {out_file}")

    # Save quantizer params so we can reconstruct or compare later
    params_file = output_dir / f"{stem}_{quantizer_name}_{bits}bit_params.json"
    with open(params_file, "w") as f:
        json.dump({
            "quantizer": quantizer_name,
            "bits": bits,
            "levels": quantizer.levels,
            "range_min": range_min,
            "range_max": range_max,
            "step_size": quantizer.step_size,
        }, f, indent=2)
    print(f"Saved quantizer params → {params_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize normalized ECG chunks.")
    parser.add_argument("--input", type=str, required=True, help="Chunks .pt filename (e.g. arrhythmia_chunks.pt); resolved relative to PROCESSED_DIR/chunks")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--quantizer", type=str, default="uniform", choices=["uniform"])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = settings.PROCESSED_DIR / "chunks" / input_path.name

    quantize_chunks(
        input_file=input_path,
        bits=args.bits,
        quantizer_name=args.quantizer,
    )