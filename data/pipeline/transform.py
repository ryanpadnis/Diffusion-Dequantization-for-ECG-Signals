"""Apply a signal transform to a chunk file (quantized or clean)."""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from data.utils import settings
from data.utils.transforms import STFTTransform, DFTTransform, HaarWaveletTransform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(transform_name: str, torch_dtype, device):
    if transform_name == "stft":
        return STFTTransform(
            n_fft=256,
            hop_length=64,
            win_length=256,
            center=True,
            onesided=True,
            torch_dtype=torch_dtype,
            device=device,
        )
    elif transform_name == "haar":
        return HaarWaveletTransform(torch_dtype=torch_dtype, device=device)
    elif transform_name == "dft":
        return DFTTransform(torch_dtype=torch_dtype, device=device)
    else:
        raise ValueError(f"Unknown transform: {transform_name}. Available: stft, haar, dft")


def transform_chunks(
    input_file: Path,
    transform_name: str = "stft",
    torch_dtype=torch.float32,
    device=DEVICE,
):
    input_file = Path(input_file)
    data = torch.load(input_file)
    signals = data["chunks"].to(dtype=torch_dtype)
    print(f"Loaded {signals.shape} from {input_file}")

    transform = build_transform(transform_name, torch_dtype, device)

    transformed = []
    for signal in tqdm(signals, desc=f"Transforming ({transform_name})"):
        signal = signal.to(device=device)
        out = transform.apply(signal.unsqueeze(0)).squeeze(0)
        transformed.append(out.cpu())

    transformed_tensor = torch.stack(transformed)

    output_dir = settings.PROCESSED_DIR / "transformed"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_file.stem  # preserves quantization info if present, e.g. "arrhythmia_chunks_uniform_4bit"
    out_file = output_dir / f"{stem}_{transform_name}.pt"
    torch.save({"signals": transformed_tensor, "record_names": data["record_names"]}, out_file)
    print(f"Saved transformed signals: {transformed_tensor.shape} → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform ECG chunks (STFT, DFT, Haar).")
    parser.add_argument("--input", type=str, required=True, help="Quantized .pt filename (e.g. arrhythmia_chunks_uniform_4bit.pt); resolved relative to PROCESSED_DIR/chunks/quantized")
    parser.add_argument("--transform", type=str, default="stft", choices=["stft", "haar", "dft"])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = settings.PROCESSED_DIR / "quantized" / input_path.name

    transform_chunks(
        input_file=input_path,
        transform_name=args.transform,
    )