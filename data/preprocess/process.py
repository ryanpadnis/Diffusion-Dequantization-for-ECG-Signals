"""
Generate different quantizations and transform combinations for vital sign signals and save as torch
"""
import numpy as np
import torch
from pathlib import Path
from transform import *
from quantize import *
from tqdm import tqdm
from typing import List
from data import settings

# Local variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32


def process_signals(quantizers: List[str], transforms: List[str], bits_list: List[int],
                    input_file: str, output_dir: str,
                    torch_dtype=torch.float32, device=torch.device('cpu')):
    """Process signals with specified quantizers, bit depths, and transforms."""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = torch.load(input_path)
    print(f"Loaded data from {input_path} with keys: {list(data.keys())}")
    
    signals = data['chunks']  # Load chunks from art_chunks.pt
    
    for quantizer_name in quantizers:
        for transform_name in transforms:
            for bits in bits_list:
                qt_signals = []  # quantized + transformed
                nqt_signals = [] # non-quantized + transformed
                qt_inv_signals = []  # inverse of quantized+transformed
                nqt_inv_signals = [] # inverse of non-quantized+transformed
                for signal in tqdm(signals, desc=f"Processing with {quantizer_name} + {transform_name} @ {bits} bits"):
                    signal = signal.to(dtype=torch_dtype, device=device)

                    # Handle 11-bit raw data if needed (clip to 0-2047)
                    raw_min, raw_max = 0, 2047
                    signal_11bit = torch.clamp(signal, raw_min, raw_max)

                    # Quantize first
                    if quantizer_name == 'uniform':
                        quantizer = UniformQuantizer(bits=bits, range_min=signal_11bit.min().item(), range_max=signal_11bit.max().item())
                    else:
                        raise ValueError(f"Unknown quantizer: {quantizer_name}")
                    quantized_signal = quantizer.quantize(signal_11bit)

                    # Then transform (quantized)
                    if transform_name == 'stft':
                        transform = STFTTransform(
                            n_fft=256,
                            hop_length=64,
                            win_length=256,
                            center=True,
                            onesided=True,
                            torch_dtype=torch_dtype,
                            device=device,
                        )
                    elif transform_name == 'haar':
                        transform = HaarWaveletTransform(torch_dtype=torch_dtype, device=device)
                    elif transform_name == 'dft':
                        transform = DFTTransform(torch_dtype=torch_dtype, device=device)
                    else:
                        raise ValueError(f"Unknown transform: {transform_name}")

                    # Quantized + transformed
                    qt_transformed = transform.apply(quantized_signal.unsqueeze(0)).squeeze(0)
                    qt_signals.append(qt_transformed.cpu())

                    # Non-quantized + transformed
                    nqt_transformed = transform.apply(signal_11bit.unsqueeze(0)).squeeze(0)
                    nqt_signals.append(nqt_transformed.cpu())

                    # Inverse (reconstruction) for quantized+transformed
                    qt_inv = transform.inverse(qt_transformed.unsqueeze(0)).squeeze(0)
                    qt_inv_signals.append(qt_inv.cpu())

                    # Inverse (reconstruction) for non-quantized+transformed
                    nqt_inv = transform.inverse(nqt_transformed.unsqueeze(0)).squeeze(0)
                    nqt_inv_signals.append(nqt_inv.cpu())

                # Save all four outputs
                def save_tensor(tensor_list, suffix):
                    out_file = settings.output_path(f"{quantizer_name}{suffix}", transform_name, bits)
                    torch.save({'signals': torch.stack(tensor_list)}, out_file)
                    print(f"Saved {suffix} signals to {out_file}")
                    print(f"  Tensor shape: {torch.stack(tensor_list).shape}")
                    print("Datatype:", torch.stack(tensor_list).dtype)

                save_tensor(qt_signals, "_qt")
                save_tensor(nqt_signals, "_nqt")
                save_tensor(qt_inv_signals, "_qt_inv")
                save_tensor(nqt_inv_signals, "_nqt_inv")


if __name__ == "__main__":
    input_file = str(settings.PROCESSED_DIR / "arythmia_chunks.pt")
    output_dir = str(settings.PROCESSED_DIR)

    process_signals(
        settings.DEFAULT_QUANTIZERS,
        settings.DEFAULT_TRANSFORMS,
        settings.DEFAULT_BITS,
        input_file,
        output_dir,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
    )
