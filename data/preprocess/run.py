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
                processed_signals = []
                for signal in tqdm(signals, desc=f"Processing with {quantizer_name} + {transform_name} @ {bits} bits"):
                    signal = signal.to(dtype=torch_dtype, device=device)

                    # Apply transform (freq bins = n_fft/2 + 1 when onesided=True)
                    if transform_name == 'stft':
                        transform = STFTTransform(torch_dtype=torch_dtype, device=device)
                    elif transform_name == 'haar':
                        transform = HaarWaveletTransform(torch_dtype=torch_dtype, device=device)
                    else:
                        raise ValueError(f"Unknown transform: {transform_name}")
                    
                    transformed_signal = transform.apply(signal)
                    
                    if quantizer_name == 'uniform':
                        quantizer = UniformQuantizer(bits=bits, range_min=transformed_signal.min().item(), range_max=transformed_signal.max().item())
                    else:
                        raise ValueError(f"Unknown quantizer: {quantizer_name}")
                    
                    quantized_signal = quantizer.quantize(transformed_signal)
                    
                    processed_signals.append(quantized_signal.cpu())
            
            # Save processed signals
            output_file = settings.output_path(quantizer_name, transform_name, bits)
            torch.save({'signals': torch.stack(processed_signals)}, output_file)
            print(f"Saved processed signals to {output_file}")


if __name__ == "__main__":
    process_signals(settings.DEFAULT_QUANTIZERS, settings.DEFAULT_TRANSFORMS, settings.DEFAULT_BITS,
                    str(settings.ART_CHUNKS_FILE), str(settings.PROCESSED_DIR), 
                    torch_dtype=TORCH_DTYPE, device=DEVICE)
