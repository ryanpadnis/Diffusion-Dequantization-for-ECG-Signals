"""
Generate different quantizations and transform combinations for vital sign signals and save as torch
"""
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))  # For transform, quantize
sys.path.insert(0, str(current_dir.parent.parent))  # For settings

from transform import *
from quantize import *
from tqdm import tqdm
from typing import List
import settings

# Local variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32


def process_signals(quantizers: List[str], transforms: List[str],
                    input_file: str, output_dir: str,
                    torch_dtype=torch.float32, device=torch.device('cpu')):
    """Process signals with specified quantizers and transforms."""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = torch.load(input_path)
    print(f"Loaded data from {input_path} with keys: {list(data.keys())}")
    
    signals = data['chunks']  # Load chunks from art_chunks.pt
    
    for quantizer_name in quantizers:
        for transform_name in transforms:
            processed_signals = []
            for signal in tqdm(signals, desc=f"Processing with {quantizer_name} + {transform_name}"):
                signal = signal.to(dtype=torch_dtype, device=device)

                # Apply transform
                if transform_name == 'stft':
                    transform = STFTTransform(torch_dtype=torch_dtype, device=device)
                elif transform_name == 'haar':
                    transform = HaarWaveletTransform(torch_dtype=torch_dtype, device=device)
                else:
                    raise ValueError(f"Unknown transform: {transform_name}")
                
                transformed_signal = transform.apply(signal)
                
                if quantizer_name == 'uniform':
                    quantizer = UniformQuantizer(bits=8, range_min=transformed_signal.min().item(), range_max=transformed_signal.max().item())
                else:
                    raise ValueError(f"Unknown quantizer: {quantizer_name}")
                
                quantized_signal = quantizer.quantize(transformed_signal)
                
                processed_signals.append(quantized_signal.cpu())
            
            # Save processed signals
            output_file = output_path / f"signals_{quantizer_name}_{transform_name}.pt"
            torch.save({'signals': torch.stack(processed_signals)}, output_file)
            print(f"Saved processed signals to {output_file}")


if __name__ == "__main__":
    process_signals(['uniform'], ['stft'], str(settings.ART_CHUNKS_FILE), str(settings.PROCESSED_DIR), 
                    torch_dtype=TORCH_DTYPE, device=DEVICE)
