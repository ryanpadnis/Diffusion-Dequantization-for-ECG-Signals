"""Chunk ART signals into fixed sequences and store as PyTorch tensors."""

import polars as pl
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data import settings

def chunk_record(record_id, df, sequence_length=512, stride=256, torch_dtype=torch.float32):
    """Chunk a single record."""
    signal = df.filter(pl.col('record_id') == record_id)['ART'].to_numpy()
    chunks = []
    
    for i in range(0, len(signal) - sequence_length, stride):
        chunk = signal[i:i+sequence_length]
        chunks.append(torch.tensor(chunk, dtype=torch_dtype))
    
    return chunks

def chunk_signals(input_file, output_dir="data/raw", 
                  sequence_length=512, stride=256, 
                  torch_dtype=torch.float32, num_workers=4):
    """Chunk continuous signals and save as PyTorch tensors."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pl.read_parquet(input_file)
    record_ids = df['record_id'].unique().to_list()
    
    print(f"Chunking {len(record_ids)} records...")
    
    all_chunks = []
    all_record_ids = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(chunk_record, rid, df, sequence_length, stride, torch_dtype): rid 
            for rid in record_ids
        }
        
        with tqdm(as_completed(futures), total=len(record_ids)) as pbar:
            for future in pbar:
                record_id = futures[future]
                chunks = future.result()
                all_chunks.extend(chunks)
                all_record_ids.extend([record_id] * len(chunks))
    
    chunk_tensors = torch.stack(all_chunks)
    record_ids_list = torch.tensor(all_record_ids, dtype=torch.long)
    
    print(f"\n✓ Created {len(all_chunks)} chunks")
    print(f"  Tensor shape: {chunk_tensors.shape}")
    
    # Save entire dataset
    torch.save({
        'chunks': chunk_tensors, 
        'record_ids': record_ids_list
    }, output_path / "art_chunks.pt")
    
    print(f"  Saved to {output_path / 'art_chunks.pt'}")

if __name__ == "__main__":
    # Sizing rule: time_frames = floor((L - n_fft)/hop) + 1
    # Choose L = hop*(T-1) + n_fft to hit a target T exactly.
    # Use COLA-friendly hop = win_length//2 with n_fft=30, win_length=30 => hop=15
    # With n_fft=30, hop=15, target T=128 ⇒ L = 15*(128-1) + 30 = 1935
    chunk_signals(settings.ART_SIGNALS_FILE,
                  output_dir=settings.RAW_DIR,
                  sequence_length=1935,
                  stride=1935,
                  num_workers=4,
                  torch_dtype=torch.float32) # can specify precision here
