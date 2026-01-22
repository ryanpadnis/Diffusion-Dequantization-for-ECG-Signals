"""Load VitalDB ART signals and save to Parquet with Polars."""
import vitaldb
import polars as pl
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def load_art_record(record_id, sampling_rate=1/100): #feel free top adjust the sampling rate
    """Load ART signal from a single record."""
    try:
        vf = vitaldb.VitalFile(record_id, ['SNUADC/ART'])
        samples = vf.to_numpy(['SNUADC/ART'], sampling_rate)
        mask = ~np.isnan(samples[:, 0])
        art_signal = samples[mask, 0]
        
        if len(art_signal) > 0:
            return {
                'ART': art_signal,
                'record_id': [record_id] * len(art_signal),
                'sample_index': np.arange(len(art_signal))
            }
    except:
        pass
    return None

def load_and_save_art_signals(record_ids, output_dir="data/data/raw", num_workers=4):
    """Load ART signals in parallel and save with Polars lazy evaluation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_art_record, rid): rid for rid in record_ids}
        
        with tqdm(as_completed(futures), total=len(record_ids), desc="Loading ART") as pbar:
            for future in pbar:
                result = future.result()
                if result is not None:
                    results.append(result)
    
    if results:
        # Convert to Polars lazy DataFrames and concatenate
        lazy_dfs = [
            pl.DataFrame(r).lazy() for r in results
        ]
        df = pl.concat(lazy_dfs)
        
        # streaming also for speed
        output_file = output_path / "art_signals.parquet"
        df.sink_parquet(str(output_file))
        
        # Get stats (need to collect for stats)
        collected = pl.read_parquet(str(output_file))
        print(f"\n✓ Saved {len(collected)} samples")
        print(f"  Records: {collected['record_id'].n_unique()}")
        print(f"  Range: {collected['ART'].min():.2f} to {collected['ART'].max():.2f} mmHg")
        print(f"  File: {output_file}")

if __name__ == "__main__":
    #range refers to the number of surgeries in VitalDB we want to access
    load_and_save_art_signals(range(1, 100), output_dir="../../data/data/raw")


