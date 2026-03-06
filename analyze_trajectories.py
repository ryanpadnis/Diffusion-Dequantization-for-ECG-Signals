#!/usr/bin/env python3
"""Simple script to analyze all_trajectories.pt files using metrics.py"""

from pathlib import Path
import torch
import json
import argparse
from diffusion.utils.metrics import compute_energy_metrics, print_energy_metrics


def load_trajectories(traj_path: Path) -> dict:
    """Load all_trajectories.pt and extract tensors into a data dict.
    
    Also loads companion files from the same directory:
      - condition_raw.pt (for 16-bit ground truth)
      - condition_spec.pt (for spectrogram-based analysis)
    """
    if not traj_path.exists():
        raise FileNotFoundError(f"File not found: {traj_path}")
    
    # Load the .pt file
    obj = torch.load(traj_path, map_location='cpu')
    
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict, got {type(obj)}")
    
    print(f"Loaded {traj_path.name}")
    print(f"Available keys: {list(obj.keys())}\n")
    
    # Build data dict for metrics.py (mirrors analysis.py structure)
    data = {}
    parent = traj_path.parent
    
    # Extract time-domain signals from all_trajectories.pt
    if 'time_domain' in obj:
        data['trajectories'] = [obj['time_domain']]
    elif 'spectrograms_denorm_mag' in obj:
        # If only spectrograms saved, we'll use them but note the limitation
        data['trajectories'] = []
        print("Note: Only spectrograms found in all_trajectories.pt, time-domain metrics will be skipped\n")
    
    # Load condition_raw.pt (this is the 16-bit ground truth / target)
    condition_raw_path = parent / 'condition_raw.pt'
    if condition_raw_path.exists():
        try:
            cond_raw = torch.load(condition_raw_path, map_location='cpu')
            if isinstance(cond_raw, dict):
                # Extract tensor from dict (try common keys)
                cond_raw = cond_raw.get('raw') or cond_raw.get('signal') or next(iter(cond_raw.values()))
            data['16bit_gt'] = cond_raw  # condition_raw IS the target
            print(f"✓ Loaded ground truth from condition_raw.pt")
        except Exception as e:
            print(f"✗ Error loading condition_raw.pt: {e}")
    
    # Load condition_spec.pt (spectrogram representation)
    condition_spec_path = parent / 'condition_spec.pt'
    if condition_spec_path.exists():
        try:
            cond_spec = torch.load(condition_spec_path, map_location='cpu')
            if isinstance(cond_spec, dict):
                cond_spec = cond_spec.get('magnitude') or cond_spec.get('mag') or next(iter(cond_spec.values()))
            data['cond_spec'] = cond_spec
            print(f"✓ Loaded spectrogram from condition_spec.pt")
        except Exception as e:
            print(f"✗ Error loading condition_spec.pt: {e}")
    
    # For 4-bit condition, try to find it (used for baseline comparison)
    # First check if it's embedded in all_trajectories
    if 'spectrograms_norm' in obj:
        # The normalized version might represent the 4-bit quantized condition
        data['gen_mag_norm'] = obj['spectrograms_norm']
    
    # Try to find explicit 4-bit files
    cond_4bit_candidates = [
        'cond_time_4bit.pt', 'condition_4bit.pt', 'condition_4bit_time.pt', 'condition_4bit_quantized.pt'
    ]
    for fname in cond_4bit_candidates:
        fpath = parent / fname
        if fpath.exists():
            try:
                c4 = torch.load(fpath, map_location='cpu')
                if isinstance(c4, dict):
                    c4 = c4.get('signal') or next(iter(c4.values()))
                data['4bit'] = c4
                print(f"✓ Loaded 4-bit condition from {fname}")
            except Exception as e:
                print(f"✗ Error loading {fname}: {e}")
            break
    
    # Extract spectrograms if present in all_trajectories.pt
    if 'spectrograms_denorm_mag' in obj:
        data['gen_mag'] = obj['spectrograms_denorm_mag']
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Analyze all_trajectories.pt files using energy metrics'
    )
    parser.add_argument(
        'trajectory_file',
        type=str,
        help='Path to all_trajectories.pt file'
    )
    parser.add_argument(
        '--n-bands',
        type=int,
        default=4,
        help='Number of frequency bands for energy analysis (default: 4)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.10,
        help='NEF threshold for quality assessment (default: 0.10 = 10%%)'
    )
    parser.add_argument(
        '--save-json',
        type=str,
        default=None,
        help='Optional: save metrics to JSON file'
    )
    
    args = parser.parse_args()
    
    traj_path = Path(args.trajectory_file)
    
    # Load data
    try:
        data = load_trajectories(traj_path)
    except Exception as e:
        print(f"Error loading trajectories: {e}")
        return
    
    # Check what we have
    has_time = bool(data.get('trajectories'))
    has_condition = torch.is_tensor(data.get('4bit'))
    has_target = torch.is_tensor(data.get('16bit_gt'))
    has_mag = torch.is_tensor(data.get('gen_mag'))
    has_cond_spec = torch.is_tensor(data.get('cond_spec'))
    
    print("\n" + "="*60)
    print("LOADED DATA SUMMARY")
    print("="*60)
    print(f"✓ Time-domain trajectories: {has_time}")
    print(f"✓ Ground truth (condition_raw): {has_target}")
    print(f"✓ 4-bit condition:          {has_condition}")
    print(f"✓ Generated magnitude STFT: {has_mag}")
    print(f"✓ Condition spectrogram:    {has_cond_spec}")
    
    # Print tensor shapes for debugging
    if has_time:
        traj = data.get('trajectories')[0]
        print(f"  └─ trajectory shape: {traj.shape}")
    if has_target:
        print(f"  └─ target (16-bit) shape: {data['16bit_gt'].shape}")
    if has_condition:
        print(f"  └─ condition (4-bit) shape: {data['4bit'].shape}")
    if has_mag:
        print(f"  └─ generated mag shape: {data['gen_mag'].shape}")
    if has_cond_spec:
        print(f"  └─ condition spec shape: {data['cond_spec'].shape}")
    print("="*60 + "\n")
    
    # Try to compute energy metrics if we have time-domain data
    if has_time and has_condition and has_target:
        print("ENERGY METRICS ANALYSIS")
        print("="*60)
        results = compute_energy_metrics(
            data,
            n_bands=args.n_bands,
            threshold=args.threshold,
        )
        
        if results:
            print_energy_metrics(results)
            
            # Save to JSON if requested
            if args.save_json:
                # Convert tensors to lists for JSON serialization
                json_data = {
                    'threshold': results['threshold'],
                    'gt_band_fracs': results['gt_band_fracs'],
                    '4bit_vs_gt': {
                        'nef': float(results['4bit_vs_gt'].nef),
                        'energy_ratio': float(results['4bit_vs_gt'].energy_ratio),
                        'envelope_nef': float(results['4bit_vs_gt'].envelope_nef),
                        'band_nefs': [float(x) for x in results['4bit_vs_gt'].band_nefs],
                    },
                    'trajectories': []
                }
                
                for entry in results['trajectories']:
                    m = entry['metrics']
                    json_data['trajectories'].append({
                        'trajectory_idx': entry['trajectory_idx'],
                        'improvement_vs_4bit_pct': float(entry['improvement_vs_4bit_pct']),
                        'nef': float(m.nef),
                        'energy_ratio': float(m.energy_ratio),
                        'envelope_nef': float(m.envelope_nef),
                        'band_nefs': [float(x) for x in m.band_nefs],
                    })
                
                with open(args.save_json, 'w') as f:
                    json.dump(json_data, f, indent=2)
                print(f"\n✓ Metrics saved to: {args.save_json}")
        else:
            print("Could not compute metrics (data mismatch)")
    else:
        print("CANNOT COMPUTE ENERGY METRICS")
        print("="*60)
        print("Energy metrics require: time-domain + 16-bit target + generated signal")
        print("\nCurrently available:")
        if not has_time:
            print("  ✗ Time-domain trajectories")
        else:
            print("  ✓ Time-domain trajectories")
        if not has_target:
            print("  ✗ Target/ground-truth (look for condition_raw.pt)")
        else:
            print("  ✓ Target/ground-truth")
        if not has_condition:
            print("  ✗ Baseline 4-bit condition (optional for improvement %)")
        else:
            print("  ✓ Baseline 4-bit condition")
        
        print("\nNote: If analyzing a sample directory with all_trajectories.pt,")
        print("ensure these companion files exist in the same directory:")
        print("  - condition_raw.pt (16-bit ground truth)")
        print("  - condition_spec.pt (spectrogram form, optional)")
    
    # Print spectrogram info if available
    if has_mag:
        mag = data['gen_mag']
        print(f"\nGenerated magnitude spectrogram shape: {mag.shape}")
        print(f"  (typically [n_trajectories, n_freq_bins, n_time_frames])")
    
    if has_cond_spec:
        spec = data['cond_spec']
        print(f"\nCondition spectrogram shape: {spec.shape}")


if __name__ == '__main__':
    main()
