"""
Diagnostic: print numeric stats for a single sample's STFT, 4-bit quantization indices,
and dequantized values using saved normalizer/quantizer metadata.
Run with: `uv run diffusion/analysis/quant_diag.py`
"""
from pathlib import Path
import pickle
import torch
import numpy as np

from data.preprocess.transform import get_transform
from data.preprocess.quantize import UniformQuantizer


def main():
    results_dir = Path('diffusion/results') / 'V1'
    normalizer_path = results_dir / 'data' / 'normalizer_params.pkl'
    if not normalizer_path.exists():
        raise FileNotFoundError(f"normalizer_params.pkl not found at {normalizer_path}")

    with open(normalizer_path, 'rb') as f:
        norm = pickle.load(f)

    pipeline_config = norm['pipeline_config'].copy()
    transform_type = pipeline_config.pop('transform', 'stft')

    transform = get_transform(transform_type, **pipeline_config, device=torch.device('cpu'))

    # Load first raw signal
    from diffusion.utils.config import DiffusionConfig
    raw_path = Path(DiffusionConfig.raw_data_path)
    raw = torch.load(raw_path)
    if isinstance(raw, dict):
        if raw.get('chunks') is not None:
            signals = raw.get('chunks')
        elif raw.get('signals') is not None:
            signals = raw.get('signals')
        else:
            raise KeyError('raw data dict missing "chunks" or "signals"')
    else:
        signals = raw

    sample = signals[0].unsqueeze(0)  # [1, L]

    # Compute spectrogram
    spec = transform.apply(sample)  # [1, F, T]
    spec = spec.squeeze(0)

    print('\n[Spec stats]')
    print('shape:', tuple(spec.shape))
    print('dtype:', spec.dtype)
    print('min/max:', float(spec.min()), float(spec.max()))
    print('mean:', float(spec.mean()))
    print('median:', float(spec.flatten().median()))
    print('p1/p99:', float(torch.quantile(spec, torch.tensor(0.01))), float(torch.quantile(spec, torch.tensor(0.99))))

    # Quantizer ranges from normalizer params
    cqmin = norm.get('cond_quant_range_min', norm['cond_min'])
    cqmax = norm.get('cond_quant_range_max', norm['cond_max'])
    rqmin = norm.get('real_quant_range_min', norm['real_min'])
    rqmax = norm.get('real_quant_range_max', norm['real_max'])
    print('\n[Quantizer ranges from normalizer_params]')
    print('cond_quant_range_min, max:', cqmin, cqmax)
    print('real_quant_range_min, max:', rqmin, rqmax)
    print('cond_min/max (norm):', norm['cond_min'], norm['cond_max'])
    print('real_min/max (norm):', norm['real_min'], norm['real_max'])

    # Create 4-bit quantizer using stored range
    q4 = UniformQuantizer(bits=4, range_min=cqmin, range_max=cqmax)
    indices4 = q4.quantize_indices(spec)
    deq4 = q4.decode(indices4)

    # Statistics for indices
    flat_idx = indices4.flatten().cpu().numpy().astype(np.int64)
    unique, counts = np.unique(flat_idx, return_counts=True)
    print('\n[4-bit indices] unique bins:', unique)
    print('counts per bin (nonzero):')
    for u, c in zip(unique, counts):
        print('  bin', int(u), 'count', int(c))

    print('\n[4-bit dequantized stats]')
    print('min/max:', float(deq4.min()), float(deq4.max()))
    print('mean:', float(deq4.mean()))
    print('n_zeros:', int((deq4 == 0).sum().item()))

    # Also show 16-bit quantization coverage
    q16 = UniformQuantizer(bits=16, range_min=rqmin, range_max=rqmax)
    indices16 = q16.quantize_indices(spec)
    unique16 = np.unique(indices16.flatten().cpu().numpy())
    print('\n[16-bit indices] unique_bins_count:', unique16.shape[0])

    # Print a small sample of spec values and corresponding 4-bit dequantized values
    print('\nSample values (freq bin 11, time 10..20):')
    try:
        r = 11
        cstart = 10
        cend = 20
        vals = spec[r, cstart:cend].cpu().numpy()
        idxs = indices4[r, cstart:cend].cpu().numpy()
        deqs = deq4[r, cstart:cend].cpu().numpy()
        for i, (v, idv, dqv) in enumerate(zip(vals, idxs, deqs)):
            print(f'  t={cstart+i}: spec={v:.6f}, idx4={int(idv)}, deq4={dqv:.6f}')
    except Exception:
        pass


if __name__ == '__main__':
    main()
