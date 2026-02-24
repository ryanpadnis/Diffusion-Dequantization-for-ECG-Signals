"""Quantization consistency diagnostics.

This script is meant to be *fast* and not require a GPU or running the model.
It validates that:
  - time-domain quantization ranges are present
  - saved sample artifacts match re-quantization with the saved ranges
  - per-run normalizer_params (if present) agrees with sample quant_params

Run from project root, e.g.:

  uv run python -m diffusion.analysis.quant_diag --version V2 --run-id 20260213_151839 --sampler-type ddpm --sample-idx 0

"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import torch

from data.utils.transforms import get_transform
from data.utils.quantizers import UniformQuantizer
from diffusion.utils.quant_debug import summarize_tensor, summarize_uniform_quantizer, summarize_quantization_usage


def _load_optional_pickle(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[quant_diag] Failed to load {path}: {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default='diffusion/results')
    p.add_argument('--version', type=str, default='V2')
    p.add_argument('--run-id', type=str, required=True)
    p.add_argument('--sampler-type', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    p.add_argument('--sample-idx', type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.results_dir) / args.version / args.run_id
    sample_dir = run_dir / 'samples' / args.sampler_type / f'sample_{int(args.sample_idx)}'
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample dir not found: {sample_dir}")

    print(f"[quant_diag] run_dir: {run_dir}")
    print(f"[quant_diag] sample_dir: {sample_dir}")

    # Load config (for pipeline_config)
    cfg_path = run_dir / 'config.pkl'
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)
    pipeline_config = (cfg.get('pipeline_config') or {}).copy()
    transform_type = pipeline_config.pop('transform', 'stft')

    # Load per-run normalizer params if available
    normalizer_path = run_dir / 'data' / 'normalizer_params.pkl'
    norm = _load_optional_pickle(normalizer_path)
    if norm is None:
        print(f"[quant_diag] NOTE: normalizer_params.pkl not found at {normalizer_path}")
        print("[quant_diag]       This should exist for training/sampling consistency.")
    else:
        print(f"[quant_diag] Loaded normalizer_params.pkl")
        for k in ['cond_time_range_min', 'cond_time_range_max', 'real_time_range_min', 'real_time_range_max', 'cond_bits', 'real_bits', 'mag_normalization']:
            if k in norm:
                print(f"  - {k}: {norm.get(k)}")

    # Load sample artifacts
    raw = torch.load(sample_dir / 'condition_raw.pt', map_location='cpu')['raw_signal']  # [1, L]
    # Optional artifacts (older runs may have these)
    t4_path = sample_dir / 'condition_4bit_time.pt'
    t16_path = sample_dir / 'ground_truth_16bit_time.pt'
    t4_obj = torch.load(t4_path, map_location='cpu') if t4_path.exists() else None
    t16_obj = torch.load(t16_path, map_location='cpu') if t16_path.exists() else None

    t4 = (t4_obj.get('time_domain_4bit') if isinstance(t4_obj, dict) else None)
    t16 = (t16_obj.get('time_domain_16bit') if isinstance(t16_obj, dict) else None)
    qp = (t4_obj.get('quant_params') if isinstance(t4_obj, dict) else None) or {}

    print("\n[Time-domain tensors]")
    summarize_tensor(raw, 'raw_time')
    if t4 is not None:
        summarize_tensor(t4, 'cond_time_saved_4bit')
    if t16 is not None:
        summarize_tensor(t16, 'target_time_saved')

    # Determine per-sample quant range directly from raw (no global params).
    cond_bits = int(cfg.get('bit_size', 4))
    real_bits = int(cfg.get('real_bit_size', 16))

    lower_pct = float(cfg.get('quantile_clip_lower', 0.0))
    upper_pct = float(cfg.get('quantile_clip_upper', 100.0))
    x = raw.detach().flatten()
    if lower_pct <= 0.0 and upper_pct >= 100.0:
        lo = float(x.min().item())
        hi = float(x.max().item())
    else:
        lo = float(torch.quantile(x, torch.tensor(lower_pct / 100.0)).item())
        hi = float(torch.quantile(x, torch.tensor(upper_pct / 100.0)).item())
    peak = max(abs(lo), abs(hi))
    if peak <= 0:
        peak = float(x.abs().max().item())
    if peak <= 0:
        peak = 1.0

    tmin = -peak
    tmax = peak
    rtmin = -peak
    rtmax = peak

    q4 = UniformQuantizer(bits=cond_bits, range_min=tmin, range_max=tmax)
    qreal = UniformQuantizer(bits=real_bits, range_min=rtmin, range_max=rtmax)

    print("\n[Time-domain quantizers]")
    summarize_uniform_quantizer(q4, f'cond_time_q{cond_bits}')
    summarize_quantization_usage(raw, q4, f'cond_time_q{cond_bits}_usage')
    summarize_uniform_quantizer(qreal, f'real_time_q{real_bits}')
    summarize_quantization_usage(raw, qreal, f'real_time_q{real_bits}_usage')

    # Compute per-sample 4-bit waveform for diagnostics.
    t4_re = q4.quantize(raw)
    summarize_tensor(t4_re, 'cond_time_recomputed_4bit')

    if t4 is not None:
        mse_t4 = float(torch.mean((t4_re - t4) ** 2).item())
        max_abs_t4 = float((t4_re - t4).abs().max().item())
        print("\n[Check: saved 4-bit waveform matches q4(raw)]")
        print(f"  mse={mse_t4:.6g} max_abs={max_abs_t4:.6g}")

    # Spectrogram sanity: ensure transform config matches
    transform = get_transform(transform_type, **pipeline_config, device=torch.device('cpu'))
    mag_raw = transform.apply(raw)
    mag_t4 = transform.apply(t4_re)
    mag_t16 = transform.apply(t16) if t16 is not None else None

    print("\n[Spectrogram magnitudes]")
    summarize_tensor(mag_raw, 'mag(raw)')
    summarize_tensor(mag_t4, 'mag(time_4bit)')
    if mag_t16 is not None:
        summarize_tensor(mag_t16, 'mag(time_target)')

    # Compare normalizer vs sample quant params
    if norm is not None:
        def _cmp(name: str, a: float, b: float) -> None:
            diff = abs(float(a) - float(b))
            print(f"  - {name}: sample={float(a):.6g} normalizer={float(b):.6g} |diff|={diff:.6g}")

        print("\n[Check: sample quant_params vs normalizer_params]")
        if norm.get('cond_time_range_min') is not None:
            _cmp('cond_time_range_min', tmin, float(norm['cond_time_range_min']))
            _cmp('cond_time_range_max', tmax, float(norm['cond_time_range_max']))
        if norm.get('real_time_range_min') is not None:
            _cmp('real_time_range_min', rtmin, float(norm['real_time_range_min']))
            _cmp('real_time_range_max', rtmax, float(norm['real_time_range_max']))


if __name__ == '__main__':
    main()
