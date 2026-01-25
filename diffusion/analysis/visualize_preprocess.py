"""\
Visualize preprocessing for a single sample.

Outputs two figures under diffusion/results/<version>/analysis/:
1) Spectrograms (transform magnitudes)
2) Time-domain waveforms (original vs reconstructed by inverting the transform)

Run from project root:

    uv run diffusion/analysis/visualize_preprocess.py

"""
from pathlib import Path
import pickle
import torch
import matplotlib.pyplot as plt
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

    device = torch.device('cpu')

    # Load a single raw signal (first chunk)
    # Load raw data using central config to avoid ambiguity
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

    sample = signals[0]  # 1D tensor

    # Compute transform magnitude + reconstruction for raw
    transform_raw = get_transform(transform_type, **pipeline_config, device=device)
    mag_raw_t = transform_raw.apply(sample.unsqueeze(0)).squeeze(0).cpu()
    recon_raw_t = transform_raw.inverse(mag_raw_t.unsqueeze(0)).squeeze(0).detach().cpu()

    # Time-domain quantization ranges (saved by runner)
    ctmin = norm.get('cond_time_range_min')
    ctmax = norm.get('cond_time_range_max')
    rtmin = norm.get('real_time_range_min')
    rtmax = norm.get('real_time_range_max')
    if ctmin is None or ctmax is None or rtmin is None or rtmax is None:
        raise KeyError('Missing time quantizer ranges in normalizer_params.pkl. Re-run `uv run diffusion/utils/runner.py`.')

    q4_time = UniformQuantizer(bits=4, range_min=float(ctmin), range_max=float(ctmax))
    q16_time = UniformQuantizer(bits=16, range_min=float(rtmin), range_max=float(rtmax))

    sample_b = sample.unsqueeze(0)
    wave4 = q4_time.quantize(sample_b).squeeze(0)
    wave16 = q16_time.quantize(sample_b).squeeze(0)

    # Compute transform magnitudes + reconstructions for quantized waveforms
    transform_4 = get_transform(transform_type, **pipeline_config, device=device)
    mag_4_t = transform_4.apply(wave4.unsqueeze(0)).squeeze(0).cpu()
    recon_4_t = transform_4.inverse(mag_4_t.unsqueeze(0)).squeeze(0).detach().cpu()

    transform_16 = get_transform(transform_type, **pipeline_config, device=device)
    mag_16_t = transform_16.apply(wave16.unsqueeze(0)).squeeze(0).cpu()
    recon_16_t = transform_16.inverse(mag_16_t.unsqueeze(0)).squeeze(0).detach().cpu()

    # -------- Figure 1: Spectrograms --------
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 9))

    axes1[0].imshow(mag_raw_t.numpy(), aspect='auto', origin='lower')
    axes1[0].set_title('Transform magnitude (raw)')

    axes1[1].imshow(mag_4_t.numpy(), aspect='auto', origin='lower')
    axes1[1].set_title('Transform magnitude (4-bit time-quantized)')

    axes1[2].imshow(mag_16_t.numpy(), aspect='auto', origin='lower')
    axes1[2].set_title('Transform magnitude (16-bit time-quantized)')

    plt.tight_layout()
    out_specs = results_dir / 'analysis' / 'preprocess_sample_0_specs.png'
    out_specs.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_specs, dpi=150)

    # -------- Figure 2: Time series (time-domain vs inverse-transform) --------
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 8), sharex='col')

    raw_np = sample.detach().cpu().numpy()
    wave4_np = wave4.detach().cpu().numpy()
    wave16_np = wave16.detach().cpu().numpy()
    recon_raw_np = recon_raw_t.numpy()
    recon_4_np = recon_4_t.numpy()
    recon_16_np = recon_16_t.numpy()

    axes2[0, 0].plot(raw_np)
    axes2[0, 0].set_title('Raw (time domain)')
    axes2[0, 0].set_xlim(0, len(raw_np) - 1)
    axes2[0, 1].plot(recon_raw_np)
    axes2[0, 1].set_title('Raw (inverse transform)')
    axes2[0, 1].set_xlim(0, len(recon_raw_np) - 1)

    axes2[1, 0].plot(wave4_np)
    axes2[1, 0].set_title('4-bit time-quantized (time domain)')
    axes2[1, 0].set_xlim(0, len(wave4_np) - 1)
    axes2[1, 1].plot(recon_4_np)
    axes2[1, 1].set_title('4-bit time-quantized (inverse transform)')
    axes2[1, 1].set_xlim(0, len(recon_4_np) - 1)

    axes2[2, 0].plot(wave16_np)
    axes2[2, 0].set_title('16-bit time-quantized (time domain)')
    axes2[2, 0].set_xlim(0, len(wave16_np) - 1)
    axes2[2, 1].plot(recon_16_np)
    axes2[2, 1].set_title('16-bit time-quantized (inverse transform)')
    axes2[2, 1].set_xlim(0, len(recon_16_np) - 1)

    plt.tight_layout()
    out_time = results_dir / 'analysis' / 'preprocess_sample_0_time.png'
    plt.savefig(out_time, dpi=150)

    print(f"Saved spectrogram visualization to: {out_specs}")
    print(f"Saved time-series visualization to: {out_time}")


if __name__ == '__main__':
    main()
