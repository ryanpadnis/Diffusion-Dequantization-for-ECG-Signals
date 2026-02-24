"""Compare raw vs 4-bit vs target-bit vs diffusion-generated outputs for one sample.

Works with both legacy and current sample layouts.

Expected layout (training-style) for a run:
    diffusion/results/<V1|V2|V3|V4>/<run_id>/samples/<ddpm|ddim>/sample_<idx>/...

How to run (copy/paste safe):
    # Analyze sample_0 from a V4 run (auto-detect ddpm vs ddim)
    uv run python -m diffusion.analysis.analysis --version V4 --run-id 20260213_202346 --sample-idx 0 --max-trajectories 1 --print-details

    # If you specifically want DDIM outputs
    uv run python -m diffusion.analysis.analysis --version V4 --run-id 20260213_202346 --sampler-type ddim --sample-idx 0

Outputs:
    Saves plots + metrics JSON under:
        diffusion/results/<version>/<run-id>/analysis/sample_<idx>/
"""

from pathlib import Path

import argparse
import json
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from data.utils.transforms import get_transform
from data.utils.quantizers import UniformQuantizer, compute_range_from_tensor


def _lowpass_fft(x: torch.Tensor, *, cutoff_hz: float, sample_rate_hz: float) -> torch.Tensor:
    """Simple ideal low-pass via rFFT masking.

    Keeps frequencies <= cutoff_hz and zeroes the rest.
    """
    x = x.detach().to(torch.float32).reshape(-1)
    n = int(x.numel())
    if n <= 0:
        return x
    sr = float(sample_rate_hz)
    cutoff = float(cutoff_hz)
    if (not np.isfinite(sr)) or sr <= 0 or (not np.isfinite(cutoff)) or cutoff <= 0:
        return x
    # Clamp cutoff at Nyquist to avoid empty/invalid masks.
    cutoff = min(cutoff, 0.5 * sr)

    X = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr).to(X.device)
    mask = (freqs <= cutoff).to(X.dtype)
    y = torch.fft.irfft(X * mask, n=n)
    return y.to(torch.float32)


def _aligned_pair(a: torch.Tensor | None, b: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    if a is None or b is None or (not torch.is_tensor(a)) or (not torch.is_tensor(b)):
        return None
    a = _squeeze_1d(a.detach().to(torch.float32)).reshape(-1)
    b = _squeeze_1d(b.detach().to(torch.float32)).reshape(-1)
    n = min(int(a.numel()), int(b.numel()))
    if n <= 0:
        return None
    return a[:n], b[:n]


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).reshape(-1)
    b = b.to(torch.float32).reshape(-1)
    a = a - torch.mean(a)
    b = b - torch.mean(b)
    denom = float(torch.sqrt(torch.sum(a * a) * torch.sum(b * b)).item())
    if denom <= 1e-12:
        return float('nan')
    return float((torch.sum(a * b) / denom).item())


def _mse_aligned(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    pair = _aligned_pair(a, b)
    if pair is None:
        return float('nan')
    x, y = pair
    return float(torch.mean((x - y) ** 2).item())


def _peak_rescale_like(ref: torch.Tensor | None, x: torch.Tensor | None) -> torch.Tensor | None:
    """Peak-rescale x to match ref peak (no lookahead other than ref choice)."""
    if ref is None or x is None or (not torch.is_tensor(ref)) or (not torch.is_tensor(x)):
        return None
    ref = _squeeze_1d(ref.detach().to(torch.float32)).reshape(-1)
    x = _squeeze_1d(x.detach().to(torch.float32)).reshape(-1)
    n = min(int(ref.numel()), int(x.numel()))
    if n <= 0:
        return x
    ref = ref[:n]
    x = x[:n]
    eps = 1e-8
    rpk = float(ref.abs().max().item())
    xpk = float(x.abs().max().item())
    if rpk <= 0 or xpk <= 0:
        return x
    return x * float(rpk / (xpk + eps))


def _reconstruct_gen_mag_lowpass_condphase(
    config: dict,
    data: dict,
    *,
    sample_dir: Path,
    sample_rate_hz: float,
    mag_lowpass_hz: float,
    mag_lowpass_kind: str,
    mag_lowpass_transition_bins: int,
) -> torch.Tensor | None:
    """Reconstruct time-domain signal from saved gen magnitude with cond phase + mag LP."""
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    transform_type = str(pipe.get('transform', 'stft')).lower()
    if transform_type != 'stft':
        return None

    raw = data.get('raw')
    if not torch.is_tensor(raw):
        return None
    raw = raw.detach().cpu().to(torch.float32)
    while raw.ndim > 2:
        raw = raw.squeeze(1)
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    length = int(raw.shape[-1])
    if length <= 0:
        return None

    gen_mag = _load_generated_denorm_mag(sample_dir, traj_idx=0, variant='default')
    if gen_mag is None or (not torch.is_tensor(gen_mag)):
        return None
    gen_mag = gen_mag.detach().cpu().to(torch.float32)
    while gen_mag.ndim > 3:
        gen_mag = gen_mag.squeeze(1)
    if gen_mag.ndim == 2:
        gen_mag = gen_mag.unsqueeze(0)

    # Compute condition phase from RAW waveform.
    pipeline_config = pipe.copy()
    pipeline_config.pop('transform', None)
    t = get_transform('stft', **pipeline_config, device=torch.device('cpu'))
    _ = t.apply(raw)
    cond_phase = getattr(t, 'phase', None)
    if cond_phase is None:
        return None
    cond_phase = _match_phase_shape(cond_phase.detach().cpu().to(torch.float32), gen_mag)

    # Low-pass the generated magnitude along frequency bins.
    n_fft = int(pipe.get('n_fft', 254))
    cutoff_bin = _phase_diag_cutoff_bin(
        float(mag_lowpass_hz),
        sample_rate_hz=float(sample_rate_hz),
        n_fft=n_fft,
        num_bins=int(gen_mag.shape[1]),
    )
    mag_lp = _lowpass_mag_bins(
        gen_mag,
        cutoff_bin=int(cutoff_bin),
        kind=str(mag_lowpass_kind),
        transition_bins=int(mag_lowpass_transition_bins),
    )

    t.phase = cond_phase
    # Ensure inverse uses the original conditioning length for center=True.
    t.input_length = int(length)
    recon = t.inverse(mag_lp).detach().cpu().to(torch.float32)
    return _squeeze_1d(recon)


def plot_minimal_time(
    *,
    cond4: torch.Tensor,
    target: torch.Tensor,
    gen0: torch.Tensor,
    gen0_lp: torch.Tensor | None,
    out_path: Path,
    title_prefix: str = '',
) -> None:
    series: list[tuple[str, torch.Tensor]] = [
        ('cond_4bit', _squeeze_1d(cond4)),
        ('target_11bit', _squeeze_1d(target)),
        ('gen_0', _squeeze_1d(gen0)),
    ]
    if gen0_lp is not None and torch.is_tensor(gen0_lp):
        series.append(('gen_0_magLP40_taper+condPhase', _squeeze_1d(gen0_lp)))

    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.6 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]

    tgt = _squeeze_1d(target).detach().to(torch.float32)
    tmin = float(tgt.min().item())
    tmax = float(tgt.max().item())
    margin = 0.05 * (tmax - tmin + 1e-8)
    shared_ylim = (tmin - margin, tmax + margin)

    for ax, (name, s) in zip(axes, series):
        ax.plot(_squeeze_1d(s).detach().cpu().numpy(), linewidth=0.7)
        ax.set_title(f"{title_prefix}{name}".strip())
        ax.set_ylim(*shared_ylim)
    axes[-1].set_xlabel('time index')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_minimal_fft_mag(
    *,
    cond4: torch.Tensor,
    target: torch.Tensor,
    gen0: torch.Tensor,
    gen0_lp: torch.Tensor | None,
    out_path: Path,
    sample_rate_hz: float,
    max_hz: float | None = None,
    title_prefix: str = '',
) -> None:
    series: list[tuple[str, torch.Tensor]] = [
        ('cond_4bit', _squeeze_1d(cond4)),
        ('target_11bit', _squeeze_1d(target)),
        ('gen_0', _squeeze_1d(gen0)),
    ]
    if gen0_lp is not None and torch.is_tensor(gen0_lp):
        series.append(('gen_0_magLP40_taper+condPhase', _squeeze_1d(gen0_lp)))

    lens = [int(_squeeze_1d(x).numel()) for _, x in series]
    n = int(min(lens))
    if n <= 4:
        return
    sr = float(sample_rate_hz)
    if (not np.isfinite(sr)) or sr <= 0:
        sr = 1.0
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr).detach().cpu().numpy()

    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.6 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, series):
        x = _squeeze_1d(s).detach().to(torch.float32).reshape(-1)[:n]
        X = torch.fft.rfft(x)
        mag = torch.abs(X).detach().cpu().numpy()
        ax.plot(freqs, mag, linewidth=0.8)
        ax.set_title(f"{title_prefix}rFFT magnitude: {name}".strip())
        ax.set_ylabel('|X|')
        if max_hz is not None and np.isfinite(float(max_hz)) and float(max_hz) > 0:
            ax.set_xlim(0.0, float(max_hz))
    axes[-1].set_xlabel('frequency (Hz)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_minimal_fft_mag_phase(
    *,
    cond4: torch.Tensor,
    target: torch.Tensor,
    gen0: torch.Tensor,
    gen0_lp: torch.Tensor | None,
    out_path: Path,
    sample_rate_hz: float,
    max_hz: float | None = None,
    phase_mag_floor_rel: float = 1e-3,
    title_prefix: str = '',
) -> None:
    """Minimal rFFT magnitude AND phase plot for (cond4, target, gen0, optional gen0_lp).

    Phase is masked to NaN where magnitude is very small to reduce visual noise.
    """
    series: list[tuple[str, torch.Tensor]] = [
        ('cond_4bit', _squeeze_1d(cond4)),
        ('target_11bit', _squeeze_1d(target)),
        ('gen_0', _squeeze_1d(gen0)),
    ]
    if gen0_lp is not None and torch.is_tensor(gen0_lp):
        series.append(('gen_0_magLP40_taper+condPhase', _squeeze_1d(gen0_lp)))

    lens = [int(_squeeze_1d(x).numel()) for _, x in series]
    n = int(min(lens)) if lens else 0
    if n <= 4:
        return

    sr = float(sample_rate_hz)
    if (not np.isfinite(sr)) or sr <= 0:
        sr = 1.0
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr).detach().cpu().numpy()

    fig, axes = plt.subplots(len(series), 2, figsize=(14, 2.6 * len(series)), sharex='col')
    if len(series) == 1:
        axes = np.array([axes])

    for row_idx, (name, s) in enumerate(series):
        x = _squeeze_1d(s).detach().to(torch.float32).reshape(-1)[:n]
        X = torch.fft.rfft(x)
        mag = torch.abs(X).detach().cpu().numpy()
        ph = torch.angle(X).detach().cpu().numpy()

        mmax = float(np.max(mag)) if mag.size else 0.0
        floor = float(phase_mag_floor_rel) * mmax
        if floor > 0:
            ph = np.where(mag >= floor, ph, np.nan)

        axm = axes[row_idx, 0]
        axp = axes[row_idx, 1]
        axm.plot(freqs, mag, linewidth=0.8)
        axm.set_title(f"{title_prefix}rFFT magnitude: {name}".strip())
        axm.set_ylabel('|X|')

        axp.plot(freqs, ph, linewidth=0.8)
        axp.set_title(f"{title_prefix}rFFT phase: {name}".strip())
        axp.set_ylabel('∠X (rad)')
        axp.set_ylim(-math.pi, math.pi)

        if max_hz is not None and np.isfinite(float(max_hz)) and float(max_hz) > 0:
            axm.set_xlim(0.0, float(max_hz))
            axp.set_xlim(0.0, float(max_hz))

    axes[-1, 0].set_xlabel('frequency (Hz)')
    axes[-1, 1].set_xlabel('frequency (Hz)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_recon_variants(
    *,
    config: dict,
    data: dict,
    sample_dir: Path,
    sample_rate_hz: float,
    griffin_lim_iters: int,
    hybrid_cutoff_hz: float | None,
    mag_lowpass_hz: float | None,
    mag_lowpass_kind: str,
    mag_lowpass_transition_bins: int,
) -> dict[str, torch.Tensor]:
    """Compute a small set of waveform reconstructions for comparison.

    Returns a dict of 1D float32 CPU tensors, cropped to the conditioning length.
    Keys include: raw, cond_4bit, target, gen0_saved, recon_condphase, recon_gl, recon_hybrid.
    """
    out: dict[str, torch.Tensor] = {}

    raw = data.get('raw')
    if torch.is_tensor(raw):
        raw = _squeeze_1d(raw.detach().cpu().to(torch.float32))
        out['raw'] = raw

    cond4 = data.get('4bit')
    if torch.is_tensor(cond4):
        out['cond_4bit'] = _squeeze_1d(cond4.detach().cpu().to(torch.float32))

    target = data.get('16bit_gt')
    if torch.is_tensor(target):
        out['target'] = _squeeze_1d(target.detach().cpu().to(torch.float32))

    gen0 = None
    if isinstance(data.get('trajectories'), list) and data['trajectories']:
        gen0 = data['trajectories'][0]
    if torch.is_tensor(gen0):
        out['gen0_saved'] = _squeeze_1d(gen0.detach().cpu().to(torch.float32))

    # Need raw + STFT pipeline to compute phase-based reconstructions.
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    if str(pipe.get('transform', 'stft')).lower() != 'stft':
        return out
    if not torch.is_tensor(raw):
        return out

    length = int(out['raw'].numel())
    if length <= 0:
        return out

    gen_mag = _load_generated_denorm_mag(sample_dir, traj_idx=0, variant='default')
    if gen_mag is None or (not torch.is_tensor(gen_mag)):
        return out
    gen_mag = gen_mag.detach().cpu().to(torch.float32)
    while gen_mag.ndim > 3:
        gen_mag = gen_mag.squeeze(1)
    if gen_mag.ndim == 2:
        gen_mag = gen_mag.unsqueeze(0)

    # Compute condition phase from RAW waveform.
    pipeline_config = pipe.copy()
    pipeline_config.pop('transform', None)
    t = get_transform('stft', **pipeline_config, device=torch.device('cpu'))
    _ = t.apply(out['raw'].reshape(1, -1))
    cond_phase = getattr(t, 'phase', None)
    if cond_phase is None:
        return out
    cond_phase = _match_phase_shape(cond_phase.detach().cpu().to(torch.float32), gen_mag)

    # Optional: low-pass the generated magnitude along frequency bins.
    mag_to_use = gen_mag
    if mag_lowpass_hz is not None and math.isfinite(float(mag_lowpass_hz)) and float(mag_lowpass_hz) > 0:
        n_fft = int(pipe.get('n_fft', 254))
        cutoff_bin = _phase_diag_cutoff_bin(
            float(mag_lowpass_hz),
            sample_rate_hz=float(sample_rate_hz),
            n_fft=n_fft,
            num_bins=int(gen_mag.shape[1]),
        )
        mag_to_use = _lowpass_mag_bins(
            gen_mag,
            cutoff_bin=int(cutoff_bin),
            kind=str(mag_lowpass_kind),
            transition_bins=int(mag_lowpass_transition_bins),
        )

    # Ensure inverse uses original conditioning length for center=True.
    t.input_length = int(length)

    # Recon A: magnitude + condition phase
    try:
        t.phase = cond_phase
        recon_condphase = t.inverse(mag_to_use).detach().cpu().to(torch.float32).reshape(-1)[:length]
        out['recon_condphase'] = _squeeze_1d(recon_condphase)
    except Exception:
        pass

    # Recon B: magnitude + Griffin-Lim phase
    n_fft = int(pipe.get('n_fft', 254))
    hop = int(pipe.get('hop_length', 57))
    win = int(pipe.get('win_length', n_fft))
    center = bool(pipe.get('center', True))
    onesided = bool(pipe.get('onesided', True))
    try:
        recon_gl, gl_phase = _griffin_lim(
            mag_to_use,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            center=center,
            onesided=onesided,
            length=length,
            n_iter=int(griffin_lim_iters),
            seed=0,
        )
        out['recon_gl'] = _squeeze_1d(recon_gl.detach().cpu().to(torch.float32).reshape(-1)[:length])
        gl_phase = _match_phase_shape(gl_phase.detach().cpu().to(torch.float32), gen_mag)
    except Exception:
        gl_phase = None

    # Recon C: hybrid phase (low bins from condition, high bins from GL)
    if gl_phase is not None and hybrid_cutoff_hz is not None and math.isfinite(float(hybrid_cutoff_hz)) and float(hybrid_cutoff_hz) > 0:
        try:
            cutoff_bin = _phase_diag_cutoff_bin(
                float(hybrid_cutoff_hz),
                sample_rate_hz=float(sample_rate_hz),
                n_fft=n_fft,
                num_bins=int(gen_mag.shape[1]),
            )
            hybrid_phase = gl_phase.clone()
            if int(cutoff_bin) > 0:
                hybrid_phase[:, : int(cutoff_bin), :] = cond_phase[:, : int(cutoff_bin), :]
            t.phase = hybrid_phase
            recon_h = t.inverse(mag_to_use).detach().cpu().to(torch.float32).reshape(-1)[:length]
            out['recon_hybrid'] = _squeeze_1d(recon_h)
        except Exception:
            pass

    return out


def plot_compact_suite(
    *,
    config: dict,
    data: dict,
    sample_dir: Path,
    out_dir: Path,
    sample_rate_hz: float,
    griffin_lim_iters: int,
    hybrid_cutoff_hz: float | None,
    mag_lowpass_hz: float | None,
    mag_lowpass_kind: str,
    mag_lowpass_transition_bins: int,
    title_prefix: str = '',
) -> dict:
    """Compact analysis: ~5 PNGs + JSON metrics.

    Produces:
      - compact_time.png
      - compact_fft_mag_phase.png
      - compact_spec_mag.png
      - compact_spec_phase.png
      - compact_metrics.png
      - compact_metrics.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _compute_recon_variants(
        config=config,
        data=data,
        sample_dir=sample_dir,
        sample_rate_hz=float(sample_rate_hz),
        griffin_lim_iters=int(griffin_lim_iters),
        hybrid_cutoff_hz=hybrid_cutoff_hz,
        mag_lowpass_hz=mag_lowpass_hz,
        mag_lowpass_kind=str(mag_lowpass_kind),
        mag_lowpass_transition_bins=int(mag_lowpass_transition_bins),
    )

    target = variants.get('target')
    cond4 = variants.get('cond_4bit')
    if target is None or cond4 is None:
        return {}

    # Pick a small set of recon methods.
    pick = [
        ('gen0_saved', variants.get('gen0_saved')),
        ('recon_condphase', variants.get('recon_condphase')),
        ('recon_gl', variants.get('recon_gl')),
        ('recon_hybrid', variants.get('recon_hybrid')),
    ]
    picked = [(n, x) for (n, x) in pick if torch.is_tensor(x)]
    if not picked:
        return {}

    # Scaling policies we’ll score.
    scale_refs: dict[str, torch.Tensor] = {
        'none': None,
        'cond4_peak': cond4,
        'target_peak': target,
    }

    metrics: dict[str, dict[str, float]] = {}
    for name, sig in picked:
        metrics[name] = {}
        for sname, ref in scale_refs.items():
            xs = sig if ref is None else _peak_rescale_like(ref, sig)
            metrics[name][sname] = _mse_aligned(xs, target)

    # For plots, use cond4_peak scaling (fair, no lookahead beyond condition).
    plot_series: list[tuple[str, torch.Tensor]] = []
    for name, sig in picked:
        xs = _peak_rescale_like(cond4, sig)
        plot_series.append((name, xs if xs is not None else sig))

    # --- Plot 1: time-domain comparison (target overlay + each variant)
    tgt = _squeeze_1d(target).detach().cpu().to(torch.float32)
    tmin = float(tgt.min().item())
    tmax = float(tgt.max().item())
    margin = 0.05 * (tmax - tmin + 1e-8)
    shared_ylim = (tmin - margin, tmax + margin)

    fig, axes = plt.subplots(len(plot_series), 1, figsize=(14, 2.4 * len(plot_series)), sharex=True)
    if len(plot_series) == 1:
        axes = [axes]
    for ax, (name, s) in zip(axes, plot_series):
        ax.plot(tgt.numpy(), linewidth=0.7, alpha=0.45, label='target')
        ax.plot(_squeeze_1d(s).detach().cpu().numpy(), linewidth=0.9, alpha=0.95, label=name)
        ax.set_title(f"{title_prefix}{name} (scaled: cond4_peak)".strip())
        ax.set_ylim(*shared_ylim)
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('time index')
    plt.tight_layout()
    fig.savefig(out_dir / 'compact_time.png', dpi=150)
    plt.close(fig)

    # --- Plot 2: rFFT magnitude + phase (target overlay)
    # Align to common length.
    lens = [int(_squeeze_1d(x).numel()) for _, x in plot_series] + [int(tgt.numel())]
    n = int(min(lens))
    sr = float(sample_rate_hz)
    if (not np.isfinite(sr)) or sr <= 0:
        sr = 1.0
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr).detach().cpu().numpy()
    Xt = torch.fft.rfft(tgt.reshape(-1)[:n])
    mag_t = torch.abs(Xt).detach().cpu().numpy()
    ph_t = torch.angle(Xt).detach().cpu().numpy()
    # Mask target phase where magnitude is tiny.
    mmax_t = float(np.max(mag_t)) if mag_t.size else 0.0
    floor_t = 1e-3 * mmax_t
    if floor_t > 0:
        ph_t = np.where(mag_t >= floor_t, ph_t, np.nan)

    fig, axes = plt.subplots(len(plot_series), 2, figsize=(14, 2.6 * len(plot_series)), sharex='col')
    if len(plot_series) == 1:
        axes = np.array([axes])
    for row_idx, (name, s) in enumerate(plot_series):
        x = _squeeze_1d(s).detach().cpu().to(torch.float32).reshape(-1)[:n]
        X = torch.fft.rfft(x)
        mag = torch.abs(X).detach().cpu().numpy()
        ph = torch.angle(X).detach().cpu().numpy()
        mmax = float(np.max(mag)) if mag.size else 0.0
        floor = 1e-3 * mmax
        if floor > 0:
            ph = np.where(mag >= floor, ph, np.nan)

        axm = axes[row_idx, 0]
        axp = axes[row_idx, 1]
        axm.plot(freqs, mag_t, linewidth=0.7, alpha=0.35, label='target')
        axm.plot(freqs, mag, linewidth=0.9, alpha=0.95, label=name)
        axm.set_title(f"{title_prefix}rFFT magnitude: {name}".strip())
        axm.set_ylabel('|X|')
        axm.legend(fontsize=8, loc='upper right')

        axp.plot(freqs, ph_t, linewidth=0.7, alpha=0.35, label='target')
        axp.plot(freqs, ph, linewidth=0.9, alpha=0.95, label=name)
        axp.set_title(f"{title_prefix}rFFT phase: {name}".strip())
        axp.set_ylabel('∠X (rad)')
        axp.set_ylim(-math.pi, math.pi)
    axes[-1, 0].set_xlabel('frequency (Hz)')
    axes[-1, 1].set_xlabel('frequency (Hz)')
    plt.tight_layout()
    fig.savefig(out_dir / 'compact_fft_mag_phase.png', dpi=150)
    plt.close(fig)

    # --- Plot 3/4: STFT magnitude + phase (2 columns: target vs variant)
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    if str(pipe.get('transform', 'stft')).lower() == 'stft':
        n_fft = int(pipe.get('n_fft', 254))
        hop = int(pipe.get('hop_length', 57))
        win = int(pipe.get('win_length', n_fft))
        center = bool(pipe.get('center', True))
        onesided = bool(pipe.get('onesided', True))

        # Target STFT once.
        Zt = _compute_stft_for_plot(tgt, n_fft=n_fft, hop_length=hop, win_length=win, center=center, onesided=onesided)
        mag_t_stft = torch.log1p(torch.abs(Zt).to(torch.float32)).detach().cpu().numpy()
        ph_t_stft = torch.angle(Zt).to(torch.float32).detach().cpu().numpy()

        # Magnitude comparisons.
        fig, axes = plt.subplots(len(plot_series), 2, figsize=(14, 3.0 * len(plot_series)), sharex=True, sharey=True)
        if len(plot_series) == 1:
            axes = np.array([axes])
        vmin = float(np.min(mag_t_stft))
        vmax = float(np.max(mag_t_stft))
        for row_idx, (name, s) in enumerate(plot_series):
            Zv = _compute_stft_for_plot(s, n_fft=n_fft, hop_length=hop, win_length=win, center=center, onesided=onesided)
            mag_v = torch.log1p(torch.abs(Zv).to(torch.float32)).detach().cpu().numpy()
            vmin = min(vmin, float(np.min(mag_v)))
            vmax = max(vmax, float(np.max(mag_v)))
        for row_idx, (name, s) in enumerate(plot_series):
            Zv = _compute_stft_for_plot(s, n_fft=n_fft, hop_length=hop, win_length=win, center=center, onesided=onesided)
            mag_v = torch.log1p(torch.abs(Zv).to(torch.float32)).detach().cpu().numpy()
            ax0 = axes[row_idx, 0]
            ax1 = axes[row_idx, 1]
            ax0.imshow(mag_t_stft, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
            ax0.set_title('target STFT mag (log1p)')
            ax0.set_ylabel('freq bin')
            ax1.imshow(mag_v, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
            ax1.set_title(f"{name} STFT mag (log1p)")
        axes[-1, 0].set_xlabel('frame')
        axes[-1, 1].set_xlabel('frame')
        plt.tight_layout()
        fig.savefig(out_dir / 'compact_spec_mag.png', dpi=150)
        plt.close(fig)

        # Phase comparisons.
        fig, axes = plt.subplots(len(plot_series), 2, figsize=(14, 3.0 * len(plot_series)), sharex=True, sharey=True)
        if len(plot_series) == 1:
            axes = np.array([axes])
        for row_idx, (name, s) in enumerate(plot_series):
            Zv = _compute_stft_for_plot(s, n_fft=n_fft, hop_length=hop, win_length=win, center=center, onesided=onesided)
            ph_v = torch.angle(Zv).to(torch.float32).detach().cpu().numpy()
            ax0 = axes[row_idx, 0]
            ax1 = axes[row_idx, 1]
            im0 = ax0.imshow(ph_t_stft, origin='lower', aspect='auto', vmin=-math.pi, vmax=math.pi, cmap='twilight')
            ax0.set_title('target STFT phase')
            ax0.set_ylabel('freq bin')
            im1 = ax1.imshow(ph_v, origin='lower', aspect='auto', vmin=-math.pi, vmax=math.pi, cmap='twilight')
            ax1.set_title(f"{name} STFT phase")
        # One shared colorbar.
        fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
        axes[-1, 0].set_xlabel('frame')
        axes[-1, 1].set_xlabel('frame')
        plt.tight_layout()
        fig.savefig(out_dir / 'compact_spec_phase.png', dpi=150)
        plt.close(fig)

    # --- Plot 5: metrics bar chart
    inv_names = [n for (n, _) in picked]
    scale_names = list(scale_refs.keys())
    vals = np.array([[metrics[n][s] for s in scale_names] for n in inv_names], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    x = np.arange(len(inv_names))
    width = 0.25
    for j, sname in enumerate(scale_names):
        ax.bar(x + (j - (len(scale_names) - 1) / 2) * width, vals[:, j], width=width, label=sname)
    ax.set_xticks(x)
    ax.set_xticklabels(inv_names, rotation=20, ha='right')
    ax.set_ylabel('MSE vs target')
    ax.set_title(f"{title_prefix}MSE by inversion & scaling".strip())
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / 'compact_metrics.png', dpi=150)
    plt.close(fig)

    out_json = {
        'sample_dir': str(sample_dir),
        'sample_rate_hz': float(sample_rate_hz),
        'griffin_lim_iters': int(griffin_lim_iters),
        'hybrid_cutoff_hz': (float(hybrid_cutoff_hz) if hybrid_cutoff_hz is not None else None),
        'mag_lowpass_hz': (float(mag_lowpass_hz) if mag_lowpass_hz is not None else None),
        'mag_lowpass_kind': str(mag_lowpass_kind),
        'mag_lowpass_transition_bins': int(mag_lowpass_transition_bins),
        'metrics': metrics,
    }
    with open(out_dir / 'compact_metrics.json', 'w') as f:
        json.dump(out_json, f, indent=2)
    return out_json


def _compute_stft_for_plot(
    x: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool,
    onesided: bool,
) -> torch.Tensor:
    x = _squeeze_1d(x.detach().to(torch.float32)).reshape(1, -1)
    window = torch.hann_window(int(win_length), periodic=True, dtype=torch.float32)
    Z = torch.stft(
        x,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=window,
        center=bool(center),
        onesided=bool(onesided),
        return_complex=True,
        normalized=False,
    )
    # [B, F, T] -> [F, T]
    return Z[0].detach().to(torch.complex64)


def _load_tensor_if_exists(path: Path, *, key_candidates: list[str] | None = None) -> torch.Tensor | None:
    if path is None or (not path.exists()):
        return None
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception:
        return None
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict) and key_candidates:
        return _extract_tensor(obj, key_candidates=key_candidates)
    return None


def _compare_compute_mag_phase(
    *,
    config: dict,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    if str(pipe.get('transform', 'stft')).lower() != 'stft':
        return None, None
    n_fft = int(pipe.get('n_fft', 254))
    hop = int(pipe.get('hop_length', 57))
    win = int(pipe.get('win_length', n_fft))
    center = bool(pipe.get('center', True))
    onesided = bool(pipe.get('onesided', True))
    Z = _compute_stft_for_plot(x, n_fft=n_fft, hop_length=hop, win_length=win, center=center, onesided=onesided)
    mag = torch.abs(Z).to(torch.float32)
    phase = torch.angle(Z).to(torch.float32)
    return mag, phase


def plot_compare_spec_mag(
    *,
    cond_mag: torch.Tensor,
    target_mag: torch.Tensor,
    gen_mag: torch.Tensor,
    gen_mag_from_time: torch.Tensor | None = None,
    extra_series: list[tuple[str, torch.Tensor]] | None = None,
    out_path: Path,
    log_scale: bool = True,
    annotation: str | None = None,
) -> None:
    series: list[tuple[str, torch.Tensor]] = [
        ('cond_4bit', cond_mag),
        ('target_11bit', target_mag),
        ('diffusion_generated (saved: spectrogram_denorm_mag)', gen_mag),
    ]
    if gen_mag_from_time is not None and torch.is_tensor(gen_mag_from_time):
        series.append(('generated (from time_domain_raw STFT mag)', gen_mag_from_time))
    if extra_series:
        for name, m in list(extra_series):
            if m is None or (not torch.is_tensor(m)):
                continue
            series.append((str(name), m))
    mats = []
    for name, m in series:
        m = m.detach().to(torch.float32)
        raw_max = float(torch.clamp(m, min=0.0).max().item())
        if bool(log_scale):
            m = torch.log1p(torch.clamp(m, min=0.0))
        mats.append((name, m.detach().cpu().numpy(), raw_max))
    vmin = min(float(np.min(a)) for _, a, _ in mats)
    vmax = max(float(np.max(a)) for _, a, _ in mats)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return

    anno_lines = len(textwrap.wrap(str(annotation), width=140)) if annotation else 0
    anno_height = max(0.0, anno_lines * 0.28 + 0.3)
    fig, axes = plt.subplots(len(mats), 1, figsize=(14, 3.0 * len(mats) + anno_height), sharex=True)
    if len(mats) == 1:
        axes = [axes]
    for ax, (name, a, raw_max) in zip(axes, mats):
        im = ax.imshow(a, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f"STFT magnitude{' (log1p)' if log_scale else ''}: {name}  [max={raw_max:.4g}]")
        ax.set_ylabel('freq bin')
        fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01)
    axes[-1].set_xlabel('frame')
    total_h = 3.0 * len(mats) + anno_height
    top_frac = max(0.5, 1.0 - anno_height / total_h) if anno_height > 0 else 0.97
    if annotation:
        fig.text(
            0.01,
            0.99,
            textwrap.fill(str(annotation), width=140),
            ha='left',
            va='top',
            fontsize=9,
            family='monospace',
        )
    plt.tight_layout(rect=[0, 0, 1, top_frac])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compare_spec_phase(
    *,
    cond_phase: torch.Tensor,
    target_phase: torch.Tensor,
    out_path: Path,
    annotation: str | None = None,
) -> None:
    series = [
        ('cond_4bit phase', cond_phase),
        ('target_11bit phase', target_phase),
    ]
    mats = [(name, p.detach().to(torch.float32).detach().cpu().numpy()) for name, p in series]
    fig, axes = plt.subplots(len(mats), 1, figsize=(14, 3.0 * len(mats)), sharex=True)
    if len(mats) == 1:
        axes = [axes]
    for ax, (name, a) in zip(axes, mats):
        im = ax.imshow(a, origin='lower', aspect='auto', vmin=-math.pi, vmax=math.pi, cmap='twilight')
        ax.set_title(name)
        ax.set_ylabel('freq bin')
        fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01)
    axes[-1].set_xlabel('frame')
    if annotation:
        fig.text(
            0.01,
            0.995,
            textwrap.fill(str(annotation), width=120),
            ha='left',
            va='top',
            fontsize=10,
            family='monospace',
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compare_time(
    *,
    cond_time: torch.Tensor,
    target_time: torch.Tensor,
    gen_time: torch.Tensor,
    extra_series: list[tuple[str, torch.Tensor]] | None = None,
    out_path: Path,
    annotation: str | None = None,
) -> None:
    series: list[tuple[str, torch.Tensor]] = [
        ('cond_4bit', _squeeze_1d(cond_time)),
        ('target_11bit', _squeeze_1d(target_time)),
        ('diffusion_generated', _squeeze_1d(gen_time)),
    ]
    if extra_series:
        for name, s in list(extra_series):
            if s is None or (not torch.is_tensor(s)):
                continue
            series.append((str(name), _squeeze_1d(s)))
    anno_lines = len(textwrap.wrap(str(annotation), width=140)) if annotation else 0
    anno_height = max(0.0, anno_lines * 0.28 + 0.3)
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.6 * len(series) + anno_height), sharex=True)
    if len(series) == 1:
        axes = [axes]
    tgt = _squeeze_1d(target_time).detach().to(torch.float32)
    tmin = float(tgt.min().item())
    tmax = float(tgt.max().item())
    margin = 0.05 * (tmax - tmin + 1e-8)
    shared_ylim = (tmin - margin, tmax + margin)
    for ax, (name, s) in zip(axes, series):
        sv = _squeeze_1d(s).detach().cpu().to(torch.float32)
        smax = float(sv.abs().max().item())
        ax.plot(sv.numpy(), linewidth=0.7)
        ax.set_title(f"{name}  [max={smax:.4g}]")
        ax.set_ylim(*shared_ylim)
    axes[-1].set_xlabel('time index')
    total_h = 2.6 * len(series) + anno_height
    top_frac = max(0.5, 1.0 - anno_height / total_h) if anno_height > 0 else 0.97
    if annotation:
        fig.text(
            0.01,
            0.99,
            textwrap.fill(str(annotation), width=140),
            ha='left',
            va='top',
            fontsize=9,
            family='monospace',
        )
    plt.tight_layout(rect=[0, 0, 1, top_frac])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compare_norm_preclamp_hist(
    *,
    cond_norm_pre: torch.Tensor,
    target_norm_pre: torch.Tensor,
    gen_norm: torch.Tensor | None,
    out_path: Path,
    title: str,
    clip_min: float = -4.0,
    clip_max: float = 4.0,
) -> dict:
    """Histogram of normalization values before clamp to [-1,1].

    Returns summary stats including out-of-range fractions.
    """
    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(torch.float32).reshape(-1)

    c = _flat(cond_norm_pre)
    t = _flat(target_norm_pre)
    g = _flat(gen_norm) if (gen_norm is not None and torch.is_tensor(gen_norm)) else None

    def _oob_frac(x: torch.Tensor) -> tuple[float, float, float]:
        n = int(x.numel())
        if n <= 0:
            return float('nan'), float('nan'), float('nan')
        lo = float(torch.mean((x < -1.0).to(torch.float32)).item())
        hi = float(torch.mean((x > 1.0).to(torch.float32)).item())
        both = float(torch.mean(((x < -1.0) | (x > 1.0)).to(torch.float32)).item())
        return lo, hi, both

    c_lo, c_hi, c_both = _oob_frac(c)
    t_lo, t_hi, t_both = _oob_frac(t)
    g_lo = g_hi = g_both = None
    if g is not None:
        g_lo, g_hi, g_both = _oob_frac(g)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.0))
    bins = 200
    # Clip tails for readability (keeps histogram stable)
    c_plot = torch.clamp(c, clip_min, clip_max).cpu().numpy()
    t_plot = torch.clamp(t, clip_min, clip_max).cpu().numpy()
    ax.hist(c_plot, bins=bins, alpha=0.55, label=f"cond preclamp (oob={c_both:.3g})", density=True)
    ax.hist(t_plot, bins=bins, alpha=0.55, label=f"target preclamp (oob={t_both:.3g})", density=True)
    if g is not None:
        g_plot = torch.clamp(g, clip_min, clip_max).cpu().numpy()
        ax.hist(g_plot, bins=bins, alpha=0.55, label=f"generated norm (saved) (oob={float(g_both):.3g})", density=True)
    ax.axvline(-1.0, color='k', linewidth=1.0, linestyle='--')
    ax.axvline(1.0, color='k', linewidth=1.0, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('normalized value')
    ax.set_ylabel('density')
    ax.set_xlim(float(clip_min), float(clip_max))
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        'cond_oob_frac': float(c_both),
        'cond_oob_low_frac': float(c_lo),
        'cond_oob_high_frac': float(c_hi),
        'target_oob_frac': float(t_both),
        'target_oob_low_frac': float(t_lo),
        'target_oob_high_frac': float(t_hi),
        'generated_oob_frac': (float(g_both) if g_both is not None else None),
        'generated_oob_low_frac': (float(g_lo) if g_lo is not None else None),
        'generated_oob_high_frac': (float(g_hi) if g_hi is not None else None),
        'cond_preclamp_min': float(c.min().item()) if int(c.numel()) else float('nan'),
        'cond_preclamp_max': float(c.max().item()) if int(c.numel()) else float('nan'),
        'target_preclamp_min': float(t.min().item()) if int(t.numel()) else float('nan'),
        'target_preclamp_max': float(t.max().item()) if int(t.numel()) else float('nan'),
        'generated_norm_min': (float(g.min().item()) if g is not None and int(g.numel()) else None),
        'generated_norm_max': (float(g.max().item()) if g is not None and int(g.numel()) else None),
    }


def plot_compare_stats_bars(
    *,
    series: dict[str, torch.Tensor],        # name -> 1D time-domain waveform
    spec_series: dict[str, torch.Tensor],   # name -> 2D spectrogram magnitude [F,T]
    out_path: Path,
    sample_rate_hz: float = 360.0,
    n_fft: int = 254,
    ecg_band_hz: tuple[float, float] = (0.5, 40.0),
    annotation: str | None = None,
) -> dict:
    """Bar chart grid: time-domain stats + spectral energy stats for each signal variant."""
    sr = float(sample_rate_hz)
    ecg_lo, ecg_hi = float(ecg_band_hz[0]), float(ecg_band_hz[1])

    # Compute freq bin range for ECG band (onesided STFT)
    n_bins = n_fft // 2 + 1
    freq_res = sr / float(n_fft)
    bin_lo = max(0, int(math.floor(ecg_lo / freq_res)))
    bin_hi = min(n_bins - 1, int(math.ceil(ecg_hi / freq_res)))

    names = list(series.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))

    def _safe(v: float) -> float:
        return float(v) if math.isfinite(float(v)) else 0.0

    stats: dict[str, dict] = {}
    for name, x in series.items():
        x = x.detach().to(torch.float32).reshape(-1)
        mag = spec_series.get(name)
        if mag is not None:
            mag = mag.detach().to(torch.float32)
            total_energy = float(torch.mean(mag ** 2).item())
            band_energy = float(torch.mean(mag[bin_lo:bin_hi + 1] ** 2).item()) if bin_hi >= bin_lo else float('nan')
            p95 = float(torch.quantile(mag.reshape(-1), 0.95).item())
        else:
            total_energy = band_energy = p95 = float('nan')
        stats[name] = {
            'mean': float(x.mean().item()),
            'std': float(x.std().item()),
            'rms': float(torch.sqrt(torch.mean(x ** 2)).item()),
            'peak': float(x.abs().max().item()),
            'spec_energy_total': total_energy,
            'spec_energy_band': band_energy,
            'spec_p95_mag': p95,
        }

    metrics = [
        ('Time mean', 'mean'),
        ('Time std', 'std'),
        ('Time RMS', 'rms'),
        ('Time peak', 'peak'),
        ('Spectral energy (all)', 'spec_energy_total'),
        (f'Spectral energy ({ecg_lo:.0f}-{ecg_hi:.0f} Hz)', 'spec_energy_band'),
        ('Spectral p95 mag', 'spec_p95_mag'),
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(2.5 * n_metrics, 4.5))
    x_pos = np.arange(len(names))

    for ax, (title, key) in zip(axes, metrics):
        vals = [_safe(stats[n][key]) for n in names]
        bars = ax.bar(x_pos, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f'{val:.3g}', ha='center', va='bottom', fontsize=6)

    if annotation:
        fig.text(0.01, 0.99, textwrap.fill(str(annotation), width=160),
                 ha='left', va='top', fontsize=7, family='monospace')

    plt.suptitle('Signal & Spectral Statistics Comparison', fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return stats


def plot_minimal_spectrogram_magnitude(
    *,
    config: dict,
    cond4: torch.Tensor,
    target: torch.Tensor,
    gen0: torch.Tensor,
    gen0_lp: torch.Tensor | None,
    out_path: Path,
    title_prefix: str = '',
    log_scale: bool = True,
) -> None:
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    if str(pipe.get('transform', 'stft')).lower() != 'stft':
        return

    n_fft = int(pipe.get('n_fft', 254))
    hop = int(pipe.get('hop_length', 57))
    win = int(pipe.get('win_length', n_fft))
    center = bool(pipe.get('center', True))
    onesided = bool(pipe.get('onesided', True))

    series: list[tuple[str, torch.Tensor]] = [
        ('cond_4bit', _squeeze_1d(cond4)),
        ('target_11bit', _squeeze_1d(target)),
        ('gen_0', _squeeze_1d(gen0)),
    ]
    if gen0_lp is not None and torch.is_tensor(gen0_lp):
        series.append(('gen_0_magLP40_taper+condPhase', _squeeze_1d(gen0_lp)))

    specs = []
    for name, s in series:
        Z = _compute_stft_for_plot(s, n_fft=n_fft, hop_length=hop, win_length=win, center=center, onesided=onesided)
        mag = torch.abs(Z).to(torch.float32)
        if bool(log_scale):
            mag = torch.log1p(mag)
        specs.append((name, mag.detach().cpu().numpy()))

    # Shared color limits for comparability.
    vmin = min(float(np.min(m)) for _, m in specs)
    vmax = max(float(np.max(m)) for _, m in specs)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return

    fig, axes = plt.subplots(len(specs), 1, figsize=(14, 3.0 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = [axes]

    for ax, (name, mag_np) in zip(axes, specs):
        im = ax.imshow(mag_np, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_title(f"{title_prefix}STFT magnitude: {name}".strip())
        ax.set_ylabel('freq bin')
        fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01)

    axes[-1].set_xlabel('frame')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_minimal_spectrogram_phase(
    *,
    config: dict,
    cond4: torch.Tensor,
    target: torch.Tensor,
    gen0: torch.Tensor,
    gen0_lp: torch.Tensor | None,
    out_path: Path,
    title_prefix: str = '',
) -> None:
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    if str(pipe.get('transform', 'stft')).lower() != 'stft':
        return

    n_fft = int(pipe.get('n_fft', 254))
    hop = int(pipe.get('hop_length', 57))
    win = int(pipe.get('win_length', n_fft))
    center = bool(pipe.get('center', True))
    onesided = bool(pipe.get('onesided', True))

    series: list[tuple[str, torch.Tensor]] = [
        ('cond_4bit', _squeeze_1d(cond4)),
        ('target_11bit', _squeeze_1d(target)),
        ('gen_0', _squeeze_1d(gen0)),
    ]
    if gen0_lp is not None and torch.is_tensor(gen0_lp):
        series.append(('gen_0_magLP40_taper+condPhase', _squeeze_1d(gen0_lp)))

    phases = []
    for name, s in series:
        Z = _compute_stft_for_plot(s, n_fft=n_fft, hop_length=hop, win_length=win, center=center, onesided=onesided)
        ph = torch.angle(Z).to(torch.float32)
        phases.append((name, ph.detach().cpu().numpy()))

    fig, axes = plt.subplots(len(phases), 1, figsize=(14, 3.0 * len(phases)), sharex=True)
    if len(phases) == 1:
        axes = [axes]

    for ax, (name, ph_np) in zip(axes, phases):
        im = ax.imshow(ph_np, origin='lower', aspect='auto', vmin=-math.pi, vmax=math.pi, cmap='twilight')
        ax.set_title(f"{title_prefix}STFT phase: {name}".strip())
        ax.set_ylabel('freq bin')
        fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01)

    axes[-1].set_xlabel('frame')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _acf(x: torch.Tensor, *, max_lag: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation function for lags [0..max_lag], normalized by r[0]."""
    x = x.detach().to(torch.float32).reshape(-1)
    n = int(x.numel())
    if n <= 1:
        return np.array([0], dtype=np.int32), np.array([np.nan], dtype=np.float32)
    x = x - torch.mean(x)
    x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
    n = int(x_np.size)
    max_lag = int(min(int(max_lag), n - 1))
    if max_lag <= 0:
        return np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)

    full = np.correlate(x_np, x_np, mode='full')
    center = n - 1
    r = full[center:center + max_lag + 1]
    r0 = float(r[0])
    if abs(r0) <= 1e-12:
        return np.arange(0, max_lag + 1, dtype=np.int32), np.full((max_lag + 1,), np.nan, dtype=np.float32)
    return np.arange(0, max_lag + 1, dtype=np.int32), (r / r0).astype(np.float32)


def _pacf_from_acf(acf: np.ndarray) -> np.ndarray:
    """Compute PACF from autocorrelation using Levinson-Durbin recursion.

    acf must include lag 0..m.
    Returns pacf[0..m] with pacf[0]=1.
    """
    r = np.asarray(acf, dtype=np.float64)
    m = int(r.size) - 1
    if m <= 0:
        return np.array([1.0], dtype=np.float32)
    if not np.isfinite(r[0]) or abs(float(r[0])) <= 1e-12:
        return np.full((m + 1,), np.nan, dtype=np.float32)

    phi = np.zeros((m + 1, m + 1), dtype=np.float64)
    sigma = np.zeros((m + 1,), dtype=np.float64)
    phi[1, 1] = float(r[1])
    sigma[1] = float(1.0 - phi[1, 1] * phi[1, 1])

    for k in range(2, m + 1):
        num = float(r[k])
        for j in range(1, k):
            num -= float(phi[k - 1, j]) * float(r[k - j])
        den = float(sigma[k - 1])
        if abs(den) <= 1e-12:
            phi[k, k] = 0.0
            sigma[k] = den
            continue
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        sigma[k] = sigma[k - 1] * (1.0 - phi[k, k] * phi[k, k])

    pacf = np.zeros((m + 1,), dtype=np.float64)
    pacf[0] = 1.0
    for k in range(1, m + 1):
        pacf[k] = phi[k, k]
    return pacf.astype(np.float32)


def _spectral_features(x: torch.Tensor) -> dict:
    x = x.detach().to(torch.float32).reshape(-1)
    if int(x.numel()) <= 0:
        return {}
    x0 = x - torch.mean(x)
    X = torch.fft.rfft(x0)
    P = (torch.abs(X) ** 2).detach().cpu().numpy()
    eps = 1e-12
    freqs = np.arange(P.size, dtype=np.float64)
    total = float(np.sum(P) + eps)
    centroid = float(np.sum(freqs * P) / total)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * P) / total))
    cdf = np.cumsum(P)
    rolloff_95 = int(np.searchsorted(cdf, 0.95 * total))
    flatness = float(np.exp(np.mean(np.log(P + eps))) / (np.mean(P) + eps))
    return {
        'time_energy': float(torch.mean(x0 * x0).item()),
        'rms': float(torch.sqrt(torch.mean(x0 * x0)).item()),
        'spec_energy': float(np.mean(P)),
        'spec_centroid_bin': centroid,
        'spec_bandwidth_bin': bandwidth,
        'spec_rolloff95_bin': rolloff_95,
        'spec_flatness': flatness,
    }


def plot_correlation_matrix(data: dict, out_path: Path) -> None:
    """Correlation matrix for just (cond_4bit, target, gen_0)."""
    cond4 = data.get('4bit')
    tgt = data.get('16bit_gt')
    gen0 = None
    if isinstance(data.get('trajectories'), list) and data['trajectories']:
        gen0 = data['trajectories'][0]

    items: list[tuple[str, torch.Tensor]] = []
    if torch.is_tensor(cond4):
        items.append(('cond_4bit', cond4))
    if torch.is_tensor(tgt):
        tb = data.get('_target_bits')
        items.append((f"target_{int(tb)}bit" if tb is not None else 'target', tgt))
    if torch.is_tensor(gen0):
        items.append(('gen_0', gen0))

    if len(items) < 2:
        return

    # Align all to the shortest length.
    lens = [int(_squeeze_1d(t).numel()) for _, t in items]
    n = int(min(lens))
    xs = [(_squeeze_1d(t).to(torch.float32).reshape(-1)[:n]) for _, t in items]
    labels = [k for k, _ in items]

    C = np.zeros((len(xs), len(xs)), dtype=np.float32)
    for i in range(len(xs)):
        for j in range(len(xs)):
            C[i, j] = float(_pearson_corr(xs[i], xs[j]))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5), constrained_layout=True)
    im = ax.imshow(C, vmin=-1.0, vmax=1.0, cmap='coolwarm')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title('Pearson correlation (aligned)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C[i, j]:.2f}", ha='center', va='center', fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acf_pacf_overlay(
    *,
    name: str,
    series: torch.Tensor,
    cond4: torch.Tensor,
    target: torch.Tensor,
    target_label: str = 'target',
    out_acf: Path,
    out_pacf: Path,
    max_lag: int = 400,
) -> None:
    pair = _aligned_pair(series, cond4)
    pair2 = _aligned_pair(series, target)
    if pair is None or pair2 is None:
        return

    # Align all three to same length.
    x = pair[0]
    c = pair[1]
    t = pair2[1]
    n = min(int(x.numel()), int(c.numel()), int(t.numel()))
    if n <= 1:
        return
    x = x[:n]
    c = c[:n]
    t = t[:n]

    max_lag = int(min(int(max_lag), n - 1))
    lags, acf_x = _acf(x, max_lag=max_lag)
    _, acf_c = _acf(c, max_lag=max_lag)
    _, acf_t = _acf(t, max_lag=max_lag)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    ax.plot(lags, acf_x, label=f'acf({name})', linewidth=1.0)
    ax.plot(lags, acf_t, label=f'acf({target_label})', linewidth=1.0)
    ax.plot(lags, acf_c, label='acf(cond_4bit)', linewidth=1.0)
    ax.axhline(0, color='k', linewidth=0.7, alpha=0.4)
    ax.set_title(f'ACF overlay (goal: match target)')
    ax.set_xlabel('lag (samples)')
    ax.set_ylabel('acf')
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    fig.savefig(out_acf, dpi=150)
    plt.close(fig)

    pacf_x = _pacf_from_acf(acf_x)
    pacf_t = _pacf_from_acf(acf_t)
    pacf_c = _pacf_from_acf(acf_c)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    ax.plot(lags, pacf_x, label=f'pacf({name})', linewidth=1.0)
    ax.plot(lags, pacf_t, label=f'pacf({target_label})', linewidth=1.0)
    ax.plot(lags, pacf_c, label='pacf(cond_4bit)', linewidth=1.0)
    ax.axhline(0, color='k', linewidth=0.7, alpha=0.4)
    ax.set_title(f'PACF overlay (goal: match target)')
    ax.set_xlabel('lag (samples)')
    ax.set_ylabel('pacf')
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    fig.savefig(out_pacf, dpi=150)
    plt.close(fig)


def compute_all_series_features(data: dict, *, max_trajectories: int = 1) -> dict[str, dict]:
    items = build_series(data, max_trajectories=int(max_trajectories), include_diff=False)
    items = [(k, _squeeze_1d(v).to(torch.float32).reshape(-1)) for k, v in items if torch.is_tensor(v)]
    if not items:
        return {}
    n = min(int(v.numel()) for _, v in items)
    if n <= 0:
        return {}
    out: dict[str, dict] = {}
    for name, v in items:
        f = _spectral_features(v[:n])
        out[str(name)] = f
    return out


def print_all_series_features(data: dict, *, max_trajectories: int = 1) -> dict[str, dict]:
    feats = compute_all_series_features(data, max_trajectories=int(max_trajectories))
    if not feats:
        return {}
    print('series_features:')
    for name, f in feats.items():
        print(
            f"  {name}: "
            f"time_energy={f['time_energy']:.6g}, rms={f['rms']:.6g}, "
            f"centroid_bin={f['spec_centroid_bin']:.2f}, bandwidth_bin={f['spec_bandwidth_bin']:.2f}, "
            f"rolloff95_bin={int(f['spec_rolloff95_bin'])}, flatness={f['spec_flatness']:.4f}"
        )
    return feats


def _squeeze_1d(x: torch.Tensor) -> torch.Tensor:
    while x.ndim > 1:
        x = x.squeeze(0)
    return x


def _first_existing_path(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


def _resolve_sample_dir(
    results_dir: Path,
    version: str,
    sample_idx: int,
    sampler_type: str | None = None,
) -> tuple[Path, str | None]:
    """Resolve sample directory.

    Supports both legacy `samples/sample_<idx>` and `samples/<sampler_type>/sample_<idx>`.
    If sampler_type is provided, it is used directly.
    Otherwise, the most recently modified matching sample directory is used.
    """
    samples_root = results_dir / version / 'samples'

    # Legacy layout: samples/sample_<idx>
    legacy = samples_root / f'sample_{sample_idx}'
    if legacy.exists():
        return legacy, None

    # New layout: samples/<sampler_type>/sample_<idx>
    if not samples_root.exists():
        return legacy, None

    if sampler_type:
        chosen = samples_root / sampler_type / f'sample_{sample_idx}'
        return chosen, sampler_type

    candidates: list[tuple[float, Path, str]] = []
    for sub in sorted([p for p in samples_root.iterdir() if p.is_dir()]):
        d = sub / f'sample_{sample_idx}'
        if d.exists():
            try:
                mtime = d.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((mtime, d, sub.name))

    if not candidates:
        return legacy, None

    # If multiple exist, pick the most recently modified.
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, chosen_dir, sampler_type = candidates[0]
    return chosen_dir, sampler_type


def _extract_tensor(obj, key_candidates: list[str] | None = None) -> torch.Tensor | None:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return _squeeze_1d(obj.to(torch.float32))

    if isinstance(obj, dict):
        if key_candidates:
            for k in key_candidates:
                if k in obj and torch.is_tensor(obj[k]):
                    return _squeeze_1d(obj[k].to(torch.float32))
        for v in obj.values():
            if torch.is_tensor(v):
                return _squeeze_1d(v.to(torch.float32))
    return None


def _quantize_generated(data: dict, bits: int = 16) -> None:
    if not data.get('trajectories'):
        return

    gt = data.get('16bit_gt')
    if gt is not None:
        range_min, range_max = compute_range_from_tensor(gt)
    else:
        stacked = torch.stack([t.detach().cpu() for t in data['trajectories']])
        range_min, range_max = compute_range_from_tensor(stacked)

    q = UniformQuantizer(bits=bits, range_min=range_min, range_max=range_max)
    data['trajectories'] = [q.quantize(t) for t in data['trajectories']]


def _unique_level_stats(x: torch.Tensor) -> dict:
    x = _squeeze_1d(x.detach().cpu().to(torch.float32))
    uniq = torch.unique(x)
    uniq_sorted = torch.sort(uniq).values
    out = {
        'unique_levels_used': int(uniq_sorted.numel()),
        'min': float(x.min().item()),
        'max': float(x.max().item()),
    }
    if uniq_sorted.numel() >= 2:
        diffs = uniq_sorted[1:] - uniq_sorted[:-1]
        diffs = diffs[diffs.abs() > 1e-8]
        if diffs.numel() > 0:
            out['estimated_step'] = float(torch.min(diffs).item())
    return out


def _infer_symmetric_time_range_from_quantized(x: torch.Tensor, bits: int) -> tuple[float, float] | None:
    """Infer [-peak, +peak] from a uniformly quantized signal.

    Assumes UniformQuantizer with symmetric range and bin-center decoding.
    """
    stats = _unique_level_stats(x)
    step = stats.get('estimated_step')
    if step is None or step <= 0:
        return None
    levels = 2 ** int(bits)
    peak = float(step) * float(levels) / 2.0
    if peak <= 0:
        return None
    return (-peak, peak)


def _ensure_time_range(data: dict, *, cond_bits: int = 4) -> tuple[float, float] | None:
    """Ensure time quantization range is present in `data`.

    Prefer reading `quant_params` if present in saved artifacts; otherwise infer
    a symmetric range from the saved conditional 4-bit waveform.
    """
    if '_time_range_min' in data and '_time_range_max' in data:
        return (float(data['_time_range_min']), float(data['_time_range_max']))

    # Try to infer from the saved conditional 4-bit waveform.
    saved_cond = data.get('4bit')
    if saved_cond is None:
        return None

    inferred = _infer_symmetric_time_range_from_quantized(saved_cond, bits=int(cond_bits))
    if inferred is None:
        return None
    data['_time_range_min'] = float(inferred[0])
    data['_time_range_max'] = float(inferred[1])
    return inferred


def _add_generated_quant_versions(data: dict, *, target_bits: int = 11, cond_bits: int = 4) -> None:
    """Add quantized versions of the generated time-domain trajectory.

    User intent: show what the diffusion output looks like when quantized down
    to 11-bit and 4-bit.

    Note: For debugging, we quantize gen0 using *its own* (min,max) range.
    This avoids the common confusion where a symmetric +/-peak range (derived
    from a different signal) appears to "rescale" the generated waveform.
    """
    trajectories = data.get('trajectories') or []
    if not trajectories:
        return

    def _quantize_and_store(prefix: str, x: torch.Tensor) -> None:
        x = x.detach().to(torch.float32).reshape(-1)
        gmin, gmax = compute_range_from_tensor(x)

        # If degenerate range, fall back to a non-zero symmetric range.
        if float(gmax) <= float(gmin):
            peak = float(x.detach().abs().max().item())
            if peak <= 0:
                peak = 1.0
            gmin, gmax = -peak, peak

        q4 = UniformQuantizer(bits=4, range_min=float(gmin), range_max=float(gmax))
        qT = UniformQuantizer(bits=int(target_bits), range_min=float(gmin), range_max=float(gmax))
        data[f'{prefix}_q4'] = q4.quantize(x)
        data[f'{prefix}_q{int(target_bits)}'] = qT.quantize(x)

    # Only add for first trajectory by default (plots can already include multiple gen_i).
    gen0 = trajectories[0]

    # Keep these for historical prints/debug.
    gmin, gmax = compute_range_from_tensor(gen0)
    data['_gen0_time_range_min'] = float(gmin)
    data['_gen0_time_range_max'] = float(gmax)

    _quantize_and_store('gen_0', gen0)

    # Also quantize the two inversion variants if present.
    if isinstance(data.get('gen_0_inv_unitrange'), torch.Tensor):
        _quantize_and_store('gen_0_inv_unitrange', data['gen_0_inv_unitrange'])
    if isinstance(data.get('gen_0_inv_rawspec'), torch.Tensor):
        _quantize_and_store('gen_0_inv_rawspec', data['gen_0_inv_rawspec'])


def _add_cond_minmax_scaled_gen0(data: dict, *, target_bits: int = 11) -> None:
    """Add a gen0 variant affinely scaled to match the condition min/max (time-domain).

    Adds:
      - gen_0_condminmax              (scaled waveform)
      - gen_0_condminmax_q4           (scaled + quantized using cond range)
      - gen_0_condminmax_q{target}    (scaled + quantized using cond range)

    Note: This uses ONLY the condition range and gen0's own range; no target lookahead.
    """
    trajectories = data.get('trajectories') or []
    cond4 = data.get('4bit')
    if (not trajectories) or (not torch.is_tensor(cond4)):
        return

    gen0 = trajectories[0].detach().to(torch.float32).reshape(-1)
    cond4 = cond4.detach().to(torch.float32).reshape(-1)
    n = min(int(gen0.numel()), int(cond4.numel()))
    if n <= 0:
        return
    gen0 = gen0[:n]
    cond4 = cond4[:n]

    cmin = float(cond4.min().item())
    cmax = float(cond4.max().item())
    gmin = float(gen0.min().item())
    gmax = float(gen0.max().item())

    if gmax <= gmin + 1e-12:
        gen_scaled = torch.clamp(gen0, min=cmin, max=cmax)
    else:
        gen01 = (gen0 - gmin) / (gmax - gmin)
        gen_scaled = gen01 * (cmax - cmin) + cmin
        gen_scaled = torch.clamp(gen_scaled, min=cmin, max=cmax)

    data['gen_0_condminmax'] = gen_scaled

    # Quantize using condition span so levels match the conditioning range.
    q4 = UniformQuantizer(bits=4, range_min=float(cmin), range_max=float(cmax))
    qT = UniformQuantizer(bits=int(target_bits), range_min=float(cmin), range_max=float(cmax))
    data['gen_0_condminmax_q4'] = q4.quantize(gen_scaled)
    data[f'gen_0_condminmax_q{int(target_bits)}'] = qT.quantize(gen_scaled)


def _fill_missing_baselines_from_raw(data: dict, config: dict) -> None:
    """Backfill missing baseline signals from the saved raw condition waveform.

    Some sampling runs only save:
      - condition_raw.pt (raw time-domain waveform)
      - trajectories (generated time-domain)

    For this project, the raw waveform is also the training target (assumed ~11-bit),
    and the 4-bit condition is derived from it using per-sample symmetric quantization.
    """

    raw = data.get('raw')
    if raw is None or (not torch.is_tensor(raw)):
        return

    # Treat raw as the target waveform when an explicit gt file isn't saved.
    if data.get('16bit_gt') is None:
        data['16bit_gt'] = raw

    # Recompute condition 4-bit baseline if it's missing.
    if data.get('4bit') is None:
        try:
            cond_bits = int((config or {}).get('bit_size', 4))
        except Exception:
            cond_bits = 4

        try:
            lower_pct = float((config or {}).get('quantile_clip_lower', 0.0))
            upper_pct = float((config or {}).get('quantile_clip_upper', 100.0))
        except Exception:
            lower_pct, upper_pct = 0.0, 100.0

        lo, hi = compute_range_from_tensor(raw, lower_pct=lower_pct, upper_pct=upper_pct)
        peak = max(abs(float(lo)), abs(float(hi)))
        if peak <= 0:
            peak = float(raw.detach().abs().max().item())
        if peak <= 0:
            peak = 1.0

        q = UniformQuantizer(bits=int(cond_bits), range_min=-peak, range_max=peak)
        data['4bit'] = q.quantize(raw)
        data['_time_range_min'] = float(-peak)
        data['_time_range_max'] = float(peak)

def load_sample_data(
    results_dir: Path,
    version: str,
    sample_idx: int = 0,
    *,
    sample_dir: Path | None = None,
    sampler_type: str | None = None,
    gen_time_key: str = 'auto',
) -> dict:
    data: dict = {}

    data['_gen_time_key'] = str(gen_time_key)

    if sample_dir is None:
        sample_dir, detected_sampler = _resolve_sample_dir(
            results_dir,
            version,
            sample_idx,
            sampler_type=sampler_type,
        )
    else:
        detected_sampler = sampler_type

    data['_sample_dir'] = sample_dir
    data['_sampler_type'] = detected_sampler

    # Load whatever exists (avoid hardcoded filenames/keys).
    cond_path = _first_existing_path(sample_dir, ['condition_raw.pt', 'condition_signal.pt', 'raw.pt'])
    if cond_path is not None:
        data['raw'] = _extract_tensor(
            torch.load(cond_path, map_location='cpu'),
            key_candidates=['raw_signal', 'condition_signal_16bit', 'condition_signal', 'signal', 'x'],
        )

    q4_path = _first_existing_path(
        sample_dir,
        ['cond_time_4bit.pt', 'condition_4bit_time.pt', 'quantized_signal.pt', 'quantized_signal_4bit.pt', 'condition_4bit.pt', 'q4.pt'],
    )
    if q4_path is not None:
        data['4bit'] = _extract_tensor(
            torch.load(q4_path, map_location='cpu'),
            key_candidates=['cond_time_4bit', 'time_domain_4bit', 'quantized_signal_4bit', 'condition_signal_4bit', 'signal_4bit', 'x'],
        )

    gt_path = _first_existing_path(
        sample_dir,
        ['target_time.pt', 'ground_truth_16bit_time.pt', 'ground_truth_signal.pt', 'ground_truth.pt', 'gt.pt', 'target.pt'],
    )
    if gt_path is not None:
        data['16bit_gt'] = _extract_tensor(
            torch.load(gt_path, map_location='cpu'),
            key_candidates=[
                'target_time',
                'time_domain_16bit',
                'ground_truth_signal_16bit',
                'ground_truth_signal',
                'gt',
                'target',
                'signal_16bit',
                'x',
            ],
        )

    # Auto-select which generated time-domain signal to analyze.
    # New sampler behavior saves condition-scaled waveform as `time_domain`.
    # Older runs saved raw waveform as `time_domain` and the scaled variant as
    # `time_domain_peak_rescaled`. We detect that case and prefer the legacy
    # scaled signal to keep old analyses meaningful.
    gen_time_key = str(gen_time_key or 'auto')
    gen_time_candidates = [
        'time_domain',
        'time_domain_peak_rescaled',
        'time_domain_raw',
        'generated_time_domain',
    ]
    if gen_time_key != 'auto':
        gen_time_candidates = [gen_time_key] + [k for k in gen_time_candidates if k != gen_time_key]

    def _append_trajs(dst: list[torch.Tensor], td: torch.Tensor | None) -> None:
        if not torch.is_tensor(td):
            return
        td = td.to(torch.float32)
        while td.ndim > 2:
            td = td.squeeze(1)
        if td.ndim == 1:
            dst.append(td)
        elif td.ndim == 2:
            dst.extend([td[i] for i in range(td.shape[0])])

    trajs: list[torch.Tensor] = []
    trajs_inv_unitrange: list[torch.Tensor] = []
    trajs_inv_rawspec: list[torch.Tensor] = []
    all_traj_path = sample_dir / 'all_trajectories.pt'
    if all_traj_path.exists():
        obj = torch.load(all_traj_path, map_location='cpu')
        td = None
        if isinstance(obj, dict):
            # Back-compat heuristic:
            # if time_domain == time_domain_raw and a legacy scaled series exists,
            # treat time_domain as raw (old behavior) and prefer the legacy scaled.
            try:
                td_default = obj.get('time_domain')
                td_raw = obj.get('time_domain_raw')
                td_legacy = obj.get('time_domain_peak_rescaled')
                if (
                    gen_time_key == 'auto'
                    and torch.is_tensor(td_default)
                    and torch.is_tensor(td_raw)
                    and torch.is_tensor(td_legacy)
                ):
                    a = td_default.to(torch.float32)
                    b = td_raw.to(torch.float32)
                    if a.shape == b.shape and torch.allclose(a, b, rtol=0.0, atol=1e-8):
                        gen_time_candidates = ['time_domain_peak_rescaled'] + [
                            k for k in gen_time_candidates if k != 'time_domain_peak_rescaled'
                        ]
            except Exception:
                pass

            for k in gen_time_candidates:
                td = obj.get(k)
                if td is not None:
                    data['_gen_time_key_resolved'] = str(k)
                    break

            # Always load the two inversion variants when present.
            _append_trajs(trajs_inv_unitrange, obj.get('time_domain_raw'))
            _append_trajs(trajs_inv_rawspec, obj.get('time_domain_from_rawspec'))
        if torch.is_tensor(td):
            _append_trajs(trajs, td)

    if not trajs:
        traj_idx = 0
        while True:
            traj_path = sample_dir / f'trajectory_{traj_idx}.pt'
            if not traj_path.exists():
                break
            traj_obj = torch.load(traj_path, map_location='cpu')

            traj_signal = None
            if isinstance(traj_obj, dict):
                for k in gen_time_candidates:
                    v = traj_obj.get(k)
                    if torch.is_tensor(v):
                        traj_signal = v
                        data['_gen_time_key_resolved'] = str(k)
                        break
            if traj_signal is None:
                traj_signal = _extract_tensor(
                    traj_obj,
                    key_candidates=[
                        *gen_time_candidates,
                        'time_domain',
                        'time_domain_raw',
                        'generated_signal',
                        'signal',
                    ],
                )

            if traj_signal is not None:
                trajs.append(traj_signal)

            # Also collect the two explicit inversion variants, if present.
            if isinstance(traj_obj, dict):
                _append_trajs(trajs_inv_unitrange, traj_obj.get('time_domain_raw'))
                _append_trajs(trajs_inv_rawspec, traj_obj.get('time_domain_from_rawspec'))
            traj_idx += 1

    data['trajectories'] = trajs
    data['trajectories_inv_unitrange'] = trajs_inv_unitrange
    data['trajectories_inv_rawspec'] = trajs_inv_rawspec

    # Expose first-trajectory variants as explicit series keys.
    if trajs_inv_unitrange:
        data['gen_0_inv_unitrange'] = trajs_inv_unitrange[0]
    if trajs_inv_rawspec:
        data['gen_0_inv_rawspec'] = trajs_inv_rawspec[0]
    return data


def _load_generated_spec(sample_dir: Path, traj_idx: int = 0) -> torch.Tensor | None:
    # Prefer explicit cached spec file (new sampler behavior)
    spec_path = sample_dir / f'trajectory_{traj_idx}_spec.pt'
    if spec_path.exists():
        obj = torch.load(spec_path, map_location='cpu')
        spec = _extract_tensor(obj, key_candidates=['spectrogram_16bit', 'spectrograms_16bit', 'spectrogram'])
        return spec

    traj_path = sample_dir / f'trajectory_{traj_idx}.pt'
    if traj_path.exists():
        obj = torch.load(traj_path, map_location='cpu')
        spec = _extract_tensor(obj, key_candidates=['spectrogram_16bit', 'spectrograms_16bit', 'spectrogram'])
        return spec

    all_traj_path = sample_dir / 'all_trajectories.pt'
    if all_traj_path.exists():
        obj = torch.load(all_traj_path, map_location='cpu')
        if isinstance(obj, dict) and torch.is_tensor(obj.get('spectrograms_16bit')):
            s = obj['spectrograms_16bit'].to(torch.float32)
            while s.ndim > 4:
                s = s.squeeze(1)
            if s.ndim == 4 and s.shape[0] > traj_idx:
                return s[traj_idx]

    return None


def _denormalize_mag(normed: torch.Tensor, mag_min: float, mag_max: float) -> torch.Tensor:
    mag = (normed + 1.0) / 2.0
    mag = mag * (float(mag_max) - float(mag_min)) + float(mag_min)
    return torch.clamp(mag, min=0.0)


def _load_generated_denorm_mag(sample_dir: Path, traj_idx: int = 0, *, variant: str = 'default') -> torch.Tensor | None:
    """Load the saved *denormalized magnitude* for a trajectory.

    This is the most direct input to an ISTFT-style reconstruction.

    Args:
        variant:
            - 'default': use the sampler's main denormalized magnitude.
            - 'rawspec': use the denormalized magnitude produced from the raw diffusion output.
    """
    variant = str(variant or 'default').strip().lower()
    key = 'spectrogram_denorm_mag_rawspec' if variant in {'rawspec', 'raw', 'no_unitrange'} else 'spectrogram_denorm_mag'

    traj_path = sample_dir / f'trajectory_{traj_idx}.pt'
    if traj_path.exists():
        obj = torch.load(traj_path, map_location='cpu')
        if isinstance(obj, dict) and torch.is_tensor(obj.get(key)):
            m = obj[key].to(torch.float32)
            # Expected shape: [B, F, T] (no channel dim)
            while m.ndim > 3:
                m = m.squeeze(1)
            return m

    all_traj_path = sample_dir / 'all_trajectories.pt'
    if all_traj_path.exists():
        obj = torch.load(all_traj_path, map_location='cpu')
        if isinstance(obj, dict):
            # all_trajectories only stores the default denorm magnitude currently.
            if key != 'spectrogram_denorm_mag':
                return None
            if torch.is_tensor(obj.get('spectrograms_denorm_mag')):
                s = obj['spectrograms_denorm_mag'].to(torch.float32)
                while s.ndim > 3:
                    s = s.squeeze(1)
                if s.ndim == 3 and s.shape[0] > traj_idx:
                    return s[traj_idx : traj_idx + 1]

    return None


def _load_generated_denorm_phase(sample_dir: Path, traj_idx: int = 0, *, variant: str = 'default') -> torch.Tensor | None:
    """Load the saved *denormalized phase* (radians) for a trajectory.

    This exists when the sampler ran with use_phase_channel=True and saved
    'spectrogram_denorm_phase' alongside 'spectrogram_denorm_mag'.

    Args:
        variant:
            - 'default': use the sampler's main predicted phase.
            - 'rawspec': use the predicted phase produced from the raw diffusion output.
    """
    variant = str(variant or 'default').strip().lower()
    key = 'spectrogram_denorm_phase_rawspec' if variant in {'rawspec', 'raw', 'no_unitrange'} else 'spectrogram_denorm_phase'

    traj_path = sample_dir / f'trajectory_{traj_idx}.pt'
    if traj_path.exists():
        obj = torch.load(traj_path, map_location='cpu')
        if isinstance(obj, dict) and torch.is_tensor(obj.get(key)):
            ph = obj[key].to(torch.float32)
            # Expected shape: [B, F, T] (no channel dim)
            while ph.ndim > 3:
                ph = ph.squeeze(1)
            return ph

    # all_trajectories does not currently store predicted phase.
    return None


def _match_phase_shape(phase: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """Pad/crop phase to match magnitude's [B, F, T] shape."""
    phase = phase.to(torch.float32)
    mag = mag.to(torch.float32)
    while phase.ndim > 3:
        phase = phase.squeeze(1)
    while mag.ndim > 3:
        mag = mag.squeeze(1)

    if phase.ndim == 2:
        phase = phase.unsqueeze(0)
    if mag.ndim == 2:
        mag = mag.unsqueeze(0)

    Bp, Fp, Tp = int(phase.shape[0]), int(phase.shape[1]), int(phase.shape[2])
    Bm, Fm, Tm = int(mag.shape[0]), int(mag.shape[1]), int(mag.shape[2])

    # Match batch by repeating if needed (common case: phase computed from single condition)
    if Bp != Bm:
        if Bp == 1:
            phase = phase.expand(Bm, -1, -1)
        else:
            phase = phase[:Bm]

    # Crop/pad frequency
    if Fp > Fm:
        phase = phase[:, :Fm, :]
    elif Fp < Fm:
        pad = Fm - Fp
        phase = torch.nn.functional.pad(phase, (0, 0, 0, pad), mode='constant', value=0.0)

    # Crop/pad time frames
    Tp = int(phase.shape[2])
    if Tp > Tm:
        phase = phase[:, :, :Tm]
    elif Tp < Tm:
        pad = Tm - Tp
        phase = torch.nn.functional.pad(phase, (0, pad, 0, 0), mode='constant', value=0.0)

    return phase


def _griffin_lim(
    mag: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool,
    onesided: bool,
    length: int,
    n_iter: int = 32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Griffin–Lim phase reconstruction for a magnitude-only STFT.

    Returns:
        (time, phase) where phase is the final STFT phase estimate (radians).
    """
    mag = mag.detach().to(torch.float32)
    while mag.ndim > 3:
        mag = mag.squeeze(1)
    if mag.ndim == 2:
        mag = mag.unsqueeze(0)

    device = torch.device('cpu')
    mag = mag.to(device=device)
    window = torch.hann_window(int(win_length), device=device, dtype=torch.float32)

    # Initialize random phase.
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    phase = (2.0 * math.pi) * torch.rand(mag.shape, generator=g, device=device, dtype=torch.float32) - math.pi

    time = None
    for _ in range(int(max(1, n_iter))):
        complex_spec = mag * torch.exp(1j * phase)
        time = torch.istft(
            complex_spec,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window=window,
            center=bool(center),
            onesided=bool(onesided),
            length=int(length),
        )
        est = torch.stft(
            time,
            n_fft=int(n_fft),
            hop_length=int(hop_length),
            win_length=int(win_length),
            window=window,
            center=bool(center),
            onesided=bool(onesided),
            return_complex=True,
            normalized=False,
        )
        phase = torch.angle(est).to(torch.float32)

        # Align to requested shape if STFT frame count differs.
        phase = _match_phase_shape(phase, mag)

    assert time is not None
    return time.to(torch.float32), phase.to(torch.float32)


def _phase_diag_cutoff_bin(cutoff_hz: float, *, sample_rate_hz: float, n_fft: int, num_bins: int) -> int:
    sr = float(sample_rate_hz)
    cutoff = float(cutoff_hz)
    if (not math.isfinite(sr)) or sr <= 0 or (not math.isfinite(cutoff)) or cutoff <= 0:
        return 0
    # rFFT bin centers match k * sr / n_fft
    bins = int(math.floor(cutoff * float(n_fft) / sr)) + 1
    return max(0, min(int(num_bins), bins))


def _lowpass_mag_bins(
    mag: torch.Tensor,
    *,
    cutoff_bin: int,
    kind: str = 'hard',
    transition_bins: int = 8,
) -> torch.Tensor:
    """Low-pass a magnitude spectrogram along frequency bins.

    Args:
        mag: [B, F, T]
        cutoff_bin: bins < cutoff_bin are kept.
        kind: 'hard' or 'taper'
        transition_bins: for 'taper', number of bins over which to roll off.
    """
    mag = mag.detach().to(torch.float32)
    while mag.ndim > 3:
        mag = mag.squeeze(1)
    if mag.ndim == 2:
        mag = mag.unsqueeze(0)

    F = int(mag.shape[1])
    c = int(max(0, min(F, cutoff_bin)))
    if c <= 0:
        return torch.zeros_like(mag)
    if c >= F:
        return mag

    kind = str(kind or 'hard').strip().lower()
    out = mag.clone()
    if kind in {'hard', 'mask'}:
        out[:, c:, :] = 0.0
        return out

    if kind in {'taper', 'soft'}:
        tb = int(max(1, transition_bins))
        start = max(0, c - tb)
        # Build a 1D frequency mask with a cosine rolloff.
        w = torch.zeros((F,), dtype=out.dtype, device=out.device)
        w[:start] = 1.0
        # Rolloff region [start, c)
        n = c - start
        if n > 0:
            t = torch.linspace(0.0, 1.0, steps=n, device=out.device, dtype=out.dtype)
            # 1 -> 0 smooth
            w[start:c] = 0.5 * (1.0 + torch.cos(math.pi * t))
        w[c:] = 0.0
        out = out * w.view(1, F, 1)
        return out

    raise ValueError(f"Unknown mag lowpass kind: {kind!r} (use 'hard' or 'taper')")


def plot_phase_recon_diagnostics(
    config: dict,
    data: dict,
    *,
    sample_dir: Path,
    out_time_path: Path,
    out_fft_path: Path,
    out_phase_path: Path | None,
    sample_rate_hz: float,
    griffin_lim_iters: int,
    hybrid_cutoff_hz: float | None,
    mag_lowpass_hz: float | None,
    mag_lowpass_kind: str,
    mag_lowpass_transition_bins: int,
    time_lowpass_hz: float | None,
    primary_phase_source: str = 'auto',
    peak_rescale_to: str = 'none',
    share_ylim: bool = True,
) -> None:
    """Diagnose magnitude-vs-phase artifacts for magnitude-only STFT models.

    Saves:
      - time overlay plot
      - FFT magnitude overlay plot
      - optional phase matrix comparison plot
    """
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    transform_type = str(pipe.get('transform', 'stft')).lower()
    if transform_type != 'stft':
        print(f"phase_diag: skipped (transform={transform_type!r}, expected 'stft')")
        return

    raw = data.get('raw')
    if not torch.is_tensor(raw):
        print('phase_diag: skipped (raw condition waveform missing)')
        return
    raw = raw.detach().cpu().to(torch.float32)
    while raw.ndim > 2:
        raw = raw.squeeze(1)
    if raw.ndim == 1:
        raw = raw.unsqueeze(0)
    length = int(raw.shape[-1])
    if length <= 0:
        print('phase_diag: skipped (empty condition waveform)')
        return

    gen_mag = _load_generated_denorm_mag(sample_dir, traj_idx=0, variant='default')
    if gen_mag is None or (not torch.is_tensor(gen_mag)):
        print('phase_diag: skipped (generated denorm magnitude missing)')
        return
    gen_mag = gen_mag.detach().cpu().to(torch.float32)
    while gen_mag.ndim > 3:
        gen_mag = gen_mag.squeeze(1)
    if gen_mag.ndim == 2:
        gen_mag = gen_mag.unsqueeze(0)

    # Optional: load model-predicted phase (radians) if present.
    gen_phase_pred = _load_generated_denorm_phase(sample_dir, traj_idx=0, variant='default')
    if torch.is_tensor(gen_phase_pred):
        gen_phase_pred = _match_phase_shape(gen_phase_pred.detach().cpu().to(torch.float32), gen_mag)
    else:
        gen_phase_pred = None

    # Compute condition phase from RAW waveform using the same transform implementation.
    pipeline_config = pipe.copy()
    pipeline_config.pop('transform', None)
    t = get_transform('stft', **pipeline_config, device=torch.device('cpu'))
    _ = t.apply(raw)
    cond_phase = getattr(t, 'phase', None)
    if cond_phase is None:
        print('phase_diag: skipped (failed to compute condition phase)')
        return
    cond_phase = _match_phase_shape(cond_phase.detach().cpu(), gen_mag)

    # Recon 1: generated magnitude + raw-condition phase.
    t.phase = cond_phase
    recon_condphase = t.inverse(gen_mag).detach().cpu().to(torch.float32)

    # Recon 1b: generated magnitude + predicted phase (if available).
    recon_predphase = None
    if gen_phase_pred is not None:
        t.phase = gen_phase_pred
        recon_predphase = t.inverse(gen_mag).detach().cpu().to(torch.float32)

    # Recon 2: generated magnitude + Griffin–Lim estimated phase.
    n_fft = int(pipe.get('n_fft', 254))
    hop = int(pipe.get('hop_length', 57))
    win = int(pipe.get('win_length', n_fft))
    center = bool(pipe.get('center', True))
    onesided = bool(pipe.get('onesided', True))
    recon_gl, gl_phase = _griffin_lim(
        gen_mag,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        center=center,
        onesided=onesided,
        length=length,
        n_iter=int(griffin_lim_iters),
        seed=0,
    )
    recon_gl = recon_gl.detach().cpu().to(torch.float32)
    gl_phase = _match_phase_shape(gl_phase.detach().cpu(), gen_mag)

    # Recon 3: hybrid phase (low bins from condition, high bins from GL).
    recon_hybrid = None
    hybrid_phase = None
    if hybrid_cutoff_hz is not None and math.isfinite(float(hybrid_cutoff_hz)) and float(hybrid_cutoff_hz) > 0:
        cutoff_bin = _phase_diag_cutoff_bin(
            float(hybrid_cutoff_hz),
            sample_rate_hz=float(sample_rate_hz),
            n_fft=n_fft,
            num_bins=int(gen_mag.shape[1]),
        )
        hybrid_phase = gl_phase.clone()
        if cutoff_bin > 0:
            hybrid_phase[:, :cutoff_bin, :] = cond_phase[:, :cutoff_bin, :]
        t.phase = hybrid_phase
        recon_hybrid = t.inverse(gen_mag).detach().cpu().to(torch.float32)

    # Recon 4+: low-pass the GENERATED MAGNITUDE then invert with chosen phase(s).
    recon_mag_lp_condphase = None
    recon_mag_lp_hybrid = None
    recon_mag_lp_predphase = None
    mag_lp = None
    mag_cutoff_bin = None
    if mag_lowpass_hz is not None and math.isfinite(float(mag_lowpass_hz)) and float(mag_lowpass_hz) > 0:
        mag_cutoff_bin = _phase_diag_cutoff_bin(
            float(mag_lowpass_hz),
            sample_rate_hz=float(sample_rate_hz),
            n_fft=n_fft,
            num_bins=int(gen_mag.shape[1]),
        )
        mag_lp = _lowpass_mag_bins(
            gen_mag,
            cutoff_bin=int(mag_cutoff_bin),
            kind=str(mag_lowpass_kind),
            transition_bins=int(mag_lowpass_transition_bins),
        )

        t.phase = cond_phase
        recon_mag_lp_condphase = t.inverse(mag_lp).detach().cpu().to(torch.float32)

        if gen_phase_pred is not None:
            t.phase = gen_phase_pred
            recon_mag_lp_predphase = t.inverse(mag_lp).detach().cpu().to(torch.float32)

        if hybrid_phase is not None:
            t.phase = hybrid_phase
            recon_mag_lp_hybrid = t.inverse(mag_lp).detach().cpu().to(torch.float32)

    # Existing sampler recon (for comparison): gen_0 time series (already saved).
    gen0 = None
    if isinstance(data.get('trajectories'), list) and data['trajectories']:
        gen0 = _squeeze_1d(data['trajectories'][0]).detach().cpu().to(torch.float32)

    target = data.get('16bit_gt')
    if torch.is_tensor(target):
        target = _squeeze_1d(target).detach().cpu().to(torch.float32)

    # Optional: peak-rescale for apples-to-apples comparison.
    # Note: the sampler's saved `time_domain` is rescaled to match the CONDITION 4-bit peak.
    peak_rescale_to = str(peak_rescale_to or 'none').strip().lower()
    if peak_rescale_to not in {'none', 'off', 'cond4', 'condition', 'raw', 'target'}:
        peak_rescale_to = 'none'

    ref_peak = None
    if peak_rescale_to in {'cond4', 'condition'}:
        c4 = data.get('4bit')
        if torch.is_tensor(c4):
            ref_peak = float(_squeeze_1d(c4).detach().cpu().to(torch.float32).abs().max().item())
    elif peak_rescale_to == 'raw':
        ref_peak = float(_squeeze_1d(raw).detach().cpu().to(torch.float32).abs().max().item())
    elif peak_rescale_to == 'target':
        if target is not None:
            ref_peak = float(target.detach().cpu().to(torch.float32).abs().max().item())

    def _peak_rescale(sig: torch.Tensor) -> torch.Tensor:
        if ref_peak is None or (not math.isfinite(float(ref_peak))) or float(ref_peak) <= 0:
            return sig
        s = sig.detach().cpu().to(torch.float32).reshape(-1)
        p = float(s.abs().max().item())
        if (not math.isfinite(p)) or p <= 1e-12:
            return sig
        return (sig * (float(ref_peak) / (p + 1e-12))).to(torch.float32)

    def _crop1(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None or (not torch.is_tensor(x)):
            return None
        x = x.reshape(-1)
        if x.numel() <= 0:
            return None
        n = x.numel()
        n0 = min(n, length)
        return x[:n0]

    def _band_energy(x: torch.Tensor, *, cutoff_hz: float) -> tuple[float, float, float]:
        """Return (low_energy, high_energy, high_to_low_ratio) using ideal rFFT bin split."""
        x = x.detach().cpu().to(torch.float32).reshape(-1)
        n = int(x.numel())
        if n <= 0:
            return 0.0, 0.0, 0.0
        X = torch.fft.rfft(x)
        P = (torch.abs(X) ** 2).to(torch.float32)
        freqs = torch.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz))
        mask = (freqs <= float(cutoff_hz)).to(P.dtype)
        low = float(torch.sum(P * mask).item())
        high = float(torch.sum(P * (1.0 - mask)).item())
        ratio = float(high / (low + 1e-12))
        return low, high, ratio

    primary_phase_source = str(primary_phase_source or 'auto').strip().lower()
    if primary_phase_source not in {'auto', 'predicted', 'raw_cond', 'condition', 'griffinlim', 'hybrid'}:
        primary_phase_source = 'auto'

    series: list[tuple[str, torch.Tensor]] = []
    if target is not None:
        series.append(('target', _crop1(target)))
    if gen0 is not None:
        series.append(('gen0_saved', _crop1(gen0)))
    if recon_predphase is not None:
        series.append(('gen_mag + phase(predicted)', _crop1(recon_predphase)))
    series.append(('gen_mag + phase(raw_cond)', _crop1(recon_condphase)))
    series.append((f'gen_mag + griffinlim({int(griffin_lim_iters)})', _crop1(recon_gl)))
    if recon_hybrid is not None and hybrid_cutoff_hz is not None:
        series.append((f'gen_mag + hybrid_phase(lp<{float(hybrid_cutoff_hz):g}Hz)', _crop1(recon_hybrid)))
    if recon_mag_lp_condphase is not None and mag_lowpass_hz is not None:
        series.append(
            (
                f'LPmag<{float(mag_lowpass_hz):g}Hz + phase(raw_cond) ({str(mag_lowpass_kind)})',
                _crop1(recon_mag_lp_condphase),
            )
        )
    if recon_mag_lp_predphase is not None and mag_lowpass_hz is not None:
        series.append(
            (
                f'LPmag<{float(mag_lowpass_hz):g}Hz + phase(predicted) ({str(mag_lowpass_kind)})',
                _crop1(recon_mag_lp_predphase),
            )
        )
    if recon_mag_lp_hybrid is not None and mag_lowpass_hz is not None and hybrid_cutoff_hz is not None:
        series.append(
            (
                f'LPmag<{float(mag_lowpass_hz):g}Hz + hybrid_phase ({str(mag_lowpass_kind)})',
                _crop1(recon_mag_lp_hybrid),
            )
        )

    # Optional: time-domain low-pass after inversion for a few key reconstructions.
    if time_lowpass_hz is not None and math.isfinite(float(time_lowpass_hz)) and float(time_lowpass_hz) > 0:
        lp = float(time_lowpass_hz)
        # Pick a small subset to avoid clutter.
        picked = []

        # Always include the primary phase recon if present.
        primary_key = None
        if primary_phase_source in {'condition', 'raw_cond'}:
            primary_key = 'gen_mag + phase(raw_cond)'
        elif primary_phase_source == 'predicted':
            primary_key = 'gen_mag + phase(predicted)'
        elif primary_phase_source == 'griffinlim':
            primary_key = f'gen_mag + griffinlim({int(griffin_lim_iters)})'
        elif primary_phase_source == 'hybrid' and (recon_hybrid is not None) and (hybrid_cutoff_hz is not None):
            primary_key = f'gen_mag + hybrid_phase(lp<{float(hybrid_cutoff_hz):g}Hz)'
        else:
            # auto
            primary_key = 'gen_mag + phase(predicted)' if (recon_predphase is not None) else 'gen_mag + phase(raw_cond)'

        for name, sig in series:
            if name in {'gen0_saved', 'gen_mag + phase(raw_cond)', 'gen_mag + phase(predicted)'} or name.startswith('LPmag<'):
                picked.append((name, sig))
        # Ensure primary is included (and only once).
        if primary_key is not None:
            for name, sig in series:
                if name == primary_key and (name, sig) not in picked:
                    picked.append((name, sig))
                    break
        for name, sig in picked:
            y = _lowpass_fft(sig, cutoff_hz=lp, sample_rate_hz=float(sample_rate_hz))
            series.append((f'LPtime@{lp:g}Hz({name})', _crop1(y)))

    series = [(k, v) for (k, v) in series if v is not None]
    if not series:
        print('phase_diag: skipped (no series to plot)')
        return

    if peak_rescale_to not in {'none', 'off'}:
        # Rescale all non-target series to the same peak for easier comparison.
        out_series: list[tuple[str, torch.Tensor]] = []
        for name, sig in series:
            if name == 'target':
                out_series.append((name, sig))
            else:
                out_series.append((name, _crop1(_peak_rescale(sig))))
        series = out_series

    # Print a compact band-energy summary to help quantify the "target dies early" effect.
    energy_cutoff = None
    if mag_lowpass_hz is not None and float(mag_lowpass_hz) > 0:
        energy_cutoff = float(mag_lowpass_hz)
    elif hybrid_cutoff_hz is not None and float(hybrid_cutoff_hz) > 0:
        energy_cutoff = float(hybrid_cutoff_hz)
    else:
        energy_cutoff = 40.0

    try:
        print(f"phase_diag: band-energy split @ {energy_cutoff:g}Hz (sr={float(sample_rate_hz):g}Hz)")
        # Compare target vs the core generated outputs.
        if target is not None:
            lo, hi, r = _band_energy(_crop1(target), cutoff_hz=energy_cutoff)
            print(f"  target:   low={lo:.3g} high={hi:.3g} high/low={r:.3g}")
        keys = ['gen0_saved', 'gen_mag + phase(raw_cond)', f'gen_mag + griffinlim({int(griffin_lim_iters)})']
        if gen_phase_pred is not None:
            keys.insert(1, 'gen_mag + phase(predicted)')
        for key in keys:
            for name, sig in series:
                if name == key:
                    lo, hi, r = _band_energy(sig, cutoff_hz=energy_cutoff)
                    print(f"  {name}: low={lo:.3g} high={hi:.3g} high/low={r:.3g}")
    except Exception as e:
        print(f"phase_diag: band-energy summary failed (non-fatal): {e}")

    # Time-domain overlay plots.
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.6 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]

    shared_ylim = None
    if bool(share_ylim) and target is not None:
        tmin = float(target.min().item())
        tmax = float(target.max().item())
        margin = 0.05 * (tmax - tmin + 1e-8)
        shared_ylim = (tmin - margin, tmax + margin)
    for ax, (name, x) in zip(axes, series):
        ax.plot(x.detach().cpu().numpy(), linewidth=0.7)
        ax.set_title(name)
        if shared_ylim is not None and name != 'target':
            ax.set_ylim(*shared_ylim)
    axes[-1].set_xlabel('time index')
    plt.tight_layout()
    fig.savefig(out_time_path, dpi=150)
    plt.close(fig)

    # FFT magnitude overlays (in bins; optionally could convert to Hz but keep consistent with existing helper).
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.6 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (name, x) in zip(axes, series):
        X = torch.fft.rfft(x.detach().cpu().to(torch.float32))
        mag = torch.abs(X).numpy()
        ax.plot(mag, linewidth=0.7)
        ax.set_title(f'fft_mag: {name}')
    axes[-1].set_xlabel('freq bin')
    plt.tight_layout()
    fig.savefig(out_fft_path, dpi=150)
    plt.close(fig)

    if out_phase_path is not None:
        mats: list[tuple[str, torch.Tensor]] = [('phase(raw_cond)', cond_phase[0])]
        if gen_phase_pred is not None:
            mats.append(('phase(predicted)', gen_phase_pred[0]))
        mats.append((f'phase(griffinlim:{int(griffin_lim_iters)})', gl_phase[0]))
        if hybrid_phase is not None:
            mats.append((f'phase(hybrid<{float(hybrid_cutoff_hz):g}Hz)', hybrid_phase[0]))

        fig, axes = plt.subplots(1, len(mats), figsize=(5.5 * len(mats), 4.2), constrained_layout=True)
        if len(mats) == 1:
            axes = [axes]
        for ax, (name, ph) in zip(axes, mats):
            im = ax.imshow(ph.detach().cpu().numpy(), aspect='auto', origin='lower', cmap='twilight')
            ax.set_title(name)
            ax.set_xlabel('time frame')
            ax.set_ylabel('freq bin')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(out_phase_path, dpi=150)
        plt.close(fig)




def build_series(data: dict, max_trajectories: int = 1, include_diff: bool = True) -> list[tuple[str, torch.Tensor]]:
    series: list[tuple[str, torch.Tensor]] = []
    raw = data.get('raw')
    q4 = data.get('4bit')
    gt = data.get('16bit_gt')

    target_bits = data.get('_target_bits')

    if raw is not None:
        series.append(('raw', raw))
    if q4 is not None:
        series.append(('cond_4bit', q4))
    if gt is not None:
        if target_bits is not None:
            series.append((f'target_{int(target_bits)}bit', gt))
        else:
            series.append(('target', gt))

    for i, traj in enumerate(data.get('trajectories', [])[:max_trajectories]):
        series.append((f'gen_{i}', traj))

    # Derived gen0 quantized variants.
    if isinstance(data.get('gen_0_q4'), torch.Tensor):
        series.append(('gen_0_q4', data['gen_0_q4']))
    if target_bits is not None:
        k = f'gen_0_q{int(target_bits)}'
        if isinstance(data.get(k), torch.Tensor):
            series.append((k, data[k]))

    # Two inversion variants for debugging scaling/inversion effects.
    # - inv_rawspec: invert the raw diffusion spectrogram output (no unit-range rescale)
    # - inv_unitrange: invert after rescaling diffusion output to [-1, 1]
    if isinstance(data.get('gen_0_inv_rawspec'), torch.Tensor):
        series.append(('gen_0_inv_rawspec', data['gen_0_inv_rawspec']))
    if isinstance(data.get('gen_0_inv_rawspec_q4'), torch.Tensor):
        series.append(('gen_0_inv_rawspec_q4', data['gen_0_inv_rawspec_q4']))
    if target_bits is not None:
        k = f'gen_0_inv_rawspec_q{int(target_bits)}'
        if isinstance(data.get(k), torch.Tensor):
            series.append((k, data[k]))

    if isinstance(data.get('gen_0_inv_unitrange'), torch.Tensor):
        series.append(('gen_0_inv_unitrange', data['gen_0_inv_unitrange']))
    if isinstance(data.get('gen_0_inv_unitrange_q4'), torch.Tensor):
        series.append(('gen_0_inv_unitrange_q4', data['gen_0_inv_unitrange_q4']))
    if target_bits is not None:
        k = f'gen_0_inv_unitrange_q{int(target_bits)}'
        if isinstance(data.get(k), torch.Tensor):
            series.append((k, data[k]))

    # Condition-range scaled gen0 variant (and its quantized views).
    if isinstance(data.get('gen_0_condminmax'), torch.Tensor):
        series.append(('gen_0_condminmax', data['gen_0_condminmax']))
    if isinstance(data.get('gen_0_condminmax_q4'), torch.Tensor):
        series.append(('gen_0_condminmax_q4', data['gen_0_condminmax_q4']))
    if target_bits is not None:
        k = f'gen_0_condminmax_q{int(target_bits)}'
        if isinstance(data.get(k), torch.Tensor):
            series.append((k, data[k]))

    if include_diff and (q4 is not None) and (gt is not None):
        n = min(int(q4.numel()), int(gt.numel()))
        label = f'diff_target-cond4'
        series.append((label, gt[:n] - q4[:n]))

    return series


def plot_time_series(data: dict, out_path: Path) -> None:
    series = build_series(data, max_trajectories=1, include_diff=True)
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.5 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]

    gt = data.get('16bit_gt')
    shared_ylim = None
    if gt is not None:
        gt_min = float(gt.min().item())
        gt_max = float(gt.max().item())
        margin = 0.05 * (gt_max - gt_min + 1e-8)
        shared_ylim = (gt_min - margin, gt_max + margin)

    for ax, (name, s) in zip(axes, series):
        ax.plot(s.detach().cpu().numpy(), linewidth=0.6)
        ax.set_title(name)
        # Keep generated panels on the *same* scale as the target.
        if shared_ylim is not None:
            is_diff = name.startswith('diff_')
            is_generated = name.startswith('gen_')
            if is_generated and (not is_diff):
                ax.set_ylim(*shared_ylim)

    axes[-1].set_xlabel('time index')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_series_lowpass(
    data: dict,
    out_path: Path,
    *,
    cutoff_hz: float,
    sample_rate_hz: float,
) -> None:
    """Overlay original vs low-pass filtered signals (time domain).

    Intended as a quick qualitative check of whether high-frequency noise dominates.
    """
    # Only plot core signals to keep the figure readable.
    series_all = build_series(data, max_trajectories=1, include_diff=False)
    keep_prefixes = ("raw", "cond_4bit", "target_", "target", "gen_0")
    series = [(k, v) for (k, v) in series_all if any(k == p or k.startswith(p) for p in keep_prefixes)]
    if not series:
        return

    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.8 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]

    gt = data.get('16bit_gt')
    shared_ylim = None
    if gt is not None:
        gt_min = float(gt.min().item())
        gt_max = float(gt.max().item())
        margin = 0.05 * (gt_max - gt_min + 1e-8)
        shared_ylim = (gt_min - margin, gt_max + margin)

    for ax, (name, s) in zip(axes, series):
        s = _squeeze_1d(s).to(torch.float32).reshape(-1)
        y = _lowpass_fft(s, cutoff_hz=float(cutoff_hz), sample_rate_hz=float(sample_rate_hz))
        ax.plot(s.detach().cpu().numpy(), linewidth=0.55, alpha=0.70, label='orig')
        ax.plot(y.detach().cpu().numpy(), linewidth=0.9, alpha=0.95, label=f'lowpass@{float(cutoff_hz):g}Hz')
        ax.set_title(name)
        if shared_ylim is not None:
            is_generated = name.startswith('gen_')
            if is_generated:
                ax.set_ylim(*shared_ylim)
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel('time index')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fft_series(data: dict, out_path: Path, max_trajectories: int = 1, max_bin: int | None = None) -> None:
    series = build_series(data, max_trajectories=max_trajectories, include_diff=True)
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 2.5 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]

    for ax, (name, s) in zip(axes, series):
        x = s.detach().cpu().to(torch.float32)
        X = torch.fft.rfft(x)
        mag = torch.abs(X).numpy()
        ax.plot(mag, linewidth=0.6)
        ax.set_title(f'fft_mag: {name}')
        if max_bin is not None:
            ax.set_xlim(0, int(max_bin))

    axes[-1].set_xlabel('freq bin')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fft_mag_phase_series(
    data: dict,
    out_path: Path,
    *,
    sample_rate_hz: float,
    max_trajectories: int = 1,
    max_hz: float | None = None,
    phase_mag_floor_rel: float = 1e-3,
) -> None:
    """Plot rFFT magnitude AND phase for aligned time-series.

    Phase is masked to NaN where magnitude is very small to reduce visual noise.
    """
    entries = build_series(data, max_trajectories=max_trajectories, include_diff=True)
    if not entries:
        return

    # Align to common length.
    lens = [int(_squeeze_1d(x).numel()) for _, x in entries if torch.is_tensor(x)]
    if not lens:
        return
    n = int(min(lens))
    if n <= 4:
        return

    sr = float(sample_rate_hz)
    if (not np.isfinite(sr)) or sr <= 0:
        sr = 1.0
    freqs = torch.fft.rfftfreq(n, d=1.0 / sr).detach().cpu().numpy()

    fig, axes = plt.subplots(len(entries), 2, figsize=(14, 2.6 * len(entries)), sharex='col')
    if len(entries) == 1:
        axes = np.array([axes])

    for row_idx, (name, s) in enumerate(entries):
        x = _squeeze_1d(s).detach().cpu().to(torch.float32).reshape(-1)[:n]
        X = torch.fft.rfft(x)
        mag = torch.abs(X).detach().cpu().numpy()
        ph = torch.angle(X).detach().cpu().numpy()

        # Mask phase where magnitude is tiny.
        mmax = float(np.max(mag)) if mag.size else 0.0
        floor = float(phase_mag_floor_rel) * mmax
        if floor > 0:
            ph = np.where(mag >= floor, ph, np.nan)

        axm = axes[row_idx, 0]
        axp = axes[row_idx, 1]
        axm.plot(freqs, mag, linewidth=0.7)
        axm.set_title(f"rFFT magnitude: {name}")
        axm.set_ylabel('|X|')

        axp.plot(freqs, ph, linewidth=0.7)
        axp.set_title(f"rFFT phase: {name}")
        axp.set_ylabel('∠X (rad)')
        axp.set_ylim(-math.pi, math.pi)

        if max_hz is not None and np.isfinite(float(max_hz)) and float(max_hz) > 0:
            axm.set_xlim(0.0, float(max_hz))
            axp.set_xlim(0.0, float(max_hz))

    axes[-1, 0].set_xlabel('frequency (Hz)')
    axes[-1, 1].set_xlabel('frequency (Hz)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stft_mag_phase_spectrograms(
    config: dict,
    data: dict,
    out_path: Path,
    *,
    max_trajectories: int = 1,
) -> None:
    """Plot STFT magnitude and phase spectrograms for time-series entries.

    This computes STFT from the *waveforms* (raw/cond/target/gen) so both magnitude and
    phase are shown consistently.
    """
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    transform_type = str(pipe.get('transform', 'stft')).lower()
    if transform_type != 'stft':
        return

    entries = build_series(data, max_trajectories=max_trajectories, include_diff=False)
    if not entries:
        return

    pipeline_config = pipe.copy()
    pipeline_config.pop('transform', None)
    t = get_transform('stft', **pipeline_config, device=torch.device('cpu'))

    mats: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, s in entries:
        x = _squeeze_1d(s).detach().cpu().to(torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        _ = t.apply(x)
        mag = getattr(t, 'magnitude', None)
        ph = getattr(t, 'phase', None)
        if not torch.is_tensor(mag) or not torch.is_tensor(ph):
            continue
        mag0 = mag[0].detach().cpu().numpy()
        ph0 = ph[0].detach().cpu().numpy()
        mats.append((name, mag0, ph0))

    if not mats:
        return

    fig, axes = plt.subplots(len(mats), 2, figsize=(14, 3.0 * len(mats)))
    if len(mats) == 1:
        axes = np.array([axes])

    for row_idx, (name, mag, ph) in enumerate(mats):
        axm = axes[row_idx, 0]
        axp = axes[row_idx, 1]
        axm.imshow(mag, aspect='auto', origin='lower')
        axm.set_title(f"STFT magnitude: {name}")
        axm.set_xlabel('time frame')
        axm.set_ylabel('freq bin')

        im = axp.imshow(ph, aspect='auto', origin='lower', cmap='twilight', vmin=-math.pi, vmax=math.pi)
        axp.set_title(f"STFT phase: {name}")
        axp.set_xlabel('time frame')
        axp.set_ylabel('freq bin')
        fig.colorbar(im, ax=axp, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_saved_generated_mag_phase(
    config: dict,
    *,
    sample_dir: Path,
    out_path: Path,
    traj_idx: int = 0,
) -> None:
    """Plot the sampler-saved denormalized generated magnitude and predicted phase (if present)."""
    pipe = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
    if str(pipe.get('transform', 'stft')).lower() != 'stft':
        return

    gen_mag = _load_generated_denorm_mag(sample_dir, traj_idx=traj_idx, variant='default')
    gen_ph = _load_generated_denorm_phase(sample_dir, traj_idx=traj_idx, variant='default')
    if gen_mag is None or (not torch.is_tensor(gen_mag)):
        return
    if gen_ph is None or (not torch.is_tensor(gen_ph)):
        # Nothing to plot beyond magnitude (already covered by other plots).
        return

    gen_mag = gen_mag.detach().cpu().to(torch.float32)
    gen_ph = gen_ph.detach().cpu().to(torch.float32)
    while gen_mag.ndim > 3:
        gen_mag = gen_mag.squeeze(1)
    while gen_ph.ndim > 3:
        gen_ph = gen_ph.squeeze(1)
    if gen_mag.ndim == 2:
        gen_mag = gen_mag.unsqueeze(0)
    if gen_ph.ndim == 2:
        gen_ph = gen_ph.unsqueeze(0)

    # Plot first item.
    mag0 = gen_mag[0].detach().cpu().numpy()
    ph0 = gen_ph[0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    axes[0].imshow(mag0, aspect='auto', origin='lower')
    axes[0].set_title('Saved generated denorm magnitude')
    axes[0].set_xlabel('time frame')
    axes[0].set_ylabel('freq bin')

    im = axes[1].imshow(ph0, aspect='auto', origin='lower', cmap='twilight', vmin=-math.pi, vmax=math.pi)
    axes[1].set_title('Saved generated predicted phase (rad)')
    axes[1].set_xlabel('time frame')
    axes[1].set_ylabel('freq bin')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stats_bars(data: dict, out_path: Path) -> None:
    """Plot simple summary stats per signal as bar charts.

    Produces a compact, per-sample diagnostic: min/max and peak/rms.
    """
    entries = build_series(data, max_trajectories=1, include_diff=False)

    if not entries:
        return

    labels = [k for k, _ in entries]
    mins = []
    maxs = []
    peaks = []
    rmss = []
    for _, x in entries:
        x = x.detach().cpu().to(torch.float32).reshape(-1)
        mins.append(float(x.min().item()))
        maxs.append(float(x.max().item()))
        peaks.append(float(x.abs().max().item()))
        rmss.append(float(torch.sqrt(torch.mean(x ** 2)).item()))

    xs = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    ax = axes[0]
    w = 0.38
    ax.bar(xs - w / 2, mins, width=w, label='min')
    ax.bar(xs + w / 2, maxs, width=w, label='max')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_title('min / max')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(xs - w / 2, peaks, width=w, label='peak |x|')
    ax.bar(xs + w / 2, rmss, width=w, label='rms')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_title('peak & rms')
    ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_mag_matrix(config: dict, signal_1d: torch.Tensor) -> np.ndarray:
    pipeline_config = (config.get('pipeline_config') or {}).copy()
    transform_type = pipeline_config.pop('transform', 'stft')
    device = torch.device('cpu')
    t = get_transform(transform_type, **pipeline_config, device=device)
    return t.apply(signal_1d.unsqueeze(0)).squeeze(0).detach().cpu().numpy()


def plot_spectrograms(config: dict, data: dict, out_path: Path, max_trajectories: int = 1) -> None:
    items = build_series(data, max_trajectories=max_trajectories, include_diff=False)
    mags: list[tuple[str, np.ndarray]] = [(name, _compute_mag_matrix(config, s)) for name, s in items]

    q4 = data.get('4bit')
    gt = data.get('16bit_gt')
    diff = None
    if (q4 is not None) and (gt is not None):
        mag4 = _compute_mag_matrix(config, q4)
        mag16 = _compute_mag_matrix(config, gt)
        tmin = min(mag4.shape[-1], mag16.shape[-1])
        fmin = min(mag4.shape[-2], mag16.shape[-2])
        diff = np.abs(mag16[:fmin, :tmin] - mag4[:fmin, :tmin])
        mags.append(('diff_|target-cond4|', diff))

    fig, axes = plt.subplots(len(mags), 1, figsize=(12, 3 * len(mags)))
    if len(mags) == 1:
        axes = [axes]

    for ax, (name, mag) in zip(axes, mags):
        ax.imshow(mag, aspect='auto', origin='lower')
        ax.set_title(name)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if diff is not None:
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 3))
        ax2.imshow(diff, aspect='auto', origin='lower')
        ax2.set_title('diff_|target-cond4|')
        plt.tight_layout()
        diff_path = out_path.with_name(out_path.stem + '_diff.png')
        fig2.savefig(diff_path, dpi=150)
        plt.close(fig2)


def plot_spectrograms_alt(config: dict, data: dict, out_path: Path, max_trajectories: int = 1) -> None:
    items = build_series(data, max_trajectories=max_trajectories, include_diff=False)
    mags: list[tuple[str, np.ndarray]] = [(name, _compute_mag_matrix(config, s)) for name, s in items]

    q4 = data.get('4bit')
    gt = data.get('16bit_gt')
    if (q4 is not None) and (gt is not None):
        mag4 = _compute_mag_matrix(config, q4)
        mag16 = _compute_mag_matrix(config, gt)
        tmin = min(mag4.shape[-1], mag16.shape[-1])
        fmin = min(mag4.shape[-2], mag16.shape[-2])
        diff = np.abs(mag16[:fmin, :tmin] - mag4[:fmin, :tmin])
        mags.append(('diff_|target-cond4|', diff))

    fig, axes = plt.subplots(len(mags), 1, figsize=(12, 3 * len(mags)))
    if len(mags) == 1:
        axes = [axes]

    for ax, (name, mag) in zip(axes, mags):
        ax.pcolormesh(mag, shading='auto')
        ax.set_title(name)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_core_series_for_specs(data: dict, *, max_trajectories: int = 1) -> list[tuple[str, torch.Tensor]]:
    items: list[tuple[str, torch.Tensor]] = []
    if torch.is_tensor(data.get('raw')):
        items.append(('raw', data['raw']))
    if torch.is_tensor(data.get('4bit')):
        items.append(('cond_4bit', data['4bit']))
    if torch.is_tensor(data.get('16bit_gt')):
        tb = data.get('_target_bits')
        items.append((f"target_{int(tb)}bit" if tb is not None else 'target', data['16bit_gt']))

    trajs = data.get('trajectories') or []
    if isinstance(trajs, list):
        for i, t in enumerate(trajs[: int(max_trajectories)]):
            if torch.is_tensor(t):
                items.append((f'gen_{i}', t))

    # Common time-domain debug variants (continuous-valued).
    for k in ['gen_0_inv_rawspec', 'gen_0_inv_unitrange', 'gen_0_condminmax']:
        if torch.is_tensor(data.get(k)):
            items.append((k, data[k]))
    return items


def plot_spectrograms_lowpass(
    config: dict,
    data: dict,
    out_path: Path,
    *,
    cutoff_hz: float,
    sample_rate_hz: float,
    max_trajectories: int = 1,
    alt: bool = False,
) -> None:
    items = _build_core_series_for_specs(data, max_trajectories=max_trajectories)
    if not items:
        return

    filtered: list[tuple[str, torch.Tensor]] = []
    for name, s in items:
        s1 = _squeeze_1d(s).to(torch.float32).reshape(-1)
        filtered.append((name, _lowpass_fft(s1, cutoff_hz=float(cutoff_hz), sample_rate_hz=float(sample_rate_hz))))

    mags: list[tuple[str, np.ndarray]] = [(name, _compute_mag_matrix(config, s)) for name, s in filtered]

    # Optional filtered diff panel.
    cond = next((t for n, t in filtered if n == 'cond_4bit'), None)
    tgt = next((t for n, t in filtered if n.startswith('target')), None)
    diff = None
    if (cond is not None) and (tgt is not None):
        mag4 = _compute_mag_matrix(config, cond)
        magt = _compute_mag_matrix(config, tgt)
        tmin = min(mag4.shape[-1], magt.shape[-1])
        fmin = min(mag4.shape[-2], magt.shape[-2])
        diff = np.abs(magt[:fmin, :tmin] - mag4[:fmin, :tmin])
        mags.append((f'diff_|target-cond4|_lowpass@{float(cutoff_hz):g}Hz', diff))

    fig, axes = plt.subplots(len(mags), 1, figsize=(12, 3 * len(mags)))
    if len(mags) == 1:
        axes = [axes]

    for ax, (name, mag) in zip(axes, mags):
        if alt:
            ax.pcolormesh(mag, shading='auto')
        else:
            ax.imshow(mag, aspect='auto', origin='lower')
        ax.set_title(name)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_metrics(data: dict) -> dict | None:
    gt = data.get('16bit_gt')
    cond_4bit = data.get('4bit')
    gen0_q4 = data.get('gen_0_q4')
    gen0_qT = None
    if data.get('_target_bits') is not None:
        gen0_qT = data.get(f"gen_0_q{int(data.get('_target_bits'))}")

    gen0_condminmax_q4 = data.get('gen_0_condminmax_q4')
    gen0_condminmax_qT = None
    if data.get('_target_bits') is not None:
        gen0_condminmax_qT = data.get(f"gen_0_condminmax_q{int(data.get('_target_bits'))}")

    trajectories = data.get('trajectories') or []

    if (gt is None) or (cond_4bit is None):
        return None

    target_bits = data.get('_target_bits')
    raw_qT = None
    if target_bits is not None:
        raw_qT = data.get(f'raw_q{int(target_bits)}')

    min_len = min(int(gt.numel()), int(cond_4bit.numel()))
    if gen0_q4 is not None:
        min_len = min(min_len, int(gen0_q4.numel()))
    if gen0_qT is not None:
        min_len = min(min_len, int(gen0_qT.numel()))
    if gen0_condminmax_q4 is not None:
        min_len = min(min_len, int(gen0_condminmax_q4.numel()))
    if gen0_condminmax_qT is not None:
        min_len = min(min_len, int(gen0_condminmax_qT.numel()))
    for traj in trajectories:
        min_len = min(min_len, int(traj.numel()))

    gt = gt[:min_len]
    cond_4bit = cond_4bit[:min_len]
    if gen0_q4 is not None:
        gen0_q4 = gen0_q4[:min_len]
    if gen0_qT is not None:
        gen0_qT = gen0_qT[:min_len]
    if gen0_condminmax_q4 is not None:
        gen0_condminmax_q4 = gen0_condminmax_q4[:min_len]
    if gen0_condminmax_qT is not None:
        gen0_condminmax_qT = gen0_condminmax_qT[:min_len]

    metrics: dict = {
        'target_bits': int(target_bits) if target_bits is not None else None,
        'mse_cond4_vs_target': float(torch.mean((cond_4bit - gt) ** 2)),
        'trajectories': [],
    }

    # If baselines computed, include the implied quantizer range.
    if '_time_range_min' in data and '_time_range_max' in data:
        metrics['cond_time_range_min'] = float(data['_time_range_min'])
        metrics['cond_time_range_max'] = float(data['_time_range_max'])

    cond_stats = _unique_level_stats(cond_4bit)
    metrics['cond4_levels_used'] = cond_stats.get('unique_levels_used')
    metrics['cond4_estimated_step'] = cond_stats.get('estimated_step')

    if gen0_q4 is not None:
        metrics['mse_quant4(gen0)_vs_target'] = float(torch.mean((gen0_q4 - gt) ** 2))
        metrics['gen0_q4_levels_used'] = _unique_level_stats(gen0_q4).get('unique_levels_used')
    if gen0_qT is not None and target_bits is not None:
        metrics[f'mse_quant{int(target_bits)}(gen0)_vs_target'] = float(torch.mean((gen0_qT - gt) ** 2))

    if gen0_condminmax_q4 is not None:
        metrics['mse_quant4(gen0_condminmax)_vs_target'] = float(torch.mean((gen0_condminmax_q4 - gt) ** 2))
    if gen0_condminmax_qT is not None and target_bits is not None:
        metrics[f'mse_quant{int(target_bits)}(gen0_condminmax)_vs_target'] = float(torch.mean((gen0_condminmax_qT - gt) ** 2))

    for i, traj in enumerate(trajectories):
        traj = traj[:min_len]
        traj_metrics = {
            'trajectory_idx': i,
            'mse_vs_target': float(torch.mean((traj - gt) ** 2)),
            'mse_vs_cond4': float(torch.mean((traj - cond_4bit) ** 2)),
        }
        metrics['trajectories'].append(traj_metrics)

    return metrics


def print_metrics(metrics: dict | None) -> None:
    if not metrics:
        print('metrics: skipped (need cond_4bit + target + optional trajectories)')
        return

    target_bits = metrics.get('target_bits')
    target_label = f"target_{int(target_bits)}bit" if target_bits is not None else "target"

    print(f"mse_cond4_vs_{target_label}: {metrics['mse_cond4_vs_target']:.6f}")
    if 'cond_time_range_min' in metrics and 'cond_time_range_max' in metrics:
        print(
            "cond_quant_range: "
            f"[{metrics['cond_time_range_min']:.6f}, {metrics['cond_time_range_max']:.6f}]"
        )
    if 'mse_quant4(gen0)_vs_target' in metrics:
        print(f"mse_quant4(gen0)_vs_{target_label}: {metrics['mse_quant4(gen0)_vs_target']:.6f}")
    if target_bits is not None and f"mse_quant{int(target_bits)}(gen0)_vs_target" in metrics:
        print(
            f"mse_quant{int(target_bits)}(gen0)_vs_{target_label}: "
            f"{metrics[f'mse_quant{int(target_bits)}(gen0)_vs_target']:.6f}"
        )

    if 'mse_quant4(gen0_condminmax)_vs_target' in metrics:
        print(f"mse_quant4(gen0_condminmax)_vs_{target_label}: {metrics['mse_quant4(gen0_condminmax)_vs_target']:.6f}")
    if target_bits is not None and f"mse_quant{int(target_bits)}(gen0_condminmax)_vs_target" in metrics:
        print(
            f"mse_quant{int(target_bits)}(gen0_condminmax)_vs_{target_label}: "
            f"{metrics[f'mse_quant{int(target_bits)}(gen0_condminmax)_vs_target']:.6f}"
        )

    if 'cond4_levels_used' in metrics:
        print(f"cond4_unique_levels_used: {int(metrics['cond4_levels_used'])}")
    if 'cond4_estimated_step' in metrics and metrics['cond4_estimated_step'] is not None:
        print(f"cond4_estimated_step: {float(metrics['cond4_estimated_step']):.6f}")

    for traj in metrics['trajectories']:
        denom = metrics['mse_cond4_vs_target']
        improvement = ((denom - traj['mse_vs_target']) / denom * 100) if denom != 0 else float('nan')
        print(
            f"traj_{traj['trajectory_idx']}: mse_vs_{target_label}={traj['mse_vs_target']:.6f}, "
            f"improvement_vs_cond4={improvement:+.2f}%"
        )


def _tensor_stats(x: torch.Tensor | None) -> dict:
    if x is None:
        return {}
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {'numel': 0}
    return {
        'shape': tuple(x.shape),
        'numel': int(x.numel()),
        'min': float(torch.min(x)),
        'max': float(torch.max(x)),
        'mean': float(torch.mean(x)),
        'std': float(torch.std(x)),
    }


def print_details(data: dict, config: dict | None = None) -> None:
    config = config or {}
    print("details:")
    print(f"  sampler_type: {data.get('_sampler_type')!r}")
    print(f"  target_bits: {data.get('_target_bits')!r}")
    print(f"  gen_time_key: {data.get('_gen_time_key')!r}")
    if data.get('_gen_time_key_resolved') is not None:
        print(f"  gen_time_key_resolved: {data.get('_gen_time_key_resolved')!r}")

    # Time-domain signals (analysis uses generic keys)
    cond_raw = data.get('raw')
    cond_4bit = data.get('4bit')
    target = data.get('16bit_gt')

    if isinstance(cond_raw, torch.Tensor):
        s = _tensor_stats(cond_raw)
        print(f"  raw (condition/target): numel={s.get('numel')}, min={s.get('min'):.6f}, max={s.get('max'):.6f}, std={s.get('std'):.6f}")
    else:
        print("  raw: (missing)")

    if isinstance(cond_4bit, torch.Tensor):
        s = _tensor_stats(cond_4bit)
        q = _unique_level_stats(cond_4bit)
        print(f"  cond_4bit: min={s.get('min'):.6f}, max={s.get('max'):.6f}, levels_used={q.get('unique_levels_used')}, est_step={q.get('estimated_step')}")
    else:
        print("  cond_4bit: (missing)")

    if isinstance(target, torch.Tensor):
        s = _tensor_stats(target)
        print(f"  target (raw/gt): min={s.get('min'):.6f}, max={s.get('max'):.6f}, std={s.get('std'):.6f}")
    else:
        print("  target: (missing)")

    # Quantizer ranges inferred/loaded
    if '_time_range_min' in data and '_time_range_max' in data:
        print(f"  inferred_time_quant_range: [{float(data['_time_range_min']):.6f}, {float(data['_time_range_max']):.6f}]")

    if '_gen0_time_range_min' in data and '_gen0_time_range_max' in data:
        print(
            f"  gen0_time_quant_range_used: "
            f"[{float(data['_gen0_time_range_min']):.6f}, {float(data['_gen0_time_range_max']):.6f}]"
        )

    # Helpful warning when gen0 blows up relative to the conditioning/target scale.
    try:
        if isinstance(target, torch.Tensor) and isinstance(data.get('trajectories'), list) and data['trajectories']:
            gen0 = data['trajectories'][0]
            if isinstance(gen0, torch.Tensor):
                tgt_peak = float(target.detach().abs().max().item())
                gen_peak = float(gen0.detach().abs().max().item())
                if tgt_peak > 0 and gen_peak / tgt_peak >= 2.0:
                    print(f"  WARNING: gen0_peak={gen_peak:.6f} is {gen_peak / tgt_peak:.2f}x target_peak={tgt_peak:.6f}")
    except Exception:
        pass

    # Generated outputs
    trajectories = data.get('trajectories') or []
    if isinstance(trajectories, list) and trajectories:
        t0 = trajectories[0]
        if isinstance(t0, torch.Tensor):
            s = _tensor_stats(t0)
            print(f"  gen0_time: min={s.get('min'):.6f}, max={s.get('max'):.6f}, std={s.get('std'):.6f}")
        print(f"  num_trajectories_loaded: {len(trajectories)}")
    else:
        print("  trajectories: (none)")

    # Generated quantized views (if present)
    if isinstance(data.get('gen_0_q4'), torch.Tensor):
        q = _unique_level_stats(data['gen_0_q4'])
        print(f"  gen0_q4: levels_used={q.get('unique_levels_used')}, est_step={q.get('estimated_step')}")
    tgt_bits = data.get('_target_bits')
    if tgt_bits is not None:
        k = f"gen_0_q{int(tgt_bits)}"
        if isinstance(data.get(k), torch.Tensor):
            q = _unique_level_stats(data[k])
            print(f"  {k}: levels_used={q.get('unique_levels_used')}, est_step={q.get('estimated_step')}")

    # Config echoes (helpful when debugging mismatches)
    if isinstance(config, dict) and config:
        print(f"  cfg.time_quantization_mode: {config.get('time_quantization_mode')!r}")
        print(f"  cfg.bit_size (cond bits): {config.get('bit_size')!r}")
        print(f"  cfg.test_holdout_count: {config.get('test_holdout_count')!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze one sampled condition vs generated output")
    p.add_argument("--results-root", type=str, default="diffusion/results", help="Root results dir (default: diffusion/results)")
    p.add_argument("--version", type=str, default="V1", help="Experiment version folder (V1/V2)")
    p.add_argument("--run-id", type=str, required=True, help="Run id folder under diffusion/results/<version>/")
    p.add_argument("--sampler-type", type=str, default="auto", choices=["auto", "ddpm", "ddim"], help="Which sampler subfolder to use")
    p.add_argument("--sample-idx", type=int, default=0, help="Sample index (sample_<idx>)")
    p.add_argument(
        "--all-samples",
        action="store_true",
        help="If set, analyze multiple samples in one run (sample_<start>..sample_<start+num-1>).",
    )
    p.add_argument(
        "--sample-idx-start",
        type=int,
        default=0,
        help="Start sample index when --all-samples is set (default: 0).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples to analyze when --all-samples is set (default: 16).",
    )
    p.add_argument("--max-trajectories", type=int, default=1, help="How many trajectories to include in plots")
    p.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Write only the generated vs 4-bit vs target comparisons: raw spectrogram mag/phase, time-domain, plus a metrics JSON with pre/post MSEs. Avoids any re-inversion of sampler outputs.",
    )
    p.add_argument(
        "--compare-lp-hz",
        type=float,
        default=40.0,
        help="For --compare: low-pass cutoff (Hz) for the diffusion_generated_lp variant (default 40.0).",
    )
    p.add_argument(
        "--compare-norm-preclamp",
        action="store_true",
        help=(
            "For --compare: write a debug plot that shows the condition/target normalization values BEFORE clamping to [-1,1] "
            "(using condition-derived scaling). Helps diagnose saturation from range mismatch."
        ),
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Compact analysis: write ~5 PNGs (time, FFT mag+phase, spec mag/phase, metrics) comparing a few inversion+scaling options plus JSON losses.",
    )
    p.add_argument(
        "--minimal",
        action="store_true",
        help="Minimal analysis: only plots cond4/target11/gen0 and gen0 with tapered mag low-pass (and writes MSE metrics).",
    )
    p.add_argument("--quantize", action="store_true", help="Quantize generated trajectories to 16-bit for comparison")
    p.add_argument("--target-bits", type=int, default=11, help="Label/assume target bit-depth for plots/metrics (default: 11)")
    p.add_argument("--print-details", action="store_true", help="Print debug details about loaded/reconstructed signals and quantization")
    p.add_argument(
        "--gen-time-key",
        type=str,
        default="auto",
        help="Which generated time-domain field to analyze. Use 'auto' to prefer peak-rescaled if present (default: auto).",
    )
    p.add_argument(
        "--lowpass-hz",
        type=float,
        default=None,
        help="If set, also save a time-series plot overlaying an FFT low-pass filtered view (Hz).",
    )
    p.add_argument(
        "--sample-rate",
        type=float,
        default=360.0,
        help="Sample rate for frequency axes / low-pass (Hz). Default 360 for MIT-BIH.",
    )

    p.add_argument(
        "--phase-diag",
        action="store_true",
        help="If set, reconstruct gen magnitude with different phase choices (raw-cond phase vs Griffin–Lim vs hybrid) and save diagnostic plots.",
    )
    p.add_argument(
        "--griffin-lim-iters",
        type=int,
        default=32,
        help="Iterations for Griffin–Lim phase estimation (used when --phase-diag).",
    )
    p.add_argument(
        "--phase-hybrid-cutoff-hz",
        type=float,
        default=None,
        help="If set with --phase-diag, use raw-cond phase for bins below this Hz and Griffin–Lim phase above it.",
    )
    p.add_argument(
        "--mag-lowpass-hz",
        type=float,
        default=None,
        help="If set with --phase-diag, low-pass the generated magnitude (Hz) before ISTFT to test band-limiting fixes.",
    )
    p.add_argument(
        "--mag-lowpass-kind",
        type=str,
        default="hard",
        choices=["hard", "taper"],
        help="How to apply the magnitude low-pass mask: hard zeroing or tapered rolloff.",
    )
    p.add_argument(
        "--mag-lowpass-transition-bins",
        type=int,
        default=8,
        help="For --mag-lowpass-kind=taper: number of bins for the rolloff region.",
    )
    p.add_argument(
        "--phase-diag-time-lowpass-hz",
        type=float,
        default=None,
        help="If set with --phase-diag, also low-pass reconstructed time signals (Hz) for a few key variants and include in the plots.",
    )
    p.add_argument(
        "--phase-diag-primary-phase",
        type=str,
        default="auto",
        choices=["auto", "predicted", "raw_cond", "griffinlim", "hybrid"],
        help="For --phase-diag: which phase source to treat as the primary recon (auto picks predicted if available).",
    )
    p.add_argument(
        "--phase-diag-peak-rescale",
        type=str,
        default="none",
        choices=["none", "cond4", "raw", "target"],
        help="For --phase-diag: optionally peak-rescale non-target reconstructions to match cond_4bit, raw, or target peak.",
    )
    p.add_argument(
        "--phase-diag-share-ylim",
        action="store_true",
        help="For --phase-diag: share y-limits using target range (helps compare amplitudes).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    results_root = Path(args.results_root)
    # Mirror training/sampling convention:
    #   results_dir=<diffusion/results>/<version>
    #   version=<run_id>
    results_dir = results_root / str(args.version)
    version = str(args.run_id)
    sample_idx = int(args.sample_idx)

    sampler_type = None if str(args.sampler_type) == "auto" else str(args.sampler_type)

    config: dict = {}
    config_path = results_dir / version / 'config.pkl'
    if not config_path.exists():
        config_path = results_dir / 'config.pkl'
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    else:
        print(f"config: missing ({config_path}); skipping spectrogram plots")

    def _analyze_one(sample_idx: int) -> dict | None:
        # Force sampler selection by setting sampler_type for resolver.
        sample_dir, detected_sampler = _resolve_sample_dir(results_dir, version, sample_idx, sampler_type=sampler_type)
        if not sample_dir.exists():
            return None

        data = load_sample_data(
            results_dir,
            version,
            sample_idx,
            sample_dir=sample_dir,
            sampler_type=detected_sampler,
            gen_time_key=str(args.gen_time_key),
        )
        data['_target_bits'] = int(args.target_bits) if args.target_bits is not None else None
        _fill_missing_baselines_from_raw(data, config)

        # Save per-sample outputs in a dedicated subdirectory.
        plot_dir = results_dir / version / 'analysis' / f'sample_{sample_idx}'
        plot_dir.mkdir(parents=True, exist_ok=True)

        sampler_tag = detected_sampler or data.get('_sampler_type')
        tag = f"_{sampler_tag}" if isinstance(sampler_tag, str) and sampler_tag else ""

        if bool(args.compare):
            # Compare using sampler-saved artifacts when available to avoid double inversion.
            # Time-domain baselines (preferred): sampler-saved target/cond.
            cond_time = _load_tensor_if_exists(sample_dir / 'cond_time_4bit.pt', key_candidates=['cond_time_4bit'])
            target_time = _load_tensor_if_exists(sample_dir / 'target_time.pt', key_candidates=['target_time'])
            if cond_time is None:
                cond_time = data.get('4bit')
            if target_time is None:
                cand = data.get('16bit_gt')
                if torch.is_tensor(cand):
                    target_time = cand
                else:
                    target_time = data.get('raw')

            # Generated time-domain: sampler-saved (already inverted).
            gen_time = None
            # In compare mode load the sampler-saved time-domain output directly.
            try:
                traj_obj = torch.load(sample_dir / 'trajectory_0.pt', map_location='cpu')
                if isinstance(traj_obj, dict):
                    if torch.is_tensor(traj_obj.get('time_domain')):
                        gen_time = traj_obj.get('time_domain')
            except Exception:
                pass
            if gen_time is None:
                # Fallback if trajectory_0.pt isn't present.
                if isinstance(data.get('trajectories_inv_unitrange'), list) and data['trajectories_inv_unitrange']:
                    gen_time = data['trajectories_inv_unitrange'][0]
                elif isinstance(data.get('trajectories'), list) and data['trajectories']:
                    gen_time = data['trajectories'][0]

            if (not torch.is_tensor(cond_time)) or (not torch.is_tensor(target_time)) or (not torch.is_tensor(gen_time)):
                print(f"sample_{sample_idx}{tag}: compare skipped (missing time-domain artifacts)")
                return None

            cond_time = _squeeze_1d(cond_time.detach().cpu().to(torch.float32))
            target_time = _squeeze_1d(target_time.detach().cpu().to(torch.float32))
            gen_time = _squeeze_1d(gen_time.detach().cpu().to(torch.float32))

            # Spectrogram magnitude/phase for baselines (computed from time).
            cond_mag, cond_phase = _compare_compute_mag_phase(config=config, x=cond_time)
            target_mag, target_phase = _compare_compute_mag_phase(config=config, x=target_time)
            if cond_mag is None or target_mag is None:
                print(f"sample_{sample_idx}{tag}: compare skipped (transform != stft)")
                return None

            # Generated pre-inversion magnitude: prefer sampler-saved denormalized magnitude.
            gen_mag = None
            try:
                traj_obj = torch.load(sample_dir / 'trajectory_0.pt', map_location='cpu')
                if isinstance(traj_obj, dict) and torch.is_tensor(traj_obj.get('spectrogram_denorm_mag')):
                    gm = traj_obj['spectrogram_denorm_mag'].detach().cpu().to(torch.float32)
                    while gm.ndim > 3:
                        gm = gm.squeeze(1)
                    if gm.ndim == 3:
                        gen_mag = gm[0]
                    elif gm.ndim == 2:
                        gen_mag = gm
            except Exception:
                gen_mag = None
            if gen_mag is None:
                # Fallback: compute from generated waveform (this is post-inversion, but better than nothing).
                gen_mag, _ = _compare_compute_mag_phase(config=config, x=gen_time)

            # Align spectrogram shapes.
            cond_mag = cond_mag.detach().cpu().to(torch.float32)
            target_mag = target_mag.detach().cpu().to(torch.float32)
            gen_mag = gen_mag.detach().cpu().to(torch.float32)
            f = min(int(cond_mag.shape[0]), int(target_mag.shape[0]), int(gen_mag.shape[0]))
            t = min(int(cond_mag.shape[1]), int(target_mag.shape[1]), int(gen_mag.shape[1]))
            cond_mag = cond_mag[:f, :t]
            target_mag = target_mag[:f, :t]
            gen_mag = gen_mag[:f, :t]
            cond_phase = cond_phase[:f, :t]
            target_phase = target_phase[:f, :t]

            # Compute comparison metrics.
            def _compute_metrics(*, gen_mag_v: torch.Tensor, gen_time_v: torch.Tensor) -> dict:
                def _rms(x: torch.Tensor) -> float:
                    x = x.detach().to(torch.float32).reshape(-1)
                    if int(x.numel()) <= 0:
                        return float('nan')
                    return float(torch.sqrt(torch.mean(x * x)).item())

                mse_spec_cond = float(torch.mean((cond_mag - target_mag) ** 2).item())
                mse_spec_gen = float(torch.mean((gen_mag_v - target_mag) ** 2).item())
                mse_spec_cond_log = float(torch.mean((torch.log1p(torch.clamp(cond_mag, min=0.0)) - torch.log1p(torch.clamp(target_mag, min=0.0))) ** 2).item())
                mse_spec_gen_log = float(torch.mean((torch.log1p(torch.clamp(gen_mag_v, min=0.0)) - torch.log1p(torch.clamp(target_mag, min=0.0))) ** 2).item())
                mse_time_cond = _mse_aligned(cond_time, target_time)
                mse_time_gen = _mse_aligned(gen_time_v, target_time)

                pair_ct = _aligned_pair(cond_time, target_time)
                pair_gt = _aligned_pair(gen_time_v, target_time)
                corr_time_cond = float('nan')
                corr_time_gen = float('nan')
                peak_ratio_cond = float('nan')
                peak_ratio_gen = float('nan')
                rms_ratio_cond = float('nan')
                rms_ratio_gen = float('nan')
                if pair_ct is not None:
                    c, tt = pair_ct
                    corr_time_cond = _pearson_corr(c, tt)
                    tpk = float(tt.abs().max().item())
                    cpk = float(c.abs().max().item())
                    peak_ratio_cond = float(cpk / (tpk + 1e-8))
                    rms_ratio_cond = float(_rms(c) / (_rms(tt) + 1e-8))
                if pair_gt is not None:
                    g, tt = pair_gt
                    corr_time_gen = _pearson_corr(g, tt)
                    tpk = float(tt.abs().max().item())
                    gpk = float(g.abs().max().item())
                    peak_ratio_gen = float(gpk / (tpk + 1e-8))
                    rms_ratio_gen = float(_rms(g) / (_rms(tt) + 1e-8))

                cm = torch.clamp(cond_mag.to(torch.float32), min=0.0).reshape(-1)
                tm = torch.clamp(target_mag.to(torch.float32), min=0.0).reshape(-1)
                gm = torch.clamp(gen_mag_v.to(torch.float32), min=0.0).reshape(-1)
                spec_mean_ratio_cond = float((torch.mean(cm) / (torch.mean(tm) + 1e-8)).item())
                spec_mean_ratio_gen = float((torch.mean(gm) / (torch.mean(tm) + 1e-8)).item())
                spec_p95_ratio_cond = float((torch.quantile(cm, 0.95) / (torch.quantile(tm, 0.95) + 1e-8)).item())
                spec_p95_ratio_gen = float((torch.quantile(gm, 0.95) / (torch.quantile(tm, 0.95) + 1e-8)).item())

                out = {
                    'mse_spec_mag(cond4_vs_target)': float(mse_spec_cond),
                    'mse_spec_mag(generated_vs_target)': float(mse_spec_gen),
                    'mse_spec_log1p(cond4_vs_target)': float(mse_spec_cond_log),
                    'mse_spec_log1p(generated_vs_target)': float(mse_spec_gen_log),
                    'mse_time(cond4_vs_target)': float(mse_time_cond),
                    'mse_time(generated_vs_target)': float(mse_time_gen),
                    'corr_time(cond4_vs_target)': float(corr_time_cond),
                    'corr_time(generated_vs_target)': float(corr_time_gen),
                    'peak_ratio_time(cond4_vs_target)': float(peak_ratio_cond),
                    'peak_ratio_time(generated_vs_target)': float(peak_ratio_gen),
                    'rms_ratio_time(cond4_vs_target)': float(rms_ratio_cond),
                    'rms_ratio_time(generated_vs_target)': float(rms_ratio_gen),
                    'mean_ratio_spec_mag(cond4_vs_target)': float(spec_mean_ratio_cond),
                    'mean_ratio_spec_mag(generated_vs_target)': float(spec_mean_ratio_gen),
                    'p95_ratio_spec_mag(cond4_vs_target)': float(spec_p95_ratio_cond),
                    'p95_ratio_spec_mag(generated_vs_target)': float(spec_p95_ratio_gen),
                }
                if np.isfinite(float(mse_spec_cond)) and float(mse_spec_cond) != 0.0:
                    out['improvement_pct_spec_mag'] = float((mse_spec_cond - mse_spec_gen) / mse_spec_cond * 100.0)
                if np.isfinite(float(mse_time_cond)) and float(mse_time_cond) != 0.0:
                    out['improvement_pct_time'] = float((mse_time_cond - mse_time_gen) / mse_time_cond * 100.0)
                return out

            metrics = _compute_metrics(gen_mag_v=gen_mag, gen_time_v=gen_time)
            annotation = " | ".join([
                f"time_mse(gen,target)={float(metrics.get('mse_time(generated_vs_target)', float('nan'))):.6g}",
                f"spec_mse(gen,target)={float(metrics.get('mse_spec_mag(generated_vs_target)', float('nan'))):.6g}",
                f"corr_time(gen,target)={float(metrics.get('corr_time(generated_vs_target)', float('nan'))):.6g}",
                f"peak_ratio(gen/target)={float(metrics.get('peak_ratio_time(generated_vs_target)', float('nan'))):.6g}",
                f"rms_ratio(gen/target)={float(metrics.get('rms_ratio_time(generated_vs_target)', float('nan'))):.6g}",
                f"baseline cond4 time_mse={float(metrics.get('mse_time(cond4_vs_target)', float('nan'))):.6g} "
                f"spec_mse={float(metrics.get('mse_spec_mag(cond4_vs_target)', float('nan'))):.6g}",
            ])

            # Compute STFT mag from generated time-domain for side-by-side with saved denorm-mag.
            gen_mag_from_time = None
            try:
                gm_t, _ = _compare_compute_mag_phase(config=config, x=gen_time)
                if gm_t is not None:
                    gm_t = gm_t.detach().cpu().to(torch.float32)
                    ff = min(int(cond_mag.shape[0]), int(target_mag.shape[0]), int(gen_mag.shape[0]), int(gm_t.shape[0]))
                    tt = min(int(cond_mag.shape[1]), int(target_mag.shape[1]), int(gen_mag.shape[1]), int(gm_t.shape[1]))
                    gen_mag_from_time = gm_t[:ff, :tt]
            except Exception:
                gen_mag_from_time = None

            plot_compare_spec_mag(
                cond_mag=cond_mag,
                target_mag=target_mag,
                gen_mag=gen_mag,
                gen_mag_from_time=gen_mag_from_time,
                out_path=plot_dir / f'compare_spec_mag{tag}.png',
                log_scale=True,
                annotation=annotation,
            )
            plot_compare_spec_phase(
                cond_phase=cond_phase,
                target_phase=target_phase,
                out_path=plot_dir / f'compare_spec_phase{tag}.png',
                annotation=annotation,
            )

            # --- 11-bit re-quantization of diffusion generated ---
            gen_11bit = None
            try:
                gt = gen_time.detach().to(torch.float32).reshape(-1)
                peak_g = float(gt.abs().max().item())
                if peak_g > 0:
                    q11 = UniformQuantizer(bits=11, range_min=-peak_g, range_max=peak_g)
                    gen_11bit = _squeeze_1d(q11.quantize(gt.unsqueeze(0)).squeeze(0))
            except Exception:
                gen_11bit = None

            # --- Low-pass filtered diffusion generated (cutoff = ECG band 40 Hz) ---
            gen_lp = None
            try:
                lp_hz = float(getattr(args, 'compare_lp_hz', 40.0) or 40.0)
                gen_lp = _squeeze_1d(_lowpass_fft(gen_time.detach().to(torch.float32), cutoff_hz=lp_hz, sample_rate_hz=float(args.sample_rate)))
            except Exception:
                gen_lp = None

            extra = []
            if gen_11bit is not None:
                extra.append(('diffusion_generated_11bit', gen_11bit))
            if gen_lp is not None:
                extra.append((f'diffusion_generated_lp{int(getattr(args, "compare_lp_hz", 40))}hz', gen_lp))

            plot_compare_time(
                cond_time=cond_time,
                target_time=target_time,
                gen_time=gen_time,
                extra_series=extra if extra else None,
                out_path=plot_dir / f'compare_time{tag}.png',
                annotation=annotation,
            )

            # --- Statistics bar chart ---
            pipe_cfg = (config.get('pipeline_config') or {}) if isinstance(config, dict) else {}
            n_fft_bar = int(pipe_cfg.get('n_fft', 254))
            time_series_bar: dict[str, torch.Tensor] = {
                'cond_4bit': cond_time,
                'target_11bit': target_time,
                'diffusion_generated': gen_time,
            }
            spec_series_bar: dict[str, torch.Tensor] = {
                'cond_4bit': cond_mag,
                'target_11bit': target_mag,
                'diffusion_generated': gen_mag,
            }
            if gen_11bit is not None:
                time_series_bar['diffusion_generated_11bit'] = gen_11bit
                gm_11bit, _ = _compare_compute_mag_phase(config=config, x=gen_11bit)
                if gm_11bit is not None:
                    f11 = min(int(gen_mag.shape[0]), int(gm_11bit.shape[0]))
                    t11 = min(int(gen_mag.shape[1]), int(gm_11bit.shape[1]))
                    spec_series_bar['diffusion_generated_11bit'] = gm_11bit[:f11, :t11]
            if gen_lp is not None:
                lp_key = f'diffusion_generated_lp{int(getattr(args, "compare_lp_hz", 40))}hz'
                time_series_bar[lp_key] = gen_lp
                gm_lp, _ = _compare_compute_mag_phase(config=config, x=gen_lp)
                if gm_lp is not None:
                    flp = min(int(gen_mag.shape[0]), int(gm_lp.shape[0]))
                    tlp = min(int(gen_mag.shape[1]), int(gm_lp.shape[1]))
                    spec_series_bar[lp_key] = gm_lp[:flp, :tlp]

            stats_out = plot_compare_stats_bars(
                series=time_series_bar,
                spec_series=spec_series_bar,
                out_path=plot_dir / f'compare_stats_bars{tag}.png',
                sample_rate_hz=float(args.sample_rate),
                n_fft=n_fft_bar,
                annotation=annotation,
            )

            out_json = {
                'sample_dir': str(sample_dir),
                'pipeline_transform': str(((config.get('pipeline_config') or {}).get('transform', '')) if isinstance(config, dict) else ''),
                'sampler_type': str(sampler_tag) if sampler_tag is not None else None,
                'metrics': metrics,
                'stats': stats_out,
            }

            # Optional: debug plot of normalization values BEFORE clamp to [-1,1].
            if bool(getattr(args, 'compare_norm_preclamp', False)):
                try:
                    # Compute per-sample cond-derived normalization stats from cond_mag (onesided STFT magnitude).
                    mag_norm_mode = str((config.get('mag_norm_mode', 'minmax') if isinstance(config, dict) else 'minmax') or 'minmax').strip().lower()
                    mag_norm_eps = float((config.get('mag_norm_epsilon', 1e-8) if isinstance(config, dict) else 1e-8) or 1e-8)
                    if mag_norm_eps <= 0 or (not math.isfinite(mag_norm_eps)):
                        mag_norm_eps = 1e-8

                    cmag = cond_mag.detach().to(torch.float32)
                    tmag = target_mag.detach().to(torch.float32)
                    if mag_norm_mode in {'absmax', 'peak', 'max'}:
                        cond_peak = cmag.max()
                        denom = float(cond_peak.item())
                        if denom < mag_norm_eps:
                            denom = 1.0
                        # 0..peak -> [-1,1]
                        cond_norm_pre = (cmag / denom) * 2.0 - 1.0
                        target_norm_pre = (tmag / denom) * 2.0 - 1.0
                    else:
                        cmin = float(cmag.min().item())
                        cmax = float(cmag.max().item())
                        denom = float(cmax - cmin)
                        if abs(denom) < mag_norm_eps:
                            denom = 1.0
                        cond_norm_pre = ((cmag - cmin) / denom) * 2.0 - 1.0
                        target_norm_pre = ((tmag - cmin) / denom) * 2.0 - 1.0

                    # Load a saved generated normalized tensor if available.
                    gen_norm_saved = None
                    try:
                        traj0 = torch.load(sample_dir / 'trajectory_0.pt', map_location='cpu')
                        if isinstance(traj0, dict):
                            gen_norm_saved = traj0.get('spectrogram_norm')
                            if torch.is_tensor(gen_norm_saved):
                                # [B,1,F,T] -> [F,T]
                                while gen_norm_saved.ndim > 2:
                                    gen_norm_saved = gen_norm_saved.squeeze(0)
                                    if gen_norm_saved.ndim == 3 and gen_norm_saved.shape[0] == 1:
                                        gen_norm_saved = gen_norm_saved[0]
                                    if gen_norm_saved.ndim == 3 and gen_norm_saved.shape[0] != 1:
                                        # If shape [C,F,T], take first channel
                                        gen_norm_saved = gen_norm_saved[0]
                    except Exception:
                        gen_norm_saved = None

                    debug_stats = plot_compare_norm_preclamp_hist(
                        cond_norm_pre=cond_norm_pre,
                        target_norm_pre=target_norm_pre,
                        gen_norm=gen_norm_saved,
                        out_path=plot_dir / f'compare_norm_preclamp_hist{tag}.png',
                        title=f"Pre-clamp normalization values (mode={mag_norm_mode})",
                    )
                    out_json['norm_preclamp_debug'] = {
                        'mag_norm_mode': str(mag_norm_mode),
                        'mag_norm_epsilon': float(mag_norm_eps),
                        **debug_stats,
                    }
                except Exception as e:
                    out_json['norm_preclamp_debug'] = {'error': str(e)}

            with open(plot_dir / f'compare_metrics{tag}.json', 'w') as f:
                json.dump(out_json, f, indent=2)

            print(
                f"sample_{sample_idx}{tag}: "
                f"time_mse={metrics.get('mse_time(generated_vs_target)', float('nan')):.6g} "
                f"spec_mse={metrics.get('mse_spec_mag(generated_vs_target)', float('nan')):.6g} "
                f"peak_ratio={metrics.get('peak_ratio_time(generated_vs_target)', float('nan')):.6g} "
                f"rms_ratio={metrics.get('rms_ratio_time(generated_vs_target)', float('nan')):.6g}"
            )
            return out_json

        # Minimal mode: only the 4 core views + MSEs.
        if bool(args.compact):
            out = plot_compact_suite(
                config=config,
                data=data,
                sample_dir=sample_dir,
                out_dir=plot_dir,
                sample_rate_hz=float(args.sample_rate),
                griffin_lim_iters=int(args.griffin_lim_iters),
                hybrid_cutoff_hz=(float(args.phase_hybrid_cutoff_hz) if args.phase_hybrid_cutoff_hz is not None else 40.0),
                mag_lowpass_hz=(float(args.mag_lowpass_hz) if args.mag_lowpass_hz is not None else None),
                mag_lowpass_kind=str(args.mag_lowpass_kind),
                mag_lowpass_transition_bins=int(args.mag_lowpass_transition_bins),
                title_prefix='',
            )
            if out:
                # Print a tiny summary for quick iteration.
                try:
                    m = out.get('metrics') if isinstance(out, dict) else None
                    if isinstance(m, dict) and m:
                        best = None
                        for inv, d in m.items():
                            if isinstance(d, dict) and 'cond4_peak' in d:
                                v = float(d['cond4_peak'])
                                if best is None or v < best[1]:
                                    best = (inv, v)
                        if best is not None:
                            print(f"sample_{sample_idx}{tag}: best(cond4_peak)={best[0]} mse={best[1]:.6g}")
                except Exception:
                    pass
                return out

        if bool(args.minimal):
            cond4 = data.get('4bit')
            target = data.get('16bit_gt')
            gen0 = None
            if isinstance(data.get('trajectories'), list) and data['trajectories']:
                gen0 = data['trajectories'][0]

            if (not torch.is_tensor(cond4)) or (not torch.is_tensor(target)) or (not torch.is_tensor(gen0)):
                return None

            # Default to mag LP=40Hz taper if user didn't specify.
            mag_lp_hz = float(args.mag_lowpass_hz) if args.mag_lowpass_hz is not None else 40.0
            gen0_lp = None
            if mag_lp_hz is not None and math.isfinite(float(mag_lp_hz)) and float(mag_lp_hz) > 0:
                gen0_lp = _reconstruct_gen_mag_lowpass_condphase(
                    config,
                    data,
                    sample_dir=sample_dir,
                    sample_rate_hz=float(args.sample_rate),
                    mag_lowpass_hz=float(mag_lp_hz),
                    mag_lowpass_kind=str(args.mag_lowpass_kind),
                    mag_lowpass_transition_bins=int(args.mag_lowpass_transition_bins),
                )

            # Peak-rescale generated recon(s) to match condition peak (matches sampler convention).
            cond4_ref = _squeeze_1d(cond4)
            gen0_scaled_tmp = _peak_rescale_like(cond4_ref, gen0)
            gen0_scaled = gen0_scaled_tmp if gen0_scaled_tmp is not None else _squeeze_1d(gen0)
            gen0_lp_scaled = _peak_rescale_like(cond4_ref, gen0_lp) if gen0_lp is not None else None

            plot_minimal_time(
                cond4=cond4_ref,
                target=_squeeze_1d(target),
                gen0=gen0_scaled,
                gen0_lp=gen0_lp_scaled,
                out_path=plot_dir / f'signals_time{tag}.png',
            )
            plot_minimal_fft_mag(
                cond4=cond4_ref,
                target=_squeeze_1d(target),
                gen0=gen0_scaled,
                gen0_lp=gen0_lp_scaled,
                out_path=plot_dir / f'signals_fft_mag{tag}.png',
                sample_rate_hz=float(args.sample_rate),
                max_hz=None,
            )

            # Minimal rFFT magnitude+phase (includes FFT phase for DDPM/gen_0).
            plot_minimal_fft_mag_phase(
                cond4=cond4_ref,
                target=_squeeze_1d(target),
                gen0=gen0_scaled,
                gen0_lp=gen0_lp_scaled,
                out_path=plot_dir / f'signals_fft_mag_phase{tag}.png',
                sample_rate_hz=float(args.sample_rate),
                max_hz=None,
            )

            # Two extra visualization files: magnitude spectrogram + phase spectrogram.
            plot_minimal_spectrogram_magnitude(
                config=config,
                cond4=cond4_ref,
                target=_squeeze_1d(target),
                gen0=gen0_scaled,
                gen0_lp=gen0_lp_scaled,
                out_path=plot_dir / f'signals_spec_mag{tag}.png',
            )
            plot_minimal_spectrogram_phase(
                config=config,
                cond4=cond4_ref,
                target=_squeeze_1d(target),
                gen0=gen0_scaled,
                gen0_lp=gen0_lp_scaled,
                out_path=plot_dir / f'signals_spec_phase{tag}.png',
            )

            # MSE metrics vs target (aligned length).
            metrics = {
                'sample_idx': int(sample_idx),
                'sampler_type': str(sampler_tag) if sampler_tag is not None else None,
                'gen_time_key': str(data.get('_gen_time_key_resolved') or data.get('_gen_time_key') or ''),
                'mse_cond4_vs_target': _mse_aligned(cond4_ref, target),
                'mse_gen0_vs_target': _mse_aligned(gen0_scaled, target),
                'mse_gen0_maglp_vs_target': _mse_aligned(gen0_lp_scaled, target) if gen0_lp_scaled is not None else float('nan'),
                'mag_lowpass_hz': float(mag_lp_hz),
                'mag_lowpass_kind': str(args.mag_lowpass_kind),
                'mag_lowpass_transition_bins': int(args.mag_lowpass_transition_bins),
            }
            with open(plot_dir / f'metrics_minimal{tag}.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            print(
                f"sample_{sample_idx}{tag}: "
                f"mse(cond4,target)={metrics['mse_cond4_vs_target']:.6g} "
                f"mse(gen0,target)={metrics['mse_gen0_vs_target']:.6g} "
                f"mse(gen0_LP,target)={metrics['mse_gen0_maglp_vs_target']:.6g}"
            )
            return metrics

        # Full mode (legacy behavior)
        _add_generated_quant_versions(
            data,
            target_bits=int(args.target_bits),
            cond_bits=int(config.get('bit_size', 4)) if isinstance(config, dict) else 4,
        )
        _add_cond_minmax_scaled_gen0(data, target_bits=int(args.target_bits))

        if bool(args.print_details):
            print_details(data, config)
        if args.quantize:
            _quantize_generated(data, bits=16)

        plot_time_series(data, plot_dir / f'time{tag}.png')
        plot_fft_mag_phase_series(
            data,
            plot_dir / f'fft_mag_phase{tag}.png',
            sample_rate_hz=float(args.sample_rate),
            max_trajectories=int(args.max_trajectories),
        )
        if bool(args.phase_diag):
            plot_phase_recon_diagnostics(
                config,
                data,
                sample_dir=sample_dir,
                out_time_path=plot_dir / f'phase_recon_time{tag}.png',
                out_fft_path=plot_dir / f'phase_recon_fft{tag}.png',
                out_phase_path=plot_dir / f'phase_matrices{tag}.png',
                sample_rate_hz=float(args.sample_rate),
                griffin_lim_iters=int(args.griffin_lim_iters),
                hybrid_cutoff_hz=(float(args.phase_hybrid_cutoff_hz) if args.phase_hybrid_cutoff_hz is not None else None),
                mag_lowpass_hz=(float(args.mag_lowpass_hz) if args.mag_lowpass_hz is not None else None),
                mag_lowpass_kind=str(args.mag_lowpass_kind),
                mag_lowpass_transition_bins=int(args.mag_lowpass_transition_bins),
                time_lowpass_hz=(float(args.phase_diag_time_lowpass_hz) if args.phase_diag_time_lowpass_hz is not None else None),
                primary_phase_source=str(args.phase_diag_primary_phase),
                peak_rescale_to=str(args.phase_diag_peak_rescale),
                share_ylim=bool(args.phase_diag_share_ylim),
            )
        if args.lowpass_hz is not None:
            plot_time_series_lowpass(
                data,
                plot_dir / f'time_lowpass{tag}.png',
                cutoff_hz=float(args.lowpass_hz),
                sample_rate_hz=float(args.sample_rate),
            )
        pipe = (config.get('pipeline_config') or {})
        if pipe:
            plot_spectrograms(config, data, plot_dir / f'specs{tag}.png', max_trajectories=int(args.max_trajectories))
            plot_spectrograms_alt(config, data, plot_dir / f'specs_alt{tag}.png', max_trajectories=int(args.max_trajectories))
            plot_stft_mag_phase_spectrograms(
                config,
                data,
                plot_dir / f'stft_mag_phase{tag}.png',
                max_trajectories=int(args.max_trajectories),
            )
            plot_saved_generated_mag_phase(
                config,
                sample_dir=sample_dir,
                out_path=plot_dir / f'gen_saved_mag_phase{tag}.png',
                traj_idx=0,
            )
            if args.lowpass_hz is not None:
                lp = float(args.lowpass_hz)
                sr = float(args.sample_rate)
                plot_spectrograms_lowpass(
                    config,
                    data,
                    plot_dir / f'specs_lowpass{tag}.png',
                    cutoff_hz=lp,
                    sample_rate_hz=sr,
                    max_trajectories=int(args.max_trajectories),
                    alt=False,
                )
                plot_spectrograms_lowpass(
                    config,
                    data,
                    plot_dir / f'specs_alt_lowpass{tag}.png',
                    cutoff_hz=lp,
                    sample_rate_hz=sr,
                    max_trajectories=int(args.max_trajectories),
                    alt=True,
                )
        else:
            print('spectrograms: skipped (pipeline_config missing/incomplete)')
        return None

    if bool(args.all_samples):
        start = int(args.sample_idx_start)
        count = int(args.num_samples)
        if count <= 0:
            return
        metrics_all: list[dict] = []
        for i in range(start, start + count):
            m = _analyze_one(int(i))
            if isinstance(m, dict):
                metrics_all.append(m)

        if bool(args.minimal) and metrics_all:
            out_path = results_dir / version / 'analysis' / 'metrics_minimal_summary.json'
            with open(out_path, 'w') as f:
                json.dump({'version': str(args.version), 'run_id': str(args.run_id), 'metrics': metrics_all}, f, indent=2)
            # Print aggregate stats.
            vals_gen = [float(m.get('mse_gen0_vs_target')) for m in metrics_all if np.isfinite(float(m.get('mse_gen0_vs_target', float('nan'))))]
            vals_lp = [float(m.get('mse_gen0_maglp_vs_target')) for m in metrics_all if np.isfinite(float(m.get('mse_gen0_maglp_vs_target', float('nan'))))]
            if vals_gen:
                print(f"minimal summary: mean mse(gen0,target)={float(np.mean(vals_gen)):.6g} over {len(vals_gen)} samples")
            if vals_lp:
                print(f"minimal summary: mean mse(gen0_LP,target)={float(np.mean(vals_lp)):.6g} over {len(vals_lp)} samples")
        return

    # Default: analyze one sample.
    _ = _analyze_one(int(sample_idx))
    return



if __name__ == "__main__":
    main()
