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

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from data.preprocess.transform import get_transform
from data.preprocess.quantize import UniformQuantizer, compute_range_from_tensor


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
        ['condition_4bit_time.pt', 'quantized_signal.pt', 'quantized_signal_4bit.pt', 'condition_4bit.pt', 'q4.pt'],
    )
    if q4_path is not None:
        data['4bit'] = _extract_tensor(
            torch.load(q4_path, map_location='cpu'),
            key_candidates=['time_domain_4bit', 'quantized_signal_4bit', 'condition_signal_4bit', 'signal_4bit', 'x'],
        )

    gt_path = _first_existing_path(
        sample_dir,
        ['ground_truth_16bit_time.pt', 'ground_truth_signal.pt', 'ground_truth.pt', 'gt.pt', 'target.pt'],
    )
    if gt_path is not None:
        data['16bit_gt'] = _extract_tensor(
            torch.load(gt_path, map_location='cpu'),
            key_candidates=[
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
    p.add_argument("--max-trajectories", type=int, default=1, help="How many trajectories to include in plots")
    p.add_argument("--quantize", action="store_true", help="Quantize generated trajectories to 16-bit for comparison")
    p.add_argument("--target-bits", type=int, default=11, help="Label/assume target bit-depth for plots/metrics (default: 11)")
    p.add_argument("--print-details", action="store_true", help="Print debug details about loaded/reconstructed signals and quantization")
    p.add_argument(
        "--gen-time-key",
        type=str,
        default="auto",
        help="Which generated time-domain field to analyze. Use 'auto' to prefer peak-rescaled if present (default: auto).",
    )
    p.add_argument("--include-specs", action="store_true", help="Also save spectrogram plots (off by default)")
    p.add_argument("--fft-max-bin", type=int, default=250, help="Max FFT bin to display")
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
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    else:
        print(f"config: missing ({config_path}); skipping spectrogram plots")

    # Force sampler selection by setting sampler_type for resolver.
    sample_dir, detected_sampler = _resolve_sample_dir(results_dir, version, sample_idx, sampler_type=sampler_type)
    if not sample_dir.exists():
        raise SystemExit(f"sample dir not found: {sample_dir}")

    data = load_sample_data(
        results_dir,
        version,
        sample_idx,
        sample_dir=sample_dir,
        sampler_type=detected_sampler,
        gen_time_key=str(args.gen_time_key),
    )

    data['_target_bits'] = int(args.target_bits) if args.target_bits is not None else None

    # If sampling didn't persist explicit 4-bit condition / target waveforms, reconstruct
    # them from the raw condition waveform so metrics can still be computed.
    _fill_missing_baselines_from_raw(data, config)

    # Add quantized views of the generated output for visual comparison.
    _add_generated_quant_versions(
        data,
        target_bits=int(args.target_bits),
        cond_bits=int(config.get('bit_size', 4)) if isinstance(config, dict) else 4,
    )

    # Add the condition-range min/max scaled gen0 variant.
    _add_cond_minmax_scaled_gen0(data, target_bits=int(args.target_bits))

    if bool(args.print_details):
        print_details(data, config)

    if args.quantize:
        _quantize_generated(data, bits=16)

    metrics = compute_metrics(data)
    print_metrics(metrics)

    # Feature descriptors across all signals (raw/cond/target/gen + quantized gen0 variants).
    features = print_all_series_features(data, max_trajectories=int(args.max_trajectories))

    # Save per-sample plots in a dedicated subdirectory.
    plot_dir = results_dir / version / 'analysis' / f'sample_{sample_idx}'
    plot_dir.mkdir(parents=True, exist_ok=True)

    sampler_tag = detected_sampler or data.get('_sampler_type')
    tag = f"_{sampler_tag}" if isinstance(sampler_tag, str) and sampler_tag else ""

    # Cleanup legacy correlation outputs (no longer produced).
    for legacy in [plot_dir / f'xcorr{tag}.png', plot_dir / f'corr_matrix{tag}.png']:
        try:
            if legacy.exists():
                legacy.unlink()
        except Exception:
            pass

    have_any = bool(build_series(data, max_trajectories=int(args.max_trajectories), include_diff=False))
    if not have_any:
        print('no signals found to plot in sample folder')
        return

    plot_time_series(data, plot_dir / f'time{tag}.png')
    plot_fft_series(
        data,
        plot_dir / f'fft{tag}.png',
        max_trajectories=int(args.max_trajectories),
        max_bin=int(args.fft_max_bin) if args.fft_max_bin is not None else None,
    )
    plot_stats_bars(data, plot_dir / f'stats{tag}.png')

    # Save features to JSON for easy diffing / downstream usage.
    try:
        with open(plot_dir / f'features{tag}.json', 'w') as f:
            json.dump(features, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"features: failed to write json (non-fatal): {e}")

    # ACF/PACF overlays: for each diffusion output (each gen_* and quantized gen0 variants),
    # overlay against both the target and conditional baselines.
    cond4 = data.get('4bit')
    tgt = data.get('16bit_gt')
    if torch.is_tensor(cond4) and torch.is_tensor(tgt):
        target_bits = data.get('_target_bits')
        target_label = f"target_{int(target_bits)}bit" if target_bits is not None else 'target'
        diffusion_items = []
        for name, s in build_series(data, max_trajectories=int(args.max_trajectories), include_diff=False):
            if not torch.is_tensor(s):
                continue
            if str(name).startswith('gen_'):
                diffusion_items.append((str(name), s))
        for name, s in diffusion_items:
            safe = name.replace('/', '_')
            plot_acf_pacf_overlay(
                name=name,
                series=s,
                cond4=cond4,
                target=tgt,
                target_label=target_label,
                out_acf=plot_dir / f'acf_{safe}{tag}.png',
                out_pacf=plot_dir / f'pacf_{safe}{tag}.png',
                max_lag=400,
            )

    if bool(args.include_specs):
        pipe = (config.get('pipeline_config') or {})
        if pipe and (pipe.get('n_fft') is not None):
            plot_spectrograms(config, data, plot_dir / f'specs{tag}.png', max_trajectories=int(args.max_trajectories))
            plot_spectrograms_alt(config, data, plot_dir / f'specs_alt{tag}.png', max_trajectories=int(args.max_trajectories))
        else:
            print('spectrograms: skipped (pipeline_config missing/incomplete)')



if __name__ == "__main__":
    main()
