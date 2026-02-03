"""Compare raw vs 4-bit vs 16-bit vs diffusion-generated outputs for one sample."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from data.preprocess.transform import get_transform
from data.preprocess.quantize import UniformQuantizer, compute_range_from_tensor


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

def load_sample_data(results_dir: Path, version: str, sample_idx: int = 0) -> dict:
    data: dict = {}

    sample_dir, sampler_type = _resolve_sample_dir(results_dir, version, sample_idx)
    data['_sample_dir'] = sample_dir
    data['_sampler_type'] = sampler_type

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

    trajs: list[torch.Tensor] = []
    all_traj_path = sample_dir / 'all_trajectories.pt'
    if all_traj_path.exists():
        obj = torch.load(all_traj_path, map_location='cpu')
        td = None
        if isinstance(obj, dict):
            td = obj.get('time_domain')
            if td is None:
                td = obj.get('generated_time_domain')
        if torch.is_tensor(td):
            td = td.to(torch.float32)
            while td.ndim > 2:
                td = td.squeeze(1)
            if td.ndim == 1:
                trajs.append(td)
            elif td.ndim == 2:
                trajs.extend([td[i] for i in range(td.shape[0])])

    if not trajs:
        traj_idx = 0
        while True:
            traj_path = sample_dir / f'trajectory_{traj_idx}.pt'
            if not traj_path.exists():
                break
            traj_obj = torch.load(traj_path, map_location='cpu')
            traj_signal = _extract_tensor(
                traj_obj,
                key_candidates=['time_domain', 'generated_signal', 'generated_time_domain', 'signal'],
            )
            if traj_signal is not None:
                trajs.append(traj_signal)
            traj_idx += 1

    data['trajectories'] = trajs
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

    if raw is not None:
        series.append(('raw', raw))
    if q4 is not None:
        series.append(('4bit', q4))
    if gt is not None:
        series.append(('16bit', gt))

    for i, traj in enumerate(data.get('trajectories', [])[:max_trajectories]):
        series.append((f'gen_{i}', traj))

    if include_diff and (q4 is not None) and (gt is not None):
        n = min(int(q4.numel()), int(gt.numel()))
        series.append(('diff_16-4', gt[:n] - q4[:n]))

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
        if shared_ylim is not None and name.startswith('gen_'):
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
        mags.append(('diff_|16-4|', diff))

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
        ax2.set_title('diff_|16-4|')
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
        mags.append(('diff_|16-4|', diff))

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
    deg_4bit = data.get('4bit')
    trajectories = data.get('trajectories') or []
    if (gt is None) or (deg_4bit is None) or (not trajectories):
        return None

    min_len = min(int(gt.numel()), int(deg_4bit.numel()))
    for traj in trajectories:
        min_len = min(min_len, int(traj.numel()))

    gt = gt[:min_len]
    deg_4bit = deg_4bit[:min_len]

    metrics = {
        'mse_4bit_vs_gt': float(torch.mean((deg_4bit - gt) ** 2)),
        'trajectories': [],
    }

    for i, traj in enumerate(trajectories):
        traj = traj[:min_len]
        traj_metrics = {
            'trajectory_idx': i,
            'mse_vs_gt': float(torch.mean((traj - gt) ** 2)),
            'mse_vs_4bit': float(torch.mean((traj - deg_4bit) ** 2)),
        }
        metrics['trajectories'].append(traj_metrics)

    return metrics


def print_metrics(metrics: dict | None) -> None:
    if not metrics:
        print('metrics: skipped (need 4bit + 16bit_gt + at least one trajectory)')
        return
    print(f"mse_4bit_vs_gt: {metrics['mse_4bit_vs_gt']:.6f}")
    for traj in metrics['trajectories']:
        denom = metrics['mse_4bit_vs_gt']
        improvement = ((denom - traj['mse_vs_gt']) / denom * 100) if denom != 0 else float('nan')
        print(f"traj_{traj['trajectory_idx']}: mse_vs_gt={traj['mse_vs_gt']:.6f}, improvement={improvement:+.2f}%")


def main() -> None:
    # Edit these if you want a different selection without CLI args.
    results_dir = Path('diffusion/results')
    version = 'V1'
    sample_idx = 0
    # Set to 'ddim' or 'ddpm' to force a specific folder; None means auto-select.
    sampler_type = "ddim"
    quantize = False
    fft_max_bin = 250

    config: dict = {}
    config_path = results_dir / version / 'config.pkl'
    if config_path.exists():
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    else:
        print(f"config: missing ({config_path}); skipping spectrogram plots")

    sample_dir, detected_sampler = _resolve_sample_dir(results_dir, version, sample_idx, sampler_type=sampler_type)
    data = load_sample_data(
        results_dir,
        version,
        sample_idx,
        sample_dir=sample_dir,
        sampler_type=detected_sampler,
    )
    if quantize:
        _quantize_generated(data, bits=16)

    metrics = compute_metrics(data)
    print_metrics(metrics)

    plot_dir = results_dir / version / 'analysis'
    plot_dir.mkdir(parents=True, exist_ok=True)

    sampler_type = data.get('_sampler_type')
    tag = f"_{sampler_type}" if isinstance(sampler_type, str) and sampler_type else ""

    have_any = bool(build_series(data, max_trajectories=1, include_diff=False))
    if not have_any:
        print('no signals found to plot in sample folder')
        return

    plot_time_series(data, plot_dir / f'sample_{sample_idx}{tag}_time.png')
    plot_fft_series(data, plot_dir / f'sample_{sample_idx}{tag}_fft.png', max_trajectories=1, max_bin=fft_max_bin)

    pipe = (config.get('pipeline_config') or {})
    if pipe and (pipe.get('n_fft') is not None):
        plot_spectrograms(config, data, plot_dir / f'sample_{sample_idx}{tag}_specs.png')
        plot_spectrograms_alt(config, data, plot_dir / f'sample_{sample_idx}{tag}_specs_alt.png', max_trajectories=1)
    else:
        print('spectrograms: skipped (pipeline_config missing/incomplete)')



if __name__ == "__main__":
    main()
