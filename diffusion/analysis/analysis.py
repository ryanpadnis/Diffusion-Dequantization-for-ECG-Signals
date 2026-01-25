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


def load_sample_data(results_dir: Path, version: str = "V1", sample_idx: int = 0) -> dict:
    sample_dir = results_dir / version / 'samples' / f'sample_{sample_idx}'

    data = {
        'raw': _squeeze_1d(torch.load(sample_dir / 'condition_raw.pt')['raw_signal']),
        '4bit': _squeeze_1d(torch.load(sample_dir / 'condition_4bit_time.pt')['time_domain_4bit']),
        '16bit_gt': _squeeze_1d(torch.load(sample_dir / 'ground_truth_16bit_time.pt')['time_domain_16bit']),
        'trajectories': [],
    }

    time_all = torch.load(sample_dir / 'all_trajectories.pt')['time_domain']
    if time_all.ndim == 1:
        data['trajectories'] = [_squeeze_1d(time_all)]
    else:
        data['trajectories'] = [_squeeze_1d(time_all[i]) for i in range(time_all.shape[0])]

    return data


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


def plot_fft_series(data: dict, out_path: Path, max_trajectories: int = 1) -> None:
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


def compute_metrics(data: dict):
    gt = data['16bit_gt']
    deg_4bit = data['4bit']
    
    min_len = min(len(gt), len(deg_4bit))
    for traj in data['trajectories']:
        min_len = min(min_len, len(traj))
    
    gt = gt[:min_len]
    deg_4bit = deg_4bit[:min_len]
    
    metrics = {
        'mse_4bit_vs_gt': float(torch.mean((deg_4bit - gt) ** 2)),
        'trajectories': []
    }
    
    for i, traj in enumerate(data['trajectories']):
        traj = traj[:min_len]
        
        traj_metrics = {
            'trajectory_idx': i,
            'mse_vs_gt': float(torch.mean((traj - gt) ** 2)),
            'mse_vs_4bit': float(torch.mean((traj - deg_4bit) ** 2)),
        }
        metrics['trajectories'].append(traj_metrics)
    
    return metrics


def print_metrics(metrics: dict):
    print(f"mse_4bit_vs_gt: {metrics['mse_4bit_vs_gt']:.6f}")
    for traj in metrics['trajectories']:
        improvement = ((metrics['mse_4bit_vs_gt'] - traj['mse_vs_gt']) / 
                      metrics['mse_4bit_vs_gt'] * 100)
        print(f"traj_{traj['trajectory_idx']}: mse_vs_gt={traj['mse_vs_gt']:.6f}, improvement={improvement:+.2f}%")


def main():
    results_dir = Path("diffusion/results")
    version = "V1"
    sample_idx = 0
    quantize = False

    config_path = results_dir / version / 'config.pkl'
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    data = load_sample_data(results_dir, version, sample_idx)
    if quantize:
        _quantize_generated(data, bits=16)

    metrics = compute_metrics(data)
    print_metrics(metrics)

    plot_dir = results_dir / version / 'analysis'

    plot_time_series(data, plot_dir / f'sample_{sample_idx}_time.png')
    plot_spectrograms(config, data, plot_dir / f'sample_{sample_idx}_specs.png')
    plot_spectrograms_alt(config, data, plot_dir / f'sample_{sample_idx}_specs_alt.png', max_trajectories=1)
    plot_fft_series(data, plot_dir / f'sample_{sample_idx}_fft.png', max_trajectories=1)


if __name__ == "__main__":
    main()
