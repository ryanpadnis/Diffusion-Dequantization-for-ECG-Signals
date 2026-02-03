"""Visualize diffusion denoising progression over time.

This script does not do inversion/denormalization. It only plots what the
sampler already saved in each progress snapshot.

Expected keys per progress snapshot:
- spectrograms_16bit: denormalized magnitude spectrograms, shape [B, 1, F, T]
- time_domain: inverted waveforms, shape [B, L]

For comparison, it overlays the condition (4-bit) and GT (16-bit) waveforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import torch


def _squeeze_1d(x: torch.Tensor) -> torch.Tensor:
    while x.ndim > 1:
        x = x.squeeze(0)
    return x


def _extract_denorm_mag_2d(obj: dict, traj_idx: int = 0) -> torch.Tensor | None:
    v = obj.get('spectrograms_16bit')
    if not torch.is_tensor(v):
        return None

    s = v.to(torch.float32)
    if s.ndim == 4:
        b = min(max(int(traj_idx), 0), int(s.shape[0]) - 1)
        return s[b, 0]
    if s.ndim == 3:
        b = min(max(int(traj_idx), 0), int(s.shape[0]) - 1)
        return s[b]
    if s.ndim == 2:
        return s
    return None


def _extract_time_1d(obj: dict, traj_idx: int = 0) -> torch.Tensor | None:
    v = obj.get('time_domain')
    if not torch.is_tensor(v):
        return None
    x = v.to(torch.float32)
    if x.ndim == 2:
        b = min(max(int(traj_idx), 0), int(x.shape[0]) - 1)
        return _squeeze_1d(x[b])
    if x.ndim == 1:
        return x
    while x.ndim > 1:
        x = x[0]
    return x


@dataclass(frozen=True)
class ProgressFrame:
    path: Path
    step_idx: int
    timestep: int


def _parse_progress_filename(p: Path) -> ProgressFrame | None:
    m = re.match(r"progress_step_(\d+)_t(\d+)\.pt$", p.name)
    if not m:
        return None
    return ProgressFrame(path=p, step_idx=int(m.group(1)), timestep=int(m.group(2)))


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

    legacy = samples_root / f'sample_{sample_idx}'
    if legacy.exists():
        return legacy, None

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

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, chosen_dir, sampler_type = candidates[0]
    return chosen_dir, sampler_type


def main() -> None:
    # Edit these if you want a different selection without CLI args.
    results_dir = Path('diffusion/results')
    version = 'V1'
    sample_idx = 0
    trajectory_idx = 0
    max_frames = 10
    # Set to 'ddim' or 'ddpm' to force a specific folder; None means auto-select.
    sampler_type_arg: str | None = None

    sample_dir, sampler_type = _resolve_sample_dir(results_dir, version, sample_idx, sampler_type=sampler_type_arg)
    progress_dir = sample_dir / 'progress'
    out_dir = results_dir / version / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    if not progress_dir.exists():
        print(f'progress: missing ({progress_dir})')
        return

    cond_ref = None
    gt_ref = None
    cond_time_path = sample_dir / 'condition_4bit_time.pt'
    if cond_time_path.exists():
        obj = torch.load(cond_time_path, map_location='cpu')
        if torch.is_tensor(obj.get('time_domain_4bit')):
            cond_ref = _squeeze_1d(obj['time_domain_4bit'].to(torch.float32))

    gt_time_path = sample_dir / 'ground_truth_16bit_time.pt'
    if gt_time_path.exists():
        obj = torch.load(gt_time_path, map_location='cpu')
        if torch.is_tensor(obj.get('time_domain_16bit')):
            gt_ref = _squeeze_1d(obj['time_domain_16bit'].to(torch.float32))

    frames: list[ProgressFrame] = []
    for p in sorted(progress_dir.glob('progress_step_*.pt')):
        fr = _parse_progress_filename(p)
        if fr is not None:
            frames.append(fr)

    if not frames:
        print(f'progress: no progress_step_*.pt found in {progress_dir}')
        return

    if len(frames) > max_frames:
        idxs = (
            torch.linspace(0, len(frames) - 1, steps=max_frames)
            .round()
            .to(torch.int64)
            .tolist()
        )
        frames = [frames[i] for i in idxs]

    nrows = len(frames)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 2.6 * nrows))
    if nrows == 1:
        axes = [axes]

    for row, fr in enumerate(frames):
        obj = torch.load(fr.path, map_location='cpu')
        mag = _extract_denorm_mag_2d(obj, traj_idx=trajectory_idx)
        wave = _extract_time_1d(obj, traj_idx=trajectory_idx)

        if mag is None:
            print(
                f"snapshot missing 'spectrograms_16bit': {fr.path.name}. "
                "Re-run sampling with progress snapshots after updating the sampler."
            )
            return
        if wave is None:
            print(
                f"snapshot missing 'time_domain': {fr.path.name}. "
                "Re-run sampling with progress snapshots after updating the sampler."
            )
            return

        ax_spec, ax_wave = axes[row]
        ax_spec.imshow(mag.detach().cpu().numpy(), aspect='auto', origin='lower')
        ax_spec.set_title(f'step={fr.step_idx}  t={fr.timestep} (denorm mag)')
        ax_spec.set_ylabel('freq')

        ax_wave.plot(wave.detach().cpu().numpy(), linewidth=0.8, label='gen')
        if cond_ref is not None:
            ax_wave.plot(cond_ref.detach().cpu().numpy(), linewidth=0.6, alpha=0.6, label='cond_4bit')
        if gt_ref is not None:
            ax_wave.plot(gt_ref.detach().cpu().numpy(), linewidth=0.6, alpha=0.6, label='gt_16bit')

        title = 'waveform (gen + references)'
        if gt_ref is not None:
            n = min(int(gt_ref.numel()), int(wave.numel()))
            mse = float(torch.mean((wave[:n] - gt_ref[:n]) ** 2).item())
            title += f'  mse_to_gt={mse:.4g}'
        ax_wave.set_title(title)
        ax_wave.set_ylabel('amp')

        if row == 0:
            ax_wave.legend(loc='upper right', fontsize=8)

    axes[-1][0].set_xlabel('time frame')
    axes[-1][1].set_xlabel('time index')

    plt.tight_layout()
    tag = f"_{sampler_type}" if isinstance(sampler_type, str) and sampler_type else ""
    out_path = out_dir / f'sample_{sample_idx}{tag}_progress_over_time.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f'wrote: {out_path}')


if __name__ == '__main__':
    main()
