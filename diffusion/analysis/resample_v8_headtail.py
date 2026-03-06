"""Resample 16 conditions from the head+tail of the holdout split.

Intended use:
- You already trained models and have checkpoints/config locally under:
    diffusion/results/<version>/<variant>/<run_id>/
- You want to regenerate sampling artifacts without retraining.
- You want the 16 conditions to be:
    first 8 + last 8 of the holdout subset (typically holdout_n=500 from end)

This overwrites (or recreates) the following folders per run:
  diffusion/results/<version>/<variant>/<run_id>/samples/<sampler>/sample_0..15/

Usage:
  uv run python -m diffusion.analysis.resample_v8_headtail \
    --results-root diffusion/results \
    --version V8 \
    --raw-data-path data/data/processed/arrhythmia_chunks.pt \
    --holdout-count 500 \
    --holdout-from-end \
    --head 8 --tail 8 \
    --checkpoint best_model.pt \
    --sampler ddpm \
    --device auto
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from diffusion.sample.sampler import DiffusionSampler


_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}.*$")


@dataclass(frozen=True)
class RunSpec:
    variant: str
    run_id: str
    run_dir: Path


def _pick_device(device: str) -> str:
    d = (device or "auto").strip().lower()
    if d != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_signals(obj: object) -> torch.Tensor:
    if isinstance(obj, dict):
        signals = obj.get("signals")
        if signals is None:
            signals = obj.get("chunks")
        if signals is None:
            raise KeyError("Expected key 'signals' or 'chunks' in loaded data")
        return signals
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Unsupported data object type for signals: {type(obj)}")


def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and _RUN_ID_RE.match(p.name) is not None


def _find_latest_run(variant_dir: Path, checkpoint_name: str) -> Optional[Path]:
    """Pick newest run_id folder that has config.pkl and the checkpoint."""
    candidates: list[Path] = []
    for run_dir in variant_dir.iterdir():
        if not _is_run_dir(run_dir):
            continue
        if not (run_dir / "config.pkl").exists():
            continue
        if not (run_dir / "checkpoints" / checkpoint_name).exists():
            continue
        candidates.append(run_dir)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name, reverse=True)[0]


def _iter_variants(version_dir: Path) -> Iterable[Path]:
    for p in sorted(version_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            yield p


def _select_head_tail(
    signals: torch.Tensor,
    *,
    holdout_count: int,
    holdout_from_end: bool,
    head: int,
    tail: int,
) -> torch.Tensor:
    if signals.ndim != 2:
        signals = signals.view(signals.shape[0], -1)

    n_total = int(signals.shape[0])
    h_count = int(holdout_count)
    if h_count > 0 and n_total > h_count:
        holdout = signals[-h_count:] if holdout_from_end else signals[:h_count]
    else:
        holdout = signals

    n = int(holdout.shape[0])
    head = int(head)
    tail = int(tail)
    need = head + tail
    if need <= 0:
        raise ValueError("head+tail must be > 0")
    if n < need:
        raise ValueError(f"Holdout has only {n} samples; need head+tail={need}")

    return torch.cat([holdout[:head], holdout[-tail:]], dim=0)


def _clear_existing_samples(run_dir: Path, *, sampler: str, num_samples: int) -> None:
    root = run_dir / "samples" / sampler
    if not root.exists():
        return
    for i in range(int(num_samples)):
        d = root / f"sample_{i}"
        if d.exists():
            shutil.rmtree(d)


def resample_all_variants(
    *,
    results_root: Path,
    version: str,
    raw_data_path: Path,
    holdout_count: int,
    holdout_from_end: bool,
    head: int,
    tail: int,
    sampler: str,
    checkpoint: str,
    device: str,
    batch_size: int,
    num_trajectories: int,
    overwrite: bool,
    num_inference_steps: Optional[int],
) -> list[RunSpec]:
    version_dir = results_root / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Missing version directory: {version_dir}")

    data_obj = torch.load(raw_data_path, map_location="cpu")
    signals = _extract_signals(data_obj)
    selected = _select_head_tail(
        signals,
        holdout_count=int(holdout_count),
        holdout_from_end=bool(holdout_from_end),
        head=int(head),
        tail=int(tail),
    )

    # Ensure selection is [N, T]
    if selected.ndim != 2:
        selected = selected.view(selected.shape[0], -1)

    picked: list[RunSpec] = []

    use_ddim = (sampler.strip().lower() == "ddim")

    for variant_dir in _iter_variants(version_dir):
        run_dir = _find_latest_run(variant_dir, checkpoint_name=str(checkpoint))
        if run_dir is None:
            continue

        if overwrite:
            _clear_existing_samples(run_dir, sampler=("ddim" if use_ddim else "ddpm"), num_samples=int(selected.shape[0]))

        sampler_obj = DiffusionSampler(
            results_dir=variant_dir,
            version=run_dir.name,
            checkpoint_name=str(checkpoint),
            device=str(device),
        )

        sampler_obj.sample(
            raw_condition_signals=selected,
            num_inference_steps=num_inference_steps,
            use_ddim=use_ddim,
            num_trajectories=int(num_trajectories),
            batch_size=int(batch_size),
        )

        picked.append(RunSpec(variant=variant_dir.name, run_id=run_dir.name, run_dir=run_dir))

    return picked


def main() -> None:
    p = argparse.ArgumentParser(description="Resample 16 conditions (head+tail of holdout) across all variants")
    p.add_argument("--results-root", type=str, default="diffusion/results")
    p.add_argument("--version", type=str, default="V8")
    p.add_argument("--raw-data-path", type=str, default="data/data/processed/arrhythmia_chunks.pt")

    p.add_argument("--holdout-count", type=int, default=500)
    p.add_argument("--holdout-from-end", action="store_true", default=True)
    p.add_argument("--holdout-from-start", action="store_true", default=False)

    p.add_argument("--head", type=int, default=8)
    p.add_argument("--tail", type=int, default=8)

    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--num-inference-steps", type=int, default=None)

    p.add_argument("--num-trajectories", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-overwrite", action="store_true", help="Do not delete existing sample_0..15 folders")

    args = p.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    version = str(args.version)
    raw_data_path = Path(args.raw_data_path).expanduser().resolve()
    if not raw_data_path.exists():
        raise SystemExit(f"Missing raw data: {raw_data_path}")

    holdout_from_end = True
    if bool(args.holdout_from_start):
        holdout_from_end = False

    device = _pick_device(str(args.device))

    picked = resample_all_variants(
        results_root=results_root,
        version=version,
        raw_data_path=raw_data_path,
        holdout_count=int(args.holdout_count),
        holdout_from_end=bool(holdout_from_end),
        head=int(args.head),
        tail=int(args.tail),
        sampler=str(args.sampler),
        checkpoint=str(args.checkpoint),
        device=str(device),
        batch_size=int(args.batch_size),
        num_trajectories=int(args.num_trajectories),
        overwrite=not bool(args.no_overwrite),
        num_inference_steps=(int(args.num_inference_steps) if args.num_inference_steps is not None else None),
    )

    if not picked:
        raise SystemExit("No variants found with a runnable checkpoint/config.")

    print("Resampled runs:")
    for r in picked:
        print(f"- {r.variant}/{r.run_id}")


if __name__ == "__main__":
    main()
