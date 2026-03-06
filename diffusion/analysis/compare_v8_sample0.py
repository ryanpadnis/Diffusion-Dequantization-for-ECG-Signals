"""Compare sample_0 across the 12 V8 model variants.

This script is meant to answer: "show me the first sample across all 12 models".
It assumes sampling artifacts are present under:

  diffusion/results/V8/<variant>/<run_id>/samples/<sampler>/sample_<k>/trajectory_<t>.pt

Each sample directory is expected to contain:
  - target_time.pt (dict with key: 'target_time')
  - cond_time_4bit.pt (dict with key: 'cond_time_4bit') [optional baseline]
  - trajectory_<t>.pt (dict with key: 'time_domain')

Outputs:
  - A PNG grid plot comparing generated vs target waveforms.
  - A CSV table of MSEs per variant.

Usage:
  uv run python -m diffusion.analysis.compare_v8_sample0 \
    --version V8 --sample-index 0 --trajectory 0 --sampler ddpm

Notes:
  - If porting to other versions, pass --version.
  - If a variant has multiple runs, the newest run with the expected sample files is used.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch


_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}.*$")


@dataclass(frozen=True)
class VariantResult:
    variant: str
    run_id: str
    mse_gen_vs_target: float
    mse_cond_vs_target: Optional[float]
    traj_path: Path
    target_path: Path


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and _RUN_ID_RE.match(path.name) is not None


def _find_latest_run_with_sample(variant_dir: Path, *, sampler: str, sample_index: int, trajectory: int) -> Optional[Path]:
    """Return the run directory containing the requested sampling artifact."""
    if not variant_dir.exists():
        return None

    # Most common layout: <variant>/<run_id>/samples/<sampler>/sample_k/trajectory_t.pt
    candidates: list[Path] = []
    for run_dir in variant_dir.iterdir():
        if not _is_run_dir(run_dir):
            continue
        traj = (
            run_dir
            / "samples"
            / sampler
            / f"sample_{sample_index}"
            / f"trajectory_{trajectory}.pt"
        )
        tgt = run_dir / "samples" / sampler / f"sample_{sample_index}" / "target_time.pt"
        if traj.exists() and tgt.exists():
            candidates.append(run_dir)

    if not candidates:
        return None

    # Newest by lexicographic run_id (timestamps sort correctly as strings).
    return sorted(candidates, key=lambda p: p.name, reverse=True)[0]


def _load_1d_time_tensor(pt_path: Path, *, key: str) -> torch.Tensor:
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and key in obj:
        x = obj[key]
    else:
        raise KeyError(f"Expected key {key!r} in {pt_path}")

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = x.detach().to(torch.float32)
    # Flatten any leading dims; keep time last.
    x = x.reshape(-1)
    return x


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).reshape(-1)
    b = b.detach().to(torch.float32).reshape(-1)
    n = min(int(a.numel()), int(b.numel()))
    if n <= 0:
        return float("nan")
    return float(torch.mean((a[:n] - b[:n]) ** 2).item())


def _iter_variants(version_dir: Path) -> Iterable[Path]:
    for p in sorted(version_dir.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            yield p


def compare(
    *,
    results_root: Path,
    version: str,
    sample_index: int,
    trajectory: int,
    sampler: str,
) -> list[VariantResult]:
    version_dir = results_root / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Missing version directory: {version_dir}")

    rows: list[VariantResult] = []

    for variant_dir in _iter_variants(version_dir):
        run_dir = _find_latest_run_with_sample(
            variant_dir,
            sampler=sampler,
            sample_index=sample_index,
            trajectory=trajectory,
        )
        if run_dir is None:
            continue

        sample_dir = run_dir / "samples" / sampler / f"sample_{sample_index}"
        traj_path = sample_dir / f"trajectory_{trajectory}.pt"
        target_path = sample_dir / "target_time.pt"
        cond_path = sample_dir / "cond_time_4bit.pt"

        gen = _load_1d_time_tensor(traj_path, key="time_domain")
        tgt = _load_1d_time_tensor(target_path, key="target_time")

        mse_gen = _mse(gen, tgt)
        mse_cond = None
        if cond_path.exists():
            try:
                cond = _load_1d_time_tensor(cond_path, key="cond_time_4bit")
                mse_cond = _mse(cond, tgt)
            except Exception:
                mse_cond = None

        rows.append(
            VariantResult(
                variant=variant_dir.name,
                run_id=run_dir.name,
                mse_gen_vs_target=mse_gen,
                mse_cond_vs_target=mse_cond,
                traj_path=traj_path,
                target_path=target_path,
            )
        )

    # Stable ordering: best MSE first.
    rows.sort(key=lambda r: r.mse_gen_vs_target)
    return rows


def _save_csv(rows: list[VariantResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "run_id", "mse_gen_vs_target", "mse_cond_vs_target", "trajectory_pt", "target_pt"])
        for r in rows:
            w.writerow(
                [
                    r.variant,
                    r.run_id,
                    f"{r.mse_gen_vs_target:.8g}",
                    ("" if r.mse_cond_vs_target is None else f"{r.mse_cond_vs_target:.8g}"),
                    str(r.traj_path),
                    str(r.target_path),
                ]
            )


def _plot(rows: list[VariantResult], *, out_png: Path, max_points: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: uv pip install matplotlib\n"
            f"Import error: {e}"
        )

    n = len(rows)
    if n == 0:
        raise SystemExit("No variants found with the requested sample artifacts.")

    # Prefer a 3x4 grid for the expected 12 variants; fall back otherwise.
    cols = 4
    rows_n = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5.0, rows_n * 2.6), squeeze=False)

    for i, r in enumerate(rows):
        ax = axes[i // cols][i % cols]
        gen = _load_1d_time_tensor(r.traj_path, key="time_domain")
        tgt = _load_1d_time_tensor(r.target_path, key="target_time")
        m = min(int(gen.numel()), int(tgt.numel()), int(max_points) if max_points > 0 else 10**18)
        gen = gen[:m].numpy()
        tgt = tgt[:m].numpy()

        ax.plot(tgt, linewidth=1.2, label="target")
        ax.plot(gen, linewidth=1.0, label="gen")

        title = f"{r.variant}\nMSE(gen,target)={r.mse_gen_vs_target:.3g}"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes.
    for j in range(n, rows_n * cols):
        axes[j // cols][j % cols].axis("off")

    # One shared legend.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare sample_0 across V8 variants and compute MSE")
    p.add_argument("--results-root", type=str, default="diffusion/results", help="Root results dir")
    p.add_argument("--version", type=str, default="V8", help="Version folder (e.g. V8)")
    p.add_argument("--sample-index", type=int, default=0, help="Sample index (sample_k)")
    p.add_argument("--trajectory", type=int, default=0, help="Trajectory index (trajectory_t)")
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Sampler subdir")
    p.add_argument("--max-points", type=int, default=5000, help="Max points to plot per waveform (0=all)")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (defaults to <results-root>/<version>/_comparisons/)",
    )
    args = p.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    version = str(args.version)
    sample_index = int(args.sample_index)
    trajectory = int(args.trajectory)
    sampler = str(args.sampler)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (results_root / version / "_comparisons")
    out_png = out_dir / f"compare_sample_{sample_index}_traj_{trajectory}_{sampler}.png"
    out_csv = out_dir / f"compare_sample_{sample_index}_traj_{trajectory}_{sampler}.csv"

    rows = compare(
        results_root=results_root,
        version=version,
        sample_index=sample_index,
        trajectory=trajectory,
        sampler=sampler,
    )

    _save_csv(rows, out_csv)
    _plot(rows, out_png=out_png, max_points=int(args.max_points))

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_csv}")
    print("\nTop MSEs (best first):")
    for r in rows[: min(12, len(rows))]:
        extra = "" if r.mse_cond_vs_target is None else f" | MSE(cond,target)={r.mse_cond_vs_target:.3g}"
        print(f"- {r.variant} ({r.run_id}): MSE(gen,target)={r.mse_gen_vs_target:.3g}{extra}")


if __name__ == "__main__":
    main()
