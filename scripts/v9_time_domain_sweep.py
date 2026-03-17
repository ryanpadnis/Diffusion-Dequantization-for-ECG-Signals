#!/usr/bin/env python3
"""
Sweep all V9 runs and compute time-domain metrics across samples.
Outputs a summary table to identify the best run and data suitable for a poster.

Usage:
    uv run python scripts/v9_time_domain_sweep.py
    uv run python scripts/v9_time_domain_sweep.py --output results/v9_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
import statistics
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from diffusion.utils.metrics import compute_energy_metrics
from diffusion.analysis.analysis_copy import (
    load_sample_data,
    _resolve_sample_dir,
    _fill_missing_baselines_from_raw,
)


def discover_v9_runs(results_root: Path) -> list[tuple[str, str]]:
    """Find all (config, run_id) pairs under V9 that have sample data."""
    v9 = results_root / "V9"
    if not v9.exists() or not v9.is_dir():
        return []

    runs: list[tuple[str, str]] = []
    for config_dir in sorted(v9.iterdir()):
        if not config_dir.is_dir() or config_dir.name.startswith("_"):
            continue
        for run_dir in sorted(config_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            # Check for samples
            samples_dir = run_dir / "samples"
            analysis_dir = run_dir / "analysis"
            has_samples = (
                samples_dir.exists()
                and any((samples_dir / d / f"sample_0").exists() for d in samples_dir.iterdir() if d.is_dir())
            ) or (
                samples_dir.exists()
                and (samples_dir / "sample_0").exists()
            )
            has_analysis = analysis_dir.exists() and (analysis_dir / "sample_0").exists()
            if has_samples or has_analysis:
                runs.append((config_dir.name, run_dir.name))
    return runs


def compute_metrics_for_run(
    results_dir: Path,
    version: str,
    sampler_type: str = "ddpm",
    max_samples: int = 20,
):
    """Load samples for a run and compute time-domain energy metrics. Returns list of per-sample results."""
    config_path = results_dir / version / "config.pkl"
    config: dict = {}
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    all_metrics: list[dict] = []
    for sample_idx in range(max_samples):
        sample_dir, detected = _resolve_sample_dir(
            results_dir, version, sample_idx, sampler_type=sampler_type
        )
        if not sample_dir.exists():
            break
        try:
            data = load_sample_data(
                results_dir,
                version,
                sample_idx,
                sample_dir=sample_dir,
                sampler_type=detected or sampler_type,
            )
            _fill_missing_baselines_from_raw(data, config)
            results = compute_energy_metrics(data, n_bands=4, threshold=0.10)
        except Exception:
            continue
        if results is None:
            continue

        baseline = results.get("4bit_vs_gt")
        mse_4bit = baseline.mse if baseline is not None else float("nan")
        # Use trajectory 0 metrics vs GT
        for entry in results.get("trajectories", []):
            m = entry.get("metrics")
            if m is None:
                continue
            all_metrics.append({
                "sample_idx": sample_idx,
                "traj_idx": entry.get("trajectory_idx", 0),
                "mse_4bit": mse_4bit,
                "nef": m.nef,
                "mse": m.mse,
                "nmse": m.nmse,
                "l1_fraction": m.l1_fraction,
                "prd": m.prd,
                "correlation": m.correlation,
                "snr_db": m.snr_db,
                "energy_ratio": m.energy_ratio,
                "nef_improvement_pct": entry.get("improvement_vs_4bit_pct", float("nan")),
                "mse_improvement_pct": entry.get("mse_improvement_pct", float("nan")),
                "l1_improvement_pct": entry.get("l1_improvement_pct", float("nan")),
            })
            break  # Use first trajectory only

    return all_metrics


def aggregate_metrics(per_sample: list[dict]) -> dict | None:
    """Compute mean ± std across samples."""
    if not per_sample:
        return None
    keys = ["mse_4bit", "nef", "mse", "nmse", "l1_fraction", "prd", "correlation", "snr_db", "energy_ratio",
            "nef_improvement_pct", "mse_improvement_pct", "l1_improvement_pct"]
    out: dict = {"n_samples": len(per_sample)}
    for k in keys:
        vals = [x[k] for x in per_sample if k in x and _isfinite(x[k])]
        if vals:
            out[f"{k}_mean"] = statistics.mean(vals)
            out[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        else:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
    return out


def _isfinite(x) -> bool:
    return isinstance(x, (int, float)) and x == x and abs(x) < float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description="V9 time-domain metrics sweep")
    ap.add_argument("--results-root", type=Path, default=ROOT / "diffusion" / "results")
    ap.add_argument("--output", type=Path, default=ROOT / "diffusion" / "results" / "V9" / "v9_time_domain_sweep.csv")
    ap.add_argument("--poster-md", type=Path, default=None, help="Write a Markdown summary for poster")
    ap.add_argument("--plots", action="store_true", help="Generate poster plots (saved to results/V9/)")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--sampler", type=str, default="ddpm")
    args = ap.parse_args()

    results_dir = args.results_root / "V9"
    runs = discover_v9_runs(args.results_root)
    if not runs:
        print("No V9 runs found.")
        return

    print(f"Found {len(runs)} V9 runs. Computing time-domain metrics (paired across configs)...")
    rows: list[dict] = []
    per_run_samples: dict[str, list[dict]] = {}  # version -> list of {sample_idx, nef, ...}

    for config, run_id in runs:
        version = f"{config}/{run_id}"
        per_sample = compute_metrics_for_run(
            results_dir, version, sampler_type=args.sampler, max_samples=args.max_samples
        )
        if not per_sample:
            continue
        per_run_samples[version] = per_sample
        agg = aggregate_metrics(per_sample)
        if agg is None:
            continue
        rows.append({
            "config": config,
            "run_id": run_id,
            "version": version,
            **{k: agg[k] for k in agg},
        })
        print(f"  {version}: {agg['n_samples']} samples, NEF={agg['nef_mean']:.4f}±{agg['nef_std']:.4f}, "
              f"PRD={agg['prd_mean']:.2f}%, corr={agg['correlation_mean']:.4f}, SNR={agg['snr_db_mean']:.2f} dB")

    # Paired comparison: for each sample_idx present in ALL runs, which config won (lowest NEF)?
    common_sample_idxs = None
    for version, samples in per_run_samples.items():
        idxs = frozenset(s["sample_idx"] for s in samples)
        common_sample_idxs = idxs if common_sample_idxs is None else common_sample_idxs & idxs
    wins: dict[str, int] = {v: 0 for v in per_run_samples}
    if common_sample_idxs and len(common_sample_idxs) >= 1:
        # Build sample_idx -> version -> nef
        by_sample: dict[int, dict[str, float]] = {}
        for version, samples in per_run_samples.items():
            for s in samples:
                if s["sample_idx"] in common_sample_idxs:
                    by_sample.setdefault(s["sample_idx"], {})[version] = s["nef"]
        for sample_idx, version_nefs in by_sample.items():
            if len(version_nefs) == len(per_run_samples):
                winner = min(version_nefs, key=version_nefs.get)
                wins[winner] = wins.get(winner, 0) + 1
        n_paired = len(common_sample_idxs)
        print(f"\n  Paired comparison ({n_paired} samples in common):")
        for version in sorted(wins, key=lambda v: -wins[v]):
            print(f"    {version}: won {wins[version]}/{n_paired} samples")
        # Add wins to rows
        for r in rows:
            r["paired_wins"] = wins.get(r["version"], 0)
            r["n_paired"] = len(common_sample_idxs)
    else:
        for r in rows:
            r["paired_wins"] = 0
            r["n_paired"] = 0

    if not rows:
        print("No metrics computed.")
        return

    # Sort by mean NEF within config (lower is better)
    rows.sort(key=lambda r: (r.get("nef_mean", float("inf")), -r.get("paired_wins", 0)))

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config", "run_id", "version", "n_samples", "n_paired", "paired_wins",
                  "mse_4bit_mean", "mse_4bit_std", "nef_mean", "nef_std", "mse_mean", "mse_std", "nmse_mean", "nmse_std",
                  "l1_fraction_mean", "l1_fraction_std", "prd_mean", "prd_std",
                  "correlation_mean", "correlation_std", "snr_db_mean", "snr_db_std",
                  "energy_ratio_mean", "energy_ratio_std",
                  "nef_improvement_pct_mean", "mse_improvement_pct_mean", "l1_improvement_pct_mean"]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {args.output}")

    # 4-bit vs reconstructed comparison table (poster-friendly)
    comparison_path = args.output.parent / "comparison_4bit_vs_recon.csv"
    with open(comparison_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "run_id", "MSE_4bit_mean", "MSE_4bit_std", "MSE_recon_mean", "MSE_recon_std", "MSE_improvement_%"])
        for r in rows:
            w.writerow([
                r.get("config", ""),
                r.get("run_id", ""),
                f"{r.get('mse_4bit_mean', float('nan')):.6g}",
                f"{r.get('mse_4bit_std', float('nan')):.6g}",
                f"{r.get('mse_mean', float('nan')):.6g}",
                f"{r.get('mse_std', float('nan')):.6g}",
                f"{r.get('mse_improvement_pct_mean', float('nan')):.1f}",
            ])
    print(f"Wrote {comparison_path}")

    # Best run summary (ranked by mean NEF)
    best = rows[0]
    print("\n" + "=" * 60)
    best_by_wins = max(rows, key=lambda r: r.get("paired_wins", 0))
    print("BEST RUN (by mean NEF):")
    print(f"  {best['version']}")
    print(f"  Paired wins: {best_by_wins['version']} ({best_by_wins.get('paired_wins', 0)}/{best_by_wins.get('n_paired', 0)} samples)")
    print(f"  NEF:    {best['nef_mean']:.4f} ± {best['nef_std']:.4f}")
    print(f"  PRD:    {best['prd_mean']:.2f}% ± {best['prd_std']:.2f}%")
    print(f"  SNR:    {best['snr_db_mean']:.2f} ± {best['snr_db_std']:.2f} dB")
    print(f"  corr:   {best['correlation_mean']:.4f} ± {best['correlation_std']:.4f}")
    print(f"  samples: {best['n_samples']}")
    print("=" * 60)

    # Optional Markdown for poster
    if args.poster_md:
        md_path = Path(args.poster_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w") as f:
            f.write("# V9 Time-Domain Reconstruction Results\n\n")
            f.write("| Config | Run ID | N | Paired wins | NEF (mean±std) | PRD (%) | Correlation | SNR (dB) |\n")
            f.write("|--------|--------|---|--------------|----------------|---------|-------------|----------|\n")
            for r in rows[:15]:
                cfg = r.get("config", "-")
                rid = r.get("run_id", "-")
                n = r.get("n_samples", 0)
                wins_str = f"{r.get('paired_wins', 0)}/{r.get('n_paired', 0)}"
                nef = f"{r['nef_mean']:.4f}±{r['nef_std']:.4f}"
                prd_val = f"{r['prd_mean']:.2f}±{r['prd_std']:.2f}"
                corr = f"{r['correlation_mean']:.4f}"
                snr = f"{r['snr_db_mean']:.2f}"
                f.write(f"| {cfg} | {rid} | {n} | {wins_str} | {nef} | {prd_val} | {corr} | {snr} |\n")
            f.write(f"\n**Best run:** `{best['version']}` (mean NEF={best['nef_mean']:.4f})\n")
        print(f"Poster summary: {md_path}")

    # Poster plots
    if args.plots and rows and per_run_samples:
        _save_poster_plots(rows, per_run_samples, args.output.parent)


def _save_poster_plots(rows: list[dict], per_run_samples: dict[str, list[dict]], out_dir: Path) -> None:
    """Generate poster-quality figures for NEF, PRD, correlation, and SNR."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    config_labels = [r["config"] for r in rows]
    x = np.arange(len(config_labels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(config_labels)))

    def _bar_plot(ax, means, stds, ylabel, title, ref_line=None, ref_label=None, higher_better=False):
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        if ref_line is not None:
            ax.axhline(y=ref_line, color="gray", linestyle="--", linewidth=1, label=ref_label)
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ref_label:
            ax.legend()
        ax.set_ylim(bottom=0 if not higher_better else None)

    def _box_plot(ax, data_by_config, ylabel, title, ref_line=None, ref_label=None):
        bp = ax.boxplot(data_by_config, labels=config_labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i])
        if ref_line is not None:
            ax.axhline(y=ref_line, color="gray", linestyle="--", linewidth=1, label=ref_label)
        ax.set_xticklabels(config_labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ref_label:
            ax.legend()
        ax.set_ylim(bottom=0 if ref_line is not None else None)

    # --- Plot 1: Bar chart of mean NEF ± std ---
    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_plot(ax, [r["nef_mean"] for r in rows], [r["nef_std"] for r in rows],
              "Noise Energy Fraction (NEF)", "ECG Reconstruction Quality by Config",
              ref_line=0.10, ref_label="Target (NEF ≤ 0.10)")
    fig.tight_layout()
    fig.savefig(out_dir / "poster_nef_by_config.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_nef_by_config.png'}")

    # --- Plot 2: Bar chart of mean PRD ± std ---
    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_plot(ax, [r["prd_mean"] for r in rows], [r["prd_std"] for r in rows],
              "Percent RMS Difference (%)", "PRD by Config (ECG standard)",
              ref_line=None)
    fig.tight_layout()
    fig.savefig(out_dir / "poster_prd_by_config.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_prd_by_config.png'}")

    # --- Plot 3: Bar chart of mean correlation (higher is better) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_plot(ax, [r["correlation_mean"] for r in rows], [r["correlation_std"] for r in rows],
              "Pearson Correlation", "Waveform Correlation with Ground Truth",
              ref_line=1.0, ref_label="Perfect (1.0)", higher_better=True)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "poster_correlation_by_config.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_correlation_by_config.png'}")

    # --- Plot 4: Bar chart of mean MSE ---
    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_plot(ax, [r["mse_mean"] for r in rows], [r["mse_std"] for r in rows],
              "Mean Squared Error", "MSE by Config", ref_line=None)
    fig.tight_layout()
    fig.savefig(out_dir / "poster_mse_by_config.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_mse_by_config.png'}")

    # --- Plot 5: Bar chart of mean L1 fraction ---
    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_plot(ax, [r["l1_fraction_mean"] for r in rows], [r["l1_fraction_std"] for r in rows],
              "L1 Fraction", "L1 Fraction by Config", ref_line=None)
    fig.tight_layout()
    fig.savefig(out_dir / "poster_l1_by_config.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_l1_by_config.png'}")

    # --- Plot 6: Bar chart of mean SNR (dB) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_plot(ax, [r["snr_db_mean"] for r in rows], [r["snr_db_std"] for r in rows],
              "SNR (dB)", "Signal-to-Noise Ratio by Config",
              ref_line=10.0, ref_label="10 dB (NEF ≈ 0.10)")
    fig.tight_layout()
    fig.savefig(out_dir / "poster_snr_by_config.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_snr_by_config.png'}")

    # --- Plot 7: Box plot of NEF distribution per config ---
    nef_by_config = []
    for r in rows:
        v = r["version"]
        nefs = [s["nef"] for s in per_run_samples.get(v, []) if _isfinite(s.get("nef"))]
        nef_by_config.append(nefs)
    fig, ax = plt.subplots(figsize=(10, 5))
    _box_plot(ax, nef_by_config, "Noise Energy Fraction (NEF)", "NEF Distribution Across Samples",
              ref_line=0.10, ref_label="Target (NEF ≤ 0.10)")
    fig.tight_layout()
    fig.savefig(out_dir / "poster_nef_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_nef_boxplot.png'}")

    # --- Plot: 4-bit vs reconstructed MSE comparison (grouped bars, clean labels) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(config_labels)
    width = 0.35
    x = np.arange(n)
    mse_4bit = [r.get("mse_4bit_mean", float("nan")) for r in rows]
    mse_4bit_std = [r.get("mse_4bit_std", 0) for r in rows]
    mse_recon = [r["mse_mean"] for r in rows]
    mse_recon_std = [r["mse_std"] for r in rows]
    bars1 = ax.bar(x - width / 2, mse_4bit, width, yerr=mse_4bit_std, capsize=3, label="4-bit (vs GT)", color="tab:orange", edgecolor="black")
    bars2 = ax.bar(x + width / 2, mse_recon, width, yerr=mse_recon_std, capsize=3, label="Reconstructed (vs GT)", color="tab:green", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("4-bit vs Reconstructed: MSE vs Ground Truth")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "poster_4bit_vs_recon_mse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_4bit_vs_recon_mse.png'}")

    # --- Plot 8: Multi-metric summary (2×3) ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    _bar_plot(axes[0, 0], [r["nef_mean"] for r in rows], [r["nef_std"] for r in rows],
              "NEF", "NEF", ref_line=0.10, ref_label="Target")
    _bar_plot(axes[0, 1], [r["prd_mean"] for r in rows], [r["prd_std"] for r in rows],
              "PRD (%)", "PRD")
    _bar_plot(axes[0, 2], [r["mse_mean"] for r in rows], [r["mse_std"] for r in rows],
              "MSE", "MSE")
    _bar_plot(axes[1, 0], [r["l1_fraction_mean"] for r in rows], [r["l1_fraction_std"] for r in rows],
              "L1 fraction", "L1")
    _bar_plot(axes[1, 1], [r["correlation_mean"] for r in rows], [r["correlation_std"] for r in rows],
              "Correlation", "Correlation", higher_better=True)
    axes[1, 1].set_ylim(0, 1.05)
    _bar_plot(axes[1, 2], [r["snr_db_mean"] for r in rows], [r["snr_db_std"] for r in rows],
              "SNR (dB)", "SNR", ref_line=10.0, ref_label="10 dB")
    for ax in axes.flat:
        ax.set_xticklabels(config_labels, rotation=45, ha="right")
    fig.suptitle("ECG Reconstruction Metrics by Config", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "poster_metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'poster_metrics_summary.png'}")


if __name__ == "__main__":
    main()
