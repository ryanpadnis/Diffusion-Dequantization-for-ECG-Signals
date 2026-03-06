#!/usr/bin/env python3
"""
Simple script demonstrating the use of metrics.py for signal quality evaluation.

This script creates sample tensors and computes energy-based metrics using
the functions from diffusion.utils.metrics, similar to how analysis.py
computes metrics for diffusion model outputs.
"""

import torch
import numpy as np
from pathlib import Path
from diffusion.utils.metrics import (
    noise_energy_fraction,
    energy_ratio,
    band_noise_energy_fractions,
    noise_energy_fraction_envelope,
    EnergyMetrics,
    compute_energy_metrics,
    print_energy_metrics,
)
from diffusion.analysis.analysis import load_sample_data, _resolve_sample_dir


def create_sample_signals():
    """Create sample reference and candidate signals for demonstration."""
    # Create a sample reference signal (ground truth)
    # Using a simple sine wave with some noise
    t = torch.linspace(0, 2 * np.pi, 1000)
    ref_signal = torch.sin(2 * t) + 0.1 * torch.randn_like(t)

    # Create a candidate signal (reconstruction) with some distortion
    # Add quantization noise and slight amplitude error
    candidate_signal = torch.round(ref_signal * 16) / 16  # 4-bit quantization effect
    candidate_signal += 0.05 * torch.randn_like(candidate_signal)  # reconstruction noise

    return ref_signal, candidate_signal


def load_real_sample(version='V7', run_id='20260218_212954', sample_idx=0, sampler_type='ddpm'):
    """Load a real sample from the diffusion results directory."""
    results_root = Path('diffusion/results')
    results_dir = results_root / version
    run_version = run_id

    sample_dir, detected_sampler = _resolve_sample_dir(
        results_dir, run_version, sample_idx, sampler_type=sampler_type
    )

    data = load_sample_data(
        results_dir, run_version, sample_idx,
        sample_dir=sample_dir,
        sampler_type=detected_sampler,
    )

    return data


def demonstrate_individual_metrics():
    """Demonstrate using individual metric functions."""
    print("=== Individual Metric Functions Demo ===")

    ref, candidate = create_sample_signals()

    nef = noise_energy_fraction(candidate, ref)
    print(".4f")

    er = energy_ratio(candidate, ref)
    print(".4f")

    band_nefs = band_noise_energy_fractions(candidate, ref, n_bands=4)
    print(f"Band NEFs (4 bands): {['.4f' for b in band_nefs]}")

    env_nef = noise_energy_fraction_envelope(candidate, ref)
    print(".4f")

    # Create EnergyMetrics object
    metrics = EnergyMetrics(
        nef=nef,
        energy_ratio=er,
        band_nefs=band_nefs,
        envelope_nef=env_nef,
        n_bands=4
    )

    print("\nEnergyMetrics summary:")
    print(metrics.summary())


def demonstrate_batch_metrics(version='V7', run_id='20260218_212954', sample_idx=0, sampler_type='ddpm'):
    """Demonstrate using compute_energy_metrics with real data from diffusion results."""
    print(f"\n=== Batch Metrics Demo (using real data: {version}/{run_id}/sample_{sample_idx}) ===")

    # Load real sample data
    data = load_real_sample(version=version, run_id=run_id, sample_idx=sample_idx, sampler_type=sampler_type)

    if not data:
        print("Failed to load sample data. Check the version, run_id, and sample_idx.")
        return

    # Compute metrics
    results = compute_energy_metrics(data, n_bands=4, threshold=0.10)

    # Print results
    print_energy_metrics(results)


def main():
    """Main function to run the demonstration."""
    print("Simple Metrics Script (using real diffusion samples)")
    print("=" * 60)

    # Configuration - change these to analyze different samples
    VERSION = 'V7'          # Which version directory (V1, V2, V3, etc.)
    RUN_ID = '20260218_212954'  # Which run ID
    SAMPLE_IDX = 0          # Which sample index (0, 1, 2, etc.)
    SAMPLER_TYPE = 'ddpm'   # 'ddpm' or 'ddim'

    print(f"Analyzing: {VERSION}/{RUN_ID}/sample_{SAMPLE_IDX} ({SAMPLER_TYPE})")
    print()

    # Set random seed for reproducible results (if needed)
    torch.manual_seed(42)
    np.random.seed(42)

    demonstrate_individual_metrics()  # Still uses synthetic data for basic demo
    demonstrate_batch_metrics(
        version=VERSION,
        run_id=RUN_ID,
        sample_idx=SAMPLE_IDX,
        sampler_type=SAMPLER_TYPE
    )

    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == '__main__':
    main()