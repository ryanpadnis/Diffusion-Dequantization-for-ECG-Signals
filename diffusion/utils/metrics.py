"""Energy-based evaluation metrics for comparing signal reconstruction quality.

Designed to complement analysis.py. Import and call from compute_metrics(), or
run standalone against any results directory using the same layout conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def noise_energy_fraction(candidate: torch.Tensor, ref: torch.Tensor) -> float:
    """Fraction of reference signal energy that is reconstruction error.
    Target: NEF <= 0.10 (10% noise energy).
    """
    noise_energy = float(torch.sum((candidate - ref) ** 2))
    ref_energy = float(torch.sum(ref ** 2))
    if ref_energy == 0.0:
        return float('inf')
    return noise_energy / ref_energy


def energy_ratio(candidate: torch.Tensor, ref: torch.Tensor) -> float:
    """Ratio of candidate total energy to reference total energy.
    """
    ref_energy = float(torch.sum(ref ** 2))
    if ref_energy == 0.0:
        return float('inf')
    return float(torch.sum(candidate ** 2)) / ref_energy


def band_noise_energy_fractions(
    candidate: torch.Tensor,
    ref: torch.Tensor,
    n_bands: int = 4,
) -> list[float]:
    """Per-frequency-band NEF.

    Splits the rfft spectrum into n_bands equal-width bands and computes
    NEF independently in each band. Reveals *where* in the spectrum the
    model is struggling even when the overall NEF looks acceptable.

    Returns list of length n_bands, each value is NEF for that band.
    """
    C = torch.fft.rfft(candidate.to(torch.float32))
    R = torch.fft.rfft(ref.to(torch.float32))
    noise_spec = C - R

    n = len(R)
    band_size = max(1, n // n_bands)
    fracs: list[float] = []

    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else n
        ref_e = float((R[start:end].abs() ** 2).sum())
        noise_e = float((noise_spec[start:end].abs() ** 2).sum())
        fracs.append(noise_e / ref_e if ref_e > 0.0 else float('inf'))

    return fracs


def noise_energy_fraction_envelope(candidate: torch.Tensor, ref: torch.Tensor) -> float:
    """NEF computed on the analytic (Hilbert) envelope rather than the raw signal.

    Useful for oscillatory/bandpass signals where carrier phase errors should
    not be penalised — only amplitude-shape errors matter.
    """
    def _envelope(x: torch.Tensor) -> torch.Tensor:
        N = x.numel()
        X = torch.fft.fft(x.to(torch.float32))
        h = torch.zeros(N, dtype=X.dtype, device=X.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        return torch.abs(torch.fft.ifft(X * h))

    return noise_energy_fraction(_envelope(candidate), _envelope(ref))


# ---------------------------------------------------------------------------
# Aggregate result container
# ---------------------------------------------------------------------------

class EnergyMetrics(NamedTuple):
    """All energy metrics for a single candidate vs. reference comparison."""
    nef: float                       # overall noise energy fraction
    energy_ratio: float              # candidate energy / ref energy
    band_nefs: list[float]           # per-band NEF
    envelope_nef: float              # NEF on amplitude envelope
    n_bands: int

    @property
    def passes_threshold(self, threshold: float = 0.10) -> bool:
        return self.nef <= threshold

    def band_labels(self) -> list[str]:
        return [f"band_{i}" for i in range(self.n_bands)]

    def summary(self, label: str = "", threshold: float = 0.10) -> str:
        status = "✓" if self.nef <= threshold else "✗"
        lines = [
            f"{label}{'  ' if label else ''}NEF: {self.nef:.4f}  "
            f"[{status} {'≤' if self.nef <= threshold else '>'}{threshold:.0%} threshold]",
            f"  energy ratio:   {self.energy_ratio:.4f}  (1.0 = matched energy)",
            f"  envelope NEF:   {self.envelope_nef:.4f}",
            f"  band NEFs ({self.n_bands}):  "
            + "  ".join(f"{v:.4f}" for v in self.band_nefs),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main compute entry point — mirrors the signature of compute_metrics()
# ---------------------------------------------------------------------------

def compute_energy_metrics(
    data: dict,
    n_bands: int = 4,
    threshold: float = 0.10,
) -> dict | None:
    """Compute energy-based metrics for all trajectories vs 4bit and GT.

    Designed to be called alongside or as a drop-in complement to
    compute_metrics() in analysis.py. Accepts the same `data` dict.

    Returns
    -------
    dict with keys:
        'threshold'       : float threshold used
        '4bit_vs_gt'      : EnergyMetrics for 4-bit degraded vs ground truth
        'trajectories'    : list of dicts, one per trajectory
        'gt_band_fracs'   : reference signal's energy fraction per band
    """
    gt = data.get('16bit_gt')
    deg_4bit = data.get('4bit')
    trajectories = data.get('trajectories') or []

    if gt is None or deg_4bit is None or not trajectories:
        return None

    # Align lengths
    min_len = min(int(gt.numel()), int(deg_4bit.numel()))
    for t in trajectories:
        min_len = min(min_len, int(t.numel()))

    gt = gt[:min_len].to(torch.float32)
    deg_4bit = deg_4bit[:min_len].to(torch.float32)

    def _metrics(candidate: torch.Tensor, ref: torch.Tensor) -> EnergyMetrics:
        return EnergyMetrics(
            nef=noise_energy_fraction(candidate, ref),
            energy_ratio=energy_ratio(candidate, ref),
            band_nefs=band_noise_energy_fractions(candidate, ref, n_bands=n_bands),
            envelope_nef=noise_energy_fraction_envelope(candidate, ref),
            n_bands=n_bands,
        )

    # GT energy distribution — useful reference for interpreting band NEFs
    gt_spec = torch.abs(torch.fft.rfft(gt)) ** 2
    gt_total = float(gt_spec.sum()) + 1e-12
    n = len(gt_spec)
    band_size = max(1, n // n_bands)
    gt_band_fracs = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else n
        gt_band_fracs.append(float(gt_spec[start:end].sum()) / gt_total)

    results: dict = {
        'threshold': threshold,
        'gt_band_fracs': gt_band_fracs,
        '4bit_vs_gt': _metrics(deg_4bit, gt),
        'trajectories': [],
    }

    for i, traj in enumerate(trajectories):
        traj = traj[:min_len].to(torch.float32)
        m = _metrics(traj, gt)
        baseline_nef = results['4bit_vs_gt'].nef
        improvement_pct = (
            (baseline_nef - m.nef) / baseline_nef * 100.0
            if baseline_nef > 0 else float('nan')
        )
        results['trajectories'].append({
            'trajectory_idx': i,
            'metrics': m,
            'improvement_vs_4bit_pct': improvement_pct,
            'vs_4bit': _metrics(traj, deg_4bit),  # how much did we move from the input
        })

    return results


# ---------------------------------------------------------------------------
# Pretty printer — mirrors print_metrics() in analysis.py
# ---------------------------------------------------------------------------

def print_energy_metrics(results: dict | None) -> None:
    if not results:
        print('energy metrics: skipped (need 4bit + 16bit_gt + at least one trajectory)')
        return

    threshold = results['threshold']

    gt_fracs = results['gt_band_fracs']
    n_bands = len(gt_fracs)
    print("── GT energy distribution ──────────────────────")
    print("  " + "  ".join(f"band_{i}: {f:.3f}" for i, f in enumerate(gt_fracs)))

    print("\n── 4bit vs GT ──────────────────────────────────")
    print(results['4bit_vs_gt'].summary(threshold=threshold))

    print("\n── Trajectories ────────────────────────────────")
    baseline_nef = results['4bit_vs_gt'].nef
    for entry in results['trajectories']:
        i = entry['trajectory_idx']
        m: EnergyMetrics = entry['metrics']
        imp = entry['improvement_vs_4bit_pct']
        print(f"\n  traj_{i}  (NEF improvement vs 4bit baseline: {imp:+.2f}%)")
        print("  " + m.summary(threshold=threshold).replace("\n", "\n  "))
        vm: EnergyMetrics = entry['vs_4bit']
        print(f"  distance from 4bit input:  NEF={vm.nef:.4f}  energy_ratio={vm.energy_ratio:.4f}")


# ---------------------------------------------------------------------------
# Standalone entry point — mirrors main() in analysis.py
# ---------------------------------------------------------------------------

def main() -> None:
    # Mirror the same defaults as analysis.py main() for easy standalone use.
    import pickle
    from diffusion.analysis.analysis import load_sample_data, _resolve_sample_dir

    results_root = Path('diffusion/results')
    version = 'V7'
    run_id = '20260218_212954'
    sample_idx = 0
    sampler_type = 'ddpm'
    n_bands = 4
    threshold = 0.10

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

    results = compute_energy_metrics(data, n_bands=n_bands, threshold=threshold)
    print_energy_metrics(results)


if __name__ == '__main__':
    main()