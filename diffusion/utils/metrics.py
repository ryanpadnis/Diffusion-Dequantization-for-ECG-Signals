"""Energy-based evaluation metrics for comparing signal reconstruction quality.

Designed to complement analysis.py. Import and call from compute_metrics(), or
run standalone against any results directory using the same layout conventions.
"""

from __future__ import annotations

import math
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


def spectral_nef(candidate_mag: torch.Tensor, ref_mag: torch.Tensor) -> float:
    """Spectral-domain NEF: fraction of reference magnitude energy that is error.

    Compares spectrogram magnitudes (not time-domain). Use when target is only
    available as target_spec.pt (V8) or for spectral-only comparison.
    """
    c = candidate_mag.detach().to(torch.float32).reshape(-1)
    r = ref_mag.detach().to(torch.float32).reshape(-1)
    n = min(int(c.numel()), int(r.numel()))
    if n <= 0:
        return float('inf')
    c, r = c[:n], r[:n]
    noise_energy = float(torch.sum((c - r) ** 2))
    ref_energy = float(torch.sum(r ** 2))
    if ref_energy == 0.0:
        return float('inf')
    return noise_energy / ref_energy


def mse(candidate: torch.Tensor, ref: torch.Tensor) -> float:
    """Mean squared error between candidate and reference."""
    c = candidate.detach().to(torch.float32).reshape(-1)
    r = ref.detach().to(torch.float32).reshape(-1)
    n = min(int(c.numel()), int(r.numel()))
    if n <= 0:
        return float('nan')
    c, r = c[:n], r[:n]
    return float(torch.mean((c - r) ** 2))


def nmse(candidate: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> float:
    """Normalized MSE: MSE / mean(ref²). Scale-invariant; equals NEF for aligned signals."""
    c = candidate.detach().to(torch.float32).reshape(-1)
    r = ref.detach().to(torch.float32).reshape(-1)
    n = min(int(c.numel()), int(r.numel()))
    if n <= 0:
        return float('nan')
    c, r = c[:n], r[:n]
    mse_val = float(torch.mean((c - r) ** 2))
    ref_power = float(torch.mean(r ** 2)) + eps
    return mse_val / ref_power


def prd(candidate: torch.Tensor, ref: torch.Tensor) -> float:
    """Percent Root-mean-square Difference (ECG literature standard).
    PRD = sqrt(sum((ref-rec)²) / sum(ref²)) * 100. Equals sqrt(NEF)*100."""
    nf = noise_energy_fraction(candidate, ref)
    if nf <= 0:
        return 0.0
    return float(math.sqrt(nf) * 100.0)


def correlation(candidate: torch.Tensor, ref: torch.Tensor) -> float:
    """Pearson correlation coefficient between candidate and reference. Range [-1, 1], 1 = perfect."""
    c = candidate.detach().to(torch.float32).reshape(-1)
    r = ref.detach().to(torch.float32).reshape(-1)
    n = min(int(c.numel()), int(r.numel()))
    if n < 2:
        return float('nan')
    c, r = c[:n], r[:n]
    c_cent = c - c.mean()
    r_cent = r - r.mean()
    num = float((c_cent * r_cent).sum())
    den = float(torch.sqrt((c_cent ** 2).sum() * (r_cent ** 2).sum())) + 1e-12
    return num / den


def snr_db(candidate: torch.Tensor, ref: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB. SNR = 10*log10(signal_power / noise_power) = -10*log10(NEF)."""
    nf = noise_energy_fraction(candidate, ref)
    if nf <= 0:
        return float('inf')
    return float(10.0 * math.log10(1.0 / nf))


def l1_fraction(candidate: torch.Tensor, ref: torch.Tensor) -> float:
    """L1 fraction: sum(|cand - ref|) / sum(|ref|). Analogous to NEF but L1-based."""
    c = candidate.detach().to(torch.float32).reshape(-1)
    r = ref.detach().to(torch.float32).reshape(-1)
    n = min(int(c.numel()), int(r.numel()))
    if n <= 0:
        return float('nan')
    c, r = c[:n], r[:n]
    ref_l1 = float(torch.sum(torch.abs(r))) + 1e-12
    return float(torch.sum(torch.abs(c - r))) / ref_l1


def spectral_energy_ratio(candidate_mag: torch.Tensor, ref_mag: torch.Tensor) -> float:
    """Ratio of candidate magnitude energy to reference magnitude energy."""
    c = candidate_mag.detach().to(torch.float32).reshape(-1)
    r = ref_mag.detach().to(torch.float32).reshape(-1)
    n = min(int(c.numel()), int(r.numel()))
    if n <= 0:
        return float('inf')
    r = r[:n]
    ref_energy = float(torch.sum(r ** 2))
    if ref_energy == 0.0:
        return float('inf')
    c = c[:n]
    return float(torch.sum(c ** 2)) / ref_energy


def band_spectral_nefs(
    candidate_mag: torch.Tensor,
    ref_mag: torch.Tensor,
    n_bands: int = 4,
) -> list[float]:
    """Per-frequency-band spectral NEF. Magnitudes treated as [F, T] (freq, time)."""
    c = candidate_mag.detach().to(torch.float32)
    r = ref_mag.detach().to(torch.float32)
    while c.ndim > 2:
        c = c.squeeze(0)
    while r.ndim > 2:
        r = r.squeeze(0)
    if c.ndim != 2 or r.ndim != 2:
        return [float('inf')] * n_bands
    F = min(c.shape[0], r.shape[0])
    T = min(c.shape[1], r.shape[1])
    c = c[:F, :T].reshape(-1)
    r = r[:F, :T].reshape(-1)
    band_size = max(1, F // n_bands)
    fracs: list[float] = []
    ref_total = float((r ** 2).sum()) + 1e-12
    for i in range(n_bands):
        start_f = i * band_size
        end_f = start_f + band_size if i < n_bands - 1 else F
        start_idx = start_f * T
        end_idx = end_f * T
        r_band = r[start_idx:end_idx]
        c_band = c[start_idx:end_idx]
        ref_e = float((r_band ** 2).sum())
        noise_e = float(((c_band - r_band) ** 2).sum())
        # Use floor to avoid NEF explosion when band has negligible ref energy
        ref_e_safe = max(ref_e, 1e-12 * ref_total)
        fracs.append(noise_e / ref_e_safe if ref_e_safe > 0.0 else 0.0)
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
    mse: float                      # mean squared error
    nmse: float                     # MSE / mean(ref²), scale-invariant (≈ NEF)
    l1_fraction: float              # L1 fraction: sum(|diff|)/sum(|ref|)
    prd: float                      # Percent Root-mean-square Difference (ECG standard)
    correlation: float              # Pearson correlation (1 = perfect)
    snr_db: float                   # Signal-to-noise ratio in dB

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
            f"  MSE: {self.mse:.6f}   NMSE: {self.nmse:.4f}   L1 fraction: {self.l1_fraction:.4f}",
            f"  PRD: {self.prd:.2f}%   correlation: {self.correlation:.4f}   SNR: {self.snr_db:.2f} dB",
            f"  energy ratio:   {self.energy_ratio:.4f}  (1.0 = matched energy)",
            f"  envelope NEF:   {self.envelope_nef:.4f}",
            f"  band NEFs ({self.n_bands}):  "
            + "  ".join(f"{v:.4f}" for v in self.band_nefs),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spectral-domain metrics (for V8 / target_spec.pt only)
# ---------------------------------------------------------------------------

class SpectralEnergyMetrics(NamedTuple):
    """Spectral-domain energy metrics (magnitude spectrograms)."""
    nef: float
    energy_ratio: float
    band_nefs: list[float]
    n_bands: int
    mse: float
    nmse: float
    l1_fraction: float

    def summary(self, label: str = "", threshold: float = 0.10) -> str:
        status = "✓" if self.nef <= threshold else "✗"
        lines = [
            f"{label}{'  ' if label else ''}spectral NEF: {self.nef:.4f}  "
            f"[{status} {'≤' if self.nef <= threshold else '>'}{threshold:.0%} threshold]",
            f"  spectral MSE: {self.mse:.6f}   NMSE: {self.nmse:.4f}   L1 fraction: {self.l1_fraction:.4f}",
            f"  spectral energy ratio: {self.energy_ratio:.4f}  (1.0 = matched)",
            f"  band NEFs ({self.n_bands}):  " + "  ".join(f"{v:.4f}" for v in self.band_nefs),
        ]
        return "\n".join(lines)


def compute_spectral_energy_metrics(
    target_mag: torch.Tensor,
    cond4_mag: torch.Tensor,
    traj_mags: list[torch.Tensor],
    n_bands: int = 4,
    threshold: float = 0.10,
) -> dict:
    """Compute spectral-domain energy metrics (works with target_spec.pt, no time-domain target needed).

    Uses magnitude spectrograms. Works for V8 (target_spec.pt only) and V9.
    """
    def _align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = a.detach().to(torch.float32)
        b = b.detach().to(torch.float32)
        while a.ndim > 2:
            a = a.squeeze(0)
        while b.ndim > 2:
            b = b.squeeze(0)
        if a.ndim != 2 or b.ndim != 2:
            return a.reshape(-1), b.reshape(-1)
        F, T = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        return a[:F, :T].reshape(-1), b[:F, :T].reshape(-1)

    def _to_2d(mag: torch.Tensor) -> torch.Tensor:
        m = mag.detach().to(torch.float32)
        while m.ndim > 2:
            m = m.squeeze(0)
        return m if m.ndim == 2 else m.unsqueeze(0)

    def _spec_metrics(cand: torch.Tensor, ref: torch.Tensor) -> SpectralEnergyMetrics:
        cb, rb = _to_2d(cand), _to_2d(ref)
        return SpectralEnergyMetrics(
            nef=spectral_nef(cand, ref),
            energy_ratio=spectral_energy_ratio(cand, ref),
            band_nefs=band_spectral_nefs(cb, rb, n_bands=n_bands),
            n_bands=n_bands,
            mse=mse(cand, ref),
            nmse=nmse(cand, ref),
            l1_fraction=l1_fraction(cand, ref),
        )

    # Target magnitude energy distribution
    r = _to_2d(target_mag)
    total = float((r ** 2).sum()) + 1e-12
    F = r.shape[0]
    band_size = max(1, F // n_bands)
    gt_band_fracs = []
    for i in range(n_bands):
        start = i * band_size
        end = start + band_size if i < n_bands - 1 else F
        gt_band_fracs.append(float((r[start:end] ** 2).sum()) / total)

    results: dict = {
        'threshold': threshold,
        'gt_band_fracs': gt_band_fracs,
        '4bit_vs_target': _spec_metrics(cond4_mag, target_mag),
        'trajectories': [],
    }

    baseline_nef = results['4bit_vs_target'].nef
    for i, traj_mag in enumerate(traj_mags):
        m = _spec_metrics(traj_mag, target_mag)
        improvement_pct = (
            (baseline_nef - m.nef) / baseline_nef * 100.0
            if baseline_nef > 0 else float('nan')
        )
        results['trajectories'].append({
            'trajectory_idx': i,
            'metrics': m,
            'improvement_vs_4bit_pct': improvement_pct,
            'vs_4bit': _spec_metrics(traj_mag, cond4_mag),
        })

    return results


def print_spectral_energy_metrics(results: dict | None) -> None:
    """Print spectral energy metrics (for target_spec.pt–based runs)."""
    if not results:
        return
    threshold = results['threshold']
    print("── Spectral energy metrics (target_spec magnitude) ─────────────")
    print("  " + "  ".join(f"band_{i}: {f:.3f}" for i, f in enumerate(results['gt_band_fracs'])))
    print("\n── 4bit vs target (spectral) ──────────────────────────────────")
    print(results['4bit_vs_target'].summary(threshold=threshold))
    print("\n── Trajectories (spectral) ────────────────────────────────────")
    baseline = results['4bit_vs_target'].nef
    for entry in results['trajectories']:
        i = entry['trajectory_idx']
        m = entry['metrics']
        imp = entry['improvement_vs_4bit_pct']
        vs4 = entry['vs_4bit']
        print(f"\n  traj_{i}  (spectral NEF improvement vs 4bit: {imp:+.2f}%)")
        print("  " + m.summary(threshold=threshold).replace("\n", "\n  "))
        print(f"  distance from 4bit (spectral):  NEF={vs4.nef:.4f}  NMSE={vs4.nmse:.4f}  L1={vs4.l1_fraction:.4f}  energy_ratio={vs4.energy_ratio:.4f}")


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
            mse=mse(candidate, ref),
            nmse=nmse(candidate, ref),
            l1_fraction=l1_fraction(candidate, ref),
            prd=prd(candidate, ref),
            correlation=correlation(candidate, ref),
            snr_db=snr_db(candidate, ref),
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

    baseline = results['4bit_vs_gt']
    for i, traj in enumerate(trajectories):
        traj = traj[:min_len].to(torch.float32)
        m = _metrics(traj, gt)
        nef_imp = (
            (baseline.nef - m.nef) / baseline.nef * 100.0
            if baseline.nef > 0 else float('nan')
        )
        mse_imp = (
            (baseline.mse - m.mse) / baseline.mse * 100.0
            if baseline.mse > 0 else float('nan')
        )
        l1_imp = (
            (baseline.l1_fraction - m.l1_fraction) / baseline.l1_fraction * 100.0
            if baseline.l1_fraction > 0 else float('nan')
        )
        results['trajectories'].append({
            'trajectory_idx': i,
            'metrics': m,
            'improvement_vs_4bit_pct': nef_imp,
            'mse_improvement_pct': mse_imp,
            'l1_improvement_pct': l1_imp,
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
    for entry in results['trajectories']:
        i = entry['trajectory_idx']
        m: EnergyMetrics = entry['metrics']
        nef_imp = entry['improvement_vs_4bit_pct']
        mse_imp = entry['mse_improvement_pct']
        l1_imp = entry['l1_improvement_pct']
        print(f"\n  traj_{i}  (vs 4bit: NEF {nef_imp:+.2f}%  MSE {mse_imp:+.2f}%  L1 {l1_imp:+.2f}%)")
        print("  " + m.summary(threshold=threshold).replace("\n", "\n  "))
        vm: EnergyMetrics = entry['vs_4bit']
        print(f"  distance from 4bit input:  NEF={vm.nef:.4f}  NMSE={vm.nmse:.4f}  L1={vm.l1_fraction:.4f}  corr={vm.correlation:.4f}  SNR={vm.snr_db:.2f}dB")


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