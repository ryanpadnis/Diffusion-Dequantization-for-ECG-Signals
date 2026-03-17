# Time-Domain ECG Reconstruction Metrics — Summary

This document describes all metrics computed in the V9 sweep (`scripts/v9_time_domain_sweep.py`) and how they are calculated.

---

## What We Compare

| Term | Meaning |
|------|---------|
| **Reference (ref)** | 16-bit ground-truth ECG waveform |
| **Candidate (cand)** | Diffusion model output (generated reconstruction) |
| **Comparison** | Candidate vs reference, aligned in length |

---

## Metrics (Formulas & Interpretation)

### 1. NEF — Noise Energy Fraction

**Formula:**
```
NEF = Σ(cand - ref)² / Σ(ref²)
```

**Interpretation:** Fraction of reference energy that is reconstruction error.  
**Target:** NEF ≤ 0.10 (10% noise energy). Lower = better.  
**Scale-invariant:** Yes.

---

### 2. MSE — Mean Squared Error

**Formula:**
```
MSE = mean((cand - ref)²)
```

**Interpretation:** Average squared error.  
**Lower = better.** Not scale-invariant; compare within same dataset.

---

### 3. NMSE — Normalized Mean Squared Error

**Formula:**
```
NMSE = MSE / mean(ref²)
```

**Interpretation:** Scale-invariant MSE; equals NEF when signals are aligned.  
**Lower = better.**

---

### 4. L1 Fraction

**Formula:**
```
L1 fraction = Σ|cand - ref| / Σ|ref|
```

**Interpretation:** L1 analog of NEF. Fraction of reference L1 norm that is error.  
**Lower = better.** More robust to outliers than L2.

---

### 5. PRD — Percent Root-mean-square Difference

**Formula:**
```
PRD = sqrt(NEF) × 100  =  sqrt(Σ(cand-ref)² / Σ(ref²)) × 100
```

**Interpretation:** Common in ECG literature.  
**Lower = better.** PRD ≈ 31.6% when NEF = 0.10.

---

### 6. Correlation — Pearson Coefficient

**Formula:**
```
corr = Σ((cand - mean(cand)) × (ref - mean(ref))) / sqrt(Σ(cand-mean)² × Σ(ref-mean)²)
```

**Interpretation:** Linear correlation between candidate and reference.  
**Range:** [-1, 1]. 1 = perfect match. Higher = better.

---

### 7. SNR — Signal-to-Noise Ratio (dB)

**Formula:**
```
SNR (dB) = 10 × log₁₀(1 / NEF) = -10 × log₁₀(NEF)
```

**Interpretation:** Standard dB measure. NEF = 0.10 ⇒ SNR ≈ 10 dB.  
**Higher = better.**

---

### 8. Energy Ratio

**Formula:**
```
energy_ratio = Σ(cand²) / Σ(ref²)
```

**Interpretation:** Ratio of candidate energy to reference.  
**Target:** ≈ 1.0. Below 1 = under-energetic; above 1 = over-energetic.

---

### 9. Band NEFs (4 bands)

**Formula:** Same as NEF, but computed per frequency band (splitting the RFFT spectrum into 4 equal bands).

**Interpretation:** Shows where in the spectrum error is concentrated. Useful for diagnosing frequency-specific issues.

---

### 10. Envelope NEF

**Formula:** NEF computed on the analytic (Hilbert) envelope instead of the raw signal.

**Interpretation:** Emphasizes amplitude-shape error, not carrier phase. Useful for ECG morphology.

---

### 11. Improvement % (vs 4-bit baseline)

**Formulas:**
```
NEF improvement  = (NEF_4bit - NEF_traj) / NEF_4bit × 100
MSE improvement  = (MSE_4bit - MSE_traj) / MSE_4bit × 100
L1 improvement   = (L1_4bit - L1_traj) / L1_4bit × 100
```

**Interpretation:** % reduction in error from 4-bit input to generated output.  
**Positive = improvement**; negative = model made it worse.

---

### 12. Paired Wins

**Meaning:** For each sample index present in all configs, the config with the lowest NEF “wins” that sample.  
**Count:** Number of samples each config won.  
**Interpretation:** Head-to-head comparison on the same ECGs.

---

## Aggregation in the Sweep

- **Per sample:** Metrics computed for trajectory_0 vs 16bit_gt.
- **Per config:** Mean ± std across all samples in that config.
- **Ranking:** By mean NEF (lower first).

---

## Poster Plots Generated

| File | Content |
|------|---------|
| `poster_nef_by_config.png` | Bar chart: mean NEF ± std |
| `poster_prd_by_config.png` | Bar chart: mean PRD ± std |
| `poster_mse_by_config.png` | Bar chart: mean MSE ± std |
| `poster_l1_by_config.png` | Bar chart: mean L1 fraction ± std |
| `poster_correlation_by_config.png` | Bar chart: mean correlation |
| `poster_snr_by_config.png` | Bar chart: mean SNR (dB) |
| `poster_nef_boxplot.png` | Box plot: NEF distribution per config |
| `poster_metrics_summary.png` | 2×3: NEF, PRD, MSE, L1, correlation, SNR |

---

## References

- NEF: standard in signal reconstruction
- PRD: ECG literature (e.g., Zigel et al.)
- SNR: 10·log₁₀(signal/noise)
