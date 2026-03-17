# Analysis metrics: what is being compared

This doc explains the **reference signals** and **what each metric compares** in `analysis_copy.py` (and `analysis.py`).

---

## 1. Reference signals (data keys)

| Key | Meaning |
|-----|--------|
| **`16bit_gt`** (target) | Ground-truth waveform: high-resolution (16-bit) reference. This is the “truth” we try to match. |
| **`4bit`** (cond_4bit) | Conditioning signal: same content as target but **quantized to 4-bit**. Used as input to the diffusion model. |
| **`gen_0`** / **`trajectories[0]`** | Diffusion model output: the **generated** waveform (first trajectory = denoised sample). |
| **`gen_0_q4`** | Generated waveform **re-quantized to 4-bit** (for fair comparison with cond_4bit bit-depth). |
| **`gen_0_qT`** | Generated waveform **re-quantized to target bits** (e.g. 8 or 16). |
| **`gen_0_condminmax_q4`** / **`gen_0_condminmax_qT`** | Same as above but the quantizer range is set from the **condition’s min/max** (per-sample range), not global. |
| **`trajectories`** | List of time-domain series: typically `[gen_0, step_1, step_2, ...]` along the sampling path. |

So in short:
- **Target** = reference (ground truth).
- **Cond 4-bit** = low-bit input to the model.
- **Gen** = model output; **gen_0_q4 / gen_0_qT** = that output quantized for comparison.

---

## 2. `compute_metrics()` — time-domain MSE and stats

All comparisons are **time-domain** (waveform vs waveform), aligned in length.

| Metric | Comparison | Interpretation |
|--------|------------|----------------|
| **`mse_cond4_vs_target`** | cond_4bit vs 16bit_gt | How much error comes from 4-bit quantization alone (baseline). |
| **`mse_quant4(gen0)_vs_target`** | gen_0_q4 vs 16bit_gt | Error of **generated** waveform (at 4-bit resolution) vs target. |
| **`mse_quantT(gen0)_vs_target`** | gen_0_qT vs 16bit_gt | Same but at **target bit-depth** T. |
| **`mse_quant4(gen0_condminmax)_vs_target`** | gen_0_condminmax_q4 vs 16bit_gt | Generated (4-bit, cond min/max range) vs target. |
| **`mse_quantT(gen0_condminmax)_vs_target`** | gen_0_condminmax_qT vs 16bit_gt | Same at target bits. |
| **Per-trajectory `mse_vs_target`** | trajectory[i] vs 16bit_gt | How close that trajectory step is to the target. |
| **Per-trajectory `mse_vs_cond4`** | trajectory[i] vs cond_4bit | How close that step is to the **condition** (4-bit). |

So:
- **vs_target** = “how close to ground truth?”
- **vs_cond4** = “how close to the conditioning signal?”

Other entries:
- **`cond4_levels_used`** / **`cond4_estimated_step`** = stats of the 4-bit condition (number of levels, step size).
- **`cond_time_range_min/max`** = quantizer range used for the condition (when available).

---

## 3. `_analyze_one()` spectrogram/time comparison metrics

Inside `_analyze_one`, `_compute_metrics(gen_mag_v, gen_time_v)` compares **three** time-domain signals and their STFTs:

- **cond_time** = condition (4-bit) in time → **cond_mag**, **cond_phase**
- **target_time** = 16-bit ground truth → **target_mag**, **target_phase**
- **gen_time_v** = one **generated** variant (e.g. after inversion) → **gen_mag_v**

So every metric is either **condition vs target** or **generated vs target**:

| Metric | Comparison | Domain |
|--------|------------|--------|
| **`mse_spec_mag(cond4_vs_target)`** | cond_mag vs target_mag | Spectrogram magnitude |
| **`mse_spec_mag(generated_vs_target)`** | gen_mag vs target_mag | Spectrogram magnitude |
| **`mse_spec_log1p(...)`** | Same, but in **log1p(magnitude)** space | Soaks up scale differences |
| **`mse_time(cond4_vs_target)`** | cond_time vs target_time | Time-domain waveform |
| **`mse_time(generated_vs_target)`** | gen_time vs target_time | Time-domain waveform |
| **`corr_time(cond4_vs_target)`** / **`corr_time(generated_vs_target)`** | Pearson correlation of aligned time series | Shape similarity |
| **`peak_ratio_time(...)`** | max\|cond\|/max\|target\| (and gen/target) | Relative peak level |
| **`rms_ratio_time(...)`** | RMS(cond)/RMS(target), RMS(gen)/RMS(target) | Relative level |
| **`mean_ratio_spec_mag(...)`** / **`p95_ratio_spec_mag(...)`** | Mean (or 95th percentile) of spectrogram magnitude ratios | Spectral level match |
| **`improvement_pct_spec_mag`** / **`improvement_pct_time`** | (mse_cond − mse_gen) / mse_cond × 100 | % MSE reduction of generated vs condition |

So here again **target** is always the reference; **cond4** and **generated** are both compared to it (and improvement is “generated vs cond4” in terms of MSE).

---

## 4. Compact analysis (`run_compact_*`) — MSE by variant and scaling

Variants compared (when present):

- **cond_4bit**, **target**, **gen0_saved**, **recon_condphase**, **recon_gl**, **recon_hybrid**

For each variant, MSE vs **target** is computed under three scaling policies:

| Scale | Meaning |
|-------|--------|
| **none** | No rescaling; raw MSE vs target. |
| **cond4_peak** | Variant is peak-rescaled to match **cond_4bit** peak (fair, no lookahead). |
| **target_peak** | Variant is peak-rescaled to match **target** peak. |

So the **comparison** is always **variant vs target**; the **metric** is MSE (and the bar chart uses `cond4_peak` for a fair comparison across inversions).

---

## 5. Energy metrics (`diffusion.utils.metrics`)

Used alongside the above; again **candidate** is compared to **reference**:

- **Noise energy fraction (NEF)** = energy of (candidate − reference) / energy(reference). Lower = closer to reference.
- **Energy ratio** = energy(candidate) / energy(reference). Near 1 = similar overall level.
- **Band NEF** = NEF per frequency band (where error is concentrated).
- **Envelope NEF** = NEF on Hilbert envelope (amplitude shape, not phase).

Here “reference” is typically the 16-bit target; “candidate” can be cond_4bit, gen_0, or any reconstruction.

---

## Summary

- **Target** = 16-bit ground truth (the reference in almost all metrics).
- **Cond 4-bit** = low-bit input; metrics compare it to target to see “quantization-only” error.
- **Generated** = model output (and its quantized versions); metrics compare it to target and sometimes to cond_4bit.
- **MSE vs target** = “how close to ground truth?”; **MSE vs cond4** = “how close to the condition?”; **improvement_%** = “how much did the model improve over the condition?”
