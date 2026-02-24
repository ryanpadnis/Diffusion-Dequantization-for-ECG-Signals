"""Sample a trained diffusion model and reconstruct outputs."""

import torch
import pickle
import math
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from diffusion.utils.torch_utils import resolve_torch_dtype


class DiffusionSampler:
    """Load a trained diffusion model and generate samples."""
    
    def __init__(self, results_dir: Path, version: str = "V1", checkpoint_name: str = "checkpoint_epoch_3.pt", device: str = "cpu"):
        """Initialize sampler."""
        self.results_dir = Path(results_dir)
        self.version = version
        self.version_dir = self.results_dir / version
        self.device = torch.device(device)
        
        config_path = self.version_dir / 'config.pkl'
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)
        print(f"[Sampler] Loaded config from: {config_path}")
        
        from diffusion.utils.diffusion_models import create_diffuser
        checkpoint_path = self.version_dir / 'checkpoints' / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Build model exactly like training, but force device/dtype per sampler.
        cfg = dict(self.config)
        cfg['device'] = str(self.device)
        self.model = create_diffuser(cfg)

        # This loads UNet + condition encoder weights (cond_encoder) since they're
        # part of the model's state_dict.
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.eval()
        print(f"[Sampler] Loaded model from: {checkpoint_path}")

    def _load_normalizer_params(self) -> Optional[dict]:
        # Kept for backwards compatibility with older workflows, but not required
        # when sampling quantizes per-sample in real time.
        cand = self.version_dir / 'data' / 'normalizer_params.pkl'
        if cand.exists():
            try:
                with open(cand, 'rb') as f:
                    norm = pickle.load(f)
                print(f"[Sampler] Loaded normalizer params from: {cand}")
                return norm
            except Exception as e:
                print(f"[Sampler] Warning: failed reading {cand} (non-fatal): {e}")
        return None

    def _compute_sampling_params(self, raw_signals: torch.Tensor, *, debug_quant: bool = False) -> dict:
        """Compute non-quantization preprocessing params.

        Time-domain quantization is intentionally per-sample and computed in real time
        from each condition waveform only (no global ranges).
        """
        from data.utils.quantizers import compute_range_from_tensor
        from data.utils.transforms import get_transform
        from data.utils.quantizers import UniformQuantizer

        from diffusion.utils.quant_debug import (
            summarize_tensor,
            summarize_uniform_quantizer,
            summarize_quantization_usage,
        )

        cond_bits = int(self.config.get('bit_size', 4))
        real_bits = int(self.config.get('real_bit_size', 16))

        pipeline_config = (self.config.get('pipeline_config') or {}).copy()
        transform_type = pipeline_config.pop('transform', 'stft')
        transform4 = get_transform(transform_type, **pipeline_config, device=self.device)
        transform16 = get_transform(transform_type, **pipeline_config, device=self.device)

        return {
            'cond_bits': cond_bits,
            'real_bits': real_bits,
            'pipeline_config': (self.config.get('pipeline_config') or {}).copy(),
        }

    def _maybe_pre_lowpass(self, raw_signals: torch.Tensor) -> torch.Tensor:
        """Optionally low-pass raw signals to match training preprocessing."""
        lp_cfg = self.config.get('pre_lowpass') if isinstance(self.config.get('pre_lowpass'), dict) else {}
        lp_cutoff_hz = float(lp_cfg.get('cutoff_hz', 0.0) or 0.0)
        lp_sample_rate_hz = float(lp_cfg.get('sample_rate_hz', 0.0) or 0.0)
        lp_enabled = bool(lp_cfg.get('enabled', False)) and lp_cutoff_hz > 0 and lp_sample_rate_hz > 0
        if not lp_enabled:
            return raw_signals

        x = raw_signals.detach().to(torch.float32)
        n = int(x.shape[-1])
        cutoff = min(float(lp_cutoff_hz), 0.5 * float(lp_sample_rate_hz))
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(n, d=1.0 / float(lp_sample_rate_hz)).to(X.device)
        mask = (freqs <= cutoff).to(X.dtype)
        return torch.fft.irfft(X * mask, n=n, dim=-1).to(torch.float32)

    def _compute_target_spec_norm(
        self,
        raw_signals: torch.Tensor,
        *,
        params: dict,
        cond_min: torch.Tensor,
        cond_denom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute target spectrogram in the *training-normalized* space.

        Training normalizes real magnitudes using ONLY condition-derived stats.
        This reproduces that so pre-inversion MSEs are apples-to-apples.
        """
        from data.utils.transforms import get_transform

        raw_signals = self._maybe_pre_lowpass(raw_signals)

        pipeline_config = params['pipeline_config'].copy()
        transform_type = pipeline_config.pop('transform', 'stft')
        transform = get_transform(transform_type, **pipeline_config, device=self.device)

        mag = transform.apply(raw_signals.to(self.device))
        phase = getattr(transform, 'phase', None)

        if mag.ndim == 3:
            mag = mag.unsqueeze(1)
        mag = mag.to(torch.float32)

        mag_norm_mode = str(self.config.get('mag_norm_mode', 'minmax') or 'minmax').strip().lower()
        if mag_norm_mode in {'none', 'off', 'identity'}:
            # No normalization — pass raw magnitude through; denorm is identity
            norm = mag
            norm = torch.clamp(norm, min=0.0)  # magnitudes stay non-negative
        elif mag_norm_mode in {'zscore', 'z_score', 'standardize'}:
            # cond_min holds mean, cond_denom holds std * clamp_sigma
            cmin = cond_min.to(torch.float32)
            denom = cond_denom.to(torch.float32)
            norm = (mag - cmin) / denom
            norm = torch.clamp(norm, -1.0, 1.0)
        elif mag_norm_mode in {'absmax', 'peak', 'max'}:
            denom = cond_denom.to(torch.float32)
            norm = mag / denom
            norm = norm * 2.0 - 1.0
            norm = torch.clamp(norm, -1.0, 1.0)
        else:
            cmin = cond_min.to(torch.float32)
            denom = cond_denom.to(torch.float32)
            norm = (mag - cmin) / denom
            norm = norm * 2.0 - 1.0
            norm = torch.clamp(norm, -1.0, 1.0)

        if phase is None:
            return norm, None

        # Phase normalized to [-1,1] (angle/pi) matches training when use_phase_channel=True.
        ph = phase.to(torch.float32)
        if ph.ndim == 3:
            ph = ph.unsqueeze(1)
        ph_norm = torch.clamp(ph / math.pi, -1.0, 1.0)
        return norm, ph_norm
    
    def preprocess_condition(self, raw_signals: torch.Tensor, params: dict, *, debug_quant: bool = False) -> torch.Tensor:
        """Preprocess condition: time-quantize -> STFT magnitude -> normalize to [-1, 1].

        IMPORTANT: This must match training normalization.
        Training uses per-sample min/max of the *condition spectrogram magnitude*
        to normalize BOTH condition and real magnitudes.
        """
        from data.utils.transforms import get_transform
        from data.utils.quantizers import UniformQuantizer

        # Optional: time-domain low-pass (must match training if enabled).
        raw_signals = self._maybe_pre_lowpass(raw_signals)

        cond_bits = int(params['cond_bits']) #use the conditional data bit sizew

        # Per-sample time-domain quantization range from this condition only.
        from data.utils.quantizers import compute_range_from_tensor
        lower_pct = float(self.config.get('quantile_clip_lower', 0.0))
        upper_pct = float(self.config.get('quantile_clip_upper', 100.0))
        lo, hi = compute_range_from_tensor(raw_signals, lower_pct, upper_pct)
        peak = max(abs(float(lo)), abs(float(hi)))
        if peak <= 0:
            peak = float(raw_signals.detach().abs().max().item())
        if peak <= 0:
            peak = 1.0
        tmin = -peak
        tmax = peak

        q_time = UniformQuantizer(bits=cond_bits, range_min=float(tmin), range_max=float(tmax))
        q_time._meta = {'source': 'per_sample', 'lower_pct': lower_pct, 'upper_pct': upper_pct, 'peak': float(peak)}
        wave_4bit = q_time.quantize(raw_signals.to(self.device))
        self._last_wave_4bit = wave_4bit.detach().clone()

        if debug_quant:
            from diffusion.utils.quant_debug import summarize_uniform_quantizer, summarize_quantization_usage, summarize_tensor
            print('[Sampler] Condition preprocess quantization diagnostics')
            summarize_uniform_quantizer(q_time, f'cond_time_q{cond_bits}')
            summarize_tensor(raw_signals[:1], 'raw_time(sample_0)')
            summarize_tensor(wave_4bit[:1], 'cond_time_quantized(sample_0)')
            summarize_quantization_usage(raw_signals[:1], q_time, f'cond_time_q{cond_bits}_usage(sample_0)')

        pipeline_config = params['pipeline_config'].copy()
        transform_type = pipeline_config.pop('transform', 'stft')
        transform = get_transform(transform_type, **pipeline_config, device=self.device)
        mag = transform.apply(wave_4bit)

        phase = getattr(transform, 'phase', None)
        if phase is None:
            self._last_phase = None
        else:
            self._last_phase = phase.detach().clone()
        self._last_input_length = int(wave_4bit.shape[-1])

        if mag.ndim == 2:
            raise ValueError(
                f"Transform '{transform_type}' returned 2D output {tuple(mag.shape)}; "
                "configure the transform to return a 2D grid (e.g. Haar out_shape=(H,W))."
            )
        if mag.ndim == 3:
            mag = mag.unsqueeze(1)  # [B, F, T] -> [B, 1, F, T]

        use_phase_channel = bool(self.config.get('use_phase_channel', False))
        if use_phase_channel:
            if phase is None:
                raise ValueError("use_phase_channel=True requires a transform that exposes `phase` (e.g. STFT/DFT).")
            if phase.ndim == 3:
                phase_ch = phase.unsqueeze(1)  # [B, F, T] -> [B, 1, F, T]
            elif phase.ndim == 4 and phase.shape[1] == 1:
                phase_ch = phase
            else:
                raise ValueError(f"Unexpected phase shape for phase channel: {tuple(phase.shape)}")

            # Normalize phase to [-1, 1] by dividing by pi.
            phase_norm = torch.clamp(phase_ch.to(torch.float32) / math.pi, -1.0, 1.0)

        # Per-sample normalization (must match diffusion/utils/runner.py)
        mag_norm_mode = str(self.config.get('mag_norm_mode', 'minmax') or 'minmax').strip().lower()
        mag_norm_eps = float(self.config.get('mag_norm_epsilon', 1e-8) or 1e-8)
        if mag_norm_eps <= 0:
            mag_norm_eps = 1e-8

        if mag_norm_mode in {'zscore', 'z_score', 'standardize'}:
            mag_norm_mode = 'zscore'
            clamp_sigma = float(self.config.get('mag_norm_clamp_sigma', 3.0) or 3.0)
            if clamp_sigma <= 0:
                clamp_sigma = 3.0
            cond_mean = mag.mean(dim=(2, 3), keepdim=True)
            cond_std = mag.std(dim=(2, 3), keepdim=True, unbiased=False)
            denom = cond_std * clamp_sigma
            denom = torch.where(denom.abs() < mag_norm_eps, torch.ones_like(denom), denom)
            cond_min = cond_mean  # store mean as offset for denorm

            normalized = (mag - cond_mean) / denom
            normalized = torch.clamp(normalized, -1.0, 1.0)
        elif mag_norm_mode in {'absmax', 'peak', 'max'}:
            mag_norm_mode = 'absmax'
            # Magnitudes are non-negative: use peak as scale and ignore min.
            cond_peak = mag.amax(dim=(2, 3), keepdim=True)
            denom = torch.where(cond_peak.abs() < mag_norm_eps, torch.ones_like(cond_peak), cond_peak)
            cond_min = torch.zeros_like(denom)
            normalized = mag / denom
            normalized = normalized * 2.0 - 1.0
            normalized = torch.clamp(normalized, -1.0, 1.0)
        elif mag_norm_mode in {'none', 'off', 'identity'}:
            mag_norm_mode = 'none'
            # Identity: no normalization — pass raw magnitude through (denom=1, min=0)
            cond_min = torch.zeros(mag.shape[0], 1, 1, 1, dtype=mag.dtype, device=mag.device)
            denom = torch.ones_like(cond_min)
            normalized = mag  # no transform; magnitudes are already non-negative
        else:
            mag_norm_mode = 'minmax'
            # Default: min/max range.
            cond_min = mag.amin(dim=(2, 3), keepdim=True)
            cond_max = mag.amax(dim=(2, 3), keepdim=True)
            denom = cond_max - cond_min
            denom = torch.where(denom.abs() < mag_norm_eps, torch.ones_like(denom), denom)

            normalized = (mag - cond_min) / denom
            normalized = normalized * 2.0 - 1.0
            normalized = torch.clamp(normalized, -1.0, 1.0)

        # Save for postprocess inversion.
        self._last_cond_min = cond_min.detach().clone()
        self._last_cond_denom = denom.detach().clone()
        self._last_cond_norm_mode = mag_norm_mode

        print(
            f"  Preprocess condition - shape: {normalized.shape}, dtype: {normalized.dtype}, "
            f"range: [{normalized.min():.4f}, {normalized.max():.4f}]"
        )
        if debug_quant:
            # Print the *per-sample* normalization statistics.
            cm0 = float(cond_min[0].item())
            cx0 = float((cond_min[0] + denom[0]).item())
            d0 = float(denom[0].item())
            print(f"  [Sampler] Per-sample cond_mag range (sample_0): min={cm0:.6g} max={cx0:.6g} denom={d0:.6g}")
        if use_phase_channel:
            # Only magnitude uses cond_min/denom. Phase stays angle/pi in [-1,1].
            normalized = torch.cat([normalized.to(torch.float32), phase_norm.to(torch.float32)], dim=1)
        return normalized

    def _rescale_norm_to_unit_range(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale an arbitrary tensor to [-1, 1] using per-sample min/max.

        This is applied to diffusion outputs before denormalization/ISTFT so the
        inversion uses the conditioning-derived magnitude scale consistently.
        """
        x = x.to(torch.float32)
        if x.ndim < 2:
            xmin = x.min()
            xmax = x.max()
        else:
            dims = tuple(range(1, x.ndim))
            xmin = x.amin(dim=dims, keepdim=True)
            xmax = x.amax(dim=dims, keepdim=True)
        denom = (xmax - xmin)
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
        x01 = (x - xmin) / denom
        y = x01 * 2.0 - 1.0
        return torch.clamp(y, -1.0, 1.0)
    
    def postprocess_generated(
        self,
        generated: torch.Tensor,
        condition_phase: Optional[torch.Tensor],
        condition_mag_norm: Optional[torch.Tensor],
        params: dict,
        *,
        cond_min: torch.Tensor | None = None,
        cond_denom: torch.Tensor | None = None,
        input_length: int | None = None,
        return_denormalized: bool = False,
    ) -> torch.Tensor:
        """Denormalize generated transform coefficients and invert to time domain.

        Denormalization is a simple scale: out * cond_denom + cond_min.
        No unit-range rescaling is applied — the model is trained on [0,1] data
        and its output is directly on the condition's magnitude scale.
        """
        from data.utils.transforms import get_transform

        pipeline_config = params['pipeline_config'].copy()
        transform_type = str(pipeline_config.get('transform', 'stft')).lower()
        magnitude_only = transform_type in {'stft', 'dft'}
        requires_phase = transform_type in {'stft', 'dft'}

        use_phase_channel = bool(self.config.get('use_phase_channel', False))

        # Inference may run in bf16/fp16. Convert to float32 for stable inversion.
        generated = generated.to(torch.float32)

        # Split channels when phase is present.
        if use_phase_channel:
            if generated.ndim != 4 or generated.shape[1] < 2:
                raise ValueError(f"use_phase_channel=True expects generated [B,2,H,W], got {tuple(generated.shape)}")
            generated_mag = generated[:, :1]
            generated_phase_norm = torch.clamp(generated[:, 1:2], -1.0, 1.0)
        else:
            generated_mag = generated
            generated_phase_norm = None
        
        _norm_mode = str(getattr(self, '_last_cond_norm_mode', self.config.get('mag_norm_mode', 'minmax')) or 'minmax').strip().lower()
        if (cond_min is not None) and (cond_denom is not None):
            cond_min = cond_min.to(torch.float32)
            cond_denom = cond_denom.to(torch.float32)
            if _norm_mode in {'zscore', 'z_score', 'standardize'}:
                # zscore denorm: x * (std * clamp_sigma) + mean
                denormalized = generated_mag * cond_denom + cond_min
            elif _norm_mode in {'none', 'off', 'identity'}:
                # identity: denom=1, min=0 → x * 1 + 0 = x
                denormalized = generated_mag
            else:
                # minmax / absmax denorm: (x + 1) / 2 * range + min
                denormalized = (generated_mag + 1.0) / 2.0
                denormalized = denormalized * cond_denom + cond_min
            if magnitude_only:
                # Magnitudes must be non-negative.
                denormalized = torch.clamp(denormalized, min=0.0)
        else:
            rmin = float(params['cond_mag_min'])
            rmax = float(params['cond_mag_max'])
            denormalized = (generated_mag + 1.0) / 2.0
            denormalized = denormalized * (float(rmax) - float(rmin)) + float(rmin)
            if magnitude_only:
                denormalized = torch.clamp(denormalized, min=0.0, max=max(float(rmax), 0.0))

        
        if denormalized.ndim == 4 and denormalized.shape[1] == 1:
            denormalized = denormalized.squeeze(1)

        # Optional: magnitude energy matching (condition-only, no target lookahead).
        # This applies a single scalar so generated mag energy matches condition mag energy.
        self._last_mag_energy_match_factor = None
        em_cfg = self.config.get('mag_energy_match') if isinstance(self.config.get('mag_energy_match'), dict) else {}
        em_enabled = bool(em_cfg.get('enabled', False))
        if em_enabled and (cond_min is not None) and (cond_denom is not None) and torch.is_tensor(condition_mag_norm):
            try:
                method = str(em_cfg.get('method', 'l2') or 'l2').strip().lower()
                clamp_min = float(em_cfg.get('clamp_min', 0.25) or 0.25)
                clamp_max = float(em_cfg.get('clamp_max', 4.0) or 4.0)
                eps = float(em_cfg.get('eps', 1e-8) or 1e-8)

                cmag = condition_mag_norm.detach().to(torch.float32)
                while cmag.ndim > 4:
                    cmag = cmag.squeeze(1)
                if cmag.ndim == 4 and cmag.shape[1] >= 2:
                    cmag = cmag[:, :1]
                if cmag.ndim == 4 and cmag.shape[1] == 1:
                    cmag = cmag.squeeze(1)  # [B,F,T]
                if cmag.ndim == 3 and cmag.shape[0] == 1:
                    cmag = cmag[0]

                # Denormalize condition magnitude using the same cond_min/cond_denom.
                cmin = cond_min.detach().to(torch.float32)
                cden = cond_denom.detach().to(torch.float32)
                while cmin.ndim > 4:
                    cmin = cmin.squeeze(1)
                while cden.ndim > 4:
                    cden = cden.squeeze(1)
                if cmin.ndim == 4 and cmin.shape[1] == 1:
                    cmin = cmin.squeeze(1)
                if cden.ndim == 4 and cden.shape[1] == 1:
                    cden = cden.squeeze(1)
                if cmin.ndim == 3 and cmin.shape[0] == 1:
                    cmin = cmin[0]
                if cden.ndim == 3 and cden.shape[0] == 1:
                    cden = cden[0]

                if _norm_mode in {'zscore', 'z_score', 'standardize'}:
                    cond_mag_denorm = cmag * cden + cmin
                elif _norm_mode in {'none', 'off', 'identity'}:
                    cond_mag_denorm = cmag  # identity: raw mag, no denorm needed
                else:
                    cond_mag_denorm = (cmag + 1.0) / 2.0
                    cond_mag_denorm = cond_mag_denorm * cden + cmin
                cond_mag_denorm = torch.clamp(cond_mag_denorm, min=0.0)

                gen_mag = denormalized.detach().to(torch.float32)
                # Align shapes
                f = min(int(cond_mag_denorm.shape[-2]), int(gen_mag.shape[-2]))
                t = min(int(cond_mag_denorm.shape[-1]), int(gen_mag.shape[-1]))
                cond_mag_denorm = cond_mag_denorm[..., :f, :t]
                gen_mag = gen_mag[..., :f, :t]

                if method in {'p95', 'quantile95'}:
                    cstat = torch.quantile(cond_mag_denorm.reshape(-1), 0.95)
                    gstat = torch.quantile(gen_mag.reshape(-1), 0.95)
                    scale = float((cstat / (gstat + eps)).item())
                else:
                    # default: L2 energy match (Frobenius norm)
                    cE = torch.mean(cond_mag_denorm * cond_mag_denorm)
                    gE = torch.mean(gen_mag * gen_mag)
                    scale = float(torch.sqrt(cE / (gE + eps)).item())

                if not math.isfinite(scale):
                    scale = 1.0
                scale = max(clamp_min, min(clamp_max, scale))

                denormalized = denormalized * float(scale)
                if magnitude_only and (cond_min is not None) and (cond_denom is not None):
                    # Keep magnitudes non-negative; allow >cond_max if scale>1, but avoid NaNs.
                    denormalized = torch.clamp(denormalized, min=0.0)
                self._last_mag_energy_match_factor = float(scale)
            except Exception:
                self._last_mag_energy_match_factor = None

        predicted_phase = None
        if use_phase_channel:
            # phase in radians in [-pi, pi]
            predicted_phase = (generated_phase_norm.squeeze(1) * math.pi).to(torch.float32)
        
        # Inverse transform back to time domain
        transform_type = pipeline_config.pop('transform', 'stft')
        
        transform = get_transform(
            transform_type,
            **pipeline_config,
            device=self.device
        )

        # Some transforms (e.g. STFT/DFT) require phase/state to invert.
        if requires_phase:
            inv_src = str(self.config.get('phase_inversion_source', 'auto') or 'auto').strip().lower()
            if inv_src not in {'auto', 'predicted', 'condition'}:
                inv_src = 'auto'

            if inv_src == 'condition':
                if condition_phase is None:
                    raise ValueError(f"Transform '{transform_type}' requires condition_phase for inversion (phase_inversion_source='condition')")
                phase = condition_phase
            elif inv_src == 'predicted':
                if predicted_phase is None:
                    raise ValueError("phase_inversion_source='predicted' but no predicted phase is available (use_phase_channel must be enabled).")
                phase = predicted_phase
            else:
                # auto
                if predicted_phase is not None:
                    phase = predicted_phase
                else:
                    if condition_phase is None:
                        raise ValueError(f"Transform '{transform_type}' requires condition_phase for inversion")
                    phase = condition_phase
            if phase.ndim == 4 and phase.shape[1] == 1:
                phase = phase.squeeze(1)
            transform.phase = phase.to(self.device)

            if transform.phase.shape != denormalized.shape:
                print(f"  [Warning] Phase shape {transform.phase.shape} != denormalized {denormalized.shape}")
                if transform.phase.shape[-1] > denormalized.shape[-1]:
                    transform.phase = transform.phase[:, :, :denormalized.shape[-1]]
                elif transform.phase.shape[-1] < denormalized.shape[-1]:
                    pad_size = denormalized.shape[-1] - transform.phase.shape[-1]
                    transform.phase = torch.nn.functional.pad(
                        transform.phase, (0, pad_size), mode='constant', value=0
                    )
        
        # For center=True, the STFT frame count is not a direct (T, hop, n_fft)
        # inversion of the original time length due to padding.
        # Use the original time length that produced the conditioning phase.
        if input_length is not None:
            transform.input_length = int(input_length)
        else:
            input_len = getattr(self, '_last_input_length', None)
            if input_len is not None:
                transform.input_length = int(input_len)
        
        time_signals = transform.inverse(denormalized)

        time_out = time_signals.cpu()
        if bool(return_denormalized):
            if predicted_phase is not None:
                return time_out, denormalized.detach().cpu(), predicted_phase.detach().cpu()
            return time_out, denormalized.detach().cpu()
        return time_out

    def _sample_trajectories(
        self,
        *,
        condition_spec: torch.Tensor,
        trajectory_indices: list[int],
        num_inference_steps: int,
        use_ddim: bool,
        H: int,
        W: int,
        progress_dir: Optional[Path] = None,
        progress_every_n_steps: Optional[int] = None,
        progress_condition_phase: Optional[torch.Tensor] = None,
        progress_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run diffusion denoising to sample multiple trajectories for a single condition.

        Args:
            condition_spec: [1, C, H, W] normalized condition spectrogram.
            trajectory_indices: List of trajectory ids (used for deterministic seeds).
            num_inference_steps: Number of denoising steps.
            use_ddim: Whether to use DDIM when model scheduler is DDPM.
            H, W: Spatial shape.

        Returns:
            Tensor of shape [len(trajectory_indices), C, H, W]
        """
        if not trajectory_indices:
            raise ValueError("trajectory_indices must be non-empty")

        condition_spec = condition_spec.to(self.device)
        condition_batch = condition_spec.repeat(len(trajectory_indices), 1, 1, 1)

        # Match model dtype (training often uses bf16).
        model_dtype = next(self.model.parameters()).dtype
        condition_batch = condition_batch.to(dtype=model_dtype)

        if progress_dir is not None and progress_every_n_steps is not None and int(progress_every_n_steps) > 0:
            progress_dir.mkdir(parents=True, exist_ok=True)

        def _should_snapshot(step_idx: int, total_steps: int) -> bool:
            n = int(progress_every_n_steps or 0)
            if n <= 0:
                return False
            return (step_idx % n == 0) or (step_idx == total_steps - 1)

        def _save_snapshot(step_idx: int, timestep, image_batch: torch.Tensor) -> None:
            if progress_dir is None:
                return
            if progress_params is None:
                return

            # Save spectrograms as a single tensor: [B, C, H, W]
            specs_norm = image_batch.detach().to('cpu').to(torch.float32)

            use_phase_channel = bool(self.config.get('use_phase_channel', False)) and specs_norm.ndim == 4 and specs_norm.shape[1] >= 2

            # Denormalize using per-sample normalization when available.
            snap_pipeline_config = (progress_params.get('pipeline_config') or {}).copy()
            snap_transform_type = snap_pipeline_config.get('transform', 'stft')
            snap_magnitude_only = str(snap_transform_type).lower() in {'stft', 'dft'}

            if use_phase_channel:
                specs_mag = specs_norm[:, :1]
                specs_phase_norm = torch.clamp(specs_norm[:, 1:2], -1.0, 1.0)

                if ('cond_min' in progress_params) and ('cond_denom' in progress_params):
                    cmin = progress_params['cond_min'].to(torch.float32)
                    cden = progress_params['cond_denom'].to(torch.float32)
                    _snap_mode = str(self.config.get('mag_norm_mode', 'minmax') or 'minmax').strip().lower()
                    if _snap_mode in {'zscore', 'z_score', 'standardize'}:
                        specs_denorm_mag = specs_mag * cden + cmin
                    else:
                        specs_denorm_mag = (specs_mag + 1.0) / 2.0
                        specs_denorm_mag = specs_denorm_mag * cden + cmin
                    if snap_magnitude_only:
                        specs_denorm_mag = torch.clamp(specs_denorm_mag, min=0.0)
                else:
                    rmin = float(progress_params.get('cond_mag_min', 0.0))
                    rmax = float(progress_params.get('cond_mag_max', 1.0))
                    specs_denorm_mag = (specs_mag + 1.0) / 2.0
                    specs_denorm_mag = specs_denorm_mag * (float(rmax) - float(rmin)) + float(rmin)
                    if snap_magnitude_only:
                        specs_denorm_mag = torch.clamp(specs_denorm_mag, min=0.0)

                specs_denorm_phase = specs_phase_norm * math.pi
            else:
                if ('cond_min' in progress_params) and ('cond_denom' in progress_params):
                    cmin = progress_params['cond_min'].to(torch.float32)
                    cden = progress_params['cond_denom'].to(torch.float32)
                    _snap_mode = str(self.config.get('mag_norm_mode', 'minmax') or 'minmax').strip().lower()
                    if _snap_mode in {'zscore', 'z_score', 'standardize'}:
                        specs_denorm = specs_norm * cden + cmin
                    else:
                        specs_denorm = (specs_norm + 1.0) / 2.0
                        specs_denorm = specs_denorm * cden + cmin
                    if snap_magnitude_only:
                        specs_denorm = torch.clamp(specs_denorm, min=0.0)
                else:
                    rmin = float(progress_params.get('cond_mag_min', 0.0))
                    rmax = float(progress_params.get('cond_mag_max', 1.0))
                    specs_denorm = (specs_norm + 1.0) / 2.0
                    specs_denorm = specs_denorm * (float(rmax) - float(rmin)) + float(rmin)
                    if snap_magnitude_only:
                        specs_denorm = torch.clamp(specs_denorm, min=0.0)

            # Also save phase-based inversion for each trajectory.
            # (Loop is fine; this is debug/analysis output.)
            times = []
            for b in range(specs_norm.shape[0]):
                gen_spec = specs_norm[b:b+1].to(self.device)
                gen_time = self.postprocess_generated(
                    gen_spec,
                    condition_phase=progress_condition_phase,
                    condition_mag_norm=None,
                    params=progress_params,
                    cond_min=progress_params.get('cond_min'),
                    cond_denom=progress_params.get('cond_denom'),
                    input_length=progress_params.get('input_length'),
                )
                times.append(gen_time)
            time_batch = torch.cat(times, dim=0)  # [B, L]

            t_val = int(timestep) if not isinstance(timestep, torch.Tensor) else int(timestep.item())
            out_path = progress_dir / f"progress_step_{step_idx:04d}_t{t_val}.pt"
            torch.save(
                {
                    'step_idx': int(step_idx),
                    'timestep': int(t_val),
                    'trajectory_indices': [int(i) for i in trajectory_indices],
                    # Backwards-compatible key (historically magnitude-only).
                    'spectrograms_16bit': (specs_denorm_mag.to(torch.float32) if use_phase_channel else specs_denorm.to(torch.float32)),
                    'spectrograms_16bit_norm': specs_norm.to(torch.float32),
                    'spectrograms_denorm_mag': (specs_denorm_mag.to(torch.float32) if use_phase_channel else None),
                    'spectrograms_denorm_phase': (specs_denorm_phase.to(torch.float32) if use_phase_channel else None),
                    'time_domain': time_batch.to(torch.float32),
                },
                out_path,
            )

        with torch.no_grad():
            all_noise = []
            C = int(condition_spec.shape[1])
            for traj_idx in trajectory_indices:
                generator = torch.Generator(device=self.device).manual_seed(int(traj_idx))
                noise = torch.randn(
                    (1, C, H, W),
                    device=self.device,
                    generator=generator,
                    dtype=model_dtype,
                )
                all_noise.append(noise)
            image = torch.cat(all_noise, dim=0)

            from diffusers import DDIMScheduler

            if use_ddim and getattr(self.model, 'scheduler_type', 'ddpm') == 'ddpm':
                scheduler = DDIMScheduler.from_config(self.model.noise_scheduler.config)
            else:
                scheduler = self.model.noise_scheduler

            scheduler.set_timesteps(int(num_inference_steps), device=self.device)
            cond_embed = self.model.encode_condition(condition_batch)

            timesteps = list(scheduler.timesteps)
            total_steps = len(timesteps)
            for step_idx, t in enumerate(tqdm(timesteps, desc=f"Denoising {len(trajectory_indices)} trajectories", leave=False)):
                noise_pred = self.model.unet(
                    image,
                    t,
                    encoder_hidden_states=cond_embed,
                    return_dict=False,
                )[0]
                image = scheduler.step(noise_pred, t, image).prev_sample

                if _should_snapshot(step_idx, total_steps):
                    _save_snapshot(step_idx, t, image)

        return image
    
    def sample(
        self,
        raw_condition_signals: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        use_ddim: Optional[bool] = None,
        num_trajectories: int = 1,
        batch_size: int = 16,
        save_progress_every_n_steps: Optional[int] = None,
        debug_quant: bool = False,
        preprocess_only: bool = False,
    ) -> dict:
        """Generate samples from raw condition signals, processing multiple samples in parallel batches."""
        if use_ddim is None:
            sched = str(self.config.get('scheduler_type', 'ddpm')).lower()
            use_ddim = (sched == 'ddim')

        if num_inference_steps is None:
            if bool(use_ddim):
                num_inference_steps = int(self.config.get('num_inference_steps', 50))
            else:
                num_inference_steps = int(self.config.get('num_noising_steps', 1000))

        num_conditions = len(raw_condition_signals)
        sampler_type = 'ddim' if bool(use_ddim) else 'ddpm'
        samples_root = self.version_dir / 'samples' / sampler_type
        samples_root.mkdir(parents=True, exist_ok=True)

        results = {
            'conditions': [],
            'generated_specs': [],
            'generated_time': [],
            'sampler_type': sampler_type,
        }

        # Global params (pipeline config, transform shapes) computed once.
        params = self._compute_sampling_params(raw_condition_signals, debug_quant=bool(debug_quant))

        print(f"[Sampler] Time-domain quantization mode: per_sample (real-time)")

        # Batch processing configuration
        BATCH_SIZE = max(1, int(batch_size))
        
        print(f"[Sampler] Starting generation for {num_conditions} conditions with batch size {BATCH_SIZE}...")
        
        for batch_start in range(0, num_conditions, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_conditions)
            current_batch_size = batch_end - batch_start
            
            batch_raw = raw_condition_signals[batch_start:batch_end]
            
            # 1. Preprocess Batch
            batch_cond_specs = []
            batch_phases = []
            batch_cond_mins = []
            batch_cond_denoms = []
            batch_input_lengths = []
            batch_cond_waves_4bit = []
            
            print(f"  Batch {batch_start//BATCH_SIZE + 1}: Preprocessing {current_batch_size} signals...")
            
            for i in range(current_batch_size):
                raw_sig = batch_raw[i:i+1] # Keep dim [1, L]
                idx = batch_start + i

                params_i = params
                
                # Setup output dir and save raw inputs
                sample_dir = samples_root / f'sample_{idx}'
                sample_dir.mkdir(parents=True, exist_ok=True)
                torch.save({'raw_signal': raw_sig.cpu()}, sample_dir / 'condition_raw.pt')

                # Get Condition Spectrogram
                cond_spec = self.preprocess_condition(raw_sig, params=params_i, debug_quant=bool(debug_quant and idx == 0))
                cond_spec = cond_spec.to(self.device)
                torch.save({'condition_spec_4bit': cond_spec.cpu()}, sample_dir / 'condition_spec.pt')

                # Save explicit time baselines to make downstream comparisons unambiguous.
                # - target_time: the (optionally pre-lowpassed) raw waveform
                # - cond_time_4bit: the quantized condition waveform used to form cond_spec
                try:
                    target_time = self._maybe_pre_lowpass(raw_sig).detach().cpu().to(torch.float32)
                except Exception:
                    target_time = raw_sig.detach().cpu().to(torch.float32)
                torch.save({'target_time': target_time}, sample_dir / 'target_time.pt')

                # Condition 4-bit waveform (used for time-domain scaling without lookahead).
                wave_4bit = getattr(self, '_last_wave_4bit', None)
                if wave_4bit is None:
                    wave_4bit = raw_sig.to(self.device)
                batch_cond_waves_4bit.append(wave_4bit.detach().cpu().to(torch.float32))
                torch.save({'cond_time_4bit': wave_4bit.detach().cpu().to(torch.float32)}, sample_dir / 'cond_time_4bit.pt')
                
                batch_cond_specs.append(cond_spec)
                # self._last_phase is set by preprocess_condition side-effect
                if self._last_phase is None:
                    batch_phases.append(None)
                else:
                    batch_phases.append(self._last_phase.clone())
                batch_cond_mins.append(getattr(self, '_last_cond_min').clone())
                batch_cond_denoms.append(getattr(self, '_last_cond_denom').clone())
                batch_input_lengths.append(int(getattr(self, '_last_input_length')))

                # Save target spectrogram in training-normalized space (and phase if available).
                try:
                    tgt_mag_norm, tgt_phase_norm = self._compute_target_spec_norm(
                        raw_sig,
                        params=params_i,
                        cond_min=batch_cond_mins[-1],
                        cond_denom=batch_cond_denoms[-1],
                    )
                    payload = {'target_spec_norm': tgt_mag_norm.detach().cpu().to(torch.float32)}
                    if tgt_phase_norm is not None:
                        payload['target_phase_norm'] = tgt_phase_norm.detach().cpu().to(torch.float32)
                    torch.save(payload, sample_dir / 'target_spec.pt')
                except Exception as e:
                    print(f"[Sampler] Warning: failed saving target_spec for sample_{idx} (non-fatal): {e}")

            # Stack conditions: [Batch, 1, H, W]
            cond_batch_stacked = torch.cat(batch_cond_specs, dim=0)

            if preprocess_only:
                print("[Sampler] preprocess_only=True; skipping diffusion and returning condition specs")
                results['conditions'].extend(batch_cond_specs)
                continue

            # Match model dtype (training often uses bf16) for all network inputs.
            model_dtype = next(self.model.parameters()).dtype
            
            # 2. Run Diffusion (Stacked)
            # We treat (Batch * Trajectories) as one giant batch for the model.
            # cond_batch_stacked: [B, 1, H, W] -> repeat to [B * T, 1, H, W]
            
            # E.g. B=2, T=3. Indices: [0, 0, 0, 1, 1, 1]
            inference_batch = cond_batch_stacked.repeat_interleave(num_trajectories, dim=0)
            inference_batch = inference_batch.to(dtype=model_dtype)
            
            # Map inference indices back to sample/trajectory indices
            # traj_map[k] = (sample_idx, traj_idx) is implicit
            trajectory_indices = list(range(current_batch_size * num_trajectories))
            
            print(f"  Batch {batch_start//BATCH_SIZE + 1}: Running diffusion on {len(inference_batch)} items (Batch={current_batch_size} * Traj={num_trajectories})...")
            
            # We reuse _sample_trajectories but pass the bloated batch directly.
            # It expects 'condition_spec' to be [1, 1, H, W] and does repeat internally.
            # We need to bypass that or adapt it.
            # Let's adapt the core logic inline here to support batched unique conditions.
            
            H, W = self.config.get('image_size', (16, 128))
            
            # Generate Noise
            # We need deterministic seeds per trajectory.
            all_noise = []
            C = int(inference_batch.shape[1])
            for k in range(len(inference_batch)):
                # seed = (global_sample_idx * 1000) + traj_idx
                sample_local_idx = k // num_trajectories
                traj_idx = k % num_trajectories
                global_sample_idx = batch_start + sample_local_idx
                seed = (global_sample_idx * 1000) + traj_idx
                
                gen = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn((1, C, H, W), device=self.device, generator=gen, dtype=model_dtype)
                all_noise.append(noise)
            
            image = torch.cat(all_noise, dim=0)
            
            # Scheduler Setup
            from diffusers import DDIMScheduler
            if use_ddim and getattr(self.model, 'scheduler_type', 'ddpm') == 'ddpm':
                scheduler = DDIMScheduler.from_config(self.model.noise_scheduler.config)
            else:
                scheduler = self.model.noise_scheduler
            scheduler.set_timesteps(int(num_inference_steps), device=self.device)
            
            # Encode conditions
            cond_embed = self.model.encode_condition(inference_batch)
            
            # Denoise Loop
            with torch.no_grad():
                for t in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
                    noise_pred = self.model.unet(
                        image, t, encoder_hidden_states=cond_embed, return_dict=False
                    )[0]
                    image = scheduler.step(noise_pred, t, image).prev_sample

            # 3. Postprocess & Save Batch Results
            generated_batch = image # [B*T, 1, H, W]
            
            print(f"  Batch {batch_start//BATCH_SIZE + 1}: Postprocessing & saving...")

            for i in range(current_batch_size):
                idx = batch_start + i
                phase = batch_phases[i]
                cond_min = batch_cond_mins[i]
                cond_denom = batch_cond_denoms[i]
                input_len = batch_input_lengths[i]
                sample_dir = samples_root / f'sample_{idx}'
                
                # Extract trajectories for this sample
                start_k = i * num_trajectories
                end_k = start_k + num_trajectories
                sample_gen_specs = generated_batch[start_k:end_k] # [T, 1, H, W]
                
                trajectories_spec = []
                trajectories_time = []
                trajectories_mag_denorm = []
                
                for t_idx in range(num_trajectories):
                    gen_spec = sample_gen_specs[t_idx:t_idx+1]

                    # Denormalize (multiply by condition scale) and invert to time domain.
                    out = self.postprocess_generated(
                        gen_spec,
                        condition_phase=phase,
                        condition_mag_norm=batch_cond_specs[i],
                        params=params,
                        cond_min=cond_min,
                        cond_denom=cond_denom,
                        input_length=input_len,
                        return_denormalized=True,
                    )

                    if isinstance(out, tuple) and len(out) == 3:
                        gen_time, gen_mag_denorm, gen_phase_pred = out
                    elif isinstance(out, tuple):
                        gen_time, gen_mag_denorm = out
                        gen_phase_pred = None
                    else:
                        gen_time = out
                        gen_mag_denorm = None
                        gen_phase_pred = None

                    trajectories_spec.append(gen_spec.detach().cpu().to(torch.float32))
                    trajectories_time.append(gen_time.cpu())
                    trajectories_mag_denorm.append(
                        gen_mag_denorm.detach().cpu().to(torch.float32)
                        if gen_mag_denorm is not None
                        else gen_spec.detach().cpu().to(torch.float32)
                    )
                    
                    torch.save({
                        'spectrogram_norm': gen_spec.detach().cpu().to(torch.float32),
                        'spectrogram_denorm_mag': gen_mag_denorm.detach().cpu().to(torch.float32) if gen_mag_denorm is not None else None,
                        'spectrogram_denorm_phase': (gen_phase_pred.detach().cpu().to(torch.float32) if gen_phase_pred is not None else None),
                        'time_domain': gen_time.cpu(),
                        'mag_energy_match_factor': getattr(self, '_last_mag_energy_match_factor', None),
                    }, sample_dir / f'trajectory_{t_idx}.pt')

                    # Save a tiny per-trajectory metrics JSON (pre/post inversion MSEs).
                    # This intentionally uses ONLY condition-derived normalization for the spec MSEs.
                    try:
                        import json

                        tgt_time = torch.load(sample_dir / 'target_time.pt', map_location='cpu')['target_time']
                        cond_time = torch.load(sample_dir / 'cond_time_4bit.pt', map_location='cpu')['cond_time_4bit']
                        tgt_spec_obj = torch.load(sample_dir / 'target_spec.pt', map_location='cpu')
                        tgt_spec_norm = tgt_spec_obj.get('target_spec_norm')
                        cond_spec_saved = torch.load(sample_dir / 'condition_spec.pt', map_location='cpu')['condition_spec_4bit']

                        def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
                            a = a.detach().to(torch.float32)
                            b = b.detach().to(torch.float32)
                            # Align by min length/shape where applicable.
                            if a.ndim == 2 and b.ndim == 2:
                                n = min(a.shape[-1], b.shape[-1])
                                return float(torch.mean((a[..., :n] - b[..., :n]) ** 2).item())
                            if a.shape != b.shape:
                                # Fallback: crop to min over each dim.
                                mins = [min(int(sa), int(sb)) for sa, sb in zip(a.shape, b.shape)]
                                slicer = tuple(slice(0, m) for m in mins)
                                a = a[slicer]
                                b = b[slicer]
                            return float(torch.mean((a - b) ** 2).item())

                        metrics = {
                            'mse_time(cond4_vs_target)': _mse(cond_time, tgt_time),
                            'mse_time(gen_vs_target)': _mse(gen_time.detach().cpu(), tgt_time),
                        }

                        if torch.is_tensor(tgt_spec_norm) and torch.is_tensor(cond_spec_saved):
                            # Compare only magnitude channel if phase channel exists.
                            if cond_spec_saved.ndim == 4 and cond_spec_saved.shape[1] >= 2:
                                metrics['mse_spec_mag(cond4_vs_target)'] = _mse(cond_spec_saved[:, :1], tgt_spec_norm[:, :1])
                            else:
                                metrics['mse_spec_mag(cond4_vs_target)'] = _mse(cond_spec_saved, tgt_spec_norm)

                        if torch.is_tensor(tgt_spec_norm):
                            metrics['mse_spec_mag(gen_norm_vs_target)'] = _mse(gen_spec.detach().cpu().to(torch.float32), tgt_spec_norm)

                        denom_pre = metrics.get('mse_spec_mag(cond4_vs_target)')
                        if denom_pre is not None and denom_pre != 0 and 'mse_spec_mag(gen_norm_vs_target)' in metrics:
                            metrics['improvement_pct_spec'] = float((denom_pre - metrics['mse_spec_mag(gen_norm_vs_target)']) / denom_pre * 100.0)

                        denom_post = metrics.get('mse_time(cond4_vs_target)')
                        if denom_post is not None and denom_post != 0:
                            metrics['improvement_pct_time'] = float((denom_post - metrics['mse_time(gen_vs_target)']) / denom_post * 100.0)

                        with open(sample_dir / f'metrics_traj_{t_idx}.json', 'w') as f:
                            json.dump(metrics, f, indent=2, sort_keys=True)
                    except Exception as e:
                        print(f"[Sampler] Warning: failed writing metrics JSON for sample_{idx} traj_{t_idx} (non-fatal): {e}")

                # Save all
                torch.save({
                    'spectrograms_norm': torch.cat(trajectories_spec, dim=0).to(torch.float32),
                    'spectrograms_denorm_mag': torch.cat(trajectories_mag_denorm, dim=0).to(torch.float32),
                    'time_domain': torch.cat(trajectories_time, dim=0),
                }, sample_dir / 'all_trajectories.pt')
                
                results['conditions'].append(cond_batch_stacked[i].cpu())
                results['generated_specs'].append(torch.cat(trajectories_spec, dim=0))
                results['generated_time'].append(torch.cat(trajectories_time, dim=0))
                
        return results
    



if __name__ == "__main__":
    import argparse
    import shutil

    def _pick_device(requested: str) -> str:
        req = (requested or "auto").strip().lower()
        if req != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    p = argparse.ArgumentParser(description="Sample from a trained diffusion model checkpoint")
    p.add_argument("--results-root", type=str, default="diffusion/results", help="Root results dir (default: diffusion/results)")
    p.add_argument("--version", type=str, default="V1", help="Experiment version folder (e.g., V3)")
    p.add_argument("--run-id", type=str, required=True, help="Run id folder under diffusion/results/<version>/")
    p.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename under checkpoints/")
    p.add_argument("--out-run-id", type=str, default=None, help="Optional output run id (defaults to --run-id; use to avoid overwriting samples)")

    p.add_argument("--num-samples", type=int, default=1, help="Number of distinct conditions to sample")
    p.add_argument("--num-trajectories", type=int, default=1, help="Number of trajectories per condition")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for sampling")
    p.add_argument("--use-ddim", action="store_true", help="Use DDIM sampler (default: DDPM)")
    p.add_argument("--num-inference-steps", type=int, default=None, help="Override inference steps (default: from config)")
    p.add_argument("--save-progress-every", type=int, default=0, help="If >0, save intermediate progress every N steps")
    p.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|mps|cuda")
    p.add_argument("--raw-data-path", type=str, default=None, help="Override raw data path (time-domain chunks .pt)")
    args = p.parse_args()

    results_root = Path(args.results_root)
    version = str(args.version)
    run_id = str(args.run_id)
    checkpoint_name = str(args.checkpoint)

    src_run_dir = results_root / version / run_id
    src_config = src_run_dir / "config.pkl"
    src_ckpt = src_run_dir / "checkpoints" / checkpoint_name

    if not src_config.exists():
        raise SystemExit(f"[Sampler] Missing config: {src_config}")
    if not src_ckpt.exists():
        raise SystemExit(f"[Sampler] Missing checkpoint: {src_ckpt}")

    out_run_id = str(args.out_run_id) if args.out_run_id else run_id
    out_run_dir = results_root / version / out_run_id
    (out_run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Ensure output folder has the expected layout for DiffusionSampler.
    if (out_run_dir / "config.pkl") != src_config:
        shutil.copy2(src_config, out_run_dir / "config.pkl")
    if (out_run_dir / "checkpoints" / checkpoint_name) != src_ckpt:
        shutil.copy2(src_ckpt, out_run_dir / "checkpoints" / checkpoint_name)

    with open(out_run_dir / "config.pkl", "rb") as f:
        config = pickle.load(f)

    device = _pick_device(str(args.device))
    print(f"[Sampler] Loading run: {version}/{run_id}")
    print(f"[Sampler] Writing samples to: {version}/{out_run_id}")
    print(f"[Sampler] Checkpoint: {checkpoint_name}")
    print(f"[Sampler] Device: {device}")

    sampler = DiffusionSampler(
        results_dir=results_root / version,
        version=out_run_id,
        checkpoint_name=checkpoint_name,
        device=device,
    )

    raw_data_path = None
    if args.raw_data_path:
        raw_data_path = Path(str(args.raw_data_path)).expanduser()
    else:
        cfg_path = str(config.get('raw_data_path') or '').strip()
        raw_data_path = Path(cfg_path).expanduser() if cfg_path else None

    if not raw_data_path or not raw_data_path.exists():
        raw_data_path = Path("data/data/raw/art_chunks.pt")

    print(f"[Sampler] Loading condition data from: {raw_data_path}")
    data = torch.load(raw_data_path)
    raw_signals = data.get('signals') or data.get('chunks')
    if raw_signals is None:
        raise SystemExit(f"[Sampler] Could not find 'signals' or 'chunks' in: {raw_data_path}")

    # Default: sample from held-out test tail (not seen during training).
    holdout_n = int(config.get('test_holdout_count', 0) or 0)
    holdout_from_end = bool(config.get('test_holdout_from_end', True))
    if holdout_n > 0 and raw_signals.shape[0] > holdout_n:
        raw_signals = raw_signals[-holdout_n:] if holdout_from_end else raw_signals[:holdout_n]
        where = "last" if holdout_from_end else "first"
        print(f"[Sampler] Using held-out test split: {where} {holdout_n} samples")
    else:
        print("[Sampler] Holdout not configured (or dataset too small); sampling from full dataset")

    num_samples = int(args.num_samples)
    num_samples = min(num_samples, int(raw_signals.shape[0]))
    selected = raw_signals[-num_samples:]
    num_trajectories = int(args.num_trajectories)
    print(f"[Sampler] Generating from {num_samples} conditions, {num_trajectories} trajectory each...")

    num_inference_steps = args.num_inference_steps
    save_progress_every = int(args.save_progress_every or 0)

    sampler.sample(
        raw_condition_signals=selected,
        num_inference_steps=num_inference_steps,
        use_ddim=bool(args.use_ddim),
        num_trajectories=num_trajectories,
        batch_size=int(args.batch_size),
        save_progress_every_n_steps=save_progress_every if save_progress_every > 0 else None,
    )

    sampler_subdir = 'ddim' if bool(args.use_ddim) else 'ddpm'
    print(f"[Sampler] Done! Results saved to: {sampler.version_dir / 'samples' / sampler_subdir}")
