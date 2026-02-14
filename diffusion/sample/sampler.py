"""Sample a trained diffusion model and reconstruct outputs."""

import torch
import pickle
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
        from data.preprocess.quantize import compute_range_from_tensor
        from data.preprocess.transform import get_transform
        from data.preprocess.quantize import UniformQuantizer

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
    
    def preprocess_condition(self, raw_signals: torch.Tensor, params: dict, *, debug_quant: bool = False) -> torch.Tensor:
        """Preprocess condition: time-quantize -> STFT magnitude -> normalize to [-1, 1].

        IMPORTANT: This must match training normalization.
        Training uses per-sample min/max of the *condition spectrogram magnitude*
        to normalize BOTH condition and real magnitudes.
        """
        from data.preprocess.transform import get_transform
        from data.preprocess.quantize import UniformQuantizer

        cond_bits = int(params['cond_bits']) #use the conditional data bit sizew

        # Per-sample time-domain quantization range from this condition only.
        from data.preprocess.quantize import compute_range_from_tensor
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

        self._last_phase = transform.phase.detach().clone()
        self._last_input_length = int(wave_4bit.shape[-1])

        if mag.ndim == 3:
            mag = mag.unsqueeze(1)  # [B, F, T] -> [B, 1, F, T]

        # Per-sample normalization (matches diffusion/utils/runner.py)
        cond_min = mag.amin(dim=(2, 3), keepdim=True)
        cond_max = mag.amax(dim=(2, 3), keepdim=True)
        denom = cond_max - cond_min
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)

        normalized = (mag - cond_min) / denom
        normalized = normalized * 2.0 - 1.0
        normalized = torch.clamp(normalized, -1.0, 1.0)

        # Save for postprocess inversion.
        self._last_cond_min = cond_min.detach().clone()
        self._last_cond_denom = denom.detach().clone()

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
        condition_phase: torch.Tensor,
        params: dict,
        *,
        cond_min: torch.Tensor | None = None,
        cond_denom: torch.Tensor | None = None,
        input_length: int | None = None,
        return_denormalized: bool = False,
        rescale_to_unit_range: bool = True,
    ) -> torch.Tensor:
        """Denormalize generated magnitude and invert transform using condition phase.

        If cond_min/cond_denom are provided, uses per-sample inversion matching training.
        Otherwise falls back to global range inversion (legacy behavior).
        """
        from data.preprocess.transform import get_transform

        # Inference may run in bf16/fp16. Convert to float32 for stable inversion.
        generated = generated.to(torch.float32)

        # Default behavior: rescale diffusion output to fill [-1, 1] per-sample.
        # This preserves relative structure while enforcing the expected normalized range
        # for denormalization using per-sample conditioning stats.
        if bool(rescale_to_unit_range):
            generated = self._rescale_norm_to_unit_range(generated)
        
        if (cond_min is not None) and (cond_denom is not None):
            cond_min = cond_min.to(torch.float32)
            cond_denom = cond_denom.to(torch.float32)
            denormalized = (generated + 1.0) / 2.0
            denormalized = denormalized * cond_denom + cond_min
            # Magnitudes must be non-negative; also cap to the conditioning max.
            cond_max = cond_min + cond_denom
            denormalized = torch.clamp(denormalized, min=0.0)
            denormalized = torch.minimum(denormalized, cond_max)
        else:
            rmin = float(params['cond_mag_min'])
            rmax = float(params['cond_mag_max'])
            denormalized = (generated + 1.0) / 2.0
            denormalized = denormalized * (float(rmax) - float(rmin)) + float(rmin)
            denormalized = torch.clamp(denormalized, min=0.0, max=max(float(rmax), 0.0))

        
        if denormalized.ndim == 4 and denormalized.shape[1] == 1:
            denormalized = denormalized.squeeze(1)
        
        # Inverse STFT to time domain
        pipeline_config = params['pipeline_config'].copy()
        transform_type = pipeline_config.pop('transform', 'stft')
        
        transform = get_transform(
            transform_type,
            **pipeline_config,
            device=self.device
        )
        
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
            condition_spec: [1, 1, H, W] normalized condition spectrogram.
            trajectory_indices: List of trajectory ids (used for deterministic seeds).
            num_inference_steps: Number of denoising steps.
            use_ddim: Whether to use DDIM when model scheduler is DDPM.
            H, W: Spatial shape.

        Returns:
            Tensor of shape [len(trajectory_indices), 1, H, W]
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
            if progress_condition_phase is None or progress_params is None:
                return

            # Save spectrograms as a single tensor: [B, 1, H, W]
            specs_norm = image_batch.detach().to('cpu').to(torch.float32)

            # Denormalize using per-sample normalization when available.
            if ('cond_min' in progress_params) and ('cond_denom' in progress_params):
                cmin = progress_params['cond_min'].to(torch.float32)
                cden = progress_params['cond_denom'].to(torch.float32)
                specs_denorm = (specs_norm + 1.0) / 2.0
                specs_denorm = specs_denorm * cden + cmin
                specs_denorm = torch.clamp(specs_denorm, min=0.0)
            else:
                rmin = float(progress_params.get('cond_mag_min', 0.0))
                rmax = float(progress_params.get('cond_mag_max', 1.0))
                specs_denorm = (specs_norm + 1.0) / 2.0
                specs_denorm = specs_denorm * (float(rmax) - float(rmin)) + float(rmin)
                specs_denorm = torch.clamp(specs_denorm, min=0.0)

            # Also save phase-based inversion for each trajectory.
            # (Loop is fine; this is debug/analysis output.)
            times = []
            for b in range(specs_norm.shape[0]):
                gen_spec = specs_norm[b:b+1].to(self.device)
                gen_time = self.postprocess_generated(
                    gen_spec,
                    condition_phase=progress_condition_phase,
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
                    'spectrograms_16bit': specs_denorm.to(torch.float32),
                    'spectrograms_16bit_norm': specs_norm.to(torch.float32),
                    'time_domain': time_batch.to(torch.float32),
                },
                out_path,
            )

        with torch.no_grad():
            all_noise = []
            for traj_idx in trajectory_indices:
                generator = torch.Generator(device=self.device).manual_seed(int(traj_idx))
                noise = torch.randn(
                    (1, 1, H, W),
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

                # Condition 4-bit waveform (used for time-domain scaling without lookahead).
                wave_4bit = getattr(self, '_last_wave_4bit', None)
                if wave_4bit is None:
                    wave_4bit = raw_sig.to(self.device)
                batch_cond_waves_4bit.append(wave_4bit.detach().cpu().to(torch.float32))
                
                batch_cond_specs.append(cond_spec)
                # self._last_phase is set by preprocess_condition side-effect
                batch_phases.append(self._last_phase.clone())
                batch_cond_mins.append(getattr(self, '_last_cond_min').clone())
                batch_cond_denoms.append(getattr(self, '_last_cond_denom').clone())
                batch_input_lengths.append(int(getattr(self, '_last_input_length')))

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
            for k in range(len(inference_batch)):
                # seed = (global_sample_idx * 1000) + traj_idx
                sample_local_idx = k // num_trajectories
                traj_idx = k % num_trajectories
                global_sample_idx = batch_start + sample_local_idx
                seed = (global_sample_idx * 1000) + traj_idx
                
                gen = torch.Generator(device=self.device).manual_seed(seed)
                noise = torch.randn((1, 1, H, W), device=self.device, generator=gen, dtype=model_dtype)
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
                trajectories_spec_raw = []
                trajectories_time = []
                trajectories_time_raw = []
                trajectories_time_from_rawspec = []
                trajectories_mag_denorm = []
                
                for t_idx in range(num_trajectories):
                    gen_spec_raw = sample_gen_specs[t_idx:t_idx+1]
                    gen_spec = self._rescale_norm_to_unit_range(gen_spec_raw)

                    # Invert transform WITHOUT unit-range rescale of the diffusion output.
                    # (This shows what happens if we take the raw diffusion normalized output as-is.)
                    gen_time_from_rawspec, gen_mag_denorm_rawspec = self.postprocess_generated(
                        gen_spec_raw,
                        condition_phase=phase,
                        params=params,
                        cond_min=cond_min,
                        cond_denom=cond_denom,
                        input_length=input_len,
                        return_denormalized=True,
                        rescale_to_unit_range=False,
                    )
                    
                    # Invert transform
                    gen_time_raw, gen_mag_denorm = self.postprocess_generated(
                        gen_spec,
                        condition_phase=phase,
                        params=params,
                        cond_min=cond_min,
                        cond_denom=cond_denom,
                        input_length=input_len,
                        return_denormalized=True,
                        rescale_to_unit_range=False,
                    )

                    # Post-ISTFT scaling (condition-only, no lookahead):
                    # scale generated waveform to match the CONDITION 4-bit waveform peak.
                    eps = 1e-8
                    target = batch_cond_waves_4bit[i].detach().cpu().to(torch.float32)
                    tgt_peak = float(target.abs().max().item())
                    gen_peak = float(gen_time_raw.abs().max().item())
                    peak_scale = (tgt_peak / (gen_peak + eps)) if (tgt_peak > 0 and gen_peak > 0) else 1.0
                    gen_time_scaled = gen_time_raw * float(peak_scale)
                    
                    trajectories_spec.append(gen_spec.cpu())
                    trajectories_spec_raw.append(gen_spec_raw.detach().cpu().to(torch.float32))
                    # Default saved output: scaled to match condition 4-bit peak.
                    trajectories_time.append(gen_time_scaled.cpu())
                    trajectories_time_raw.append(gen_time_raw.cpu())
                    trajectories_time_from_rawspec.append(gen_time_from_rawspec.cpu())
                    trajectories_mag_denorm.append(gen_mag_denorm.cpu())
                    
                    torch.save({
                        # Historical naming kept: this is the *normalized* spectrogram output.
                        'spectrogram_16bit': gen_spec.cpu(),
                        # Pre-inversion diffusion output after unit-range rescale to [-1, 1].
                        'spectrogram_norm': gen_spec.detach().cpu().to(torch.float32),
                        'spectrogram_norm_unitrange': gen_spec.detach().cpu().to(torch.float32),

                        # Pre-inversion raw diffusion output (no unit-range rescale).
                        'spectrogram_norm_raw': gen_spec_raw.detach().cpu().to(torch.float32),
                        'spectrogram_norm_prerescale': gen_spec_raw.detach().cpu().to(torch.float32),
                        'spectrogram_denorm_mag': gen_mag_denorm.detach().cpu().to(torch.float32),
                        'spectrogram_denorm_mag_rawspec': gen_mag_denorm_rawspec.detach().cpu().to(torch.float32),

                        # Time-domain reconstructions
                        'time_domain': gen_time_scaled.cpu(),
                        'time_domain_raw': gen_time_raw.cpu(),
                        'time_domain_from_rawspec': gen_time_from_rawspec.cpu(),
                        'time_rescale_peak_factor': float(peak_scale),
                    }, sample_dir / f'trajectory_{t_idx}.pt')

                # Save all
                torch.save({
                    'spectrograms_16bit': torch.cat(trajectories_spec, dim=0),
                    # Pre-inversion diffusion spectrograms
                    'spectrograms_norm': torch.cat(trajectories_spec, dim=0).to(torch.float32),
                    'spectrograms_norm_unitrange': torch.cat(trajectories_spec, dim=0).to(torch.float32),
                    'spectrograms_norm_raw': torch.cat(trajectories_spec_raw, dim=0),
                    'time_domain': torch.cat(trajectories_time, dim=0),
                    'time_domain_raw': torch.cat(trajectories_time_raw, dim=0),
                    'time_domain_from_rawspec': torch.cat(trajectories_time_from_rawspec, dim=0),
                    'spectrograms_denorm_mag': torch.cat(trajectories_mag_denorm, dim=0).to(torch.float32),
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
