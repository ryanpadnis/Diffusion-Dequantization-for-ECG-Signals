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

    def _compute_sampling_params(self, raw_signals: torch.Tensor) -> dict:
        """Compute preprocessing parameters from the provided testing set."""
        from data.preprocess.quantize import compute_range_from_tensor
        from data.preprocess.transform import get_transform
        from data.preprocess.quantize import UniformQuantizer

        cond_bits = int(self.config.get('bit_size', 4))
        real_bits = int(self.config.get('real_bit_size', 16))

        lower_pct = float(self.config.get('quantile_clip_lower', 0.0))
        upper_pct = float(self.config.get('quantile_clip_upper', 100.0))

        lo, hi = compute_range_from_tensor(raw_signals, lower_pct, upper_pct)
        peak = max(abs(lo), abs(hi))
        if peak <= 0:
            peak = float(raw_signals.abs().max().item())
        if peak <= 0:
            peak = 1.0

        q4 = UniformQuantizer(bits=cond_bits, range_min=-peak, range_max=peak)
        q16 = UniformQuantizer(bits=real_bits, range_min=-peak, range_max=peak)

        wave4 = q4.quantize(raw_signals.to(self.device))
        wave16 = q16.quantize(raw_signals.to(self.device))

        pipeline_config = (self.config.get('pipeline_config') or {}).copy()
        transform_type = pipeline_config.pop('transform', 'stft')
        transform4 = get_transform(transform_type, **pipeline_config, device=self.device)
        transform16 = get_transform(transform_type, **pipeline_config, device=self.device)

        cond_mag = transform4.apply(wave4)
        real_mag = transform16.apply(wave16)

        cond_mag_min = float(cond_mag.min().item())
        cond_mag_max = float(cond_mag.max().item())
        real_mag_min = float(real_mag.min().item())
        real_mag_max = float(real_mag.max().item())

        return {
            'cond_bits': cond_bits,
            'real_bits': real_bits,
            'time_range_min': float(q4.range_min),
            'time_range_max': float(q4.range_max),
            'cond_mag_min': cond_mag_min,
            'cond_mag_max': cond_mag_max,
            'real_mag_min': real_mag_min,
            'real_mag_max': real_mag_max,
            'pipeline_config': (self.config.get('pipeline_config') or {}).copy(),
        }
    
    def preprocess_condition(self, raw_signals: torch.Tensor, params: dict) -> torch.Tensor:
        """Preprocess condition: time-quantize -> STFT magnitude -> normalize to [-1, 1]."""
        from data.preprocess.transform import get_transform
        from data.preprocess.quantize import UniformQuantizer

        cond_bits = int(params['cond_bits']) #use the conditional data bit sizew
        tmin = float(params['time_range_min'])
        tmax = float(params['time_range_max'])

        q_time = UniformQuantizer(bits=cond_bits, range_min=tmin, range_max=tmax)
        wave_4bit = q_time.quantize(raw_signals.to(self.device))

        pipeline_config = params['pipeline_config'].copy()
        transform_type = pipeline_config.pop('transform', 'stft')
        transform = get_transform(transform_type, **pipeline_config, device=self.device)
        mag = transform.apply(wave_4bit)

        self._last_phase = transform.phase.detach().clone()
        self._last_input_length = int(wave_4bit.shape[-1])

        cmin = float(params['cond_mag_min'])
        cmax = float(params['cond_mag_max'])

        if mag.ndim == 3:
            mag = mag.unsqueeze(1)

        normalized = (mag - float(cmin)) / (float(cmax) - float(cmin) + 1e-8)
        normalized = normalized * 2.0 - 1.0
        #confirm sixze and datatype
        print(f"  Preprocess condition - shape: {normalized.shape}, dtype: {normalized.dtype}, range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        return normalized
    
    def postprocess_generated(self, generated: torch.Tensor, condition_phase: torch.Tensor, params: dict) -> torch.Tensor:
        """Denormalize generated magnitude and invert transform using condition phase."""
        from data.preprocess.transform import get_transform
        
        rmin = float(params['cond_mag_min'])
        rmax = float(params['cond_mag_max'])

        denormalized = (generated + 1.0) / 2.0
        denormalized = denormalized * (float(rmax) - float(rmin)) + float(rmin)

        
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
        input_len = getattr(self, '_last_input_length', None)
        if input_len is not None:
            transform.input_length = int(input_len)
        
        time_signals = transform.inverse(denormalized)

    
        
        return time_signals.cpu()

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

            # Denormalize using conditional magnitude ranges (same as postprocess inversion).
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
                    dtype=condition_batch.dtype,
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
        save_progress_every_n_steps: Optional[int] = None,
    ) -> dict:
        """Generate samples from raw condition signals."""
        if use_ddim is None:
            sched = str(self.config.get('scheduler_type', 'ddpm')).lower()
            use_ddim = (sched == 'ddim')

        # Training often uses 1000 noising steps, but DDIM sampling typically
        # uses far fewer inference steps.
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
        
        condition_phases = {}

        H, W = self.config.get('image_size', (16, 128))

        params = self._compute_sampling_params(raw_condition_signals)
        
        for cond_idx in tqdm(range(num_conditions), desc="Processing conditions"):
            raw_signal = raw_condition_signals[cond_idx:cond_idx+1]
            self._last_input_length = int(raw_signal.shape[-1])
            
            print(f"\n[Sample {cond_idx}] raw range: [{raw_signal.min():.2f}, {raw_signal.max():.2f}]")
            
            sample_dir = samples_root / f'sample_{cond_idx}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({'raw_signal': raw_signal.cpu()}, sample_dir / 'condition_raw.pt')
            
            from data.preprocess.transform import get_transform
            from data.preprocess.quantize import UniformQuantizer

            q4_time = UniformQuantizer(bits=int(params['cond_bits']), range_min=float(params['time_range_min']), range_max=float(params['time_range_max']))
            q16_time = UniformQuantizer(bits=int(params['real_bits']), range_min=float(params['time_range_min']), range_max=float(params['time_range_max']))

            time_4bit = q4_time.quantize(raw_signal.to(self.device))
            time_16bit = q16_time.quantize(raw_signal.to(self.device))

            torch.save({'time_domain_4bit': time_4bit.cpu()}, sample_dir / 'condition_4bit_time.pt')
            torch.save({'time_domain_16bit': time_16bit.cpu()}, sample_dir / 'ground_truth_16bit_time.pt')

            print(f"  4-bit range:  [{time_4bit.min():.2f}, {time_4bit.max():.2f}]")
            print(f"  16-bit range: [{time_16bit.min():.2f}, {time_16bit.max():.2f}]")

            pipeline_config = params['pipeline_config'].copy()
            transform_type = pipeline_config.pop('transform', 'stft')
            transform = get_transform(transform_type, **pipeline_config, device=self.device)
            spectrogram_4 = transform.apply(time_4bit)
            condition_phases[cond_idx] = transform.phase.clone()
            print(f"  STFT(4-bit) mag range: [{spectrogram_4.min():.2f}, {spectrogram_4.max():.2f}]")
            
            condition_spec = self.preprocess_condition(raw_signal, params=params)
            condition_spec = condition_spec.to(self.device)
            print(f"  Condition norm range: [{condition_spec.min():.4f}, {condition_spec.max():.4f}]")

            
            torch.save({'condition_spec_4bit': condition_spec.cpu()}, sample_dir / 'condition_spec.pt')

            # Cache "before inversion" (generated spectrograms) per trajectory.
            # If progress snapshots are requested, we force a fresh denoising run so
            # the intermediate snapshots match the current model run.
            force_progress = (save_progress_every_n_steps is not None) and (int(save_progress_every_n_steps) > 0)
            cached_specs: dict[int, torch.Tensor] = {}
            missing_indices: list[int] = []

            if force_progress:
                missing_indices = [int(i) for i in range(int(num_trajectories))]
                print('  Progress snapshots enabled: forcing resample (ignoring cached specs).')
            else:
                for traj_idx in range(num_trajectories):
                    spec_path = sample_dir / f'trajectory_{traj_idx}_spec.pt'
                    legacy_path = sample_dir / f'trajectory_{traj_idx}.pt'
                    loaded = None
                    if spec_path.exists():
                        obj = torch.load(spec_path, map_location='cpu')
                        if isinstance(obj, dict):
                            loaded = obj.get('spectrogram_16bit') or obj.get('spectrogram_16bit_norm') or obj.get('spectrogram')
                        else:
                            loaded = obj
                    elif legacy_path.exists():
                        obj = torch.load(legacy_path, map_location='cpu')
                        if isinstance(obj, dict):
                            loaded = obj.get('spectrogram_16bit')

                    if loaded is None:
                        missing_indices.append(traj_idx)
                    else:
                        cached_specs[traj_idx] = torch.as_tensor(loaded).to(dtype=torch.float32)

            if missing_indices:
                print(f"  Sampling {len(missing_indices)}/{num_trajectories} missing trajectories (cached: {num_trajectories - len(missing_indices)})")

                generated_missing = self._sample_trajectories(
                    condition_spec=condition_spec,
                    trajectory_indices=[int(i) for i in missing_indices],
                    num_inference_steps=int(num_inference_steps),
                    use_ddim=bool(use_ddim),
                    H=int(H),
                    W=int(W),
                    progress_dir=(sample_dir / 'progress') if (save_progress_every_n_steps is not None and int(save_progress_every_n_steps) > 0) else None,
                    progress_every_n_steps=save_progress_every_n_steps,
                    progress_condition_phase=condition_phases[cond_idx],
                    progress_params=params,
                )

                # Save cached specs and fill into cached_specs dict
                for j, traj_idx in enumerate(missing_indices):
                    spec = generated_missing[j:j+1].detach().to('cpu')
                    torch.save({'spectrogram_16bit': spec}, sample_dir / f'trajectory_{traj_idx}_spec.pt')
                    cached_specs[int(traj_idx)] = spec.to(dtype=torch.float32)
            else:
                print(f"  All {num_trajectories} trajectories already cached; skipping diffusion sampling.")

            # Assemble batch in original trajectory order.
            generated_specs_batch = torch.cat(
                [cached_specs[i] for i in range(num_trajectories)],
                dim=0,
            ).to(device=self.device, dtype=condition_spec.dtype)

            print(
                f"  Using generated batch - shape: {generated_specs_batch.shape}, dtype: {generated_specs_batch.dtype}, "
                f"range: [{generated_specs_batch.min():.4f}, {generated_specs_batch.max():.4f}]"
            )
            
            # Postprocess each trajectory (need to do separately due to phase)
            trajectories_spec = []
            trajectories_time = []
            
            for traj_idx in range(num_trajectories):
                generated_spec = generated_specs_batch[traj_idx:traj_idx+1]  # Keep batch dim [1, 1, H, W]
                
                # Postprocess to time domain using stored phase
                generated_time = self.postprocess_generated(
                    generated_spec, 
                    condition_phase=condition_phases[cond_idx],
                    params=params,
                )
                print(f"    Trajectory {traj_idx} time range: [{generated_time.min():.2f}, {generated_time.max():.2f}]")
                
                trajectories_spec.append(generated_spec.cpu())
                trajectories_time.append(generated_time.cpu())
                
                # Save individual trajectory
                torch.save({
                    'spectrogram_16bit': generated_spec.cpu(),
                    'time_domain': generated_time.cpu()
                }, sample_dir / f'trajectory_{traj_idx}.pt')
            
            # Save all trajectories for this condition
            torch.save({
                'spectrograms_16bit': torch.cat(trajectories_spec, dim=0),
                'time_domain': torch.cat(trajectories_time, dim=0)
            }, sample_dir / 'all_trajectories.pt')
            
            results['conditions'].append(condition_spec.cpu())
            results['generated_specs'].append(torch.cat(trajectories_spec, dim=0))
            results['generated_time'].append(torch.cat(trajectories_time, dim=0))
        
        return results
    



if __name__ == "__main__":
    # Load config to get paths and settings
    results_dir = Path("diffusion/results")
    version = "V1"
    
    config_path = results_dir / version / 'config.pkl'
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # Initialize sampler with epoch 3 checkpoint
    sampler = DiffusionSampler(
        results_dir=results_dir,
        version=version,
        checkpoint_name="checkpoint_epoch_15.pt",
        device="cpu"
    )
    raw_data_path = Path(str(config.get('raw_data_path') or '')).expanduser()
    if not str(raw_data_path):
        raw_data_path = Path("data/data/raw/art_chunks.pt")

    print(f"\n[Sampler] Loading condition data from: {raw_data_path}")
    data = torch.load(raw_data_path)
    raw_signals = data.get('signals') or data.get('chunks')
  
   
    # Default: sample from a held-out test tail (not seen during training).
    holdout_n = int(config.get('test_holdout_count', 0) or 0)
    if holdout_n > 0 and raw_signals.shape[0] > holdout_n:
        raw_signals = raw_signals[-holdout_n:]
        print(f"[Sampler] Using held-out test split: last {holdout_n} samples")
    else:
        print("[Sampler] Holdout not configured (or dataset too small); sampling from full dataset")

    num_samples = 1  # number of different conditions to sample
    raw_signals = raw_signals[:num_samples]
    print(type(raw_signals))
    num_trajectories = 1 
    print(f"[Sampler] Generating from {num_samples} conditions, {num_trajectories} trajectory each...")

    

    # Default to a practical DDIM sampling length.
    num_inference_steps = int(config.get('num_inference_steps', 50))
    save_progress_every = 5


    results = sampler.sample(
        raw_condition_signals=raw_signals,
        num_inference_steps=num_inference_steps,
        use_ddim=True, 
        num_trajectories=num_trajectories,
        save_progress_every_n_steps=save_progress_every,
    )

    # Quick visualization for the first sample/trajectory
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        sampler_subdir = 'ddim'
        sample_dir = sampler.version_dir / 'samples' / sampler_subdir / 'sample_0'
        cond_time = torch.load(sample_dir / 'condition_4bit_time.pt')['time_domain_4bit'][0].cpu().numpy()
        gt_time = torch.load(sample_dir / 'ground_truth_16bit_time.pt')['time_domain_16bit'][0].cpu().numpy()
        cond_spec = torch.load(sample_dir / 'condition_spec.pt')['condition_spec_4bit'][0, 0].cpu().numpy()
        traj = torch.load(sample_dir / 'trajectory_0.pt')
        gen_spec = traj['spectrogram_16bit'][0, 0].cpu().numpy()
        gen_time = traj['time_domain'][0].cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(cond_time, label='cond 4-bit', linewidth=1)
        ax.plot(gt_time, label='gt 16-bit', linewidth=1, alpha=0.8)
        ax.plot(gen_time, label='generated (inverted)', linewidth=1, alpha=0.8)
        ax.set_title('Time domain')
        ax.legend(loc='best', fontsize=8)

        ax = axes[0, 1]
        im = ax.imshow(cond_spec, aspect='auto', origin='lower')
        ax.set_title('Condition spec (normalized)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 0]
        im = ax.imshow(gen_spec, aspect='auto', origin='lower')
        ax.set_title('Generated spec (normalized)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 1]
        err = gt_time - gen_time
        ax.plot(err, color='tab:red', linewidth=1)
        ax.set_title('Error (gt - generated)')

        out_path = sample_dir / 'preview_epoch1_sample0.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Sampler] Wrote preview: {out_path}")
    except Exception as e:
        print(f"[Sampler] Preview plot failed: {e}")
    
    print(f"[Sampler] Done! Results saved to: {sampler.version_dir / 'samples'}")
