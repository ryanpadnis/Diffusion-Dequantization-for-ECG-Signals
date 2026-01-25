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
        
        from diffusion.utils.model import ConditionalDiffuser
        checkpoint_path = self.version_dir / 'checkpoints' / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = ConditionalDiffuser(
            self.config,
            unet_type=self.config.get('unet_type', 'conditional'),
            scheduler_type=self.config.get('scheduler_type', 'ddpm')
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        dtype = resolve_torch_dtype(self.config, device=self.device)
        self.model = self.model.to(device=self.device, dtype=dtype)
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

        cond_bits = int(params['cond_bits'])
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
        return normalized
    
    def postprocess_generated(self, generated: torch.Tensor, condition_phase: torch.Tensor, params: dict) -> torch.Tensor:
        """Denormalize generated magnitude and invert transform using condition phase."""
        from data.preprocess.transform import get_transform
        
        rmin = float(params['real_mag_min'])
        rmax = float(params['real_mag_max'])

        denormalized = (generated + 1.0) / 2.0
        denormalized = denormalized * (float(rmax) - float(rmin)) + float(rmin)

        # Model tensors include a channel dim; STFTTransform.inverse expects [B, F, T].
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
        
        time_frames = denormalized.shape[-1]
        hop = pipeline_config.get('hop_length')
        n_fft = pipeline_config.get('n_fft')
        transform.input_length = (time_frames - 1) * hop + n_fft
        
        time_signals = transform.inverse(denormalized)

    
        
        return time_signals.cpu()
    
    def sample(
        self,
        raw_condition_signals: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        use_ddim: Optional[bool] = None,
        num_trajectories: int = 1,
    ) -> dict:
        """Generate samples from raw condition signals."""
        if num_inference_steps is None:
            num_inference_steps = int(self.config.get('num_noising_steps', 1000))

        if use_ddim is None:
            sched = str(self.config.get('scheduler_type', 'ddpm')).lower()
            use_ddim = (sched == 'ddim')

        num_conditions = len(raw_condition_signals)
        results = {
            'conditions': [],
            'generated_specs': [],
            'generated_time': []
        }
        
        condition_phases = {}

        pipeline_config = (self.config.get('pipeline_config') or {})
        expected_length = (self.config['image_size'][1] - 1) * pipeline_config['hop_length'] + pipeline_config['n_fft']

        params = self._compute_sampling_params(raw_condition_signals)
        
        for cond_idx in tqdm(range(num_conditions), desc="Processing conditions"):
            raw_signal = raw_condition_signals[cond_idx:cond_idx+1]
            
            current_length = raw_signal.shape[-1]
            if current_length > expected_length:
                raw_signal = raw_signal[:, :expected_length]
            elif current_length < expected_length:
                pad_size = expected_length - current_length
                raw_signal = torch.nn.functional.pad(raw_signal, (0, pad_size), mode='constant', value=0)
            
            print(f"\n[Sample {cond_idx}] raw range: [{raw_signal.min():.2f}, {raw_signal.max():.2f}]")
            
            sample_dir = self.version_dir / 'samples' / f'sample_{cond_idx}'
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

            condition_batch = condition_spec.repeat(num_trajectories, 1, 1, 1)
            print(f"  Generating {num_trajectories} trajectories in batch - condition shape: {condition_batch.shape}")
            
            with torch.no_grad():
                all_noise = []
                for traj_idx in range(num_trajectories):
                    generator = torch.Generator(device=self.device).manual_seed(traj_idx)
                    noise = torch.randn(
                        (1, 1, 16, 128),
                        device=self.device,
                        generator=generator,
                        dtype=condition_batch.dtype,
                    )
                    all_noise.append(noise)
                
                batched_noise = torch.cat(all_noise, dim=0)

                from diffusers import DDIMScheduler
                
                if use_ddim and self.model.scheduler_type == 'ddpm':
                    scheduler = DDIMScheduler.from_config(self.model.noise_scheduler.config)
                else:
                    scheduler = self.model.noise_scheduler
                
                scheduler.set_timesteps(num_inference_steps, device=self.device)
                
                image = batched_noise
                cond_embed = self.model.encode_condition(condition_batch)

                for t in tqdm(scheduler.timesteps, desc=f"Denoising {num_trajectories} trajectories", leave=False):
                    noise_pred = self.model.unet(
                        image,
                        t,
                        encoder_hidden_states=cond_embed,
                        return_dict=False
                    )[0]
                    image = scheduler.step(noise_pred, t, image).prev_sample
                
                generated_specs_batch = image  # [num_trajectories, 1, 16, 128]
                
                print(f"  Generated batch - shape: {generated_specs_batch.shape}, dtype: {generated_specs_batch.dtype}, range: [{generated_specs_batch.min():.4f}, {generated_specs_batch.max():.4f}]")
            
            # Postprocess each trajectory (need to do separately due to phase)
            trajectories_spec = []
            trajectories_time = []
            
            for traj_idx in range(num_trajectories):
                generated_spec = generated_specs_batch[traj_idx:traj_idx+1]  # Keep batch dim [1, 1, 16, 128]
                
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
        checkpoint_name="checkpoint_epoch_3.pt",
        device="cpu"
    )
    
    # Load raw condition data from config
    raw_data_path = Path(config['raw_data_path'])
    print(f"\n[Sampler] Loading condition data from: {raw_data_path}")
    data = torch.load(raw_data_path)
    raw_signals = data.get('signals') or data.get('chunks')
    if raw_signals is None:
        raise ValueError("Expected 'signals' or 'chunks' key in data")
   
    num_samples = 1 #different conditions
    raw_signals = raw_signals[:num_samples]
    num_trajectories = 1 #repeat conditon across ther batch for diversity
    print(f"[Sampler] Generating from {num_samples} conditions, {num_trajectories} trajectory each...")
    
    # Allow env var override without argparse.
    import os
    env_steps = os.environ.get('SAMPLE_STEPS')
    num_inference_steps = int(env_steps) if env_steps is not None else None

    # Show which scale is used for diffusion output inversion.
    print(f"[Sampler] Output inversion scale: {config.get('output_scale', 'real')} (real=target/16-bit, cond=condition/4-bit)")
    default_steps = config.get('num_noising_steps', 1000)
    print(f"[Sampler] Inference steps: {num_inference_steps if num_inference_steps is not None else default_steps}")
    print(f"[Sampler] Scheduler type: {config.get('scheduler_type', 'ddpm')}")

    results = sampler.sample(
        raw_condition_signals=raw_signals,
        num_inference_steps=num_inference_steps,
        use_ddim=None,
        num_trajectories=num_trajectories,
    )
    
    print(f"[Sampler] Done! Results saved to: {sampler.version_dir / 'samples'}")
