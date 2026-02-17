"""Runnable training runner with direct torch transforms.

Orchestrates: load raw data → transform (on device) → quantize → train → save.
Uses apply() directly for GPU speed instead of sklearn pipeline numpy conversion.
"""

import os
from pathlib import Path
from typing import Optional

import io
import tempfile

import torch

from data.preprocess.transform import get_transform, AVAILABLE_TRANSFORMS
from data.preprocess.quantize import UniformQuantizer, compute_range_from_tensor


def _per_sample_symmetric_peak(signals_time: torch.Tensor, *, lower_pct: float, upper_pct: float) -> torch.Tensor:
    """Compute per-sample symmetric peak for time-domain quantization.

    Returns peak per sample with shape [N, 1] suitable for broadcasting.
    Uses quantiles when lower/upper pct are not [0, 100].
    """
    x = signals_time
    if x.ndim != 2:
        x = x.view(x.shape[0], -1)

    if lower_pct <= 0.0 and upper_pct >= 100.0:
        lo = x.amin(dim=1)
        hi = x.amax(dim=1)
    else:
        q_lo = float(lower_pct) / 100.0
        q_hi = float(upper_pct) / 100.0
        # torch.quantile supports dim; this is much faster than a Python loop.
        lo = torch.quantile(x, torch.tensor(q_lo, device=x.device), dim=1)
        hi = torch.quantile(x, torch.tensor(q_hi, device=x.device), dim=1)

    peak = torch.maximum(lo.abs(), hi.abs())
    # avoid zero range
    peak = torch.where(peak <= 0, x.abs().amax(dim=1), peak)
    peak = torch.where(peak <= 0, torch.ones_like(peak), peak)
    return peak.view(-1, 1)


def _uniform_quantize_per_sample(signals_time: torch.Tensor, *, bits: int, peak: torch.Tensor) -> torch.Tensor:
    """Uniformly quantize each sample using its own symmetric range [-peak_i, peak_i].

    Returns quantized float values at bin centers (like UniformQuantizer.quantize).
    """
    levels = float(2 ** int(bits))
    # step: (range_max - range_min) / levels = (2*peak)/levels
    step = (2.0 * peak) / levels  # [N,1]
    step = torch.where(step.abs() < 1e-12, torch.ones_like(step), step)

    # Clip and compute indices
    x = signals_time
    if x.ndim != 2:
        x = x.view(x.shape[0], -1)

    x_clip = torch.clamp(x, -peak, peak)
    # indices = floor((x - (-peak)) / step) = floor((x + peak)/step)
    idx = torch.floor((x_clip + peak) / step)
    idx = torch.clamp(idx, 0, levels - 1)
    # decode to centers: idx*step + (-peak) + 0.5*step
    q = idx * step - peak + (0.5 * step)
    return q.view_as(signals_time)


def _resolve_torch_dtype(config: dict, device: str) -> torch.dtype:
    dtype_val = config.get('torch_dtype', 'float32')
    if isinstance(dtype_val, torch.dtype):
        dtype = dtype_val
    else:
        dtype = getattr(torch, str(dtype_val), torch.float32)

    dev = torch.device(device)
    if dev.type in {'cpu', 'mps'} and dtype == torch.float16:
        return torch.float32
    return dtype


def construct_train_filename(quantizer_type: str, transform_type: str, bits: int, cond: bool = False) -> str:
    """Construct filename for processed training data based on parameters.
    
    Args:
        quantizer_type: Type of quantizer (e.g., 'uniform')
        transform_type: Type of transform (e.g., 'stft')
        bits: Number of bits for quantization
        cond: If True, use 'train_cond' prefix; otherwise use 'train_data'
    """
    prefix = "train_cond" if cond else "train_data"
    return f"{prefix}_{quantizer_type}_{transform_type}_{bits}bit.pt"





def load_data(data_path: Path) -> torch.Tensor:
    """Load raw data (signals or chunks) from .pt file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    data = torch.load(data_path)
    
    # Handle both dict and direct tensor formats
    if isinstance(data, dict):
        signals = data.get("signals")
        if signals is None:
            signals = data.get("chunks")
        if signals is None:
            raise KeyError("Expected key 'signals' or 'chunks' in loaded data")
    else:
        signals = data

    return signals


def create_transform(transform_config: dict, device: str):
    """Create transform object from pipeline_config using registry.
    
    Prefers `pipeline_config` inside the overall training config for consistency
    across training and sampling.
    """
    device_obj = torch.device(device)
    pipeline_config = transform_config.get('pipeline_config', {})
    transform_name = pipeline_config.get('transform', 'stft')

    extra_kwargs = {k: v for k, v in pipeline_config.items() if k != 'transform'}

    if transform_name == 'stft':
        return get_transform(
            'stft',
            n_fft=pipeline_config.get('n_fft', 30),
            hop_length=pipeline_config.get('hop_length', 64),
            win_length=pipeline_config.get('win_length', None),
            onesided=pipeline_config.get('onesided', True),
            center=pipeline_config.get('center', False),
            device=device_obj,
        )
    else:
        return get_transform(transform_name, **extra_kwargs, device=device_obj)


def create_quantizer(config: dict, signals: torch.Tensor) -> UniformQuantizer:
    """Create quantizer from config and compute range from signals."""
    bits = config.get('bits', 4)
    range_min = float(signals.min().item())
    range_max = float(signals.max().item())


    q = UniformQuantizer(bits=bits, range_min=range_min, range_max=range_max)
    return q


def create_time_quantizer(config: dict, signals_time: torch.Tensor, bits: int) -> UniformQuantizer:
    """Create a uniform quantizer for time-domain signals.

    Uses percentile clipping and a symmetric range around 0 for stability.
    """
    lower_pct = float(config.get('quantile_clip_lower', 0.0))
    upper_pct = float(config.get('quantile_clip_upper', 100.0))

    lo, hi = compute_range_from_tensor(signals_time, lower_pct, upper_pct)
    peak = max(abs(lo), abs(hi))
    if peak <= 0:
        peak = float(signals_time.detach().abs().max().item())
    if peak <= 0:
        peak = 1.0

    q = UniformQuantizer(bits=bits, range_min=-peak, range_max=peak)
    q._meta = {
        'lower_pct': lower_pct,
        'upper_pct': upper_pct,
        'symmetric': True,
        'lo': float(lo),
        'hi': float(hi),
        'peak': float(peak),
    }
    return q


def build_diffuser(config: dict):
    """Build diffuser model, optimizer, and scheduler from config."""
    from diffusion.utils.diffusion_models import create_diffuser, get_optimizer, get_lr_scheduler
    diffuser = create_diffuser(config)
    optimizer = get_optimizer(diffuser, config)
    lr_scheduler = None
    
    print(f"[build_diffuser] Created ConditionalDiffuser")
    print(f"  - Image size: {config.get('image_size', (128, 64))}")
    print(f"  - Channels: {config.get('in_channels', 1)}")
    print(f"  - Optimizer: {type(optimizer).__name__}")
    
    return diffuser, optimizer, lr_scheduler


def process_data_before_training(config:dict):
    """Process raw data into condition and real datasets for training."""
    #need to consider whether the s3 tag is in use for where to load eveyrhting into
    #this is decuplped form the training loop so that we can do the processing step in s3 not the cluster
    epochs = config.get('epochs', 1)
    device = config.get('device', 'cpu')
    
    print(f"[train_diffuser] Device: {device}")

    # Optional: store/load preprocessed data directly in S3.
    # This is useful on ephemeral clusters where local disk shouldn't be the source of truth.
    s3_data_uri = str(config.get('s3_data_uri') or '').strip() or None

    # Initialize to satisfy control flow when force_preprocess is set.
    cond_data = None
    real_data = None

    # Always define these so normalizer metadata construction is safe even when
    # we load already-preprocessed tensors (e.g. from S3) and never build time-domain quantizers.
    cond_time_quantizer = None
    real_time_quantizer = None

    def _extract_signals(loaded):
        # Mirror load_data() behavior but for already-loaded objects.
        if isinstance(loaded, dict):
            signals = loaded.get('signals')
            if signals is None:
                signals = loaded.get('chunks')
            if signals is None:
                raise KeyError("Expected key 'signals' or 'chunks' in loaded data")
            return signals
        return loaded

    def _s3_client():
        # Lazy import to avoid requiring boto3 unless S3 features are used.
        from diffusion.aws.s3_io import _require_boto3

        boto3 = _require_boto3()
        return boto3.client('s3')

    def _s3_get_torch(s3_uri: str):
        from diffusion.aws.s3_io import split_s3_uri
        from botocore.exceptions import ClientError  # type: ignore

        bucket, key = split_s3_uri(s3_uri)
        if not key:
            raise ValueError(f"S3 URI must include a key: {s3_uri}")

        client = _s3_client()
        try:
            resp = client.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = str(e.response.get('Error', {}).get('Code', ''))
            if code in {'NoSuchKey', '404', 'NotFound'}:
                raise FileNotFoundError(s3_uri) from e
            raise

        body = resp['Body']
        # torch.load requires a seekable file-like object.
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as f:
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
            f.seek(0)
            return torch.load(f)

    def _s3_get_bytes(s3_uri: str) -> bytes:
        from diffusion.aws.s3_io import split_s3_uri
        from botocore.exceptions import ClientError  # type: ignore

        bucket, key = split_s3_uri(s3_uri)
        if not key:
            raise ValueError(f"S3 URI must include a key: {s3_uri}")

        client = _s3_client()
        try:
            resp = client.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = str(e.response.get('Error', {}).get('Code', ''))
            if code in {'NoSuchKey', '404', 'NotFound'}:
                raise FileNotFoundError(s3_uri) from e
            raise
        return resp['Body'].read()

    def _s3_put_torch(obj, s3_uri: str) -> None:
        from diffusion.aws.s3_io import split_s3_uri

        bucket, key = split_s3_uri(s3_uri)
        if not key:
            raise ValueError(f"S3 URI must include a key: {s3_uri}")

        client = _s3_client()
        # Avoid holding an extra full copy in RAM for large tensors by spooling to tmp.
        with tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024) as f:
            torch.save(obj, f)
            f.seek(0)
            client.upload_fileobj(f, bucket, key)

    def _s3_put_bytes(data: bytes, s3_uri: str) -> None:
        from diffusion.aws.s3_io import split_s3_uri

        bucket, key = split_s3_uri(s3_uri)
        if not key:
            raise ValueError(f"S3 URI must include a key: {s3_uri}")
        client = _s3_client()
        client.put_object(Bucket=bucket, Key=key, Body=data)

    # Get paths and processing parameters from config
    data_dir_val = config.get('data_dir')
    data_dir = Path(data_dir_val) if data_dir_val else None
    raw_data_path = Path(config.get('raw_data_path'))
    quantizer_type = config.get('quantizer_type', 'uniform')
    transform_type = config.get('transform_type', 'stft')
    cond_bits = config.get('bit_size', 4)  # Condition dataset bit depth
    real_bits = config.get('real_bit_size', 16)  # Real data bit depth
    
    # Get directories from config
    checkpoint_dir = Path(config.get('checkpoint_dir'))
    samples_dir = Path(config.get('samples_dir'))
    logs_dir = Path(config.get('logs_dir'))

    from datetime import datetime
    run_id = str(config.get('run_id') or datetime.now().strftime('%Y%m%d_%H%M%S'))
    logs_dir = logs_dir / run_id
    config['logs_dir'] = str(logs_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    if s3_data_uri is None:
        if data_dir is None:
            raise ValueError("config['data_dir'] must be set when not using S3 for preprocessed data")
        os.makedirs(data_dir, exist_ok=True)
    
    # Save config for later loading by sampler
    import pickle
    config_path = Path(config.get('results_dir')) / 'config.pkl'
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"[train_diffuser] Config saved to: {config_path}")
    
    # Real data versus conditonal (hgih vs low bit depth)
    cond_filename = construct_train_filename(quantizer_type, transform_type, cond_bits, cond=True)
    real_filename = construct_train_filename(quantizer_type, transform_type, real_bits, cond=False)
    
    if s3_data_uri is not None:
        from diffusion.aws.s3_io import normalize_s3_uri, join_s3_uri

        s3_data_uri = normalize_s3_uri(s3_data_uri)
        cond_data_s3 = join_s3_uri(s3_data_uri, cond_filename)
        real_data_s3 = join_s3_uri(s3_data_uri, real_filename)
        normalizer_s3 = join_s3_uri(s3_data_uri, 'normalizer_params.pkl')
        cond_data_path = None
        real_data_path = None
    else:
        assert data_dir is not None
        cond_data_path = data_dir / cond_filename
        real_data_path = data_dir / real_filename
  
    force_preprocess = bool(config.get('force_preprocess', False))

    # If a fixed test holdout is enabled, avoid silently reusing preprocessed
    # datasets from a different regime.
    test_holdout_count = int(config.get('test_holdout_count', 0) or 0)
    test_holdout_from_end = bool(config.get('test_holdout_from_end', True))
    if test_holdout_count > 0 and not force_preprocess:
        try:
            if s3_data_uri is None and data_dir is not None:
                normalizer_path = data_dir / 'normalizer_params.pkl'
                if normalizer_path.exists():
                    import pickle
                    with open(normalizer_path, 'rb') as f:
                        norm_prev = pickle.load(f)
                    prev_n = int(norm_prev.get('test_holdout_count', 0) or 0)
                    prev_end = bool(norm_prev.get('test_holdout_from_end', True))
                    if prev_n != test_holdout_count or prev_end != test_holdout_from_end:
                        print('[train_diffuser] Holdout settings changed; forcing preprocess.')
                        force_preprocess = True
                else:
                    print('[train_diffuser] Holdout enabled but no normalizer metadata found; forcing preprocess.')
                    force_preprocess = True
            elif s3_data_uri is not None:
                # For S3 workflows, do not force preprocessing. We will reuse cached tensors from S3
                # to avoid requiring raw_data_path on remote clusters.
                pass
        except Exception:
            if s3_data_uri is None:
                print('[train_diffuser] Warning: could not validate existing holdout metadata; forcing preprocess.')
                force_preprocess = True
            else:
                print('[train_diffuser] Warning: could not validate existing holdout metadata for S3 run; continuing without forcing preprocess.')

    if (not force_preprocess) and s3_data_uri is not None:
        # Try S3 first.
        try:
            print(f"[train_diffuser] Loading pre-processed datasets from S3:")
            print(f"  - Condition (4-bit): {cond_data_s3}")
            cond_loaded = _s3_get_torch(cond_data_s3)
            print(f"  - Real data (16-bit): {real_data_s3}")
            real_loaded = _s3_get_torch(real_data_s3)

            # Best-effort: load normalizer metadata too for debugging consistency.
            try:
                import pickle as _pickle
                norm_bytes = _s3_get_bytes(normalizer_s3)
                norm = _pickle.loads(norm_bytes)
                print(f"[train_diffuser] Loaded normalizer_params.pkl from S3: {normalizer_s3}")
                for k in [
                    'time_quantization_mode',
                    'time_quant_clip_lower',
                    'time_quant_clip_upper',
                    'cond_bits',
                    'real_bits',
                    'mag_normalization',
                ]:
                    if k in norm:
                        print(f"  - {k}: {norm.get(k)}")
            except Exception as e:
                print(f"[train_diffuser] Warning: could not load/parse normalizer_params.pkl from S3 (non-fatal): {e}")

            cond_data = _extract_signals(cond_loaded).to(device)
            real_data = _extract_signals(real_loaded).to(device)

            # Channel dimension for 2-d unet
            if cond_data.ndim == 3:
                cond_data = cond_data.unsqueeze(1)
            if real_data.ndim == 3:
                real_data = real_data.unsqueeze(1)

            print(f"[train_diffuser] *** LOADED FROM S3 ***")
            print(f"[train_diffuser] *** CODE VERSION: 2026-02-05-v2 ***")
            print(f"[train_diffuser] Condition shape: {cond_data.shape} (samples: {cond_data.shape[0]})")
            print(f"[train_diffuser] Real data shape: {real_data.shape} (samples: {real_data.shape[0]})")
            print(f"[train_diffuser] Expected batches with batch_size=16: {cond_data.shape[0] // 16}")
        except FileNotFoundError:
            raise ValueError(
                "Preprocessed data not found in S3. This run is configured for S3-only data loading. "
                "Prepare and upload tensors first (use --prepare-s3-data or --prepare-s3-data-if-missing), "
                "or set force_preprocess=True only in an environment where raw_data_path exists. "
                f"Missing one or more of: {cond_data_s3}, {real_data_s3}"
            )

    if cond_data is not None and real_data is not None:
        pass
    elif (not force_preprocess) and s3_data_uri is None and cond_data_path.exists() and real_data_path.exists():
        # Load pre-processed data directly
        print(f"[train_diffuser] Loading pre-processed datasets:")
        print(f"  - Condition (4-bit): {cond_data_path}")
        cond_data = load_data(cond_data_path).to(device)
        print(f"  - Real data (16-bit): {real_data_path}")
        real_data = load_data(real_data_path).to(device)
        
        # Channel dimension for 2-d uent
        if cond_data.ndim == 3:
            cond_data = cond_data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        if real_data.ndim == 3:
            real_data = real_data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        
        print(f"[train_diffuser] Condition shape: {cond_data.shape}")
        print(f"[train_diffuser] Real data shape: {real_data.shape}")
        
    elif raw_data_path.exists():
        # Load raw data and process to both bit depths
        print(f"[train_diffuser] Loading raw data from: {raw_data_path}")
        signals = load_data(raw_data_path)
        signals = signals.to(device)
        print(f"[train_diffuser] Loaded raw signals shape: {signals.shape}")

        # Reserve a fixed hold-out test set so training/validation never uses it.
        if test_holdout_count > 0:
            n_total = int(signals.shape[0])
            n_hold = min(int(test_holdout_count), n_total)
            if n_hold <= 0 or n_hold >= n_total:
                raise ValueError(
                    f"Invalid test_holdout_count={test_holdout_count} for dataset size {n_total}. "
                    "Set test_holdout_count to a smaller positive value."
                )

            if test_holdout_from_end:
                train_signals = signals[:-n_hold]
                test_signals = signals[-n_hold:]
                test_slice = (n_total - n_hold, n_total)
            else:
                train_signals = signals[n_hold:]
                test_signals = signals[:n_hold]
                test_slice = (0, n_hold)

            print(
                f"[train_diffuser] Holding out {n_hold}/{n_total} samples for test "
                f"(slice={test_slice[0]}:{test_slice[1]}). Training uses {train_signals.shape[0]} samples."
            )
            # Replace signals used for preprocessing/training.
            signals = train_signals

        # Quantize in TIME DOMAIN first to create degraded (4-bit) and target waveforms.
        # If raw is already at `real_bits`, we skip quantizing the target waveform.
        assume_raw_is_real_bits = bool(config.get('assume_raw_is_real_bits', False))
        if assume_raw_is_real_bits:
            print(
                f"[train_diffuser] Time-domain: quantize condition to {cond_bits}-bit; "
                f"target uses raw (assumed {real_bits}-bit)"
            )
        else:
            print(f"[train_diffuser] Time-domain quantization: {cond_bits}-bit condition, {real_bits}-bit target")
        time_quant_mode = str(config.get('time_quantization_mode') or 'global').strip().lower()
        if time_quant_mode not in {'per_sample', 'global'}:
            print(f"[train_diffuser] Warning: unknown time_quantization_mode={time_quant_mode!r}; defaulting to per_sample")
            time_quant_mode = 'per_sample'

        lower_pct = float(config.get('quantile_clip_lower', 0.0))
        upper_pct = float(config.get('quantile_clip_upper', 100.0))

        if time_quant_mode == 'per_sample':
            print(f"[train_diffuser] Time-domain quantization mode: per_sample (pct={lower_pct:g}-{upper_pct:g})")
            peak = _per_sample_symmetric_peak(signals, lower_pct=lower_pct, upper_pct=upper_pct)  # [N,1]
            cond_time = _uniform_quantize_per_sample(signals, bits=int(cond_bits), peak=peak)
            real_time = signals if assume_raw_is_real_bits else _uniform_quantize_per_sample(signals, bits=int(real_bits), peak=peak)

            # For debug printing we still build a representative quantizer from sample_0 only.
            try:
                from diffusion.utils.quant_debug import summarize_tensor, summarize_uniform_quantizer, summarize_quantization_usage

                peak0 = float(peak[0].item())
                q0 = UniformQuantizer(bits=int(cond_bits), range_min=-peak0, range_max=peak0)
                q0._meta = {'source': 'per_sample', 'sample': 0, 'peak': peak0, 'lower_pct': lower_pct, 'upper_pct': upper_pct}
                print("[train_diffuser] Time-domain quant diagnostics (sample_0)")
                summarize_tensor(signals[:1], "raw_time(sample_0)")
                summarize_uniform_quantizer(q0, f"cond_time_q{int(cond_bits)}(sample_0)")
                summarize_quantization_usage(signals[:1], q0, f"cond_time_q{int(cond_bits)}_usage(sample_0)")
            except Exception as e:
                print(f"[train_diffuser] Quant diagnostics failed (non-fatal): {e}")

            # No single global range exists in this mode.
            cond_time_quantizer = None
            real_time_quantizer = None
        else:
            print(f"[train_diffuser] Time-domain quantization mode: global (pct={lower_pct:g}-{upper_pct:g})")
            cond_time_quantizer = create_time_quantizer(config, signals, bits=cond_bits)
            real_time_quantizer = create_time_quantizer(config, signals, bits=real_bits)

            # Debug prints for quantization consistency.
            try:
                from diffusion.utils.quant_debug import (
                    summarize_tensor,
                    summarize_uniform_quantizer,
                    summarize_quantization_usage,
                )

                print("[train_diffuser] Time-domain quantization diagnostics (train split)")
                summarize_tensor(signals[:1], "raw_time(sample_0)")
                summarize_uniform_quantizer(cond_time_quantizer, f"cond_time_q{int(cond_bits)}")
                summarize_quantization_usage(signals[:1], cond_time_quantizer, f"cond_time_q{int(cond_bits)}_usage(sample_0)")
                summarize_uniform_quantizer(real_time_quantizer, f"real_time_q{int(real_bits)}")
                summarize_quantization_usage(signals[:1], real_time_quantizer, f"real_time_q{int(real_bits)}_usage(sample_0)")
            except Exception as e:
                print(f"[train_diffuser] Quant diagnostics failed (non-fatal): {e}")

            # Build simulated waveforms
            cond_time = cond_time_quantizer.quantize(signals)
            real_time = signals if assume_raw_is_real_bits else real_time_quantizer.quantize(signals)

        # Apply transform (STFT) to both waveforms to get matrices (spectrogram magnitudes)
        transform_cond = create_transform(config, device)
        transform_real = create_transform(config, device)

        cond_mag = transform_cond.apply(cond_time)
        real_mag = transform_real.apply(real_time)
        print(f"[train_diffuser] After transform shapes: cond {cond_mag.shape}, real {real_mag.shape}")

        cond_data = cond_mag
        real_data = real_mag
        
    
        if cond_data.ndim == 3:
            cond_data = cond_data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        if real_data.ndim == 3:
            real_data = real_data.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
        
        print(f"[train_diffuser] Condition data shape: {cond_data.shape}")
        print(f"[train_diffuser] Real data shape: {real_data.shape}")
        
        if s3_data_uri is not None:
            from diffusion.aws.s3_io import s3_object_exists

            if (not force_preprocess) and s3_object_exists(cond_data_s3):
                print(f"[train_diffuser] Condition dataset already exists in S3; skipping upload: {cond_data_s3}")
            else:
                print(f"[train_diffuser] Uploading condition spectrograms to: {cond_data_s3}")
                _s3_put_torch({'signals': cond_data.to(torch.float32).cpu()}, cond_data_s3)

            if (not force_preprocess) and s3_object_exists(real_data_s3):
                print(f"[train_diffuser] Real dataset already exists in S3; skipping upload: {real_data_s3}")
            else:
                print(f"[train_diffuser] Uploading real spectrograms to: {real_data_s3}")
                _s3_put_torch({'signals': real_data.to(torch.float32).cpu()}, real_data_s3)

            print(f"[train_diffuser] Dataset S3 check/upload complete")
        else:
            print(f"[train_diffuser] Saving condition spectrograms to: {cond_data_path}")
            torch.save({'signals': cond_data.to(torch.float32).cpu()}, cond_data_path)
            print(f"[train_diffuser] Saving real spectrograms to: {real_data_path}")
            torch.save({'signals': real_data.to(torch.float32).cpu()}, real_data_path)
            print(f"[train_diffuser] Saved both datasets")
    else:
        if s3_data_uri is not None and not force_preprocess:
            raise ValueError(
                "Raw data is not available locally, and this run is configured to load preprocessed tensors from S3. "
                "Prepare and upload tensors first (use --prepare-s3-data), then rerun training. "
                f"raw_data_path={raw_data_path}"
            )
        raise ValueError(f"Raw data not found at {raw_data_path}")
    
    print(f"[train_diffuser] Computing magnitude ranges (full dataset)")
    cond_mag_min = float(cond_data.min().item())
    cond_mag_max = float(cond_data.max().item())
    real_mag_min = float(real_data.min().item())
    real_mag_max = float(real_data.max().item())

    # Limit dataset size for faster local testing (training subset only)
    max_samples = config.get('max_samples')
    if max_samples and max_samples < len(cond_data):
        print(f"[train_diffuser] Using subset for training: {max_samples}/{len(cond_data)} samples")
        indices = torch.randperm(len(cond_data))[:max_samples]
        cond_data = cond_data[indices]
        real_data = real_data[indices]

    # Limit to specific number of batches for quick testing (training subset only)
    batch_size = config.get('batch_size', 16)
    max_batches = config.get('max_batches')
    if max_batches:
        max_samples_for_batches = max_batches * batch_size
        if max_samples_for_batches < len(cond_data):
            print(f"[train_diffuser] Limiting training to {max_batches} batches ({max_samples_for_batches} samples)")
            cond_data = cond_data[:max_samples_for_batches]
            real_data = real_data[:max_samples_for_batches]

    # Normalize using ONLY condition-derived ranges (deployment-aligned).
    #
    # For each sample i, compute cond_min[i], cond_max[i] over (H,W), then:
    #   cond_norm = (cond - cond_min) / (cond_max-cond_min) -> [-1, 1]
    #   real_norm = (real - cond_min) / (cond_max-cond_min) -> may exceed [-1, 1]
    print(f"[train_diffuser] Normalizing spectrogram magnitudes using per-sample condition min/max")

    cond_data = cond_data.to(torch.float32)
    real_data = real_data.to(torch.float32)

    # Per-sample condition ranges: shape [N, 1, 1, 1]
    cond_min = cond_data.amin(dim=(2, 3), keepdim=True)
    cond_max = cond_data.amax(dim=(2, 3), keepdim=True)
    denom = cond_max - cond_min
    denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)

    cond_data = (cond_data - cond_min) / denom
    cond_data = cond_data * 2.0 - 1.0
    real_data = (real_data - cond_min) / denom
    real_data = real_data * 2.0 - 1.0
    
    # Clip to [-1, 1] to prevent numerical instability
    cond_data = torch.clamp(cond_data, -1.0, 1.0)
    real_data = torch.clamp(real_data, -1.0, 1.0)

    # Cast tensors to specified dtype 
    train_dtype = _resolve_torch_dtype(config, device)
    cond_data = cond_data.to(dtype=train_dtype)
    real_data = real_data.to(dtype=train_dtype)

    print(f"  - Condition mag range (full): [{cond_mag_min:.6f}, {cond_mag_max:.6f}]")
    print(f"  - Real mag range (full):      [{real_mag_min:.6f}, {real_mag_max:.6f}]")
    print(f"  - Normalized condition range: [{cond_data.min():.4f}, {cond_data.max():.4f}]")
    print(f"  - Normalized real range:      [{real_data.min():.4f}, {real_data.max():.4f}]")
    
    normalizer_params = {
        'time_quantization_mode': str(config.get('time_quantization_mode') or 'global'),
        'cond_time_range_min': (cond_time_quantizer.range_min if cond_time_quantizer is not None else None),
        'cond_time_range_max': (cond_time_quantizer.range_max if cond_time_quantizer is not None else None),
        'real_time_range_min': (real_time_quantizer.range_min if real_time_quantizer is not None else None),
        'real_time_range_max': (real_time_quantizer.range_max if real_time_quantizer is not None else None),
        'time_quant_clip_lower': (getattr(cond_time_quantizer, '_meta', {}).get('lower_pct') if cond_time_quantizer is not None else None),
        'time_quant_clip_upper': (getattr(cond_time_quantizer, '_meta', {}).get('upper_pct') if cond_time_quantizer is not None else None),
        'cond_mag_min': cond_mag_min,
        'cond_mag_max': cond_mag_max,
        'real_mag_min': real_mag_min,
        'real_mag_max': real_mag_max,
        'mag_normalization': 'condition_per_sample',
        'test_holdout_count': int(test_holdout_count),
        'test_holdout_from_end': bool(test_holdout_from_end),
        'cond_bits': int(cond_bits),
        'real_bits': int(real_bits),
        'transform_type': transform_type,
        'pipeline_config': config.get('pipeline_config'),
    }

    # Always save a per-run copy of normalizer_params locally (inside results/data)
    # so samplers/analysis can remain consistent even when the canonical copy lives in S3.
    try:
        if data_dir is not None:
            data_dir.mkdir(parents=True, exist_ok=True)
            normalizer_path = data_dir / 'normalizer_params.pkl'
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer_params, f)
            print(f"[train_diffuser] Normalizer params saved to: {normalizer_path}")
    except Exception as e:
        print(f"[train_diffuser] Warning: could not save local normalizer_params.pkl (non-fatal): {e}")

    if s3_data_uri is not None:
        from diffusion.aws.s3_io import s3_object_exists

        if (not force_preprocess) and s3_object_exists(normalizer_s3):
            print(f"[train_diffuser] Normalizer params already exist in S3; skipping upload: {normalizer_s3}")
        else:
            # Upload params as a small pickle blob.
            buf = io.BytesIO()
            pickle.dump(normalizer_params, buf)
            _s3_put_bytes(buf.getvalue(), normalizer_s3)
            print(f"[train_diffuser] Normalizer params uploaded to: {normalizer_s3}")
    else:
        # In non-S3 mode we already saved above; keep behavior but don't double-print.
        pass
    return cond_data, real_data, checkpoint_dir, samples_dir, logs_dir, epochs




def train_diffuser(real_data,cond_data, checkpoint_dir, samples_dir, logs_dir, epochs, config: dict):
    """
    Trains a diffusionmodel using the config and the data needed
    args:
        data: data to be used for training
        config: configuration dictionary
    
    """

    # Print a definitive version stamp to check for stale code
    # Note: `config` is a dict (from DiffusionConfig.to_dict())
    config_version = config.get("__version__", "N/A")
    print(f"[train_diffuser] *** CONFIG VERSION: {config_version} ***")

    # Build diffuser and train
    diffuser, optimizer, lr_scheduler = build_diffuser(config)
    
    from diffusion.utils.trainer import DiffusionTrainer
    from diffusion.utils.diffusion_models import get_lr_scheduler
    
    # Calculate total training steps for lr scheduler
    batch_size = config.get('batch_size', 16)
    num_samples = len(cond_data)
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
 
    warmup_steps = config.get('lr_warmup_steps')
    if warmup_steps is None:
        warmup_steps = steps_per_epoch
    print(f"[train_diffuser] LR warmup: {warmup_steps} steps ({warmup_steps/steps_per_epoch:.1f} epochs)")
    config['lr_warmup_steps'] = warmup_steps

    lr_scheduler = get_lr_scheduler(optimizer, config, total_steps)
    
    trainer = DiffusionTrainer(
        model=diffuser,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoints_dir=checkpoint_dir,
        samples_dir=samples_dir,
        logs_dir=logs_dir,
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        mixed_precision=config.get('mixed_precision'),
        validation_split=config.get('validation_split', 0.1),
        save_every_n_epochs=config.get('save_every_n_epochs', 1),
        validate_every_n_epochs=config.get('validate_every_n_epochs', 1),
        log_advanced_metrics=config.get('log_advanced_metrics', True),
        advanced_metrics_every_n_steps=config.get('advanced_metrics_every_n_steps', 50),
        energy_curve_every_n_steps=config.get('energy_curve_every_n_steps', 200),
    )
    
    trainer.fit(
        cond_data=cond_data,
        real_data=real_data,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=config.get('num_workers', 0),
        resume_from=config.get('resume_from'),
    )
    print(f"[train_diffuser] Outputs saved to:")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Samples:     {samples_dir}")
    print(f"  - Logs:        {logs_dir}")



if __name__ == '__main__':
    from diffusion.utils.config import DiffusionConfig
    config = DiffusionConfig.to_dict()
    train_diffuser(config)
