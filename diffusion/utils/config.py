"""
Docstring for diffusion.utils.config
"""

import os
import diffusion.settings as settings
from sklearn.pipeline import Pipeline

class DiffusionConfig:
    version = "V1"

    #immutaable project paths
    diffusion_root = settings.DIFFUSION_ROOT
    project_root = settings.PROJECT_ROOT
    data_root = settings.PROJECT_ROOT / "data"
    raw_data_dir = data_root / "data" / "raw"

    raw_data_path = raw_data_dir / "art_chunks.pt"
    
    #version specific
    checkpoint_dir = diffusion_root / version / "checkpoints"
    logs_dir = diffusion_root / version / "logs"
    samples_dir = diffusion_root / version / "samples"
    data_dir = diffusion_root / version / "data"
    data_path = data_dir / "train_data.pt"

    #Diffusion Model Parameters
    unet_type = "UNet2DConditional"
    diffusion_type = "Gaussian"
    noise_schedule = "linear"
    num_noising_steps = 1000
    sampler_type = "ddpm"
    image_size = (16, 128)  # Example size (height, width)
    in_channels = 1 #num features
    out_channels = 1

    #conditioning
    bit_size = 4
    quantizer_type = "uniform"
    transform_type = "stft"
    #dictionery for the  data pipeline
    pipeline_config = {
        "transform": "stft",
        "n_fft": 30,
        "hop_length": 64,
        "onesided": True,
    }
    #tuning parameters
    learning_rate = 1e-4
    batch_size = 16
    num_epochs = 100
    save_checkpoint_every = 10
    validate_every = 5
    validation_split = 0.1
    random_seed = 42

