# Tensorboard & S3 Artifact Management

## Tensorboard Logging

### Already Configured

Tensorboard logging is **already enabled** in the trainer:
- Logs are written to `logs_dir/diffusion_training/` (configured in training)
- Logs are automatically flushed after each epoch and at training completion
- Metrics logged: train/val loss, learning rate, memory stats, advanced metrics
- Uses Accelerate's built-in tensorboard integration
- Compatible with S3 syncing (logs sync to S3 every 30s during training)

### Verify Tensorboard During Training

Training will print the tensorboard logs location:
```
[Trainer] Tensorboard logs: /path/to/logs/diffusion_training
```

On remote GPU worker, logs are synced to S3:
```
s3://YOUR_BUCKET/ee269project/diffusion-results/V1/20260204_215551/logs/
```

### View Tensorboard Locally

After downloading logs from S3:

```bash
# Download your run
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project \
  --region us-east-1 \
  --run-id 20260212_211925

# Start tensorboard
uv run tensorboard --logdir  s3://anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project/diffusion-results/V1/20260212_211925/logs/20260212_211925/diffusion_training

# Open browser
open http://localhost:6006
```

### View Tensorboard from S3 (without downloading)

```bash
# Download only logs folder
aws s3 sync s3://YOUR_BUCKET/ee269project/diffusion-results/V1/20260204_215551/logs/ \
  ./temp_logs/

uv run tensorboard --logdir ./temp_logs
```

## S3 Artifact Management

### Download Artifacts

**Download specific run (keep in S3):**
```bash
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --run-id 20260204_215551
```

**Download and DELETE from S3:**
```bash
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --run-id 20260204_215551 \
  --delete-after
```

**Download ALL runs:**
```bash
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --all
```

**Download ALL runs and DELETE from S3:**
```bash
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --all \
  --delete-after
```

**Include preprocessed data cache:**
```bash
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --all \
  --include-data
```

**Dry run (preview what will happen):**
```bash
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --all \
  --delete-after \
  --dry-run
```

### Local Directory Structure

After download, files are stored in the same structure:

```
diffusion/results/V1/
├── 20260204_215551/
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   ├── checkpoint_epoch_1.pt
│   │   └── ...
│   ├── logs/
│   │   └── diffusion_training/
│   │       └── events.out.tfevents.*  # Tensorboard logs
│   ├── samples/
│   │   ├── epoch_1_samples.png
│   │   └── ...
│   └── config.pkl
└── 20260205_103045/
    └── ...
```

### Safety Features

1. **Confirmation prompt** when using `--delete-after` (unless `--dry-run`)
2. **Dry run mode** shows what would happen without making changes
3. **Batch deletion** for efficiency (up to 1000 files per request)
4. **Progress bars** for both download and deletion

### Options Reference

```
--s3 <uri>           S3 URI (required)
--region <region>    AWS region (default: us-east-1)
--run-id <id>        Download specific run
--all                Download all runs
--local-dir <path>   Custom local directory (default: diffusion/results/V1/)
--delete-after       Delete from S3 after download (DESTRUCTIVE!)
--dry-run            Preview without making changes
--include-data       Also download preprocessed data cache
```

## Workflow Examples

### After Training

```bash
# 1. Training completed, download everything and clean up S3
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --run-id 20260204_215551 \
  --delete-after

# 2. View tensorboard
uv run tensorboard --logdir diffusion/results/V1/20260204_215551/logs

# 3. Load checkpoint for sampling (if needed locally)
# Checkpoint is now at: diffusion/results/V1/20260204_215551/checkpoints/best_model.pt
```

### Keep S3, Download for Analysis

```bash
# Download without deleting
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --run-id 20260204_215551

# Analyze locally
uv run tensorboard --logdir diffusion/results/V1/20260204_215551/logs
```

### Clean Up Old Runs

```bash
# Preview what would be deleted
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --all \
  --delete-after \
  --dry-run

# If satisfied, run without --dry-run
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --all \
  --delete-after
```

### Backup Preprocessed Data

```bash
# Download preprocessed cache + all runs
uv run python -m diffusion.aws.download_s3_artifacts \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --all \
  --include-data \
  --delete-after
```

## Tensorboard Metrics

The following metrics are logged during training:

### Per Epoch
- `train/loss`: Training loss
- `train/learning_rate`: Current learning rate
- `train/epoch`: Epoch number
- `val/loss`: Validation loss (when validation runs)

### Memory Stats
- `memory/gpu_allocated_gb`: GPU memory allocated
- `memory/gpu_reserved_gb`: GPU memory reserved
- `memory/ram_used_gb`: System RAM used

### Advanced Metrics (optional, configurable)
- Batch statistics (mean, std, min, max, RMS)
- Energy curves (signal energy over time)
- Gradient statistics

### Configuration

Metrics are configured in `diffusion/utils/trainer.py`:

```python
DiffusionTrainer(
    log_advanced_metrics=True,           # Enable detailed metrics
    advanced_metrics_every_n_steps=50,   # Log batch stats every 50 steps
    energy_curve_every_n_steps=200,      # Log energy curves every 200 steps
)
```

## Troubleshooting

### Tensorboard not starting

```bash
# Install tensorboard
uv pip install tensorboard

# Or use project env
uv run tensorboard --logdir <path>
```

### No logs in tensorboard

Check that logs directory exists and has event files:
```bash
ls -R diffusion/results/V1/20260204_215551/logs/
```

Should see files like `events.out.tfevents.*`

### Download script fails

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Verify bucket access
aws s3 ls s3://YOUR_BUCKET/ee269project/

# Check region matches
aws configure get region
```

### S3 deletion fails

- Ensure IAM permissions include `s3:DeleteObject`
- Check bucket policies don't prevent deletion
- Use `--dry-run` first to verify
