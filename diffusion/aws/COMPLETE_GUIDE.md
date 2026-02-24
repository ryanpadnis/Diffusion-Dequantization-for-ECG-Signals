# Complete Guide: AWS/Anyscale Training & Sampling

Full guide for running diffusion model training and sampling on Anyscale GPU workers with S3 storage.

## Quick Start

**Training:**
```bash
# Uses default Anyscale S3 bucket
uv run python -m diffusion.aws.anyscale_train \
  --workspace EE269-spot2 \
  --region us-east-1

# Or specify custom S3 bucket
uv run python -m diffusion.aws.anyscale_train \
  --workspace EE269-spot2 \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1
```

**Sampling:**
```bash
# Uses default Anyscale S3 bucket
uv run python -m diffusion.aws.anyscale_sample \
  --workspace EE269-spot2 \
  --region us-east-1 \
  --run-id 20260204_215551 \
  --checkpoint best_model.pt \
  --num-samples 100
```

**Important:** Press **Ctrl+C** to safely stop training/sampling. The script will:
1. Stop the local process
2. Automatically stop remote Ray jobs on the cluster
3. Clean up gracefully

Press Ctrl+C twice to force quit (remote jobs may continue running).

## Prerequisites

### 1. AWS Account Setup

1. **Create IAM user** with programmatic access
2. **Get credentials**: Access Key ID + Secret Access Key
3. **Required permissions**:
   - EC2: Launch instances, manage security groups
   - S3: Read/write to your bucket
   - IAM: PassRole for instance profiles

### 2. Install AWS CLI

```bash
brew install awscli
aws configure
# Enter: Access Key ID, Secret Access Key, us-east-1, json
```

Verify:
```bash
aws sts get-caller-identity
```

### 3. Create S3 Bucket

```bash
aws s3 mb s3://YOUR_BUCKET --region us-east-1
```

### 4. Install Anyscale CLI

```bash
uv pip install anyscale
# or: pip install anyscale
```

## Anyscale Setup

### Step 1: Create Anyscale Cloud

Register your AWS account with Anyscale (one-time):

```bash
uv run anyscale cloud register \
  --provider aws \
  --region us-east-1 \
  --name EE269 \
  --vpc-id vpc-XXXXX \
  --subnet-ids subnet-XXXXX,subnet-YYYYY \
  --security-group-ids sg-XXXXX \
  --anyscale-iam-role-id arn:aws:iam::ACCOUNT:role/anyscale-cluster-role \
  --instance-iam-role-id arn:aws:iam::ACCOUNT:role/anyscale-node-role \
  --cloud-storage-bucket-name YOUR_BUCKET \
  --cloud-storage-bucket-region us-east-1 \
  --functional-verify workspace \
  --yes
```

**Note**: Anyscale provides guided setup via their console that creates these resources automatically.

### Step 2: Create Compute Config

Edit `diffusion/aws/anyscale_compute_config_gpu_use1.yaml`:

```yaml
cloud: EE269

head_node:
  instance_type: m5.xlarge  # CPU-only head node

worker_nodes:
  - name: gpu-worker
    instance_type: g6.xlarge  # 1× L4 GPU (~$0.25/hr spot)
    min_nodes: 0               # Auto-scales from 0
    max_nodes: 1
    market_type: SPOT          # Use SPOT for cost savings
```

**GPU Options** (us-east-1):
- `g4dn.xlarge`: 1× T4, 4 vCPU, ~$0.15/hr spot (slower)
- `g5.xlarge`: 1× A10G, 4 vCPU, ~$0.35/hr spot (3x faster)
- `g6.xlarge`: 1× L4, 4 vCPU, ~$0.25/hr spot (2.5x faster) **← Recommended**

Create compute config:
```bash
uv run anyscale compute-config create \
  -n ee269-gpu-use1 \
  -f diffusion/aws/anyscale_compute_config_gpu_use1.yaml
```

### Step 3: Create Workspace

```bash
uv run anyscale workspace_v2 create \
  --name EE269-spot2 \
  --cloud EE269 \
  --compute-config ee269-gpu-use1
```

### Step 4: Start Workspace

```bash
uv run anyscale workspace_v2 start --name EE269-spot2
uv run anyscale workspace_v2 wait --name EE269-spot2 --state RUNNING
```

**Verify GPU availability:**
```bash
uv run anyscale workspace_v2 run_command --name EE269-spot2 'nvidia-smi'
```

## Training

### Run Training

The script automatically:
- Checks if preprocessed data exists in S3
- Prepares data if missing (one-time)
- Installs minimal deps on GPU worker
- Runs training with periodic S3 sync
- Saves checkpoints, logs, and samples to S3

```bash
uv run python -m diffusion.aws.anyscale_train \
  --workspace EE269-spot2 \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1
```

**Common options:**
```bash
# Override epochs/batches for quick test
--epochs 5 --max-batches 10

# Use different GPU count
--ray-num-gpus 1.0

# Adjust S3 sync interval (default: 30s)
--s3-sync-interval 60.0

# Create run bundle (tarball of all artifacts)
--bundle-run
```

### Training Configuration

Edit `diffusion/utils/config.py`:

```python
# Data
test_holdout_count = 500        # Test set size
batch_size = 16                 # Batch size
epochs = 30                     # Total epochs

# Model
torch_dtype = "bfloat16"        # Use bfloat16 (prevents NaN)
mixed_precision = "bf16"        # Accelerator mixed precision

# Training
learning_rate = 1e-4
max_samples = None              # None = use all data
max_batches = None              # None = full epoch
```

### Monitor Training

Training prints progress in your terminal:
```
Epoch 1/30 [Train]:  24%|██▍| 22/90 [01:29<04:37, 4.08s/it, loss=0.6758, lr=2.20e-05]
```

**Terminate training properly:**
- **Press Ctrl+C once**: Stops local script AND stops remote Ray jobs automatically
- **Press Ctrl+C twice**: Force quit (remote jobs may continue)

The script now handles interrupts gracefully and cleans up remote jobs for you!

### Check S3 Artifacts

After training:
```bash
aws s3 ls s3://YOUR_BUCKET/ee269project/diffusion-results/V1/ --recursive
```

Structure:
```
diffusion-results/V1/20260204_215551/
  ├── checkpoints/
  │   ├── best_model.pt
  │   ├── checkpoint_epoch_1.pt
  │   └── checkpoint_epoch_30.pt
  ├── logs/
  │   └── training.log
  ├── samples/
  │   ├── epoch_1_samples.png
  │   └── epoch_30_samples.png
  └── config.pkl
```

## Sampling

### Generate Samples from Trained Model

```bash
uv run python -m diffusion.aws.anyscale_sample \
  --workspace EE269-spot2 \
  --s3 s3://YOUR_BUCKET/ee269project \
  --region us-east-1 \
  --run-id 20260204_215551 \
  --checkpoint best_model.pt \
  --num-samples 100
```

**Options:**
- `--run-id`: Training run timestamp (see S3 diffusion-results/V1/)
- `--checkpoint`: Checkpoint file (best_model.pt, checkpoint_epoch_N.pt)
- `--num-samples`: Number of samples to generate

Samples are uploaded to:
```
s3://YOUR_BUCKET/ee269project/diffusion-results/V1/20260204_215551/samples/generated_samples.pt
```

## Cost Management

### Understanding Costs

**Compute (per-second billing, 60s minimum):**
- Head node (m5.xlarge): ~$0.10/hr (runs continuously while workspace is RUNNING)
- GPU worker (g6.xlarge): ~$0.25/hr spot (only during training/sampling jobs)
- GPU worker auto-terminates when idle (`min_nodes: 0`)

**Storage:**
- S3: ~$0.023/GB/month (negligible for datasets <10GB)

**Example 30-epoch training:**
- Training time: ~40 minutes on L4 GPU
- Head node: 40 min × $0.10/hr = $0.07
- GPU worker: 40 min × $0.25/hr = $0.17
- **Total: ~$0.24**

### View Costs

**Anyscale Console:**
```bash
open https://console.anyscale.com/organization/usage
```

**AWS Cost Explorer:**
```bash
# Current month costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --group-by Type=SERVICE

# Or open AWS Console
open https://us-east-1.console.aws.amazon.com/costmanagement/home#/dashboard
```

### Minimize Costs

1. **Terminate workspace when done:**
   ```bash
   uv run anyscale workspace_v2 terminate --name EE269-spot2 --yes
   ```

2. **Use spot instances** (already configured in compute config)

3. **Clean up old S3 runs:**
   ```bash
   # List runs
   aws s3 ls s3://YOUR_BUCKET/ee269project/diffusion-results/V1/
   
   # Delete specific run
   aws s3 rm s3://YOUR_BUCKET/ee269project/diffusion-results/V1/20260204_215551/ --recursive
   ```

4. **Reduce S3 sync frequency:**
   ```bash
   --s3-sync-interval 120.0  # Sync every 2 minutes instead of 30s
   ```

## Troubleshooting

### Spot Capacity Issues

**Error:** `MaxSpotInstanceCountExceeded` or workspace stuck in STARTING

**Solution 1: Wait and retry** (spot capacity changes frequently)
```bash
uv run anyscale workspace_v2 terminate --name EE269-spot2 --yes
# Wait 5-10 minutes
uv run anyscale workspace_v2 start --name EE269-spot2
```

**Solution 2: Switch to on-demand** (edit compute config)
```yaml
market_type: ON_DEMAND  # Instead of SPOT
```

**Solution 3: Try different region** (create new cloud in us-west-2)

### Check Spot Quota

```bash
uv run aws service-quotas get-service-quota \
  --region us-east-1 \
  --service-code ec2 \
  --quota-code L-34B43A08 \
  --query 'Quota.Value' \
  --output text
```

g6.xlarge requires 4 vCPU. If quota shows 4 or less, request increase:
```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-34B43A08 \
  --desired-value 16 \
  --region us-east-1
```

### NaN Losses

**Symptom:** Loss becomes NaN during training

**Causes:**
1. Using `torch_dtype = "float16"` (insufficient precision)
2. Data outside [-1, 1] range

**Solution:** Already fixed in config
```python
torch_dtype = "bfloat16"     # More stable than fp16
mixed_precision = "bf16"      # Use bf16 mixed precision
# Data is clamped to [-1, 1] in runner.py
```

### Pin Memory Error

**Error:** `cannot pin 'torch.cuda.HalfTensor' only dense CPU tensors can be pinned`

**Cause:** Data already on CUDA device

**Solution:** Already fixed in trainer.py
```python
use_pin_memory = cond_data.device.type == 'cpu'
```

### Dependencies Not Found

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Cause:** GPU worker doesn't have dependencies

**Solution:** Script automatically installs deps. If it fails:
```bash
# SSH into workspace
uv run anyscale workspace_v2 ssh --name EE269-spot2

# Manually install
python -m pip install -r diffusion/aws/requirements_train_minimal.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu118
```

### S3 Access Denied

**Error:** `botocore.exceptions.ClientError: An error occurred (AccessDenied)`

**Cause:** Missing S3 permissions

**Solution:** Add policy to instance role:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR_BUCKET",
        "arn:aws:s3:::YOUR_BUCKET/*"
      ]
    }
  ]
}
```

### Ray Job Still Running After Ctrl+C

**Issue:** Local script exits but remote training continues

**Solution:** Explicitly stop the job
```bash
# List jobs
uv run anyscale workspace_v2 run_command --name EE269-spot2 'ray job list'

# Stop specific job
uv run anyscale workspace_v2 run_command --name EE269-spot2 'ray job stop <job_id>'

# Or terminate entire workspace
uv run anyscale workspace_v2 terminate --name EE269-spot2 --yes
```

## Architecture

### Data Flow

1. **Local preprocessing** (one-time):
   - Raw data → Quantize → STFT → Normalize → Save to S3
   - Cached at: `s3://YOUR_BUCKET/ee269project/data/V1/`

2. **Training** (on GPU worker):
   - Load preprocessed tensors from S3
   - Train diffusion model
   - Sync checkpoints/logs/samples to S3 every 30s

3. **Sampling** (on GPU worker):
   - Load checkpoint from S3
   - Generate samples
   - Upload samples to S3

### File Structure

```
diffusion/
├── aws/
│   ├── anyscale_train.py           # Training launcher
│   ├── anyscale_sample.py          # Sampling launcher
│   ├── anyscale_compute_config_gpu_use1.yaml  # Compute config
│   ├── requirements_train_minimal.txt  # Minimal deps for workers
│   └── COMPLETE_GUIDE.md           # This file
├── utils/
│   ├── config.py                   # Training configuration
│   ├── runner.py                   # Preprocessing + training orchestration
│   ├── trainer.py                  # DiffusionTrainer class
│   └── diffusion_models.py        # UNet + scheduler
├── train/
│   ├── ray_train.py                # Ray training entrypoint
│   └── ray_sample.py               # Ray sampling entrypoint (TBD)
└── sample/
    └── sampler.py                  # Sampling utilities
```

### Key Configuration Files

**Training config:** `diffusion/utils/config.py`
- Data parameters (batch_size, holdout_count)
- Model architecture (unet_type, num_noising_steps)
- Training hyperparameters (learning_rate, epochs)
- Quantization/normalization settings

**Compute config:** `diffusion/aws/anyscale_compute_config_gpu_use1.yaml`
- Instance types (head + workers)
- Autoscaling (min/max nodes)
- Market type (SPOT vs ON_DEMAND)

**Dependencies:** `diffusion/aws/requirements_train_minimal.txt`
- Minimal training deps (no EDA/data tools)
- torch installed separately via PyTorch index

## Advanced Usage

### Multi-GPU Training

Edit compute config for multiple GPUs:
```yaml
worker_nodes:
  - name: gpu-worker
    instance_type: g6.12xlarge  # 4× L4 GPUs
    min_nodes: 0
    max_nodes: 1
    market_type: SPOT
```

Launch with more GPUs:
```bash
--ray-num-gpus 4.0
```

### Custom PyTorch Version

```bash
uv run python -m diffusion.aws.anyscale_train \
  --torch torch==2.1.0 \
  --torch-index-url https://download.pytorch.org/whl/cu121 \
  ...
```

### SSH Access to Workspace

```bash
uv run anyscale workspace_v2 ssh --name EE269-spot2

# Inside workspace:
nvidia-smi
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
uv run python -m diffusion.train.ray_train --ray-address auto --ray-num-gpus 1 --s3 s3://YOUR_BUCKET/ee269project
```

### Download S3 Artifacts Locally

```bash
# Download entire run
aws s3 sync s3://YOUR_BUCKET/ee269project/diffusion-results/V1/20260204_215551/ \
  ~/Downloads/run_20260204_215551/

# Download specific checkpoint
aws s3 cp s3://YOUR_BUCKET/ee269project/diffusion-results/V1/20260204_215551/checkpoints/best_model.pt \
  ~/Downloads/
```

### Monitor Training via Anyscale Console

```bash
open https://console.anyscale.com/workspaces
```

Click your workspace → View Ray Dashboard → Jobs/Metrics

## Summary of Key Commands

```bash
# Setup (one-time)
aws configure
uv run anyscale cloud register ...
uv run anyscale compute-config create ...
uv run anyscale workspace_v2 create ...

# Start workspace
uv run anyscale workspace_v2 start --name EE269-spot2

# Train (uses default Anyscale S3 bucket)
uv run python -m diffusion.aws.anyscale_train \
  --workspace EE269-spot2 \
  --region us-east-1

# Sample (uses default Anyscale S3 bucket)
uv run python -m diffusion.aws.anyscale_sample \
  --workspace EE269-spot2 \
  --region us-east-1 \
  --run-id 20260204_215551 \
  --checkpoint best_model.pt \
  --num-samples 100

# Download artifacts from S3 (uses default bucket, optionally delete)
uv run python -m diffusion.aws.download_s3_artifacts \
  --region us-east-1 \
  --run-id 20260217_234950_v2_lowfreqloss \
  --delete-after

# View tensorboard
uv run tensorboard --logdir diffusion/results/V1/20260217_234950/logs
/Users/ryanpadnis/EE269Project/diffusion/results/V2/20260217_234950
# Stop workspace (to save costs)
uv run anyscale workspace_v2 terminate --name EE269-spot2 --yes

# Check costs
open https://console.anyscale.com/organization/usage
```

## Additional Guides

- **[Tensorboard & S3 Artifacts](TENSORBOARD_S3_GUIDE.md)** - Tensorboard setup and S3 download/cleanup scripts
- **[Original AWS/S3 Setup](README.md)** - Legacy setup documentation

## Getting Help

- **Anyscale Docs:** https://docs.anyscale.com
- **Ray Docs:** https://docs.ray.io
- **AWS CLI Docs:** https://awscli.amazonaws.com/v2/documentation/api/latest/index.html
- **Project Issues:** Contact maintainer
