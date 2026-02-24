# AWS / S3 setup (local + Anyscale)

This folder contains optional helpers for syncing run artifacts to S3.

## Sharing one run’s samples via GitHub

If you need to commit *one specific run’s* `diffusion/results/<version>/<run-id>/samples/**` to git (for teammates/analysis), use:
- [diffusion/aws/README_COMMIT_DIFFUSION_DATA.md](diffusion/aws/README_COMMIT_DIFFUSION_DATA.md)

It covers the repo’s `.gitignore` allowlist pattern + Git LFS setup for >100MB `.pt` files.

## Your bucket

This repo originally used an S3 bucket in `us-east-2` ("use2").

If you're switching to `us-east-1` ("use1"), create/use a bucket in `us-east-1` and use that region consistently in:
- `diffusion.aws.check_s3 --region ...`
- `diffusion.train.ray_train --s3-region ...`

Example (old):
- Bucket: `ee269--use2-az1--x-s3`
- Region: `us-east-2`

## Local machine setup (Mac)

**Goal:** make `boto3` able to authenticate so `--s3` uploads work.

You have two common ways to provide credentials:

### Option A: AWS CLI config files (recommended)

1) Install AWS CLI (if you don’t already have it):

- `brew install awscli`

2) Configure credentials + default region:

- `aws configure`
  - AWS Access Key ID: (from your IAM user)
  - AWS Secret Access Key: (from your IAM user)
  - Default region name: `us-east-1`
  - Default output format: `json`

This writes to `~/.aws/credentials` and `~/.aws/config`.

### Option B: Environment variables (temporary)

In your terminal:

- `export AWS_ACCESS_KEY_ID=...`
- `export AWS_SECRET_ACCESS_KEY=...`
- `export AWS_DEFAULT_REGION=us-east-2`

If you're using `us-east-1`, set:

- `export AWS_DEFAULT_REGION=us-east-1`

This only lasts for that shell session.

## Verify access (smoke test)

Run:

- `uv run python -m diffusion.aws.check_s3 --bucket ee269--use2-az1--x-s3 --region us-east-2`

If you're on `us-east-1`, use `--region us-east-1`.

This will:
- call STS `GetCallerIdentity`
- write a tiny test object under `s3://<bucket>/ee269project/smoke-test/...`
- delete it

## Upload training artifacts

Once credentials work:

Example:

- `uv run python -m diffusion.train.ray_train --s3 s3://YOUR_BUCKET/ee269project --s3-region us-east-1 --keep-last 3`

Notes:
- This uploads `checkpoints/`, `samples/`, `logs/`, and the derived `data/` artifacts (saved tensors/normalizer params).
- It does **not** upload your raw dataset (`data/data/raw`).
- Old runs are pruned under the S3 prefix (keep-last-N).

### Important: idempotent S3 data upload

When `s3_data_uri` is configured (as in `ray_train` when you pass `--s3`), preprocessing artifacts are uploaded to:

- `s3://ee269--use2-az1--x-s3/ee269project/data/V1/`

If these objects already exist, the code skips re-uploading them (unless `force_preprocess=True`). This avoids repeatedly uploading the large `train_*.pt` tensors.

### Important: faster artifact syncing

Artifact syncing to S3 is incremental: repeated syncs only upload changed files. If you want to avoid any periodic syncing overhead, set:

- `--s3-sync-interval 0`

## Anyscale (Ray on AWS) setup

If you don’t want to manage EC2 keys/AMIs/security groups yourself, Anyscale is usually the easiest way to run a GPU Ray cluster.

### Python / Ray requirement

Ray requires Python >= 3.10. This repo is configured to work with `uv` + Python 3.11.

If you need to set it up locally:

- `uv python install 3.11`
- `uv venv --python 3.11`
- `uv sync`

### What you need in AWS

You still need an IAM role Anyscale can use to create instances in your AWS account *and* your cluster nodes must be able to write to the S3 Express directory bucket.

- For node access to the directory bucket, use the policy template:
  - [diffusion/aws/iam_policy_ray_nodes.json](diffusion/aws/iam_policy_ray_nodes.json)
  - Key permission for directory buckets: `s3express:CreateSession`

### Steps (high level)

1) Create an Anyscale account + workspace.
2) In Anyscale, connect your AWS account (they provide a guided flow / CloudFormation).
3) Ensure the node role/instance profile used by your clusters includes the S3/S3 Express permissions above.
4) Configure the workspace compute to include a GPU *worker*.
  - Recommended cheapest 1-GPU worker in `us-east-2`: `g4dn.xlarge` (1x T4).
  - Keep head node CPU-only/cheap; don’t schedule work on the head.
5) Start the workspace and run training on the GPU worker.

### CLI workflow (recommended)

Assuming your workspace is named `EE269`.

If you are moving from `us-east-2` to `us-east-1`, you generally create a *new* Anyscale Cloud in `us-east-1`, then a compute config in that cloud, then a new workspace pointing at that cloud.

1) Register a new Anyscale Cloud in `us-east-1` (AWS IDs required):

- `uv run anyscale cloud register --provider aws --region us-east-1 --name EE269-use1 --vpc-id <vpc-...> --subnet-ids <subnet-...>,<subnet-...> --security-group-ids <sg-...> --anyscale-iam-role-id <arn:aws:iam::...:role/...> --instance-iam-role-id <arn:aws:iam::...:role/...> --cloud-storage-bucket-name s3://YOUR_BUCKET --cloud-storage-bucket-region us-east-1 --functional-verify workspace --yes`

2) Create a compute config in that cloud (includes 1 GPU worker):

- Edit `diffusion/aws/anyscale_compute_config_gpu_use1.yaml` (set `cloud:` to `EE269-use1`, and pick instance types)
- `uv run anyscale compute-config create -n ee269-gpu-use1 -f diffusion/aws/anyscale_compute_config_gpu_use1.yaml`

3) Create a workspace in that cloud using that compute config:

- `uv run anyscale workspace_v2 create --name EE269-use1 --cloud EE269-use1 --compute-config ee269-gpu-use1`

1) Start the workspace:

- `uv run anyscale workspace_v2 start --name EE269-use1`

2) Wait until it’s running:

- `uv run anyscale workspace_v2 wait --name EE269-use1 --state RUNNING`

3) Run sanity checks inside the workspace (no SSH needed):

- `uv run anyscale workspace_v2 run_command --name EE269-use1 'nvidia-smi'`
- `uv run anyscale workspace_v2 run_command --name EE269-use1 "python -c \"import ray; ray.init(address='auto'); print(ray.cluster_resources()); ray.shutdown()\""`

You should see `'GPU': 1.0` (or more) in `ray.cluster_resources()`.

4) Verify S3 access from the workspace nodes (directory-bucket permissions):

- `uv run anyscale workspace_v2 run_command --name EE269-use1 'uv run python -m diffusion.aws.check_s3 --bucket YOUR_BUCKET --region us-east-1'`

5) Run training on the GPU worker:

- `uv run anyscale workspace_v2 run_command --name EE269-use1 'uv run python -m diffusion.train.ray_train --ray-address auto --ray-num-gpus 1 --s3 s3://YOUR_BUCKET/ee269project --s3-region us-east-1 --keep-last 3'`

### Minimal worker installs (recommended)

On Anyscale, packages you install on the head node do **not** automatically exist on newly launched GPU workers.

If you rely on per-run bootstrapping, install only what training needs instead of `pip install -e .` (which pulls in EDA/data deps).

Inside the workspace:

- `python -m pip install -r diffusion/aws/requirements_train_minimal.txt`
- `python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu118`

Then run:

- `python -m diffusion.train.ray_train --ray-address auto --ray-num-gpus 1 --s3 s3://YOUR_BUCKET/ee269project --s3-region us-east-1 --bundle-run`

### Avoid per-run installs (recommended)

If you don’t want to wait for package installation every time a new GPU worker launches, bake the minimal deps into the workspace environment:

- Use [diffusion/aws/requirements_anyscale_gpu_cu128.txt](diffusion/aws/requirements_anyscale_gpu_cu128.txt) (updated to use cu118)

Create a new workspace with `--requirements`, or update an existing workspace (typically requires it to be TERMINATED):

- `uv run anyscale workspace_v2 update <WORKSPACE_ID> -r diffusion/aws/requirements_anyscale_gpu_cu128.txt`

This installs GPU-enabled PyTorch + only the deps needed for training/S3 sync (not the whole project).

### One-command launcher (recommended)

From your local terminal (runs `anyscale workspace_v2 run_command` under the hood):

- `uv run python -m diffusion.aws.anyscale_train --workspace EE269 --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --bundle-run`

This will:

- Start a 1-GPU Ray task to ensure minimal deps are installed on the GPU worker.
- Launch training with S3 syncing and optional `--bundle-run`.

### SSH workflow (alternative)

If you prefer an interactive shell:

- `uv run anyscale workspace_v2 ssh --name EE269`

If your workspace is named `EE269-use1`:

- `uv run anyscale workspace_v2 ssh --name EE269-use1`

Then (inside the workspace):

- `nvidia-smi`
- `python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources()); ray.shutdown()"`
- `uv run python -m diffusion.train.ray_train --ray-address auto --ray-num-gpus 1 --s3 s3://ee269--use2-az1--x-s3/ee269project --s3-region us-east-2 --keep-last 3`

Example:

- `uv run python -m diffusion.train.ray_train --ray-address auto --ray-num-gpus 1 --s3 s3://YOUR_BUCKET/ee269project --s3-region us-east-1 --keep-last 3`

### How to confirm it’s using the right workspace + GPU

- If you run without `--ray-address`, you are running locally (not Anyscale).
- If `ray.cluster_resources()` shows GPUs and `--ray-num-gpus 1` is set, the training task will run on a GPU node.
- `ray_train` prints the exact S3 run prefix at startup (this is the definitive “right bucket/path” check).

### Troubleshooting

#### EC2 error: “instance type is not eligible for Free Tier”

If the workspace/cluster logs show errors like:

- `InvalidParameterCombination: The specified instance type is not eligible for Free Tier ... free-tier-eligible=true`

then your AWS account/role is restricted to **Free Tier eligible** instance types. GPU instances like `g4dn.xlarge` are **not** Free Tier eligible, so you cannot start GPU workers (or GPU heads) until that restriction is removed.

What to do:

- If you control the AWS account: remove/relax the policy that constrains `ec2:RunInstances` to Free Tier eligible instance types (often an AWS Organizations SCP, permission boundary, or sandbox guardrail).
- If you don’t control it (class/sandbox account): you’ll need a different AWS account with GPU permissions, or accept CPU-only training.

To list Free Tier eligible instance types (CPU-only) in your region:

- `aws ec2 describe-instance-types --region us-east-2 --filters Name=free-tier-eligible,Values=true --query 'InstanceTypes[].InstanceType' --output text`

Notes:

- Spot vs on-demand does not fix Free Tier restrictions.
- A “GPU worker” requires a non-Free-Tier instance type, so GPU training implies paid usage (or credits).

If you paste the exact “AWS connection” and “cluster node role” fields Anyscale shows you (or a screenshot of the cluster config page), I can tell you exactly where to attach the policy and what to pick for the cheapest GPU option.

## S3 cost control helpers

If you want to minimize S3 storage charges, you can delete cached tensors and/or run artifacts under your project prefix.

- Delete cached preprocessed dataset (dry-run by default):
  - `uv run python -m diffusion.aws.cleanup_project_s3 --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --what data`
- Delete cached preprocessed dataset (actually delete):
  - `uv run python -m diffusion.aws.cleanup_project_s3 --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --what data --yes`
- Download an S3 prefix into your home directory:
  - `uv run python -m diffusion.aws.s3_to_home --prefix s3://YOUR_BUCKET/ee269project/data/V1 --region us-east-1 --dst ~/ee269project_s3_backup`
- Recreate (recompute+upload) the preprocessed cache after deletion:
  - `uv run python -m diffusion.aws.recreate_s3_data --s3 s3://YOUR_BUCKET/ee269project --region us-east-1`
