"""One-command launcher for training on an Anyscale workspace.

Runs locally (your laptop) and uses the Anyscale CLI to execute commands inside
an existing workspace.

Goals:
- Bootstrap only *minimal* training deps on the GPU worker (not the full project).
- Avoid repeated heavy installs by checking imports first.
- Launch training with S3 syncing + optional run bundle.

Example:
  uv run python -m diffusion.aws.anyscale_train \
    --workspace EE269 \
    --s3 s3://anyscale-production-data-.../ee269project \
    --region us-east-1 \
    --bundle-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import signal
import sys
import tarfile
import tempfile
import time
from typing import List, Optional


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--workspace", type=str, default="EE269-ondemand", help="Anyscale workspace name")
    p.add_argument("--region", type=str, default="us-east-1", help="AWS region (default: us-east-1)")
    p.add_argument(
        "--s3",
        type=str,
        default="s3://anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project",
        help="S3 bucket or prefix to use (default: Anyscale production bucket)",
    )
    p.add_argument("--ray-num-gpus", type=float, default=1.0, help="GPUs to request for training")
    p.add_argument("--s3-sync-interval", type=float, default=30.0, help="Periodic S3 sync interval (seconds)")
    p.add_argument("--bundle-run", action="store_true", help="Create+upload a single run bundle tarball")
    p.add_argument("--bundle-include-data", action="store_true", help="Include results data/ in the run bundle")
    p.add_argument("--force-preprocess", action="store_true", help="Force regenerate preprocessed data from raw dataset (uploads to S3)")
    p.add_argument("--preprocess-locally", action="store_true", help="Run preprocessing locally BEFORE training (forces regeneration with current config)")
    p.add_argument("--resume-latest", action="store_true", help="Resume training from the latest S3 run checkpoint")

    # S3 data is automatically checked and prepared if missing (no flag needed)

    # Optional passthroughs to shorten smoke tests.
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument(
        "--torch",
        type=str,
        default="torch==2.10.0",
        help="Torch spec to install on the GPU worker (default: torch==2.10.0)",
    )
    p.add_argument(
        "--torch-index-url",
        type=str,
        default="https://download.pytorch.org/whl/cu118",
        help="Index URL to use for torch wheels (default: PyTorch cu118 index)",
    )
    p.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config key passed to ray_train: KEY=VALUE (repeatable). "
            "Examples: version=V8, run_name=uniform__linear, quantizer_type=lloyd_max, noise_schedule_type=cosine."
        ),
    )

    # Sampling after training (runs on the same GPU node, same remote session).
    p.add_argument("--sample-after-train", action="store_true",
                   help="Run sampling immediately after training on the same cluster session.")
    p.add_argument("--sample-version", type=str, default=None,
                   help="Version path for sampling, e.g. V8/uniform__linear (default: derived from config overrides).")
    p.add_argument("--sample-checkpoint", type=str, default="best_model.pt")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--num-trajectories", type=int, default=1)

    return p.parse_args()


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _anyscale_cmd_prefix() -> List[str]:
    # Prefer using uv so we use the repo-local venv/lock.
    if _have("uv"):
        return ["uv", "run", "anyscale"]
    return ["anyscale"]


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, prefix) for s3://bucket[/optional/prefix]."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def _join_s3(bucket: str, *parts: str) -> str:
    key = "/".join([p.strip("/") for p in parts if str(p).strip("/")])
    return f"s3://{bucket}/{key}" if key else f"s3://{bucket}"


def _git_sha_short(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root))
        return out.decode("utf-8").strip()
    except Exception:
        return "nogit"


def _make_code_bundle(repo_root: Path) -> Path:
    """Create a small tar.gz of the repo (excluding big artifacts) and return its path."""
    excludes_dir_parts = {
        ".git",
        ".venv",
        "__pycache__",
        "ee269project.egg-info",
    }

    def should_skip(path: Path) -> bool:
        rel = path.relative_to(repo_root)
        parts = set(rel.parts)
        if parts & excludes_dir_parts:
            return True

        rel_posix = rel.as_posix()
        if rel_posix.startswith("diffusion/results/"):
            return True
        if rel_posix.startswith("data/data/raw/"):
            return True
        if rel_posix.startswith("data/data/processed/"):
            return True
        if rel.suffix in {".pt", ".pth", ".ckpt"}:
            return True
        return False

    tmpdir = Path(tempfile.mkdtemp(prefix="ee269_code_bundle_"))
    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sha = _git_sha_short(repo_root)

    # Content hash makes bundle keys stable-ish and helps debugging.
    h = hashlib.sha1()
    for p in sorted(repo_root.rglob("*.py")):
        if should_skip(p):
            continue
        try:
            h.update(p.read_bytes())
        except Exception:
            continue
    digest = h.hexdigest()[:10]

    bundle_path = tmpdir / f"code_{ts}_{sha}_{digest}.tar.gz"
    with tarfile.open(bundle_path, mode="w:gz") as tf:
        for path in repo_root.rglob("*"):
            if not path.is_file():
                continue
            if should_skip(path):
                continue
            arcname = path.relative_to(repo_root)
            tf.add(path, arcname=str(arcname))

    return bundle_path


def _upload_file_to_s3(*, local_path: Path, s3_uri: str, region: str) -> None:
    bucket, key = _parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"Expected full s3://bucket/key URI, got: {s3_uri!r}")
    try:
        import boto3
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "boto3 is required to upload the code bundle. Install with: uv pip install boto3"
        ) from e

    s3 = boto3.client("s3", region_name=str(region) if region else None)
    s3.upload_file(str(local_path), bucket, key)


def _remote_script(args: argparse.Namespace, *, code_bundle_s3_uri: str) -> str:
    # This string is executed *inside the workspace*.
    # It installs minimal deps on a GPU worker via a 1-GPU Ray task, then launches training.

    train_flags: List[str] = [
        "python",
        "-m",
        "diffusion.train.ray_train",
        "--ray-address",
        "auto",
        "--ray-num-gpus",
        str(float(args.ray_num_gpus)),
        "--s3",
        str(args.s3),
        "--s3-region",
        str(args.region),
        "--s3-sync-interval",
        str(float(args.s3_sync_interval)),
    ]

    if args.bundle_run:
        train_flags.append("--bundle-run")
    if args.bundle_include_data:
        train_flags.append("--bundle-include-data")
    # NOTE: We intentionally do NOT pass --force-preprocess into the remote Ray training.
    # Raw data is not bundled to the workspace, so remote force-preprocess is typically disabled
    # (or would fail). If the user wants regeneration, we run preprocessing BEFORE Ray on the
    # local machine via --preprocess-locally (triggered below in main).
    if args.resume_latest:
        train_flags.extend(["--resume-from-s3", "latest"])
    if args.epochs is not None:
        train_flags.extend(["--epochs", str(int(args.epochs))])
    if args.max_batches is not None:
        train_flags.extend(["--max-batches", str(int(args.max_batches))])
    if args.max_samples is not None:
        train_flags.extend(["--max-samples", str(int(args.max_samples))])
    for override in (args.config_override or []):
        train_flags.extend(["--config-override", override])

    train_cmd = " ".join(shlex.quote(x) for x in train_flags)

    # Build optional post-training sampling command (same GPU session).
    sample_block = ""
    if getattr(args, 'sample_after_train', False):
        # Derive version/run_name from config overrides.
        overrides_dict = {}
        for ov in (args.config_override or []):
            k, _, v = ov.partition("=")
            overrides_dict[k.strip()] = v.strip()
        version = getattr(args, 'sample_version', None) or (
            f"{overrides_dict.get('version', 'V8')}/{overrides_dict.get('run_name', 'run')}"
        )
        sample_flags = [
            "python", "-m", "diffusion.train.ray_sample",
            "--ray-address", "auto",
            "--ray-num-gpus", str(float(args.ray_num_gpus)),
            "--s3", str(args.s3),
            "--s3-region", str(args.region),
            "--version", version,
            "--checkpoint", str(getattr(args, 'sample_checkpoint', 'best_model.pt')),
            "--num-samples", str(int(getattr(args, 'num_samples', 16))),
            "--num-trajectories", str(int(getattr(args, 'num_trajectories', 1))),
        ]
        sample_cmd_str = " ".join(shlex.quote(x) for x in sample_flags)
        sample_block = f"""
# Resolve the run_id written by training and pass it to sampling.
RUN_ID=$(python - <<'PY'
import os, sys
try:
    from diffusion.aws.s3_io import normalize_s3_uri, join_s3_uri, list_run_prefixes
    s3_base = {str(args.s3)!r}
    version = {version!r}
    prefix = join_s3_uri(normalize_s3_uri(s3_base), 'diffusion-results', version)
    runs = sorted(list_run_prefixes(prefix), key=lambda x: x[1])
    if runs:
        print(runs[-1][0].rstrip('/').split('/')[-1])
    else:
        print('NOTFOUND', file=sys.stderr); sys.exit(1)
except Exception as e:
    print(f'ERROR: {{e}}', file=sys.stderr); sys.exit(1)
PY
)
echo "[anyscale_train] Sampling with run_id=$RUN_ID"
{sample_cmd_str} --run-id "$RUN_ID"
"""

    # Use a bootstrap task to ensure the worker has deps. We do a quick import test first.
    # Note: this installs into the worker's Python environment; it persists for the node lifetime.
    torch_spec_py = repr(str(args.torch))
    torch_index_url_py = repr(str(args.torch_index_url))

    script = f"""set -e

export AWS_DEFAULT_REGION={shlex.quote(str(args.region))}
export AWS_REGION={shlex.quote(str(args.region))}

# IMPORTANT: Avoid using ~/default (stale workspace checkout).
# We always run from a fresh code bundle uploaded from your laptop.
# Use a stable working directory (not /tmp) so paths are predictable.
WORKDIR="$HOME/ee269_run/current"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
export WORKDIR
echo "[anyscale_train] Using WORKDIR=$WORKDIR"

# Ensure boto3 exists on the head node for downloading the bundle.
python -c "import boto3" >/dev/null 2>&1 || python -m pip install -q boto3

python - <<'PY'
import os
import tarfile
from pathlib import Path

import boto3

bundle_uri = {code_bundle_s3_uri!r}
if not bundle_uri.startswith("s3://"):
    raise RuntimeError(f"Expected s3:// bundle uri, got: {{bundle_uri}}")
rest = bundle_uri[len("s3://"):]
bucket, _, key = rest.partition("/")
if not bucket or not key:
    raise RuntimeError(f"Bad bundle uri: {{bundle_uri}}")

workdir = Path(os.environ.get("WORKDIR") or ".").resolve()
tar_path = workdir / "code_bundle.tar.gz"

s3 = boto3.client("s3")
s3.download_file(bucket, key, str(tar_path))

with tarfile.open(tar_path, mode="r:gz") as tf:
    tf.extractall(path=str(workdir))

print(f"[anyscale_train] Downloaded+extracted code bundle to {{workdir}}")
PY

cd "$WORKDIR"

    # Make the working directory importable in a robust way.
    # Editable install is lightweight and ensures `python -m diffusion...` resolves to this code.
    python -m pip install -q -e . || true

echo "[anyscale_train] Debug: pwd=$(pwd)"
echo "[anyscale_train] Debug: config.py max_batches line:"
python - <<'PY'
from pathlib import Path

p = Path('diffusion/utils/config.py')
if p.exists():
    for line in p.read_text().splitlines():
        if 'max_batches' in line:
            print('  ', line)
            break
else:
    print('  (missing diffusion/utils/config.py in WORKDIR)')
PY

echo "[anyscale_train] Debug: import locations:"
python - <<'PY'
import diffusion
from diffusion.utils.config import DiffusionConfig
import diffusion.utils.config as cfg_mod

print('  diffusion.__file__         =', getattr(diffusion, '__file__', None))
print('  diffusion.utils.config.__file__ =', getattr(cfg_mod, '__file__', None))
print('  DiffusionConfig.__version__ =', getattr(DiffusionConfig, '__version__', None))
print('  DiffusionConfig.max_batches =', getattr(DiffusionConfig, 'max_batches', None))
PY

echo "[anyscale_train] Verifying config from bundle..."
python - <<'PY'
from diffusion.utils.config import DiffusionConfig

print(
    "  -> config.__version__=%s max_batches=%s"
    % (getattr(DiffusionConfig, "__version__", "N/A"), getattr(DiffusionConfig, "max_batches", None))
)
PY

# IMPORTANT: Do NOT `ray stop` on Ctrl+C (that kills the whole cluster).
trap 'echo "[anyscale_train] Interrupt received. Exiting without ray stop."; exit 130' INT TERM

# Bootstrap minimal dependencies on the GPU worker node.
python - <<'PY'
import os, time, ray
ray.init(address='auto')

@ray.remote(num_gpus=1)
def ensure_training_deps():
    import sys, subprocess
    # Minimal deps for training, should match requirements_train_minimal.txt
    packages = ["torch", "diffusers", "accelerate", "boto3", "tensorboard"]
    try:
        for pkg in packages:
            __import__(pkg)
        import torch
        return {{'status': 'already_installed', 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}}
    except ImportError:
        pass # Must install

    print("-> Installing minimal training dependencies on worker...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'diffusion/aws/requirements_train_minimal.txt'])
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        {torch_spec_py},
        '--index-url', {torch_index_url_py}
    ])
    import torch
    return {{'status': 'installed', 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}}

print('[anyscale_train] Ensuring worker has dependencies...')
start=time.time()
try:
    # Wait up to 10 minutes for a GPU worker to become available and install deps.
    result = ray.get(ensure_training_deps.remote(), timeout=600)
    print(f"[anyscale_train] Worker ready: {{result}}")
except Exception as e:
    print(f"[anyscale_train] ERROR: Failed to prepare worker node: {{e}}")
    # Print cluster status to help debug autoscaler issues.
    print("[anyscale_train] Current cluster resources:", ray.cluster_resources())
    raise

print('[anyscale_train] Deps ready in {{round(time.time()-start, 1)}}s')
ray.shutdown()
PY

# Launch the main training script
{train_cmd}

{sample_block}"""
    return script


# Global to track remote process for signal handling
_remote_process: Optional[subprocess.Popen] = None
_workspace_name: Optional[str] = None
_interrupt_count = 0


def _stop_remote_jobs(workspace: str) -> None:
    """Stop all Ray jobs running in the workspace."""
    print("\n[anyscale_train] Stopping remote Ray jobs...")
    prefix = _anyscale_cmd_prefix()
    
    # List jobs
    list_cmd = [*prefix, "workspace_v2", "run_command", "--name", workspace, "ray job list --format json"]
    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout:
            # Try to parse job IDs and stop them
            import json
            try:
                jobs = json.loads(result.stdout)
                if isinstance(jobs, list):
                    for job in jobs:
                        job_id = job.get("job_id") or job.get("submission_id")
                        status = job.get("status", "")
                        if job_id and status in ["RUNNING", "PENDING"]:
                            print(f"  Stopping job: {job_id}")
                            stop_cmd = [*prefix, "workspace_v2", "run_command", "--name", workspace, f"ray job stop {job_id}"]
                            subprocess.run(stop_cmd, timeout=10)
            except (json.JSONDecodeError, KeyError, TypeError):
                # Fallback: just try to stop all jobs
                stop_all_cmd = [*prefix, "workspace_v2", "run_command", "--name", workspace, "ray job stop --all || true"]
                subprocess.run(stop_all_cmd, timeout=10)
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Warning: Could not stop remote jobs: {e}")
    
    print("[anyscale_train] Remote job cleanup complete.")


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _interrupt_count, _remote_process, _workspace_name
    
    _interrupt_count += 1
    
    if _interrupt_count == 1:
        print("\n\n⚠️  Interrupt received (Ctrl+C). Stopping training...")
        print("    Press Ctrl+C again to force quit.\n")
        
        # Send SIGINT to the remote process (will propagate to ray_train.py)
        if _remote_process and _remote_process.poll() is None:
            _remote_process.send_signal(signal.SIGINT)
            print("[anyscale_train] Sent interrupt signal to remote command...")
            
            # Best-effort: stop Ray *jobs* (do not ray stop the cluster).
            if _workspace_name:
                print("[anyscale_train] Attempting to stop Ray jobs on workspace...")
                prefix = _anyscale_cmd_prefix()
                try:
                    stop_cmd = [
                        *prefix,
                        "workspace_v2",
                        "run_command",
                        "--name",
                        _workspace_name,
                        "ray job stop --all || true",
                    ]
                    subprocess.run(stop_cmd, timeout=5, capture_output=True)
                except Exception:
                    pass  # Best effort
            
            try:
                _remote_process.wait(timeout=10)
                print("[anyscale_train] Remote command stopped gracefully.")
            except subprocess.TimeoutExpired:
                print("[anyscale_train] Timeout waiting for remote command; terminating...")
                _remote_process.terminate()
                try:
                    _remote_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    _remote_process.kill()
        
        sys.exit(0)
    else:
        print("\n⚠️  Force quit!")
        # Kill immediately and stop Ray
        if _remote_process and _remote_process.poll() is None:
            _remote_process.kill()
        if _workspace_name:
            prefix = _anyscale_cmd_prefix()
            try:
                stop_cmd = [
                    *prefix,
                    "workspace_v2",
                    "run_command",
                    "--name",
                    _workspace_name,
                    "ray job stop --all || true",
                ]
                subprocess.run(stop_cmd, timeout=3, capture_output=True)
            except Exception:
                pass
        sys.exit(1)


def main() -> None:
    global _workspace_name, _remote_process
    
    args = _parse_args()
    _workspace_name = args.workspace
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, _signal_handler)
    
    # Local preprocessing option: regenerate data with current config before training
    # If --force-preprocess is requested, do it BEFORE Ray by regenerating+uploading locally.
    # (This also avoids relying on remote raw dataset availability.)
    if args.force_preprocess and args.s3:
        args.preprocess_locally = True

    if args.preprocess_locally and args.s3:
        local_cmd = [
            "python",
            "-m",
            "diffusion.train.ray_train",
            "--no-ray",
            "--prepare-s3-data-if-missing",  # Skip if already in S3
            "--s3",
            str(args.s3),
            "--s3-region",
            str(args.region),
        ]
        for override in (args.config_override or []):
            local_cmd.extend(["--config-override", override])

        # Use uv if available (consistent interpreter / deps).
        if _have("uv"):
            local_cmd = ["uv", "run", *local_cmd]

        print("[anyscale_train] Checking S3 for existing preprocessed data...")
        print("[anyscale_train] (Will preprocess and upload only if missing.)")
        subprocess.run(local_cmd, check=True)
        print("[anyscale_train] Preprocessing check complete.")
    
    # Skip automatic S3 data check - let the remote workspace use existing S3 data
    # (Local preprocessing fails on Mac due to torch compatibility issues)
    # If you need preprocessing, use --preprocess-locally flag

    # Create + upload a small code bundle to S3 for this run.
    # This avoids stale ~/default checkouts and avoids rsync workspace push failures.
    repo_root = Path(__file__).resolve().parents[2]
    bundle_path = _make_code_bundle(repo_root)
    bucket, prefix = _parse_s3_uri(str(args.s3))
    bundle_key_prefix = f"{prefix}/code_bundles" if prefix else "code_bundles"
    code_bundle_s3_uri = _join_s3(bucket, bundle_key_prefix, bundle_path.name)
    print(f"[anyscale_train] Uploading code bundle to {code_bundle_s3_uri} ...")
    _upload_file_to_s3(local_path=bundle_path, s3_uri=code_bundle_s3_uri, region=str(args.region))
    print("[anyscale_train] Code bundle uploaded.")

    prefix = _anyscale_cmd_prefix()

    # One remote command string (bash). Needs to be a single CLI arg.
    remote = _remote_script(args, code_bundle_s3_uri=code_bundle_s3_uri)

    cmd = [
        *prefix,
        "workspace_v2",
        "run_command",
        "--name",
        args.workspace,
        remote,
    ]

    # Make region available to the CLI (mostly for any local AWS calls).
    env = os.environ.copy()
    env.setdefault("AWS_DEFAULT_REGION", str(args.region))
    env.setdefault("AWS_REGION", str(args.region))

    print("\n💡 Tip: Press Ctrl+C to stop training and clean up remote jobs.\n")
    
    _remote_process = subprocess.Popen(cmd, env=env)
    try:
        _remote_process.wait()
    except KeyboardInterrupt:
        _signal_handler(signal.SIGINT, None)

    rc = _remote_process.returncode
    if rc != 0:
        print(f"[anyscale_train] Remote command exited with code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
