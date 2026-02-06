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
import os
import shlex
import shutil
import subprocess
import signal
import sys
import time
from typing import List, Optional


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--workspace", type=str, default="EE269-spot2", help="Anyscale workspace name")
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


def _remote_script(args: argparse.Namespace) -> str:
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
    if args.force_preprocess:
        train_flags.append("--force-preprocess")
    if args.epochs is not None:
        train_flags.extend(["--epochs", str(int(args.epochs))])
    if args.max_batches is not None:
        train_flags.extend(["--max-batches", str(int(args.max_batches))])
    if args.max_samples is not None:
        train_flags.extend(["--max-samples", str(int(args.max_samples))])

    train_cmd = " ".join(shlex.quote(x) for x in train_flags)

    # Use a bootstrap task to ensure the worker has deps. We do a quick import test first.
    # Note: this installs into the worker's Python environment; it persists for the node lifetime.
    torch_spec_py = repr(str(args.torch))
    torch_index_url_py = repr(str(args.torch_index_url))

    script = f"""set -e
cd ~/default

# Set up signal handler to cancel Ray tasks on interrupt
trap 'echo "Interrupt received, cancelling Ray tasks..."; ray stop || true; exit 130' INT TERM

# Let Ray's runtime_env handle code syncing.
# This script just ensures deps and launches the training.
echo "[anyscale_train] Starting training script..."
echo "[anyscale_train] Code is packaged by Ray's runtime_env."
echo "[anyscale_train] Verifying config from runtime..."

python -c "from diffusion.utils.config import DiffusionConfig; print(f'  -> Imported config with max_batches={{DiffusionConfig.max_batches}}')"

export AWS_DEFAULT_REGION={shlex.quote(str(args.region))}
export AWS_REGION={shlex.quote(str(args.region))}

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
"""
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
            
            # Also try to stop Ray directly on the workspace
            if _workspace_name:
                print("[anyscale_train] Attempting to stop Ray tasks on workspace...")
                prefix = _anyscale_cmd_prefix()
                try:
                    stop_cmd = [*prefix, "workspace_v2", "run_command", "--name", _workspace_name, "ray stop || true"]
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
                stop_cmd = [*prefix, "workspace_v2", "run_command", "--name", _workspace_name, "ray stop"]
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
    if args.preprocess_locally and args.s3:
        local_cmd = [
            "python",
            "-m",
            "diffusion.train.ray_train",
            "--no-ray",
            "--prepare-s3-data",  # Force regeneration (not --prepare-s3-data-if-missing)
            "--s3",
            str(args.s3),
            "--s3-region",
            str(args.region),
        ]

        # Use uv if available (consistent interpreter / deps).
        if _have("uv"):
            local_cmd = ["uv", "run", *local_cmd]

        print("[anyscale_train] Running local preprocessing with current config...")
        print("[anyscale_train] This will regenerate and upload preprocessed data to S3.")
        subprocess.run(local_cmd, check=True)
        print("[anyscale_train] Local preprocessing complete.")
    
    # Skip automatic S3 data check - let the remote workspace use existing S3 data
    # (Local preprocessing fails on Mac due to torch compatibility issues)
    # If you need preprocessing, use --preprocess-locally flag

    # The `anyscale workspace_v2 run_command` will use Ray's runtime_env
    # to sync the local code. We don't need manual push/sync anymore.
    prefix = _anyscale_cmd_prefix()

    # One remote command string (bash). Needs to be a single CLI arg.
    remote = _remote_script(args)

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


if __name__ == "__main__":
    main()
