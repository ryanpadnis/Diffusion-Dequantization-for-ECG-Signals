"""One-command launcher for sampling from trained models on an Anyscale workspace.

Runs locally (your laptop) and uses the Anyscale CLI to execute sampling inside
an existing workspace with GPU workers.

Goals:
- Load trained checkpoint from S3
- Generate samples efficiently on GPU worker
- Upload samples back to S3

Example:
  uv run python -m diffusion.aws.anyscale_sample \
    --workspace EE269-spot2 \
    --s3 s3://anyscale-production-data-.../ee269project \
    --region us-east-1 \
    --run-id 20260204_215551 \
    --checkpoint best_model.pt \
    --num-samples 100
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import signal
import sys
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
    p.add_argument("--run-id", type=str, required=True, help="Run ID to sample from (e.g. 20260204_215551)")
    p.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename (default: best_model.pt)")
    p.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    p.add_argument("--ray-num-gpus", type=float, default=1.0, help="GPUs to request for sampling")
    
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


def _remote_script(args: argparse.Namespace) -> str:
    """Generate the remote script to run inside the workspace."""
    
    sample_flags: List[str] = [
        "python",
        "-m",
        "diffusion.train.ray_sample",
        "--ray-address",
        "auto",
        "--ray-num-gpus",
        str(float(args.ray_num_gpus)),
        "--s3",
        str(args.s3),
        "--s3-region",
        str(args.region),
        "--run-id",
        str(args.run_id),
        "--checkpoint",
        str(args.checkpoint),
        "--num-samples",
        str(int(args.num_samples)),
    ]

    sample_cmd = " ".join(shlex.quote(x) for x in sample_flags)

    torch_spec_py = repr(str(args.torch))
    torch_index_url_py = repr(str(args.torch_index_url))

    script = f"""set -e

cd ~/default

export AWS_DEFAULT_REGION={shlex.quote(str(args.region))}
export AWS_REGION={shlex.quote(str(args.region))}

python - <<'PY'
import os, time, ray
ray.init(address='auto')

@ray.remote(num_gpus=1)
def ensure_sampling_deps():
    import sys, subprocess

    def _ok():
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            import boto3  # noqa: F401
            return True
        except Exception:
            return False

    if _ok():
        import torch
        return {{'status': 'already_installed', 'torch': getattr(torch, '__version__', None), 'cuda': torch.cuda.is_available()}}

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'diffusion/aws/requirements_train_minimal.txt'])
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        {torch_spec_py},
        '--index-url', {torch_index_url_py}
    ])

    import torch
    return {{'status': 'installed', 'torch': getattr(torch, '__version__', None), 'cuda': torch.cuda.is_available()}}

print('[anyscale_sample] ensuring worker deps...')
start=time.time()
last_err = None
ref = None
for attempt in range(1, 4):
    try:
        ref = ensure_sampling_deps.remote()
        t0 = time.time()
        while True:
            ready, _ = ray.wait([ref], timeout=10.0)
            if ready:
                print(ray.get(ref))
                last_err = None
                ref = None
                break

            if int(time.time() - t0) % 30 == 0:
                print('[anyscale_sample] waiting for GPU worker... cluster_resources:', ray.cluster_resources())

            if time.time() - t0 > 600.0:
                raise TimeoutError('Timed out waiting for GPU worker / dep install task to complete.')

        if last_err is None:
            break
    except Exception as e:
        last_err = e
        print("[anyscale_sample] ensure_sampling_deps failed (attempt %d/3): %s: %s" % (attempt, type(e).__name__, e))
        time.sleep(10.0 * attempt)

if last_err is not None:
    raise last_err
print('[anyscale_sample] deps done in', round(time.time()-start, 1), 's')
ray.shutdown()
PY

{sample_cmd}
"""
    return script


# Global to track remote process for signal handling
_remote_process: Optional[subprocess.Popen] = None
_workspace_name: Optional[str] = None
_interrupt_count = 0


def _stop_remote_jobs(workspace: str) -> None:
    """Stop all Ray jobs running in the workspace."""
    print("\n[anyscale_sample] Stopping remote Ray jobs...")
    prefix = _anyscale_cmd_prefix()
    
    list_cmd = [*prefix, "workspace_v2", "run_command", "--name", workspace, "ray job list --format json"]
    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout:
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
                stop_all_cmd = [*prefix, "workspace_v2", "run_command", "--name", workspace, "ray job stop --all || true"]
                subprocess.run(stop_all_cmd, timeout=10)
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Warning: Could not stop remote jobs: {e}")
    
    print("[anyscale_sample] Remote job cleanup complete.")


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _interrupt_count, _remote_process, _workspace_name
    
    _interrupt_count += 1
    
    if _interrupt_count == 1:
        print("\n\n⚠️  Interrupt received (Ctrl+C). Stopping sampling...")
        print("    Press Ctrl+C again to force quit.\n")
        
        # Send SIGINT to the remote process (will propagate to ray_sample.py)
        if _remote_process and _remote_process.poll() is None:
            _remote_process.send_signal(signal.SIGINT)
            print("[anyscale_sample] Sent interrupt signal to remote command...")
            try:
                _remote_process.wait(timeout=10)
                print("[anyscale_sample] Remote command stopped gracefully.")
            except subprocess.TimeoutExpired:
                print("[anyscale_sample] Timeout waiting for remote command; terminating...")
                _remote_process.terminate()
                try:
                    _remote_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    _remote_process.kill()
        
        sys.exit(0)
    else:
        print("\n⚠️  Force quit!")
        # Kill immediately
        if _remote_process and _remote_process.poll() is None:
            _remote_process.kill()
        sys.exit(1)


def main() -> None:
    global _workspace_name, _remote_process
    
    args = _parse_args()
    _workspace_name = args.workspace
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, _signal_handler)

    # Run Anyscale CLI from local machine.
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

    # Make region available to the CLI.
    env = os.environ.copy()
    env.setdefault("AWS_DEFAULT_REGION", str(args.region))
    env.setdefault("AWS_REGION", str(args.region))

    print("\n💡 Tip: Press Ctrl+C to stop sampling and clean up remote jobs.\n")
    
    _remote_process = subprocess.Popen(cmd, env=env)
    try:
        _remote_process.wait()
    except KeyboardInterrupt:
        _signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
