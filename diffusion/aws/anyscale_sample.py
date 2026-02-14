"""One-command launcher for sampling from trained models on an Anyscale workspace.

Runs locally (your laptop) and uses the Anyscale CLI to execute sampling inside
an existing workspace with GPU workers.

Goals:
- Load trained checkpoint from S3
- Generate samples efficiently on GPU worker
- Upload samples back to S3
- Match reliability of training script (code bundling).

Example:
    uv run python -m diffusion.aws.anyscale_sample --workspace EE269-ondemand --s3 s3://anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project --region us-east-1 --version V4 --run-id 20260213_202346 --checkpoint best_model.pt --num-samples 16 --num-trajectories 1

    # V4 shortcut wrapper:
    uv run python -m diffusion.aws.anyscale_sample_v4 --workspace EE269-ondemand --s3 s3://anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project --region us-east-1 --run-id 20260213_202346 --checkpoint best_model.pt --num-samples 16 --num-trajectories 1
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


def _git_sha_short(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo_root))
        return out.decode("utf-8").strip() or "nogit"
    except Exception:
        return "nogit"


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
    p.add_argument(
        "--run-id",
        type=str,
        default="latest",
        help="Run ID to sample from (e.g. 20260204_215551). Use 'latest' to auto-pick, or use 'root' for root layout.",
    )
    p.add_argument("--version", type=str, default="V1", help="Experiment version (default: V1)")
    p.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename (default: best_model.pt)")
    p.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    p.add_argument("--num-trajectories", type=int, default=1, help="Number of trajectories per sample")
    p.add_argument("--use-ddim", action="store_true", help="Use DDIM sampler (default: DDPM)")
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
    """Join parts into an s3:// URI."""
    path = "/".join(str(p).strip("/") for p in parts)
    return f"s3://{bucket}/{path}"


def _make_code_bundle(root_dir: Path) -> Path:
    """Create a gzipped tarball of the current source code.

    Mirrors the training launcher: aggressively excludes big artifacts so bundling is fast.
    Uses a content hash so identical code can reuse the same bundle key.
    """
    repo_root = root_dir

    excludes_dir_parts = {
        ".git",
        ".github",
        ".vscode",
        ".idea",
        "__pycache__",
        "venv",
        ".venv",
        "node_modules",
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

    tmpdir = Path(tempfile.mkdtemp(prefix="ee269_code_bundle_sample_"))
    sha = _git_sha_short(repo_root)

    h = hashlib.sha1()
    for p in sorted(repo_root.rglob("*.py")):
        if should_skip(p):
            continue
        try:
            h.update(p.read_bytes())
        except Exception:
            continue
    digest = h.hexdigest()[:10]

    # Stable-ish name: if code doesn't change, bundle name doesn't change.
    bundle_path = tmpdir / f"code_{sha}_{digest}.tar.gz"

    t0 = time.time()
    with tarfile.open(bundle_path, mode="w:gz") as tf:
        for path in repo_root.rglob("*"):
            if not path.is_file():
                continue
            if should_skip(path):
                continue
            arcname = path.relative_to(repo_root)
            tf.add(path, arcname=str(arcname))

    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    print(f"[anyscale_sample] Bundle created in {time.time()-t0:.1f}s: {bundle_path} ({size_mb:.1f} MB)")
    return bundle_path


def _upload_file_to_s3(local_path: Path, s3_uri: str, region: str) -> None:
    """Upload a local file to S3 using boto3.

    Also skips upload if the object already exists (bundle caching).
    """
    bucket, key = _parse_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"Expected full s3://bucket/key URI, got: {s3_uri!r}")
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ModuleNotFoundError as e:
        raise RuntimeError("boto3 is required locally to upload the code bundle") from e

    s3 = boto3.client("s3", region_name=str(region) if region else None)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        print(f"[anyscale_sample] Bundle already exists on S3; skipping upload: {s3_uri}")
        return
    except ClientError:
        pass

    print(f"[anyscale_sample] Uploading bundle to {s3_uri} ...")
    s3.upload_file(str(local_path), bucket, key)
    print("[anyscale_sample] Bundle upload complete.")


def _ensure_raw_chunks_in_s3(*, s3_base: str, region: str) -> None:
    """Make sure raw time-domain chunks exist in S3 for remote sampling.

    Remote sampling needs a time-domain chunks file (arrhythmia_chunks.pt) to compute
    conditioning and invert generated spectrograms back to time domain.
    """

    local_raw = Path("data/data/processed/arrhythmia_chunks.pt")
    if not local_raw.exists():
        print(f"[anyscale_sample] Local raw chunks not found; skipping upload: {local_raw}")
        return

    # Upload to the canonical project-relative location in S3.
    target_s3 = f"{str(s3_base).rstrip('/')}/data/data/processed/arrhythmia_chunks.pt"

    # Reuse the same boto3 logic as bundle upload.
    bucket, key = _parse_s3_uri(target_s3)
    if not key:
        raise ValueError(f"Expected full s3://bucket/key URI, got: {target_s3!r}")

    try:
        import boto3
        from botocore.exceptions import ClientError
    except ModuleNotFoundError as e:
        raise RuntimeError("boto3 is required locally to upload raw chunks") from e

    s3 = boto3.client("s3", region_name=str(region) if region else None)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        print(f"[anyscale_sample] Raw chunks already exist on S3; skipping upload: {target_s3}")
        return
    except ClientError:
        pass

    print(f"[anyscale_sample] Raw chunks missing on S3; uploading: {local_raw} -> {target_s3}")
    s3.upload_file(str(local_raw), bucket, key)
    print("[anyscale_sample] Raw chunks upload complete.")


def _remote_script(args: argparse.Namespace, code_bundle_s3_uri: str) -> str:
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
        "--version",
        str(args.version),
        "--checkpoint",
        str(args.checkpoint),
        "--num-samples",
        str(int(args.num_samples)),
        "--num-trajectories",
        str(int(args.num_trajectories)),
    ]
    if args.use_ddim:
        sample_flags.append("--use-ddim")
        
    sample_cmd = " ".join(shlex.quote(x) for x in sample_flags)

    torch_spec_py = repr(str(args.torch))
    torch_index_url_py = repr(str(args.torch_index_url))

    script = f"""set -e

export AWS_DEFAULT_REGION={shlex.quote(str(args.region))}
export AWS_REGION={shlex.quote(str(args.region))}

# WORKDIR Setup
WORKDIR="$HOME/ee269_sample/current"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
export WORKDIR
echo "[anyscale_sample] Using WORKDIR=$WORKDIR"

# Ensure boto3 exists on head node
python -c "import boto3" >/dev/null 2>&1 || python -m pip install -q boto3

# Download and extract code bundle
python - <<'PY'
import os
import tarfile
from pathlib import Path
import boto3

bundle_uri = {code_bundle_s3_uri!r}
workdir = Path(os.environ.get("WORKDIR") or ".").resolve()
if bundle_uri.startswith("s3://"):
    bucket = bundle_uri.replace("s3://", "").split("/")[0]
    key = "/".join(bundle_uri.replace("s3://", "").split("/")[1:])
else:
    # Handle non-S3 path if necessary, but we assume S3
    bucket, key = None, None

print(f"[anyscale_sample] Downloading code from {{bundle_uri}}...")
s3 = boto3.client("s3")
s3.download_file(bucket, key, str(workdir / "code_bundle.tar.gz"))

with tarfile.open(workdir / "code_bundle.tar.gz", mode="r:gz") as tf:
    tf.extractall(path=str(workdir))
print(f"[anyscale_sample] Code extracted.")
PY

cd "$WORKDIR"

# Make importable
python -m pip install -q -e . || true

# Bootstap Worker Dependencies
python - <<'PY'
import os, time, ray
ray.init(address='auto')

@ray.remote(num_gpus=1)
def ensure_sampling_deps():
    import sys, subprocess

    def _ok():
        try:
            import torch
            import diffusers
            import boto3
            import accelerate
            return True
        except ImportError:
            return False

    if _ok():
        import torch
        if torch.cuda.is_available():
             return {{'status': 'fast_check_ok', 'torch': torch.__version__}}

    print("-> Installing dependencies on worker...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip'])
    
    pkgs = ["numpy", "scipy", "diffusers", "accelerate", "safetensors", "tensorboard", "tqdm", "boto3"]
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])
    
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q',
        {torch_spec_py},
        '--index-url', {torch_index_url_py}
    ])

    import torch
    return {{'status': 'installed', 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}}

print('[anyscale_sample] Checking worker deps...')
try:
    res = ray.get(ensure_sampling_deps.remote(), timeout=600)
    print(f"[anyscale_sample] Worker ready: {{res}}")
except Exception as e:
    print(f"[anyscale_sample] Error on worker: {{e}}")
    raise
ray.shutdown()
PY

# Run Sampling
{sample_cmd}
"""
    return script


# Global to track remote process for signal handling
_remote_process: Optional[subprocess.Popen] = None
_workspace_name: Optional[str] = None
_interrupt_count = 0


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _interrupt_count, _remote_process, _workspace_name
    
    _interrupt_count += 1
    
    if _interrupt_count == 1:
        print("\n\n⚠️  Interrupt received (Ctrl+C). Stopping sampling...")
        print("    Press Ctrl+C again to force quit.\n")
        
        if _remote_process and _remote_process.poll() is None:
            _remote_process.send_signal(signal.SIGINT)
            try:
                _remote_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _remote_process.terminate()
        sys.exit(0)
    else:
        print("\n⚠️  Force quit!")
        if _remote_process:
            _remote_process.kill()
        sys.exit(1)


def main() -> None:
    global _workspace_name, _remote_process
    
    args = _parse_args()
    _workspace_name = args.workspace
    
    signal.signal(signal.SIGINT, _signal_handler)

    # 0. Ensure raw chunks exist in S3 so remote sampling can preprocess.
    _ensure_raw_chunks_in_s3(s3_base=str(args.s3), region=str(args.region))

    # 1. Create & Upload Bundle
    repo_root = Path(__file__).resolve().parents[2]
    bundle_path = _make_code_bundle(repo_root)
    bucket, prefix = _parse_s3_uri(str(args.s3))
    bundle_key_prefix = f"{prefix}/code_bundles" if prefix else "code_bundles"
    code_bundle_s3_uri = _join_s3(bucket, bundle_key_prefix, bundle_path.name)
    print(f"[anyscale_sample] Uploading code bundle to {code_bundle_s3_uri} ...")
    _upload_file_to_s3(local_path=bundle_path, s3_uri=code_bundle_s3_uri, region=str(args.region))

    # 2. Run Remote
    prefix = _anyscale_cmd_prefix()
    remote = _remote_script(args, code_bundle_s3_uri=code_bundle_s3_uri)

    cmd = [
        *prefix,
        "workspace_v2",
        "run_command",
        "--name",
        args.workspace,
        remote,
    ]

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
