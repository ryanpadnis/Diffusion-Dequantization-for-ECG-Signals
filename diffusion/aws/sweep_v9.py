"""V9 sweep: same 12-model grid as V8, but sampling uses front+back (head+tail).

This runs 4 quantizers × 3 schedules = 12 sequential train+sample runs on Anyscale.

Key behavior change vs V8:
- Training is unchanged.
- Sampling calls `diffusion.train.ray_sample` with `--select headtail --head 8 --tail 8`
  (and explicitly sets holdout size to 500 from the end).

Usage
-----
  uv run python -m diffusion.aws.sweep_v9               # full sweep
  uv run python -m diffusion.aws.sweep_v9 --dry-run     # print remote script, no execution
  uv run python -m diffusion.aws.sweep_v9 --no-preprocess  # skip local preprocessing
  uv run python -m diffusion.aws.sweep_v9 --skip-sample    # train only
  uv run python -m diffusion.aws.sweep_v9 --only uniform__linear uniform__cosine
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import List, Optional

# Re-use helpers from anyscale_train to avoid duplication.
from diffusion.aws.anyscale_train import (
    _make_code_bundle,
    _upload_file_to_s3,
    _parse_s3_uri,
    _join_s3,
    _have,
    _anyscale_cmd_prefix,
)


DEFAULT_S3 = "s3://anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project"
DEFAULT_WORKSPACE = "EE269-ondemand"
DEFAULT_REGION = "us-east-1"
DEFAULT_TORCH_SPEC = "torch==2.10.0"
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
VERSION = "V9"

QUANTIZERS: List[str] = ["uniform", "lloyd_max", "mu_law", "dithered_uniform"]
SCHEDULES: List[str] = ["linear", "cosine", "sigmoid"]

DEFAULT_NUM_SAMPLES = 16
DEFAULT_NUM_TRAJECTORIES = 1

# Sampling selection: "front+back"
DEFAULT_SELECT = "headtail"
DEFAULT_HEAD = 8
DEFAULT_TAIL = 8
DEFAULT_HOLDOUT_COUNT = 500
DEFAULT_HOLDOUT_FROM_END = True


def _s3_object_exists(*, s3_uri: str, region: str) -> bool:
    try:
        import boto3
    except ModuleNotFoundError:
        return False

    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {s3_uri!r}")
    rest = s3_uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected full s3://bucket/key URI, got {s3_uri!r}")

    s3 = boto3.client("s3", region_name=str(region) if region else None)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _ensure_raw_chunks_in_s3(*, repo_root: Path, args: argparse.Namespace) -> None:
    raw_path = (repo_root / str(args.raw_data_path)).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw chunks file: {raw_path}")

    # Upload to S3 under the same repo-relative key so ray_sample's search finds it.
    bucket, prefix = _parse_s3_uri(str(args.s3))
    rel_key = str(raw_path.relative_to(repo_root)).replace("\\", "/")
    s3_uri = _join_s3(bucket, prefix, rel_key)

    if _s3_object_exists(s3_uri=s3_uri, region=str(args.region)):
        print(f"[sweep_v9] Raw chunks already in S3: {s3_uri}")
        return

    size_gb = raw_path.stat().st_size / (1024**3)
    print(f"[sweep_v9] Uploading raw chunks to S3 ({size_gb:.2f} GB) → {s3_uri}")
    _upload_file_to_s3(local_path=raw_path, s3_uri=s3_uri, region=str(args.region))
    print("[sweep_v9] Raw chunks uploaded.")


def model_name(quantizer: str, schedule: str) -> str:
    return f"{quantizer}__{schedule}"


def _print_banner(text: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {text}\n{bar}\n", flush=True)


def _ensure_workspace_running(workspace: str, *, timeout_s: int = 300) -> None:
    """Wake the workspace if sleeping and wait until RUNNING."""
    prefix = _anyscale_cmd_prefix()
    print(f"[sweep_v9] Starting workspace '{workspace}' (no-op if already running)...")
    try:
        subprocess.run([*prefix, "workspace_v2", "start", "--name", workspace], check=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[sweep_v9] Warning: workspace start: {e} — continuing anyway")

    deadline = time.time() + timeout_s
    last_status = "unknown"
    while time.time() < deadline:
        try:
            result = subprocess.run(
                [*prefix, "workspace_v2", "status", "--name", workspace],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            combined = (result.stdout + result.stderr).decode("utf-8", errors="replace").lower()
            last_status = combined.strip()
            if "running" in combined:
                print("[sweep_v9] Workspace is RUNNING.")
                return
        except Exception:
            pass
        last_line = last_status.splitlines()[-1].strip() if last_status else "..."
        print(f"[sweep_v9] Workspace not ready yet ({last_line!r}) — waiting...")
        time.sleep(15)

    raise RuntimeError(
        f"Workspace '{workspace}' did not reach RUNNING within {timeout_s}s "
        f"(last: {last_status!r})"
    )


def _run_local_preprocess(*, quantizer: str, args: argparse.Namespace) -> None:
    """Run preprocessing locally for one quantizer type and upload data to S3."""
    name = model_name(quantizer, "linear")
    local_cmd = [
        "python",
        "-m",
        "diffusion.train.ray_train",
        "--no-ray",
        "--prepare-s3-data",
        "--s3",
        str(args.s3),
        "--s3-region",
        str(args.region),
        "--config-override",
        f"version={VERSION}",
        "--config-override",
        f"run_name={name}",
        "--config-override",
        f"quantizer_type={quantizer}",
        "--config-override",
        "noise_schedule_type=linear",
        "--config-override",
        "beta_type=linear",
    ]
    for extra in (args.extra_override or []):
        local_cmd.extend(["--config-override", extra])
    if _have("uv"):
        local_cmd = ["uv", "run", *local_cmd]

    print(f"[sweep_v9] Preprocessing locally for quantizer={quantizer}...")
    subprocess.run(local_cmd, check=True)
    print(f"[sweep_v9] Preprocessing complete for quantizer={quantizer}.")


def _build_remote_script(*, selected: list, args: argparse.Namespace, code_bundle_s3_uri: str) -> str:
    """Build a single bash script that runs ALL selected models on the cluster."""

    region_q = shlex.quote(str(args.region))
    torch_spec_py = repr(DEFAULT_TORCH_SPEC)
    torch_index_url_py = repr(DEFAULT_TORCH_INDEX)
    total = len(selected)

    bucket, s3_prefix = _parse_s3_uri(str(args.s3))

    model_blocks: List[str] = []
    for i, (quantizer, schedule) in enumerate(selected, start=1):
        name = model_name(quantizer, schedule)
        version_path = f"{VERSION}/{name}"
        results_prefix = "/".join(filter(None, [s3_prefix, "diffusion-results", VERSION, name]))

        train_flags = [
            "python",
            "-m",
            "diffusion.train.ray_train",
            "--ray-address",
            "auto",
            "--ray-num-gpus",
            "1",
            "--s3",
            str(args.s3),
            "--s3-region",
            str(args.region),
            "--s3-sync-interval",
            "30",
            "--bundle-run",
            "--config-override",
            f"version={VERSION}",
            "--config-override",
            f"run_name={name}",
            "--config-override",
            f"quantizer_type={quantizer}",
            "--config-override",
            f"noise_schedule_type={schedule}",
            "--config-override",
            f"beta_type={schedule}",
        ]
        if args.epochs is not None:
            train_flags.extend(["--epochs", str(args.epochs)])
        for extra in (args.extra_override or []):
            train_flags.extend(["--config-override", extra])
        train_cmd_str = " ".join(shlex.quote(x) for x in train_flags)

        trained_var = f"TRAINED_{i}"
        sampled_var = f"SAMPLED_{i}"
        run_id_var = f"RUN_ID_{i}"

        check_trained_block = f"""\
set +e
{trained_var}=$(python -c "
import boto3, sys
try:
    pag = boto3.client('s3').get_paginator('list_objects_v2')
    for page in pag.paginate(Bucket={bucket!r}, Prefix={results_prefix!r}):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('best_model.pt'):
                print('yes'); sys.exit(0)
    print('no')
except Exception:
    print('no')
" 2>/dev/null)
set -e
echo "[sweep_v9] {name}: already_trained=${{{trained_var}}}" """

        if args.skip_sample:
            sample_section = 'echo "[sweep_v9] Sampling skipped (--skip-sample)."'
            check_sampled_block = ""
        else:
            sample_flags = [
                "python",
                "-m",
                "diffusion.train.ray_sample",
                "--ray-address",
                "auto",
                "--ray-num-gpus",
                "1",
                "--s3",
                str(args.s3),
                "--s3-region",
                str(args.region),
                "--version",
                version_path,
                "--checkpoint",
                "best_model.pt",
                "--num-samples",
                str(args.num_samples),
                "--num-trajectories",
                str(args.num_trajectories),
                "--select",
                DEFAULT_SELECT,
                "--head",
                str(DEFAULT_HEAD),
                "--tail",
                str(DEFAULT_TAIL),
                "--holdout-count",
                str(DEFAULT_HOLDOUT_COUNT),
            ]
            if DEFAULT_HOLDOUT_FROM_END:
                sample_flags.append("--holdout-from-end")
            sample_cmd_str = " ".join(shlex.quote(x) for x in sample_flags)

            check_sampled_block = f"""\
set +e
{sampled_var}=$(python -c "
import boto3, sys
try:
    pag = boto3.client('s3').get_paginator('list_objects_v2')
    for page in pag.paginate(Bucket={bucket!r}, Prefix={results_prefix!r}):
        for obj in page.get('Contents', []):
            k = obj['Key']
            if '/samples/' in k and (k.endswith('.pt') or k.endswith('.npy') or k.endswith('.png')):
                print('yes'); sys.exit(0)
    print('no')
except Exception:
    print('no')
" 2>/dev/null)
set -e
echo "[sweep_v9] {name}: already_sampled=${{{sampled_var}}}" """

            sample_section = f"""\
{check_sampled_block}
if [ "${{{sampled_var}}}" = "yes" ]; then
    echo "[sweep_v9] SKIP sampling {name} — samples already in S3"
else
    echo "[sweep_v9] Resolving run_id for {name}..."
    {run_id_var}=$(python - <<'__PY__'
import sys
try:
    from diffusion.aws.s3_io import normalize_s3_uri, join_s3_uri, list_run_prefixes, s3_object_exists
    runs = sorted(
        list_run_prefixes(join_s3_uri(normalize_s3_uri({str(args.s3)!r}), 'diffusion-results', {version_path!r})),
        key=lambda x: x[1]
    )
    # Pick the newest run that actually has the checkpoint uploaded.
    for pfx, _lm in reversed(runs):
        cfg = join_s3_uri(pfx, 'config.pkl')
        ckpt = join_s3_uri(pfx, 'checkpoints', 'best_model.pt')
        try:
            if s3_object_exists(cfg) and s3_object_exists(ckpt):
                print(pfx.rstrip('/').split('/')[-1])
                sys.exit(0)
        except Exception:
            continue
    print('NOTFOUND', file=sys.stderr); sys.exit(1)
except Exception as e:
    print(f'ERROR: {{e}}', file=sys.stderr); sys.exit(1)
__PY__
)
    echo "[sweep_v9] run_id=${{{run_id_var}}}"
    {sample_cmd_str} --run-id "${{{run_id_var}}}"
fi"""

        model_blocks.append(
            f"""\
echo ""
echo "========================================================================"
echo "  [{i}/{total}]  {name}  —  TRAIN{'' if args.skip_sample else '+SAMPLE'}"
echo "========================================================================"
{check_trained_block}
if [ "${{{trained_var}}}" = "yes" ]; then
    echo "[sweep_v9] SKIP training {name} — best_model.pt already in S3"
else
    echo "[sweep_v9] Training {name}..."
    {train_cmd_str}
    echo "[sweep_v9] Training done: {name}"
fi
{sample_section}
echo "[sweep_v9] Finished: {name}"
"""
        )

    all_model_blocks = "\n".join(model_blocks)

    return f"""\
set -e

export AWS_DEFAULT_REGION={region_q}
export AWS_REGION={region_q}

WORKDIR=\"$HOME/ee269_run/current\"
rm -rf \"$WORKDIR\"
mkdir -p \"$WORKDIR\"
export WORKDIR
echo \"[sweep_v9] WORKDIR=$WORKDIR\"

python -c \"import boto3\" >/dev/null 2>&1 || python -m pip install -q boto3

python - <<'__PY__'
import os, tarfile
from pathlib import Path
import boto3

bundle_uri = {code_bundle_s3_uri!r}
rest = bundle_uri[len('s3://'):]
bucket, _, key = rest.partition('/')
workdir = Path(os.environ.get('WORKDIR') or '.').resolve()
tar_path = workdir / 'code_bundle.tar.gz'
boto3.client('s3').download_file(bucket, key, str(tar_path))
with tarfile.open(tar_path, mode='r:gz') as tf:
    tf.extractall(path=str(workdir))
print(f\"[sweep_v9] Bundle extracted to {{workdir}}\")
__PY__

cd \"$WORKDIR\"
python -m pip install -q -e . || true

trap 'echo \"[sweep_v9] Interrupted.\"; exit 130' INT TERM

python - <<'__PY__'
import time, ray
ray.init(address='auto')

@ray.remote(num_gpus=1)
def ensure_deps():
    import sys, subprocess
    packages = ['torch', 'diffusers', 'accelerate', 'boto3', 'tensorboard']
    try:
        for p in packages:
            __import__(p)
        import torch
        return {{'ok': True, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}}
    except ImportError:
        pass
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'diffusion/aws/requirements_train_minimal.txt'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', {torch_spec_py}, '--index-url', {torch_index_url_py}])
    import torch
    return {{'ok': True, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}}

print('[sweep_v9] Bootstrapping GPU worker deps...')
t = time.time()
result = ray.get(ensure_deps.remote(), timeout=600)
print(f'[sweep_v9] Worker ready in {{round(time.time()-t, 1)}}s: {{result}}')
ray.shutdown()
__PY__

echo \"[sweep_v9] Starting sweep: {total} models\"

{all_model_blocks}

echo \"\"
echo \"[sweep_v9] ============================================================\"
echo \"[sweep_v9] ALL {total} MODELS COMPLETE.\"
echo \"[sweep_v9] ============================================================\"
echo \"SWEEP_ALL_DONE_SUCCESS\"\n"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V9 sweep: train+sample all 12 models in ONE cluster session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    p.add_argument("--s3", default=DEFAULT_S3)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs for every run.")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    p.add_argument("--num-trajectories", type=int, default=DEFAULT_NUM_TRAJECTORIES)
    p.add_argument(
        "--raw-data-path",
        type=str,
        default="data/data/processed/arrhythmia_chunks.pt",
        help="Local raw time-domain chunks file to upload to S3 for sampling.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the remote script and exit without submitting.")
    p.add_argument("--no-preprocess", action="store_true", help="Skip local preprocessing (data already in S3).")
    p.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="MODEL_NAME",
        help="Only include these models (e.g. uniform__linear lloyd_max__cosine).",
    )
    p.add_argument("--skip-sample", action="store_true", help="Train only — no sampling.")
    p.add_argument(
        "--extra-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra --config-override passed to every train run (repeatable).",
    )
    return p.parse_args()


_remote_process: Optional[subprocess.Popen] = None
_interrupt_count = 0


def _signal_handler(signum, frame) -> None:
    global _interrupt_count, _remote_process
    _interrupt_count += 1
    if _interrupt_count == 1:
        print("\n⚠️  Ctrl+C received. Stopping sweep (press again to force quit).")
        if _remote_process and _remote_process.poll() is None:
            _remote_process.send_signal(signal.SIGINT)
            try:
                _remote_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _remote_process.terminate()
        sys.exit(1)
    else:
        if _remote_process:
            _remote_process.kill()
        sys.exit(1)


def main() -> None:
    global _remote_process
    args = _parse_args()
    signal.signal(signal.SIGINT, _signal_handler)

    all_combos = list(product(QUANTIZERS, SCHEDULES))

    if args.only:
        valid = {model_name(q, s) for q, s in all_combos}
        unknown = [n for n in args.only if n not in valid]
        if unknown:
            print(f"[sweep_v9] ERROR: unknown model(s): {unknown}")
            print(f"[sweep_v9] Valid: {sorted(valid)}")
            sys.exit(1)
        selected = [(q, s) for q, s in all_combos if model_name(q, s) in args.only]
    else:
        selected = all_combos

    total = len(selected)
    unique_quantizers = list(dict.fromkeys(q for q, _ in selected))

    _print_banner(f"V9 Sweep  |  {total} models  |  1 cluster session  |  workspace={args.workspace}")
    print(f"  Models     : {[model_name(q, s) for q, s in selected]}")
    print(f"  S3 base    : {args.s3}")
    if args.skip_sample:
        print("  Sampling   : disabled")
    else:
        print(
            f"  Sampling   : {args.num_samples} samples, {args.num_trajectories} traj "
            f"(select={DEFAULT_SELECT}, head={DEFAULT_HEAD}, tail={DEFAULT_TAIL}, holdout={DEFAULT_HOLDOUT_COUNT})"
        )
    print(
        f"  Preprocess : {'disabled (--no-preprocess)' if args.no_preprocess else f'{len(unique_quantizers)} quantizer type(s) locally'}"
    )
    if args.dry_run:
        print("\n  *** DRY RUN — will print remote script only ***")
    print()

    repo_root = Path(__file__).resolve().parents[2]

    if args.dry_run:
        script = _build_remote_script(selected=selected, args=args, code_bundle_s3_uri="<code_bundle_s3_uri>")
        print("=" * 72)
        print("REMOTE SCRIPT (would be submitted via anyscale workspace_v2 run_command):")
        print("=" * 72)
        print(script)
        return

    # ---- Step 1: Local preprocessing (all quantizers first) ----
    if not args.no_preprocess:
        _print_banner("Step 1 / 4  —  Local preprocessing")
        for quantizer in unique_quantizers:
            _run_local_preprocess(quantizer=quantizer, args=args)
        print("[sweep_v9] All preprocessing complete.")

    # ---- Step 2: Ensure raw time-domain chunks exist in S3 for sampling ----
    _print_banner("Step 2 / 4  —  Ensuring raw chunks in S3")
    _ensure_raw_chunks_in_s3(repo_root=repo_root, args=args)

    # ---- Step 3: Upload code bundle ----
    _print_banner("Step 3 / 4  —  Uploading code bundle")
    bundle_path = _make_code_bundle(repo_root)
    bucket, prefix = _parse_s3_uri(str(args.s3))
    bundle_key_prefix = f"{prefix}/code_bundles" if prefix else "code_bundles"
    code_bundle_s3_uri = _join_s3(bucket, bundle_key_prefix, bundle_path.name)
    print(f"[sweep_v9] Uploading code bundle → {code_bundle_s3_uri}")
    _upload_file_to_s3(local_path=bundle_path, s3_uri=code_bundle_s3_uri, region=str(args.region))
    print("[sweep_v9] Code bundle uploaded.")

    # ---- Step 4: Submit to cluster ----
    _print_banner("Step 4 / 4  —  Submitting to cluster")
    _ensure_workspace_running(args.workspace)

    remote_script = _build_remote_script(selected=selected, args=args, code_bundle_s3_uri=code_bundle_s3_uri)

    anyscale_prefix = _anyscale_cmd_prefix()
    cmd = [*anyscale_prefix, "workspace_v2", "run_command", "--name", args.workspace, remote_script]

    env = os.environ.copy()
    env.setdefault("AWS_DEFAULT_REGION", str(args.region))
    env.setdefault("AWS_REGION", str(args.region))

    print(f"\n[sweep_v9] Submitting {total}-model sweep as a single remote session...")
    print("💡 Tip: Press Ctrl+C to interrupt.\n")
    print("💡 Note: anyscale run_command may exit 0 on remote failure — we check for")
    print("         a SWEEP_ALL_DONE_SUCCESS sentinel in the output stream.\n")

    max_attempts = 3
    last_output: str = ""
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"[sweep_v9] Retrying submission (attempt {attempt}/{max_attempts})...")
            _ensure_workspace_running(args.workspace)

        t_start = time.time()
        _remote_process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sentinel_seen = False
        out_lines: list[str] = []
        try:
            assert _remote_process.stdout is not None
            for raw_line in _remote_process.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                print(line, end="", flush=True)
                if "SWEEP_ALL_DONE_SUCCESS" in line:
                    sentinel_seen = True
                # Keep a bounded tail of output for retry decision.
                out_lines.append(line)
                if len(out_lines) > 200:
                    out_lines = out_lines[-200:]
            _remote_process.wait()
        except KeyboardInterrupt:
            _signal_handler(signal.SIGINT, None)

        rc = _remote_process.returncode
        elapsed = (time.time() - t_start) / 60
        last_output = "".join(out_lines)

        if rc == 0 and sentinel_seen:
            _print_banner(f"Sweep complete  |  total time: {elapsed:.1f} min")
            print("[sweep_v9] All models completed successfully!")
            return

        # Special-case: Anyscale occasionally errors if workspace is not fully RUNNING.
        if (
            attempt < max_attempts
            and ("Workspace must be running" in last_output or "WorkspaceState.RUNNING" in last_output)
        ):
            print("[sweep_v9] Workspace not ready for run_command yet — waiting 30s then retrying...")
            time.sleep(30)
            continue

        reason = f"rc={rc}" if rc != 0 else "sentinel not found (remote script exited early)"
        _print_banner(f"Sweep FAILED  |  {reason}  |  elapsed: {elapsed:.1f} min")
        print(f"[sweep_v9] Remote script did not complete successfully ({reason}).")
        print("[sweep_v9] Check the output above for the model that failed.")
        sys.exit(rc if rc != 0 else 1)


if __name__ == "__main__":
    main()
