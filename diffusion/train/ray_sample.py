"""Sample from a trained model using Ray.

This script runs on the cluster (or locally via --no-ray).
It matches the user's request to:
1. Load a model checkpoint.
2. Pick N different samples from the test set (holdout).
3. Generate outputs for each.
4. Save artifacts (condition, target, generated) to S3.
"""
import argparse
import os
import pickle
import torch
import shutil
from pathlib import Path

# Fix python path if running as script
import sys
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from diffusion.sample.sampler import DiffusionSampler
from diffusion.utils.config import DiffusionConfig
from diffusion.aws.s3_io import split_s3_uri, join_s3_uri, download_file, upload_dir_incremental, s3_object_exists


def _extract_signals(data_obj: object) -> torch.Tensor:
    if isinstance(data_obj, dict):
        signals = data_obj.get("signals")
        if signals is None:
            signals = data_obj.get("chunks")
        if signals is None:
            raise KeyError("Expected key 'signals' or 'chunks' in loaded data")
        return signals
    if torch.is_tensor(data_obj):
        return data_obj
    raise TypeError(f"Unsupported data object type for signals: {type(data_obj)}")


def _resolve_raw_data_path(*, config: dict, s3_base: str, version: str, run_id: str) -> Path:
    """Resolve a local raw data file path, downloading from S3 if necessary.

    Sampling needs raw time-domain chunks so the sampler can compute condition phase
    and invert back to time-domain.
    """

    cfg_val = str(config.get("raw_data_path") or "").strip()
    candidate_local_paths: list[Path] = []

    if cfg_val:
        p = Path(cfg_val).expanduser()
        candidate_local_paths.append(p)
        if not p.is_absolute():
            # Also try relative to current working directory.
            candidate_local_paths.append(Path.cwd() / p)

    # Common fallbacks seen across this repo.
    candidate_local_paths.extend(
        [
            Path("data/data/raw/art_chunks.pt"),
            Path("data/data/processed/arrhythmia_chunks.pt"),
            Path("data/data/processed/non_arrhythmia_chunks.pt"),
        ]
    )

    for lp in candidate_local_paths:
        if lp.exists():
            return lp

    # If nothing exists locally, attempt to download from S3.
    # We treat s3_base as the project root prefix, so relative paths map directly.
    s3_candidates: list[tuple[str, Path]] = []

    def _add_s3_candidate(rel_posix: str, local_dest: Path) -> None:
        rel_posix = str(rel_posix).lstrip("/")
        if not rel_posix:
            return
        s3_candidates.append((join_s3_uri(s3_base, rel_posix), local_dest))

    # Prefer downloading into the expected relative location inside the repo.
    if cfg_val:
        p = Path(cfg_val)
        if not p.is_absolute():
            _add_s3_candidate(p.as_posix(), Path.cwd() / p)
        else:
            # Absolute path in config: we can't mirror it safely; download into CWD under data/.
            _add_s3_candidate(p.name, Path("data") / p.name)

    # Hardcoded known locations (single and double 'data' variants).
    for rel in [
        "data/data/raw/art_chunks.pt",
        "data/raw/art_chunks.pt",
        "data/data/processed/arrhythmia_chunks.pt",
        "data/processed/arrhythmia_chunks.pt",
        "data/data/processed/non_arrhythmia_chunks.pt",
        "data/processed/non_arrhythmia_chunks.pt",
        # Some runs store raw chunks under versioned S3 data prefixes.
        f"data/{version}/arrhythmia_chunks.pt",
        f"data/{version}/art_chunks.pt",
        # Some runs may have raw chunks under the run prefix.
        f"diffusion-results/{version}/{run_id}/data/arrhythmia_chunks.pt",
        f"diffusion-results/{version}/{run_id}/data/art_chunks.pt",
    ]:
        _add_s3_candidate(rel, Path(rel))

    for s3_uri, local_dest in s3_candidates:
        try:
            if not s3_object_exists(s3_uri):
                continue
            print(f"[ray_sample] Found raw data in S3: {s3_uri}")
            download_file(s3_uri, local_dest)
            if local_dest.exists():
                return local_dest
        except Exception as e:
            print(f"[ray_sample] Warning: failed to download candidate raw data {s3_uri}: {e}")

    # Last resort: search within a few likely S3 prefixes for any object ending with the needed filename.
    wanted_names = []
    if cfg_val:
        wanted_names.append(Path(cfg_val).name)
    wanted_names.extend(["arrhythmia_chunks.pt", "art_chunks.pt", "non_arrhythmia_chunks.pt"])
    wanted_names = [n for n in wanted_names if n]

    def _search_prefix(prefix_uri: str, *, max_scan: int = 20000) -> Path | None:
        try:
            import boto3  # type: ignore
        except Exception:
            return None

        bucket, prefix_key = split_s3_uri(prefix_uri)
        if not bucket:
            return None

        client = boto3.client("s3")
        scanned = 0
        token = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix_key, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = client.list_objects_v2(**kwargs)
            contents = resp.get("Contents") or []
            for obj in contents:
                key = str(obj.get("Key") or "")
                scanned += 1
                if scanned > max_scan:
                    return None
                for name in wanted_names:
                    if key.endswith("/" + name) or key.endswith(name):
                        s3_uri = f"s3://{bucket}/{key}"
                        # Download into a stable local path.
                        local_dest = Path("data") / name
                        print(f"[ray_sample] Found raw data by search: {s3_uri}")
                        download_file(s3_uri, local_dest)
                        if local_dest.exists():
                            return local_dest

            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if not token:
                    return None
            else:
                return None

    search_prefixes = [
        join_s3_uri(s3_base, "data"),
        join_s3_uri(s3_base, "data", "data"),
        join_s3_uri(s3_base, "diffusion-results", version, run_id),
        join_s3_uri(s3_base, "diffusion-results", version, run_id, "data"),
    ]
    for pref in search_prefixes:
        found = _search_prefix(pref)
        if found is not None:
            return found

    raise FileNotFoundError(
        "Raw data needed for sampling was not found locally or in S3. "
        "Expected a time-domain chunks file (e.g., art_chunks.pt or arrhythmia_chunks.pt). "
        f"config['raw_data_path']={cfg_val!r}; s3_base={s3_base}; version={version}; run_id={run_id}"
    )

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s3", type=str, required=True, help="S3 bucket/prefix (e.g. s3://bucket/project)")
    p.add_argument("--s3-region", type=str, default="us-east-1")
    p.add_argument("--run-id", type=str, required=True, help="Run ID to load config/checkpoint from")
    p.add_argument("--version", type=str, default="V1", help="Experiment version (default: V1)")
    p.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint name")
    p.add_argument("--num-samples", type=int, default=16, help="Number of DISTINCT test samples to process")
    p.add_argument("--num-trajectories", type=int, default=1, help="Number of random variations per test sample")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for sampling")
    p.add_argument("--use-ddim", action="store_true", help="Use DDIM sampler")
    
    # Ray args (consumed by launcher, but good to have)
    p.add_argument("--ray-address", type=str, default=None)
    p.add_argument("--ray-num-gpus", type=float, default=1.0)
    return p.parse_args()

def _run_sampling_job(args: dict) -> None:
    
    # Configure AWS
    s3_region = str(args.get("s3_region") or "")
    if s3_region:
        os.environ["AWS_DEFAULT_REGION"] = s3_region
        os.environ["AWS_REGION"] = s3_region

    # 1. Setup paths
    s3_base = str(args["s3"])
    run_id = str(args["run_id"])
    
    # We assume standard structure: s3://.../diffusion-results/V1/<run_id>/...
    # But let's find where config.pkl is.
    # Usually: <s3_base>/diffusion-results/<version>/<run_id>/config.pkl
    # Or just <s3_base>/diffusion-results/V1/<run_id> if s3_base is the project root.
    
    # Let's verify the structure by constructing the likely path
    # The runner uses: s3_run_prefix = s3_base / "diffusion-results" / version / run_id
    # We need to access that.
    
    version = str(args.get("version") or "V1")
    s3_run_prefix = join_s3_uri(s3_base, "diffusion-results", version, run_id)
    
    local_results_dir = Path("diffusion/results")
    local_run_dir = local_results_dir / version / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[ray_sample] Working directory: {os.getcwd()}")
    print(f"[ray_sample] Syncing config and checkpoint from {s3_run_prefix}...")

    # 2. Download config and checkpoint
    config_s3 = join_s3_uri(s3_run_prefix, "config.pkl")
    config_local = local_run_dir / "config.pkl"
    
    checkpoint_name = str(args.get("checkpoint") or "best_model.pt")
    ckpt_s3 = join_s3_uri(s3_run_prefix, "checkpoints", checkpoint_name)
    ckpt_local = local_run_dir / "checkpoints" / checkpoint_name
    ckpt_local.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        download_file(config_s3, config_local)
        print(f"[ray_sample] Downloaded config")
    except Exception as e:
        print(f"[ray_sample] Failed to download config: {e}")
        # fallback? 
        raise

    try:
        download_file(ckpt_s3, ckpt_local)
        print(f"[ray_sample] Downloaded checkpoint: {checkpoint_name}")
    except Exception as e:
        print(f"[ray_sample] Failed to download checkpoint: {e}")
        raise

    # 3. Initialize Sampler
    # We need to mock the directory structure Sampler expects.
    # It expects: results_dir / version / checkpoints / ...
    # We have: diffusion/results / V1 / 2026... / checkpoints
    # Sampler init: results_dir=..., version=...
    # It looks for config at results_dir/version/config.pkl. 
    # But our download put it at diffusion/results/V1/2026.../config.pkl
    # Valid layout for Sampler is:
    #   root/V1/config.pkl
    #   root/V1/checkpoints/file.pt
    # So we should treat `diffusion/results/V1/2026...` as the "version root" roughly, 
    # OR we need to restructure.
    
    # Let's just create a temporary symlink structure or move files to match Sampler expectations?
    # Actually, Sampler takes `results_dir` and `version`.
    # It constructs `version_dir = results_dir / version`.
    # If we pass results_dir="diffusion/results/V1", version=run_id, it might work?
    #   version_dir = diffusion/results/V1/2026...
    #   config -> diffusion/results/V1/2026.../config.pkl
    #   ckpt -> diffusion/results/V1/2026.../checkpoints/ckpt.pt
    # That matches exactly what we downloaded!
    
    print(f"[ray_sample] Initializing sampler with root={local_results_dir}/{version}, ver={run_id}")
    
    try:
        sampler = DiffusionSampler(
            results_dir=local_results_dir / version,  # This is the parent of run_id
            version=run_id,         # This is the "version" (subfolder)
            checkpoint_name=checkpoint_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"[ray_sample] Error initializing sampler: {e}")
        raise

    # 4. Load Raw Data for conditions (time-domain chunks)
    raw_path = _resolve_raw_data_path(config=sampler.config, s3_base=s3_base, version=version, run_id=run_id)
    print(f"[ray_sample] Loading raw time-domain signals from {raw_path}")
    signals = _extract_signals(torch.load(raw_path))
        
    # Select N *distinct* samples from the fixed test holdout.
    holdout_n = int(sampler.config.get('test_holdout_count', 0) or 0)
    holdout_from_end = bool(sampler.config.get('test_holdout_from_end', True))

    total_files = int(signals.shape[0]) if hasattr(signals, "shape") else len(signals)
    if holdout_n > 0 and total_files > holdout_n:
        if holdout_from_end:
            available_pool = signals[-holdout_n:]
            print(f"[ray_sample] Selecting from holdout set (last {holdout_n} items)")
        else:
            available_pool = signals[:holdout_n]
            print(f"[ray_sample] Selecting from holdout set (first {holdout_n} items)")
    else:
        available_pool = signals
        print(f"[ray_sample] Selecting from full dataset ({total_files} items)")
        
    num_needed = int(args.get("num_samples") or 16)
    if len(available_pool) < num_needed:
        print(f"[ray_sample] Warning: Requested {num_needed} samples but only {len(available_pool)} available.")
        num_needed = len(available_pool)
        
    # Pick distinct indices (deterministically if possible for reproducibility)
    # We just take the *last* N samples to be consistent? Or random?
    # User said "preprocess 16 different segments... going backwards".
    # So taking the last N is a good strategy.
    
    # Last N:
    selected_signals = available_pool[-num_needed:]
    # Reverse to match "stack them jointly going backwards" comment?
    # selected_signals = selected_signals.flip(dims=[0]) 
    
    print(f"[ray_sample] Selected {len(selected_signals)} samples for generation.")
    
    # 5. Run Sampling
    # sampler.sample() handles the loop over conditions.
    # It saves to results_dir/version/samples/...
    
    # We might want to clear old samples?
    # shutil.rmtree(local_run_dir / "samples", ignore_errors=True)
    
    results = sampler.sample(
        raw_condition_signals=selected_signals,
        num_inference_steps=None, # Use config default
        use_ddim=bool(args.get("use_ddim") or False),
        num_trajectories=int(args.get("num_trajectories") or 1),
        batch_size=int(args.get("batch_size") or 16),
    )
    
    # 6. Upload Results to S3
    print("[ray_sample] Sampling complete. Uploading artifacts to S3...")
    
    # Sampler saves to: <version_dir>/samples/ddim/...
    # local_run_dir is <version_dir>
    samples_local = local_run_dir / "samples"
    
    # Destination: <s3_run_prefix>/samples
    samples_s3 = join_s3_uri(s3_run_prefix, "samples")
    
    state_path = local_run_dir / ".s3_upload_state_samples.json"
    upload_dir_incremental(samples_local, samples_s3, state_path=state_path)
    print(f"[ray_sample] Uploaded samples to {samples_s3}")


def main():
    args = _parse_args()

    # Schedule the actual sampling on a Ray worker (GPU), not on the head node.
    # This avoids CUDA-unavailable head nodes and matches the training workflow.
    try:
        import ray  # type: ignore
    except Exception as e:
        raise RuntimeError("ray is required for remote sampling") from e

    addr = args.ray_address or "auto"
    ray.init(address=addr, ignore_reinit_error=True)

    num_gpus = float(getattr(args, "ray_num_gpus", 1.0) or 1.0)

    @ray.remote(num_gpus=num_gpus)
    def _remote_job(args_dict: dict) -> None:
        _run_sampling_job(args_dict)

    ray.get(_remote_job.remote(vars(args)))

if __name__ == "__main__":
    main()
