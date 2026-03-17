"""Download training/sampling artifacts from S3 and optionally delete from S3.

This script:
1. Downloads all run artifacts from S3 to local directory structure
2. Preserves the same folder structure (diffusion-results/<version>/<run_id>/)
3. Optionally deletes the S3 data after successful download

Downloaded artifacts include:
- Checkpoints (best_model.pt, checkpoint_epoch_*.pt)
- Tensorboard logs (logs/diffusion_training/)
- Generated samples (samples/*.png, samples/*.pt)
- Config and metadata (config.pkl)

Usage Examples (copy/paste safe):

    # Download essentials only (fast - best for most cases)
    uv run python -m diffusion.aws.download_s3_artifacts --version V4 --run-id 20260213_202346 --best-only
    uv run diffusion/aws/download_s3_artifacts.py --version V1/Sigmoid --run-id 20260216_161119 

    # Download logs and samples only (skip all checkpoints)
    uv run python -m diffusion.aws.download_s3_artifacts --version V4 --run-id 20260213_202346 --no-checkpoints

    # Download samples/ ONLY (fastest - just the generated .pt/.png files)
    uv run python -m diffusion.aws.download_s3_artifacts --version V7 --run-id 20260218_2000313 --samples-only

    # Download specific run (keep in S3)
    uv run python -m diffusion.aws.download_s3_artifacts --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --version V4 --run-id 20260213_202346

    # Download specific run and DELETE from S3 after
    uv run python -m diffusion.aws.download_s3_artifacts --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --version V4 --run-id 20260213_202346 --delete-after

    # Download ALL runs (keep in S3)
    uv run python -m diffusion.aws.download_s3_artifacts --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --version V4 --all

    # Download ALL runs and DELETE from S3 after
    uv run python -m diffusion.aws.download_s3_artifacts --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --version V4 --all --delete-after

    # Preview what will happen (dry run)
    uv run python -m diffusion.aws.download_s3_artifacts --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --version V4 --all --delete-after --dry-run

    # Download everything including preprocessed data cache
    uv run python -m diffusion.aws.download_s3_artifacts --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --version V4 --all --include-data

After downloading, view tensorboard logs:
    uv run tensorboard --logdir diffusion/results/V4/20260213_202346/logs
  open http://localhost:6006

Safety Features:
- Confirmation prompt when using --delete-after (unless --dry-run)
- Dry run mode shows what would happen without making changes
- Progress bars for both download and deletion
- Batch deletion for efficiency (up to 1000 files per request)

Options:
  --s3 <uri>           S3 URI (required) - e.g., s3://bucket/ee269project
  --region <region>    AWS region (default: us-east-1)
  --run-id <id>        Download specific run (e.g., 20260204_215551)
  --all                Download all runs (mutually exclusive with --run-id)
  --local-dir <path>   Custom local directory (default: diffusion/results/V1/)
    --data-only          Download ONLY the preprocessed data cache under data/<version>/
  --delete-after       Delete from S3 after download (DESTRUCTIVE!)
  --dry-run            Preview without making changes
  --include-data       Also download preprocessed data cache
  --best-only          Only download best_model.pt (skip checkpoint_epoch_*.pt) - RECOMMENDED
  --no-checkpoints     Skip all checkpoints (only logs and samples)

For detailed documentation, see:
  diffusion/aws/TENSORBOARD_S3_GUIDE.md
  diffusion/aws/COMPLETE_GUIDE.md
"""

from __future__ import annotations

import argparse
import boto3
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download S3 artifacts and optionally clean up S3")
    p.add_argument(
        "--s3",
        type=str,
        default="s3://anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project",
        help="S3 URI prefix (default: Anyscale production bucket)",
    )
    p.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    p.add_argument(
        "--version",
        type=str,
        default="V1",
        help="Experiment version under diffusion-results/ (default: V1)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Specific run to download (e.g. 20260204_215551, or model/run_id like "
            "uniform__linear/20260204_215551). Mutually exclusive with --all."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download all runs. Mutually exclusive with --run-id.",
    )

    p.add_argument(
        "--data-only",
        action="store_true",
        help="Download ONLY the preprocessed data cache under data/<version>/ (no diffusion-results runs).",
    )
    p.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help=(
            "Local destination directory. Default: ./diffusion/results/<version>/ "
            "(or ./s3_downloads/data_<version>/ when using --data-only)."
        ),
    )
    p.add_argument(
        "--delete-after",
        action="store_true",
        help="Delete from S3 after successful download (DESTRUCTIVE - use with caution!)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded/deleted without doing it",
    )
    p.add_argument(
        "--include-data",
        action="store_true",
        help="Also download preprocessed data cache (train_*.pt, normalizer_params.pkl)",
    )
    p.add_argument(
        "--best-only",
        action="store_true",
        help="Only download best_model.pt (skip checkpoint_epoch_*.pt)",
    )
    p.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Skip all checkpoints (only download logs and samples)",
    )
    p.add_argument(
        "--samples-only",
        action="store_true",
        help="Only download files under samples/ (skip checkpoints, logs, config, data cache)",
    )

    return p.parse_args()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"S3 URI must start with s3://: {uri}")
    
    uri = uri[5:]  # Remove s3://
    if "/" in uri:
        bucket, prefix = uri.split("/", 1)
    else:
        bucket, prefix = uri, ""
    
    return bucket, prefix


_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}")


def _looks_like_run_id(component: str) -> bool:
    component = str(component or "").strip()
    if not component:
        return False
    return bool(_RUN_ID_RE.match(component))


def _list_run_paths(s3_client, bucket: str, prefix: str, version: str = "V1") -> List[str]:
    """List run paths under s3://bucket/prefix/diffusion-results/<version>/.

    Supports two layouts:
      1) diffusion-results/<version>/<run_id>/...
      2) diffusion-results/<version>/<model>/<run_id>/...

    Returns run paths relative to diffusion-results/<version>/, e.g.:
      - "20260204_215551"
      - "uniform__linear/20260204_215551"
    """
    version = str(version or "V1").strip() or "V1"
    runs_prefix = f"{prefix}/diffusion-results/{version}/".lstrip("/")
    
    paginator = s3_client.get_paginator("list_objects_v2")
    run_paths: set[str] = set()
    
    for page in paginator.paginate(Bucket=bucket, Prefix=runs_prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            child_prefix = common_prefix["Prefix"]
            child = child_prefix[len(runs_prefix) :].rstrip("/")
            if not child:
                continue

            if _looks_like_run_id(child):
                # Flat layout: diffusion-results/<version>/<run_id>/
                run_paths.add(child)
                continue

            # Nested layout: diffusion-results/<version>/<model>/<run_id>/
            nested_prefix = f"{runs_prefix}{child}/"
            for page2 in paginator.paginate(Bucket=bucket, Prefix=nested_prefix, Delimiter="/"):
                for cp2 in page2.get("CommonPrefixes", []):
                    maybe_run = cp2["Prefix"][len(nested_prefix) :].rstrip("/")
                    if not maybe_run:
                        continue
                    if _looks_like_run_id(maybe_run):
                        run_paths.add(f"{child}/{maybe_run}")

    return sorted(run_paths)


def _get_run_size(s3_client, bucket: str, prefix: str, run_path: str, version: str = "V1") -> tuple[int, int]:
    """Get total size of a run in bytes. Returns (num_files, total_bytes)."""
    version = str(version or "V1").strip() or "V1"
    run_path = str(run_path).strip().strip("/")
    s3_run_prefix = f"{prefix}/diffusion-results/{version}/{run_path}/".lstrip("/")
    
    paginator = s3_client.get_paginator("list_objects_v2")
    total_bytes = 0
    num_files = 0
    
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_run_prefix):
        for obj in page.get("Contents", []):
            total_bytes += obj.get("Size", 0)
            num_files += 1
    
    return num_files, total_bytes


def _format_size(bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def _should_download(key: str, best_only: bool, no_checkpoints: bool, samples_only: bool = False) -> bool:
    """Check if a file should be downloaded based on filters."""
    if samples_only:
        return "/samples/" in key

    # Always download config.pkl
    if key.endswith("config.pkl"):
        return True
    
    if no_checkpoints and "/checkpoints/" in key:
        return False
    
    if best_only and "/checkpoints/" in key:
        # Only download best_model.pt, skip checkpoint_epoch_*.pt
        if "checkpoint_epoch_" in key:
            return False
    
    return True


def _download_run(
    s3_client,
    bucket: str,
    prefix: str,
    run_path: str,
    version: str,
    local_base: Path,
    dry_run: bool = False,
    best_only: bool = False,
    no_checkpoints: bool = False,
    samples_only: bool = False,
) -> tuple[List[str], List[str]]:
    """Download files for a run. Returns (downloaded_keys, all_keys_in_run)."""
    version = str(version or "V1").strip() or "V1"
    run_path = str(run_path).strip().strip("/")
    s3_run_prefix = f"{prefix}/diffusion-results/{version}/{run_path}/".lstrip("/")
    local_run_dir = local_base / Path(run_path)
    
    if not dry_run:
        local_run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Download] Run: {run_path}")
    print(f"  S3: s3://{bucket}/{s3_run_prefix}")
    print(f"  Local: {local_run_dir}")
    if best_only:
        print(f"  Filter: best_model.pt + logs + samples + config only")
    if no_checkpoints:
        print(f"  Filter: skipping all checkpoints")
    if samples_only:
        print(f"  Filter: samples/ only")
    
    paginator = s3_client.get_paginator("list_objects_v2")
    downloaded_keys = []
    
    # Collect all objects first
    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_run_prefix):
        all_objects.extend(page.get("Contents", []))
    
    if not all_objects:
        print(f"  No files found for run {run_path}")
        return [], []
    
    all_keys = [obj["Key"] for obj in all_objects]
    
    # Filter objects based on options
    filtered_objects = [obj for obj in all_objects if _should_download(obj["Key"], best_only, no_checkpoints, samples_only)]
    
    print(f"  Found {len(all_objects)} files total, downloading {len(filtered_objects)} files")
    
    if dry_run:
        for obj in filtered_objects[:5]:  # Show first 5
            print(f"    Would download: {obj['Key']}")
        if len(filtered_objects) > 5:
            print(f"    ... and {len(filtered_objects) - 5} more files")
        skipped = len(all_objects) - len(filtered_objects)
        if skipped > 0:
            print(f"  Would skip {skipped} files (filtered)")
        return [obj["Key"] for obj in filtered_objects], all_keys
    
    # Download in parallel (16 threads — S3 handles concurrent GETs well).
    def _dl(obj):
        s3_key = obj["Key"]
        relative_path = s3_key[len(s3_run_prefix):]
        local_path = local_run_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, s3_key, str(local_path))
        return s3_key

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_dl, obj): obj for obj in filtered_objects}
        with tqdm(total=len(futures), desc=f"  Downloading {run_path}", unit="file") as bar:
            for fut in as_completed(futures):
                downloaded_keys.append(fut.result())
                bar.update(1)

    return downloaded_keys, all_keys


def _download_data_cache(
    s3_client,
    bucket: str,
    prefix: str,
    version: str,
    local_data_dir: Path,
    dry_run: bool = False,
) -> List[str]:
    """Download preprocessed data cache. Returns list of S3 keys downloaded."""
    version = str(version or "V1").strip() or "V1"
    s3_data_prefix = f"{prefix}/data/{version}/".lstrip("/")

    local_data_dir = Path(local_data_dir).resolve()
    
    if not dry_run:
        local_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Download] Preprocessed data cache")
    print(f"  S3: s3://{bucket}/{s3_data_prefix}")
    print(f"  Local: {local_data_dir}")
    
    paginator = s3_client.get_paginator("list_objects_v2")
    downloaded_keys = []
    
    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_data_prefix):
        all_objects.extend(page.get("Contents", []))
    
    if not all_objects:
        print(f"  No preprocessed data found")
        return []
    
    print(f"  Found {len(all_objects)} files")
    
    if dry_run:
        for obj in all_objects:
            print(f"    Would download: {obj['Key']}")
        return [obj["Key"] for obj in all_objects]
    
    for obj in tqdm(all_objects, desc="  Downloading data cache", unit="file"):
        s3_key = obj["Key"]
        relative_path = s3_key[len(s3_data_prefix) :]
        local_path = local_data_dir / relative_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if same size already exists locally.
        try:
            size_s3 = int(obj.get("Size") or 0)
            if local_path.exists() and local_path.stat().st_size == size_s3:
                continue
        except Exception:
            pass

        s3_client.download_file(bucket, s3_key, str(local_path))
        downloaded_keys.append(s3_key)
    
    return downloaded_keys


def _delete_s3_keys(
    s3_client,
    bucket: str,
    keys: List[str],
    dry_run: bool = False,
) -> None:
    """Delete a list of S3 keys."""
    if not keys:
        return
    
    print(f"\n[Delete] Removing {len(keys)} files from S3...")
    
    if dry_run:
        print(f"  Would delete {len(keys)} files (showing first 5):")
        for key in keys[:5]:
            print(f"    {key}")
        if len(keys) > 5:
            print(f"    ... and {len(keys) - 5} more")
        return
    
    # Delete in batches (S3 supports up to 1000 per request)
    batch_size = 1000
    for i in tqdm(range(0, len(keys), batch_size), desc="  Deleting batches", unit="batch"):
        batch = keys[i:i+batch_size]
        delete_objects = [{"Key": key} for key in batch]
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": delete_objects, "Quiet": True}
        )
    
    print(f"  ✅ Deleted {len(keys)} files from S3")


def main() -> None:
    args = _parse_args()

    version = str(args.version or "V1").strip() or "V1"
    
    if args.data_only:
        if args.run_id or args.all:
            raise SystemExit("Error: --data-only cannot be combined with --run-id/--all")
    else:
        if args.run_id and args.all:
            raise SystemExit("Error: --run-id and --all are mutually exclusive")

        if not args.run_id and not args.all:
            raise SystemExit("Error: must specify either --run-id or --all (or use --data-only)")
    
    bucket, prefix = _parse_s3_uri(args.s3)

    # Default local directory
    if args.data_only:
        local_data_dir = Path(args.local_dir) if args.local_dir else (Path(__file__).resolve().parents[2] / "s3_downloads" / f"data_{version}")
        local_data_dir = local_data_dir.resolve()
        local_data_dir.mkdir(parents=True, exist_ok=True)
        local_base = local_data_dir
    else:
        if args.local_dir:
            local_base = Path(args.local_dir)
        else:
            # Default to diffusion/results/<version>/
            local_base = Path(__file__).parent.parent / "results" / version

        local_base = local_base.resolve()
        local_base.mkdir(parents=True, exist_ok=True)
    
    print(f"[Setup]")
    print(f"  S3 bucket: {bucket}")
    print(f"  S3 prefix: {prefix}")
    print(f"  Local dir: {local_base}")
    print(f"  Region: {args.region}")
    print(f"  Delete after: {args.delete_after}")
    print(f"  Dry run: {args.dry_run}")
    print()

    if args.data_only:
        if args.delete_after:
            raise SystemExit("Error: --delete-after is not supported with --data-only")

        s3_client = boto3.client("s3", region_name=args.region)
        downloaded = _download_data_cache(
            s3_client,
            bucket,
            prefix,
            version,
            local_base,
            dry_run=args.dry_run,
        )
        print("\n[Summary]")
        print(f"  Downloaded: {len(downloaded)} files")
        print(f"  Local dir: {local_base}")
        if not args.dry_run:
            print("\n✅ Done!")
        else:
            print("\n✅ Dry run complete (no changes made)")
        return
    
    if args.delete_after and not args.dry_run:
        response = input("⚠️  WARNING: This will DELETE data from S3 after download. Continue? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return
    
    s3_client = boto3.client("s3", region_name=args.region)
    
    # Determine which runs to download
    if args.all:
        print("[Scanning] Finding all runs in S3...")
        run_paths = _list_run_paths(s3_client, bucket, prefix, version)
        if not run_paths:
            print("No runs found in S3.")
            return
        preview = ", ".join(run_paths[:20])
        suffix = "" if len(run_paths) <= 20 else f" ... (+{len(run_paths) - 20} more)"
        print(f"Found {len(run_paths)} runs: {preview}{suffix}")
        print()
    else:
        run_paths = [args.run_id]
    
    # Show S3 size before
    if not args.dry_run:
        print("[S3 Storage Before]")
        total_files_before = 0
        total_bytes_before = 0
        for run_path in run_paths:
            num_files, num_bytes = _get_run_size(s3_client, bucket, prefix, run_path, version)
            total_files_before += num_files
            total_bytes_before += num_bytes
            print(f"  {run_path}: {num_files} files, {_format_size(num_bytes)}")
        print(f"  TOTAL: {total_files_before} files, {_format_size(total_bytes_before)}")
        print()
    
    # Download each run
    all_downloaded_keys = []
    all_keys_to_delete = []  # ALL keys in the runs (for --delete-after)
    for run_path in run_paths:
        downloaded_keys, all_run_keys = _download_run(
            s3_client, bucket, prefix, run_path, version, local_base,
            dry_run=args.dry_run,
            best_only=args.best_only,
            no_checkpoints=args.no_checkpoints,
            samples_only=args.samples_only,
        )
        all_downloaded_keys.extend(downloaded_keys)
        all_keys_to_delete.extend(all_run_keys)
        print()
    
    # Optionally download preprocessed data cache
    if args.include_data:
        data_dir = local_base.parent.parent / "data" / version
        data_keys = _download_data_cache(s3_client, bucket, prefix, version, data_dir, dry_run=args.dry_run)
        all_downloaded_keys.extend(data_keys)
        # Don't delete data cache
        print()
    
    # Summary
    print(f"[Summary]")
    print(f"  Downloaded: {len(all_downloaded_keys)} files")
    print(f"  Local dir: {local_base}")
    
    # Delete from S3 if requested (delete ALL files in the run, not just downloaded)
    if args.delete_after and all_keys_to_delete:
        print(f"\n⚠️  --delete-after: Will delete ALL {len(all_keys_to_delete)} files in the run(s) from S3")
        if args.best_only or args.no_checkpoints:
            print(f"  (Including filtered files that were not downloaded)")
        _delete_s3_keys(s3_client, bucket, all_keys_to_delete, dry_run=args.dry_run)
        
        # Show S3 size after
        if not args.dry_run:
            print("\n[S3 Storage After]")
            total_files_after = 0
            total_bytes_after = 0
            for run_path in run_paths:
                num_files, num_bytes = _get_run_size(s3_client, bucket, prefix, run_path, version)
                total_files_after += num_files
                total_bytes_after += num_bytes
                if num_files > 0:
                    print(f"  {run_path}: {num_files} files, {_format_size(num_bytes)}")
            if total_files_after == 0:
                print(f"  All runs deleted from S3 ✅")
            else:
                print(f"  TOTAL: {total_files_after} files, {_format_size(total_bytes_after)}")
            
            saved = total_bytes_before - total_bytes_after
            print(f"\n  💾 Freed: {_format_size(saved)} from S3")
    
    if not args.dry_run:
        print("\n✅ Done!")
    else:
        print("\n✅ Dry run complete (no changes made)")


if __name__ == "__main__":
    main()


