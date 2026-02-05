"""Download all training/sampling artifacts from S3 and optionally delete from S3.

This script:
1. Downloads all run artifacts from S3 to local directory structure
2. Preserves the same folder structure (diffusion-results/V1/<run_id>/)
3. Optionally deletes the S3 data after successful download

Downloaded artifacts include:
- Checkpoints (best_model.pt, checkpoint_epoch_*.pt)
- Tensorboard logs (logs/diffusion_training/)
- Generated samples (samples/*.png, samples/*.pt)
- Config and metadata (config.pkl)

Usage Examples:

  # Download essentials only (fast - best for most cases)
  uv run python -m diffusion.aws.download_s3_artifacts \\
    --run-id 20260204_215551 \\
    --best-only

  # Download logs and samples only (skip all checkpoints)
  uv run python -m diffusion.aws.download_s3_artifacts \\
    --run-id 20260204_215551 \\
    --no-checkpoints

  # Download specific run (keep in S3)
  uv run python -m diffusion.aws.download_s3_artifacts \\
    --s3 s3://YOUR_BUCKET/ee269project \\
    --region us-east-1 \\
    --run-id 20260204_215551

  # Download specific run and DELETE from S3 after
  uv run python -m diffusion.aws.download_s3_artifacts \\
    --s3 s3://YOUR_BUCKET/ee269project \\
    --region us-east-1 \\
    --run-id 20260204_215551 \\
    --delete-after

  # Download ALL runs (keep in S3)
  uv run python -m diffusion.aws.download_s3_artifacts \\
    --s3 s3://YOUR_BUCKET/ee269project \\
    --region us-east-1 \\
    --all

  # Download ALL runs and DELETE from S3 after
  uv run python -m diffusion.aws.download_s3_artifacts \\
    --s3 s3://YOUR_BUCKET/ee269project \\
    --region us-east-1 \\
    --all \\
    --delete-after

  # Preview what will happen (dry run)
  uv run python -m diffusion.aws.download_s3_artifacts \\
    --s3 s3://YOUR_BUCKET/ee269project \\
    --region us-east-1 \\
    --all \\
    --delete-after \\
    --dry-run

  # Download everything including preprocessed data cache
  uv run python -m diffusion.aws.download_s3_artifacts \\
    --s3 s3://YOUR_BUCKET/ee269project \\
    --region us-east-1 \\
    --all \\
    --include-data

After downloading, view tensorboard logs:
  uv run tensorboard --logdir diffusion/results/V1/20260204_215551/logs
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
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to download (e.g. 20260204_215551). Mutually exclusive with --all.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Download all runs. Mutually exclusive with --run-id.",
    )
    p.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local destination directory (default: ./diffusion/results/V1/)",
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


def _list_runs(s3_client, bucket: str, prefix: str) -> List[str]:
    """List all run IDs under s3://bucket/prefix/diffusion-results/V1/."""
    runs_prefix = f"{prefix}/diffusion-results/V1/".lstrip("/")
    
    paginator = s3_client.get_paginator("list_objects_v2")
    run_ids = set()
    
    for page in paginator.paginate(Bucket=bucket, Prefix=runs_prefix, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            run_path = common_prefix["Prefix"]
            # Extract run_id from: prefix/diffusion-results/V1/20260204_215551/
            run_id = run_path.rstrip("/").split("/")[-1]
            run_ids.add(run_id)
    
    return sorted(run_ids)


def _get_run_size(s3_client, bucket: str, prefix: str, run_id: str) -> tuple[int, int]:
    """Get total size of a run in bytes. Returns (num_files, total_bytes)."""
    s3_run_prefix = f"{prefix}/diffusion-results/V1/{run_id}/".lstrip("/")
    
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


def _should_download(key: str, best_only: bool, no_checkpoints: bool) -> bool:
    """Check if a file should be downloaded based on filters."""
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
    run_id: str,
    local_base: Path,
    dry_run: bool = False,
    best_only: bool = False,
    no_checkpoints: bool = False,
) -> tuple[List[str], List[str]]:
    """Download files for a run. Returns (downloaded_keys, all_keys_in_run)."""
    s3_run_prefix = f"{prefix}/diffusion-results/V1/{run_id}/".lstrip("/")
    local_run_dir = local_base / run_id
    
    if not dry_run:
        local_run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Download] Run: {run_id}")
    print(f"  S3: s3://{bucket}/{s3_run_prefix}")
    print(f"  Local: {local_run_dir}")
    if best_only:
        print(f"  Filter: best_model.pt + logs + samples + config only")
    if no_checkpoints:
        print(f"  Filter: skipping all checkpoints")
    
    paginator = s3_client.get_paginator("list_objects_v2")
    downloaded_keys = []
    
    # Collect all objects first
    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_run_prefix):
        all_objects.extend(page.get("Contents", []))
    
    if not all_objects:
        print(f"  No files found for run {run_id}")
        return [], []
    
    all_keys = [obj["Key"] for obj in all_objects]
    
    # Filter objects based on options
    filtered_objects = [obj for obj in all_objects if _should_download(obj["Key"], best_only, no_checkpoints)]
    
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
    
    # Download with progress bar
    for obj in tqdm(filtered_objects, desc=f"  Downloading {run_id}", unit="file"):
        s3_key = obj["Key"]
        # Compute relative path: remove s3_run_prefix from s3_key
        relative_path = s3_key[len(s3_run_prefix):]
        local_path = local_run_dir / relative_path
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, s3_key, str(local_path))
        downloaded_keys.append(s3_key)
    
    return downloaded_keys, all_keys


def _download_data_cache(
    s3_client,
    bucket: str,
    prefix: str,
    local_base: Path,
    dry_run: bool = False,
) -> List[str]:
    """Download preprocessed data cache. Returns list of S3 keys downloaded."""
    s3_data_prefix = f"{prefix}/data/V1/".lstrip("/")
    local_data_dir = local_base.parent.parent / "data" / "V1"
    
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
        relative_path = s3_key[len(s3_data_prefix):]
        local_path = local_data_dir / relative_path
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
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
    
    if args.run_id and args.all:
        raise SystemExit("Error: --run-id and --all are mutually exclusive")
    
    if not args.run_id and not args.all:
        raise SystemExit("Error: must specify either --run-id or --all")
    
    bucket, prefix = _parse_s3_uri(args.s3)
    
    # Default local directory
    if args.local_dir:
        local_base = Path(args.local_dir)
    else:
        # Default to diffusion/results/V1/
        local_base = Path(__file__).parent.parent / "results" / "V1"
    
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
    
    if args.delete_after and not args.dry_run:
        response = input("⚠️  WARNING: This will DELETE data from S3 after download. Continue? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return
    
    s3_client = boto3.client("s3", region_name=args.region)
    
    # Determine which runs to download
    if args.all:
        print("[Scanning] Finding all runs in S3...")
        run_ids = _list_runs(s3_client, bucket, prefix)
        if not run_ids:
            print("No runs found in S3.")
            return
        print(f"Found {len(run_ids)} runs: {', '.join(run_ids)}")
        print()
    else:
        run_ids = [args.run_id]
    
    # Show S3 size before
    if not args.dry_run:
        print("[S3 Storage Before]")
        total_files_before = 0
        total_bytes_before = 0
        for run_id in run_ids:
            num_files, num_bytes = _get_run_size(s3_client, bucket, prefix, run_id)
            total_files_before += num_files
            total_bytes_before += num_bytes
            print(f"  {run_id}: {num_files} files, {_format_size(num_bytes)}")
        print(f"  TOTAL: {total_files_before} files, {_format_size(total_bytes_before)}")
        print()
    
    # Download each run
    all_downloaded_keys = []
    all_keys_to_delete = []  # ALL keys in the runs (for --delete-after)
    for run_id in run_ids:
        downloaded_keys, all_run_keys = _download_run(
            s3_client, bucket, prefix, run_id, local_base, 
            dry_run=args.dry_run,
            best_only=args.best_only,
            no_checkpoints=args.no_checkpoints
        )
        all_downloaded_keys.extend(downloaded_keys)
        all_keys_to_delete.extend(all_run_keys)
        print()
    
    # Optionally download preprocessed data cache
    if args.include_data:
        data_keys = _download_data_cache(s3_client, bucket, prefix, local_base, dry_run=args.dry_run)
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
            for run_id in run_ids:
                num_files, num_bytes = _get_run_size(s3_client, bucket, prefix, run_id)
                total_files_after += num_files
                total_bytes_after += num_bytes
                if num_files > 0:
                    print(f"  {run_id}: {num_files} files, {_format_size(num_bytes)}")
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
