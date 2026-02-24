"""Safe-ish cleanup helper for project S3 prefixes.

This is a convenience wrapper around deleting the S3 prefixes this project uses:
- <base>/data/<version>/              (preprocessed tensors + normalizer)
- <base>/diffusion-results/<version>/ (checkpoints/logs/samples per run)

Examples:
  # Dry-run (prints object counts and bytes)
  uv run python -m diffusion.aws.cleanup_project_s3 \
        --region us-east-1 --what data

  # Actually delete the cached preprocessed dataset
  uv run python -m diffusion.aws.cleanup_project_s3 \
        --region us-east-1 --what data --yes

  # Delete both dataset cache and run artifacts
  uv run python -m diffusion.aws.cleanup_project_s3 \
        --region us-east-1 --what all --yes

Notes:
- Destructive. Defaults to dry-run.
- Uses boto3 credentials from your environment.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from diffusion.aws.s3_io import delete_prefix, join_s3_uri, normalize_s3_uri, split_s3_uri


@dataclass(frozen=True)
class PrefixSummary:
    prefix: str
    bucket: str
    key_prefix: str
    objects: int
    bytes_total: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--s3",
        default="s3://anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project",
        help=(
            "Base S3 URI prefix, e.g. s3://bucket/ee269project "
            "(default: anyscale-production-data-cld-uvdckbb6ukmk9fu8g3nxxudemt/ee269project)"
        ),
    )
    p.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    p.add_argument("--version", default="V7", help="Dataset/results version (default: V1)")
    p.add_argument(
        "--what",
        choices=["data", "results", "all"],
        default="data",
        help="What to delete (default: data)",
    )
    p.add_argument("--yes", action="store_true", help="Actually delete (otherwise dry-run)")
    return p.parse_args()


def _require_boto3_client(region: str):
    from diffusion.aws.s3_io import _require_boto3

    boto3 = _require_boto3()
    return boto3.client("s3", region_name=region)


def _summarize_prefix(*, client, prefix_uri: str) -> PrefixSummary:
    prefix_uri = normalize_s3_uri(prefix_uri)
    bucket, key_prefix = split_s3_uri(prefix_uri)

    # Always list as a "directory" prefix.
    if key_prefix and not key_prefix.endswith("/"):
        key_prefix = key_prefix + "/"

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)

    n = 0
    total = 0
    for page in pages:
        for obj in page.get("Contents", []) or []:
            n += 1
            total += int(obj.get("Size") or 0)

    return PrefixSummary(
        prefix=f"s3://{bucket}/{key_prefix}" if key_prefix else f"s3://{bucket}",
        bucket=bucket,
        key_prefix=key_prefix,
        objects=n,
        bytes_total=total,
    )


def _fmt_bytes(n: int) -> str:
    n = int(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024 or unit == "TiB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def main() -> None:
    args = _parse_args()

    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_DEFAULT_REGION", str(args.region))
    os.environ.setdefault("AWS_REGION", str(args.region))

    base = normalize_s3_uri(args.s3)
    version = str(args.version)

    data_prefix = join_s3_uri(base, "data", version)
    results_prefix = join_s3_uri(base, "diffusion-results", version)

    targets: list[str] = []
    if args.what in {"data", "all"}:
        targets.append(data_prefix)
    if args.what in {"results", "all"}:
        targets.append(results_prefix)

    client = _require_boto3_client(str(args.region))

    summaries: list[PrefixSummary] = []
    for t in targets:
        s = _summarize_prefix(client=client, prefix_uri=t)
        summaries.append(s)

    print("[cleanup_project_s3] Targets:")
    for s in summaries:
        print(f"  - {s.prefix}\tobjects={s.objects}\tbytes={_fmt_bytes(s.bytes_total)}")

    if not args.yes:
        print("[cleanup_project_s3] Dry-run only. Re-run with --yes to actually delete.")
        return

    for s in summaries:
        if s.objects <= 0:
            print(f"[cleanup_project_s3] Skipping empty prefix: {s.prefix}")
            continue
        print(f"[cleanup_project_s3] Deleting: {s.prefix}")
        delete_prefix(s.prefix)

    print("[cleanup_project_s3] Done.")


if __name__ == "__main__":
    main()
