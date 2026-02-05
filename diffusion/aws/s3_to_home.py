"""Download an S3 prefix to a local directory (home-friendly).

Example:
  uv run python -m diffusion.aws.s3_to_home \
    --prefix s3://YOUR_BUCKET/ee269project/data/V1 \
    --region us-east-1 \
    --dst ~/ee269project_s3_backup

This is intentionally simple: it mirrors keys under the prefix into dst.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from diffusion.aws.s3_io import normalize_s3_uri, split_s3_uri


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--prefix", required=True, help="S3 prefix to download, e.g. s3://bucket/ee269project")
    p.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    p.add_argument(
        "--dst",
        default=str(Path.home() / "ee269project_s3"),
        help="Destination directory (default: ~/ee269project_s3)",
    )
    return p.parse_args()


def _require_boto3_client(region: str):
    from diffusion.aws.s3_io import _require_boto3

    boto3 = _require_boto3()
    return boto3.client("s3", region_name=region)


def main() -> None:
    args = _parse_args()

    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_DEFAULT_REGION", str(args.region))
    os.environ.setdefault("AWS_REGION", str(args.region))

    s3_prefix = normalize_s3_uri(args.prefix)
    bucket, key_prefix = split_s3_uri(s3_prefix)

    # Always treat as a prefix.
    if key_prefix and not key_prefix.endswith("/"):
        key_prefix = key_prefix + "/"

    dst_root = Path(os.path.expanduser(args.dst)).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    client = _require_boto3_client(str(args.region))

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)

    downloaded = 0
    skipped = 0

    for page in pages:
        for obj in page.get("Contents", []) or []:
            key = str(obj.get("Key") or "")
            if not key:
                continue
            if key.endswith("/"):
                continue

            rel = key[len(key_prefix) :] if key_prefix and key.startswith(key_prefix) else key
            rel = rel.lstrip("/")
            if not rel:
                continue

            local_path = dst_root / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if same size already exists locally.
            try:
                size_s3 = int(obj.get("Size") or 0)
                if local_path.exists() and local_path.stat().st_size == size_s3:
                    skipped += 1
                    continue
            except Exception:
                pass

            client.download_file(bucket, key, str(local_path))
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"[s3_to_home] downloaded={downloaded} skipped={skipped}")

    print(f"[s3_to_home] Done. downloaded={downloaded} skipped={skipped} -> {dst_root}")


if __name__ == "__main__":
    main()
