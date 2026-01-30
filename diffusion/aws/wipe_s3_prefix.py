"""Wipe an S3 prefix (recursive delete), with S3 Express directory-bucket friendly behavior.

This is intended for deleting everything under a project prefix, e.g.:
  s3://ee269--use2-az1--x-s3/ee269project/

Dry-run (shows counts):
  uv run python -m diffusion.aws.wipe_s3_prefix --prefix "s3://.../ee269project" \
    --region us-east-2

Actually delete:
  uv run python -m diffusion.aws.wipe_s3_prefix --prefix "s3://.../ee269project" \
    --region us-east-2 --yes

Notes:
- Directory buckets can keep "directory entries" (keys ending in '/'). This script
  deletes all objects found under the prefix and then attempts to delete the
  prefix "directory object" itself.
- Destructive. Use with care.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from diffusion.aws.s3_io import normalize_s3_uri, split_s3_uri


def _maybe_set_repo_aws_files() -> None:
    aws_dir = Path(__file__).resolve().parent
    repo_creds = aws_dir / "credentials"
    repo_cfg = aws_dir / "config"

    if "AWS_SHARED_CREDENTIALS_FILE" not in os.environ and repo_creds.exists():
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(repo_creds)
    if "AWS_CONFIG_FILE" not in os.environ and repo_cfg.exists():
        os.environ["AWS_CONFIG_FILE"] = str(repo_cfg)


def _require_boto3_client(region: str):
    from diffusion.aws.s3_io import _require_boto3

    boto3 = _require_boto3()
    return boto3.client("s3", region_name=region)


def _chunks(xs: Iterable[str], n: int) -> Iterable[list[str]]:
    buf: list[str] = []
    for x in xs:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--prefix", required=True, help="S3 URI prefix to wipe (e.g. s3://bucket/ee269project)")
    p.add_argument("--region", default="us-east-2", help="AWS region (default: us-east-2)")
    p.add_argument("--yes", action="store_true", help="Actually delete (otherwise dry-run)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_DEFAULT_REGION", str(args.region))
    os.environ.setdefault("AWS_REGION", str(args.region))
    _maybe_set_repo_aws_files()

    s3_prefix_uri = normalize_s3_uri(args.prefix)
    bucket, key_prefix = split_s3_uri(s3_prefix_uri)

    # S3 Express directory buckets require prefixes to end with '/'.
    if key_prefix and not key_prefix.endswith("/"):
        key_prefix = key_prefix + "/"

    client = _require_boto3_client(str(args.region))

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)

    keys: list[str] = []
    for page in pages:
        for obj in page.get("Contents", []) or []:
            k = obj.get("Key")
            if k:
                keys.append(k)

    print(f"Prefix: s3://{bucket}/{key_prefix}")
    print(f"Objects matched: {len(keys)}")

    if not args.yes:
        print("Dry-run only. Re-run with --yes to actually delete.")
        return

    deleted = 0
    for batch in _chunks(keys, 1000):
        resp = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch]},
        )
        deleted += len(resp.get("Deleted", []) or batch)

    # Attempt to delete the "directory object" itself.
    if key_prefix:
        try:
            client.delete_object(Bucket=bucket, Key=key_prefix)
        except Exception:
            pass

    print(f"Deleted (approx): {deleted}")
    print("Done.")


if __name__ == "__main__":
    main()
