from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


def _maybe_set_repo_aws_files(credentials_file: str | None, config_file: str | None) -> None:
    """Prefer repo-local AWS config if present.

    This supports keeping credentials under `diffusion/aws/` while still
    avoiding committing them (they are gitignored).
    """

    if credentials_file:
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = credentials_file
    if config_file:
        os.environ["AWS_CONFIG_FILE"] = config_file

    # If explicit paths weren't provided, try repo-local defaults.
    if "AWS_SHARED_CREDENTIALS_FILE" not in os.environ:
        repo_creds = Path(__file__).resolve().parent / "credentials"
        if repo_creds.exists():
            os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(repo_creds)

    if "AWS_CONFIG_FILE" not in os.environ:
        repo_cfg = Path(__file__).resolve().parent / "config"
        if repo_cfg.exists():
            os.environ["AWS_CONFIG_FILE"] = str(repo_cfg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--bucket", type=str, required=True)
    p.add_argument("--region", type=str, default="us-east-1")
    p.add_argument("--prefix", type=str, default="ee269project/smoke-test")
    p.add_argument("--no-write", action="store_true", help="Only validate listing/head; do not put/delete test object.")
    p.add_argument(
        "--debug-auth",
        action="store_true",
        help="Print which credential source is being used (never prints secrets).",
    )
    p.add_argument(
        "--credentials-file",
        type=str,
        default=None,
        help="Path to an AWS shared credentials file. If omitted, will use AWS_SHARED_CREDENTIALS_FILE or diffusion/aws/credentials if present.",
    )
    p.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to an AWS config file. If omitted, will use AWS_CONFIG_FILE or diffusion/aws/config if present.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Avoid long hangs when boto3 tries instance metadata on non-EC2 machines.
    # But do NOT disable IMDS on EC2/Anyscale nodes, because instance-role auth
    # depends on it.
    if os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or os.environ.get("AWS_ACCESS_KEY_ID"):
        os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

    _maybe_set_repo_aws_files(args.credentials_file, args.config_file)

    session = boto3.session.Session(region_name=args.region)

    if args.debug_auth:
        print("Auth debug:")
        print(f"  AWS_SHARED_CREDENTIALS_FILE={os.environ.get('AWS_SHARED_CREDENTIALS_FILE','')}")
        print(f"  AWS_CONFIG_FILE={os.environ.get('AWS_CONFIG_FILE','')}")
        creds = session.get_credentials()
        if creds is None:
            print("  resolved_credentials=None")
        else:
            frozen = creds.get_frozen_credentials()
            ak = getattr(frozen, "access_key", "") or ""
            ak_tail = ak[-4:] if len(ak) >= 4 else ak
            print(f"  provider_method={getattr(creds, 'method', '')}")
            print(f"  access_key_endswith={ak_tail}")

    # 1) STS identity
    try:
        sts = session.client("sts")
        ident = sts.get_caller_identity()
        print("STS identity:")
        print(ident)
    except NoCredentialsError as e:
        raise SystemExit(
            "No AWS credentials found. Put them in ~/.aws/credentials OR in diffusion/aws/credentials (gitignored), or set AWS_* env vars."
        ) from e

    # 2) Check bucket access
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=args.bucket)
        print(f"Bucket access OK: {args.bucket}")
    except ClientError as e:
        raise SystemExit(f"Cannot access bucket {args.bucket}: {e}") from e

    if args.no_write:
        print("Skipping write test (--no-write).")
        return

    # 3) Write + delete a tiny object
    key = f"{args.prefix.rstrip('/')}/_smoke_{int(time.time())}.txt"
    try:
        s3.put_object(Bucket=args.bucket, Key=key, Body=b"ok")
        print(f"Wrote test object: s3://{args.bucket}/{key}")
    except ClientError as e:
        raise SystemExit(f"Failed to write object: {e}") from e

    try:
        s3.delete_object(Bucket=args.bucket, Key=key)
        print("Deleted test object.")
    except ClientError as e:
        raise SystemExit(f"Failed to delete test object: {e}") from e


if __name__ == "__main__":
    main()
