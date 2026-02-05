"""Recreate (recompute + upload) the preprocessed training cache in S3.

This runs the local preprocessing pipeline and uploads the artifacts to:
  <base>/data/<version>/

Example:
  uv run python -m diffusion.aws.recreate_s3_data \
    --s3 s3://YOUR_BUCKET/ee269project --region us-east-1 --version V1

Notes:
- Requires the raw dataset to be present locally (per your DiffusionConfig).
- This uses the same code path as `diffusion.train.ray_train --prepare-s3-data`.
"""

from __future__ import annotations

import argparse
import os
import subprocess


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--s3", required=True, help="Base S3 URI prefix, e.g. s3://bucket/ee269project")
    p.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_DEFAULT_REGION", str(args.region))
    os.environ.setdefault("AWS_REGION", str(args.region))

    cmd = [
        "python",
        "-m",
        "diffusion.train.ray_train",
        "--no-ray",
        "--prepare-s3-data",
        "--s3",
        str(args.s3),
        "--s3-region",
        str(args.region),
    ]

    # Prefer uv when available (matches repo environment).
    try:
        import shutil

        if shutil.which("uv") is not None:
            cmd = ["uv", "run", *cmd]
    except Exception:
        pass

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
