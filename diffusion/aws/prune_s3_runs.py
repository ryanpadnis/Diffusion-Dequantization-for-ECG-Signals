"""Delete recent runs from an S3 prefix (safe by default).

Typical usage (dry-run):
  uv run python -m diffusion.aws.prune_s3_runs \
    --runs-prefix "s3://ee269--use2-az1--x-s3/ee269project/diffusion-results/V1" \
    --delete-last 3

Actually delete:
  uv run python -m diffusion.aws.prune_s3_runs \
    --runs-prefix "s3://ee269--use2-az1--x-s3/ee269project/diffusion-results/V1" \
    --delete-last 3 --yes

Notes:
- This deletes whole *run* prefixes (folders) under the parent prefix.
- It does not delete the entire project unless you point it at that.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from diffusion.aws.s3_io import delete_prefix, list_run_prefixes, normalize_s3_uri


def _maybe_set_repo_aws_files() -> None:
    """Prefer repo-local AWS config/credentials if present (gitignored)."""

    aws_dir = Path(__file__).resolve().parent
    repo_creds = aws_dir / "credentials"
    repo_cfg = aws_dir / "config"

    if "AWS_SHARED_CREDENTIALS_FILE" not in os.environ and repo_creds.exists():
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(repo_creds)
    if "AWS_CONFIG_FILE" not in os.environ and repo_cfg.exists():
        os.environ["AWS_CONFIG_FILE"] = str(repo_cfg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--runs-prefix",
        required=True,
        help=(
            "S3 URI of the runs parent prefix (e.g. "
            "s3://bucket/ee269project/diffusion-results/V1)."
        ),
    )
    p.add_argument(
        "--delete-last",
        type=int,
        default=0,
        help="Delete the most recent N run prefixes (0 disables).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform deletions (otherwise dry-run).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Avoid long hangs when boto3 tries instance metadata on non-EC2 machines.
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    _maybe_set_repo_aws_files()

    runs_prefix = normalize_s3_uri(args.runs_prefix)
    delete_last = int(args.delete_last or 0)
    if delete_last <= 0:
        print("Nothing to do: --delete-last must be > 0")
        return

    runs = list(list_run_prefixes(runs_prefix))
    # list_run_prefixes returns (prefix_uri, last_modified_iso); empty iso sorts oldest.
    runs.sort(key=lambda x: x[1])

    if not runs:
        print(f"No run prefixes found under: {runs_prefix}")
        return

    to_delete = list(reversed(runs))[:delete_last]

    print(f"Runs prefix: {runs_prefix}")
    print(f"Matched runs: {len(runs)}")
    print(f"Deleting most recent: {len(to_delete)}")

    for prefix_uri, last_modified in to_delete:
        print(f"  - {prefix_uri} (last_modified={last_modified or 'unknown'})")

    if not args.yes:
        print("Dry-run only. Re-run with --yes to actually delete.")
        return

    for prefix_uri, _ in to_delete:
        print(f"Deleting: {prefix_uri}")
        delete_prefix(prefix_uri)

    print("Done.")


if __name__ == "__main__":
    main()
