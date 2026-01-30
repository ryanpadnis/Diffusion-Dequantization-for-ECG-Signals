from __future__ import annotations

import os
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


class S3DependencyError(RuntimeError):
    pass


def _require_boto3():
    try:
        import boto3  # type: ignore

        return boto3
    except Exception as e:
        raise S3DependencyError(
            "boto3 is required for S3 operations. Install with: uv add boto3"
        ) from e


def is_s3_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith("s3://")


def normalize_s3_uri(uri_or_bucket: str) -> str:
    uri_or_bucket = str(uri_or_bucket).strip()
    if uri_or_bucket.startswith("s3://"):
        return uri_or_bucket.rstrip("/")
    return f"s3://{uri_or_bucket}".rstrip("/")


def split_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Return (bucket, key_prefix_without_leading_slash)."""
    s3_uri = normalize_s3_uri(s3_uri)
    remainder = s3_uri[len("s3://") :]
    if "/" in remainder:
        bucket, key = remainder.split("/", 1)
        return bucket, key.rstrip("/")
    return remainder, ""


def join_s3_uri(base_s3_uri: str, *parts: str) -> str:
    base = normalize_s3_uri(base_s3_uri)
    bucket, prefix = split_s3_uri(base)
    extra = [p.strip("/") for p in parts if str(p).strip("/")]
    key = "/".join([p for p in [prefix, *extra] if p])
    return f"s3://{bucket}/{key}".rstrip("/") if key else f"s3://{bucket}"


def upload_file(local_path: str | Path, s3_uri: str) -> None:
    boto3 = _require_boto3()
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(str(local_path))

    bucket, key = split_s3_uri(s3_uri)
    if not key:
        # If someone passed s3://bucket with no key, use filename.
        key = local_path.name

    client = boto3.client("s3")
    client.upload_file(str(local_path), bucket, key)


def download_file(s3_uri: str, local_path: str | Path) -> None:
    boto3 = _require_boto3()
    bucket, key = split_s3_uri(s3_uri)
    if not key:
        raise ValueError(f"S3 URI must include a key: {s3_uri}")

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3")
    client.download_file(bucket, key, str(local_path))


def upload_dir(local_dir: str | Path, s3_prefix_uri: str, *, exclude_globs: Optional[Sequence[str]] = None) -> None:
    """Recursively upload a local directory to an S3 prefix.

    Example:
      upload_dir("diffusion/results/V1", "s3://my-bucket/ee269/results/V1/run123")

    This will mirror the directory structure under the prefix.
    """
    boto3 = _require_boto3()
    from fnmatch import fnmatch

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(str(local_dir))

    exclude_globs = list(exclude_globs or [])
    s3_prefix_uri = normalize_s3_uri(s3_prefix_uri)
    bucket, base_key = split_s3_uri(s3_prefix_uri)

    client = boto3.client("s3")

    for root, _, files in os.walk(local_dir):
        root_path = Path(root)
        for fname in files:
            src = root_path / fname
            rel = src.relative_to(local_dir).as_posix()

            if any(fnmatch(rel, pat) for pat in exclude_globs):
                continue

            key = "/".join([p for p in [base_key, rel] if p])
            client.upload_file(str(src), bucket, key)


def upload_dir_incremental(
    local_dir: str | Path,
    s3_prefix_uri: str,
    *,
    state_path: str | Path,
    exclude_globs: Optional[Sequence[str]] = None,
) -> int:
    """Incrementally upload a directory to S3 (only new/changed files).

    This avoids re-uploading all TensorBoard event files / checkpoints on every sync.

    Returns:
        Number of files uploaded.
    """

    boto3 = _require_boto3()
    from fnmatch import fnmatch

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(str(local_dir))

    exclude_globs = list(exclude_globs or [])
    s3_prefix_uri = normalize_s3_uri(s3_prefix_uri)
    bucket, base_key = split_s3_uri(s3_prefix_uri)

    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        prev = json.loads(state_path.read_text()) if state_path.exists() else {}
    except Exception:
        prev = {}

    client = boto3.client("s3")
    uploaded = 0
    cur: dict[str, dict[str, int]] = {}

    for root, _, files in os.walk(local_dir):
        root_path = Path(root)
        for fname in files:
            src = root_path / fname
            rel = src.relative_to(local_dir).as_posix()

            if any(fnmatch(rel, pat) for pat in exclude_globs):
                continue

            try:
                st = src.stat()
            except FileNotFoundError:
                continue

            meta = {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}
            cur[rel] = meta

            prev_meta = prev.get(rel)
            if prev_meta and int(prev_meta.get("size", -1)) == meta["size"] and int(prev_meta.get("mtime_ns", -1)) == meta["mtime_ns"]:
                continue

            key = "/".join([p for p in [base_key, rel] if p])
            client.upload_file(str(src), bucket, key)
            uploaded += 1

    # Persist current snapshot (also prunes deleted files).
    try:
        state_path.write_text(json.dumps(cur))
    except Exception:
        pass

    return uploaded


def s3_object_exists(s3_uri: str) -> bool:
    """Return True if the exact S3 object exists (best-effort).

    Uses HeadObject which is cheaper than listing. For directory-bucket (S3 Express)
    semantics, this checks only the object key, not prefix existence.
    """

    boto3 = _require_boto3()
    bucket, key = split_s3_uri(s3_uri)
    if not key:
        return False

    client = boto3.client("s3")
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        # botocore is an implementation detail of boto3, but this is the
        # standard way to detect "missing object".
        try:
            from botocore.exceptions import ClientError  # type: ignore

            if isinstance(e, ClientError):
                code = str((e.response or {}).get("Error", {}).get("Code", ""))
                if code in {"404", "NoSuchKey", "NotFound"}:
                    return False
        except Exception:
            pass
        raise


def list_run_prefixes(s3_parent_prefix_uri: str) -> Sequence[Tuple[str, str]]:
    """List direct child prefixes under an S3 prefix.

    Returns a list of (prefix_uri, last_modified_iso) for sorting/pruning.

    Implementation note: we infer last_modified by looking at the newest object
    under each child prefix (cheap enough for small numbers of runs).
    """
    boto3 = _require_boto3()

    s3_parent_prefix_uri = normalize_s3_uri(s3_parent_prefix_uri)
    bucket, parent_key = split_s3_uri(s3_parent_prefix_uri)

    client = boto3.client("s3")

    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=(parent_key + "/") if parent_key else "", Delimiter="/")

    child_prefixes: list[str] = []
    for page in pages:
        for cp in page.get("CommonPrefixes", []) or []:
            child_prefixes.append(cp["Prefix"].rstrip("/"))

    results: list[Tuple[str, str]] = []
    # Determine last_modified per prefix by scanning a small number of objects.
    for child_key_prefix in child_prefixes:
        newest = None
        # S3 Express directory buckets require prefixes to end with a delimiter.
        resp = client.list_objects_v2(
            Bucket=bucket,
            Prefix=(child_key_prefix + "/") if child_key_prefix else "",
            MaxKeys=50,
        )
        for obj in resp.get("Contents", []) or []:
            lm = obj.get("LastModified")
            if lm is None:
                continue
            if newest is None or lm > newest:
                newest = lm
        newest_iso = newest.isoformat() if newest is not None else ""
        results.append((f"s3://{bucket}/{child_key_prefix}", newest_iso))

    return results


def delete_prefix(s3_prefix_uri: str) -> None:
    """Delete all objects under prefix."""
    boto3 = _require_boto3()

    s3_prefix_uri = normalize_s3_uri(s3_prefix_uri)
    bucket, prefix = split_s3_uri(s3_prefix_uri)

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=(prefix + "/") if prefix else ""):
        objs = page.get("Contents", []) or []
        if not objs:
            continue
        # Batch deletes in chunks of 1000
        to_delete = [{"Key": o["Key"]} for o in objs if "Key" in o]
        for i in range(0, len(to_delete), 1000):
            client.delete_objects(Bucket=bucket, Delete={"Objects": to_delete[i : i + 1000]})


def prune_runs(s3_runs_parent_uri: str, *, keep_last_n: int = 3) -> None:
    """Keep only the most-recent N run prefixes under the given parent prefix."""
    keep_last_n = int(keep_last_n)
    if keep_last_n <= 0:
        return

    runs = list(list_run_prefixes(s3_runs_parent_uri))
    # Sort by last_modified (empty strings last)
    runs.sort(key=lambda x: x[1])

    if len(runs) <= keep_last_n:
        return

    to_delete = runs[: len(runs) - keep_last_n]
    for prefix_uri, _ in to_delete:
        delete_prefix(prefix_uri)
