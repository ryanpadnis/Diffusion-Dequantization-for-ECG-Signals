from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path


def maybe_set_repo_aws_files() -> None:
	"""Prefer repo-local AWS config if present.

	Supports keeping credentials under `diffusion/aws/` (gitignored) while still
	allowing boto3/awscli to authenticate.
	"""

	aws_dir = Path(__file__).resolve().parent
	repo_creds = aws_dir / "credentials"
	repo_cfg = aws_dir / "config"

	if "AWS_SHARED_CREDENTIALS_FILE" not in os.environ and repo_creds.exists():
		os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(repo_creds)
	if "AWS_CONFIG_FILE" not in os.environ and repo_cfg.exists():
		os.environ["AWS_CONFIG_FILE"] = str(repo_cfg)


def configure_aws_env(*, region: str) -> None:
	"""Set sane AWS env defaults for local/dev and Ray cluster nodes."""

	os.environ.setdefault("AWS_DEFAULT_REGION", str(region))
	os.environ.setdefault("AWS_REGION", str(region))
	# Prefer repo-local credentials if present.
	# Only disable IMDS when we are explicitly using local static credentials;
	# on EC2/Anyscale we usually WANT IMDS for instance role auth.
	maybe_set_repo_aws_files()
	if os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
		os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")


def s3_write_text(s3_uri: str, text: str) -> None:
	"""Upload a small text blob to S3 by writing a temp file first."""

	from diffusion.aws.s3_io import upload_file

	with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
		f.write(text)
		f.flush()
		upload_file(f.name, s3_uri)


def s3_sync_loop(
	*,
	stop_event: threading.Event,
	interval_s: float,
	s3_run_prefix: str,
	artifact_dirs: dict[str, Path],
	results_dir: Path,
	log_prefix: str = "[ray_train]",
) -> None:
	"""Periodically sync local artifact dirs to S3 while training runs."""

	from diffusion.aws.s3_io import join_s3_uri, upload_dir_incremental, upload_file

	deadline = time.time() + max(0.0, interval_s)
	while not stop_event.is_set():
		now = time.time()
		if now < deadline:
			stop_event.wait(timeout=max(0.1, deadline - now))
			continue
		deadline = time.time() + max(0.0, interval_s)

		try:
			for name, local_dir in artifact_dirs.items():
				if local_dir.exists():
					dst = join_s3_uri(s3_run_prefix, name)
					state_path = results_dir / f".s3sync_state_{name}.json"
					upload_dir_incremental(local_dir, dst, state_path=state_path)
			# config.pkl (runner writes it under results_dir)
			config_pkl = results_dir / "config.pkl"
			if config_pkl.exists():
				upload_file(config_pkl, join_s3_uri(s3_run_prefix, "config.pkl"))
		except Exception as e:
			print(f"{log_prefix} S3 sync warning: {e}")
