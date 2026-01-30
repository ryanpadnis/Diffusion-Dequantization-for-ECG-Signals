"""AWS/Anyscale entrypoint for running training and syncing artifacts to S3.

This intentionally does NOT set up Ray cluster infrastructure yet.
It is a thin wrapper around `diffusion.utils.runner.train_diffuser`.

Usage:
  uv run python -m diffusion.train.ray_train
  uv run python -m diffusion.train.ray_train --s3 my-bucket
  uv run python -m diffusion.train.ray_train --s3 s3://my-bucket/ee269project --keep-last 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import threading

from diffusion.aws.ray_support import configure_aws_env, s3_sync_loop, s3_write_text

from diffusion.utils.config import DiffusionConfig
from diffusion.utils.runner import train_diffuser


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(add_help=True)
	p.add_argument(
		"--no-ray",
		action="store_true",
		help="Run training directly in this process (no Ray).",
	)
	p.add_argument(
		"--ray-address",
		type=str,
		default=None,
		help="Ray address to connect to (e.g. 'auto' or 'ray://...'). If unset, starts local Ray.",
	)
	p.add_argument(
		"--ray-num-gpus",
		type=float,
		default=0.0,
		help="GPUs to request for the training task when using Ray (default: 0).",
	)
	p.add_argument(
		"--epochs",
		type=int,
		default=None,
		help="Override number of training epochs (config['epochs']).",
	)
	p.add_argument(
		"--max-batches",
		type=int,
		default=None,
		help="Override max_batches for quick tests.",
	)
	p.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Override max_samples for quick tests.",
	)
	p.add_argument(
		"--force-preprocess",
		action="store_true",
		help="Force re-preprocessing even if datasets already exist (overrides config['force_preprocess']).",
	)
	p.add_argument(
		"--s3",
		type=str,
		default=None,
		help="If set, upload the run artifacts to this S3 bucket or s3:// URI prefix.",
	)
	p.add_argument(
		"--keep-last",
		type=int,
		default=3,
		help="When using --s3, keep only the most recent N runs under the S3 runs prefix.",
	)
	p.add_argument(
		"--s3-region",
		type=str,
		default="us-east-2",
		help="AWS region for S3/STS (default: us-east-2).",
	)
	p.add_argument(
		"--run-id",
		type=str,
		default=None,
		help="Optional run id (defaults to timestamp in runner).",
	)
	p.add_argument(
		"--s3-sync-interval",
		type=float,
		default=30.0,
		help="When using --s3, periodically sync artifacts during training (seconds; 0 disables).",
	)
	return p.parse_args()


def _run_with_ray(*, config: dict, address: str | None, num_gpus: float) -> None:
	"""Run training inside a Ray task (local or on a cluster)."""
	import ray

	if address is None:
		# Local dev: bring up an embedded Ray runtime.
		ray.init(ignore_reinit_error=True)
	else:
		addr = "auto" if address.strip().lower() == "auto" else address
		ray.init(address=addr)

	@ray.remote(num_gpus=float(num_gpus or 0.0))
	def _train_task(cfg: dict) -> None:
		import torch
		from diffusion.utils.runner import train_diffuser as _train

		# If Ray assigned us GPUs, default to CUDA unless explicitly overridden.
		# Ray sets CUDA_VISIBLE_DEVICES so "cuda:0" maps to the assigned GPU.
		requested = float(num_gpus or 0.0)
		if requested > 0 and str(cfg.get("device") or "").lower() in {"", "cpu"}:
			cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
		dev = str(cfg.get("device") or "").lower()
		if dev.startswith("cuda"):
			if torch.cuda.is_available():
				try:
					torch.cuda.set_device(0)
				except Exception:
					pass
			else:
				# Misconfigured environment (or CPU-only image). Don't crash.
				cfg["device"] = "cpu"

		_train(cfg)

	try:
		ray.get(_train_task.remote(config))
	finally:
		# If we started local Ray, clean it up.
		if address is None:
			ray.shutdown()


def _as_path_str(x) -> str:
	# DiffusionConfig uses Path objects; runner expects strings in config for some keys.
	return str(x) if x is not None else ""


def _ensure_run_id(config: dict) -> str:
	run_id = str(config.get("run_id") or "").strip()
	if run_id:
		return run_id
	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	config["run_id"] = run_id
	return run_id


def main() -> None:
	args = _parse_args()
	configure_aws_env(region=str(args.s3_region))

	config = DiffusionConfig.to_dict()
	# CLI overrides
	if args.epochs is not None:
		config["epochs"] = int(args.epochs)
	if args.max_batches is not None:
		config["max_batches"] = int(args.max_batches)
	if args.max_samples is not None:
		config["max_samples"] = int(args.max_samples)
	# Default to NOT forcing preprocessing for S3-based workflows unless explicitly requested.
	# (DiffusionConfig defaults may be tuned for local iteration.)
	if args.force_preprocess:
		config["force_preprocess"] = True
	else:
		config["force_preprocess"] = False
	if args.run_id:
		config["run_id"] = args.run_id
	run_id = _ensure_run_id(config)

	# Ensure config paths are strings (some code pickles config).
	for k in ["results_dir", "checkpoint_dir", "logs_dir", "samples_dir", "data_dir", "raw_data_path"]:
		if k in config:
			config[k] = _as_path_str(config[k])

	# If syncing to S3, isolate outputs by run_id so pruning is safe.
	s3_runs_parent = None
	s3_run_prefix = None
	s3_sync_stop = None
	s3_sync_thread = None
	if args.s3:
		# Put checkpoints/samples under per-run subdirs (logs already does this inside runner).
		if run_id:
			results_dir = Path(config["results_dir"])
			config["checkpoint_dir"] = str(results_dir / "checkpoints" / run_id)
			config["samples_dir"] = str(results_dir / "samples" / run_id)
			# Ensure derived training artifacts (saved tensors, normalizer params, etc.)
			# are also run-scoped, so they can be mirrored to S3 per-run.
			config["data_dir"] = str(results_dir / "data" / run_id)

		# Proactively create visible S3 artifacts at run start so you can confirm uploads immediately.
		from diffusion.aws.s3_io import normalize_s3_uri, join_s3_uri
		s3_base = normalize_s3_uri(args.s3)
		version = str(getattr(DiffusionConfig, "version", "V1"))
		# Put derived preprocessed data in a shared S3 location (not per-run).
		# This makes cluster runs stateless and avoids writing big tensors into the repo.
		config["s3_data_uri"] = join_s3_uri(s3_base, "data", version)
		s3_runs_parent = join_s3_uri(s3_base, "diffusion-results", version)
		s3_run_prefix = join_s3_uri(s3_runs_parent, run_id)
		print(f"[ray_train] S3 run prefix: {s3_run_prefix}")
		# 'Touch' the common prefixes by writing small marker objects.
		s3_write_text(join_s3_uri(s3_run_prefix, "_RUN_STARTED.txt"), f"started={run_id}\n")
		s3_write_text(join_s3_uri(s3_run_prefix, "checkpoints", "_PLACEHOLDER"), "")
		s3_write_text(join_s3_uri(s3_run_prefix, "samples", "_PLACEHOLDER"), "")
		s3_write_text(join_s3_uri(s3_run_prefix, "logs", "_PLACEHOLDER"), "")
		s3_write_text(join_s3_uri(s3_run_prefix, "data", "_PLACEHOLDER"), "")
		print("[ray_train] Wrote S3 run-start marker + placeholders.")

		# Start periodic S3 sync so you can see data populate while training is running.
		interval_s = float(getattr(args, "s3_sync_interval", 30.0) or 0.0)
		if interval_s > 0:
			artifact_dirs = {
				"checkpoints": Path(config["checkpoint_dir"]),
				"samples": Path(config["samples_dir"]),
				"logs": Path(config["logs_dir"]),
				"data": Path(config["data_dir"]),
			}
			results_dir = Path(config["results_dir"])
			s3_sync_stop = threading.Event()
			s3_sync_thread = threading.Thread(
				target=s3_sync_loop,
				kwargs={
					"stop_event": s3_sync_stop,
					"interval_s": interval_s,
					"s3_run_prefix": s3_run_prefix,
					"artifact_dirs": artifact_dirs,
					"results_dir": results_dir,
				},
				daemon=True,
			)
			s3_sync_thread.start()
			print(f"[ray_train] Started periodic S3 sync every {interval_s:.1f}s")

	train_interrupted = False
	try:
		if args.no_ray:
			train_diffuser(config)
		else:
			# If you're requesting GPUs via Ray, train on CUDA by default.
			if float(args.ray_num_gpus or 0.0) > 0 and str(config.get("device") or "").lower() in {"", "cpu"}:
				config["device"] = "cuda"
			_run_with_ray(config=config, address=args.ray_address, num_gpus=float(args.ray_num_gpus))
	except KeyboardInterrupt:
		train_interrupted = True
		print("[ray_train] Training interrupted (Ctrl+C). Will still attempt S3 upload of any artifacts written so far.")
		# Fall through to S3 upload.
	finally:
		# Stop background sync thread before final upload.
		if s3_sync_stop is not None:
			s3_sync_stop.set()
			if s3_sync_thread is not None:
				s3_sync_thread.join(timeout=5.0)
				print("[ray_train] Stopped periodic S3 sync")
	if args.s3:
		from diffusion.aws.s3_io import normalize_s3_uri, join_s3_uri, upload_dir, prune_runs
		from diffusion.aws.s3_io import upload_dir_incremental

		s3_base = normalize_s3_uri(args.s3)
		results_dir = Path(config["results_dir"])  # local path
		run_id = str(config.get("run_id") or "")

		# Upload destination:
		#   s3://bucket[/prefix]/diffusion-results/<version>/<run_id>/...
		# Keep it simple and deterministic.
		version = str(getattr(DiffusionConfig, "version", "V1"))
		if run_id:
			s3_runs_parent = join_s3_uri(s3_base, "diffusion-results", version)
			s3_run_prefix = join_s3_uri(s3_runs_parent, run_id)
		else:
			s3_runs_parent = join_s3_uri(s3_base, "diffusion-results", version)
			s3_run_prefix = s3_runs_parent

		# Upload the artifact subfolders plus derived training data under results_dir/data.
		# This intentionally does not upload the raw dataset (data/data/raw).
		artifact_dirs = {
			"checkpoints": Path(config["checkpoint_dir"]),
			"samples": Path(config["samples_dir"]),
			"logs": Path(config["logs_dir"]),
			"data": Path(config["data_dir"]),
		}

		print(f"[ray_train] Uploading artifacts to: {s3_run_prefix}")
		if train_interrupted:
			print("[ray_train] Note: run may be incomplete; uploading whatever exists locally.")

		for name, local_dir in artifact_dirs.items():
			if local_dir.exists():
				dst = join_s3_uri(s3_run_prefix, name)
				print(f"[ray_train] Uploading {name}: {local_dir} -> {dst}")
				state_path = results_dir / f".s3sync_state_{name}.json"
				uploaded = upload_dir_incremental(local_dir, dst, state_path=state_path)
				print(f"[ray_train]   - Uploaded {uploaded} changed files")
			else:
				print(f"[ray_train] Skipping {name} (not found): {local_dir}")

		# Upload config.pkl if present (runner writes it under results_dir).
		config_pkl = results_dir / "config.pkl"
		if config_pkl.exists():
			from diffusion.aws.s3_io import upload_file
			dst = join_s3_uri(s3_run_prefix, "config.pkl")
			print(f"[ray_train] Uploading config: {config_pkl} -> {dst}")
			upload_file(config_pkl, dst)

		# Prune older runs (best-effort).
		if args.keep_last is not None and int(args.keep_last) > 0:
			print(f"[ray_train] Pruning S3 runs under {s3_runs_parent} (keep_last_n={int(args.keep_last)})")
			prune_runs(s3_runs_parent, keep_last_n=int(args.keep_last))

	# Preserve previous behavior: if training was interrupted, reflect that in exit code.
	if train_interrupted:
		raise SystemExit(130)


if __name__ == "__main__":
	main()


