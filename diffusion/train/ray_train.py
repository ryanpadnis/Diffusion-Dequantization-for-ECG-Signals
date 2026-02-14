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
import json
import os
import subprocess
import tarfile
import signal
import sys

from diffusion.aws.ray_support import configure_aws_env, s3_sync_loop, s3_write_text

from diffusion.utils.config import DiffusionConfig
from diffusion.utils.runner import process_data_before_training, train_diffuser


def _parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(add_help=True)
	p.add_argument(
		"--no-ray",
		action="store_true",
		help="Run training in this process (no Ray).",
	)
	p.add_argument(
		"--ray-address",
		type=str,
		default=None,
		help="Ray address (e.g. 'auto' or 'ray://...'). If unset, runs local Ray.",
	)
	p.add_argument(
		"--ray-num-gpus",
		type=float,
		default=0.0,
		help="GPUs to request for the Ray training task.",
	)
	p.add_argument(
		"--ray-runtime-env",
		type=str,
		default="auto",
		choices=["auto", "none"],
		help=(
			"Ray runtime_env behavior when using --ray-address. "
			"'auto' packages this repo as working_dir so workers can import it; 'none' disables packaging."
		),
	)
	p.add_argument(
		"--ray-runtime-pip",
		action="append",
		default=[],
		help=(
			"Optional pip packages to install into Ray runtime_env (repeatable). "
			"Use for lightweight deps; for torch/CUDA prefer baking into the Anyscale image/compute-config."
		),
	)
	p.add_argument(
		"--ray-runtime-pip-requirements",
		type=str,
		default=None,
		help="Optional requirements.txt-style file to install into Ray runtime_env.",
	)
	p.add_argument(
		"--ray-runtime-pip-install-option",
		action="append",
		default=[],
		help=(
			"Extra pip install option(s) for Ray runtime_env (repeatable), e.g. "
			"--ray-runtime-pip-install-option=--extra-index-url --ray-runtime-pip-install-option=https://..."
		),
	)
	p.add_argument(
		"--ray-runtime-env-var",
		action="append",
		default=[],
		help="Environment variables to set in Ray runtime_env, as KEY=VALUE (repeatable).",
	)
	p.add_argument(
		"--epochs",
		type=int,
		default=None,
		help="Override config['epochs'].",
	)
	p.add_argument(
		"--max-batches",
		type=int,
		default=None,
		help="Override config['max_batches'].",
	)
	p.add_argument(
		"--resume-from-s3",
		type=str,
		default=None,
		help="Resume training from an S3 run. Pass 'latest' or a specific run_id.",
	)
	p.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Override config['max_samples'].",
	)
	p.add_argument(
		"--force-preprocess",
		action="store_true",
		help="Force re-preprocessing.",
	)
	p.add_argument(
		"--prepare-s3-data",
		action="store_true",
		help="Preprocess locally and upload preprocessed datasets + normalizer to S3, then exit.",
	)
	p.add_argument(
		"--prepare-s3-data-if-missing",
		action="store_true",
		help=(
			"Ensure preprocessed datasets + normalizer exist in S3, then exit. "
			"If they already exist, this does nothing; otherwise it preprocesses locally and uploads."
		),
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
		default="us-east-1",
		help="AWS region for S3/STS (default: us-east-1).",
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
	p.add_argument(
		"--bundle-run",
		action="store_true",
		help=(
			"If set, create a single tar.gz bundle of run artifacts under results_dir/bundles/ "
			"and upload it to S3 when --s3 is enabled."
		),
	)
	p.add_argument(
		"--bundle-include-data",
		action="store_true",
		help="If set, include results_dir/data/<run_id> in the run bundle (can be large).",
	)
	return p.parse_args()


def _parse_kv_list(items: list[str]) -> dict[str, str]:
	out: dict[str, str] = {}
	for item in items or []:
		if not item:
			continue
		if "=" not in item:
			raise ValueError(f"Expected KEY=VALUE, got: {item}")
		k, v = item.split("=", 1)
		k = k.strip()
		if not k:
			raise ValueError(f"Expected non-empty KEY in: {item}")
		out[k] = v
	return out


def _read_requirements(path: str) -> list[str]:
	pkgs: list[str] = []
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(str(p))
	for raw in p.read_text().splitlines():
		line = raw.strip()
		if not line or line.startswith("#"):
			continue
		# Skip common pip directives in this simple helper; users can pass via --ray-runtime-pip-install-option.
		if line.startswith("--"):
			continue
		pkgs.append(line)
	return pkgs


def _sanitize_for_ray(obj):
	"""Convert config objects into plain Python types for Ray serialization.

	Key goal: the worker should be able to deserialize the config even before
	optional deps (notably torch) are installed.
	"""
	from pathlib import Path

	if obj is None:
		return None
	if isinstance(obj, (str, int, float, bool)):
		return obj
	if isinstance(obj, Path):
		return str(obj)
	if isinstance(obj, (list, tuple)):
		return [_sanitize_for_ray(x) for x in obj]
	if isinstance(obj, dict):
		return {str(k): _sanitize_for_ray(v) for k, v in obj.items()}

	# Avoid importing optional deps; detect by module/name.
	t = type(obj)
	mod = getattr(t, "__module__", "") or ""
	name = getattr(t, "__name__", "") or ""

	# torch.dtype serializes as a global in torch; convert to a simple string like "float32".
	if mod.startswith("torch") and name == "dtype":
		s = str(obj)
		# Common format: "torch.float32".
		return s.split(".")[-1] if "." in s else s

	# torch.device -> "cuda" / "cpu" / "cuda:0".
	if mod.startswith("torch") and name == "device":
		return str(obj)

	# Numpy scalars / other scalar-like objects.
	if hasattr(obj, "item") and callable(getattr(obj, "item")):
		try:
			return obj.item()
		except Exception:
			pass

	# Last resort: string repr.
	return str(obj)


def _write_run_metadata(*, results_dir: Path, run_id: str, args: argparse.Namespace, config: dict) -> Path:
	meta_dir = results_dir / "bundles" / run_id
	meta_dir.mkdir(parents=True, exist_ok=True)
	meta_path = meta_dir / "run_metadata.json"

	git = {}
	try:
		# Best-effort: these commands may fail if git isn't available or repo isn't a git checkout.
		repo_root = Path(__file__).resolve().parents[2]
		git["commit"] = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
		git["status"] = subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], text=True).strip()
	except Exception:
		git = {}

	pip_freeze = ""
	try:
		pip_freeze = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
	except Exception:
		pip_freeze = ""

	payload = {
		"run_id": run_id,
		"created_at": datetime.now().isoformat(),
		"argv": getattr(args, "__dict__", {}),
		"config": config,
		"git": git,
		"pip_freeze": pip_freeze,
	}
	meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
	return meta_path


def _create_run_bundle(
	*,
	results_dir: Path,
	run_id: str,
	artifact_dirs: dict[str, Path],
	include_data: bool,
	args: argparse.Namespace,
	config: dict,
) -> Path:
	(bundle_root := results_dir / "bundles").mkdir(parents=True, exist_ok=True)
	bundle_path = bundle_root / f"{run_id}.tar.gz"

	# Always include a metadata JSON alongside the bundle contents.
	_write_run_metadata(results_dir=results_dir, run_id=run_id, args=args, config=config)

	with tarfile.open(bundle_path, "w:gz") as tf:
		# Add config.pkl if present.
		config_pkl = results_dir / "config.pkl"
		if config_pkl.exists():
			tf.add(config_pkl, arcname="config.pkl")

		# Add the metadata directory for this run.
		meta_dir = results_dir / "bundles" / run_id
		if meta_dir.exists():
			tf.add(meta_dir, arcname=f"bundles/{run_id}")

		for name, local_dir in artifact_dirs.items():
			if name == "data" and not include_data:
				continue
			if local_dir.exists():
				# Store paths relative to results_dir if possible.
				try:
					arcname = str(local_dir.relative_to(results_dir))
				except Exception:
					arcname = f"artifacts/{name}"
				tf.add(local_dir, arcname=arcname)

	return bundle_path


def _run_with_ray(
	*,
	config: dict,
	address: str | None,
	num_gpus: float,
	runtime_env_mode: str,
	runtime_pip: list[str],
	runtime_pip_requirements: str | None,
	runtime_pip_install_options: list[str],
	runtime_env_vars: dict[str, str],
) -> None:
	"""Run training inside a Ray task (local or on a cluster)."""
	import ray
	from pathlib import Path

	# Only set runtime_env packaging when connecting to a remote cluster.
	# This ships the repo code to workers so imports work without rsync/push.
	is_remote = address is not None
	runtime_env = None
	if is_remote and runtime_env_mode != "none":
		repo_root = Path(__file__).resolve().parents[2]
		excludes = [
			".git",
			".venv",
			"__pycache__",
			"**/__pycache__",
			"diffusion/results",
			"data/data/raw",
			"data/data/processed",
			"ee269project.egg-info",
			"**/*.pt",
			"**/*.pth",
		]
		runtime_env = {
			"working_dir": str(repo_root),
			"excludes": excludes,
		}

		pip_pkgs = list(runtime_pip or [])
		if runtime_pip_requirements:
			pip_pkgs.extend(_read_requirements(runtime_pip_requirements))
		pip_pkgs = [p for p in pip_pkgs if str(p).strip()]

		if pip_pkgs or (runtime_pip_install_options and len(runtime_pip_install_options) > 0):
			pip_spec: dict = {"packages": pip_pkgs}
			if runtime_pip_install_options:
				pip_spec["pip_install_options"] = list(runtime_pip_install_options)
			runtime_env["pip"] = pip_spec

		if runtime_env_vars:
			runtime_env["env_vars"] = dict(runtime_env_vars)

	if address is None:
		# Local dev: bring up an embedded Ray runtime.
		ray.init(ignore_reinit_error=True)
	else:
		addr = "auto" if address.strip().lower() == "auto" else address
		ray.init(address=addr, runtime_env=runtime_env)

	@ray.remote(num_gpus=float(num_gpus or 0.0))
	def _train_task(cfg: dict) -> None:
		try:
			import torch
		except ModuleNotFoundError as e:
			raise RuntimeError(
				"PyTorch is not available on this Ray worker. "
				"In Anyscale, packages installed on the head node do not automatically exist on GPU workers. "
				"Fix by baking torch into the workspace image/compute-config setup, or by using Ray runtime_env pip "
				"(note: GPU-enabled torch usually requires an extra index URL / specific wheel)."
			) from e
		import threading
		from pathlib import Path

		from diffusion.aws.ray_support import configure_aws_env as _cfg_aws
		from diffusion.aws.ray_support import s3_sync_loop as _s3_sync_loop
		from diffusion.aws.s3_io import join_s3_uri as _join_s3
		from diffusion.aws.s3_io import upload_file as _s3_upload_file
		from diffusion.aws.s3_io import upload_dir_incremental as _s3_upload_dir_inc
		from diffusion.utils.runner import process_data_before_training as _prep
		from diffusion.utils.runner import train_diffuser as _train

		# Handle resume from S3 on the worker node.
		if cfg.get("_resume_s3_uri"):
			print(f"[ray_train worker] Found request to resume from: {cfg['_resume_s3_uri']}")
			try:
				from diffusion.aws.s3_io import download_file
				# Download to a stable path on the worker.
				# Use config['results_dir'] if absolute, or resolve it.
				# But results_dir is from config, which might be user laptop path.
				# Use a local tmp file or relative path.
				local_ckpt = Path("resumed_checkpoint.pt").resolve()
				print(f"[ray_train worker] Downloading checkpoint from S3 to: {local_ckpt}")
				download_file(cfg["_resume_s3_uri"], local_ckpt)
				cfg["resume_from"] = str(local_ckpt)
				print(f"[ray_train worker] Successfully downloaded checkpoint.")
			except Exception as e:
				print(f"[ray_train worker] Failed to download checkpoint: {e}")
				# Don't crash; start fresh if resume fails.
				pass

		
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

		# Ensure AWS env is set in the worker.
		region = str(cfg.get("s3_region") or os.environ.get("AWS_DEFAULT_REGION") or "")
		if region:
			_cfg_aws(region=region)

		# If configured, start periodic S3 sync from the worker.
		s3_run_prefix = str(cfg.get("s3_run_prefix") or "").strip() or None
		interval_s = float(cfg.get("s3_sync_interval") or 0.0)
		stop_event = None
		sync_thread = None
		if s3_run_prefix and interval_s > 0:
			artifact_dirs = {
				"checkpoints": Path(str(cfg.get("checkpoint_dir"))),
				"samples": Path(str(cfg.get("samples_dir"))),
				"logs": Path(str(cfg.get("logs_dir"))),
				"data": Path(str(cfg.get("data_dir"))),
			}
			results_dir = Path(str(cfg.get("results_dir")))
			stop_event = threading.Event()
			sync_thread = threading.Thread(
				target=_s3_sync_loop,
				kwargs={
					"stop_event": stop_event,
					"interval_s": interval_s,
					"s3_run_prefix": s3_run_prefix,
					"artifact_dirs": artifact_dirs,
					"results_dir": results_dir,
					"log_prefix": "[ray_train.worker]",
				},
				daemon=True,
			)
			sync_thread.start()
			print(f"[ray_train.worker] Started periodic S3 sync every {interval_s:.1f}s -> {s3_run_prefix}")

		try:
			try:
				import json as _json
				print("[ray_train.worker] Config dump:\n" + _json.dumps(cfg, indent=2, sort_keys=True))
			except Exception as e:
				print(f"[ray_train.worker] Config dump failed (non-fatal): {e}")

			cond_data, real_data, ckpt_dir, samp_dir, lg_dir, n_epochs = _prep(cfg)
			_train(real_data, cond_data, ckpt_dir, samp_dir, lg_dir, int(n_epochs), cfg)
		finally:
			if stop_event is not None:
				stop_event.set()
				if sync_thread is not None:
					sync_thread.join(timeout=10.0)

		# Final S3 upload from worker (best-effort) since artifacts are written here.
		if s3_run_prefix:
			artifact_dirs = {
				"checkpoints": Path(str(cfg.get("checkpoint_dir"))),
				"samples": Path(str(cfg.get("samples_dir"))),
				"logs": Path(str(cfg.get("logs_dir"))),
				"data": Path(str(cfg.get("data_dir"))),
			}
			results_dir = Path(str(cfg.get("results_dir")))
			for name, local_dir in artifact_dirs.items():
				if local_dir.exists():
					dst = _join_s3(s3_run_prefix, name)
					state_path = results_dir / f".s3sync_state_{name}.json"
					_s3_upload_dir_inc(local_dir, dst, state_path=state_path)

			# Upload config.pkl if present.
			config_pkl = results_dir / "config.pkl"
			if config_pkl.exists():
				_s3_upload_file(config_pkl, _join_s3(s3_run_prefix, "config.pkl"))

			# Optionally create + upload a single bundle.
			run_id = str(cfg.get("run_id") or "").strip()
			if bool(cfg.get("bundle_run")) and run_id:
				bundle_path = _create_run_bundle(
					results_dir=results_dir,
					run_id=run_id,
					artifact_dirs=artifact_dirs,
					include_data=bool(cfg.get("bundle_include_data")),
					args=argparse.Namespace(),
					config=cfg,
				)
				_s3_upload_file(bundle_path, _join_s3(s3_run_prefix, "bundle", bundle_path.name))

	# Global for signal handler to cancel the task
	_current_task_ref = None
	_cancelled = False

	def _cancel_handler(signum, frame):
		nonlocal _cancelled
		if _cancelled:
			print("\n⚠️  Force quit!")
			sys.exit(1)
		_cancelled = True
		print("\n⚠️  Interrupt received. Cancelling Ray task...")
		if _current_task_ref:
			try:
				ray.cancel(_current_task_ref, force=True)
				print("[ray_train] Task cancelled.")
			except Exception as e:
				print(f"[ray_train] Could not cancel task: {e}")
		sys.exit(0)

	# Set up signal handler
	original_handler = signal.signal(signal.SIGINT, _cancel_handler)

	try:
		cfg_clean = _sanitize_for_ray(config)
		_current_task_ref = _train_task.remote(cfg_clean)
		ray.get(_current_task_ref)
	except KeyboardInterrupt:
		_cancel_handler(signal.SIGINT, None)
	finally:
		# Restore original handler
		signal.signal(signal.SIGINT, original_handler)
		# Always shut down Ray connection (local or remote)
		try:
			ray.shutdown()
			print("[ray_train] Ray shutdown complete.")
		except Exception as e:
			print(f"[ray_train] Ray shutdown error (non-fatal): {e}")


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


def _prepare_s3(
	*,
	args: argparse.Namespace,
	config: dict,
	run_id: str,
) -> tuple[str, str, threading.Event | None, threading.Thread | None]:
	"""Prepare S3 paths, write run markers, and optionally start periodic sync.

	Returns:
	  (s3_runs_parent, s3_run_prefix, stop_event, sync_thread)
	"""
	from diffusion.aws.s3_io import normalize_s3_uri, join_s3_uri

	# Put checkpoints/samples under per-run subdirs (logs already does this inside runner).
	if run_id:
		results_dir = Path(config["results_dir"])
		config["checkpoint_dir"] = str(results_dir / "checkpoints" / run_id)
		config["samples_dir"] = str(results_dir / "samples" / run_id)
		# Ensure derived training artifacts are also run-scoped.
		config["data_dir"] = str(results_dir / "data" / run_id)

	s3_base = normalize_s3_uri(args.s3)
	version = str(getattr(DiffusionConfig, "version", "V1"))

	# Put derived preprocessed data in a shared S3 location (not per-run).
	config["s3_data_uri"] = join_s3_uri(s3_base, "data", version)
	s3_runs_parent = join_s3_uri(s3_base, "diffusion-results", version)
	s3_run_prefix = join_s3_uri(s3_runs_parent, run_id) if run_id else s3_runs_parent

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
	if interval_s <= 0:
		return s3_runs_parent, s3_run_prefix, None, None

	artifact_dirs = {
		"checkpoints": Path(config["checkpoint_dir"]),
		"samples": Path(config["samples_dir"]),
		"logs": Path(config["logs_dir"]),
		"data": Path(config["data_dir"]),
	}
	results_dir = Path(config["results_dir"])
	stop_event = threading.Event()
	sync_thread = threading.Thread(
		target=s3_sync_loop,
		kwargs={
			"stop_event": stop_event,
			"interval_s": interval_s,
			"s3_run_prefix": s3_run_prefix,
			"artifact_dirs": artifact_dirs,
			"results_dir": results_dir,
		},
		daemon=True,
	)
	sync_thread.start()
	print(f"[ray_train] Started periodic S3 sync every {interval_s:.1f}s")
	return s3_runs_parent, s3_run_prefix, stop_event, sync_thread


def _construct_train_filename(*, quantizer_type: str, transform_type: str, bits: int, cond: bool) -> str:
	prefix = "train_cond" if cond else "train_data"
	return f"{prefix}_{quantizer_type}_{transform_type}_{bits}bit.pt"


def _s3_preprocessed_objects_exist(*, s3_data_uri: str, config: dict) -> bool:
	from diffusion.aws.s3_io import join_s3_uri, s3_object_exists

	quantizer_type = str(config.get("quantizer_type", "uniform"))
	transform_type = str(config.get("transform_type", "stft"))
	cond_bits = int(config.get("bit_size", 4) or 4)
	real_bits = int(config.get("real_bit_size", 16) or 16)

	cond_name = _construct_train_filename(
		quantizer_type=quantizer_type,
		transform_type=transform_type,
		bits=cond_bits,
		cond=True,
	)
	real_name = _construct_train_filename(
		quantizer_type=quantizer_type,
		transform_type=transform_type,
		bits=real_bits,
		cond=False,
	)

	cond_s3 = join_s3_uri(s3_data_uri, cond_name)
	real_s3 = join_s3_uri(s3_data_uri, real_name)
	norm_s3 = join_s3_uri(s3_data_uri, "normalizer_params.pkl")

	try:
		return bool(s3_object_exists(cond_s3) and s3_object_exists(real_s3) and s3_object_exists(norm_s3))
	except Exception as e:
		raise RuntimeError(
			"Failed to check S3 for preprocessed objects. "
			"This requires AWS credentials and boto3."
		) from e

		

def main() -> None:
	args = _parse_args()
	configure_aws_env(region=str(args.s3_region))

	config = DiffusionConfig.to_dict()

	# Apply command-line arg overrides
	if args.epochs is not None:
		config["epochs"] = int(args.epochs)
		config["num_epochs"] = int(args.epochs)
	if args.max_batches is not None:
		config["max_batches"] = int(args.max_batches)
	if args.max_samples is not None:
		config["max_samples"] = int(args.max_samples)

	if args.resume_from_s3:
		if not args.s3:
			print("[ray_train] Error: --resume-from-s3 requires --s3")
			sys.exit(1)

		print(f"[ray_train] Attempting to resume from S3 run: {args.resume_from_s3}")
		from diffusion.aws.s3_io import (
			normalize_s3_uri, join_s3_uri, split_s3_uri, list_run_prefixes, download_file
		)
		import boto3

		# Reconstruct where runs live
		s3_base = normalize_s3_uri(args.s3)
		version = str(getattr(DiffusionConfig, "version", "V1"))
		s3_runs_parent = join_s3_uri(s3_base, "diffusion-results", version)

		target = args.resume_from_s3
		if target.strip().lower() == "latest":
			print(f"[ray_train] Listing runs in {s3_runs_parent}...")
			try:
				runs = list_run_prefixes(s3_runs_parent)
				if not runs:
					print("[ray_train] No existing runs found to resume from.")
					target = None
				else:
					runs.sort(key=lambda x: x[1])  # Sort by timestamp
					target_uri = runs[-1][0]
					target = target_uri.rstrip("/").split("/")[-1]
					print(f"[ray_train] Resolved 'latest' to run_id: {target}")
			except Exception as e:
				print(f"[ray_train] Warning: Failed to list runs: {e}")
				target = None

		if target:
			chk_prefix = join_s3_uri(s3_runs_parent, target, "checkpoints")
			bucket, prefix = split_s3_uri(chk_prefix)
			s3 = boto3.client("s3", region_name=args.s3_region)

			print(f"[ray_train] Looking for checkpoints in {chk_prefix}...")
			try:
				resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/")
				contents = resp.get("Contents", [])

				best_key = None
				# Prefer best_model.pt
				for obj in contents:
					if obj["Key"].endswith("/best_model.pt"):
						best_key = obj["Key"]
						break

				# Fallback to latest checkpoint_epoch_*.pt
				if not best_key:
					candidates = []
					for obj in contents:
						k = obj["Key"]
						if k.endswith(".pt") and "checkpoint_epoch_" in k:
							candidates.append((k, obj["LastModified"]))
					if candidates:
						candidates.sort(key=lambda x: x[1])
						best_key = candidates[-1][0]

				if best_key:
					s3_uri = f"s3://{bucket}/{best_key}"
					print(f"[ray_train] Resolved resume target: {s3_uri}")
					# Pass S3 URI to worker; do not download on head node.
					config["_resume_s3_uri"] = s3_uri
				else:
					print(f"[ray_train] Warning: No valid checkpoints found in {chk_prefix}")

			except Exception as e:
				print(f"[ray_train] Warning: Failed to fetch checkpoint metadata: {e}")

	run_id = _ensure_run_id(config)

	for k in ["results_dir", "checkpoint_dir", "logs_dir", "samples_dir", "data_dir", "raw_data_path"]:
		if k in config:
			config[k] = _as_path_str(config[k])

	# If syncing to S3, isolate outputs by run_id so pruning is safe.
	s3_runs_parent = None
	s3_run_prefix = None
	s3_sync_stop = None
	s3_sync_thread = None
	is_remote = args.ray_address is not None
	if args.s3:
		# For remote Ray runs, artifacts are written on the worker. We still prepare S3
		# prefixes/markers here, but we avoid running a head-node sync thread.
		if is_remote:
			prev_interval = float(getattr(args, "s3_sync_interval", 0.0) or 0.0)
			setattr(args, "s3_sync_interval", 0.0)
			s3_runs_parent, s3_run_prefix, s3_sync_stop, s3_sync_thread = _prepare_s3(
				args=args,
				config=config,
				run_id=run_id,
			)
			setattr(args, "s3_sync_interval", prev_interval)
		else:
			s3_runs_parent, s3_run_prefix, s3_sync_stop, s3_sync_thread = _prepare_s3(
				args=args,
				config=config,
				run_id=run_id,
			)

	train_interrupted = False
	if bool(getattr(args, "prepare_s3_data", False)) and bool(getattr(args, "prepare_s3_data_if_missing", False)):
		raise SystemExit("Use only one of --prepare-s3-data or --prepare-s3-data-if-missing")

	# One-time helper: preprocess locally and upload to S3, then exit.
	# This is what you want before running remote Ray training.
	if args.prepare_s3_data:
		if not args.s3:
			raise SystemExit("--prepare-s3-data requires --s3 s3://bucket/prefix")
		# Ensure we actually regenerate/upload.
		config["force_preprocess"] = True
		process_data_before_training(config)
		print("[ray_train] S3 data preparation complete; exiting.")
		# Stop background sync thread before exit.
		if s3_sync_stop is not None:
			s3_sync_stop.set()
			if s3_sync_thread is not None:
				s3_sync_thread.join(timeout=5.0)
				print("[ray_train] Stopped periodic S3 sync")
		return

	# One-time helper: ensure preprocessed data exists in S3, but avoid redoing work.
	if bool(getattr(args, "prepare_s3_data_if_missing", False)):
		if not args.s3:
			raise SystemExit("--prepare-s3-data-if-missing requires --s3 s3://bucket/prefix")
		s3_data_uri = str(config.get("s3_data_uri") or "").strip()
		if not s3_data_uri:
			raise SystemExit("Internal error: expected config['s3_data_uri'] to be set when using --s3")
		if _s3_preprocessed_objects_exist(s3_data_uri=s3_data_uri, config=config):
			print(f"[ray_train] Preprocessed S3 data already present; skipping: {s3_data_uri}")
		else:
			print(f"[ray_train] Preprocessed S3 data missing; preprocessing+uploading to: {s3_data_uri}")
			config["force_preprocess"] = True
			process_data_before_training(config)
		print("[ray_train] S3 data check complete; exiting.")
		# Stop background sync thread before exit.
		if s3_sync_stop is not None:
			s3_sync_stop.set()
			if s3_sync_thread is not None:
				s3_sync_thread.join(timeout=5.0)
				print("[ray_train] Stopped periodic S3 sync")
		return

	# Local (no-ray) path: preprocess locally then train locally.
	if args.no_ray:
		cond_data, real_data, checkpoint_dir, samples_dir, logs_dir, epochs = process_data_before_training(config)
		train_diffuser(real_data, cond_data, checkpoint_dir, samples_dir, logs_dir, epochs, config)
		# Stop background sync thread before final upload.
		if s3_sync_stop is not None:
			s3_sync_stop.set()
			if s3_sync_thread is not None:
				s3_sync_thread.join(timeout=5.0)
				print("[ray_train] Stopped periodic S3 sync")
		# Proceed to final S3 upload section below.
		goto_final_upload = True
	else:
		goto_final_upload = False

	runtime_env_vars = _parse_kv_list(list(getattr(args, "ray_runtime_env_var", []) or []))
	# Ensure AWS region is present on workers when using runtime_env.
	if str(args.s3_region or "").strip():
		runtime_env_vars.setdefault("AWS_DEFAULT_REGION", str(args.s3_region))
		runtime_env_vars.setdefault("AWS_REGION", str(args.s3_region))
	try:
		if not goto_final_upload:
			# Remote Ray runs must use S3 for data (otherwise Ray would try to ship the raw dataset).
			is_remote = args.ray_address is not None
			if is_remote and not args.s3:
				raise SystemExit(
					"Remote Ray training requires --s3 so the cluster can load preprocessed data from S3. "
					"First run: uv run python -m diffusion.train.ray_train --s3 s3://... --prepare-s3-data"
				)
			# For remote clusters, never force preprocessing (raw dataset won't be present on nodes).
			if is_remote and bool(config.get("force_preprocess", False)):
				print("[ray_train] Warning: force_preprocess=True on remote Ray; disabling to avoid missing raw dataset on cluster.")
				config["force_preprocess"] = False

			# Ray train on CUDA by default.
			if float(args.ray_num_gpus or 0.0) > 0 and str(config.get("device") or "").lower() in {"", "cpu"}:
				config["device"] = "cuda"

			# Plumb S3 + bundle settings into worker config for remote Ray runs.
			config["s3_region"] = str(args.s3_region)
			if args.s3 and s3_run_prefix:
				config["s3_run_prefix"] = str(s3_run_prefix)
				config["s3_sync_interval"] = float(getattr(args, "s3_sync_interval", 30.0) or 0.0)
			config["bundle_run"] = bool(getattr(args, "bundle_run", False))
			config["bundle_include_data"] = bool(getattr(args, "bundle_include_data", False))

			_run_with_ray(
				config=config,
				address=args.ray_address,
				num_gpus=float(args.ray_num_gpus),
				runtime_env_mode=str(getattr(args, "ray_runtime_env", "auto")),
				runtime_pip=list(getattr(args, "ray_runtime_pip", []) or []),
				runtime_pip_requirements=getattr(args, "ray_runtime_pip_requirements", None),
				runtime_pip_install_options=list(getattr(args, "ray_runtime_pip_install_option", []) or []),
				runtime_env_vars=runtime_env_vars,
			)
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

	# Optional: create a single local bundle of the run artifacts.
	# For remote Ray runs, artifacts are written on workers and bundling happens there.
	bundle_path: Path | None = None
	run_id = str(config.get("run_id") or "")
	if (not is_remote) and bool(getattr(args, "bundle_run", False)) and run_id:
		try:
			results_dir = Path(config["results_dir"])
			artifact_dirs = {
				"checkpoints": Path(config["checkpoint_dir"]),
				"samples": Path(config["samples_dir"]),
				"logs": Path(config["logs_dir"]),
				"data": Path(config["data_dir"]),
			}
			bundle_path = _create_run_bundle(
				results_dir=results_dir,
				run_id=run_id,
				artifact_dirs=artifact_dirs,
				include_data=bool(getattr(args, "bundle_include_data", False)),
				args=args,
				config=config,
			)
			print(f"[ray_train] Created run bundle: {bundle_path}")
		except Exception as e:
			print(f"[ray_train] Warning: failed to create run bundle: {e}")

	if args.s3 and (not is_remote):
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

		# Upload the run bundle if it was created.
		if bundle_path is not None and bundle_path.exists():
			from diffusion.aws.s3_io import upload_file
			bundle_dst = join_s3_uri(s3_run_prefix, "bundle", bundle_path.name)
			print(f"[ray_train] Uploading run bundle: {bundle_path} -> {bundle_dst}")
			upload_file(bundle_path, bundle_dst)

		# Prune older runs (best-effort).
		if args.keep_last is not None and int(args.keep_last) > 0:
			print(f"[ray_train] Pruning S3 runs under {s3_runs_parent} (keep_last_n={int(args.keep_last)})")
			prune_runs(s3_runs_parent, keep_last_n=int(args.keep_last))

	# For remote runs, the worker uploads; we still prune from the head.
	if args.s3 and is_remote:
		from diffusion.aws.s3_io import normalize_s3_uri, join_s3_uri, prune_runs
		s3_base = normalize_s3_uri(args.s3)
		version = str(getattr(DiffusionConfig, "version", "V1"))
		s3_runs_parent = join_s3_uri(s3_base, "diffusion-results", version)
		if args.keep_last is not None and int(args.keep_last) > 0:
			print(f"[ray_train] Pruning S3 runs under {s3_runs_parent} (keep_last_n={int(args.keep_last)})")
			prune_runs(s3_runs_parent, keep_last_n=int(args.keep_last))

	# Preserve previous behavior: if training was interrupted, reflect that in exit code.
	if train_interrupted:
		raise SystemExit(130)


if __name__ == "__main__":
	main()


