# Committing specific diffusion run data (samples) to GitHub

GitHub rejects normal Git blobs larger than 100MB. For diffusion artifacts, you generally want:
- Keep `checkpoints/` and `logs/` out of git.
- Commit *only* one specific run’s `samples/` subtree for sharing/analysis.
- Use **Git LFS** for any large `.pt` files in that subtree (common: `condition_raw.pt`).

This repo is already set up to allow committing a single run’s samples via `.gitignore`.

## 0) Pick the run you want to share

Decide:
- `VERSION` (e.g. `V4`)
- `RUN_ID` (e.g. `20260213_202346`)

Local path:
- `diffusion/results/<VERSION>/<RUN_ID>/samples/`

## 1) (Optional) Download the run artifacts from S3

If you don’t already have the run locally:

- `uv run python -m diffusion.aws.download_s3_artifacts --version V4 --run-id 20260213_202346 --best-only`

This downloads into:
- `diffusion/results/V4/20260213_202346/`

## 2) Update `.gitignore` to allow ONLY that run’s samples

Open `.gitignore` and update the *three* parent-directory unignore lines and the `samples/` allowlist to match your run.

Example (current pattern):
- `!diffusion/results/V4/`
- `!diffusion/results/V4/20260213_202346/`
- `!diffusion/results/V4/20260213_202346/samples/`
- `!diffusion/results/V4/20260213_202346/samples/**`

Keep everything else under `diffusion/results/` ignored.

## 3) Set up Git LFS (required for >100MB)

One-time install (macOS):
- `brew install git-lfs`

One-time init (per machine):
- `git lfs install`

Track only the large files you expect in this run.

Most common (recommended):
- `git lfs track "diffusion/results/V4/20260213_202346/samples/**/condition_raw.pt"`

This creates/updates `.gitattributes`.

## 4) Stage and commit the samples

Stage the LFS rules + the samples (force add is often needed the first time):
- `git add .gitattributes`
- `git add -f diffusion/results/V4/20260213_202346/samples`

Commit:
- `git commit -m "Add diffusion samples for V4/20260213_202346"`

If you already committed and push failed due to the 100MB limit, amend after enabling LFS:
- `git add .gitattributes diffusion/results/V4/20260213_202346/samples/**/condition_raw.pt`
- `git commit --amend --no-edit`

Verify what’s in LFS:
- `git lfs ls-files`

## 5) Push

- `git push`

If you amended an already-pushed commit, you may need:
- `git push --force-with-lease`

## For teammates (getting the data)

They must have Git LFS installed to pull the large `.pt` files:
- `brew install git-lfs`
- `git lfs install`

Then after cloning/pulling:
- `git lfs pull`

## Recommended “minimal share” set

If repo size becomes an issue, consider committing only:
- `condition_spec.pt` (conditioning spectrogram)
- `trajectory_0.pt` (one generated output)
- a few `sample_<idx>/` folders (not all 16)

…and omit `condition_raw.pt` entirely.
