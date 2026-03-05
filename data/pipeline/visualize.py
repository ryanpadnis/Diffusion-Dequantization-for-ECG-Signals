"""Visualize ECG chunks: per-record first chunks, holdout set, and diffusion sampling set."""

import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from data.utils import settings

CHUNKS_DIR = settings.PROCESSED_DIR / "chunks"
NORMALIZED_DIR = settings.PROCESSED_DIR / "normalized"
RAW_KAGGLE_DIR = settings.RAW_DIR / "mitdb_kaggle"
PLOTS_DIR = settings.PROCESSED_DIR / "plots"

# These mirror diffusion/utils/config.py defaults
TEST_HOLDOUT_COUNT = 500
DIFFUSION_NUM_SAMPLES = 16
CHUNK_LENGTH = 3600
CHUNK_STRIDE = 3600


def load_chunks(label: str = "arrhythmia", chunks_dir: Path = CHUNKS_DIR):
    """Load a chunks .pt file and return the dict with 'chunks' and 'record_names'."""
    pt_path = chunks_dir / f"{label}_chunks.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found")
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def load_first_chunks(chunks_dir: Path):
    results = {}
    for label in ["arrhythmia", "non_arrhythmia"]:
        pt_path = chunks_dir / f"{label}_chunks.pt"
        if not pt_path.exists():
            print(f"  Warning: {pt_path} not found, skipping.")
            continue
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        results[label] = {"chunks": data["chunks"], "names": data["record_names"]}
        print(f"  {label}: {data['chunks'].shape[0]} chunks from {len(set(data['record_names']))} records")
    return results


def get_first_chunk_per_record(chunks: torch.Tensor, names: list[str]):
    seen = {}
    for i, name in enumerate(names):
        if name not in seen:
            seen[name] = chunks[i]
    return list(seen.keys()), list(seen.values())


# ---------------------------------------------------------------------------
# 1) Print chunk counts
# ---------------------------------------------------------------------------
def print_chunk_info(chunks_dir: Path = CHUNKS_DIR):
    """Print the number of chunks in each .pt file found in chunks_dir."""
    found_any = False
    for label in ["arrhythmia", "non_arrhythmia"]:
        pt_path = chunks_dir / f"{label}_chunks.pt"
        if not pt_path.exists():
            continue
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        chunks = data["chunks"]
        names = data["record_names"]
        n_records = len(set(names))
        found_any = True
        print(f"{label}: {chunks.shape[0]} chunks  (shape {tuple(chunks.shape)}, {n_records} unique records)")
    if not found_any:
        print(f"No chunk files found in {chunks_dir}")


# ---------------------------------------------------------------------------
# 2) Visualize the last DIFFUSION_NUM_SAMPLES chunks from the holdout tail
# ---------------------------------------------------------------------------
def plot_holdout_samples(
    label: str = "arrhythmia",
    holdout_count: int = TEST_HOLDOUT_COUNT,
    num_samples: int = DIFFUSION_NUM_SAMPLES,
    chunks_dir: Path = CHUNKS_DIR,
    save: bool = False,
):
    """
    Grab the last `holdout_count` chunks (the test holdout) and then take the
    last `num_samples` from that set — matching what the diffusion sampler uses.
    Plots each chunk as a subplot.
    """
    data = load_chunks(label, chunks_dir)
    all_chunks = data["chunks"]
    all_names = data["record_names"]
    n_total = all_chunks.shape[0]

    print(f"Total {label} chunks: {n_total}")
    print(f"Holdout (last {holdout_count}): indices [{n_total - holdout_count}:{n_total}]")

    holdout_chunks = all_chunks[-holdout_count:]
    holdout_names = all_names[-holdout_count:]

    # The diffusion sampler takes the *last* num_samples from the holdout
    sample_chunks = holdout_chunks[-num_samples:]
    sample_names = holdout_names[-num_samples:]

    abs_start = n_total - holdout_count + (holdout_count - num_samples)
    print(
        f"Sampling last {num_samples} of holdout: "
        f"absolute indices [{abs_start}:{abs_start + num_samples}]"
    )

    n = len(sample_chunks)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    fig.suptitle(
        f"Diffusion Sampling Set — last {num_samples} of {holdout_count}-chunk holdout  ({label})",
        fontsize=12,
        fontweight="bold",
    )

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            sig = sample_chunks[i].numpy()
            ax.plot(sig, linewidth=0.5, color="#e05c5c", alpha=0.85)
            ax.set_title(f"#{abs_start + i}  ({sample_names[i]})", fontsize=7)
            ax.set_ylim(-2.5, 7.0)
            ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.4, color="gray")
            ax.set_axisbelow(True)
            ax.set_xticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_visible(False)

    plt.tight_layout()

    if save:
        out = PLOTS_DIR / f"holdout_last{num_samples}_{label}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3) Original: first chunk per record (strip plot)
# ---------------------------------------------------------------------------
def plot_first_chunks(chunks_dir: Path, save: bool = False):
    data = load_first_chunks(chunks_dir)
    if not data:
        print("No chunk files found.")
        return

    all_records = []
    for label, d in data.items():
        rec_names, rec_chunks = get_first_chunk_per_record(d["chunks"], d["names"])
        for name, chunk in zip(rec_names, rec_chunks):
            all_records.append((name, chunk, label))

    n = len(all_records)
    label_colors = {"arrhythmia": "#e05c5c", "non_arrhythmia": "#5c8fe0"}

    fig, axes = plt.subplots(n, 1, figsize=(14, n * 3.0))
    if n == 1:
        axes = [axes]

    fig.suptitle("First Chunk per Recording", fontsize=13, fontweight="bold", x=0.5, y=1.002)

    for ax, (name, chunk, label) in zip(axes, all_records):
        signal = chunk.numpy()
        ax.plot(signal, linewidth=0.6, color=label_colors[label], alpha=0.9)

        ax.set_ylabel(name, fontsize=7, rotation=0, labelpad=80, va="center", ha="right")
        ax.set_ylim(-2.5, 7.0)

        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, color="gray")
        ax.set_axisbelow(True)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    handles = [plt.Line2D([0], [0], color=c, linewidth=2, label=l.replace("_", " ").title())
               for l, c in label_colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.8)

    plt.tight_layout()

    if save:
        output_path = PLOTS_DIR / "first_chunks.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 4) Plot a continuous (pre-chunked) signal region matching the last N chunks
# ---------------------------------------------------------------------------
def plot_continuous_tail(
    record: str = "234_ekg",
    num_chunks: int = DIFFUSION_NUM_SAMPLES,
    chunk_length: int = CHUNK_LENGTH,
    stride: int = CHUNK_STRIDE,
    normalized_dir: Path = NORMALIZED_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    save: bool = False,
):
    """
    Load the full normalized signal for `record` and plot the continuous
    region that corresponds to the last `num_chunks` chunks of that record.
    """
    norm_path = normalized_dir / f"{record}_normalized.pt"
    if not norm_path.exists():
        raise FileNotFoundError(f"{norm_path} not found")

    signal = torch.load(norm_path, map_location="cpu", weights_only=False)
    total_samples = signal.shape[0]

    # Reproduce the chunking to find how many chunks this record yields
    n_chunks_total = (total_samples - chunk_length) // stride
    if n_chunks_total <= 0:
        raise ValueError(f"Signal too short ({total_samples}) for chunk_length={chunk_length}")

    tail_start_chunk = max(0, n_chunks_total - num_chunks)
    sample_start = tail_start_chunk * stride
    sample_end = min((n_chunks_total - 1) * stride + chunk_length, total_samples)

    print(f"Record: {record}")
    print(f"Full signal: {total_samples} samples")
    print(f"Total chunks: {n_chunks_total}  (chunk_length={chunk_length}, stride={stride})")
    print(f"Last {num_chunks} chunks → sample range [{sample_start}:{sample_end}]  "
          f"({sample_end - sample_start} samples)")

    region = signal[sample_start:sample_end].numpy()

    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.plot(region, linewidth=0.4, color="#e05c5c", alpha=0.9)

    # Draw faint vertical lines at chunk boundaries
    for i in range(1, num_chunks):
        x = i * stride
        ax.axvline(x, color="gray", linewidth=0.4, linestyle="--", alpha=0.35)

    ax.set_title(
        f"{record} — continuous signal for last {num_chunks} chunks  "
        f"(samples {sample_start}–{sample_end})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Sample index (within region)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save:
        out = PLOTS_DIR / f"continuous_last{num_chunks}_{record}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 5) Plot continuous pre-normalization (raw CSV) region for last N chunks
# ---------------------------------------------------------------------------
def plot_continuous_tail_raw(
    record: str = "234_ekg",
    lead: str = "MLII",
    num_chunks: int = DIFFUSION_NUM_SAMPLES,
    chunk_length: int = CHUNK_LENGTH,
    stride: int = CHUNK_STRIDE,
    raw_dir: Path = RAW_KAGGLE_DIR,
    save: bool = False,
):
    """
    Load the raw (pre-normalization) CSV for `record`, extract the `lead`
    column, and plot the region corresponding to the last `num_chunks` chunks.
    """
    csv_path = raw_dir / f"{record}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    if lead not in df.columns:
        raise KeyError(f"Lead '{lead}' not in {csv_path.name}. Available: {list(df.columns)}")

    signal = df[lead].values
    total_samples = len(signal)

    n_chunks_total = (total_samples - chunk_length) // stride
    if n_chunks_total <= 0:
        raise ValueError(f"Signal too short ({total_samples}) for chunk_length={chunk_length}")

    tail_start_chunk = max(0, n_chunks_total - num_chunks)
    sample_start = tail_start_chunk * stride
    sample_end = min((n_chunks_total - 1) * stride + chunk_length, total_samples)

    print(f"Record: {record}  (raw, lead={lead})")
    print(f"Full signal: {total_samples} samples")
    print(f"Total chunks: {n_chunks_total}  (chunk_length={chunk_length}, stride={stride})")
    print(f"Last {num_chunks} chunks → sample range [{sample_start}:{sample_end}]  "
          f"({sample_end - sample_start} samples)")

    region = signal[sample_start:sample_end]

    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.plot(region, linewidth=0.4, color="#3a7ebf", alpha=0.9)

    for i in range(1, num_chunks):
        x = i * stride
        ax.axvline(x, color="gray", linewidth=0.4, linestyle="--", alpha=0.35)

    ax.set_title(
        f"{record} (raw {lead}) — last {num_chunks} chunks  "
        f"(samples {sample_start}–{sample_end})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Sample index (within region)")
    ax.set_ylabel(f"{lead} (mV)")
    ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save:
        out = PLOTS_DIR / f"continuous_raw_last{num_chunks}_{record}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--chunks-dir", type=str, default=str(CHUNKS_DIR))
    common.add_argument("--save", action="store_true", help="Save plots to processed/plots/")

    parser = argparse.ArgumentParser(description="Visualize ECG chunks.", parents=[common])
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Print chunk counts", parents=[common])
    sub.add_parser("first-chunks", help="Plot first chunk per record (original)", parents=[common])

    holdout_p = sub.add_parser("holdout", help="Plot diffusion sampling set from holdout tail", parents=[common])
    holdout_p.add_argument("--label", type=str, default="arrhythmia")
    holdout_p.add_argument("--holdout-count", type=int, default=TEST_HOLDOUT_COUNT)
    holdout_p.add_argument("--num-samples", type=int, default=DIFFUSION_NUM_SAMPLES)

    cont_p = sub.add_parser("continuous", help="Plot continuous (pre-chunked) signal for last N chunks", parents=[common])
    cont_p.add_argument("--record", type=str, default="234_ekg")
    cont_p.add_argument("--num-chunks", type=int, default=DIFFUSION_NUM_SAMPLES)

    raw_p = sub.add_parser("continuous-raw", help="Plot continuous pre-normalization signal for last N chunks", parents=[common])
    raw_p.add_argument("--record", type=str, default="234_ekg")
    raw_p.add_argument("--lead", type=str, default="MLII")
    raw_p.add_argument("--num-chunks", type=int, default=DIFFUSION_NUM_SAMPLES)

    args = parser.parse_args()
    chunks_dir = Path(args.chunks_dir)

    if args.command == "info":
        print_chunk_info(chunks_dir)
    elif args.command == "holdout":
        plot_holdout_samples(
            label=args.label,
            holdout_count=args.holdout_count,
            num_samples=args.num_samples,
            chunks_dir=chunks_dir,
            save=args.save,
        )
    elif args.command == "first-chunks":
        plot_first_chunks(chunks_dir, save=args.save)
    elif args.command == "continuous":
        plot_continuous_tail(
            record=args.record,
            num_chunks=args.num_chunks,
            save=args.save,
        )
    elif args.command == "continuous-raw":
        plot_continuous_tail_raw(
            record=args.record,
            lead=args.lead,
            num_chunks=args.num_chunks,
            save=args.save,
        )
    else:
        print_chunk_info(chunks_dir)
        print()
        plot_holdout_samples(chunks_dir=chunks_dir, save=args.save)
