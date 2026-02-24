"""Visualize the first chunk of each signal as a clean vertical strip plot."""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from data.utils import settings


def load_first_chunks(chunks_dir: Path):
    results = {}
    for label in ["arrhythmia", "non_arrhythmia"]:
        pt_path = chunks_dir / f"{label}_chunks.pt"
        if not pt_path.exists():
            print(f"  Warning: {pt_path} not found, skipping.")
            continue
        data = torch.load(pt_path)
        results[label] = {"chunks": data["chunks"], "names": data["record_names"]}
        print(f"  {label}: {data['chunks'].shape[0]} chunks from {len(set(data['record_names']))} records")
    return results


def get_first_chunk_per_record(chunks: torch.Tensor, names: list[str]):
    seen = {}
    for i, name in enumerate(names):
        if name not in seen:
            seen[name] = chunks[i]
    return list(seen.keys()), list(seen.values())


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
        # ax.yaxis.set_tick_params(labelsize=6)
        # ax.set_yticks([-2, -1, 0, 1, 2])
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.set_ylim(-2.5, 7.0)  # consistent across all signals

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
        output_path = settings.PROCESSED_DIR / "plots" / "first_chunks.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize first chunk of each ECG recording.")
    parser.add_argument("--chunks-dir", type=str, default=str(settings.PROCESSED_DIR / "chunks"))
    parser.add_argument("--save", action="store_true", help="Save plot to processed/plots/")
    args = parser.parse_args()

    plot_first_chunks(
        chunks_dir=Path(args.chunks_dir),
        save=args.save,
    )
