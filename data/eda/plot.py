"""EDA: best-case STFT inversion sanity check.

Goal (per your request): don't touch preprocessing; in this script, compute an STFT
directly from the original ECG chunk and invert it. This shows the *best possible*
reconstruction quality for a given STFT parameterization.

Outputs (per sample) on a single subplot grid:
- Original time series
- Reconstructed time series (ISTFT of complex STFT)
- Error time series (orig - recon)
- Original magnitude spectrogram
- Reconstructed magnitude spectrogram
- Spectrogram difference (abs)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import settings


def compute_spectral_energy_db(spectrogram):
    energy = np.sum(np.square(spectrogram))
    return 10 * np.log10(energy + 1e-12)

def compute_snr_mse(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
    return snr, mse


def _load_original_chunks() -> torch.Tensor:
    processed = Path(settings.PROCESSED_DIR)
    orig_file = processed / "arythmia_chunks.pt"
    orig_obj = torch.load(orig_file, map_location="cpu")
    if isinstance(orig_obj, dict) and "chunks" in orig_obj:
        return orig_obj["chunks"]
    if isinstance(orig_obj, dict) and "signals" in orig_obj:
        return orig_obj["signals"]
    raise ValueError(f"Unexpected format in {orig_file}")


def _stft_and_inverse(
    x: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool,
    onesided: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (orig_ts, recon_ts, orig_mag, recon_mag)."""
    x = x.to(dtype=torch.float32, device="cpu")
    window = torch.hann_window(win_length)

    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=center,
        onesided=onesided,
        normalized=False,
    )

    recon = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        onesided=onesided,
        normalized=False,
        length=int(x.shape[0]),
    )

    spec_recon = torch.stft(
        recon,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=center,
        onesided=onesided,
        normalized=False,
    )

    orig_mag = torch.abs(spec).cpu().numpy()
    recon_mag = torch.abs(spec_recon).cpu().numpy()

    return x.cpu().numpy(), recon.cpu().numpy(), orig_mag, recon_mag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--win-length", type=int, default=256)
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Disable centered STFT padding (may reduce ISTFT quality at boundaries)",
    )
    parser.add_argument("--save", action="store_true", help="Save subplot figures as PNGs")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures interactively")
    args = parser.parse_args()

    center = not args.no_center

    if args.hop_length > args.win_length:
        raise ValueError("Require hop_length <= win_length for invertible ISTFT")
    if args.win_length > args.n_fft:
        raise ValueError("Require win_length <= n_fft")

    orig = _load_original_chunks()
    total = int(orig.shape[0])

    start = max(0, min(args.start, total - 1))
    end = min(total, start + max(1, args.num_samples))

    out_dir = Path(settings.PROCESSED_DIR)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(start, end):
        orig_ts, recon_ts, orig_mag, recon_mag = _stft_and_inverse(
            orig[i],
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            win_length=int(args.win_length),
            center=center,
            onesided=True,
        )

        err_ts = orig_ts - recon_ts

        # --- Statistics ---
        snr, mse = compute_snr_mse(orig_ts, recon_ts)
        orig_energy_db = compute_spectral_energy_db(orig_mag)
        recon_energy_db = compute_spectral_energy_db(recon_mag)

        fig, axes = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True)
        (ax00, ax01), (ax10, ax11), (ax20, ax21) = axes

        ax00.plot(orig_ts, linewidth=0.9)
        ax00.set_title(f"Sample {i} - Original")

        ax01.plot(recon_ts, linewidth=0.9)
        ax01.set_title(f"Reconstruction (ISTFT of complex STFT)\nSNR: {snr:.2f} dB, MSE: {mse:.4g}")

        # Log-magnitude spectrograms for visibility
        orig_show = np.log10(orig_mag + 1e-8)
        recon_show = np.log10(recon_mag + 1e-8)
        diff_show = np.abs(orig_show - recon_show)

        im0 = ax10.imshow(orig_show, aspect="auto", origin="lower", cmap="viridis")
        ax10.set_title(f"Original |STFT| (log10)\nEnergy: {orig_energy_db:.2f} dB")
        fig.colorbar(im0, ax=ax10, fraction=0.046, pad=0.04)

        im1 = ax11.imshow(recon_show, aspect="auto", origin="lower", cmap="viridis")
        ax11.set_title(f"Recon |STFT| (log10)\nEnergy: {recon_energy_db:.2f} dB")
        fig.colorbar(im1, ax=ax11, fraction=0.046, pad=0.04)

        ax20.plot(err_ts, linewidth=0.9)
        ax20.set_title("Error (orig - recon)")

        im2 = ax21.imshow(diff_show, aspect="auto", origin="lower", cmap="magma")
        ax21.set_title("|log10|STFT| diff|")
        fig.colorbar(im2, ax=ax21, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"STFT inversion sanity check (n_fft={args.n_fft}, hop={args.hop_length}, win={args.win_length}, center={center})",
            fontsize=13,
        )

        np.save(out_dir / f"istft_recon_sample_{i}.npy", recon_ts)

        if args.save:
            fig.savefig(out_dir / f"eda_sample_{i}.png", dpi=150)

        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
