"""
Sample plotting code for the signals that have been preprocessed or not
"""

import matplotlib.pyplot as plt
import torch
from data import settings



if __name__ == "__main__":
    # Use the 4-bit uniform STFT output
    DATA_FILE = settings.output_path("uniform", "stft", 4)
    data = torch.load(DATA_FILE)
    signals = data['signals']  # Shape: [N, freq_bins, time_frames]
    num_plots = min(5, signals.shape[0])

    print(f"Loaded {signals.shape[0]} spectrograms from {DATA_FILE}")

    for i in range(num_plots):
        spec = signals[i].cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        # Spectrogram
        im = axs[0].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axs[0].set_title(f"Spectrogram {i+1}")
        axs[0].set_xlabel("Time Frame")
        axs[0].set_ylabel("Frequency Bin")
        fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, label='Magnitude')

        # Time profile (average over frequency)
        time_profile = spec.mean(axis=0)
        axs[1].plot(time_profile)
        axs[1].set_title("Mean over frequency (time profile)")
        axs[1].set_xlabel("Time Frame")
        axs[1].set_ylabel("Magnitude")

        # Frequency profile (average over time)
        freq_profile = spec.mean(axis=1)
        axs[2].plot(freq_profile)
        axs[2].set_title("Mean over time (frequency profile)")
        axs[2].set_xlabel("Frequency Bin")
        axs[2].set_ylabel("Magnitude")

        plt.tight_layout()
        plt.show()
        print(f"Plotted spectrogram {i+1} with shape {spec.shape}")
    