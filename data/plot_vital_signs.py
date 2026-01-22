import vitaldb
import matplotlib.pyplot as plt
import numpy as np

def linear_quantize(data, bits=16):
    """Linear quantization to specified bit depth"""
    # Handle NaNs
    mask = ~np.isnan(data)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    scaled = np.full_like(data, np.nan)
    scaled[mask] = (data[mask] - vmin) / (vmax - vmin) * (2**bits - 1)
    return np.round(scaled)

track_names = ['SNUADC/ART', 'HR']
vf = vitaldb.VitalFile(1, track_names)
samples = vf.to_numpy(track_names, 1/100)

# Quantize to different bit depths (keep as floats)
quantized_16bit = np.round((samples - np.nanmin(samples, axis=0)) / (np.nanmax(samples, axis=0) - np.nanmin(samples, axis=0)) * (2**16 - 1))
quantized_12bit = np.round((samples - np.nanmin(samples, axis=0)) / (np.nanmax(samples, axis=0) - np.nanmin(samples, axis=0)) * (2**12 - 1))
quantized_8bit = np.round((samples - np.nanmin(samples, axis=0)) / (np.nanmax(samples, axis=0) - np.nanmin(samples, axis=0)) * (2**8 - 1))

# Take subset for visualization (first 5000 samples)
subset = slice(0, 5000)

plt.figure(figsize=(20, 15))

# ===== ARTERIAL PRESSURE =====
# Original
plt.subplot(3, 2, 1)
plt.plot(samples[subset, 0], linewidth=1)
plt.title('ART - Original (32-bit float)')
plt.ylabel('mmHg')
plt.grid()

# 16-bit quantized
plt.subplot(3, 2, 2)
plt.plot(quantized_16bit[subset, 0], linewidth=1, color='orange')
plt.title('ART - 16-bit Quantized')
plt.ylabel('Quantized value')
plt.grid()

# 12-bit quantized
plt.subplot(3, 2, 3)
plt.plot(quantized_12bit[subset, 0], linewidth=1, color='green')
plt.title('ART - 12-bit Quantized')
plt.ylabel('Quantized value')
plt.grid()

# 8-bit quantized
plt.subplot(3, 2, 4)
plt.plot(quantized_8bit[subset, 0], linewidth=1, color='red')
plt.title('ART - 8-bit Quantized')
plt.ylabel('Quantized value')
plt.grid()

# ===== HEART RATE =====
# Original
plt.subplot(3, 2, 5)
plt.plot(samples[subset, 1], linewidth=1, color='blue')
plt.title('HR - Original (32-bit float)')
plt.ylabel('bpm')
plt.xlabel('Samples')
plt.grid()

# 16-bit quantized
plt.subplot(3, 2, 6)
plt.plot(quantized_16bit[subset, 1], linewidth=1, color='purple')
plt.title('HR - 16-bit Quantized')
plt.ylabel('Quantized value')
plt.xlabel('Samples')
plt.grid()

plt.tight_layout()
plt.show()

# Print stats
print("Quantization Stats:")
print(f"Original ART range: {np.nanmin(samples[:, 0]):.2f} to {np.nanmax(samples[:, 0]):.2f}")
print(f"Original HR range: {np.nanmin(samples[:, 1]):.2f} to {np.nanmax(samples[:, 1]):.2f}")
print(f"16-bit ART range: {np.nanmin(quantized_16bit[:, 0]):.0f} to {np.nanmax(quantized_16bit[:, 0]):.0f}")
print(f"12-bit ART range: {np.nanmin(quantized_12bit[:, 0]):.0f} to {np.nanmax(quantized_12bit[:, 0]):.0f}")
print(f"8-bit ART range: {np.nanmin(quantized_8bit[:, 0]):.0f} to {np.nanmax(quantized_8bit[:, 0]):.0f}")



