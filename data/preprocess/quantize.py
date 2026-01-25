"""
Generic quantizers and their analuytics andf utilizites using abc module.  Torch only
"""
import torch
from abc import ABC, abstractmethod


class Quantizer(ABC):
    bits: int # number of bits to quantize to
    levels: int  # number of quantization levels (2**bits)
    quantizer_type: str  # 'uniform', 'non-uniform', etc.
    range_min: float
    range_max: float
   
    @abstractmethod
    def quantize(self, signal: torch.Tensor) -> torch.Tensor:
        """Quantize the input signal."""
        raise NotImplementedError

    @abstractmethod
    def dequantize(self, quantized_signal: torch.Tensor) -> torch.Tensor:
        """Dequantize the input signal."""
        raise NotImplementedError
    
    @staticmethod
    def compute_snr_db(original: torch.Tensor, quantized: torch.Tensor) -> float:
        """Compute Signal-to-Noise Ratio (SNR) in dB."""
        noise = original - quantized
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    
    @staticmethod
    def compute_mse(original: torch.Tensor, quantized: torch.Tensor) -> float:
        """Compute Mean Squared Error (MSE) between original and quantized signals."""
        mse = torch.mean((original - quantized) ** 2)
        return mse.item()
    
    @staticmethod
    def compute_snr(original: torch.Tensor, quantized: torch.Tensor) -> float:
        """Compute SNR general"""
        noise = original - quantized
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean(noise ** 2)
        snr = signal_power / noise_power
        return snr.item()
    


class UniformQuantizer(Quantizer):
    def __init__(self, bits: int, range_min: float, range_max: float):
        self.bits = bits
        self.levels = 2 ** bits
        self.quantizer_type = 'uniform'
        self.range_min = range_min
        self.range_max = range_max
        # step size uses `levels` so indices range 0..levels-1 and centers sit at +0.5*step
        self.step_size = (range_max - range_min) / float(self.levels)

    def quantize_indices(self, signal: torch.Tensor) -> torch.Tensor:
        """Quantize the input signal and return integer bin indices (0..levels-1)."""
        clipped_signal = torch.clamp(signal, self.range_min, self.range_max)
        indices = torch.floor((clipped_signal - self.range_min) / self.step_size)
        indices = torch.clamp(indices, 0, self.levels - 1)
        return indices

    def quantize(self, signal: torch.Tensor) -> torch.Tensor:
        """Uniformly quantize the input signal and return quantized VALUES.

        This returns a float tensor whose values lie on the quantizer's discrete levels.
        It matches the common DSP definition of quantization: x -> Q(x).
        """
        indices = self.quantize_indices(signal)
        return self.decode(indices)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode integer indices into representative quantized values (bin centers)."""
        return indices * self.step_size + self.range_min + (self.step_size / 2.0)

    # Legacy name kept for backwards compatibility, but pipeline code should not need it.
    def dequantize(self, quantized_signal: torch.Tensor) -> torch.Tensor:
        """Legacy alias for decode()."""
        return self.decode(quantized_signal)


def compute_range_from_tensor(x: torch.Tensor, lower_pct: float = 0.0, upper_pct: float = 100.0) -> (float, float):
    """Compute a robust [min, max] range from tensor `x` using percentiles.

    Args:
        x: Tensor of values (any shape).
        lower_pct: Lower percentile (0-100).
        upper_pct: Upper percentile (0-100).

    Returns:
        (min_val, max_val) as Python floats.
    """
    # Move to CPU for quantile computation if needed
    x_cpu = x.detach().cpu().flatten()
    if lower_pct <= 0.0 and upper_pct >= 100.0:
        return float(x_cpu.min().item()), float(x_cpu.max().item())

    # torch.quantile expects values between 0 and 1
    lower_q = lower_pct / 100.0
    upper_q = upper_pct / 100.0
    try:
        lo = torch.quantile(x_cpu, torch.tensor(lower_q))
        hi = torch.quantile(x_cpu, torch.tensor(upper_q))
    except Exception:
        # Fallback for older torch versions: use numpy
        import numpy as _np
        arr = x_cpu.numpy()
        lo = _np.percentile(arr, lower_pct)
        hi = _np.percentile(arr, upper_pct)
        return float(lo), float(hi)

    return float(lo.item()), float(hi.item())
    

##etc and etc for other quantizers

    


    





