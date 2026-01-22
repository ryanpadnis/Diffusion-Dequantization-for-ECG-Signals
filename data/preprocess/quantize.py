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
        self.step_size = (range_max - range_min) / self.levels

    def quantize(self, signal: torch.Tensor) -> torch.Tensor:
        """Uniformly quantize the input signal."""
        clipped_signal = torch.clamp(signal, self.range_min, self.range_max)
        quantized_signal = torch.round((clipped_signal - self.range_min) / self.step_size)
        return quantized_signal

    def dequantize(self, quantized_signal: torch.Tensor) -> torch.Tensor:
        """Dequantize the input signal."""
        dequantized_signal = quantized_signal * self.step_size + self.range_min + self.step_size / 2
        return dequantized_signal
    

##etc and etc for other quantizers

    


    





