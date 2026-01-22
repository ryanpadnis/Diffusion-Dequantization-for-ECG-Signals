"""
Generic quantizers 
"""
from abc import ABC, abstractmethodq 

class Quantizer(ABC):
    @abstractmethod
    def quantize(self, samples):
        pass
    @abstractmethod
    def dequantize(self, quantized_samples):
        pass
    