"""Signal transforms with PyTorch backend and abstract base class."""

import torch
from abc import ABC, abstractmethod


# Abstract base class for all transforms
class SignalTransform(ABC):
    """Abstract base class for signal transforms."""
    
    def __init__(self, torch_dtype=torch.float32, device=torch.device('cpu')):
        self.torch_dtype = torch_dtype
        self.device = device
        self.magnitude = None
        self.phase = None
    
    @abstractmethod
    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply transform to signal. Returns magnitude/coefficients."""
        pass
    
    @abstractmethod
    def inverse(self, coeffs: torch.Tensor = None) -> torch.Tensor:
        """Inverse transform. If None, uses stored magnitude/phase."""
        pass


# STFT Transform
class STFTTransform(SignalTransform):
    """Short-Time Fourier Transform."""
    
    def __init__(self, n_fft=30, hop_length=64, onesided=True, torch_dtype=torch.float32, device=torch.device('cpu')):
        super().__init__(torch_dtype, device)
        self.type = 'stft'
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.onesided = onesided
        self.params = {'n_fft': n_fft, 'hop_length': hop_length, 'onesided': onesided}

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply STFT to input signal. Handles batched data.
        
        Args:
            signal: [batch_size, signal_length] or [signal_length]
                    Example: [batch_size, 8158] for 128 frames with n_fft=30, hop=64
        
        Returns:
            stft_mag: [batch_size, freq_bins, time_frames] or [freq_bins, time_frames]
                      Magnitude spectrogram (2D for single, 3D for batch)
        """
        signal = signal.to(dtype=self.torch_dtype, device=self.device)
        
        # Compute STFT (handles batches automatically)
        stft_complex = torch.stft(
            signal, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            onesided=self.onesided,
            return_complex=True,
            window=torch.hann_window(self.n_fft, device=self.device),
            normalized=False,
            center=True
        )
        
        # Store magnitude and phase for perfect reconstruction
        self.magnitude = torch.abs(stft_complex)
        self.phase = torch.angle(stft_complex)
        
        return self.magnitude
    
    def inverse(self, stft_mag: torch.Tensor = None) -> torch.Tensor:
        """
        Inverse STFT using stored phase.
        
        Args:
            stft_mag: Optional magnitude. If None, uses stored self.magnitude
        
        Returns:
            signal: [batch_size, signal_length] or [signal_length]
        """
        if stft_mag is None:
            stft_mag = self.magnitude
        
        stft_mag = stft_mag.to(dtype=self.torch_dtype, device=self.device)
        complex_spec = stft_mag * torch.exp(1j * self.phase)
        
        signal = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=self.device),
            center=True
        )
        
        return signal


# DFT Transform
class DFTTransform(SignalTransform):
    """Discrete Fourier Transform."""
    
    def __init__(self, torch_dtype=torch.float32, device=torch.device('cpu')):
        super().__init__(torch_dtype, device)
        self.type = 'dft'
        self.params = {}

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply FFT, return magnitude."""
        signal = signal.to(dtype=self.torch_dtype, device=self.device)
        dft_complex = torch.fft.fft(signal)
        self.magnitude = torch.abs(dft_complex)
        self.phase = torch.angle(dft_complex)
        return self.magnitude

    def inverse(self, dft_mag: torch.Tensor = None) -> torch.Tensor:
        """Inverse FFT using stored phase."""
        if dft_mag is None:
            dft_mag = self.magnitude
        dft_mag = dft_mag.to(dtype=self.torch_dtype, device=self.device)
        complex_spec = dft_mag * torch.exp(1j * self.phase)
        return torch.fft.ifft(complex_spec).real


# Haar Wavelet Transform
class HaarWaveletTransform(SignalTransform):
    """Haar Wavelet decomposition."""
    
    def __init__(self, torch_dtype=torch.float32, device=torch.device('cpu')):
        super().__init__(torch_dtype, device)
        self.type = 'haar_wavelet'
        self.params = {}

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Haar wavelet decomposition."""
        signal = signal.to(dtype=self.torch_dtype, device=self.device)
        n = signal.shape[-1]
        output = signal.clone()
        while n > 1:
            avg = (output[..., :n:2] + output[..., 1:n:2]) / 2
            diff = (output[..., :n:2] - output[..., 1:n:2]) / 2
            output[..., :n//2] = avg
            output[..., n//2:n] = diff
            n //= 2
        self.magnitude = output
        return output

    def inverse(self, coeffs: torch.Tensor = None) -> torch.Tensor:
        """Inverse Haar wavelet decomposition."""
        if coeffs is None:
            coeffs = self.magnitude
        coeffs = coeffs.to(dtype=self.torch_dtype, device=self.device)
        n = 1
        length = coeffs.shape[-1]
        output = coeffs.clone()
        while n * 2 <= length:
            avg = output[..., :n]
            diff = output[..., n:2*n]
            output[..., :2*n:2] = avg + diff
            output[..., 1:2*n:2] = avg - diff
            n *= 2
        return output



AVAILABLE_TRANSFORMS = {
    'stft': STFTTransform,
    'dft': DFTTransform,
    'haar': HaarWaveletTransform,
}


def get_transform(name: str, **kwargs) -> SignalTransform:
    """Get transform by name from registry.
    
    Args:
        name: Transform name ('stft', 'dft', 'haar')
        **kwargs: Arguments to pass to transform constructor
    
    Returns:
        SignalTransform instance
    """
    if name not in AVAILABLE_TRANSFORMS:
        raise ValueError(f"Unknown transform: {name}. Available: {list(AVAILABLE_TRANSFORMS.keys())}")
    return AVAILABLE_TRANSFORMS[name](**kwargs)




