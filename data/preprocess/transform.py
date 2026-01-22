"""
Torch native transformers for vital sign signals using abc modules
"""

from abc import ABC, abstractmethod
import torch


class Transform(ABC):
    type: str  # 'normalize', 'standardize', dft, etc.
    params: dict  # parameters for the transform
    torch_dtype: torch.dtype  # data type for torch tensors
    device: torch.device  # device to perform computations on
    magnitude: torch.Tensor = None  # storage for magnitude
    phase: torch.Tensor = None  # storage for phase

    @abstractmethod
    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply the transform to the input signal."""
        raise NotImplementedError
    
    @abstractmethod
    def inverse(self, transformed_signal: torch.Tensor) -> torch.Tensor:
        """Inverse the transform to recover the original signal."""
        raise NotImplementedError

    
#stft
class STFTTransform(Transform):
    def __init__(self, n_fft=30, hop_length=64, onesided=True, torch_dtype=torch.float32, device=torch.device('cpu')):
        self.type = 'stft'
        self.params = {'n_fft': n_fft, 'hop_length': hop_length, 'onesided': onesided} # n is the window size, hop is the overlap size
        self.torch_dtype = torch_dtype
        self.device = device
    
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
        n_fft = self.params['n_fft']
        hop_length = self.params['hop_length']
        onesided = self.params['onesided']
        
        # Compute STFT (should handle batches automatically)
        stft_complex = torch.stft(
            signal, 
            n_fft=n_fft, 
            hop_length=hop_length,
            onesided=onesided,
            return_complex=True,
            window=torch.hann_window(n_fft, device=self.device),
            normalized=False,
            center=True
        )
        
        # Store magnitude and phase for perfect reconstruction
        self.magnitude = torch.abs(stft_complex)
        self.phase = torch.angle(stft_complex)
        
        # Return magnitude (2D or 3D depending on input)
        return self.magnitude
    
    def inverse(self, stft_mag: torch.Tensor = None) -> torch.Tensor:
        """
        Inverse STFT using stored phase or Griffin-Lim if phase not available.
        
        Args:
            stft_mag: Optional magnitude. If None, uses stored self.magnitude
        
        Returns:
            signal: [batch_size, signal_length] or [signal_length]
        """
        n_fft = self.params['n_fft']
        hop_length = self.params['hop_length']
        
        # Use stored magnitude if not provided
        if stft_mag is None:
            stft_mag = self.magnitude
        
        stft_mag = stft_mag.to(dtype=self.torch_dtype, device=self.device)
        
        complex_spec = stft_mag * torch.exp(1j * self.phase)
        signal = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=self.device),
            center=True
        )

        return signal



class DFTTransform(Transform):
    def __init__(self, torch_dtype=torch.float32, device=torch.device('cpu')):
        self.type = 'dft'
        self.params = {}
        self.torch_dtype = torch_dtype
        self.device = device
    
    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply DFT to input signal. Handles batched data.
        
        Args:
            signal: [batch_size, signal_length] or [signal_length]
        
        Returns:
            dft_mag: [batch_size, freq_bins] or [freq_bins]
                      Magnitude spectrum (2D for single, 3D for batch)
        """
        signal = signal.to(dtype=self.torch_dtype, device=self.device)
        
        # Compute DFT (handles batches automatically)
        dft_complex = torch.fft.fft(signal)
        
        # Store magnitude and phase for perfect reconstruction
        self.magnitude = torch.abs(dft_complex)
        self.phase = torch.angle(dft_complex)
        
        return self.magnitude
    
    def inverse(self, dft_mag: torch.Tensor = None) -> torch.Tensor:
        """
        Inverse DFT using stored phase.
        
        Args:
            dft_mag: Optional magnitude. If None, uses stored self.magnitude
        
        Returns:
            signal: [batch_size, signal_length] or [signal_length]
        """
        # Use stored magnitude if not provided
        if dft_mag is None:
            dft_mag = self.magnitude
        
        dft_mag = dft_mag.to(dtype=self.torch_dtype, device=self.device)
        
        complex_spec = dft_mag * torch.exp(1j * self.phase)
        signal = torch.fft.ifft(complex_spec).real

        return signal
    

class HaarWaveletTransform(Transform):
    def __init__(self, torch_dtype=torch.float32, device=torch.device('cpu')):
        self.type = 'haar_wavelet'
        self.params = {}
        self.torch_dtype = torch_dtype
        self.device = device
    
    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply Haar Wavelet Transform to input signal. Handles batched data.
        
        Args:
            signal: [batch_size, signal_length] or [signal_length]
        
        Returns:
            coeffs: Transformed coefficients
        """
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
        """
        Inverse Haar Wavelet Transform.
        
        Args:
            coeffs: Transformed coefficients. If None, uses stored self.magnitude
        
        Returns:
            signal: Reconstructed signal
        """
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




