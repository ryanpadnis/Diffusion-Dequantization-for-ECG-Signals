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

    def __init__(
        self,
        n_fft=30,
        hop_length=64,
        onesided=True,
        win_length=None,
        center=False,
        out_shape: tuple[int, int] | None = None,
        torch_dtype=torch.float32,
        device=torch.device('cpu'),
    ):
        super().__init__(torch_dtype, device)
        self.type = 'stft'
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.onesided = onesided
        self.center = center
        self.out_shape = tuple(out_shape) if out_shape is not None else None

        self.win_length = win_length if win_length is not None else n_fft
        if self.win_length > self.n_fft:
            self.win_length = self.n_fft

        self.input_length = None
        self.params = {
            'n_fft': n_fft,
            'hop_length': hop_length,
            'win_length': self.win_length,
            'onesided': onesided,
            'center': center,
            'out_shape': self.out_shape,
        }

    def _apply_out_shape(self, mag: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad/crop STFT outputs to out_shape=(H,W) if set.

        - H must match the number of STFT frequency bins (no implicit resampling).
          If you want H bins, choose n_fft such that H = n_fft//2 + 1 (onesided=True).
        - W is padded with zeros (or cropped) along time frames.
        """
        if self.out_shape is None:
            return mag, phase

        H, W = int(self.out_shape[0]), int(self.out_shape[1])
        if H <= 0 or W <= 0:
            raise ValueError(f"out_shape must be positive, got {self.out_shape}")

        cur_H = int(mag.shape[-2])
        cur_W = int(mag.shape[-1])

        if cur_H != H:
            raise ValueError(
                f"STFT produced {cur_H} freq bins but out_shape requests {H}. "
                f"Set n_fft so that (n_fft//2 + 1) == {H} (with onesided=True)."
            )

        if cur_W == W:
            return mag, phase

        if cur_W > W:
            mag = mag[..., :W]
            phase = phase[..., :W]
            return mag, phase

        pad = W - cur_W
        mag = torch.nn.functional.pad(mag, (0, pad), mode='constant', value=0.0)
        phase = torch.nn.functional.pad(phase, (0, pad), mode='constant', value=0.0)
        return mag, phase

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
        self.input_length = signal.shape[-1]

        window = torch.hann_window(self.win_length, device=self.device)
        stft_complex = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            onesided=self.onesided,
            return_complex=True,
            window=window,
            normalized=False,
            center=self.center,
        )
        
    
        self.magnitude = torch.abs(stft_complex)
        self.phase = torch.angle(stft_complex)

        self.magnitude, self.phase = self._apply_out_shape(self.magnitude, self.phase)
        
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
        if self.phase is None:
            raise ValueError("STFTTransform.inverse() requires phase to be set (call apply() first or set transform.phase).")

        complex_spec = stft_mag * torch.exp(1j * self.phase)
        window = torch.hann_window(self.win_length, device=self.device)

        length = getattr(self, 'input_length', None)

        # torch.istft requires hop_length <= win_length (and win_length <= n_fft).
        # Our pipeline sometimes uses hop_length > win_length to hit a desired frame count.
        # In that case we do a simple overlap-add ISTFT implementation.
        if 0 < self.hop_length <= self.win_length:
            try:
                signal = torch.istft(
                    complex_spec,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=window,
                    center=self.center,
                    onesided=self.onesided,
                    length=length,
                )
                return signal
            except RuntimeError:
                # Fall through to manual overlap-add.
                pass

        # Manual overlap-add ISTFT (supports hop_length > win_length)
        if complex_spec.ndim == 2:
            complex_spec = complex_spec.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        # [B, F, T] -> [B, n_fft, T]
        frames = torch.fft.irfft(complex_spec, n=self.n_fft, dim=-2)
        frames = frames.transpose(-2, -1)  # [B, T, n_fft]

        time_frames = frames.shape[-2]
        inferred_length = (time_frames - 1) * self.hop_length + self.n_fft
        signal_length = int(length) if length is not None else inferred_length

        # Pad window to n_fft (torch.stft pads win_length<n_fft centered)
        if self.win_length < self.n_fft:
            pad_total = self.n_fft - self.win_length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            window_full = torch.nn.functional.pad(window, (pad_left, pad_right))
        else:
            window_full = window

        window_full = window_full.to(dtype=frames.dtype, device=frames.device)

        signal = torch.zeros((frames.shape[0], signal_length), device=frames.device, dtype=frames.dtype)
        window_envelope = torch.zeros_like(signal)

        for t in range(time_frames):
            start = t * self.hop_length
            end = start + self.n_fft
            if start >= signal_length:
                break

            if end > signal_length:
                frame = frames[:, t, : signal_length - start]
                win = window_full[: signal_length - start]
                end = signal_length
            else:
                frame = frames[:, t, :]
                win = window_full

            signal[:, start:end] += frame * win
            window_envelope[:, start:end] += win.pow(2)

        eps = 1e-8
        nonzero = window_envelope > eps
        signal[nonzero] = signal[nonzero] / window_envelope[nonzero]

        if squeezed:
            signal = signal.squeeze(0)
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
    
    def __init__(
        self,
        out_shape: tuple[int, int] | None = None,
        target_len: int | None = None,
        torch_dtype=torch.float32,
        device=torch.device('cpu'),
    ):
        super().__init__(torch_dtype, device)
        self.type = 'haar'
        self.out_shape = tuple(out_shape) if out_shape is not None else None
        self.target_len = int(target_len) if target_len is not None else None
        self.input_length: int | None = None
        self.padded_length: int | None = None
        self.params = {
            'out_shape': self.out_shape,
            'target_len': self.target_len,
        }

    @staticmethod
    def _next_pow2(n: int) -> int:
        n = int(n)
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def _resolve_target_len(self, input_len: int) -> int:
        if self.out_shape is not None:
            h, w = int(self.out_shape[0]), int(self.out_shape[1])
            if h <= 0 or w <= 0:
                raise ValueError(f"out_shape must be positive, got: {self.out_shape}")
            return int(h * w)
        if self.target_len is not None:
            return int(self.target_len)
        return self._next_pow2(int(input_len))

    @staticmethod
    def _haar_forward_1d(x: torch.Tensor) -> torch.Tensor:
        """Full Haar decomposition along the last dim (length must be power of 2)."""
        n = int(x.shape[-1])
        if n & (n - 1) != 0:
            raise ValueError(f"Haar forward requires power-of-two length, got {n}")
        y = x.clone()
        cur = n
        inv_sqrt2 = float(2.0 ** -0.5)
        while cur > 1:
            # Clone because we write back into y and these are views.
            a = y[..., :cur:2].clone()
            b = y[..., 1:cur:2].clone()
            avg = (a + b) * inv_sqrt2
            diff = (a - b) * inv_sqrt2
            half = cur // 2
            y[..., :half] = avg
            y[..., half:cur] = diff
            cur = half
        return y

    @staticmethod
    def _haar_inverse_1d(coeffs: torch.Tensor) -> torch.Tensor:
        """Inverse full Haar decomposition along the last dim."""
        n_total = int(coeffs.shape[-1])
        if n_total & (n_total - 1) != 0:
            raise ValueError(f"Haar inverse requires power-of-two length, got {n_total}")
        y = coeffs.clone()
        cur = 1
        inv_sqrt2 = float(2.0 ** -0.5)
        while cur * 2 <= n_total:
            # Clone because we write back into y and these are views.
            avg = y[..., :cur].clone()
            diff = y[..., cur:2 * cur].clone()
            # Inverse of forward with sqrt(2) normalization.
            y[..., :2 * cur:2] = (avg + diff) * inv_sqrt2
            y[..., 1:2 * cur:2] = (avg - diff) * inv_sqrt2
            cur *= 2
        return y

    def apply(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Haar wavelet decomposition.

        Supports arbitrary input lengths by zero-padding to a power-of-two.
        If `out_shape` is set, returns coefficients reshaped to [B, H, W].
        """
        x = signal.to(dtype=self.torch_dtype, device=self.device)
        squeezed = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeezed = True
        if x.ndim != 2:
            raise ValueError(f"HaarWaveletTransform.apply expects [B, L] or [L], got {tuple(x.shape)}")

        self.input_length = int(x.shape[-1])
        target_len = self._resolve_target_len(self.input_length)
        if target_len < self.input_length:
            raise ValueError(f"target_len ({target_len}) < input_length ({self.input_length})")
        target_len = self._next_pow2(target_len)
        self.padded_length = int(target_len)

        if x.shape[-1] < target_len:
            x = torch.nn.functional.pad(x, (0, target_len - x.shape[-1]), mode='constant', value=0.0)

        coeffs_1d = self._haar_forward_1d(x)

        if self.out_shape is not None:
            h, w = int(self.out_shape[0]), int(self.out_shape[1])
            need = h * w
            if need != target_len:
                raise ValueError(f"out_shape {self.out_shape} implies len={need}, but padded_length={target_len}")
            coeffs = coeffs_1d.view(coeffs_1d.shape[0], h, w)
        else:
            coeffs = coeffs_1d

        self.magnitude = coeffs
        if squeezed:
            return coeffs.squeeze(0)
        return coeffs

    def inverse(self, coeffs: torch.Tensor = None) -> torch.Tensor:
        """Inverse Haar wavelet decomposition.

        If `out_shape` was used in apply(), accepts [B,H,W] and returns [B,L].
        Crops back to the original input length when known.
        """
        if coeffs is None:
            coeffs = self.magnitude
        if coeffs is None:
            raise ValueError("HaarWaveletTransform.inverse() requires coeffs or a prior apply() call")

        y = coeffs.to(dtype=self.torch_dtype, device=self.device)
        squeezed = False
        if y.ndim == 2 and self.out_shape is not None:
            y = y.unsqueeze(0)
            squeezed = True
        if y.ndim == 3:
            # [B,H,W] -> [B, H*W]
            y = y.reshape(y.shape[0], -1)
        elif y.ndim == 2:
            # already [B,L]
            pass
        elif y.ndim == 1:
            y = y.unsqueeze(0)
            squeezed = True
        else:
            raise ValueError(f"HaarWaveletTransform.inverse expects [B,L], [L], [B,H,W], or [H,W]; got {tuple(y.shape)}")

        recon = self._haar_inverse_1d(y)

        out_len = int(self.input_length) if self.input_length is not None else recon.shape[-1]
        recon = recon[..., :out_len]

        if squeezed:
            return recon.squeeze(0)
        return recon


class NormalizationTransform(SignalTransform):
    """Normalize data to a target range.

    This supports two use-cases:
    - Explicit min/max provided (used by diffusion runner/sampler)
    - If not provided, infer min/max from the first `apply()` call
    """

    def __init__(
        self,
        data_min: float = None,
        data_max: float = None,
        target_min: float = -1.0,
        target_max: float = 1.0,
        device=None,
        torch_dtype=torch.float32,
    ):
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        super().__init__(torch_dtype, device)
        self.type = 'normalize'
        self.data_min = data_min
        self.data_max = data_max
        self.target_min = target_min
        self.target_max = target_max
        self.params = {
            'data_min': data_min,
            'data_max': data_max,
            'target_min': target_min,
            'target_max': target_max,
        }

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(dtype=self.torch_dtype, device=self.device)

        if self.data_min is None or self.data_max is None:
            self.data_min = float(data.min().item())
            self.data_max = float(data.max().item())
            self.params['data_min'] = self.data_min
            self.params['data_max'] = self.data_max

        normalized01 = (data - self.data_min) / (self.data_max - self.data_min + 1e-8)
        return normalized01 * (self.target_max - self.target_min) + self.target_min

    def inverse(self, data: torch.Tensor = None) -> torch.Tensor:
        if data is None:
            raise ValueError("Data must be provided for inverse().")
        data = data.to(dtype=self.torch_dtype, device=self.device)
        denorm01 = (data - self.target_min) / (self.target_max - self.target_min + 1e-8)
        return denorm01 * (self.data_max - self.data_min) + self.data_min


AVAILABLE_TRANSFORMS = {
    'stft': STFTTransform,
    'dft': DFTTransform,
    'haar': HaarWaveletTransform,
    'normalize': NormalizationTransform,
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




