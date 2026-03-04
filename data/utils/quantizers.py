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


import math


class LloydMaxQuantizer(Quantizer):
    """Optimal non-uniform (Lloyd-Max) quantizer.

    Minimizes quantization MSE by iteratively fitting bin boundaries and
    centroids to the empirical distribution of the training data.

    Usage::
        q = LloydMaxQuantizer(bits=4, range_min=-1.0, range_max=1.0)
        q.fit(training_data)          # fit codebook once
        out = q.quantize(new_signal)  # apply to any signal in range
    """

    def __init__(
        self,
        bits: int,
        range_min: float,
        range_max: float,
        max_iter: int = 100,
        tol: float = 1e-7,
    ):
        self.bits = bits
        self.levels = 2 ** bits
        self.quantizer_type = "lloyd_max"
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.max_iter = max_iter
        self.tol = tol

        # Initialise with uniform codebook (will be refined by fit()).
        step = (self.range_max - self.range_min) / float(self.levels)
        self.centroids: torch.Tensor = torch.linspace(
            self.range_min + 0.5 * step,
            self.range_max - 0.5 * step,
            self.levels,
        )
        self.boundaries: torch.Tensor = torch.linspace(
            self.range_min, self.range_max, self.levels + 1
        )

    def fit(self, data: torch.Tensor) -> "LloydMaxQuantizer":
        """Fit the Lloyd-Max codebook to the provided data tensor.

        Iterates until centroids converge or ``max_iter`` is reached.
        Returns *self* for chaining.
        """
        x = data.detach().cpu().float().flatten()
        x = torch.clamp(x, self.range_min, self.range_max)

        centroids = self.centroids.clone().float()
        rmin, rmax = self.range_min, self.range_max

        for _ in range(self.max_iter):
            # Step 1: decision boundaries = midpoints of adjacent centroids.
            midpoints = (centroids[:-1] + centroids[1:]) / 2.0
            boundaries = torch.cat(
                [torch.tensor([rmin]), midpoints, torch.tensor([rmax])]
            )

            # Step 2: new centroids = conditional mean inside each bin.
            new_centroids = torch.empty_like(centroids)
            for k in range(self.levels):
                lo = float(boundaries[k].item())
                hi = float(boundaries[k + 1].item())
                mask = (x >= lo) & (x < hi) if k < self.levels - 1 else (x >= lo) & (x <= hi)
                pts = x[mask]
                new_centroids[k] = pts.mean() if len(pts) > 0 else (lo + hi) / 2.0

            delta = (new_centroids - centroids).abs().max().item()
            centroids = new_centroids
            if delta < self.tol:
                break

        self.centroids = centroids
        midpoints = (centroids[:-1] + centroids[1:]) / 2.0
        self.boundaries = torch.cat(
            [torch.tensor([rmin]), midpoints, torch.tensor([rmax])]
        )
        return self

    def quantize(self, signal: torch.Tensor) -> torch.Tensor:
        """Map each sample to its nearest Lloyd-Max centroid."""
        x = signal.float()
        x_clip = torch.clamp(x, self.range_min, self.range_max)
        # Nearest centroid via vectorised L1 distance.
        diff = (x_clip.unsqueeze(-1) - self.centroids.to(x.device)).abs()  # [..., K]
        idx = diff.argmin(dim=-1)
        return self.centroids.to(x.device)[idx].to(signal.dtype)

    def dequantize(self, quantized_signal: torch.Tensor) -> torch.Tensor:
        """Already in decoded space; pass through."""
        return quantized_signal


class MuLawQuantizer(Quantizer):
    """μ-law companding quantizer.

    Applies logarithmic μ-law compression to the normalised signal, then
    performs uniform quantisation in the compressed domain and expands back.
    Well-suited for signals with high dynamic range (e.g. speech / ECG).

    The compression function follows ITU-T G.711 (μ = 255 by default)::

        y = sgn(x̂) · ln(1 + μ|x̂|) / ln(1 + μ),  x̂ = x / peak ∈ [-1, 1]
    """

    def __init__(
        self,
        bits: int,
        range_min: float,
        range_max: float,
        mu: float = 255.0,
    ):
        self.bits = bits
        self.levels = 2 ** bits
        self.quantizer_type = "mu_law"
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.mu = float(mu)
        self.step_size = 2.0 / float(self.levels)  # step in compressed [-1, 1] domain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compress(self, x_norm: torch.Tensor) -> torch.Tensor:
        """μ-law compression: x̂ ∈ [-1,1] → y ∈ [-1,1]."""
        mu = self.mu
        return x_norm.sign() * torch.log1p(mu * x_norm.abs()) / math.log(1.0 + mu)

    def _expand(self, y: torch.Tensor) -> torch.Tensor:
        """μ-law expansion (inverse): y ∈ [-1,1] → x̂ ∈ [-1,1]."""
        mu = self.mu
        return y.sign() * ((1.0 + mu) ** y.abs() - 1.0) / mu

    # ------------------------------------------------------------------
    # Quantizer interface
    # ------------------------------------------------------------------

    def quantize(self, signal: torch.Tensor) -> torch.Tensor:
        """Compress → uniform-quantise → expand → scale back."""
        x = signal.float()
        peak = max(abs(self.range_min), abs(self.range_max))
        peak = max(peak, 1e-12)

        x_norm = torch.clamp(x / peak, -1.0, 1.0)
        y = self._compress(x_norm)          # in [-1, 1]

        levels = float(self.levels)
        step = 2.0 / levels
        idx = torch.floor((y + 1.0) / step).clamp(0, levels - 1)
        y_q = idx * step - 1.0 + 0.5 * step  # bin centres in compressed domain

        x_q_norm = self._expand(y_q)
        return (x_q_norm * peak).to(signal.dtype)

    def dequantize(self, quantized_signal: torch.Tensor) -> torch.Tensor:
        """Already decoded; pass through."""
        return quantized_signal


class DitheredUniformQuantizer(Quantizer):
    """Uniform quantizer with TPDF (triangular probability density function) dither.

    Adding dither noise before quantisation trades correlated harmonic distortion
    for flat, signal-independent noise — the classic technique used in audio mastering
    and high-fidelity DSP.  The dither amplitude equals ±1 LSB (one quantisation step),
    which is the minimum necessary to linearise the quantiser.

    The dither is added in the *time domain* prior to quantisation and is NOT subtracted
    afterwards (the same convention used in perceptual audio codecs).  This means the
    quantisation error becomes uncorrelated white noise with variance ≈ Δ²/6 (same as
    un-dithered uniform) but without harmonic distortion artifacts.

    TPDF is generated as the sum of two independent uniform [-Δ/2, Δ/2] r.v.s,
    giving a triangular distribution on [-Δ, Δ].
    """

    def __init__(
        self,
        bits: int,
        range_min: float,
        range_max: float,
        seed: int | None = None,
    ):
        self.bits = bits
        self.levels = 2 ** bits
        self.quantizer_type = "dithered_uniform"
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.step_size = (self.range_max - self.range_min) / float(self.levels)
        self._seed = seed
        self._rng: torch.Generator | None = None
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)

    def _tpdf_dither(self, shape: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate TPDF dither: sum of two uniform [-step/2, step/2] random variables."""
        half = self.step_size / 2.0
        # Two independent U[-half, half] variables.
        kwargs = {} if self._rng is None else {"generator": self._rng}
        u1 = torch.zeros(shape, device=device).uniform_(-half, half, **kwargs)
        u2 = torch.zeros(shape, device=device).uniform_(-half, half, **kwargs)
        return (u1 + u2).to(dtype)

    def quantize(self, signal: torch.Tensor) -> torch.Tensor:
        """Add TPDF dither, then uniformly quantise; return decoded bin-centre values."""
        dither = self._tpdf_dither(signal.shape, signal.device, signal.dtype)
        x_dithered = torch.clamp(signal + dither, self.range_min, self.range_max)

        indices = torch.floor((x_dithered - self.range_min) / self.step_size)
        indices = torch.clamp(indices, 0, self.levels - 1)
        return (indices * self.step_size + self.range_min + self.step_size / 2.0).to(signal.dtype)

    def dequantize(self, quantized_signal: torch.Tensor) -> torch.Tensor:
        """Already decoded; pass through."""
        return quantized_signal
