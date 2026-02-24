"""Quantization debugging helpers.

These utilities are intentionally lightweight and print-focused so we can
validate that training/sampling/analysis all use consistent quantization ranges.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UniformQuantizerSpec:
    bits: int
    range_min: float
    range_max: float
    step_size: float
    levels: int


def _subsample_flat(x: torch.Tensor, max_elems: int = 200_000) -> torch.Tensor:
    x = x.detach().flatten()
    if x.numel() <= max_elems:
        return x
    # Deterministic stride sampling (no RNG needed).
    stride = int(max(1, x.numel() // max_elems))
    return x[::stride]


def summarize_tensor(
    x: torch.Tensor,
    name: str,
    *,
    quantiles: tuple[float, float, float] = (0.01, 0.5, 0.99),
    max_elems: int = 200_000,
) -> None:
    xs = _subsample_flat(x, max_elems=max_elems).to(torch.float32).cpu()
    q_lo, q_med, q_hi = quantiles
    try:
        qv = torch.quantile(xs, torch.tensor([q_lo, q_med, q_hi]))
        qv = [float(v) for v in qv]
    except Exception:
        qv = [float(xs.min().item()), float(xs.median().item()), float(xs.max().item())]

    print(
        f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={float(xs.min()):.6g} max={float(xs.max()):.6g} "
        f"mean={float(xs.mean()):.6g} std={float(xs.std(unbiased=False)):.6g} "
        f"q{int(q_lo*100)}={qv[0]:.6g} q50={qv[1]:.6g} q{int(q_hi*100)}={qv[2]:.6g}"
    )


def uniform_quantizer_spec(q) -> UniformQuantizerSpec:
    # Works with data.preprocess.quantize.UniformQuantizer.
    bits = int(getattr(q, "bits"))
    levels = int(getattr(q, "levels"))
    range_min = float(getattr(q, "range_min"))
    range_max = float(getattr(q, "range_max"))
    step_size = float(getattr(q, "step_size"))
    return UniformQuantizerSpec(bits=bits, range_min=range_min, range_max=range_max, step_size=step_size, levels=levels)


def summarize_uniform_quantizer(q, name: str) -> None:
    spec = uniform_quantizer_spec(q)
    print(
        f"[{name}] bits={spec.bits} levels={spec.levels} "
        f"range=[{spec.range_min:.6g}, {spec.range_max:.6g}] step={spec.step_size:.6g}"
    )

    meta = getattr(q, "_meta", None)
    if isinstance(meta, dict) and meta:
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
        print(f"[{name}] meta: {meta_str}")


def summarize_quantization_usage(
    x: torch.Tensor,
    q,
    name: str,
    *,
    max_elems: int = 200_000,
    max_bins_to_print: int = 32,
) -> None:
    """Print how many quantization bins are actually being used.

    Uses q.quantize_indices(x) and reports unique bin count and the most common bins.
    """
    # Subsample before quantizing indices to avoid huge CPU work.
    xs = _subsample_flat(x, max_elems=max_elems)
    idx = q.quantize_indices(xs)
    idx = idx.detach().to(torch.int64).cpu()

    unique, counts = torch.unique(idx, return_counts=True)
    n_unique = int(unique.numel())
    print(f"[{name}] used_bins={n_unique}/{int(getattr(q, 'levels', 0) or 0)}")
    if n_unique == 0:
        return

    # Show most common bins.
    order = torch.argsort(counts, descending=True)
    unique = unique[order]
    counts = counts[order]

    to_show = min(int(max_bins_to_print), int(unique.numel()))
    pairs = []
    for i in range(to_show):
        pairs.append(f"{int(unique[i])}:{int(counts[i])}")
    print(f"[{name}] top_bins(count): " + " ".join(pairs))
