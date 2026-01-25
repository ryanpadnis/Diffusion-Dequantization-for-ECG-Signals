"""Small torch helpers shared across training/sampling/model."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import torch


TorchDTypeLike = Union[str, torch.dtype]


def resolve_torch_dtype(config: Mapping[str, Any], device: Optional[torch.device] = None) -> torch.dtype:
    """Resolve desired torch dtype from config with safe CPU fallback.

    Expects config['torch_dtype'] as a string like 'float16'/'float32' or a torch.dtype.
    If device is CPU and dtype is float16, falls back to float32.
    """
    dtype_val: TorchDTypeLike = config.get("torch_dtype", "float32")  # type: ignore[assignment]

    if isinstance(dtype_val, torch.dtype):
        dtype = dtype_val
    else:
        dtype = getattr(torch, str(dtype_val), torch.float32)

    dev = device
    if dev is None:
        dev = torch.device(config.get("device", "cpu"))

    if dev.type == "cpu" and dtype == torch.float16:
        return torch.float32

    return dtype
