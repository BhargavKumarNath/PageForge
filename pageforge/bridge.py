"""Zero-copy tensor exchange between PyTorch and CuPy via DLPack.

Both directions share the same device allocation: writing through one view is
visible through the other, and nothing is copied to the host.
"""

from __future__ import annotations

import cupy as cp
import torch


def torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    """View a CUDA torch tensor as a CuPy array (no copy)."""
    if not x.is_cuda:
        raise ValueError("torch_to_cupy needs a CUDA tensor")
    return cp.from_dlpack(x.detach())


def cupy_to_torch(x: cp.ndarray) -> torch.Tensor:
    """View a CuPy array as a torch tensor (no copy)."""
    return torch.from_dlpack(x)


def shares_memory(a: torch.Tensor, b: cp.ndarray) -> bool:
    """True when the torch tensor and CuPy array start at the same device address."""
    return a.data_ptr() == b.data.ptr
