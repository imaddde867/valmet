from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Prefer Metal (MPS) on Apple Silicon, then CUDA, else CPU."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
