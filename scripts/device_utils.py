from __future__ import annotations

import os
from typing import Final

import torch

VALMET_DEVICE_ENV: Final[str] = "VALMET_DEVICE"


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def pick_device() -> torch.device:
    """Prefer Metal (MPS) on Apple Silicon, allow env overrides, else CUDA/CPU."""
    override = os.environ.get(VALMET_DEVICE_ENV)
    if override:
        normalized = override.strip().lower()
        if normalized in {"mps", "metal"} and _mps_available():
            return torch.device("mps")
        if normalized in {"cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda")
        if normalized in {"cpu", "host"}:
            return torch.device("cpu")
        print(
            f"Requested device '{override}' unavailable. "
            "Falling back to automatic selection."
        )

    if _mps_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


__all__ = ["pick_device", "VALMET_DEVICE_ENV"]
