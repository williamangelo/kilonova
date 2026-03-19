"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class TrainConfig:
    """Resolved configuration passed to the training runner."""

    data_dir: Path
    run_dir: Path | None
    model: str
    num_iterations: int
    batch_size: int
    learning_rate: float
    data_fraction: float | None
    eval_every: int
    device: torch.device


def resolve_device(device_spec: str) -> torch.device:
    """Resolve the requested device string into a torch.device."""
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_spec == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

    return torch.device(device_spec)


def detect_compute_dtype(device: torch.device) -> tuple[torch.dtype, str]:
    """auto-detect optimal compute dtype based on GPU capability.

    bf16 on Ampere+ (SM 80+), fp32 otherwise.
    """
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"CUDA SM {capability[0]}{capability[1]} (bf16)"
        return torch.float32, f"CUDA SM {capability[0]}{capability[1]} (pre-Ampere, fp32)"
    return torch.float32, f"no CUDA ({device.type})"


__all__ = ["TrainConfig", "resolve_device", "detect_compute_dtype"]
