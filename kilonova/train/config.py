"""Training configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import warnings

import torch

from osmium.utils.platform import is_macos

WarnFunc = Callable[[str], None]


@dataclass(slots=True)
class TrainConfig:
    """Resolved configuration passed to the training runner."""

    data_dir: Path
    run_dir: Path | None
    max_tokens: int | None
    data_fraction: float | None
    model: str
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int | None
    eval_freq: int
    eval_iter: int
    compile_model: bool
    mixed_precision: bool
    gradient_accumulation_steps: int
    device: torch.device
    num_workers: int
    max_grad_norm: float
    warmup_steps: int | None
    min_lr: float | None

    @classmethod
    def from_click(cls, params: Mapping[str, Any], warn: WarnFunc | None = None) -> "TrainConfig":
        """Construct TrainConfig from parsed Click parameters."""
        values = dict(params)
        device_spec = values.pop("device")
        mixed_precision_request = values.pop("mixed_precision")
        compile_request = values.pop("compile_model")
        num_workers_request = values.pop("num_workers")

        device = resolve_device(device_spec)

        # mixed precision: only works on CUDA
        mixed_precision = _resolve_mixed_precision(
            requested=mixed_precision_request,
            device=device,
            warn=warn,
        )

        # torch.compile: enabled by default on CUDA and CPU
        compile_model = _resolve_compile_model(compile_request)

        # num_workers: must be 0 on macOS due to multiprocessing issues
        num_workers = _resolve_num_workers(
            requested=num_workers_request,
            warn=warn,
        )

        return cls(
            data_dir=Path(values["data_dir"]),
            run_dir=Path(values["run_dir"]) if values.get("run_dir") else None,
            max_tokens=values["max_tokens"],
            data_fraction=values["data_fraction"],
            model=values["model"],
            epochs=values["epochs"],
            batch_size=values["batch_size"],
            learning_rate=values["learning_rate"],
            patience=values["patience"],
            eval_freq=values["eval_freq"],
            eval_iter=values["eval_iter"],
            compile_model=compile_model,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=values["gradient_accumulation_steps"],
            device=device,
            num_workers=num_workers,
            max_grad_norm=values["max_grad_norm"],
            warmup_steps=values["warmup_steps"],
            min_lr=values["min_lr"],
        )


def resolve_device(device_spec: str) -> torch.device:
    """Resolve the requested device string into a torch.device."""
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_spec == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

    return torch.device(device_spec)


def _resolve_mixed_precision(
    requested: bool | None,
    device: torch.device,
    warn: WarnFunc | None,
) -> bool:
    """resolve mixed precision setting with CPU check.

    mixed precision (fp16) only works on CUDA devices. warn and disable on CPU.
    """
    if device.type == "cpu":
        if requested is True:
            _emit_warning(
                warn,
                "mixed precision does not work on CPU, disabling. use CUDA/MPS for fp16 training.",
            )
        return False

    # enable by default on CUDA
    if device.type == "cuda":
        return True if requested is None else requested

    # disable on other devices (e.g., mps)
    return False if requested is None else requested


def _resolve_compile_model(requested: bool | None) -> bool:
    """resolve torch.compile setting.

    torch.compile provides 20-40% speedup on both CPU and CUDA, enabled by default.
    """
    # enable by default on all devices (CPU, CUDA, MPS)
    return True if requested is None else requested


def _resolve_num_workers(
    requested: int,
    warn: WarnFunc | None,
) -> int:
    """resolve num_workers with macOS check.

    macOS has multiprocessing issues with PyTorch DataLoader.
    force num_workers=0 on macOS.

    note: on macOS, consider using multiprocessing_context='spawn' in DataLoader
    instead of fork (default). however, this requires code changes in the dataloader
    factory. for now, we disable workers entirely on macOS.
    """
    if is_macos() and requested > 0:
        _emit_warning(
            warn,
            f"DataLoader workers (num_workers={requested}) not supported on macOS due to multiprocessing issues. "
            "forcing num_workers=0. consider using threading or spawn context in future.",
        )
        return 0

    return requested


def _emit_warning(warn: WarnFunc | None, message: str) -> None:
    """Emit warnings either through provided callback or python warnings."""
    if warn is not None:
        warn(message)
    else:
        warnings.warn(message, RuntimeWarning, stacklevel=2)


__all__ = ["TrainConfig", "resolve_device"]
