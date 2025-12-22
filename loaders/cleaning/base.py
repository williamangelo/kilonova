"""Base cleaner interface for dataset cleaning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseCleaner(ABC):
    """abstract base class for dataset cleaners"""

    @abstractmethod
    def clean(self, input_dir: Path, output_dir: Path) -> None:
        """clean a dataset from input_dir and save to output_dir"""
        raise NotImplementedError


__all__ = ["BaseCleaner"]
