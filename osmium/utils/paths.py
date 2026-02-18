"""Path resolution utilities using conventions with override support."""

from __future__ import annotations

from pathlib import Path


class PathResolver:
    """resolves paths using conventions with override support"""

    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)

    def raw_dir(self, dataset: str, override: Path | None = None) -> Path:
        """resolve raw data directory for a dataset"""
        return override or self.base_dir / "raw" / dataset

    def clean_dir(self, dataset: str, override: Path | None = None) -> Path:
        """resolve clean data directory for a dataset"""
        return override or self.base_dir / "clean" / dataset

    def processed_dir(self, dataset: str, override: Path | None = None) -> Path:
        """resolve processed data directory for a dataset"""
        return override or self.base_dir / "processed" / dataset

    def run_dir(self, name: str) -> Path:
        """resolve training run directory"""
        return self.base_dir / "runs" / name



__all__ = ["PathResolver"]
