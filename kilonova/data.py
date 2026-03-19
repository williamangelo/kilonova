"""
Training data utilities.

Provides memory-mapped datasets and dataloader factories
for pre-tokenized binary data.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """Map-style dataset backed by a memory-mapped binary token file.

    Each sample is a (input, target) pair where target is input shifted
    right by one token.

    Args:
        bin_path: Path to the .bin file containing tokens
        max_length: Context window size (sequence length)
        stride: Step size between sequences (default: max_length, no overlap)
        max_tokens: Optional limit on number of tokens to use
    """

    def __init__(
        self,
        bin_path: str | Path,
        max_length: int,
        stride: int | None = None,
        max_tokens: int | None = None,
    ):
        self.bin_path = Path(bin_path)
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length

        # determine dtype from sibling metadata.json, fall back to uint16
        meta_path = self.bin_path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            dtype_name = meta.get("dtype", "uint16")
            self.dtype = np.uint16 if dtype_name == "uint16" else np.uint32
        else:
            self.dtype = np.uint16

        self.tokens = np.memmap(self.bin_path, dtype=self.dtype, mode='r')

        if max_tokens is not None and max_tokens < len(self.tokens):
            self.num_tokens = max_tokens
        else:
            self.num_tokens = len(self.tokens)

        # each sample needs max_length + 1 tokens (input + shifted target)
        self.num_samples = max(0, (self.num_tokens - self.max_length - 1) // self.stride + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"index {idx} out of range for dataset of size {self.num_samples}")
        start = idx * self.stride
        input_tokens = np.array(self.tokens[start:start + self.max_length], dtype=np.int64)
        target_tokens = np.array(self.tokens[start + 1:start + self.max_length + 1], dtype=np.int64)
        return torch.from_numpy(input_tokens), torch.from_numpy(target_tokens)


def load_meta(data_dir: str | Path) -> dict:
    """Load metadata from a preprocessed dataset directory."""
    meta_path = Path(data_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {data_dir}. "
            "Run the dataset preparation script first (e.g. uv run scripts/gutenberg.py)"
        )

    with open(meta_path) as f:
        return json.load(f)


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    max_length: int,
    stride: int | None = None,
    max_tokens: int | None = None,
    data_fraction: float | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders from preprocessed data.

    Args:
        data_dir: Directory containing train.bin, val.bin, and metadata.json
        batch_size: Batch size for training
        max_length: Context window size (sequence length)
        stride: Step size between sequences (default: max_length)
        max_tokens: Maximum number of tokens to use (for small runs)
        data_fraction: Fraction of data to use (alternative to max_tokens)
        shuffle: Whether to shuffle training data each epoch
        num_workers: Number of DataLoader workers (0 for main process)
        drop_last: Whether to drop incomplete final batch
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_path = Path(data_dir)
    train_path = data_path / "train.bin"
    val_path = data_path / "val.bin"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.bin not found in {data_dir}. "
            "Run the dataset preparation script first (e.g. uv run scripts/gutenberg.py)"
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"val.bin not found in {data_dir}. "
            "Run the dataset preparation script first (e.g. uv run scripts/gutenberg.py)"
        )

    meta = load_meta(data_dir)

    # calculate max_tokens from data_fraction if specified
    if data_fraction is not None and max_tokens is None:
        max_tokens = int(meta["train_tokens"] * data_fraction)
        val_max_tokens = int(meta["val_tokens"] * data_fraction)
    elif max_tokens is not None:
        # treat max_tokens as total budget and split according to original train/val ratio
        total_tokens = meta["train_tokens"] + meta["val_tokens"]
        train_ratio = meta["train_tokens"] / total_tokens
        val_ratio = meta["val_tokens"] / total_tokens

        total_budget = max_tokens
        max_tokens = int(total_budget * train_ratio)
        val_max_tokens = int(total_budget * val_ratio)
    else:
        val_max_tokens = None

    train_dataset = TokenDataset(
        bin_path=train_path,
        max_length=max_length,
        stride=stride,
        max_tokens=max_tokens,
    )

    val_dataset = TokenDataset(
        bin_path=val_path,
        max_length=max_length,
        stride=stride,
        max_tokens=val_max_tokens,
    )

    logger.info(f"Loaded preprocessed data from {data_dir}")
    logger.info(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")

    if max_tokens and val_max_tokens:
        logger.info(f"Token limit: {max_tokens:,} train, {val_max_tokens:,} val ({max_tokens + val_max_tokens:,} total)")

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        generator=generator,  # val loader uses shuffle=False, no generator needed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader
