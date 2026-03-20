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
    data_fraction: float | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders from preprocessed data.

    Args:
        data_dir: Directory containing train.bin, val.bin, and metadata.json
        batch_size: Batch size for training
        max_length: Context window size (sequence length)
        data_fraction: Fraction of data to use (optional)

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

    if data_fraction is not None:
        max_tokens = int(meta["train_tokens"] * data_fraction)
        val_max_tokens = int(meta["val_tokens"] * data_fraction)
    else:
        max_tokens = None
        val_max_tokens = None

    train_dataset = TokenDataset(
        bin_path=train_path,
        max_length=max_length,
        max_tokens=max_tokens,
    )

    val_dataset = TokenDataset(
        bin_path=val_path,
        max_length=max_length,
        max_tokens=val_max_tokens,
    )

    logger.info(f"Loaded preprocessed data from {data_dir}")
    logger.info(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")

    if max_tokens is not None and val_max_tokens is not None:
        logger.info(f"Token limit: {max_tokens:,} train, {val_max_tokens:,} val ({max_tokens + val_max_tokens:,} total)")

    generator = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
