"""
Map-style dataset for pre-tokenized binary data.

Uses memory-mapped files for zero-copy random access to token data,
enabling training on datasets larger than available RAM.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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
