"""
Memory-efficient streaming dataset for pre-tokenized binary data.

Uses memory-mapped files for zero-copy access to token data,
enabling training on datasets larger than available RAM.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset


class StreamingTokenDataset(IterableDataset):
    """Memory-efficient streaming dataset that reads from pre-tokenized binary files.

    Uses numpy memory mapping for zero-copy access. Only the pages being
    actively read are loaded into memory, enabling training on datasets
    much larger than available RAM.

    Automatically partitions data across DataLoader workers to avoid
    duplicate samples when num_workers > 0.

    Args:
        bin_path: Path to the .bin file containing tokens
        max_length: Context window size (sequence length)
        stride: Step size between sequences (default: max_length, no overlap)
        max_tokens: Optional limit on number of tokens to use
        shuffle_chunks: Whether to shuffle chunk order each epoch
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        bin_path: str | Path,
        max_length: int,
        stride: int | None = None,
        max_tokens: int | None = None,
        shuffle_chunks: bool = True,
        seed: int = 42,
    ):
        self.bin_path = Path(bin_path)
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        self.shuffle_chunks = shuffle_chunks
        self.base_seed = seed
        self.seed = seed

        # determine dtype from metadata.json if available
        meta_path = self.bin_path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            dtype_name = meta.get("dtype", "uint16")
            self.dtype = np.uint16 if dtype_name == "uint16" else np.uint32
        else:
            self.dtype = np.uint16

        # memory-map the file
        self.tokens = np.memmap(self.bin_path, dtype=self.dtype, mode='r')

        if max_tokens is not None and max_tokens < len(self.tokens):
            self.num_tokens = max_tokens
        else:
            self.num_tokens = len(self.tokens)

        # each sequence needs max_length + 1 tokens (input + shifted target)
        self.num_samples = max(0, (self.num_tokens - self.max_length - 1) // self.stride + 1)

    def __len__(self):
        """Return number of samples (for progress bars and logging)."""
        return self.num_samples

    def __iter__(self):
        """Iterate through sequences with automatic worker partitioning."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rng = random.Random(self.seed)

        if self.shuffle_chunks:
            # shuffle all indices first, then partition
            indices = list(range(self.num_samples))
            rng.shuffle(indices)

            # each worker gets every nth sample
            worker_indices = indices[worker_id::num_workers]
        else:
            # contiguous partition for each worker
            worker_indices = range(worker_id, self.num_samples, num_workers)

        for idx in worker_indices:
            start_idx = idx * self.stride

            # slice and copy the small chunk (avoids keeping memmap references)
            input_tokens = np.array(self.tokens[start_idx:start_idx + self.max_length], dtype=np.int64)
            target_tokens = np.array(self.tokens[start_idx + 1:start_idx + self.max_length + 1], dtype=np.int64)

            yield torch.from_numpy(input_tokens), torch.from_numpy(target_tokens)

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across epochs."""
        self.seed = self.base_seed + epoch
