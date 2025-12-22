"""
Data loading utilities for GPT training.

Provides memory-efficient streaming dataloaders that read from
pre-tokenized binary files.

Usage:
    # Preprocessing (run once)
    uv run python -m loaders.preprocessing --input data/raw --output data/processed

    # Training
    from loaders import create_dataloaders
    train_loader, val_loader = create_dataloaders("data/processed", batch_size=8)
"""

from .dataloader import create_dataloaders, StreamingTokenDataset

__all__ = [
    "create_dataloaders",
    "StreamingTokenDataset",
]
