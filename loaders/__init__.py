"""
Data loading utilities for GPT training.

Provides memory-mapped dataloaders that read from
pre-tokenized binary files.

Usage:
    from loaders import create_dataloaders
    train_loader, val_loader = create_dataloaders("data/processed", batch_size=8)
"""

from .dataloader import create_dataloaders, TokenDataset

__all__ = [
    "create_dataloaders",
    "TokenDataset",
]
