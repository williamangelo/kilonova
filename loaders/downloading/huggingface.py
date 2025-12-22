"""Generic HuggingFace dataset downloader."""

from __future__ import annotations

from pathlib import Path

import click
from datasets import load_dataset


def download_from_huggingface(source: str, output_dir: Path, split: str = "train") -> None:
    """download a dataset from HuggingFace and save to disk

    args:
        source: huggingface dataset identifier (e.g., 'pg19' or 'username/dataset')
        output_dir: directory where the dataset will be saved
        split: dataset split to download (default: 'train')
    """
    click.echo(f"Downloading from HuggingFace: {source}")
    click.echo(f"Split: {split}")
    click.echo("This may take a while depending on your connection speed...")

    # ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # download dataset
    dataset = load_dataset(source, split=split)

    click.echo(f"\nDownloaded {len(dataset)} items")
    click.echo(f"Dataset fields: {dataset.column_names}")

    # save to disk
    click.echo(f"\nSaving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))

    click.secho(f"✓ Dataset successfully saved to {output_dir.absolute()}", fg="green")


__all__ = ["download_from_huggingface"]
