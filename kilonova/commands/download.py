"""Download command implementation."""

from __future__ import annotations

from pathlib import Path

import click

from loaders.cleaning.registry import get_source, is_registered
from loaders.downloading import download_from_huggingface
from kilonova.utils import PathResolver


def download_dataset(dataset: str, output: Path | None) -> None:
    """download a dataset from the registry or HuggingFace

    args:
        dataset: dataset name (registered) or huggingface:repo/path format
        output: optional custom output directory
    """
    resolver = PathResolver()

    # determine source and output directory
    if dataset.startswith("huggingface:"):
        # generic huggingface dataset
        source = dataset.replace("huggingface:", "")
        dataset_name = source.split("/")[-1]  # extract name from repo/path
        output_dir = output or resolver.raw_dir(dataset_name)
    elif is_registered(dataset):
        # built-in dataset from registry
        source = get_source(dataset)
        if source and source.startswith("huggingface:"):
            source = source.replace("huggingface:", "")
        output_dir = output or resolver.raw_dir(dataset)
    else:
        raise click.ClickException(
            f"Dataset '{dataset}' not found in registry.\n"
            f"Use format: huggingface:username/dataset for custom datasets."
        )

    # download from huggingface
    download_from_huggingface(source, output_dir)

    # show next steps
    click.echo(f"\nNext steps:")
    click.echo(f"  kilonova clean {dataset}")


__all__ = ["download_dataset"]
