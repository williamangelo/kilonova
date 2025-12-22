"""Clean command implementation."""

from __future__ import annotations

from pathlib import Path

import click

from loaders.cleaning import get_cleaner, is_registered
from loaders.cleaning.generic import GenericCleaner
from osmium.utils import PathResolver


def clean_dataset(dataset: str, input_dir: Path | None, output: Path | None) -> None:
    """clean a dataset with dataset-specific or generic cleaning

    args:
        dataset: dataset name
        input_dir: optional custom input directory
        output: optional custom output directory
    """
    resolver = PathResolver()

    # resolve input and output directories
    input_path = input_dir or resolver.raw_dir(dataset)
    output_path = output or resolver.clean_dir(dataset)

    # validate input directory exists
    if not input_path.exists():
        raise click.ClickException(
            f"Input directory not found: {input_path}\n"
            f"Run: osmium download {dataset}"
        )

    # get cleaner (dataset-specific or generic)
    cleaner_class = get_cleaner(dataset) if is_registered(dataset) else GenericCleaner

    if cleaner_class is None:
        # registered but no custom cleaner defined, use generic
        click.echo(f"No custom cleaner for '{dataset}', using generic cleaner")
        cleaner_class = GenericCleaner
    elif cleaner_class != GenericCleaner and is_registered(dataset):
        click.echo(f"Using dataset-specific cleaner for '{dataset}'")
    else:
        click.echo(f"Using generic cleaner for '{dataset}'")

    # instantiate and run cleaner
    cleaner = cleaner_class()
    cleaner.clean(input_path, output_path)

    # show next steps
    click.echo(f"\nNext steps:")
    click.echo(f"  osmium preprocess {dataset}")


__all__ = ["clean_dataset"]
