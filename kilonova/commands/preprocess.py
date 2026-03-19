"""Preprocess command implementation."""

from __future__ import annotations

from pathlib import Path

import click

from loaders.preprocessing import preprocess_dataset
from osmium.utils import PathResolver


def preprocess_data(dataset: str, input_dir: Path | None, output: Path | None, train_split: float) -> None:
    """tokenize text files into binary format

    args:
        dataset: dataset name
        input_dir: optional custom input directory
        output: optional custom output directory
        train_split: train/validation split ratio
    """
    resolver = PathResolver()

    # resolve input and output directories
    input_path = input_dir or resolver.clean_dir(dataset)
    output_path = output or resolver.processed_dir(dataset)

    # validate input directory exists
    if not input_path.exists():
        raise click.ClickException(
            f"Input directory not found: {input_path}\n"
            f"Run: osmium clean {dataset}"
        )

    click.echo(f"Preprocessing dataset: {dataset}")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Train/validation split: {train_split:.2f}/{1-train_split:.2f}")

    # run preprocessing
    preprocess_dataset(
        input_dir=str(input_path),
        output_dir=str(output_path),
        tokenizer_name="gpt2",
        train_ratio=train_split,
        num_files=None,
    )

    click.secho(f"✓ Dataset preprocessed and saved to {output_path.absolute()}", fg="green")

    # show next steps
    click.echo(f"\nNext steps:")
    click.echo(f"  osmium train gpt2-small --data {dataset}")


__all__ = ["preprocess_data"]
