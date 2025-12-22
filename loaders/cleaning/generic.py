"""Generic text cleaner for unregistered datasets."""

from __future__ import annotations

import re
from pathlib import Path

import click
from datasets import load_from_disk

from loaders.cleaning.base import BaseCleaner


class GenericCleaner(BaseCleaner):
    """generic text cleaner with whitespace normalization and encoding fixes"""

    def clean(self, input_dir: Path, output_dir: Path) -> None:
        """clean a dataset with generic text cleaning operations"""
        click.echo(f"Loading dataset from {input_dir}")
        dataset = load_from_disk(str(input_dir))

        click.echo("Applying generic text cleaning...")
        cleaned_dataset = dataset.map(
            self._clean_text,
            desc="Cleaning text",
        )

        # ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"Saving cleaned dataset to {output_dir}")
        cleaned_dataset.save_to_disk(str(output_dir))

        click.secho(f"✓ Dataset cleaned and saved to {output_dir.absolute()}", fg="green")

    def _clean_text(self, example: dict) -> dict:
        """apply generic cleaning to a text example"""
        # find text field (usually 'text', 'content', or similar)
        text_field = self._find_text_field(example)
        if not text_field:
            return example

        text = example[text_field]

        # normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # remove trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]

        # join and normalize multiple newlines to max 2
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # normalize multiple spaces to single space
        text = re.sub(r' {2,}', ' ', text)

        example[text_field] = text
        return example

    def _find_text_field(self, example: dict) -> str | None:
        """find the main text field in the dataset example"""
        common_text_fields = ['text', 'content', 'body', 'article', 'passage']
        for field in common_text_fields:
            if field in example:
                return field
        # fallback: return first string field
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 100:  # likely the main text
                return key
        return None


__all__ = ["GenericCleaner"]
