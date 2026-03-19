"""Train command implementation."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import click

from models.architectures import is_registered as is_model_registered, list_models
from kilonova.train import resolve_device, run_training


def train_model(
    architecture: str,
    data: str,
    num_iterations: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    data_fraction: float | None,
    eval_every: int,
    device: str,
) -> None:
    """train a model using preprocessed data"""
    if not is_model_registered(architecture):
        raise click.ClickException(
            f"Unknown architecture: {architecture}\n"
            f"Available: {', '.join(list_models())}"
        )

    data_dir = Path("data/processed") / data

    if not data_dir.exists():
        raise click.ClickException(
            f"Dataset not preprocessed: {data}\n"
            f"Run the dataset preparation script first (e.g. uv run scripts/gutenberg.py)"
        )

    if not (data_dir / "train.bin").exists() or not (data_dir / "val.bin").exists():
        raise click.ClickException(
            f"Dataset incomplete: missing train.bin or val.bin in {data_dir}\n"
            f"Run the dataset preparation script first (e.g. uv run scripts/gutenberg.py)"
        )

    date = datetime.now().strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:7]
    run_id = f"run-{date}-{short_id}"
    run_dir = Path("data/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\nTraining run: {run_id}")
    click.echo(f"Architecture: {architecture}")
    click.echo(f"Dataset: {data}")
    click.echo(f"Run directory: {run_dir}\n")

    run_training(
        model_name=architecture,
        data_dir=data_dir,
        run_dir=run_dir,
        device=resolve_device(device),
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        data_fraction=data_fraction,
        eval_every=eval_every,
    )

    click.secho(f"\n✓ Training complete: {run_id}", fg="green")
    click.echo(f"\nNext steps:")
    click.echo(f"  kilonova evaluate {run_id}")


__all__ = ["train_model"]
