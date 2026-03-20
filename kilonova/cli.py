"""Click-based command-line interface for Kilonova."""

from __future__ import annotations

import click

from kilonova.train import train_model
from models.architectures import MODEL_REGISTRY


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
def cli() -> None:
    """kilonova - CLI for training LLMs from scratch."""


@cli.command(help="Train a model on preprocessed data.")
@click.argument("architecture", type=click.Choice(tuple(MODEL_REGISTRY.keys())))
@click.option("--data", required=True, help="dataset name (uses data/processed/<dataset>/).")
@click.option("--num-iterations", type=click.IntRange(min=1), default=1000, show_default=True, help="total optimizer steps.")
@click.option("--batch-size", type=click.IntRange(min=1), default=2, show_default=True, help="micro-batch size per accumulation step.")
@click.option("--grad-accum-steps", type=click.IntRange(min=1), default=2, show_default=True, help="gradient accumulation steps (effective batch = batch-size * grad-accum-steps).")
@click.option("--learning-rate", type=float, default=4e-4, show_default=True, help="peak learning rate.")
@click.option("--data-fraction", type=float, default=None, help="fraction of data to use (0-1).")
@click.option("--eval-every", type=int, default=250, show_default=True, help="eval every N steps (-1 to disable).")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", show_default=True, help="device for training.")
def train(architecture: str, data: str, num_iterations: int, batch_size: int, grad_accum_steps: int, learning_rate: float, data_fraction: float | None, eval_every: int, device: str) -> None:
    """Train a model on preprocessed data."""
    try:
        train_model(
            model_name=architecture,
            data=data,
            device=device,
            num_iterations=num_iterations,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            learning_rate=learning_rate,
            data_fraction=data_fraction,
            eval_every=eval_every,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e


def main() -> None:
    """Allow setuptools entrypoints to reference a stable callable."""
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
