"""Click-based command-line interface for Kilonova."""

from __future__ import annotations

from pathlib import Path

import click

from kilonova.commands.evaluate import evaluate_model as evaluate_model_impl
from kilonova.commands.generate import generate_cmd
from kilonova.commands.train import train_model
from models.architectures import MODEL_REGISTRY


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
def cli() -> None:
    """kilonova - unified CLI for training LLMs from scratch."""


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
    """Train a GPT model on preprocessed data."""
    train_model(
        architecture=architecture,
        data=data,
        num_iterations=num_iterations,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
        data_fraction=data_fraction,
        eval_every=eval_every,
        device=device,
    )


@cli.command(help="Evaluate a trained model.")
@click.argument("model")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", help="device for evaluation.")
@click.option("--output", type=click.Path(path_type=Path), help="save results to file.")
@click.option("--prompt", type=str, help="test prompt for generation.")
def evaluate(model: str, device: str, output: Path | None, prompt: str | None) -> None:
    """Run comprehensive evaluation test suite on a trained model."""
    evaluate_model_impl(model, device, output, prompt)


@cli.command(help="Generate text from a trained model.")
@click.argument("model")
@click.option("--prompt", type=str, help="generation prompt (omit for interactive mode).")
@click.option("--interactive", is_flag=True, help="force interactive mode.")
@click.option("--temp", type=float, default=1.0, show_default=True, help="sampling temperature.")
@click.option("--max-tokens", type=int, default=50, show_default=True, help="maximum tokens to generate.")
@click.option("--top-k", type=int, help="top-k sampling.")
@click.option("--top-p", type=float, help="nucleus sampling.")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", show_default=True, help="device for generation.")
def generate(model: str, prompt: str | None, interactive: bool, temp: float, max_tokens: int, top_k: int | None, top_p: float | None, device: str) -> None:
    """Interactive text generation from a trained model."""
    generate_cmd(model, prompt, interactive, temp, max_tokens, top_k, top_p, device)


def main() -> None:
    """Allow setuptools entrypoints to reference a stable callable."""
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
