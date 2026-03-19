"""Click-based command-line interface for Osmium."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from osmium.commands.clean import clean_dataset
from osmium.commands.download import download_dataset
from osmium.commands.evaluate import evaluate_model as evaluate_model_impl
from osmium.commands.generate import generate_cmd
from osmium.commands.info import info_cmd
from osmium.commands.list_cmd import list_datasets_cmd, list_models_cmd
from osmium.commands.preprocess import preprocess_data
from osmium.commands.train import train_model
from models.architectures import MODEL_REGISTRY


def _warn(message: str) -> None:
    """Emit a warning message via Click."""
    click.secho(f"Warning: {message}", err=True, fg="yellow")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
def cli() -> None:
    """osmium - unified CLI for training LLMs from scratch."""


@cli.command(help="Download a dataset.")
@click.argument("dataset")
@click.option("--output", type=click.Path(path_type=Path), help="Output directory (default: data/raw/<dataset>).")
def download(dataset: str, output: Path | None) -> None:
    """Download a dataset from the registry or HuggingFace."""
    download_dataset(dataset, output)


@cli.command(help="Clean a dataset using dataset-specific logic.")
@click.argument("dataset")
@click.option("--input", "input_dir", type=click.Path(path_type=Path), help="Input directory (default: data/raw/<dataset>).")
@click.option("--output", type=click.Path(path_type=Path), help="Output directory (default: data/clean/<dataset>).")
def clean(dataset: str, input_dir: Path | None, output: Path | None) -> None:
    """Clean a dataset with dataset-specific or generic cleaning."""
    clean_dataset(dataset, input_dir, output)


@cli.command(help="Preprocess (tokenize) cleaned data.")
@click.argument("dataset")
@click.option("--input", "input_dir", type=click.Path(path_type=Path), help="Input directory (default: data/clean/<dataset>).")
@click.option("--output", type=click.Path(path_type=Path), help="Output directory (default: data/processed/<dataset>).")
@click.option("--train-split", type=float, default=0.95, show_default=True, help="Train/validation split ratio.")
def preprocess(dataset: str, input_dir: Path | None, output: Path | None, train_split: float) -> None:
    """Tokenize text files into binary format."""
    preprocess_data(dataset, input_dir, output, train_split)


@cli.command(help="Train a model using preprocessed data. Defaults follow modern LLM training practices (single-epoch, checkpoint-based evaluation).")
@click.argument("architecture", type=click.Choice(tuple(MODEL_REGISTRY.keys())))
@click.option("--data", required=True, help="Dataset name (uses data/processed/<dataset>/).")
@click.option("--name", type=str, help="Experiment name (auto-generated from config filename if using --config).")
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Load hyperparameters from YAML file.")
@click.option("--max-tokens", type=int, default=None, help="Cap the number of tokens to use.")
@click.option("--data-fraction", type=float, default=None, help="Fraction of data to consume (0-1).")
@click.option("--epochs", type=click.IntRange(min=1), default=1, show_default=True, help="Training epochs.")
@click.option("--batch-size", type=click.IntRange(min=1), default=2, show_default=True, help="Batch size per step.")
@click.option("--learning-rate", type=float, default=0.0004, show_default=True, help="Optimizer learning rate.")
@click.option("--patience", type=int, default=None, help="Early stopping patience (None=disabled, trains to completion).")
@click.option("--eval-freq", type=click.IntRange(min=1), default=500, show_default=True, help="Eval frequency in steps.")
@click.option("--eval-iter", type=click.IntRange(min=1), default=50, show_default=True, help="Batches to use for eval.")
@click.option("--mixed-precision/--no-mixed-precision", default=None, help="Toggle fp16 mixed precision.")
@click.option("--compile/--no-compile", "compile_model", default=None, help="Toggle torch.compile for speedups.")
@click.option("--gradient-accumulation-steps", type=click.IntRange(min=1), default=4, show_default=True, help="Number of gradient accumulation steps.")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", show_default=True, help="Device preference for training.")
@click.option("--num-workers", type=click.IntRange(min=0), default=0, show_default=True, help="Number of dataloader workers.")
@click.option("--max-grad-norm", type=float, default=1.0, show_default=True, help="Gradient clipping threshold (0 to disable).")
@click.option("--warmup-steps", type=int, default=None, help="Override LR warmup steps.")
@click.option("--min-lr", type=float, default=None, help="Minimum LR after cosine decay.")
def train(architecture: str, data: str, name: str | None, config: Path | None, **kwargs: Any) -> None:
    """Train a GPT model using preprocessed data."""
    train_model(architecture, data, name, config, kwargs)


@cli.command(help="Evaluate a trained model.")
@click.argument("model")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", help="Device for evaluation.")
@click.option("--output", type=click.Path(path_type=Path), help="Save results to file.")
@click.option("--prompt", type=str, help="Test prompt for generation.")
def evaluate(model: str, device: str, output: Path | None, prompt: str | None) -> None:
    """Run comprehensive evaluation test suite on a trained model."""
    evaluate_model_impl(model, device, output, prompt)


@cli.command(help="Generate text from a trained model.")
@click.argument("model")
@click.option("--prompt", type=str, help="Generation prompt (omit for interactive mode).")
@click.option("--interactive", is_flag=True, help="Force interactive mode.")
@click.option("--temp", type=float, default=1.0, show_default=True, help="Sampling temperature.")
@click.option("--max-tokens", type=int, default=50, show_default=True, help="Maximum tokens to generate.")
@click.option("--top-k", type=int, help="Top-k sampling.")
@click.option("--top-p", type=float, help="Nucleus sampling.")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto", show_default=True, help="Device for generation.")
def generate(model: str, prompt: str | None, interactive: bool, temp: float, max_tokens: int, top_k: int | None, top_p: float | None, device: str) -> None:
    """Interactive text generation from a trained model."""
    generate_cmd(model, prompt, interactive, temp, max_tokens, top_k, top_p, device)


@cli.group(help="List available datasets or models.")
def list() -> None:
    """List resources (datasets, models)."""


@list.command(name="datasets", help="Show available datasets and their status.")
def list_datasets() -> None:
    """List all available datasets."""
    list_datasets_cmd()


@list.command(name="models", help="Show training runs and checkpoints.")
def list_models() -> None:
    """List all trained models."""
    list_models_cmd()


@cli.command(help="Display detailed information about a dataset or model.")
@click.argument("name")
def info(name: str) -> None:
    """Show detailed information about a dataset or model."""
    info_cmd(name)


def main() -> None:
    """Allow setuptools entrypoints to reference a stable callable."""
    cli(standalone_mode=True)


if __name__ == "__main__":
    main()
