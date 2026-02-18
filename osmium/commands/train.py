"""Train command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from models.registry import is_registered as is_model_registered, list_models
from osmium.train.config import TrainConfig
from osmium.train.runner import run_training
from osmium.utils import PathResolver
from osmium.utils.config import load_yaml_config, save_yaml_config


def train_model(
    architecture: str,
    data: str,
    name: str | None,
    config: Path | None,
    cli_params: dict[str, Any],
) -> None:
    """train a model using preprocessed data

    config precedence (lowest to highest):
        1. hardcoded defaults (defined in this function)
        2. YAML config file (--config flag)
        3. CLI parameters (explicit flags like --epochs, --batch-size)

    defaults follow modern LLM training practices:
        - single-epoch training through full dataset (chinchilla optimal)
        - no early stopping (checkpoint-based selection post-training)
        - eval every 500 steps (reduced noise, matches SOTA practices)

    args:
        architecture: model architecture from registry
        data: dataset name (uses data/processed/<dataset>/)
        name: experiment name
        config: path to YAML config file
        cli_params: CLI parameters from Click
    """
    # validate architecture
    if not is_model_registered(architecture):
        raise click.ClickException(
            f"Unknown architecture: {architecture}\n"
            f"Available: {', '.join(list_models())}"
        )

    resolver = PathResolver()
    data_dir = resolver.processed_dir(data)

    # validate dataset is preprocessed
    if not data_dir.exists():
        raise click.ClickException(
            f"Dataset not preprocessed: {data}\n"
            f"Run: osmium preprocess {data}"
        )

    # validate train.bin and val.bin exist
    if not (data_dir / "train.bin").exists() or not (data_dir / "val.bin").exists():
        raise click.ClickException(
            f"Dataset incomplete: missing train.bin or val.bin in {data_dir}\n"
            f"Run: osmium preprocess {data}"
        )

    # determine experiment name
    if config and not name:
        # auto-name from config filename (without .yaml extension)
        name = config.stem
    elif not name and not config:
        raise click.ClickException(
            "Either --name or --config must be provided.\n"
            "Use --name for quick experiments or --config for config-driven runs."
        )

    # create run directory
    run_dir = resolver.run_dir(name)
    if run_dir.exists():
        raise click.ClickException(
            f"Run '{name}' already exists at {run_dir}\n"
            f"Choose a different --name"
        )

    # hardcoded defaults based on modern LLM training research (GPT-3, LLaMA, Chinchilla)
    # config precedence: defaults < YAML config < CLI params
    defaults = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.0004,
        "patience": None,  # disable early stopping (modern approach: checkpoint-based selection)
        "eval_freq": 500,  # evaluate every 500 steps (reduced noise, matches SOTA)
        "eval_iter": 50,
        "gradient_accumulation_steps": 4,
        "device": "auto",
        "num_workers": 0,
        "max_grad_norm": 1.0,
        "max_tokens": None,  # use full dataset (chinchilla optimal: ~20 tokens per parameter)
        "data_fraction": None,
        "mixed_precision": None,
        "compile_model": None,
        "warmup_steps": None,
        "min_lr": None,
    }
    config_values = defaults.copy()

    # load config from file if provided (overrides defaults)
    if config:
        click.echo(f"Loading config from {config}")
        yaml_config = load_yaml_config(config)
        config_values.update(yaml_config)

    # merge CLI parameters (CLI takes precedence over everything)
    # only override if CLI param differs from the hardcoded default
    for key, value in cli_params.items():
        if key in config_values and value != defaults.get(key):
            config_values[key] = value

    # add required fields
    config_values["data_dir"] = data_dir
    config_values["model"] = architecture
    config_values["run_dir"] = run_dir

    # create TrainConfig
    train_config = TrainConfig.from_click(config_values, warn=lambda msg: click.secho(f"Warning: {msg}", fg="yellow"))

    # setup run directory
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config snapshot
    config_snapshot = {
        "architecture": architecture,
        "dataset": data,
        "name": name,
        **{k: str(v) if isinstance(v, Path) else v for k, v in config_values.items() if k != "data_dir"},
    }
    save_yaml_config(config_snapshot, run_dir / "config.yaml")

    click.echo(f"\nTraining run: {name}")
    click.echo(f"Architecture: {architecture}")
    click.echo(f"Dataset: {data}")
    click.echo(f"Run directory: {run_dir}")
    click.echo()

    # run training
    run_training(train_config)

    click.secho(f"\n✓ Training complete for '{name}'", fg="green")
    click.echo(f"\nNext steps:")
    click.echo(f"  osmium evaluate {name}")


__all__ = ["train_model"]
