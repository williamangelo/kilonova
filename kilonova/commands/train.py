"""Train command implementation."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from models.architectures import is_registered as is_model_registered, list_models
from kilonova.train.config import TrainConfig
from kilonova.train.runner import run_training


def _generate_run_id() -> str:
    """generate a unique run id: run-YYYYMMDD-<7char_uuid>"""
    date = datetime.now().strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:7]
    return f"run-{date}-{short_id}"


def _write_metadata(run_dir: Path, run_id: str, architecture: str, dataset: str,
                     notes: str | None, hyperparams: dict) -> None:
    """write initial metadata.json for a run"""
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "architecture": architecture,
        "dataset": dataset,
        "hyperparams": hyperparams,
        "best_val_loss": None,
        "total_tokens_seen": None,
        "duration_seconds": None,
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _update_metadata(run_dir: Path, best_val_loss: float | None,
                      total_tokens_seen: int | None, duration_seconds: float | None) -> None:
    """update metadata.json with training results"""
    meta_path = run_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    metadata["best_val_loss"] = best_val_loss
    metadata["total_tokens_seen"] = total_tokens_seen
    metadata["duration_seconds"] = duration_seconds

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def train_model(
    architecture: str,
    data: str,
    notes: str | None,
    cli_params: dict[str, Any],
) -> None:
    """train a model using preprocessed data"""
    # validate architecture
    if not is_model_registered(architecture):
        raise click.ClickException(
            f"Unknown architecture: {architecture}\n"
            f"Available: {', '.join(list_models())}"
        )

    data_dir = Path("data/processed") / data

    # validate dataset is preprocessed
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

    # generate run id and create directory
    run_id = _generate_run_id()
    run_dir = Path("data/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # build config from defaults + CLI params
    defaults = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.0004,
        "patience": None,
        "eval_freq": 100,
        "eval_iter": 50,
        "gradient_accumulation_steps": 4,
        "device": "auto",
        "num_workers": 0,
        "max_grad_norm": 1.0,
        "max_tokens": None,
        "data_fraction": None,
        "mixed_precision": None,
        "compile_model": None,
        "warmup_steps": None,
        "min_lr": None,
    }
    config_values = defaults.copy()

    # merge CLI parameters (only override if value differs from default)
    for key, value in cli_params.items():
        if key in config_values and value != defaults.get(key):
            config_values[key] = value

    # add required fields
    config_values["data_dir"] = data_dir
    config_values["model"] = architecture
    config_values["run_dir"] = run_dir

    # write metadata
    hyperparams = {k: v for k, v in config_values.items()
                   if k not in ("data_dir", "model", "run_dir")}
    _write_metadata(run_dir, run_id, architecture, data, notes, hyperparams)

    # create TrainConfig
    train_config = TrainConfig.from_click(
        config_values,
        warn=lambda msg: click.secho(f"Warning: {msg}", fg="yellow"),
    )

    click.echo(f"\nTraining run: {run_id}")
    click.echo(f"Architecture: {architecture}")
    click.echo(f"Dataset: {data}")
    click.echo(f"Run directory: {run_dir}")
    if notes:
        click.echo(f"Notes: {notes}")
    click.echo()

    # run training and capture results
    start_time = time.time()
    train_losses, val_losses, tokens_seen, lrs = run_training(train_config)
    duration = time.time() - start_time

    # update metadata with results
    best_val = min(val_losses) if val_losses else None
    total_tokens = tokens_seen[-1] if tokens_seen else None
    _update_metadata(run_dir, best_val, total_tokens, duration)

    click.secho(f"\n✓ Training complete: {run_id}", fg="green")
    click.echo(f"\nNext steps:")
    click.echo(f"  kilonova evaluate {run_id}")


__all__ = ["train_model"]
