"""Info command implementation."""

from __future__ import annotations

import json
from pathlib import Path

import click

from loaders.cleaning.registry import DATASET_REGISTRY, get_source
from kilonova.utils import PathResolver


def info_cmd(name: str) -> None:
    """display detailed information about a dataset or model

    auto-detects whether name refers to a dataset or training run
    """
    resolver = PathResolver()

    # check if it's a training run first
    run_dir = resolver.run_dir(name)
    if run_dir.exists():
        _display_model_info(name, run_dir)
        return

    # check if it's a dataset
    raw_dir = resolver.raw_dir(name)
    if raw_dir.exists() or name in DATASET_REGISTRY:
        _display_dataset_info(name, resolver)
        return

    # not found
    raise click.ClickException(
        f"'{name}' not found as dataset or training run.\n"
        f"Use 'kilonova list datasets' or 'kilonova list models' to see available options."
    )


def _display_dataset_info(dataset: str, resolver: PathResolver) -> None:
    """display detailed information about a dataset"""
    raw_dir = resolver.raw_dir(dataset)
    clean_dir = resolver.clean_dir(dataset)
    processed_dir = resolver.processed_dir(dataset)

    # dataset header
    click.echo(f"\nDataset: {dataset}")

    # status indicators
    downloaded = raw_dir.exists() and any(raw_dir.iterdir())
    cleaned = clean_dir.exists() and any(clean_dir.iterdir())
    processed = processed_dir.exists() and (processed_dir / "train.bin").exists()

    status_parts = []
    if processed:
        status_parts.append("✓ processed")
    elif cleaned:
        status_parts.append("✓ cleaned")
    elif downloaded:
        status_parts.append("✓ downloaded")
    else:
        status_parts.append("✗ not downloaded")

    click.echo(f"Status: {' '.join(status_parts)}")

    # source (for registered datasets)
    source = get_source(dataset)
    if source:
        click.echo(f"Source: {source}")

    click.echo()

    # directory sizes
    if downloaded:
        raw_size = _get_dir_size(raw_dir)
        click.echo(f"Raw data:      {raw_dir}  {_format_size(raw_size)}")
    else:
        click.echo(f"Raw data:      {raw_dir}  (not downloaded)")

    if cleaned:
        clean_size = _get_dir_size(clean_dir)
        click.echo(f"Clean data:    {clean_dir}  {_format_size(clean_size)}")
    else:
        click.echo(f"Clean data:    {clean_dir}  (not cleaned)")

    if processed:
        click.echo(f"Processed:     {processed_dir}")
    else:
        click.echo(f"Processed:     {processed_dir}  (not processed)")

    # metadata from processed data
    if processed:
        metadata_path = processed_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open() as f:
                    metadata = json.load(f)

                click.echo()
                click.echo(f"Tokenizer:     {metadata.get('tokenizer_name', 'unknown')}")
                click.echo(f"Vocabulary:    {metadata.get('vocab_size', '?'):,} tokens")

                train_tokens = metadata.get('train_tokens', 0)
                val_tokens = metadata.get('val_tokens', 0)
                total_tokens = train_tokens + val_tokens

                click.echo(f"Training:      {train_tokens:,} tokens")
                click.echo(f"Validation:    {val_tokens:,} tokens")

                if total_tokens > 0:
                    train_ratio = train_tokens / total_tokens
                    val_ratio = val_tokens / total_tokens
                    click.echo(f"Split ratio:   {train_ratio:.2f} / {val_ratio:.2f}")
            except (json.JSONDecodeError, KeyError) as e:
                click.echo(f"\nWarning: Could not read metadata: {e}")


def _display_model_info(run_name: str, run_dir: Path) -> None:
    """display detailed information about a training run"""
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"
    checkpoints_dir = run_dir / "checkpoints"
    arch_path = run_dir / "model_architecture.json"

    click.echo(f"\nRun: {run_name}")

    # determine status
    if (checkpoints_dir / "best.pth").exists():
        status = "completed"
    elif checkpoints_dir.exists() and any(checkpoints_dir.glob("*.pth")):
        status = "in progress"
    else:
        status = "initialized"

    click.echo(f"Status: {status}")

    # creation time from directory
    if run_dir.exists():
        created_time = run_dir.stat().st_ctime
        from datetime import datetime
        created_dt = datetime.fromtimestamp(created_time)
        click.echo(f"Created: {created_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    click.echo()

    # read config for details
    architecture = "unknown"
    dataset = "unknown"
    config_file = "unknown"
    hyperparams = {}

    if config_path.exists():
        try:
            import yaml
            with config_path.open() as f:
                config = yaml.safe_load(f)

            architecture = config.get("architecture", "unknown")
            dataset = config.get("dataset", "unknown")
            config_file = config.get("config_file", "unknown")

            # extract hyperparameters
            hyperparams = {
                k: v for k, v in config.items()
                if k not in ["architecture", "dataset", "config_file", "device"]
            }
        except (yaml.YAMLError, ImportError) as e:
            click.echo(f"Warning: Could not read config: {e}")

    click.echo(f"Architecture:  {architecture}")
    click.echo(f"Dataset:       {dataset}")
    if config_file and config_file != "unknown":
        click.echo(f"Config:        {config_file}")

    # model parameters from architecture file
    if arch_path.exists():
        try:
            with arch_path.open() as f:
                arch_config = json.load(f)

            # calculate parameters (rough estimate)
            # this would need actual model instantiation for exact count
            # for now just show the config
            click.echo(f"\nModel configuration:")
            for key, value in arch_config.items():
                click.echo(f"  {key}: {value}")
        except (json.JSONDecodeError, KeyError):
            pass

    # checkpoint information
    if checkpoints_dir.exists():
        checkpoints = sorted(checkpoints_dir.glob("epoch-*.pth"))
        best_checkpoint = checkpoints_dir / "best.pth"

        num_epochs = len(checkpoints)
        click.echo(f"\nTraining epochs:  {num_epochs}")

        # try to extract best validation loss
        if best_checkpoint.exists():
            try:
                import torch
                checkpoint = torch.load(best_checkpoint, map_location="cpu")

                best_loss = checkpoint.get("val_loss", "?")
                best_epoch = checkpoint.get("epoch", "?")

                if best_loss != "?":
                    click.echo(f"Best val loss:    {best_loss:.4f} (epoch {best_epoch})")

                # show last checkpoint
                if checkpoints:
                    last_checkpoint = checkpoints[-1]
                    click.echo(f"Last checkpoint:  {last_checkpoint}")
            except (ImportError, Exception) as e:
                click.echo(f"Warning: Could not read checkpoint: {e}")

    # hyperparameters
    if hyperparams:
        click.echo(f"\nHyperparameters:")
        for key, value in sorted(hyperparams.items()):
            click.echo(f"  {key}: {value}")


def _get_dir_size(path: Path) -> int:
    """calculate total size of all files in directory"""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        return 0
    return total


def _format_size(size_bytes: int) -> str:
    """format byte size into human-readable format"""
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


__all__ = ["info_cmd"]
