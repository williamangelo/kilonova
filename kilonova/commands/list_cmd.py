"""List command implementations."""

from __future__ import annotations

import json
from pathlib import Path

import click

from loaders.cleaning.registry import DATASET_REGISTRY
from osmium.utils import PathResolver


def list_datasets_cmd() -> None:
    """list all available datasets and their status"""
    resolver = PathResolver()

    click.echo("\nAvailable datasets:")

    # track counts
    built_in_count = len(DATASET_REGISTRY)
    custom_count = 0

    # check for datasets in data/raw that aren't registered
    raw_base = resolver.base_dir / "raw"
    if raw_base.exists():
        for dataset_dir in sorted(raw_base.iterdir()):
            if dataset_dir.is_dir() and dataset_dir.name not in DATASET_REGISTRY:
                custom_count += 1

    # show registered datasets
    for dataset_name in sorted(DATASET_REGISTRY.keys()):
        _display_dataset_status(dataset_name, resolver)

    # show custom (unregistered) datasets
    if raw_base.exists():
        for dataset_dir in sorted(raw_base.iterdir()):
            if dataset_dir.is_dir() and dataset_dir.name not in DATASET_REGISTRY:
                _display_dataset_status(dataset_dir.name, resolver, is_custom=True)

    # summary
    click.echo(f"\nBuilt-in datasets: {built_in_count}")
    click.echo(f"Custom datasets: {custom_count}")


def _display_dataset_status(dataset: str, resolver: PathResolver, is_custom: bool = False) -> None:
    """display status line for a single dataset"""
    raw_dir = resolver.raw_dir(dataset)
    clean_dir = resolver.clean_dir(dataset)
    processed_dir = resolver.processed_dir(dataset)

    # check status of each stage
    downloaded = raw_dir.exists() and any(raw_dir.iterdir())
    cleaned = clean_dir.exists() and any(clean_dir.iterdir())
    processed = processed_dir.exists() and (processed_dir / "train.bin").exists()

    # format status indicators
    download_status = "✓ downloaded" if downloaded else "✗ not downloaded"
    clean_status = "✓ cleaned" if cleaned else "✗ not cleaned"
    process_status = "✓ processed" if processed else "✗ not processed"

    # build output line
    custom_marker = " (custom)" if is_custom else ""
    click.echo(f"  {dataset:20} {download_status:20} {clean_status:20} {process_status}{custom_marker}")


def list_models_cmd() -> None:
    """list all trained models and their checkpoints"""
    resolver = PathResolver()
    runs_dir = resolver.base_dir / "runs"

    if not runs_dir.exists():
        click.echo("\nNo training runs found.")
        click.echo(f"Run directory: {runs_dir}")
        return

    click.echo("\nTraining runs:")

    # find all run directories
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    if not run_dirs:
        click.echo("  (none)")
        click.echo(f"\nTotal runs: 0")
        return

    # display each run
    for run_dir in run_dirs:
        _display_model_status(run_dir)

    click.echo(f"\nTotal runs: {len(run_dirs)}")


def _display_model_status(run_dir: Path) -> None:
    """display status line for a single training run"""
    run_name = run_dir.name

    # load config to get architecture and dataset
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"
    checkpoints_dir = run_dir / "checkpoints"

    # extract info from config/metadata
    architecture = "unknown"
    dataset = "unknown"
    epochs_trained = "?"
    best_loss = "?"

    # try to read metadata first for dataset info
    if metadata_path.exists():
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)
                dataset = metadata.get("dataset", "unknown")
        except (json.JSONDecodeError, KeyError):
            pass

    # try to read config for architecture and training info
    if config_path.exists():
        try:
            import yaml
            with config_path.open() as f:
                config = yaml.safe_load(f)
                architecture = config.get("architecture", "unknown")
                dataset = config.get("dataset", dataset)  # override if available
                epochs_trained = config.get("epochs", "?")
        except (yaml.YAMLError, KeyError, ImportError):
            pass

    # count checkpoints and find best
    num_checkpoints = 0
    best_checkpoint = None
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pth"))
        num_checkpoints = len(checkpoints)
        best_checkpoint = checkpoints_dir / "best.pth"
        if best_checkpoint.exists():
            # try to extract loss from checkpoint (requires torch)
            try:
                import torch
                checkpoint = torch.load(best_checkpoint, map_location="cpu")
                if "val_loss" in checkpoint:
                    best_loss = f"{checkpoint['val_loss']:.2f}"
            except (ImportError, Exception):
                best_loss = "?"

    # format epoch info
    if num_checkpoints > 0:
        epoch_info = f"{num_checkpoints} checkpoints"
    else:
        epoch_info = "no checkpoints"

    # build output line
    click.echo(f"  {run_name:25} {architecture:15} {dataset:15} {epoch_info:20} best: {best_loss}")


__all__ = ["list_datasets_cmd", "list_models_cmd"]
