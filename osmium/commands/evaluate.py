"""Evaluate command implementation."""

from __future__ import annotations

from pathlib import Path

import click

from osmium.utils import PathResolver


def evaluate_model(model: str, device: str, output: Path | None, prompt: str | None) -> None:
    """run comprehensive evaluation test suite on a trained model

    args:
        model: model name (run name) or path to checkpoint
        device: device for evaluation (auto, cuda, cpu)
        output: optional output file for results
        prompt: optional test prompt for generation
    """
    resolver = PathResolver()

    # determine if model is a run name or checkpoint path
    model_path = Path(model)
    if not model_path.exists():
        # assume it's a run name
        run_dir = resolver.run_dir(model)
        if not run_dir.exists():
            raise click.ClickException(
                f"Run '{model}' not found at {run_dir}\n"
                f"Available runs in data/runs/"
            )

        # use best.pth checkpoint if available, otherwise latest
        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            raise click.ClickException(
                f"No checkpoints found for run '{model}' at {checkpoint_dir}"
            )

        best_checkpoint = checkpoint_dir / "best.pth"
        if best_checkpoint.exists():
            model_path = best_checkpoint
        else:
            # find latest checkpoint
            checkpoints = sorted(checkpoint_dir.glob("epoch-*.pth"))
            if not checkpoints:
                raise click.ClickException(
                    f"No checkpoints found in {checkpoint_dir}"
                )
            model_path = checkpoints[-1]

        click.echo(f"Using checkpoint: {model_path}")

    # validate checkpoint exists
    if not model_path.exists():
        raise click.ClickException(f"Checkpoint not found: {model_path}")

    # TODO: implement actual evaluation logic
    # for now, just show placeholder
    click.echo(f"\nEvaluating model: {model_path}")
    click.echo(f"Device: {device}")
    if prompt:
        click.echo(f"Test prompt: {prompt}")
    if output:
        click.echo(f"Output file: {output}")

    click.echo("\n[Evaluation logic to be implemented]")
    click.echo("This will run comprehensive tests including:")
    click.echo("  - Token length variation")
    click.echo("  - Temperature comparison")
    click.echo("  - Sampling strategies (top-k, top-p)")
    click.echo("  - Multiple evaluation prompts")

    click.secho("\n✓ Evaluation complete", fg="green")


__all__ = ["evaluate_model"]
