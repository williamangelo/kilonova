"""Legacy shim that delegates to the Click-based Kilonova CLI."""

from __future__ import annotations

import sys
from typing import Sequence

import click

from kilonova.cli import cli
from kilonova.train.config import TrainConfig  # Re-export for compatibility
from kilonova.train.runner import run_training  # Re-export for compatibility

__all__ = ["main", "TrainConfig", "run_training"]


def main(argv: Sequence[str] | None = None) -> int:
    """Invoke the Kilonova CLI as if `python train.py` was called directly."""
    args = list(argv) if argv is not None else sys.argv[1:]
    try:
        cli.main(args=["train", *args], prog_name="train.py", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
