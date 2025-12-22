"""Configuration file handling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """load configuration from a YAML file

    args:
        config_path: path to YAML config file

    returns:
        dictionary of configuration values
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}


def save_yaml_config(config: dict[str, Any], output_path: Path) -> None:
    """save configuration to a YAML file

    args:
        config: dictionary of configuration values
        output_path: path to save YAML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


__all__ = ["load_yaml_config", "save_yaml_config"]
