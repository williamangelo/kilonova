"""Model architecture registry."""

from models.architectures import (
    register_architecture,
    get_architecture_class,
    get_model_config,
    is_registered,
    list_models,
    list_architectures,
)

__all__ = [
    "register_architecture",
    "get_architecture_class",
    "get_model_config",
    "is_registered",
    "list_models",
    "list_architectures",
]
