"""Model architecture registry."""

from models.architectures import (
    register_architecture,
    get_architecture_class,
    get_model_config,
    is_registered,
    list_models,
    list_architectures,
)

# import architecture modules to trigger registration via @register_architecture
import models.gpt2  # noqa: F401
import models.gpt2_from_scratch  # noqa: F401

__all__ = [
    "register_architecture",
    "get_architecture_class",
    "get_model_config",
    "is_registered",
    "list_models",
    "list_architectures",
]
