"""Architecture and model registry system."""

from __future__ import annotations

from typing import Any

ARCHITECTURE_REGISTRY: dict[str, Any] = {}
MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_architecture(name: str):
    """decorator to register architecture classes.

    the class may define a MODELS dict mapping model names to config overrides.
    shared defaults go in DEFAULTS. both are optional.

    usage:
        @register_architecture("gpt2")
        class GPT2(nn.Module):
            DEFAULTS = {"vocab_size": 50257, "context_length": 1024, ...}
            MODELS = {
                "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
                ...
            }
    """
    def decorator(cls):
        if name in ARCHITECTURE_REGISTRY:
            import warnings
            warnings.warn(
                f"Architecture '{name}' already registered. "
                f"Overwriting {ARCHITECTURE_REGISTRY[name].__name__} with {cls.__name__}",
                UserWarning,
                stacklevel=2
            )
        ARCHITECTURE_REGISTRY[name] = cls

        # collect model configs from class
        defaults = getattr(cls, "DEFAULTS", {})
        models = getattr(cls, "MODELS", {})
        for model_name, overrides in models.items():
            MODEL_REGISTRY[model_name] = {
                "architecture": name,
                **defaults,
                **overrides,
            }

        return cls
    return decorator


def get_architecture_class(name: str):
    """get architecture class by name."""
    if name not in ARCHITECTURE_REGISTRY:
        available = ", ".join(ARCHITECTURE_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture: {name}\n"
            f"Available: {available}"
        )
    return ARCHITECTURE_REGISTRY[name]


def get_model_config(model_name: str) -> dict[str, Any]:
    """get complete model config by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name].copy()


def is_registered(model_name: str) -> bool:
    """check if a model config is registered."""
    return model_name in MODEL_REGISTRY


def list_models() -> list[str]:
    """list all registered model configs."""
    return list(MODEL_REGISTRY.keys())


def list_architectures() -> list[str]:
    """list all registered architectures."""
    return list(ARCHITECTURE_REGISTRY.keys())


__all__ = [
    "ARCHITECTURE_REGISTRY",
    "MODEL_REGISTRY",
    "register_architecture",
    "get_architecture_class",
    "get_model_config",
    "is_registered",
    "list_models",
    "list_architectures",
]
