"""Dataset registry mapping dataset names to sources and cleaners."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from loaders.cleaning.base import BaseCleaner


class DatasetEntry(TypedDict):
    """dataset registry entry"""

    source: str
    cleaner: type[BaseCleaner] | None


# placeholder for cleaner classes - will be implemented when converting scripts
DATASET_REGISTRY: dict[str, DatasetEntry] = {
    "gutenberg": {
        "source": "huggingface:pg19",
        "cleaner": None,  # TODO: implement GutenbergCleaner
    },
    "verdict": {
        "source": "huggingface:the_verdict",
        "cleaner": None,  # TODO: implement VerdictCleaner
    },
}


def get_cleaner(dataset: str) -> type[BaseCleaner] | None:
    """get the cleaner class for a dataset, if registered"""
    if dataset not in DATASET_REGISTRY:
        return None
    return DATASET_REGISTRY[dataset]["cleaner"]


def get_source(dataset: str) -> str | None:
    """get the huggingface source for a dataset, if registered"""
    if dataset not in DATASET_REGISTRY:
        return None
    return DATASET_REGISTRY[dataset]["source"]


def is_registered(dataset: str) -> bool:
    """check if a dataset is registered"""
    return dataset in DATASET_REGISTRY


__all__ = ["DATASET_REGISTRY", "get_cleaner", "get_source", "is_registered"]
