"""Tests for dataset registry."""

import pytest

from loaders.cleaning.registry import get_cleaner, get_source, is_registered as is_dataset_registered


class TestDatasetRegistry:
    """tests for dataset registry behavior"""

    def test_is_registered_returns_true_for_known_datasets(self):
        assert is_dataset_registered("gutenberg")
        assert is_dataset_registered("verdict")

    def test_is_registered_returns_false_for_unknown_datasets(self):
        assert not is_dataset_registered("nonexistent")
        assert not is_dataset_registered("")
        assert not is_dataset_registered("GUTENBERG")  # case sensitive

    def test_get_source_returns_huggingface_path_for_known_dataset(self):
        source = get_source("gutenberg")
        assert source is not None
        assert source.startswith("huggingface:")

    def test_get_source_returns_none_for_unknown_dataset(self):
        assert get_source("nonexistent") is None

    def test_get_cleaner_returns_none_when_not_implemented(self):
        cleaner = get_cleaner("gutenberg")
        assert cleaner is None or callable(cleaner)

    def test_get_cleaner_returns_none_for_unknown_dataset(self):
        assert get_cleaner("nonexistent") is None
