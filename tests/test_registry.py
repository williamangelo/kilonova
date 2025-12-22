"""Tests for dataset and model registries."""

import pytest

from loaders.cleaning.registry import get_cleaner, get_source, is_registered as is_dataset_registered
from models.registry import get_model_config, is_registered as is_model_registered, list_architectures


class TestDatasetRegistry:
    """tests for dataset registry behavior"""

    def test_is_registered_returns_true_for_known_datasets(self):
        # these are the built-in datasets
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
        # cleaners are optional - None means use generic cleaner
        cleaner = get_cleaner("gutenberg")
        # should be None or a class, but not raise
        assert cleaner is None or callable(cleaner)

    def test_get_cleaner_returns_none_for_unknown_dataset(self):
        assert get_cleaner("nonexistent") is None


class TestModelRegistry:
    """tests for model registry behavior"""

    def test_get_model_config_returns_complete_config(self):
        """get_model_config should return all config fields including architecture."""
        config = get_model_config("gpt2-small")

        # architecture field should be present
        assert "architecture" in config
        assert config["architecture"] == "gpt"

        # existing fields
        assert "vocab_size" in config
        assert "context_length" in config
        assert "emb_dim" in config
        assert "n_layers" in config
        assert "n_heads" in config
        assert "drop_rate" in config
        assert "qkv_bias" in config

    def test_get_model_config_returns_copy_not_reference(self):
        """modifying returned config should not affect registry"""
        config1 = get_model_config("gpt2-small")
        original_emb_dim = config1["emb_dim"]

        config1["emb_dim"] = 99999

        config2 = get_model_config("gpt2-small")
        assert config2["emb_dim"] == original_emb_dim

    def test_get_model_config_raises_for_unknown_architecture(self):
        with pytest.raises(ValueError) as exc_info:
            get_model_config("nonexistent")

        assert "Unknown architecture" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_get_model_config_error_lists_available_architectures(self):
        with pytest.raises(ValueError) as exc_info:
            get_model_config("nonexistent")

        error_msg = str(exc_info.value)
        # should list available options
        assert "gpt2-small" in error_msg or "Available" in error_msg

    def test_is_registered_returns_true_for_all_listed_architectures(self):
        for arch in list_architectures():
            assert is_model_registered(arch)

    def test_is_registered_returns_false_for_unknown(self):
        assert not is_model_registered("nonexistent")
        assert not is_model_registered("")

    def test_list_architectures_returns_non_empty_list(self):
        archs = list_architectures()
        assert len(archs) > 0
        assert all(isinstance(a, str) for a in archs)

    def test_model_configs_have_valid_dimensions(self):
        """sanity check that model dimensions are internally consistent"""
        for arch in list_architectures():
            config = get_model_config(arch)
            # emb_dim must be divisible by n_heads
            assert config["emb_dim"] % config["n_heads"] == 0, f"{arch} has invalid dimensions"
