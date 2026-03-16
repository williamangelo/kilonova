"""Tests for architecture and model registry system."""

from __future__ import annotations

import pytest

from models.architectures import (
    ARCHITECTURE_REGISTRY,
    MODEL_REGISTRY,
    register_architecture,
    get_architecture_class,
    get_model_config,
    is_registered,
    list_architectures,
    list_models,
)


class TestRegisterArchitecture:
    """test architecture registration decorator."""

    def test_registers_class(self):
        @register_architecture("test_arch")
        class TestModel:
            pass

        assert "test_arch" in ARCHITECTURE_REGISTRY
        assert ARCHITECTURE_REGISTRY["test_arch"] is TestModel

    def test_returns_original_class(self):
        @register_architecture("test_arch2")
        class TestModel:
            pass

        assert TestModel.__name__ == "TestModel"

    def test_warns_on_duplicate(self):
        @register_architecture("duplicate_test")
        class FirstModel:
            pass

        with pytest.warns(UserWarning) as warning_list:
            @register_architecture("duplicate_test")
            class SecondModel:
                pass

            warning_msg = str(warning_list[0].message)
            assert "duplicate_test" in warning_msg
            assert "FirstModel" in warning_msg
            assert "SecondModel" in warning_msg

        assert ARCHITECTURE_REGISTRY["duplicate_test"] is SecondModel

    def test_collects_models_from_class(self):
        @register_architecture("test_with_models")
        class TestModel:
            DEFAULTS = {"vocab_size": 100, "drop_rate": 0.1}
            MODELS = {
                "test-small": {"emb_dim": 64, "n_layers": 2},
                "test-large": {"emb_dim": 128, "n_layers": 4},
            }

        assert "test-small" in MODEL_REGISTRY
        assert "test-large" in MODEL_REGISTRY

        config = MODEL_REGISTRY["test-small"]
        assert config["architecture"] == "test_with_models"
        assert config["vocab_size"] == 100
        assert config["drop_rate"] == 0.1
        assert config["emb_dim"] == 64
        assert config["n_layers"] == 2

    def test_model_overrides_take_precedence_over_defaults(self):
        @register_architecture("test_override")
        class TestModel:
            DEFAULTS = {"vocab_size": 100, "emb_dim": 32}
            MODELS = {"test-override": {"emb_dim": 64}}

        config = MODEL_REGISTRY["test-override"]
        assert config["emb_dim"] == 64
        assert config["vocab_size"] == 100

    def test_works_without_models_or_defaults(self):
        @register_architecture("bare_arch")
        class BareModel:
            pass

        assert "bare_arch" in ARCHITECTURE_REGISTRY
        assert get_architecture_class("bare_arch") is BareModel


class TestGetArchitectureClass:
    """test architecture class lookup."""

    def test_returns_registered_class(self):
        @register_architecture("lookup_test")
        class LookupModel:
            pass

        assert get_architecture_class("lookup_test") is LookupModel

    def test_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown architecture: nonexistent"):
            get_architecture_class("nonexistent")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="Available:"):
            get_architecture_class("unknown")


class TestModelConfig:
    """test model config lookup."""

    def test_returns_complete_config(self):
        config = get_model_config("gpt2-small")

        assert config["architecture"] == "gpt2"
        assert "vocab_size" in config
        assert "context_length" in config
        assert "emb_dim" in config
        assert "n_layers" in config
        assert "n_heads" in config
        assert "drop_rate" in config
        assert "qkv_bias" in config

    def test_returns_copy(self):
        config1 = get_model_config("gpt2-small")
        original = config1["emb_dim"]
        config1["emb_dim"] = 99999

        config2 = get_model_config("gpt2-small")
        assert config2["emb_dim"] == original

    def test_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_config("nonexistent")

    def test_all_gpt2_models_have_valid_dimensions(self):
        for model in list_models():
            config = get_model_config(model)
            if "emb_dim" in config and "n_heads" in config:
                assert config["emb_dim"] % config["n_heads"] == 0, f"{model} has invalid dimensions"

    def test_all_models_have_architecture(self):
        for model in list_models():
            config = get_model_config(model)
            assert "architecture" in config
            assert config["architecture"] in ARCHITECTURE_REGISTRY


class TestGPT2Registration:
    """test that built-in models are properly registered via auto-discovery."""

    def test_gpt2_architecture_registered(self):
        assert "gpt2" in list_architectures()

    def test_gpt2_from_scratch_architecture_registered(self):
        assert "gpt2_from_scratch" in list_architectures()

    def test_gpt2_models_registered(self):
        models = list_models()
        assert "gpt2-small" in models
        assert "gpt2-medium" in models
        assert "gpt2-large" in models
        assert "gpt2-xlarge" in models

    def test_gpt2_from_scratch_models_registered(self):
        models = list_models()
        assert "gpt2-from-scratch-small" in models
        assert "gpt2-from-scratch-medium" in models
        assert "gpt2-from-scratch-large" in models
        assert "gpt2-from-scratch-xlarge" in models

    def test_gpt2_and_from_scratch_share_same_config_shape(self):
        gpt2 = get_model_config("gpt2-small")
        scratch = get_model_config("gpt2-from-scratch-small")

        # same hyperparams, different architecture
        assert gpt2["architecture"] != scratch["architecture"]
        assert gpt2["emb_dim"] == scratch["emb_dim"]
        assert gpt2["n_layers"] == scratch["n_layers"]
        assert gpt2["n_heads"] == scratch["n_heads"]
        assert gpt2["vocab_size"] == scratch["vocab_size"]


class TestListFunctions:
    """test list_models and list_architectures."""

    def test_list_models_returns_non_empty(self):
        models = list_models()
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_list_architectures_returns_non_empty(self):
        archs = list_architectures()
        assert len(archs) > 0
        assert all(isinstance(a, str) for a in archs)

    def test_is_registered_matches_list(self):
        for model in list_models():
            assert is_registered(model)

    def test_is_registered_false_for_unknown(self):
        assert not is_registered("nonexistent")
        assert not is_registered("")


class TestModelsModule:
    """test models package exports."""

    def test_exports_architecture_functions(self):
        import models

        assert hasattr(models, "register_architecture")
        assert hasattr(models, "get_architecture_class")
        assert hasattr(models, "list_architectures")

    def test_exports_model_functions(self):
        import models

        assert hasattr(models, "get_model_config")
        assert hasattr(models, "is_registered")
        assert hasattr(models, "list_models")

    def test_exports_loader(self):
        import models

        assert hasattr(models, "create_model_from_config")

    def test_auto_discovers_architectures(self):
        import models

        assert "gpt2" in models.list_architectures()
        assert "gpt2_from_scratch" in models.list_architectures()
