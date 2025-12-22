"""Tests for shared model loading utilities."""

from __future__ import annotations

import pytest

from models.loader import create_model_from_config
from models.architectures import register_architecture


class DummyModel:
    """dummy model for testing."""
    def __init__(self, config):
        self.config = config


class TestCreateModelFromConfig:
    """test create_model_from_config function."""

    def test_creates_model_with_registered_architecture(self, monkeypatch):
        """should create model instance using registered architecture."""
        # register dummy architecture
        register_architecture("dummy")(DummyModel)

        # mock get_model_config to return test config
        def mock_get_config(name):
            return {
                "architecture": "dummy",
                "param1": "value1",
                "param2": 42,
            }

        monkeypatch.setattr("models.loader.get_model_config", mock_get_config)

        model, config = create_model_from_config("test-model")

        assert isinstance(model, DummyModel)
        assert config["architecture"] == "dummy"
        assert config["param1"] == "value1"
        assert config["param2"] == 42

    def test_raises_if_architecture_field_missing(self, monkeypatch):
        """should raise ValueError if config missing architecture field."""
        def mock_get_config(name):
            return {"param1": "value1"}

        monkeypatch.setattr("models.loader.get_model_config", mock_get_config)

        with pytest.raises(ValueError, match="missing 'architecture' field"):
            create_model_from_config("test-model")

    def test_raises_if_architecture_not_registered(self, monkeypatch):
        """should raise ValueError if architecture not in registry."""
        def mock_get_config(name):
            return {"architecture": "nonexistent"}

        monkeypatch.setattr("models.loader.get_model_config", mock_get_config)

        with pytest.raises(ValueError, match="Unknown architecture: nonexistent"):
            create_model_from_config("test-model")

    def test_restores_architecture_field_in_config(self, monkeypatch):
        """should restore architecture field after model creation."""
        register_architecture("dummy2")(DummyModel)

        def mock_get_config(name):
            return {
                "architecture": "dummy2",
                "param": "value",
            }

        monkeypatch.setattr("models.loader.get_model_config", mock_get_config)

        model, config = create_model_from_config("test-model")

        # architecture field should be restored
        assert "architecture" in config
        assert config["architecture"] == "dummy2"
