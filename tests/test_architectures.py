"""Tests for architecture registration system."""

from __future__ import annotations

import pytest

from models.architectures import (
    ARCHITECTURE_REGISTRY,
    register_architecture,
    get_architecture_class,
    list_architectures,
)


class TestRegisterArchitecture:
    """test architecture registration decorator."""

    def test_decorator_registers_class_in_registry(self):
        """decorator should add class to ARCHITECTURE_REGISTRY."""
        @register_architecture("test_arch")
        class TestModel:
            pass

        assert "test_arch" in ARCHITECTURE_REGISTRY
        assert ARCHITECTURE_REGISTRY["test_arch"] is TestModel

    def test_decorator_returns_original_class(self):
        """decorator should return the class unchanged."""
        @register_architecture("test_arch2")
        class TestModel:
            pass

        assert TestModel.__name__ == "TestModel"


class TestGetArchitectureClass:
    """test architecture class lookup."""

    def test_returns_registered_class(self):
        """should return class for registered architecture."""
        @register_architecture("gpt_test")
        class GPTTest:
            pass

        result = get_architecture_class("gpt_test")
        assert result is GPTTest

    def test_raises_for_unknown_architecture(self):
        """should raise ValueError for unknown architecture."""
        with pytest.raises(ValueError, match="Unknown architecture: nonexistent"):
            get_architecture_class("nonexistent")

    def test_error_message_lists_available_architectures(self):
        """error should list available architectures."""
        with pytest.raises(ValueError, match="Available:"):
            get_architecture_class("unknown")


class TestListArchitectures:
    """test listing registered architectures."""

    def test_returns_list_of_registered_names(self):
        """should return list of all registered architecture names."""
        @register_architecture("arch1")
        class Arch1:
            pass

        @register_architecture("arch2")
        class Arch2:
            pass

        result = list_architectures()
        assert isinstance(result, list)
        assert "arch1" in result
        assert "arch2" in result
