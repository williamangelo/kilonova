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

    def test_warns_on_duplicate_registration(self):
        """decorator should warn when overwriting an existing architecture."""
        @register_architecture("duplicate_test")
        class FirstModel:
            pass

        # should warn and show both old and new class names
        with pytest.warns(UserWarning) as warning_list:
            @register_architecture("duplicate_test")
            class SecondModel:
                pass

            warning_msg = str(warning_list[0].message)
            assert "Architecture 'duplicate_test' already registered" in warning_msg
            assert "FirstModel" in warning_msg
            assert "SecondModel" in warning_msg

        # verify it still overwrites correctly
        assert ARCHITECTURE_REGISTRY["duplicate_test"] is SecondModel


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


class TestGPTRegistration:
    """test that GPT architecture is properly registered."""

    def test_gpt_registered_in_architecture_registry(self):
        """GPT should be registered as 'gpt' architecture."""
        from models.gpt import GPT
        from models.architectures import get_architecture_class

        gpt_class = get_architecture_class("gpt")
        assert gpt_class is GPT

    def test_gpt_in_list_architectures(self):
        """gpt should appear in list of architectures."""
        from models.gpt import GPT  # trigger import
        from models.architectures import list_architectures

        architectures = list_architectures()
        assert "gpt" in architectures
