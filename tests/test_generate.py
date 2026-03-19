"""Tests for generate command implementation."""

from unittest.mock import patch

import pytest
import torch

from kilonova.train.config import resolve_device


class TestResolveDevice:
    """tests for device resolution logic"""

    def test_explicit_cpu_returns_cpu(self):
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_explicit_cuda_returns_cuda(self):
        with patch("kilonova.train.config.torch.cuda.is_available", return_value=True):
            device = resolve_device("cuda")
            assert device == torch.device("cuda")

    def test_explicit_cuda_raises_when_unavailable(self):
        with patch("kilonova.train.config.torch.cuda.is_available", return_value=False):
            with pytest.raises(ValueError, match="CUDA device requested"):
                resolve_device("cuda")

    def test_auto_with_cuda_available_returns_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            device = resolve_device("auto")
            assert device == torch.device("cuda")

    def test_auto_without_cuda_returns_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            device = resolve_device("auto")
            assert device == torch.device("cpu")
