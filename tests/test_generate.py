"""Tests for generate command implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from kilonova.commands.generate import resolve_checkpoint
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


class TestResolveCheckpoint:
    """tests for checkpoint path resolution"""

    def test_returns_path_if_file_exists(self, tmp_path):
        checkpoint = tmp_path / "model.pth"
        checkpoint.touch()

        result = resolve_checkpoint(str(checkpoint))
        assert result == checkpoint

    def test_resolves_run_name_to_best_checkpoint(self, tmp_path):
        # create run directory structure
        run_dir = tmp_path / "data" / "runs" / "my-run"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        best = checkpoint_dir / "best.pth"
        best.touch()

        with patch("kilonova.commands.generate.PathResolver") as mock_resolver:
            mock_resolver.return_value.run_dir.return_value = run_dir
            result = resolve_checkpoint("my-run")
            assert result == best

    def test_resolves_run_name_to_latest_epoch_if_no_best(self, tmp_path):
        run_dir = tmp_path / "data" / "runs" / "my-run"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        # create epoch checkpoints but no best.pth
        (checkpoint_dir / "epoch-001.pth").touch()
        (checkpoint_dir / "epoch-002.pth").touch()
        (checkpoint_dir / "epoch-003.pth").touch()

        with patch("kilonova.commands.generate.PathResolver") as mock_resolver:
            mock_resolver.return_value.run_dir.return_value = run_dir
            result = resolve_checkpoint("my-run")
            assert result == checkpoint_dir / "epoch-003.pth"

    def test_raises_for_nonexistent_run(self, tmp_path):
        run_dir = tmp_path / "data" / "runs" / "nonexistent"

        with patch("kilonova.commands.generate.PathResolver") as mock_resolver:
            mock_resolver.return_value.run_dir.return_value = run_dir
            with pytest.raises(Exception) as exc_info:
                resolve_checkpoint("nonexistent")
            assert "not found" in str(exc_info.value)

    def test_raises_for_run_with_no_checkpoints(self, tmp_path):
        run_dir = tmp_path / "data" / "runs" / "empty-run"
        run_dir.mkdir(parents=True)
        # no checkpoints directory

        with patch("kilonova.commands.generate.PathResolver") as mock_resolver:
            mock_resolver.return_value.run_dir.return_value = run_dir
            with pytest.raises(Exception) as exc_info:
                resolve_checkpoint("empty-run")
            assert "checkpoints" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    def test_raises_for_empty_checkpoints_directory(self, tmp_path):
        run_dir = tmp_path / "data" / "runs" / "empty-checkpoints"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        # empty checkpoints directory

        with patch("kilonova.commands.generate.PathResolver") as mock_resolver:
            mock_resolver.return_value.run_dir.return_value = run_dir
            with pytest.raises(Exception) as exc_info:
                resolve_checkpoint("empty-checkpoints")
            assert "No checkpoints found" in str(exc_info.value)
