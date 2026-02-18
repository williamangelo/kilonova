"""Tests for path resolution utilities."""

from pathlib import Path

import pytest

from osmium.utils.paths import PathResolver


class TestPathResolver:
    """tests for PathResolver"""

    def test_default_base_dir(self):
        resolver = PathResolver()
        assert resolver.base_dir == Path("data")

    def test_custom_base_dir(self):
        resolver = PathResolver(base_dir="/custom/path")
        assert resolver.base_dir == Path("/custom/path")

    def test_raw_dir_default(self):
        resolver = PathResolver()
        assert resolver.raw_dir("gutenberg") == Path("data/raw/gutenberg")

    def test_raw_dir_with_override(self):
        resolver = PathResolver()
        override = Path("/custom/raw")
        assert resolver.raw_dir("gutenberg", override=override) == override

    def test_clean_dir_default(self):
        resolver = PathResolver()
        assert resolver.clean_dir("gutenberg") == Path("data/clean/gutenberg")

    def test_clean_dir_with_override(self):
        resolver = PathResolver()
        override = Path("/custom/clean")
        assert resolver.clean_dir("gutenberg", override=override) == override

    def test_processed_dir_default(self):
        resolver = PathResolver()
        assert resolver.processed_dir("gutenberg") == Path("data/processed/gutenberg")

    def test_processed_dir_with_override(self):
        resolver = PathResolver()
        override = Path("/custom/processed")
        assert resolver.processed_dir("gutenberg", override=override) == override

    def test_run_dir(self):
        resolver = PathResolver()
        assert resolver.run_dir("baseline") == Path("data/runs/baseline")

