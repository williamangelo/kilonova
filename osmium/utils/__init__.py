"""Shared utilities for osmium CLI."""

from osmium.utils.paths import PathResolver
from osmium.utils.platform import is_macos, is_linux, get_platform_name

__all__ = ["PathResolver", "is_macos", "is_linux", "get_platform_name"]
