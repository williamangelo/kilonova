"""Platform detection utilities for kilonova."""

import platform


def is_macos() -> bool:
    """check if the current platform is macOS.

    returns:
        True if running on macOS (Darwin), False otherwise
    """
    return platform.system() == "Darwin"


