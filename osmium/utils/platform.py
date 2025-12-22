"""Platform detection utilities for osmium."""

import platform


def is_macos() -> bool:
    """check if the current platform is macOS.

    returns:
        True if running on macOS (Darwin), False otherwise
    """
    return platform.system() == "Darwin"


def is_linux() -> bool:
    """check if the current platform is Linux.

    returns:
        True if running on Linux, False otherwise
    """
    return platform.system() == "Linux"


def get_platform_name() -> str:
    """get the current platform name.

    returns:
        Platform name (e.g., 'Darwin', 'Linux', 'Windows')
    """
    return platform.system()
