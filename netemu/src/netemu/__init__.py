"""
netemu - Linux tc/netem network emulation wrapper for Python.

This package provides a Python interface to Linux tc/netem for network
condition emulation. It supports delay, jitter, packet loss, bandwidth
limiting, corruption, reordering, and duplication.

Example:
    >>> from netemu import NetworkEmulator, NetworkProfile
    >>> emulator = NetworkEmulator(interface="eth0")
    >>> emulator.apply_settings(delay_ms=100, loss_pct=1.0)
    True
    >>> emulator.clear()
    True

Using profiles:
    >>> emulator = NetworkEmulator(profiles_path="profiles.yaml")
    >>> emulator.apply_profile("poor_cellular")
    True
"""

from .emulator import NetworkEmulator, apply_profile, clear_profile
from .exceptions import (
    CommandFailedError,
    NetEmuError,
    ProfileLoadError,
    ProfileNotFoundError,
    SudoNotAvailableError,
)
from .profile import NetworkProfile

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "NetworkEmulator",
    "NetworkProfile",
    # Exceptions
    "NetEmuError",
    "SudoNotAvailableError",
    "ProfileNotFoundError",
    "CommandFailedError",
    "ProfileLoadError",
    # Convenience functions
    "apply_profile",
    "clear_profile",
    # Version
    "__version__",
]
