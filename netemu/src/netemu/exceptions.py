"""
Custom exceptions for the netemu package.
"""


class NetEmuError(Exception):
    """Base exception for all netemu errors."""

    pass


class SudoNotAvailableError(NetEmuError):
    """
    Raised when sudo access is required but not available.

    Network emulation requires root privileges to execute tc commands.
    Configure passwordless sudo for tc/ip commands or run as root.
    """

    def __init__(self, message: str = "Sudo access is required for network emulation"):
        super().__init__(message)


class ProfileNotFoundError(NetEmuError):
    """
    Raised when a requested network profile is not found.

    Check that the profile name is correct and the profiles file
    has been loaded properly.
    """

    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        super().__init__(f"Network profile not found: {profile_name}")


class CommandFailedError(NetEmuError):
    """
    Raised when a tc/ip command fails to execute.

    This may indicate insufficient permissions, invalid parameters,
    or missing kernel modules.
    """

    def __init__(self, command: str, returncode: int, stderr: str = ""):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        message = f"Command failed (exit {returncode}): {command}"
        if stderr:
            message += f"\nStderr: {stderr}"
        super().__init__(message)


class ProfileLoadError(NetEmuError):
    """
    Raised when profile configuration file cannot be loaded.

    Check that the file exists, is valid YAML, and has the expected structure.
    """

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        message = f"Failed to load profiles from: {path}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)
