"""Exceptions used throughout the package."""


class PackageRootNotFoundError(IOError):
    """Error for when the package root is not found."""

    def __init__(self, message: str | None = None) -> None:
        """Create an exception for when the package root is not found."""
        super().__init__(message or "Package root could not be found.")
