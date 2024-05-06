"""Common utility functions for working with jupyter notebooks."""

import os

from jupyter_utils import exceptions


def find_project_root(file: str = "pyproject.toml") -> str:
    """Find the package root."""
    cwd = os.getcwd()
    while not os.path.exists(os.path.join(cwd, file)):
        cwd = os.path.dirname(cwd)
        if cwd == os.path.sep:
            raise exceptions.PackageRootNotFoundError(
                f"Failed to find package root as no {file} exists in the tree."
            )
    return cwd
