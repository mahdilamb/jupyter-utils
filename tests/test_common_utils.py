"""Test common utility functions."""

import os
import tempfile

import pytest

from jupyter_utils import common as common_utils
from jupyter_utils import exceptions


def test_find_package_root(create_working_directory_mocker):
    """Test that finding the package root works."""
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        create_working_directory_mocker(tmp_dir),
    ):
        with pytest.raises(exceptions.PackageRootNotFoundError):
            common_utils.find_project_root()
            print(common_utils.find_project_root())
        with open(os.path.join(tmp_dir, "pyproject.toml"), "w"):
            ...
        assert os.path.relpath(common_utils.find_project_root(), tmp_dir) == ".", (
            "Expected project root to be the temporary directory containing the "
            + "pyproject file."
        )


if __name__ == "__main__":
    pytest.main([__file__])
