"""Test common utility functions."""


import pytest

from jupyter_utils import functions as func_utils


def test_return_arg():
    """Test that finding the package root works."""

    def returns_tuple() -> tuple[int, str, bool]:
        return 1, "1", True

    assert (
        func_utils.return_arg(returns_tuple, 0)() == 1
    ), "Expected first arg to be `1`."
    assert (
        func_utils.return_arg(returns_tuple, 1)() == "1"
    ), 'Expected second arg to be "1".'
    assert (
        func_utils.return_arg(returns_tuple, 2)() is True
    ), "Expected this arg to be `True`."


if __name__ == "__main__":
    pytest.main([__file__])
