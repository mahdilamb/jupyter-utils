"""Test random number generator utility functions."""

import sys
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
import torch

from magicbox import random as random_utils


def assert_all_equal(values: Sequence[npt.ArrayLike], msg: str | None = None):
    """Test that all arrays are equal."""
    for a, b in zip(values[:-1], values[1:], strict=False):
        np.testing.assert_array_equal(a, b, err_msg=msg)


def assert_not_all_equal(values: Sequence[npt.ArrayLike], msg: str | None = None):
    """Test that all arrays are not equal."""
    failed = False
    for a, b in zip(values[:-1], values[1:], strict=False):
        try:
            np.testing.assert_array_equal(a, b, err_msg=msg)
            failed = True
            break
        except AssertionError:
            ...
    if failed:
        raise AssertionError(msg or "values are not the same")


@pytest.mark.parametrize(("seed",), [(42,), (None,)])
def test_numpy_seed(seed: int | None):
    """Test that numpy seed can be set so that determinicity is maintained."""
    if seed is not None:
        values = []
        for _ in range(32):
            with random_utils.seed_context(seed):
                values.append(np.random.randint(-sys.maxsize, sys.maxsize, 128))
        assert_all_equal(values, "Expected the arrays to be deterministic.")
    else:
        assert_not_all_equal(
            [np.random.randint(-sys.maxsize, sys.maxsize, 128) for _ in range(32)],
            "Expected the arrays not to be deterministic.",
        )


@pytest.mark.parametrize(("seed",), [(42,), (None,)])
def test_torch_seed(seed: int | None):
    """Test that torch seed can be set so that determinicity is maintained."""
    if seed is not None:
        values = []
        for _ in range(32):
            with random_utils.seed_context(seed):
                values.append(
                    torch.randint(low=-sys.maxsize, high=sys.maxsize, size=(128,))
                )
        assert_all_equal(values, "Expected the arrays to be deterministic.")
    else:
        assert_not_all_equal(
            [
                torch.randint(low=-sys.maxsize, high=sys.maxsize, size=(128,))
                for _ in range(32)
            ],
            "Expected the arrays not to be deterministic.",
        )


if __name__ == "__main__":
    pytest.main([__file__])
