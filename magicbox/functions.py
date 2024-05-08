"""Utility functions for functional programming."""

from collections.abc import Callable
from typing import ParamSpec, TypeVar

T = TypeVar("T", bound=tuple)
P = ParamSpec("P")


def return_arg(fn: Callable[P, T], arg: int = 0) -> Callable[P, T]:
    """Get a specific argument from a function that returns a tuple.

    Args:
        fn (Callable[P, T]): The function to modify,
        arg (int, optional): The arg to get. Defaults to 0.
    """

    def call(*args: P.args, **kwargs: P.kwargs):
        val = fn(*args, **kwargs)
        return val[arg]

    return call
