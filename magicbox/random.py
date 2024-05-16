"""Utility functions for working with random number generators."""

import contextlib
import random
from typing import Any, Literal

import numpy as np


def get_states() -> (
    dict[
        Literal[
            "numpy",
            "random",
            "torch",
            "torch.cuda",
            "torch.backends.cudnn.deterministic",
            "torch.backends.cudnn.benchmark",
        ],
        Any,
    ]
):
    result = {"random": random.getstate(), "numpy": np.random.get_state()}
    try:
        import torch

        result["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            result["torch.cuda"] = torch.cuda.get_rng_state_all()
            result["torch.backends.cudnn.deterministic"] = (
                torch.backends.cudnn.deterministic
            )
            result["torch.backends.cudnn.benchmark"] = torch.backends.cudnn.benchmark
    except ImportError:
        ...
    return result


_INITIAL_SEEDS = get_states()


def set_seed(seed: int):
    """Set the seed for various packages."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        ...


def reset_states():
    """Reset the states to the initial state."""
    random.setstate(_INITIAL_SEEDS["random"])
    np.random.set_state(_INITIAL_SEEDS["numpy"])
    try:
        import torch

        torch.set_rng_state(_INITIAL_SEEDS["torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(_INITIAL_SEEDS["torch.cuda"])
            torch.backends.cudnn.deterministic = _INITIAL_SEEDS[
                "torch.backends.cudnn.deterministic"
            ]
            torch.backends.cudnn.benchmark = _INITIAL_SEEDS[
                "torch.backends.cudnn.benchmark"
            ]
    except ImportError:
        ...


@contextlib.contextmanager
def seed_context(seed: int):
    """Context manager for using a specific seed."""
    try:
        set_seed(seed)
        yield
    finally:
        reset_states()
