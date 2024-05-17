"""Utility functions for working with pytorch."""

import re
from collections.abc import Callable, Sequence
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

import numpy as np
import torch

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__file__)

Model = TypeVar("Model", bound=torch.nn.Module)
Index: TypeAlias = Sequence[int] | Sequence[str] | slice | int | str


class WeightsFreezingFunction(Generic[Model], Protocol):
    """Protocol for a function that freezes layers."""

    def __call__(
        self,
        model: Model,
        verbose: bool = False,
    ) -> Model:
        """Freeze the layers of a model."""
        ...


class WeightsFreezerMetaclass(Generic[Model], type):
    """Metaclass for weight freezing."""

    def __getitem__(cls, index: Index) -> WeightsFreezingFunction[Model]:
        """Get a weight freezing function based on the supplied index.

        If the index is:
        - a string: use regex
        - an int, sequence of ints, sequence of strings: freeze specific parameters.
        - a slice: freeze a range of layers.
        """
        if isinstance(index, int):
            index = (index,)

        def freeze_weights(
            model: Model,
            verbose: bool = False,
        ) -> Model:
            """Freeze the weights of a model."""
            if isinstance(index, str):
                if verbose:
                    logger.info(f"Freezing layers matching r'{index}'")
                pattern = re.compile(index)
                for name, param in model.named_parameters():
                    param.requires_grad = pattern.match(name) is None

            elif isinstance(index, slice):
                # Freeze by slice
                all_parameters = tuple(model.named_parameters())[index]
                if verbose:
                    logger.info(
                        "Freezing layers: " + repr([name for name, _ in all_parameters])
                    )
                for _, param in all_parameters:
                    param.requires_grad = False

            elif index:
                if verbose:
                    logger.info(f"Freezing layers: {index}")
                if isinstance(index[0], int):
                    for i, param in enumerate(model.parameters()):
                        param.requires_grad = i not in index
                else:
                    for name, param in model.named_parameters():
                        param.requires_grad = name not in index
            return model

        return freeze_weights


class WeightsFreezer(Generic[Model], metaclass=WeightsFreezerMetaclass):
    """Class for freezing layers."""


def early_stopping(
    model: torch.nn.Module,
    min_delta: float = 0,
    patience: int = 0,
    verbose: Literal[0, 1] = 0,
    mode: Literal["min", "max"] = "min",
    baseline: float | None = None,
    restore_best_weights: bool = False,
    start_from_epoch: int = 0,
) -> Callable[[float], bool]:
    """Create a callable to check if training should be stopped.

    To restart, use the `.restart` method of the function.
    """
    if mode == "min":
        monitor_op = np.less
    elif mode == "max":
        monitor_op = np.greater
    if monitor_op == np.greater:
        min_delta *= 1
    else:
        min_delta *= -1
    wait = 0
    stopped_epoch = 0
    best = np.Inf if monitor_op == np.less else -np.Inf
    best_weights: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epoch = 0

    def reset():
        nonlocal epoch, best_weights, wait, best, stopped_epoch, best_epoch
        wait = 0
        stopped_epoch = 0
        best = np.Inf if monitor_op == np.less else -np.Inf
        best_weights = None
        best_epoch = 0
        epoch = 0

    early_stopping.reset = reset

    def should_stop(current: float | None) -> bool:
        nonlocal epoch, best_weights, wait, best, stopped_epoch, best_epoch
        current_epoch = epoch
        epoch += 1
        if current is None or current_epoch < start_from_epoch:
            return False
        if restore_best_weights and best_weights is None:
            best_weights = model.state_dict()
        wait += 1
        if monitor_op(current - min_delta, best):
            best = current
            best_epoch = current_epoch
            if restore_best_weights:
                best_weights = model.state_dict()
            if baseline is None or monitor_op(current - min_delta, baseline):
                wait = 0
        if wait >= patience and current_epoch > 0:
            stopped_epoch = current_epoch
            if restore_best_weights and best_weights is not None:
                if verbose > 0:
                    logger.info(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{best_epoch + 1}."
                    )
                model.load_state_dict(best_weights)
            return True
        return False

    return should_stop
