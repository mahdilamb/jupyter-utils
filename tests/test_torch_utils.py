"""Test the torch utils."""

import collections
from collections.abc import Sequence

import pytest
import torch

import magicbox.torch as torch_utils


@pytest.fixture(scope="session")
def neural_network():
    """Create a neural network."""
    model = torch.nn.Sequential(
        collections.OrderedDict(
            **{
                f"layer_{i}": layer
                for i, layer in enumerate(
                    (torch.nn.Linear(1, 2), *(torch.nn.Linear(2, 2) for _ in range(20)))
                )
            }
        )
    )

    yield model


@pytest.mark.parametrize(("freeze",), [((2,),)])
def test_freeze_by_index(freeze: Sequence[int], neural_network):
    """Test freezing by index."""
    model = neural_network
    assert all(
        param.requires_grad for param in model.parameters()
    ), "Expected all params to be trainable."
    model = torch_utils.WeightsFreezer[freeze](model)
    assert (
        tuple(
            i for i, param in enumerate(model.parameters()) if not param.requires_grad
        )
        == freeze
    ), "Expected the requested layers to not require_grad."


if __name__ == "__main__":
    pytest.main([__file__])
