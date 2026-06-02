"""Small utilities that avoid core GPT-2 model math."""

from __future__ import annotations

import random
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and MLX random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for an MLX module.

    Args:
        model: An ``mlx.nn.Module`` exposing ``parameters()``.

    Returns:
        Total number of scalar parameters.
    """
    # previously: hand-rolled recursion over dict/list/tuple parameter trees.
    #   def count_value(value): ... isinstance(value, mx.array) -> np.prod(shape)
    # mlx.utils.tree_flatten is the idiomatic way to walk an MLX parameter tree;
    # it flattens the nested dict/list structure into (name, array) leaves.
    # Without a `destination`, it returns the list-of-tuples form.
    leaves = cast(list[tuple[str, mx.array]], tree_flatten(model.parameters()))
    return sum(int(np.prod(array.shape)) for _, array in leaves)

