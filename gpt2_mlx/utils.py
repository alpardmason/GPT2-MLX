"""Small utilities that avoid core GPT-2 model math."""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and MLX random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def count_parameters(model: object) -> int:
    """Count trainable parameters for an MLX module-like object.

    Args:
        model: An object exposing ``parameters()`` in the style of ``mlx.nn``.

    Returns:
        Total number of scalar parameters.
    """
    parameters = model.parameters() if hasattr(model, "parameters") else {}

    def count_value(value: object) -> int:
        if isinstance(value, mx.array):
            return int(np.prod(value.shape))
        if isinstance(value, dict):
            return sum(count_value(child) for child in value.values())
        if isinstance(value, list | tuple):
            return sum(count_value(child) for child in value)
        return 0

    return count_value(parameters)

