"""Small data helpers for token-level language modeling."""

from __future__ import annotations

import mlx.core as mx


def make_lm_batch(
    tokens: list[int] | mx.array,
    batch_size: int,
    block_size: int,
) -> tuple[mx.array, mx.array]:
    """Create a simple next-token language modeling batch.

    Args:
        tokens: A one-dimensional token stream with length at least
            ``batch_size * block_size + 1``.
        batch_size: Number of rows in the returned batch.
        block_size: Number of time steps per row.

    Returns:
        A tuple ``(x, y)`` where both arrays have shape ``[B, T]``.
        ``y`` is shifted one token to the right relative to ``x``.
    """
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}."
        raise ValueError(msg)
    if block_size <= 0:
        msg = f"block_size must be positive, got {block_size}."
        raise ValueError(msg)

    token_array = mx.array(tokens, dtype=mx.int32)
    required = batch_size * block_size + 1
    if token_array.ndim != 1:
        msg = f"tokens must be one-dimensional, got shape {token_array.shape}."
        raise ValueError(msg)
    if token_array.shape[0] < required:
        msg = (
            f"Need at least {required} tokens for batch_size={batch_size} and "
            f"block_size={block_size}, got {token_array.shape[0]}."
        )
        raise ValueError(msg)

    x = token_array[: batch_size * block_size].reshape(batch_size, block_size)
    y = token_array[1:required].reshape(batch_size, block_size)
    return x, y
