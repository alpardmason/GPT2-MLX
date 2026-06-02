"""Contract tests for causal mask behavior."""

from __future__ import annotations

import mlx.core as mx
import pytest

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.model import CausalSelfAttention


@pytest.mark.todo
def test_causal_mask_contract() -> None:
    """A causal mask should allow current/past positions and block future ones."""
    config = GPTConfig.tiny()
    attention = CausalSelfAttention(config)
    seq_len = 4

    mask = attention.create_causal_mask(seq_len)

    assert mask.shape in {
        (seq_len, seq_len),
        (1, seq_len, seq_len),
        (1, 1, seq_len, seq_len),
    }

    materialized = mx.array(mask).reshape(-1, seq_len, seq_len)[0]
    for row in range(seq_len):
        for col in range(seq_len):
            value = bool(materialized[row, col].item())
            if col <= row:
                assert value, f"position {row} should be allowed to attend to {col}"
            else:
                assert not value, f"position {row} should not attend to future {col}"

