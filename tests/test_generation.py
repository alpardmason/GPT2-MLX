"""Contract tests for autoregressive generation."""

from __future__ import annotations

import mlx.core as mx

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.model import GPT


def test_generate_appends_requested_number_of_tokens() -> None:
    """Generation should return the prompt plus max_new_tokens."""
    config = GPTConfig.tiny()
    model = GPT(config)
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

    out = model.generate(prompt, max_new_tokens=5)

    assert out.shape == (1, 8)

