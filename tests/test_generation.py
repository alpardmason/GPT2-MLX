"""Contract tests for autoregressive generation."""

from __future__ import annotations

import mlx.core as mx
import pytest

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.model import GPT
from gpt2_mlx.utils import set_seed


@pytest.mark.todo
def test_generate_appends_requested_number_of_tokens() -> None:
    """Generation should return the prompt plus max_new_tokens."""
    config = GPTConfig.tiny()
    model = GPT(config)
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

    out = model.generate(prompt, max_new_tokens=5)

    assert out.shape == (1, 8)


@pytest.mark.todo
def test_generate_past_block_size() -> None:
    """A prompt longer than block_size must still generate without error.

    This forces the implementation to crop the context to the last
    ``block_size`` tokens before each model call, instead of feeding an
    ever-growing sequence past the position-embedding limit.
    """
    config = GPTConfig.tiny()
    model = GPT(config)
    prompt_len = config.block_size + 3
    prompt = mx.arange(prompt_len, dtype=mx.int32).reshape(1, prompt_len)

    out = model.generate(prompt, max_new_tokens=4)

    assert out.shape == (1, prompt_len + 4)


@pytest.mark.todo
def test_greedy_generation_deterministic() -> None:
    """With a fixed seed, generation should be reproducible."""
    config = GPTConfig.tiny()
    model = GPT(config)
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)

    set_seed(0)
    first = model.generate(prompt, max_new_tokens=5)
    set_seed(0)
    second = model.generate(prompt, max_new_tokens=5)

    assert bool(mx.array_equal(first, second).item()), (
        "Generation is not reproducible under a fixed seed."
    )

