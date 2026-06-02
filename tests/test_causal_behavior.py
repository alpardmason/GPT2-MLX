"""Behavioral contract tests for causal information flow.

These tests are intentionally RED until the learner implements the forward
passes. They assert the single most important property of a causal language
model: outputs at position ``t`` must not depend on tokens at positions ``> t``.
Shape-only tests cannot catch a leaky (bidirectional) attention implementation;
these can.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.model import GPT, CausalSelfAttention


@pytest.mark.todo
def test_no_future_leakage() -> None:
    """Changing future tokens must not change earlier-position logits."""
    config = GPTConfig.tiny()
    model = GPT(config)

    base = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    # Identical prefix [1, 2], different suffix from position 2 onward.
    perturbed = mx.array([[1, 2, 9, 5]], dtype=mx.int32)

    logits_base = model(base)
    logits_perturbed = model(perturbed)

    prefix_len = 2
    diff = mx.abs(
        logits_base[:, :prefix_len, :] - logits_perturbed[:, :prefix_len, :]
    ).max()
    assert diff.item() < 1e-5, (
        "Logits at shared prefix positions changed when future tokens changed, "
        "which means attention is leaking information from the future."
    )


@pytest.mark.todo
def test_attention_respects_causality() -> None:
    """Attention output at a position must ignore later positions."""
    config = GPTConfig.tiny()
    attention = CausalSelfAttention(config)

    mx.random.seed(0)
    x = mx.random.normal((1, 4, config.n_embd))
    perturbed = mx.array(x)
    # Perturb only the last time step.
    perturbed[:, 3, :] = perturbed[:, 3, :] + 10.0

    y = attention(x)
    y_perturbed = attention(perturbed)

    diff = mx.abs(y[:, :3, :] - y_perturbed[:, :3, :]).max()
    assert diff.item() < 1e-5, (
        "Attention outputs at positions 0..2 changed when position 3 changed; "
        "the causal mask is not being applied correctly."
    )
