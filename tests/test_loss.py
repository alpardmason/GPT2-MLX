"""Sanity contract test for the next-token loss.

Intentionally RED until ``GPT.loss`` is implemented. A correctly implemented
cross-entropy loss on an untrained model with ``V`` classes should be close to
``ln(V)`` (uniform-prediction baseline). This rejects fakes such as returning
``logits.mean()`` that happen to be scalar but are not a real loss.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.model import GPT


@pytest.mark.todo
def test_loss_near_ln_vocab() -> None:
    """Untrained loss should sit in a sane band around ln(vocab_size)."""
    config = GPTConfig.tiny()
    model = GPT(config)

    mx.random.seed(0)
    idx = mx.random.randint(0, config.vocab_size, (2, 4)).astype(mx.int32)
    targets = mx.random.randint(0, config.vocab_size, (2, 4)).astype(mx.int32)

    loss = model.loss(idx, targets)
    value = loss.item()

    baseline = math.log(config.vocab_size)
    assert value > 0.0, "Loss must be positive for an untrained model."
    assert value < 3.0 * baseline, (
        f"Untrained loss {value:.3f} is implausibly far from the uniform "
        f"baseline ln(vocab_size)={baseline:.3f}; check the loss computation."
    )
