"""Shape and construction contract tests for the GPT-2 MLX skeleton."""

from __future__ import annotations

import mlx.core as mx
import pytest

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.data import make_lm_batch
from gpt2_mlx.model import GPT, MLP, Block, CausalSelfAttention


def test_tiny_config_is_valid() -> None:
    """The tiny test config should satisfy core architecture invariants."""
    config = GPTConfig.tiny()

    assert config.vocab_size == 64
    assert config.block_size == 8
    assert config.n_embd % config.n_head == 0


def test_invalid_head_width_raises() -> None:
    """Embedding width must divide evenly across attention heads."""
    with pytest.raises(ValueError, match="n_embd must be divisible"):
        GPTConfig(n_embd=10, n_head=3)


def test_make_lm_batch_shapes_and_shift() -> None:
    """Batch helper should create aligned next-token inputs and targets."""
    x, y = make_lm_batch(list(range(9)), batch_size=2, block_size=4)

    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    assert x[0, 0].item() == 0
    assert y[0, 0].item() == 1
    assert x[1, 3].item() == 7
    assert y[1, 3].item() == 8


def test_model_instantiates() -> None:
    """The skeleton should be constructible before forward logic exists."""
    model = GPT(GPTConfig.tiny())

    assert isinstance(model, GPT)


def test_attention_output_shape_contract() -> None:
    """Attention should map [B, T, C] to [B, T, C] once implemented."""
    config = GPTConfig.tiny()
    attention = CausalSelfAttention(config)
    x = mx.zeros((2, 4, config.n_embd))

    y = attention(x)

    assert y.shape == x.shape


def test_mlp_output_shape_contract() -> None:
    """MLP should map [B, T, C] to [B, T, C] once implemented."""
    config = GPTConfig.tiny()
    mlp = MLP(config)
    x = mx.zeros((2, 4, config.n_embd))

    y = mlp(x)

    assert y.shape == x.shape


def test_block_output_shape_contract() -> None:
    """A block should preserve the residual stream shape once implemented."""
    config = GPTConfig.tiny()
    block = Block(config)
    x = mx.zeros((2, 4, config.n_embd))

    y = block(x)

    assert y.shape == x.shape


def test_gpt_logits_shape_contract() -> None:
    """GPT should map token IDs [B, T] to logits [B, T, vocab_size]."""
    config = GPTConfig.tiny()
    model = GPT(config)
    idx = mx.zeros((2, 4), dtype=mx.int32)

    logits = model(idx)

    assert logits.shape == (2, 4, config.vocab_size)


def test_gpt_loss_is_scalar_contract() -> None:
    """Language modeling loss should return a scalar once implemented."""
    config = GPTConfig.tiny()
    model = GPT(config)
    idx = mx.zeros((2, 4), dtype=mx.int32)
    targets = mx.ones((2, 4), dtype=mx.int32)

    loss = model.loss(idx, targets)

    assert loss.shape == ()
