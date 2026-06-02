"""Tests for small non-core helpers in gpt2_mlx.utils."""

from __future__ import annotations

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.model import GPT
from gpt2_mlx.utils import count_parameters


def test_count_parameters_returns_positive_int() -> None:
    """Parameter counting works right after construction (no forward needed)."""
    model = GPT(GPTConfig.tiny())

    total = count_parameters(model)

    assert isinstance(total, int)
    assert total > 0


def test_count_parameters_matches_hand_computed_tiny_total() -> None:
    """Cross-check the count against a hand-derived figure for the tiny config.

    tiny: vocab=64, block=8, n_layer=2, n_head=2, n_embd=16, bias=True.

    Embeddings:
        token:    64 * 16 = 1024
        position:  8 * 16 = 128
    Per block (x2):
        ln_1:                 weight 16 + bias 16 = 32
        attn.qkv_proj: 16*48 = 768 + bias 48      = 816
        attn.out_proj: 16*16 = 256 + bias 16      = 272
        ln_2:                 weight 16 + bias 16 = 32
        mlp.fc:        16*64 = 1024 + bias 64     = 1088
        mlp.proj:      64*16 = 1024 + bias 16     = 1040
        block total                              = 3280
    ln_f:                     weight 16 + bias 16 = 32
    lm_head:       16*64 = 1024 (bias=False)      = 1024
    Total = 1024 + 128 + 2*3280 + 32 + 1024 = 8768
    """
    model = GPT(GPTConfig.tiny())

    assert count_parameters(model) == 8768
