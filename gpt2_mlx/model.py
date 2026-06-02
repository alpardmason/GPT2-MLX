"""GPT-2 module skeletons for an MLX learning project.

The classes in this file intentionally leave the core transformer logic as TODO
holes. The surrounding structure is present so you can focus on implementing the
important model concepts yourself.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from gpt2_mlx.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention skeleton.

    Expected input shape:
        x: ``[B, T, C]``

    Expected output shape:
        y: ``[B, T, C]``

    Where:
        B is batch size, T is sequence length, and C is ``config.n_embd``.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def create_causal_mask(self, seq_len: int) -> mx.array:
        """Create a causal attention mask for a sequence length.

        Args:
            seq_len: Sequence length ``T``.

        Returns:
            A mask whose shape can broadcast over attention scores shaped
            ``[B, H, T, T]``.

        TODO:
            Design and implement a causal masking strategy where each token can
            attend to itself and previous tokens, but not future tokens.
        """
        # Shape target: broadcast-compatible with attention scores [B, H, T, T].
        raise NotImplementedError(
            "TODO: implement causal mask creation for attention scores [B, H, T, T]."
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply causal self-attention.

        Args:
            x: Residual stream activations with shape ``[B, T, C]``.

        Returns:
            Updated residual stream activations with shape ``[B, T, C]``.

        TODO:
            Implement causal self-attention end to end:
            1. project x to qkv
            2. split q, k, v
            3. reshape into heads
            4. compute scaled causal attention
            5. combine heads
            6. apply output projection
        """
        # Input shape: [B, T, C].
        # Output shape: [B, T, C].
        raise NotImplementedError(
            "TODO: implement CausalSelfAttention.__call__ from [B, T, C] to [B, T, C]."
        )


class MLP(nn.Module):
    """GPT-style feed-forward network skeleton.

    Expected input shape:
        x: ``[B, T, C]``

    Expected output shape:
        y: ``[B, T, C]``
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply the transformer feed-forward sublayer.

        Args:
            x: Residual stream activations with shape ``[B, T, C]``.

        Returns:
            Updated residual stream activations with shape ``[B, T, C]``.

        TODO:
            Implement the full GPT-style MLP transformation, including the
            hidden expansion, nonlinearity, projection back to the residual
            width, and any configured dropout behavior.
        """
        # Input shape: [B, T, C].
        # Output shape: [B, T, C].
        raise NotImplementedError(
            "TODO: implement MLP.__call__ from [B, T, C] to [B, T, C]."
        )


class Block(nn.Module):
    """Transformer block skeleton.

    Expected input shape:
        x: ``[B, T, C]``

    Expected output shape:
        y: ``[B, T, C]``
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd, affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, affine=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply one transformer block.

        Args:
            x: Residual stream activations with shape ``[B, T, C]``.

        Returns:
            Updated residual stream activations with shape ``[B, T, C]``.

        TODO:
            Implement the transformer block as a complete concept, including
            normalization, residual paths, causal self-attention, and MLP
            composition.
        """
        # Input shape: [B, T, C].
        # Output shape: [B, T, C].
        raise NotImplementedError(
            "TODO: implement Block.__call__ from [B, T, C] to [B, T, C]."
        )


class GPT(nn.Module):
    """Decoder-only GPT model skeleton.

    Expected input shape:
        idx: ``[B, T]`` integer token IDs

    Expected output shape:
        logits: ``[B, T, vocab_size]``
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, affine=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, idx: mx.array) -> mx.array:
        """Run the GPT forward pass.

        Args:
            idx: Integer token IDs with shape ``[B, T]``.

        Returns:
            Logits with shape ``[B, T, vocab_size]``.

        TODO:
            Implement the full GPT forward contract:
            1. validate or handle sequence length against block_size
            2. create token and position embeddings
            3. combine embeddings into the residual stream
            4. apply each transformer block
            5. apply final normalization
            6. project to vocabulary logits
            7. optionally handle weight tying once you choose to implement it
        """
        # Input shape: [B, T].
        # Output shape: [B, T, vocab_size].
        raise NotImplementedError(
            "TODO: implement GPT.__call__ from token IDs [B, T] to logits [B, T, V]."
        )

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        """Compute next-token language modeling loss.

        Args:
            idx: Integer token IDs with shape ``[B, T]``.
            targets: Target token IDs with shape ``[B, T]``.

        Returns:
            A scalar loss array.

        TODO:
            Implement the language modeling loss using the logits produced by
            ``self(idx)`` and the provided next-token targets.
        """
        # idx shape: [B, T].
        # targets shape: [B, T].
        # output shape: scalar [].
        raise NotImplementedError(
            "TODO: implement GPT.loss for logits [B, T, V] and targets [B, T]."
        )

    def generate(
        self,
        idx: mx.array,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> mx.array:
        """Autoregressively append new tokens to a prompt.

        Args:
            idx: Integer token IDs with shape ``[B, T]``.
            max_new_tokens: Number of tokens to append.
            temperature: Sampling temperature for the future implementation.

        Returns:
            Integer token IDs with shape ``[B, T + max_new_tokens]``.

        TODO:
            Implement the generation loop as a complete concept, including
            context cropping, model calls, next-token selection, and appending
            generated tokens.
        """
        # Input shape: [B, T].
        # Output shape: [B, T + max_new_tokens].
        raise NotImplementedError(
            "TODO: implement GPT.generate from [B, T] to [B, T + max_new_tokens]."
        )


__all__ = [
    "Block",
    "CausalSelfAttention",
    "GPT",
    "GPTConfig",
    "MLP",
]

