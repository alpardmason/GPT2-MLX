"""Configuration objects for the educational GPT-2 MLX skeleton."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Architecture settings for a GPT-style decoder-only transformer.

    Attributes:
        vocab_size: Number of token IDs the model can represent.
        block_size: Maximum sequence length, also called context length.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads per block.
        n_embd: Width of the residual stream and token embeddings.
        dropout: Dropout probability used by modules that support dropout.
        bias: Whether linear and normalization layers should include bias terms.
        tie_weights: Whether token embeddings and LM head weights should be tied.
    """

    vocab_size: int = 50_257
    block_size: int = 1_024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = False

    def __post_init__(self) -> None:
        """Validate basic architectural invariants."""
        positive_int_fields = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
        }
        for field_name, value in positive_int_fields.items():
            if value <= 0:
                msg = f"{field_name} must be positive, got {value}."
                raise ValueError(msg)

        if self.n_embd % self.n_head != 0:
            msg = (
                "n_embd must be divisible by n_head so each attention head has "
                f"an equal width, got n_embd={self.n_embd}, n_head={self.n_head}."
            )
            raise ValueError(msg)

        if not 0.0 <= self.dropout < 1.0:
            msg = f"dropout must be in [0.0, 1.0), got {self.dropout}."
            raise ValueError(msg)

    @classmethod
    def tiny(cls) -> GPTConfig:
        """Return a very small config for tests and learning experiments."""
        return cls(
            vocab_size=64,
            block_size=8,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
            tie_weights=False,
        )

