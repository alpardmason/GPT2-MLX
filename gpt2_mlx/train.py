"""Training entry points for the GPT-2 MLX learning skeleton."""

from __future__ import annotations

import argparse

import mlx.core as mx

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.data import make_lm_batch
from gpt2_mlx.model import GPT


def compute_loss(model: GPT, x: mx.array, y: mx.array) -> mx.array:
    """Compute training loss for a token batch.

    Args:
        model: A ``GPT`` instance.
        x: Input token IDs with shape ``[B, T]``.
        y: Target token IDs with shape ``[B, T]``.

    Returns:
        A scalar loss array.

    TODO:
        Implement loss computation after you implement ``GPT.__call__`` and
        decide where the language modeling loss should live.
    """
    # x shape: [B, T].
    # y shape: [B, T].
    # output shape: scalar [].
    raise NotImplementedError(
        "TODO: implement training loss computation for x [B, T] and y [B, T]."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the training skeleton."""
    parser = argparse.ArgumentParser(description="Train a GPT-2 MLX skeleton.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8)
    return parser


def main() -> None:
    """Run the training skeleton until the loss TODO is reached."""
    args = build_arg_parser().parse_args()
    config = GPTConfig.tiny()
    model = GPT(config)
    tokens = list(range(args.batch_size * args.block_size + 1))
    x, y = make_lm_batch(tokens, args.batch_size, args.block_size)
    compute_loss(model, x, y)


if __name__ == "__main__":
    main()

