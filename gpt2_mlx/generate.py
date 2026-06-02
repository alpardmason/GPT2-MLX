"""Generation entry points for the GPT-2 MLX learning skeleton."""

from __future__ import annotations

import argparse

import mlx.core as mx

from gpt2_mlx.config import GPTConfig
from gpt2_mlx.model import GPT


def generate_text(
    model: GPT,
    prompt_tokens: mx.array,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> mx.array:
    """Generate tokens from a prompt using the model's generation contract.

    Args:
        model: A ``GPT`` instance.
        prompt_tokens: Integer token IDs with shape ``[B, T]``.
        max_new_tokens: Number of new tokens to append.
        temperature: Sampling temperature for the future implementation.

    Returns:
        Integer token IDs with shape ``[B, T + max_new_tokens]``.

    TODO:
        Implement autoregressive generation conceptually:
        1. repeatedly crop context if needed
        2. call the model on the current token sequence
        3. select the next-token logits
        4. sample or choose the next token
        5. append it to the running sequence

    This function intentionally delegates to ``GPT.generate`` so the core
    generation task lives with the model contract.
    """
    return model.generate(
        idx=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the generation skeleton."""
    parser = argparse.ArgumentParser(description="Generate with a GPT-2 MLX skeleton.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser


def main() -> None:
    """Run the generation skeleton until the model TODO is reached."""
    args = build_arg_parser().parse_args()
    config = GPTConfig.tiny()
    model = GPT(config)
    prompt = mx.array([[0]], dtype=mx.int32)
    generate_text(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

