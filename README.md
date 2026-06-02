# GPT-2 MLX Learning Skeleton

This repository is an educational skeleton for reimplementing a small GPT-2 style
language model from scratch in MLX.

The goal is to help you internalize the architecture, not to outsource the
implementation. The package includes class definitions, method signatures, shape
contracts, docstrings, and tests. The important GPT-2 logic is intentionally left
as larger conceptual TODOs.

## Branches

If you are using this project as a learner, start from the `skeleton` branch.
It contains the course-project scaffold with intentionally unfinished GPT-2
implementation tasks.

```bash
git clone <repo-url>
cd GPT2-MLX
git switch skeleton
```

The `main` branch is where the project owner may continue their own learning
implementation. Use `skeleton` when you want the clean starting point.

## What Is Intentionally Missing

You should implement these pieces yourself:

- causal self-attention
- causal mask creation
- GPT-style MLP forward pass
- transformer block composition
- GPT forward pass
- next-token language modeling loss
- autoregressive generation
- optional weight tying

The skeleton does not include KV cache support, Hugging Face GPT-2 weight
loading, checkpointing, or a complete training loop. Those are good later
extensions after the core model is working.

## Project Layout

```text
gpt2_mlx/
  config.py      # GPTConfig dataclass
  model.py       # GPT-2 module skeletons
  data.py        # simple token batch helpers
  train.py       # training CLI skeleton
  generate.py    # generation CLI skeleton
  utils.py       # small non-core helpers
tests/
  test_shapes.py
  test_attention_mask.py
  test_generation.py
```

## Setup

```bash
uv sync
```

## Tests

```bash
uv run pytest
```

Some tests are expected to fail initially. They describe the contracts you need
to satisfy as you implement each conceptual TODO.

You can also run linting with:

```bash
uv run ruff check .
```
