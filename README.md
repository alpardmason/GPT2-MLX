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
  test_shapes.py           # config, construction, and shape contracts
  test_attention_mask.py   # causal mask contract
  test_causal_behavior.py  # no-future-leakage behavioral contracts
  test_generation.py       # autoregressive generation contracts
  test_loss.py             # next-token loss sanity contract
  test_utils.py            # non-core helper tests
.github/
  workflows/ci.yml         # lint, type check, scaffold-gated tests
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
to satisfy as you implement each conceptual TODO. These intentionally-red tests
carry the `todo` marker, so you can run just the scaffold/infra tests (the same
subset CI gates on) with:

```bash
uv run pytest -m "not todo"
```

As you implement each concept, its `todo`-marked tests should turn green.

## Linting and Type Checking

```bash
uv run ruff check .
uv run pyright
```

Continuous integration (`.github/workflows/ci.yml`) runs linting, type checking,
and `pytest -m "not todo"` on pushes and pull requests to `main` and `skeleton`.
