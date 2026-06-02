# AGENTS.md

## Project Introduction

This repository is an educational GPT-2 implementation project in Python using
MLX (`mlx.core` and `mlx.nn`). Its purpose is to help the human learner
internalize the core GPT-2 architecture by filling in intentionally unfinished
conceptual TODOs.

The current codebase is a skeleton, not a complete model. Imports,
configuration, module construction, simple data helpers, and tests are present,
but the important transformer logic is deliberately left unimplemented.

Core learning tasks intentionally left to the user:

- causal self-attention from `x: [B, T, C]` to `y: [B, T, C]`
- causal mask creation compatible with attention scores `[B, H, T, T]`
- GPT-style MLP from `[B, T, C]` to `[B, T, C]`
- transformer block composition
- GPT forward pass from token IDs `[B, T]` to logits `[B, T, vocab_size]`
- next-token language modeling loss
- autoregressive generation
- optional weight tying in a later pass

Do not treat failing TODO tests as regressions. Several tests are intentionally
red until the learner implements the corresponding concepts.

## Tech Stack and Environment

- Python 3.12
- MLX / `mlx.core` / `mlx.nn`
- `uv` for environment and dependency management
- `pytest` for tests
- `ruff` for linting and import checks
- `pyproject.toml` for project configuration

Use `pathlib.Path` for path handling in new Python code. Keep public functions
and methods fully type annotated.

## Supported Commands

Install or sync dependencies:

```bash
uv sync
```

Run the full test suite:

```bash
uv run pytest
```

Expected current result: some tests fail with `NotImplementedError` because
they are contract tests for learner-owned TODOs.

Run linting:

```bash
uv run ruff check .
```

Run a specific test file:

```bash
uv run pytest tests/test_shapes.py
```

Run generation or training skeletons:

```bash
uv run python -m gpt2_mlx.generate
uv run python -m gpt2_mlx.train
```

Expected current result: these reach intentional TODOs rather than completing
generation or training.

## Requirements for Code Generation

When generating or modifying code in this repository:

- Preserve the project as a course-style learning scaffold unless the user
  explicitly asks to implement a specific core concept.
- Do not implement complete GPT-2 logic on the user's behalf by default.
- Do not hide complete solutions in comments, helper functions, tests, or
  one-line snippets.
- Prefer larger conceptual TODOs over tiny line-level TODOs.
- Keep shape comments and docstrings for all unfinished model methods.
- Use `NotImplementedError` for intentionally unfinished learner tasks.
- Keep tests focused on contracts and observable behavior, not reference
  implementations.
- Use `uv` commands for Python tooling.
- Prefer correctness, readability, and maintainability over cleverness.
- Avoid adding scope-creep features in pass 1:
  - KV cache
  - Hugging Face GPT-2 weight loading
  - checkpoint save/load
  - advanced sampling
  - distributed training
  - mixed precision tuning
  - polished full training loop

If the user asks for help implementing a core concept, support them with
principles, shape reasoning, debugging, and targeted hints before offering code.
When code is requested, implement only the requested concept and keep the change
small.

## Professional Review Instructions

When reviewing future user work, take a course-instructor plus senior-engineer
stance.

The goal of every review is to help students learn like professional AI
engineers, work like professional AI engineers, and steadily close the gap
between a toy GPT-2 reimplementation and production-quality LLM engineering.

Use this structure for technical answers:

1. Recommendation
2. Rationale
3. Alternatives

Default review workflow:

- Do not change code automatically during review.
- Write a review report under `reports/` for the student to read, learn from,
  and apply themselves.
- Use a clear filename such as
  `reports/review-YYYY-MM-DD-short-topic.md`.
- Include findings, rationale, suggested fixes, relevant tests, and learning
  references.
- Include a "What to learn next" section that connects the review findings to
  concrete study topics.
- If the student explicitly asks you to improve code for them, preserve the
  original student version as comments, add concise analysis/suggestion comments,
  and then implement the requested improved version.
- Keep the original commented version close to the changed code so the student
  can compare before and after.

For code reviews, lead with findings ordered by severity. Focus on:

- correctness of tensor shapes and broadcasting
- causal masking behavior
- information leakage from future tokens
- residual path and normalization placement
- logits and target alignment for next-token loss
- MLX API correctness
- numerical stability
- test coverage of edge cases
- maintainability and readability
- inefficient practices and avoidable memory or compute waste
- whether the implementation matches MLX best practices
- whether the code would still be understandable to a learner after optimization

Do not rewrite the user's implementation wholesale unless explicitly asked.
Prefer actionable review comments, minimal patches, and questions that help the
learner reason from first principles.

Review standards:

- Check tests before modifying code when feasible.
- Treat intentionally red TODO tests as expected until the corresponding concept
  has been implemented.
- Distinguish learning mistakes from style preferences.
- Call out scope creep directly.
- Suggest additional tests when a concept is implemented.
- Avoid approving code that works only by hardcoding test cases.
- Review performance as engineering discipline, not as premature optimization:
  identify inefficient bad practices, explain why they matter, and suggest
  best-practice alternatives.
- Assume students usually have a single consumer-level Apple Silicon machine.
  The reference local constraint is an Apple Silicon M3 Pro with 36 GB unified
  memory; other students may have less.
- Prefer recommendations that make the minimal model runnable and debuggable on
  local hardware before suggesting production-scale techniques.
- Flag memory risks such as unnecessarily materialized large tensors, avoidable
  copies, excessive context lengths, oversized configs, and tests that require
  unrealistic hardware.
- When discussing scale-up paths, separate local learning guidance from
  server-side and multi-GPU production guidance.
- Provide further reading links when useful, especially for modern LLM
  techniques and scaling topics.
- Cite reputable sources for production-scale topics, such as official
  framework documentation, well-known engineering blogs, research papers, or
  high-quality course notes.
- Make it clear when a technique is appropriate for this educational GPT-2
  project versus something to study for production systems later.
- After a core concept has been implemented, ask for or suggest lightweight
  benchmark notes covering memory use, tokens per second, and maximum context
  length tested on local hardware.
- Once training or generation exists, include evaluation notes for model
  behavior, generation quality, regressions, and limitations.

Suggested review report sections:

- Summary
- Findings ordered by severity
- Correctness and shape analysis
- Efficiency and resource analysis
- Tests to add or update
- Local-hardware guidance
- Production bridge and further reading
- Benchmarks and evaluation notes
- What to learn next

Useful production-bridge topics to reference when relevant:

- activation checkpointing and memory/computation tradeoffs
- mixed precision and numerical stability
- KV cache for inference
- tokenizer and checkpoint compatibility
- distributed data parallelism, tensor parallelism, and pipeline parallelism
- optimizer memory usage
- evaluation, observability, and regression testing for LLM systems
- serving latency, throughput, batching, and caching

## Common Pitfalls to Watch For

- Off-by-one target shifting in language modeling loss.
- Causal masks with the right shape but inverted semantics.
- Accidentally allowing attention to future positions.
- Reshaping heads in a way that swaps time and head dimensions.
- Forgetting to combine heads back to `[B, T, C]`.
- Applying the LM head to the wrong tensor shape.
- Passing sequences longer than `block_size` without a clear policy.
- Implementing generation that repeatedly feeds uncropped contexts past
  `block_size`.
- Adding advanced features before the minimal model works.

## Technical Decision Records

- Decision: Use a small package layout under `gpt2_mlx/`.
  Context: The user requested named files and runnable imports.
  Alternatives considered: flat top-level modules.
  Rationale: A package keeps imports clean while staying easy to navigate.

- Decision: Use intentionally failing contract tests.
  Context: The project is a learning skeleton.
  Alternatives considered: marking TODO tests as `xfail`.
  Rationale: Red tests make the learning targets visible and concrete.

- Decision: Exclude KV cache and Hugging Face loading from pass 1.
  Context: The goal is internalizing GPT-2 basics.
  Alternatives considered: adding future compatibility hooks immediately.
  Rationale: Those features add surface area before the core architecture is
  understood.
